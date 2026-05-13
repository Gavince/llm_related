from transformers import AutoModelForCausalLM, AutoModel, AutoModelForSequenceClassification, AutoTokenizer, PreTrainedModel
from dataclasses import dataclass
from typing import Optional, Union, Tuple
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
# from torch.utils.tensorboard import SummaryWriter
from typing import Callable, Dict, List, Optional, Tuple, Union, Any
from copy import deepcopy
from datasets import load_dataset
from reward_func import *
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'


class GSM8KDataset(Dataset):
    """GSM8K 中文数据集封装。

    约定每条样本返回:
    - prompt: 问题文本
    - answer: 标准答案（用于奖励函数）
    """
    def __init__(self, data_path, tokenizer):
        """加载 HuggingFace datasets 格式的数据集。"""
        self.tokenizer = tokenizer
        data = load_dataset(data_path)
        self.data = data['train']
  
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        sample = self.data[index]
        # prompt = self.tokenizer.apply_chat_template(sample['prompt'], tokenize=False, add_generation_prompt=True)
        answer = sample['answer_only']
        prompt = sample['question_zh-cn']
        return {'prompt': prompt, 'answer': answer}


@dataclass
class Samples:
    """单个 group（同一 prompt 的多次采样）中间结果容器。"""
    prompt_response_ids: torch.Tensor
    response_ids: torch.Tensor
    prompt: Any
    answer: Any
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    num_actions: Union[int, torch.Tensor]
    response_length: int


class GRPOArguments:
    """训练超参数配置（简化版，无 dataclass）。"""
    output_dir = './output'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lr = 0.000001
    save_steps = 100
    epoch = 3
    num_generations = 4 # 组内样本数
    max_prompt_length = 256 # 最大输入长度
    max_generate_length = 256 # 最大输出长度
    reward_weights : List[float] = None # 奖励的权重（多个奖励函数）
    beta = 0.0 # KL散度的系数，为0则忽略KL散度，即不使用参考模型
    clip_eps_high = 0.28
    clip_eps_low = 0.2
    gradient_accumulation_steps = 2 # 梯度累加
    num_iterations = 2 # 采样一次样本训练模型轮数
    batch_size = 2

class GRPOTrainer:
    """GRPO/DAPO 训练器。

    训练主流程:
    1) generate_samples: 对每个 prompt 采样多条 response（组内采样）
    2) generate_experiences: 计算奖励、优势、old/ref log_probs
    3) compute_loss: 计算 token-level PPO/GRPO/DAPO 损失
    4) train_step/train: 梯度累积 + 多轮迭代更新
    """
    def __init__(self,
        model = None,
        reward_funcs: Union[List[str], List[Callable]] = None,
        args = None,
        train_dataset: Optional[Union[Dataset]] = None,
        eval_dataset: Optional[Union[Dataset]] = None,
        tokenizer = None,
        reward_tokenizers = None):

        self.args = args
        # 加载模型
        if isinstance(model, str):
            model = AutoModelForCausalLM.from_pretrained(model)
        self.model = model.to(self.args.device)
        
        # 是否使用参考模型
        self.ref_model = None
        if self.args.beta != 0.0:
            self.ref_model = deepcopy(model)
            self.ref_model.eval()
    
        
        if isinstance(tokenizer, str):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        
        self.tokenizer = self.get_tokenizer(tokenizer)
        
        
        if isinstance(reward_funcs, str):
            reward_funcs = [reward_funcs]
        
        for i, reward_func in enumerate(reward_funcs):
            # 如果奖励函数为字符串，表示使用的是奖励模型，则加载模型
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1).to(self.args.device)
        
        self.reward_funcs = reward_funcs
        
        if reward_tokenizers is None:
            reward_tokenizers = [None] * len(reward_funcs)
            
        elif isinstance(reward_tokenizers, str):
            reward_tokenizers = [reward_tokenizers]
            
        else:
            if len(reward_tokenizers) != len(reward_funcs):
                raise ValueError("Length of reward_tokenizers must be equal to the number of reward_funcs.")
            
        for i, (reward_tokenizer, reward_func) in enumerate(zip(reward_tokenizers, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_tokenizer is None:
                    reward_tokenizer = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_tokenizer.pad_token_id is None:
                    reward_tokenizer.pad_token = reward_tokenizer.eos_token
                
                reward_func.config.pad_token_id = reward_tokenizer.pad_token_id
                reward_tokenizers[i] = reward_tokenizer
        self.reward_tokenizers = reward_tokenizers
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        # 缓存已经生成的数据的一个批次的数据，可供模型多次训练迭代，无需重新生成
        self.input_buffer = [None] * self.args.gradient_accumulation_steps
        
        # 模型更新的次数
        self.update_steps = 0 

    def get_tokenizer(self, tokenizer):
        """统一 tokenizer 设置，训练时使用左填充。"""
        tokenizer.padding_side = "left"
        return tokenizer
    
    # 生成样本，以组为单位
    def generate_samples(self, inputs):
        """按 prompt 生成 group 样本。

        参数:
            inputs: DataLoader 返回的 batch，至少包含 `prompt`。
        返回:
            List[Samples]，每个元素对应一个 prompt 的 group 数据。
        """
        samples_list = []
        self.model.eval()
        prompts = [prompt for prompt in inputs['prompt']]
        answers = [None] * len(prompts)
        
        if 'answer' in inputs:
            answers = [answer for answer in inputs['answer']]
        
        max_length = self.args.max_generate_length + self.args.max_prompt_length
        for prompt, answer in zip(prompts, answers):
            # 应用聊天模板，加入系统提示词
            input_text = self.tokenizer.apply_chat_template([{"role": "system", 'content': SYSTEM_PROMPT}, {"role": "user", 'content': prompt}]
                                                            , add_generation_prompt=True
                                                            , tokenize=False)
            
            # 生成一个 group 的输入（同一个 prompt 重复 num_generations 次）
            inputs = self.tokenizer([input_text] * self.args.num_generations
                                    , padding='max_length'
                                    , max_length=self.args.max_prompt_length
                                    , truncation=True
                                    , return_tensors='pt')
            prompt_ids = inputs['input_ids']
            with torch.no_grad():
                prompt_response_ids = self.model.generate(**inputs.to(self.args.device), 
                                    max_new_tokens = self.args.max_generate_length,
                                    temperature=0.9,
                                    top_p = 1,
                                    top_k = 50)
                
            # 对齐长度到 max_prompt_length + max_generate_length，便于后续 batch 化。
            if prompt_response_ids.size(1) >= max_length:
                prompt_response_ids = prompt_response_ids[:, :max_length]
            else:
                prompt_response_ids = torch.cat([prompt_response_ids, torch.full((prompt_response_ids.size(0), max_length - prompt_response_ids.size(1)), fill_value=self.tokenizer.pad_token_id, device=prompt_response_ids.device)], dim=1)
          
            # attention_mask: prompt+response 全序列有效位
            attention_mask = (prompt_response_ids.ne(self.tokenizer.pad_token_id)).to(dtype=torch.long)
            # response_ids: 仅响应区间，用于奖励和 action 定位
            response_ids = prompt_response_ids[:, prompt_ids.size(1):]
            # action_mask: 仅在 response 区域中，非 eos/pad 的 token 参与策略优化
            action_mask = (response_ids.ne(self.tokenizer.eos_token_id) & response_ids.ne(self.tokenizer.pad_token_id)).to(dtype=torch.long)
        

            # 存储的是一个group的数据
            samples = Samples(
                prompt_response_ids=prompt_response_ids,
                response_ids=response_ids,
                prompt = prompt,
                answer = answer,
                attention_mask=attention_mask,
                action_mask=action_mask,
                num_actions=action_mask.size(1),
                response_length=action_mask.float().sum(dim=-1)
            )
            samples_list.append(samples)

        return samples_list
    
    # 生成经验(优势、token的概率分布)
    def generate_experiences(self, inputs):
        """将采样结果转换为 PPO/GRPO 训练经验。

        关键中间量:
            rewards_per_func: [num_reward_funcs, num_generations]
            rewards:          [num_generations]
            advantages:       [num_generations] (句子粒度，组内标准化)
            old/ref log_prob: [num_generations, num_actions]
        """
        
        self.model.eval()
        # [Samples(prompt_response_ids, response_ids, prompt, answer, attention_mask, action_mask, num_actions, response_length), ...]
        samples_list = self.generate_samples(inputs)
        batch_prompt_response_ids = []
        batch_attention_mask = []
        batch_action_mask = []
        batch_advantages = []
        batch_old_action_log_probs = []
        batch_ref_action_log_probs = []
        # 遍历batch中每一个prompt，将其对应的n组输出也一起拼接
        for samples in samples_list:
            prompt_response_ids = samples.prompt_response_ids # shape: (num_generations, seq_len)
            response_ids = samples.response_ids # shape: (num_generations, seq_len)
            answer = samples.answer
            attention_mask = samples.attention_mask # shape: (num_generations, seq_len)
            action_mask = samples.action_mask # shape: (num_generations, num_actions)
            num_actions = samples.num_actions
            prompt = samples.prompt
            
            with torch.no_grad():
                
                # 存储各个奖励函数在一个group内各个响应的奖励
                rewards_per_func = torch.zeros(len(self.reward_funcs), self.args.num_generations, device=self.args.device)
                
                # 将输出转换成文本
                response_texts = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
                prompt_texts = [prompt] * len(response_texts)
                # 奖励模型常基于“完整对话（prompt+response）”评分。
                prompt_response_texts = [prompt + response for prompt, response in zip(prompt_texts, response_texts)]
                
                for i, (reward_func, reward_tokenizer) in enumerate(
                    zip(self.reward_funcs, self.reward_tokenizers)
                ):
                    if isinstance(reward_func, PreTrainedModel):
                        with torch.inference_mode():
                            reward_model_inputs = reward_tokenizer(prompt_response_texts, return_tensors="pt", padding=True)
                            rewards_per_func[i] = reward_func(**reward_model_inputs.to(self.args.device)).logits.squeeze(-1)
                    
                    else:
                        answers = [answer] * len(prompt_texts)
                        # 输出为list，每个元素为单个响应的奖励值
                        output_reward_func = reward_func(prompts=prompt_texts, responses=response_texts, answers=answers)
                        # 奖励函数返回 None 时置为 NaN，便于后续定位异常样本/奖励函数。
                        output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]
                        rewards_per_func[i] = torch.tensor(output_reward_func, dtype=torch.float32, device=self.args.device)
                
                # rewards_per_func: [num_funcs, num_generations], 每一条都应该有一个单独的奖励值
                if not self.args.reward_weights:
                    self.args.reward_weights = [1.0] * len(self.reward_funcs)
                if len(self.args.reward_weights) != len(self.reward_funcs):
                    raise ValueError("The number of reward weights must be equal to the number of reward functions.")
                # 乘以各个奖励函数的权重
                rewards = rewards_per_func * torch.tensor(self.args.reward_weights, dtype=torch.float32, device=rewards_per_func.device).unsqueeze(1)
                # rewards: [num_funcs, num_generations]
                rewards = rewards.sum(dim=0) # shape: [num_generations]
                print(f'rewards: {rewards}')
                
                # 组内均值方差标准化（GRPO核心）：只比较同一 prompt 下多条采样结果。
                mean_group_rewards = rewards.mean()
                std_group_rewards = rewards.std()
                
                # GRPO的优势是句子粒度的，而非token粒度的
                advantages = (rewards - mean_group_rewards) / (std_group_rewards + 1e-8) # shape: [num_generations]
                # 统计优势中非零元素的数量，如果为0，则说明该组中的优势全为0，舍弃该组数据(对更新模型没有用)
                nonzero_num = advantages.count_nonzero().item()
                if nonzero_num == 0:
                    continue
                # 将有收益的数据加载保存，但是会不满足bacth data的size，需要后续拼接
                batch_advantages.append(advantages)
                
                # 计算策略模型输出token的概率
                # old_action_log_probs: 采样策略在动作区间 token 上的对数概率
                old_action_log_probs = self.get_action_log_probs(self.model, prompt_response_ids, attention_mask, num_actions)
                batch_old_action_log_probs.append(old_action_log_probs)
                
                # 是否使用参考模型
                if self.ref_model:
                    # ref_action_log_probs: 用于 KL 正则项
                    ref_action_log_probs = self.get_action_log_probs(self.ref_model, prompt_response_ids, attention_mask, num_actions)
                    batch_ref_action_log_probs.append(ref_action_log_probs)
                    
                
                batch_prompt_response_ids.append(prompt_response_ids)
                batch_attention_mask.append(attention_mask)
                batch_action_mask.append(action_mask)
        
               
        return {
            "prompt_response_ids": batch_prompt_response_ids,
            "attention_mask": batch_attention_mask,
            "action_mask": batch_action_mask,
            "old_action_log_probs": batch_old_action_log_probs,
            "ref_action_log_probs": batch_ref_action_log_probs if self.ref_model else None,
            "advantages": batch_advantages,
        }
    
    def compute_loss(self, model, inputs):
        """计算 GRPO/DAPO 目标。

        输入形状（cat 后）:
            prompt_response_ids: [batch_size * num_generations, seq_len]
            action_mask:         [batch_size * num_generations, num_actions]
            advantages:          [batch_size * num_generations]
        """
        
        prompt_response_ids = inputs['prompt_response_ids']
        attention_mask = inputs['attention_mask']
        action_mask = inputs['action_mask']
        num_actions = action_mask.size(1)
        # 当前策略在动作 token 上的 log_prob
        action_log_probs = self.get_action_log_probs(model, prompt_response_ids, attention_mask, num_actions)
        
        if self.args.beta != 0.0:
            
            ref_action_log_probs = inputs['ref_action_log_probs']
            # log_ratio > 0 表示参考模型概率更高；log_ratio < 0 表示当前模型概率更高。
            log_ratio = ref_action_log_probs - action_log_probs 
            log_ratio = log_ratio * action_mask
            
            # k3: log_ratio.exp() - 1 - log_ratio
            k3 = log_ratio.exp() - 1 - log_ratio
        
        advantages = inputs['advantages']
        
        # num_iterations == 1 时，old policy 退化为当前 policy（detach）以保持公式一致。
        old_action_log_probs = inputs['old_action_log_probs'] if self.args.num_iterations > 1 else action_log_probs.detach()
        coef_1 = torch.exp(action_log_probs - old_action_log_probs) # 重要性采样 shape: [batch_size * num_generations, num_actions]
        coef_2 = torch.clamp(coef_1, 1 - self.args.clip_eps_low, 1 + self.args.clip_eps_high)
        # 句子级优势广播到 token 维：同一 response 的每个 token 共享同一 advantage。
        per_token_loss1 = coef_1 * advantages.unsqueeze(1) # 一个序列中每个token的优势是一样的
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2) # shape: [batch_size * num_generations, num_actions]
        per_token_loss = per_token_loss * action_mask  
        if self.args.beta != 0.0:
            per_token_loss = per_token_loss + self.args.beta * k3
        
        # GRPO loss
        # loss = (per_token_loss.sum(dim=1) / action_mask.sum(dim=1)) shape: [batch_size * num_generations]
        # loss = loss.mean()
        
        
        # DAPO loss
        # per_token_loss = per_token_loss.view(-1, self.args.num_generations, num_actions) #  shape: [batch_size, num_generations, num_actions]
        # loss = per_token_loss.sum(-1).sum(-1) / action_mask.sum(-1).sum(-1) # shape: [batch_size]
        # loss = loss.mean()
        
        # 当前启用的聚合方式：先在组内所有 token 上求平均，再对 batch 求均值。
        per_token_loss = per_token_loss.view(-1, self.args.num_generations, num_actions) #  shape: [batch_size, num_generations, num_actions]
        action_mask = action_mask.view(-1, self.args.num_generations, num_actions)
        loss = per_token_loss.sum(-1).sum(-1) / action_mask.sum(-1).sum(-1) # shape: [batch_size]
        loss = loss.mean()
        
        return loss


    def get_action_log_probs(self, model, input_ids, attention_mask, num_actions):
        """提取动作区间 token 的 teacher-forcing log_probs。

        步骤:
        1) logits[:, :-1, :] 与 labels=input_ids[:, 1:] 对齐
        2) gather 出真实 label 的对数概率
        3) 仅保留最后 num_actions（对应 response 动作区间）
        """
        
        # 计算策略模型输出token的概率
        output = model(input_ids, attention_mask=attention_mask)
        logits = output.logits
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
        log_probs_labels = log_probs.gather(dim=-1, index=input_ids[:, 1:].unsqueeze(-1))
        action_log_probs = log_probs_labels.squeeze(-1)[:, -num_actions:]
        return action_log_probs

    
    
    def train_step(self, model, inputs, optimizer, step):
        """单个微步训练（支持梯度累积）。"""
        model.train()
        # scaler = torch.amp.GradScaler()
        # with torch.amp.autocast(device_type='cuda'):
        loss = self.compute_loss(model, inputs)
        loss = loss / self.args.gradient_accumulation_steps
        # loss = scaler.scale(loss)
        loss.backward()
        if (step + 1) % self.args.gradient_accumulation_steps == 0:
            
            optimizer.step()
            optimizer.zero_grad()
            # scaler.unscale_(optimizer)
            # scaler.step(optimizer)
            # scaler.update()
        
            # writer.add_scalar("grpo_loss", loss.item(), self.update_steps)
            print(f"step: {self.update_steps}/{self.global_steps}  grpo_loss: {loss.item():.8f}")
        torch.cuda.empty_cache()

    def train(self):
        """训练入口。

        逻辑:
        - 不断采样并写入 buffer（按 group 存）
        - 凑够训练 batch 后拼接为 tensor
        - 根据 gradient_accumulation_steps 触发优化器更新
        - 每次更新可复用同一批经验做 num_iterations 轮优化
        """
        self.global_steps = self.args.num_iterations * self.args.epoch * len(self.train_dataset) // (self.args.batch_size * self.args.gradient_accumulation_steps)
        for _ in range(self.args.epoch):
            
            dataloader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True)
            buffer = {'prompt_response_ids':[],
                      'attention_mask':[],
                      'action_mask':[],
                      'old_action_log_probs':[],
                      'ref_action_log_probs':[],
                      'advantages':[]}
            idx = 0
            for batch in dataloader:
                # {'prompt': ['足球队的财务主管必须为其球队的 16 名球员购买装备。每件装备包括一件 25 美元的球衣、一条 15.20 美元的短裤和一双 6.80 美元的袜子。队伍中所有玩家的所有装备要多少钱？', '詹姆斯在锻炼时受伤了。三天后疼痛就减轻了，但他知道伤势至少需要五倍的时间才能完全愈合。之后，他想再等三天再开始锻炼。如果他想等三周后再开始举重，那么他需要多长时间才能再次举重？']
                # , 'answer': tensor([752,  39])}
                inputs = self.generate_experiences(batch)
                """
                {'prompt_response_ids': [tensor([[151643, 151643, 151643,  ..., 151643, 151643, 151643],
                [151643, 151643, 151643,  ..., 151643, 151643, 151643],
                [151643, 151643, 151643,  ...,     17,     15, 100252],
                [151643, 151643, 151643,  ...,     20,     17, 101237]])]
                , 'attention_mask': [tensor([[0, 0, 0,  ..., 0, 0, 0],
                [0, 0, 0,  ..., 0, 0, 0],
                [0, 0, 0,  ..., 1, 1, 1],
                [0, 0, 0,  ..., 1, 1, 1]])], 'action_mask': [tensor([[1, 1, 1,  ..., 0, 0, 0],
                [1, 1, 1,  ..., 0, 0, 0],
                [1, 1, 1,  ..., 1, 1, 1],
                [1, 1, 1,  ..., 1, 1, 1]])], 'old_action_log_probs': [tensor([[-6.6509e-04, -1.2123e-04, -3.0386e-01,  ..., -1.5479e+01,
                -1.5379e+01, -1.4846e+01],
                [-6.6509e-04, -1.2123e-04, -3.0386e-01,  ..., -1.4115e+01,
                -1.3853e+01, -1.4426e+01],
                [-6.6509e-04, -1.2123e-04, -3.0386e-01,  ..., -1.1921e-07,
                -2.3075e-03, -1.1264e-03],
                [-6.6509e-04, -1.2123e-04, -3.0386e-01,  ..., -2.3842e-07,
                -4.1723e-06, -2.3041e-04]])], 'ref_action_log_probs': None, 'advantages': [tensor([ 1.4142,  0.0000, -0.7071, -0.7071])]}
                """
                # buffer 中每个元素是“一个 prompt 对应的 group 张量”，后续再统一 cat。
                buffer['prompt_response_ids']+=inputs['prompt_response_ids']
                buffer['attention_mask']+=inputs['attention_mask']
                buffer['action_mask'] += inputs['action_mask']
                buffer['old_action_log_probs'] += inputs['old_action_log_probs']
                if self.ref_model is not None:
                    buffer['ref_action_log_probs'] += inputs['ref_action_log_probs']
                else:
                    buffer['ref_action_log_probs'] = None
                
                buffer['advantages'] +=inputs['advantages']
                
             
                # 如果生成的样本batch_size小于设定的batch_size，说明生成数据过程中有舍弃数据，需要继续采样，凑够一个完整的batch_size
         
                if len(buffer['prompt_response_ids']) < self.args.batch_size:
                    continue
                
                if self.ref_model is not None:
                    # 取前 batch_size 个 group，沿 group 维拼接为 [batch_size * num_generations, ...]
                    inputs = {k: v[:self.args.batch_size] for k, v in buffer.items()}
                    inputs = {k: torch.cat(v, dim=0) for k, v in inputs.items()}
                    buffer = {k: v[self.args.batch_size:] for k, v in buffer.items()}
                    
                else:
                    inputs = {k: v[:self.args.batch_size] for k, v in buffer.items() if k != 'ref_action_log_probs'}
                    inputs = {k: torch.cat(v, dim=0) for k, v in inputs.items()}
                    inputs['ref_action_log_probs'] = None
                    buffer = {k: v[self.args.batch_size:] for k, v in buffer.items() if k != 'ref_action_log_probs'}
                    buffer['ref_action_log_probs'] = None
                self.input_buffer[idx % self.args.gradient_accumulation_steps] = inputs

                if (idx + 1) % self.args.gradient_accumulation_steps == 0:
                   
                    # 同一批经验可重复训练 num_iterations 轮（on-policy 近似下的小步复用）。
                    for _ in range(self.args.num_iterations):
                        for step, inputs in enumerate(self.input_buffer):
                            self.train_step(self.model, inputs, self.optimizer, step)
                        
                        self.update_steps += 1
                        if self.update_steps % self.args.save_steps == 0:
                            self.model.save_pretrained(self.args.output_dir + f'/checkpoint_{self.update_steps}')
                            self.tokenizer.save_pretrained(self.args.output_dir + f'/checkpoint_{self.update_steps}')
                
                idx += 1
                   
                del inputs

    def save_model(self):
        """保存最终模型与 tokenizer。"""
        self.model.save_pretrained(self.args.output_dir)
        self.tokenizer.save_pretrained(self.args.output_dir)           

if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    
    SYSTEM_PROMPT = """
按照如下格式回答问题：
<think>
你的思考过程
</think>
<answer>
你的回答
</answer>
"""
    
    args = GRPOArguments()
    
    # writer = SummaryWriter('./runs')
    # 策略模型
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B')
    model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B')
    # 奖励函数
    # reward_model = '/home/user/Downloads/reward-model-deberta-v3-large-v2'
    # reward_tokenizer = AutoTokenizer.from_pretrained('/home/user/Downloads/reward-model-deberta-v3-large-v2')
    

    
    
    prompts_dataset = GSM8KDataset('swulling/gsm8k_chinese', tokenizer)
  
    trainer = GRPOTrainer(model=model,
                          reward_funcs = [correctness_reward, digit_reward, hard_format_reward, mark_reward],
                          args=args,
                          train_dataset=prompts_dataset,
                          tokenizer=tokenizer)
    trainer.train()
    trainer.save_model()
    

