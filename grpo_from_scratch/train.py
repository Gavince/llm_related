"""
GRPO (Group Relative Policy Optimization) 训练器实现
核心思想：在一组生成的响应中，通过相对优势（组内归一化的奖励）进行策略优化
"""

from transformers import AutoModelForCausalLM, AutoModel, AutoModelForSequenceClassification, AutoTokenizer, PreTrainedModel
from dataclasses import dataclass
from typing import Optional, Union, Tuple
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
# from torch.utils.tensorboard import SummaryWriter
from typing import Callable, Dict, List, Optional, Tuple, Union, Any
from copy import deepcopy
from datasets import load_dataset
from reward_func import *
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'


class GSM8KDataset(Dataset):
    """
    GSM8K数学问题数据集
    - 加载中文版GSM8K数据集
    - 返回问题(prompt)和答案(answer)
    """
    def __init__(self, data_path, tokenizer):
        self.tokenizer = tokenizer
        data = load_dataset(data_path)
        self.data = data['train']
  
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        sample = self.data[index]
        # prompt = self.tokenizer.apply_chat_template(sample['prompt'], tokenize=False, add_generation_prompt=True)
        answer = sample['answer_only']  # 仅包含答案数值
        prompt = sample['question_zh-cn']  # 中文问题
        return {'prompt': prompt, 'answer': answer}


@dataclass
class Samples:
    """
    存储一个Group内所有生成样本的数据结构
    - prompt_response_ids: 完整序列(prompt+response)的token ids
    - response_ids: 仅响应部分的token ids
    - prompt: 原始问题文本
    - answer: 标准答案
    - attention_mask: 注意力掩码(非pad位置为1)
    - action_mask: 动作掩码(非eos和pad的response token为1)
    - num_actions: 动作数量(响应序列长度)
    - response_length: 实际响应长度
    """
    prompt_response_ids: torch.Tensor  # [num_generations, seq_len]
    response_ids: torch.Tensor         # [num_generations, response_len]
    prompt: Any                        # 原始prompt文本
    answer: Any                        # 标准答案
    attention_mask: Optional[torch.LongTensor]  # [num_generations, seq_len]
    action_mask: Optional[torch.BoolTensor]     # [num_generations, response_len]
    num_actions: Union[int, torch.Tensor]       # 响应序列长度
    response_length: int                        # 实际响应长度


class GRPOArguments:
    """
    GRPO训练超参数配置
    """
    output_dir = './output'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lr = 0.000001  # 学习率
    save_steps = 100  # 保存检查点的步数间隔
    epoch = 3  # 训练轮数
    num_generations = 2  # 每个prompt生成的响应数量(组大小)
    max_prompt_length = 256  # 最大输入长度
    max_generate_length = 256  # 最大生成长度
    reward_weights : List[float] = None  # 各奖励函数的权重(支持多奖励加权)
    beta = 0.0  # KL散度系数，>0时使用参考模型约束，=0时无KL约束
    clip_eps = 0.2  # PPO裁剪系数，控制策略更新幅度
    gradient_accumulation_steps = 2  # 梯度累积步数
    num_iterations = 1  # 每批样本的训练迭代次数(重复利用采样数据)
    batch_size = 2  # 批次大小(每批包含多少个prompt)

class GRPOTrainer:
    def __init__(self,
        model = None,
        reward_funcs: Union[List[str], List[Callable]] = None,
        args = None,
        train_dataset: Optional[Union[Dataset]] = None,
        eval_dataset: Optional[Union[Dataset]] = None,
        tokenizer = None,
        reward_tokenizers = None):
        """
        初始化GRPO训练器
        
        参数:
            model: 策略模型(待训练的生成模型)
            reward_funcs: 奖励函数列表，可以是函数或奖励模型路径
            args: 训练参数配置
            train_dataset: 训练数据集
            eval_dataset: 验证数据集
            tokenizer: 分词器
            reward_tokenizers: 奖励模型对应的分词器列表
        """
        self.args = args
        
        # === 1. 加载策略模型 ===
        if isinstance(model, str):
            model = AutoModelForCausalLM.from_pretrained(model)
        self.model = model.to(self.args.device)
        
        # === 2. 可选：创建参考模型(用于KL散度约束) ===
        self.ref_model = None
        if self.args.beta != 0.0:
            self.ref_model = deepcopy(model)  # 深拷贝当前模型作为参考
            self.ref_model.eval()  # 参考模型始终处于评估模式
    
        # === 3. 初始化分词器 ===
        if isinstance(tokenizer, str):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.tokenizer = self.get_tokenizer(tokenizer)
        
        # === 4. 初始化奖励函数 ===
        # 支持两种奖励函数：自定义函数 或 奖励模型
        if isinstance(reward_funcs, str):
            reward_funcs = [reward_funcs]
        
        for i, reward_func in enumerate(reward_funcs):
            # 如果是字符串路径，则加载奖励模型
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1).to(self.args.device)
        
        self.reward_funcs = reward_funcs
        
        # === 5. 初始化奖励模型的分词器 ===
        if reward_tokenizers is None:
            reward_tokenizers = [None] * len(reward_funcs)
        elif isinstance(reward_tokenizers, str):
            reward_tokenizers = [reward_tokenizers]
        else:
            if len(reward_tokenizers) != len(reward_funcs):
                raise ValueError("奖励分词器数量必须与奖励函数数量一致")
            
        # 为每个奖励模型配置分词器
        for i, (reward_tokenizer, reward_func) in enumerate(zip(reward_tokenizers, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_tokenizer is None:
                    reward_tokenizer = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_tokenizer.pad_token_id is None:
                    reward_tokenizer.pad_token = reward_tokenizer.eos_token
                
                reward_func.config.pad_token_id = reward_tokenizer.pad_token_id
                reward_tokenizers[i] = reward_tokenizer
        self.reward_tokenizers = reward_tokenizers
        
        # === 6. 初始化优化器 ===
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        
        # === 6.5 初始化混合精度训练（V100加速优化） ===
        self.scaler = torch.cuda.amp.GradScaler()
        
        # === 7. 设置数据集 ===
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        # === 8. 初始化训练状态 ===
        # 经验缓冲区：缓存生成的样本，支持多次迭代训练而无需重新生成
        self.input_buffer = [None] * self.args.gradient_accumulation_steps
        # 模型更新步数计数器
        self.update_steps = 0 
    def get_tokenizer(self, tokenizer):
        """
        配置分词器
        - 左填充：确保生成时从右侧开始生成新token
        """
        tokenizer.padding_side = "left"
        return tokenizer
    
    def generate_samples(self, inputs):
        """
        生成样本(以组为单位)
        
        核心流程：
        1. 对每个prompt，生成num_generations个响应(形成一个group)
        2. 构建attention_mask和action_mask
        3. 返回Samples对象列表
        
        参数:
            inputs: 包含'prompt'和'answer'的字典
        
        返回:
            samples_list: Samples对象列表，每个对象代表一个group
        """
        samples_list = []
        self.model.eval()  # 生成阶段不需要梯度
        # ['马里昂在动物救助中心接收的海龟比玛莎多了 20 只海龟，他们去动物救助中心参加动物救助日活动。如果玛莎收到了 40 只海龟，那么他们一共收到了多少只海龟？'
        # , '比尔担心另一场流行病正在囤积卫生纸。比尔每天上厕所 3 次，每次使用 5 格卫生纸。如果比尔有 1000 卷卫生纸，每卷有 300 平方卫生纸，他的卫生纸供应量能持续多少天？']
        prompts = [prompt for prompt in inputs['prompt']]
        """{'prompt': ['马里昂在动物救助中心接收的海龟比玛莎多了 20 只海龟，他们去动物救助中心参加动物救助日活动。如果玛莎收到了 40 只海龟，那么他们一共收到了多少只海龟？'
        , '比尔担心另一场流行病正在囤积卫生纸。比尔每天上厕所 3 次，每次使用 5 格卫生纸。如果比尔有 1000 卷卫生纸，每卷有 300 平方卫生纸，他的卫生纸供应量能持续多少天？']
        , 'answer': tensor([  100, 20000])}"""
        answers = [None] * len(prompts)
        
        if 'answer' in inputs:
            answers = [answer for answer in inputs['answer']]
        
        max_length = self.args.max_generate_length + self.args.max_prompt_length
        
        for prompt, answer in zip(prompts, answers):
            # === 步骤1：构建输入文本(应用聊天模板) ===
            input_text = self.tokenizer.apply_chat_template(
                [{"role": "system", 'content': SYSTEM_PROMPT}, 
                 {"role": "user", 'content': prompt}], 
                add_generation_prompt=True, 
                tokenize=False
            )
            
            # === 步骤2：为同一个prompt创建num_generations份输入 ===
            # 复制相同输入，生成多个不同响应(通过采样实现多样性)
            inputs = self.tokenizer(
                [input_text] * self.args.num_generations, 
                padding='max_length', 
                max_length=self.args.max_prompt_length, 
                truncation=True, 
                return_tensors='pt'
            )
            prompt_ids = inputs['input_ids']
            
            # === 步骤3：生成响应（使用混合精度加速） ===
            with torch.no_grad():
                with torch.cuda.amp.autocast():  # 混合精度加速生成
                    prompt_response_ids = self.model.generate(
                        **inputs.to(self.args.device), 
                        max_new_tokens=self.args.max_generate_length,
                        temperature=1.2,  # 采样温度，控制随机性
                        top_p=1,          # nucleus采样
                        top_k=50,          # top-k采样
                        do_sample=True
                    )
            
            # === 步骤4：统一序列长度(padding或截断) ===
            if prompt_response_ids.size(1) >= max_length:
                prompt_response_ids = prompt_response_ids[:, :max_length]
            else:
                
                prompt_response_ids = torch.cat([
                    prompt_response_ids, 
                    torch.full(
                        (prompt_response_ids.size(0), max_length - prompt_response_ids.size(1)), 
                        fill_value=self.tokenizer.pad_token_id, 
                        device=prompt_response_ids.device
                    )
                ], dim=1)
          
            # === 步骤5：构建掩码 ===
            # attention_mask: 非pad位置为1,此处为prompt+response的token
            attention_mask = (prompt_response_ids.ne(self.tokenizer.pad_token_id)).to(dtype=torch.long)
            
            # response_ids: 仅保留响应部分
            response_ids = prompt_response_ids[:, prompt_ids.size(1):]
            
            # action_mask: 排除eos和pad的响应token(仅对有效生成内容计算损失)
            action_mask = (response_ids.ne(self.tokenizer.eos_token_id) & 
                          response_ids.ne(self.tokenizer.pad_token_id)).to(dtype=torch.long)
        
            # === 步骤6：封装为Samples对象 ===
            samples = Samples(
                prompt_response_ids=prompt_response_ids,  # 完整序列
                response_ids=response_ids,                # 响应部分
                prompt=prompt,                            # 原始问题
                answer=answer,                            # 标准答案
                attention_mask=attention_mask,            # 注意力掩码
                action_mask=action_mask,                  # 动作掩码
                num_actions=action_mask.size(1),          # 动作数量
                response_length=action_mask.float().sum(dim=-1)  # 实际响应长度
            )
            samples_list.append(samples)
            
            # 及时释放中间变量，节省内存
            del inputs, prompt_ids
            torch.cuda.empty_cache()

        return samples_list
    
    def generate_experiences(self, inputs):
        """
        生成训练所需的经验数据
        
        核心流程：
        1. 生成样本(generate_samples)
        2. 计算各样本的动作对数概率(策略模型 + 可选参考模型)
        3. 计算奖励(多个奖励函数加权)
        4. 计算优势(组内归一化) - GRPO的核心
        
        参数:
            inputs: 包含'prompt'和'answer'的字典
        
        返回:
            dict: 包含训练所需的所有数据(ids, masks, log_probs, advantages等)
        """
        self.model.eval()
        
        # === 步骤1：生成样本 ===
        # 获取samples_list，每个samples包含一个group的样本数据,xing zhuang
        samples_list = self.generate_samples(inputs)
        
        # === 步骤2：初始化批次数据容器 ===
        batch_prompt_response_ids = []
        batch_attention_mask = []
        batch_action_mask = []
        batch_advantages = []
        batch_old_action_log_probs = []
        batch_ref_action_log_probs = []
        
        # === 步骤3：遍历每个group，计算优势和概率 ===
        # len(samples_list) = batch_size 
        # len(samples[0]) = num_generations
        for samples in samples_list:
            # 提取samples数据
            prompt_response_ids = samples.prompt_response_ids  # [num_generations, seq_len]
            response_ids = samples.response_ids                # [num_generations, response_len]
            answer = samples.answer
            attention_mask = samples.attention_mask            # [num_generations, seq_len]
            action_mask = samples.action_mask                  # [num_generations, response_len]
            num_actions = samples.num_actions
            prompt = samples.prompt
            
            # 添加到批次容器
            batch_prompt_response_ids.append(prompt_response_ids)
            batch_attention_mask.append(attention_mask)
            batch_action_mask.append(action_mask)
            
            with torch.no_grad():
                # === 3.1 计算当前策略模型的动作对数概率(old policy) ===
                old_action_log_probs = self.get_action_log_probs(
                    self.model, prompt_response_ids, attention_mask, num_actions
                )
                batch_old_action_log_probs.append(old_action_log_probs)
                
                # === 3.2 可选：计算参考模型的动作对数概率(用于KL约束) ===
                if self.ref_model:
                    ref_action_log_probs = self.get_action_log_probs(
                        self.ref_model, prompt_response_ids, attention_mask, num_actions
                    )
                    batch_ref_action_log_probs.append(ref_action_log_probs)
                # === 3.3 计算奖励(支持多奖励函数) ===
                # 存储各奖励函数对group内各响应的评分
                rewards_per_func = torch.zeros(
                    len(self.reward_funcs), self.args.num_generations, device=self.args.device
                )
                
                # 解码生成文本
                response_texts = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
                prompt_texts = [prompt] * len(response_texts)
                # prompt拼接上response, 形成完整的prompt_response_texts
                prompt_response_texts = [prompt + response for prompt, response in zip(prompt_texts, response_texts)]
                
                # 遍历所有奖励函数,计算单个组的奖励函数得分
                for i, (reward_func, reward_tokenizer) in enumerate(
                    zip(self.reward_funcs, self.reward_tokenizers)
                ):
                    # 情况1：奖励模型(PreTrainedModel)
                    if isinstance(reward_func, PreTrainedModel):
                        with torch.inference_mode():
                            reward_model_inputs = reward_tokenizer(
                                prompt_response_texts, return_tensors="pt", padding=True
                            )
                            rewards_per_func[i] = reward_func(
                                **reward_model_inputs.to(self.args.device)
                            ).logits.squeeze(-1)
                    
                    # 情况2：自定义奖励函数
                    else:
                        answers = [answer] * len(prompt_texts)
                        output_reward_func = reward_func(
                            prompts=prompt_texts, responses=response_texts, answers=answers
                        )
                        # 处理None值(某些奖励函数可能返回None)
                        output_reward_func = [reward if reward is not None else torch.nan 
                                             for reward in output_reward_func]
                        rewards_per_func[i] = torch.tensor(
                            output_reward_func, dtype=torch.float32, device=self.args.device
                        )
                # === 3.4 多奖励加权融合 ===
                # rewards_per_func: [num_funcs, num_generations]
                if not self.args.reward_weights:
                    self.args.reward_weights = [1.0] * len(self.reward_funcs)
                if len(self.args.reward_weights) != len(self.reward_funcs):
                    raise ValueError("奖励权重数量必须与奖励函数数量一致")
                
                # 加权求和：每个奖励函数乘以对应权重
                rewards = rewards_per_func * torch.tensor(
                    self.args.reward_weights, dtype=torch.float32, device=rewards_per_func.device
                ).unsqueeze(1)
                
                # 汇总所有奖励函数的得分
                rewards = rewards.sum(dim=0)  # [num_generations]
                print(f'rewards: {rewards}')
                
                # 及时释放中间变量，节省内存
                del rewards_per_func, response_texts, prompt_texts, prompt_response_texts
                
                # === 3.5 计算优势(GRPO核心) ===
                # 组内标准化：优势 = (奖励 - 组均值) / 组标准差
                mean_group_rewards = rewards.mean()
                std_group_rewards = rewards.std()
                
                # GRPO的优势是**句子级别**的，而非token级别
                # 同一个响应的所有token共享相同的优势值
                advantages = (rewards - mean_group_rewards) / (std_group_rewards + 1e-8)  # [num_generations]
                batch_advantages.append(advantages)
        # === 步骤4：合并所有group的数据 ===
        return {
            "prompt_response_ids": torch.cat(batch_prompt_response_ids, dim=0),
            "attention_mask": torch.cat(batch_attention_mask, dim=0),
            "action_mask": torch.cat(batch_action_mask, dim=0),
            "old_action_log_probs": torch.cat(batch_old_action_log_probs, dim=0),
            "ref_action_log_probs": torch.cat(batch_ref_action_log_probs, dim=0) if self.ref_model else None,
            "advantages": torch.cat(batch_advantages, dim=0),
        }
    
    def compute_loss(self, model, inputs):
        """
        计算GRPO损失
        
        损失组成：
        1. PPO-Clip损失：min(ratio * A, clip(ratio) * A)
        2. 可选的KL散度惩罚：beta * KL(ref || policy)
        
        公式：
        - ratio = exp(log π_θ - log π_old)
        - clip(ratio) = clamp(ratio, 1-ε, 1+ε)
        - L = -min(ratio * A, clip(ratio) * A) + β * KL
        
        参数:
            model: 当前策略模型
            inputs: generate_experiences返回的数据字典
        
        返回:
            loss: 标量损失值
        """
        # === 步骤1：计算当前策略的动作对数概率 ===
        prompt_response_ids = inputs['prompt_response_ids']
        attention_mask = inputs['attention_mask']
        action_mask = inputs['action_mask']
        num_actions = action_mask.size(1)
        
        # 前向传播获取当前策略的log概率,shape: [batch_size * num_generations, num_actions]
        action_log_probs = self.get_action_log_probs(
            model, prompt_response_ids, attention_mask, num_actions
        )
        
        # === 步骤2：可选的KL散度惩罚 ===
        if self.args.beta != 0.0:
            ref_action_log_probs = inputs['ref_action_log_probs']
            log_ratio = ref_action_log_probs - action_log_probs  # log(π_ref / π_θ)
            log_ratio = log_ratio * action_mask
            
            # KL散度的近似：k3 = exp(log_ratio) - 1 - log_ratio
            # 这是 KL(ref || policy) 的泰勒展开形式
            k3 = log_ratio.exp() - 1 - log_ratio
        
        # === 步骤3：PPO-Clip损失 ===
        advantages = inputs['advantages']  # [batch_size * num_generations]
        
        # 重要性采样比率：ratio = π_θ(a|s) / π_old(a|s)
        old_action_log_probs = (inputs['old_action_log_probs'] 
                                if self.args.num_iterations > 1 
                                else action_log_probs.detach())
        
        coef_1 = torch.exp(action_log_probs - old_action_log_probs)  # ratio
        coef_2 = torch.clamp(coef_1, 1 - self.args.clip_eps, 1 + self.args.clip_eps)  # clipped ratio
        
        # 两种损失：原始损失 vs 裁剪损失
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)  # ratio * A
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)  # clip(ratio) * A
        
        # 取较小值(更保守的策略更新)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        
        # 仅对有效token计算损失
        per_token_loss = per_token_loss * action_mask
        
        # 添加KL惩罚项
        if self.args.beta != 0.0:
            per_token_loss = per_token_loss + self.args.beta * k3
        
        # === 步骤4：聚合损失 ===
        # 每个序列的损失 = token损失之和 / 有效token数
        loss = per_token_loss.sum(dim=1) / action_mask.sum(dim=1)  # [batch_size * num_generations]
        loss = loss.mean()  # 批次平均损失
        
        return loss

    def get_action_log_probs(self, model, input_ids, attention_mask, num_actions):
        """
        计算模型输出token的对数概率
        
        流程：
        1. 前向传播获取logits
        2. 计算log_softmax
        3. 提取实际生成token的对数概率
        4. 仅保留response部分的概率
        
        参数:
            model: 语言模型
            input_ids: 输入token ids [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            num_actions: 响应序列长度
        
        返回:
            action_log_probs: 响应token的对数概率 [batch_size, num_actions]
        """
        # 前向传播（使用混合精度加速）
        with torch.cuda.amp.autocast():
            output = model(input_ids, attention_mask=attention_mask)
            # torch.Size([4, 512, 151936])
            logits = output.logits  # [batch_size, seq_len, vocab_size]
        
        # 计算对数概率分布
        # 最后一个位置应该为eos_token_id
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)  # 去掉最后一个位置
        
        # 提取实际生成token的对数概率, 自回归模型，上一个时刻的输出是下一个时刻的输入
        # input_ids[:, 1:] 是真实生成的token (shifted by 1)
        log_probs_labels = log_probs.gather(
            dim=-1, index=input_ids[:, 1:].unsqueeze(-1)
        )
        
        # 仅保留response部分的对数概率
        action_log_probs = log_probs_labels.squeeze(-1)[:, -num_actions:]
        
        # 及时释放大张量，节省内存
        del logits, log_probs, log_probs_labels
        
        return action_log_probs
    
    def train_step(self, model, inputs, optimizer, step):
        """
        执行单步训练
        
        流程：
        1. 计算损失
        2. 反向传播(支持梯度累积)
        3. 更新参数
        
        参数:
            model: 待训练模型
            inputs: 训练数据
            optimizer: 优化器
            step: 当前步数(用于判断是否执行优化器更新)
        """
        model.train()
        
        # === 步骤1：计算损失（使用混合精度） ===
        with torch.cuda.amp.autocast():
            loss = self.compute_loss(model, inputs)
        
        # === 步骤2：梯度累积(缩放损失) ===
        loss = loss / self.args.gradient_accumulation_steps
        
        # === 步骤3：混合精度反向传播 ===
        self.scaler.scale(loss).backward()
        
        # === 步骤4：条件更新参数(仅在累积足够步数后) ===
        if (step + 1) % self.args.gradient_accumulation_steps == 0:
            # 梯度裁剪（防止梯度爆炸，提升稳定性）
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 更新参数
            self.scaler.step(optimizer)
            self.scaler.update()
            optimizer.zero_grad()
            
            # 打印训练进度（每10步打印一次，减少IO阻塞）
            if self.update_steps % 10 == 0 or self.update_steps == 1:
                print(f"step: {self.update_steps}/{self.global_steps}  grpo_loss: {loss.item() * self.args.gradient_accumulation_steps:.8f}")
        
        # 释放显存
        torch.cuda.empty_cache()

    def train(self):
        """
        主训练循环
        
        流程：
        1. 外层循环：遍历epoch
        2. 中层循环：遍历数据批次
        3. 生成经验(采样+计算奖励+优势)
        4. 内层循环：多次迭代训练(复用采样数据)
        5. 周期性保存检查点
        """
        # 计算总训练步数
        self.global_steps = (self.args.num_iterations * self.args.epoch * 
                            len(self.train_dataset) // 
                            (self.args.batch_size * self.args.gradient_accumulation_steps))
        
        for _ in range(self.args.epoch):
            dataloader = DataLoader(
                self.train_dataset, 
                batch_size=self.args.batch_size, 
                shuffle=True,
                num_workers=4,  # 多进程加载数据，加速数据准备
                pin_memory=True if self.args.device == 'cuda' else False  # GPU加速数据传输
            )
            
            for idx, batch in enumerate(dataloader):
                # === 步骤1：生成经验数据 ===
                """{'prompt_response_ids': tensor([[151643, 151643, 151643,  ..., 100158,   3837, 104710],
                    [151643, 151643, 151643,  ..., 100369, 107278,   3837],
                    [151643, 151643, 151643,  ..., 101492,   3407, 101889],
                    [151643, 151643, 151643,  ...,  42411, 104209,     21]]), 'attention_mask': tensor([[0, 0, 0,  ..., 1, 1, 1],
                    [0, 0, 0,  ..., 1, 1, 1],
                    [0, 0, 0,  ..., 1, 1, 1],
                    [0, 0, 0,  ..., 1, 1, 1]]), 'action_mask': tensor([[1, 1, 1,  ..., 1, 1, 1],
                    [1, 1, 1,  ..., 1, 1, 1],
                    [1, 1, 1,  ..., 1, 1, 1],
                    [1, 1, 1,  ..., 1, 1, 1]]), 'old_action_log_probs': tensor([[-5.7788e-04, -4.6848e-05, -1.2721e+00,  ..., -6.4996e-01,
                    -1.7188e+00, -1.1591e+00],
                    [-5.7788e-04, -4.6848e-05, -3.4095e-01,  ..., -2.8362e+00,
                    -1.6663e-01, -3.6102e-01],
                    [-4.0475e-04, -1.7166e-05, -3.7063e-01,  ..., -1.4287e-03,
                    -4.9276e-01, -1.7851e+00],
                    [-4.0475e-04, -1.7166e-05, -3.7063e-01,  ..., -1.4440e+00,
                    -4.5085e-02, -1.0126e-03]]), 'ref_action_log_probs': None, 'advantages': tensor([0., 0., 0., 0.])}"""
                inputs = self.generate_experiences(batch)
                
                # === 步骤2：存入缓冲区(用于梯度累积) ===
                # 首先需要存储梯度累积的inputs, gradient_accumulation_steps=4
                # 当idx=0时，存储inputs到input_buffer[0]
                # 当idx=1时，存储inputs到input_buffer[1]
                # 当idx=2时，存储inputs到input_buffer[2]
                # 当idx=3时，存储inputs到input_buffer[3]
                # 当idx=4时，存储inputs到input_buffer[0]
                # 当idx=5时，存储inputs到input_buffer[1]
                # 当idx=6时，存储inputs到input_buffer[2]
                # 当idx=7时，存储inputs到input_buffer[3]
                self.input_buffer[idx % self.args.gradient_accumulation_steps] = inputs
                
                # === 步骤3：累积足够步数后，执行训练 ===
                if (idx + 1) % self.args.gradient_accumulation_steps == 0:
                    """
                    step=0 → 计算 loss → loss /=4 → backward（累加 1/4 梯度）→ 不 step
                    step=1 → 同上，累加第 2/4 梯度
                    step=2 → 累加第 3/4
                    step=3 → 累加第 4/4 → (3+1)%4==0 → optimizer.step() + zero_grad() → 打印 loss
                    第二次迭代（iter=1）：
                        又对相同的 4 个 inputs 各执行一次 train_step
                        梯度再次累加（又累加了 4 次 1/4 梯度）
                        在 step=3 时再次 step() + zero_grad() + 打印
                    """
                    # 多次迭代训练(复用采样数据，提高样本利用率)
                    for _ in range(self.args.num_iterations):
                        for step, inputs in enumerate(self.input_buffer):
                            self.train_step(self.model, inputs, self.optimizer, step)
                        
                        self.update_steps += 1
                        
                        # === 步骤4：周期性保存检查点 ===
                        if self.update_steps % self.args.save_steps == 0:
                            checkpoint_path = f'{self.args.output_dir}/checkpoint_{self.update_steps}'
                            self.model.save_pretrained(checkpoint_path)
                            self.tokenizer.save_pretrained(checkpoint_path)
                        
                del inputs  # 释放内存
    
    def save_model(self):
        """保存最终模型"""
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
    

