#!/usr/bin/env bash
# sync-and-protect.sh
# 目标：
#   1) 同步 upstream/main -> 本地 main
#   2) 把 main 更新带到开发分支（默认 rebase）
#   3) 保护并恢复本地未提交改动（可含未跟踪文件）
#   4) 可选推送 main 与 dev，保护学习分支不丢失

set -euo pipefail
IFS=$'\n\t'

# ─── 默认配置（可通过环境变量覆盖） ───────────────────────────────────
: "${UPSTREAM_REMOTE:=upstream}"
: "${UPSTREAM_BRANCH:=main}"
: "${DEFAULT_DEV_BRANCH:=dev_0905}"

PUSH=true
PUSH_MAIN=true
USE_REBASE=true
FORCE_PUSH_MAIN=false
STASH_UNTRACKED=true
RETURN_TO_ORIG_BRANCH=true
QUIET=false

DEV_BRANCH="$DEFAULT_DEV_BRANCH"
ORIG_BRANCH=""
STASHED=false
TOTAL_STEPS=6
CURRENT_STEP=0

STATUS_PREFLIGHT="未执行"
STATUS_STASH="未执行"
STATUS_MAIN="未执行"
STATUS_DEV="未执行"
STATUS_PUSH_DEV="未执行"
STATUS_RESTORE="未执行"

usage() {
    echo "用法: $0 [dev-branch] [options]"
    echo ""
    echo "选项："
    echo "  --no-push              不推送任何内容到 origin"
    echo "  --no-push-main         不推送 main 到 origin"
    echo "  --allow-force-main     允许 main 使用 --force-with-lease（默认关闭）"
    echo "  --merge                开发分支使用 merge（默认 rebase）"
    echo "  --no-stash-untracked   stash 时不包含未跟踪文件（默认包含）"
    echo "  --stay-on-dev          结束后停在开发分支（默认切回原分支）"
    echo "  --quiet                精简输出（仅显示步骤与摘要）"
    echo "  --help, -h             显示帮助"
    echo ""
    echo "关键场景示例："
    echo "  1) 日常同步（推荐：同步 main + 更新 dev + 推送备份）"
    echo "     $0 dev_0905"
    echo ""
    echo "  2) 仅本地同步，不推远端（演练/测试流程）"
    echo "     $0 dev_0905 --no-push"
    echo ""
    echo "  3) 保持停在开发分支，便于继续学习"
    echo "     $0 dev_0905 --stay-on-dev"
    echo ""
    echo "  4) 不推 main，只推 dev（常见于只想备份学习分支）"
    echo "     $0 dev_0905 --no-push-main"
    echo ""
    echo "  5) 开发分支改用 merge（不改写 dev 提交历史）"
    echo "     $0 dev_0905 --merge"
    echo ""
    echo "  6) stash 不包含未跟踪文件（保留本地临时文件在工作区）"
    echo "     $0 dev_0905 --no-stash-untracked"
    echo ""
    echo "  7) main 允许强推（仅在你明确知道风险时使用）"
    echo "     $0 dev_0905 --allow-force-main"
    echo ""
    echo "  8) 组合场景：不推 main + 停留 dev + merge 更新"
    echo "     $0 dev_0905 --no-push-main --stay-on-dev --merge"
    echo ""
    echo "  9) 只同步到本地并停留 dev（离线学习常用）"
    echo "     $0 dev_0905 --no-push --stay-on-dev"
    echo ""
    echo "  10) 恢复默认行为（无需参数）"
    echo "     $0"
}

info() {
    $QUIET || echo "→ $*"
}

ok() {
    $QUIET || echo "✓ $*"
}

warn() {
    echo "! $*"
}

fail() {
    echo "× $*" >&2
    exit 1
}

step() {
    CURRENT_STEP=$((CURRENT_STEP + 1))
    echo ""
    echo "============================================================"
    echo "[$CURRENT_STEP/$TOTAL_STEPS] $*"
    echo "============================================================"
}

out() {
    $QUIET || echo "$*"
}

run_git() {
    if $QUIET; then
        git "$@" >/dev/null
    else
        git "$@"
    fi
}

is_git_busy() {
    [[ -d .git/rebase-merge ]] \
    || [[ -d .git/rebase-apply ]] \
    || git rev-parse --verify MERGE_HEAD >/dev/null 2>&1
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --no-push) PUSH=false ;;
            --no-push-main) PUSH_MAIN=false ;;
            --allow-force-main) FORCE_PUSH_MAIN=true ;;
            --merge) USE_REBASE=false ;;
            --no-stash-untracked) STASH_UNTRACKED=false ;;
            --stay-on-dev) RETURN_TO_ORIG_BRANCH=false ;;
            --quiet) QUIET=true ;;
            --help|-h) usage; exit 0 ;;
            -*)
                fail "未知选项: $1"
                ;;
            *)
                DEV_BRANCH="$1"
                ;;
        esac
        shift
    done
}

preflight_checks() {
    info "检查远程与分支状态..."
    git rev-parse --is-inside-work-tree >/dev/null 2>&1 || {
        fail "当前目录不是 Git 仓库，请先 cd 到仓库根目录再运行脚本。"
    }

    ORIG_BRANCH="$(git branch --show-current)"

    git remote get-url origin &>/dev/null || {
        fail "未找到 origin remote，请先执行：git remote add origin <你的仓库URL>"
    }

    git remote get-url "$UPSTREAM_REMOTE" &>/dev/null || {
        fail "未找到 upstream remote，请先执行：git remote add upstream <上游URL>"
    }

    git show-ref --verify --quiet "refs/heads/$DEV_BRANCH" || {
        git branch --list >&2
        fail "本地没有分支 $DEV_BRANCH"
    }

    if is_git_busy; then
        echo "  建议先执行：git status" >&2
        echo "  rebase 场景：git rebase --continue 或 git rebase --abort" >&2
        echo "  merge  场景：git merge --continue  或 git merge --abort" >&2
        fail "检测到仓库当前处于 rebase/merge 进行中，请先处理完再运行脚本。"
    fi

    STATUS_PREFLIGHT="成功"
    ok "前置检查通过"
}

print_summary() {
    $QUIET && return
    echo "┌──────────────────────────────────────┐"
    echo "│ 同步 upstream → main → $DEV_BRANCH   │"
    echo "└──────────────────────────────────────┘"
    echo "  当前分支： $ORIG_BRANCH"
    echo "  开发分支： $DEV_BRANCH"
    echo "  推送保护： $( $PUSH && echo '开启' || echo '关闭 (--no-push)' )"
    echo "  main 推送： $( $PUSH_MAIN && echo '开启' || echo '关闭 (--no-push-main)' )"
    echo "  main 强推： $( $FORCE_PUSH_MAIN && echo '开启 (--allow-force-main)' || echo '关闭（默认安全推送）' )"
    echo "  更新方式： $( $USE_REBASE && echo 'rebase (推荐学习场景)' || echo 'merge' )"
}

stash_workspace_if_needed() {
    local has_untracked=false
    if $STASH_UNTRACKED && [[ -n "$(git ls-files --others --exclude-standard)" ]]; then
        has_untracked=true
    fi

    if ! git diff --quiet --exit-code || ! git diff --cached --quiet || $has_untracked; then
        info "检测到未提交更改，自动 stash 保护..."
        local stash_msg="WIP-auto-$(date '+%Y%m%d-%H%M') before sync"
        if $STASH_UNTRACKED; then
            run_git stash push -u -m "$stash_msg" || fail "stash 失败"
        else
            run_git stash push -m "$stash_msg" || fail "stash 失败"
        fi
        STASHED=true
        STATUS_STASH="已stash"
        ok "工作区已保护到 stash"
        return
    fi

    STATUS_STASH="无需stash"
    ok "工作区干净，无需 stash"
}

update_main() {
    info "更新 main 分支..."
    run_git fetch --prune "$UPSTREAM_REMOTE"
    run_git fetch --prune origin

    if git merge-base --is-ancestor "${UPSTREAM_REMOTE}/${UPSTREAM_BRANCH}" main &>/dev/null; then
        out "  main 已是最新的"
        STATUS_MAIN="已最新"
        ok "main 无需更新"
        return
    fi

    run_git checkout main
    out "  rebase upstream/$UPSTREAM_BRANCH -> main ..."
    run_git rebase "${UPSTREAM_REMOTE}/${UPSTREAM_BRANCH}" || {
        echo "  请手动解决后执行：git rebase --continue 或 git rebase --abort" >&2
        fail "main rebase 冲突"
    }

    if ! $PUSH || ! $PUSH_MAIN; then
        out "  (跳过 main push)"
        STATUS_MAIN="已更新(未推送)"
        ok "main 已更新（按参数跳过推送）"
        return
    fi

    if $FORCE_PUSH_MAIN; then
        info "推送 main 到 origin (--force-with-lease)..."
        run_git push --force-with-lease origin main || fail "main 强推失败（可能远程有新提交）"
    else
        info "安全推送 main 到 origin（仅快进，不重写历史）..."
        run_git push origin main || fail "main 推送失败（非快进或远程有新提交）"
    fi

    STATUS_MAIN="已更新并推送"
    ok "main 已更新并推送"
}

wait_user_finish_rebase() {
    echo ""
    echo "┌────────────────────────────────────────────────────────────┐"
    echo "│                  ⚠️  rebase 冲突！                        │"
    echo "│  Git 已自动 stash 你的未提交更改，请现在手动解决冲突      │"
    echo "└────────────────────────────────────────────────────────────┘"
    echo ""
    echo "推荐解决步骤（在 VSCode / GitLens 中操作最方便）："
    echo "  1. git status                  查看冲突文件列表"
    echo "  2. 打开冲突文件，解决标记（<<<<<<< HEAD ... ======= ... >>>>>>>）"
    echo "     • modify/delete 冲突： git add（保留改动）或 git rm（跟随上游删除）"
    echo "  3. git add <已解决的文件>"
    echo "  4. git rebase --continue       继续处理下一个 commit"
    echo "     （或 git rebase --skip       跳过当前 commit）"
    echo "     （或 git rebase --abort      完全放弃本次 rebase）"
    echo ""
    read -r -p "解决完成并执行 --continue 后，按 Enter 继续..."

    is_git_busy && fail "检测到 rebase/merge 仍未结束，请先处理完。"
    [[ "$(git branch --show-current)" == "$DEV_BRANCH" ]] \
        || fail "当前分支不是 $DEV_BRANCH，可能执行了 abort 或切到了其他分支。"
}

update_dev_branch() {
    info "更新 $DEV_BRANCH ..."
    run_git checkout "$DEV_BRANCH"

    if git merge-base --is-ancestor main "$DEV_BRANCH" &>/dev/null; then
        out "  $DEV_BRANCH 已包含 main 最新内容"
        STATUS_DEV="已最新"
        ok "$DEV_BRANCH 无需更新"
        return
    fi

    if $USE_REBASE; then
        out "  rebase main -> $DEV_BRANCH (使用 --autostash)..."
        if run_git rebase --autostash main; then
            out "  rebase 成功"
        else
            wait_user_finish_rebase
            out "  rebase 已完成，autostash 自动恢复工作区"
        fi
        STATUS_DEV="rebase完成"
        ok "$DEV_BRANCH 已完成 rebase 更新"
        return
    fi

    out "  merge main -> $DEV_BRANCH ..."
    run_git merge --no-edit main || fail "merge 冲突！请手动解决后 git commit"
    STATUS_DEV="merge完成"
    ok "$DEV_BRANCH 已完成 merge 更新"
}

push_dev_branch() {
    if ! $PUSH; then
        out "  (--no-push，跳过 $DEV_BRANCH push)"
        STATUS_PUSH_DEV="跳过"
        ok "已按参数跳过 dev 推送"
        return
    fi

    info "推送 $DEV_BRANCH 到 origin (--force-with-lease)..."
    if run_git push --force-with-lease origin "$DEV_BRANCH"; then
        STATUS_PUSH_DEV="成功"
        ok "已推送 $DEV_BRANCH 到远程"
    else
        STATUS_PUSH_DEV="失败"
        echo "× push 失败，可能远程有新提交"
        echo "  建议先： git pull --rebase origin $DEV_BRANCH 再重试"
        fail "dev 推送失败，为避免误判已备份，脚本中止。"
    fi
}

restore_workspace() {
    if $STASHED; then
        info "恢复之前 stash 的工作区..."
        run_git stash pop || warn "pop 可能有冲突，请手动 git stash apply 并检查"
    else
        info "无需恢复 stash"
    fi

    if $RETURN_TO_ORIG_BRANCH && [[ "$ORIG_BRANCH" != "$DEV_BRANCH" ]]; then
        info "切回脚本启动前分支：$ORIG_BRANCH"
        run_git checkout "$ORIG_BRANCH"
    else
        info "保持当前分支不变"
    fi

    STATUS_RESTORE="完成"
    ok "收尾处理完成"
}

print_execution_summary() {
    echo ""
    echo "---------------- 执行摘要 ----------------"
    echo "前置检查        : $STATUS_PREFLIGHT"
    echo "工作区保护      : $STATUS_STASH"
    echo "main 更新       : $STATUS_MAIN"
    echo "dev 更新        : $STATUS_DEV"
    echo "dev 推送        : $STATUS_PUSH_DEV"
    echo "收尾恢复        : $STATUS_RESTORE"
    echo "------------------------------------------"
}

main() {
    parse_args "$@"

    step "前置检查"
    preflight_checks

    print_summary

    step "保护当前工作区"
    stash_workspace_if_needed

    step "更新 main"
    update_main

    step "更新开发分支"
    update_dev_branch

    step "推送开发分支"
    push_dev_branch

    step "恢复现场"
    restore_workspace

    print_execution_summary

    echo ""
    if $QUIET; then
        echo "✓ 同步 & 保护完成"
        echo "  当前分支： $(git branch --show-current)"
    else
        ok "同步 & 保护完成"
        echo "  当前分支： $(git branch --show-current)"
        echo "  最近提交概览："
        git log --oneline --graph -n 6
    fi
}

main "$@"