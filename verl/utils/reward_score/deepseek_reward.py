import math
import re
from collections import Counter

from Levenshtein import ratio as levenshtein_ratio

TOKENIZER = AutoTokenizer.from_pretrained("microsoft/Phi-4-reasoning-plus")

L_MAX= 31744
def parse_solution_with_box(solution_str: str) -> tuple[str | None, str | None, bool]:
    """
    <think>{思考過程}</think>{...}\\boxed{答え} という形式の文字列を解析します。

    Args:
        solution_str: モデルが生成した出力文字列。

    Returns:
        タプル (thinking_process, answer, is_format_valid)。
        フォーマットが不正な場合は (None, None, False) を返します。
    """
    # <think>タグ内の思考過程と、\boxed{}タグ内の答えを抽出する正規表現
    # '.*?' は、<think>タグと\boxed{}タグの間の任意の文字列にマッチします。
    # re.DOTALL は、'.'が改行文字にもマッチするようにします。
    pattern = "<think>(.*?)</think>.*?\\boxed\{(.*?)\}"
    match = re.search(pattern, solution_str, re.DOTALL)

    if match:
        # グループ1が<think>タグの内容
        thinking_process = match.group(1).strip()
        # グループ2が\boxed{}タグの内容
        answer = match.group(2).strip()
        is_format_valid = True
        return thinking_process, answer, is_format_valid
    else:
        # パターンにマッチしない場合はフォーマットが不正と判断
        return None, None, False
def parse_solution(solution_str: str) -> tuple[str | None, str | None, bool]:
    """
    <think>...</think>{answer} という構造の文字列を解析する。

    Args:
        solution_str: モデルの生成出力。

    Returns:
        タプル (thinking_process, answer, is_format_valid)
    """
    # 正規表現を使用して<think>ブロックとそれに続く回答を抽出
    match = re.search(r"<think>(.*?)</think>(.*)", solution_str, re.DOTALL)

    if match:
        thinking_process = match.group(1).strip()
        answer = match.group(2).strip()
        return thinking_process, answer, True
    else:
        # thinkタグが見つからない、または形式が不正
        return None, None, False


def compute_score(solution_str: str, ground_truth: str, truth_reasoning: str):
    """
    Phi-4-reasoning論文で説明されている報酬関数に基づいて最終的なスコアを計算します。

    Args:
        solution_str: モデルから生成された完全なテキスト。(tokenizedではなく、文字列形式)
        ground_truth: 正解。
        is_incomplete: 生成がシーケンス終了トークンなしで不完全な場合はTrue。
    """
    # 1. 出力文字列を解析し、フォーマットを検証
    thinking_process, answer, is_format_valid = parse_solution(solution_str)
    thinking_process_truth,_, is_format_valid_truth = parse_solution(truth_reasoning)
    if is_format_valid_truth:
        truth_reasoning = thinking_process_truth
    # 2. フォーマット違反のオーバーライドを処理
    # <think>タグが不正な場合は is_format_valid が False になる
    if is_format_valid:
        r_format = 1.0
    else:
        r_format = 0
    # 3. 回答が正解かどうかを報酬に反映
    is_correct = (answer is not None and str(answer) == str(ground_truth))
    if is_correct:
        # 正解の場合、正解度報酬を1.0にスケーリング
        r_acc_scaled = 1.0
    else:
        r_acc_scaled = 0.0  # Assign a default value for r_acc_scaled
    # Levenshtein距離を使用してスコアを計算
    r_leven = levenshtein_ratio(str(thinking_process), str(truth_reasoning))
    
    final_score = r_format + r_acc_scaled + r_leven # 平均を取ることでスコアを正規化

    return final_score