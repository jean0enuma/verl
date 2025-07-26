import math
import re
from collections import Counter

# 論文のセクション4.1および4.2から引用した定数
# 報酬関数で使用する長さのパラメータ
L_MAX = 31744
L_POS_CONTROL = 25600
L_NEG_CONTROL = 3702

# 報酬値の範囲
R_MAX_POS = 1.0
R_MIN_POS = 0.5
R_MAX_NEG = -0.5
R_MIN_NEG = -1.0

# 最終的な報酬の重み
W_ACC = 8 / 13
W_REP = 1 / 13

# 繰り返しペナルティのパラメータ
NGRAM_SIZE = 5
NGRAM_FREQ_THRESHOLD = 5


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

def _compute_repetition_penalty(text: str) -> float:
    """
    n-gramの頻度に基づいて繰り返しペナルティを計算します。
    """
    words = text.split()
    if len(words) < NGRAM_SIZE:
        return 0.0

    ngrams = [" ".join(words[i:i+NGRAM_SIZE]) for i in range(len(words) - NGRAM_SIZE + 1)]
    if not ngrams:
        return 0.0

    ngram_counts = Counter(ngrams)
    frequent_ngrams = {k: v for k, v in ngram_counts.items() if v > NGRAM_FREQ_THRESHOLD}

    if not frequent_ngrams:
        return 0.0

    term1 = len(frequent_ngrams) / len(ngrams)
    max_freq = max(frequent_ngrams.values())
    total_possible_ngrams = len(words) / NGRAM_SIZE if len(words) > 0 else 1
    term2 = max_freq / total_possible_ngrams

    penalty = -max(term1, term2)
    return penalty

def compute_score(solution_str: str, ground_truth: str, is_incomplete: bool = False):
    """
    Phi-4-reasoning論文で説明されている報酬関数に基づいて最終的なスコアを計算します。

    Args:
        solution_str: モデルから生成された完全なテキスト。
        ground_truth: 正解。
        is_incomplete: 生成がシーケンス終了トークンなしで不完全な場合はTrue。
    """
    # 1. 出力文字列を解析し、フォーマットを検証
    thinking_process, answer, is_format_valid = parse_solution(solution_str)

    # 2. フォーマット違反のオーバーライドを処理
    # <think>タグが不正な場合は is_format_valid が False になる
    if not is_format_valid:
        r_acc_scaled = -1.0
    # 生成が不完全な場合
    elif is_incomplete:
        r_acc_scaled = -0.5
    else:
        # 3. フォーマットが正常な場合、長さ認識型の正解度報酬を計算
        is_correct = (answer is not None and str(answer) == str(ground_truth))

        # 注記: 論文ではトークン長が使用されていますが、ここでは単語数を代理として使用します。
        # 正確な実装には、トークナイザが必要です。
        L = len(solution_str.split())

        if is_correct:
            rho_plus = min(1.0, max(0, L - L_POS_CONTROL) / (L_MAX - L_POS_CONTROL))
            cos_term = 0.5 * (R_MAX_POS - R_MIN_POS) * (1 + math.cos(math.pi * rho_plus))
            r_acc_scaled = R_MIN_POS + cos_term
        else:
            rho_minus = min(1.0, L / L_NEG_CONTROL)
            cos_term = 0.5 * (R_MIN_NEG - R_MAX_NEG) * (1 + math.cos(math.pi * rho_minus))
            r_acc_scaled = R_MAX_NEG + cos_term

    # 4. 繰り返しペナルティを計算 (文字列全体を対象)
    r_rep = _compute_repetition_penalty(solution_str)

    # 5. 最終的な重み付きスコアを計算
    final_score = (W_ACC * r_acc_scaled) + (W_REP * r_rep)

    return final_score