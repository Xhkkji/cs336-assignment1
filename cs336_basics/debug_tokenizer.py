from tokenizer_others import *


# 测试用例1：简单英文句子
def test_pretokenize():
    text1 = "I don't like apples"
    result1 = iter_pretokenize(text1)
    print(f"输入: '{text1}'")
    print(f"输出: {result1}")
    print(f"解码回字符串: {[token.decode('utf-8') for token in result1]}")
    print()

special_tokens = ["<|endoftext|>", "<|unk|>", "<|pad|>",]
t = BPETokenizer(1000, special_tokens)
t.train("../data/TinyStoriesV2-GPT4-valid.txt")

# test_pretokenize()