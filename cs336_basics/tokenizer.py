import regex as re
# from pretokenization_example import find_chunk_boundaries
from tqdm import *

import os
from typing import BinaryIO


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
# PAT用法：
# text_example = "some text that i'll pre-tokenize"
# # 使用 finditer 遍历匹配项
# matches = re.finditer(PAT, text_example)
# for match in matches:
#     print(f"匹配内容: '{match.group()}', 位置: {match.span()}")

# print(max([("A", "B"), ("A", "C"), ("B", "ZZ"), ("BA", "A")]))

def pretokenize(text:str) -> list[bytes]:
    # 使用正则表达式进行预分词
    match_tokens = re.finditer(PAT, text)
    str_tokens = [match.group() for match in match_tokens]
    byte_tokens = [s.encode("utf-8") for s in str_tokens]
    # print(byte_tokens)
    return byte_tokens

def Parallelizing_chunking(path:str):
    with open(path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        # print(boundaries)
        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        chunks = []
        for start, end in tqdm(zip(boundaries[:-1], boundaries[1:])):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunk = pretokenize(chunk)
            # chunk = chunk.replace("\n", "")
            # chunk = chunk.split(' ')
            # Run pre-tokenization on your chunk and store the counts for each pre-token
            chunks.append(chunk)
        return chunks

text = "some text that i'll pre-tokenize"
class BPETokenizer:
    # def __init__(self, vocab_size: int, special_tokens: list[str] | None = None):
    def __init__(self, vocab_size: int, special_tokens: list[str]):
        """
        vocab_size: int A positive integer that defines the maximum final vocabulary size (including the
            initial byte vocabulary, vocabulary items produced from merging, and any special tokens).
        special_tokens: list[str] A list of strings to add to the vocabulary. These special tokens do not
            otherwise affect BPE training.

        Your BPE training function should return the resulting vocabulary and merges:

        vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]]
        vocab: dict[int, bytes] The tokenizer vocabulary, a mapping from int (token ID in the vocabulary) to bytes (token bytes).
        merges: list[tuple[bytes, bytes]] A list of BPE merges produced from training. Each list item
            is a tuple of bytes (<token1>, <token2>), representing that <token1> was merged with
            <token2>. The merges should be ordered by order of creation
        """
        # 用于储存词对对应的出现次数，比如((l,o):5)
        self.dic_wordnear2num = dict[tuple[bytes], int]

        self.freq = {}
        self.dic_word2num = {}
        self.char_seq_freq = {}  # 转换后：{('l','o','w','</w>'): 1, ...}
        self.pair_freq = {}
        
        


    def train(self, input_path: str):
        """
        input_path: str Path to a text file with BPE tokenizer training data.
        """
        # print(input_path)
        chunks = Parallelizing_chunking(input_path)
        # print(chunks)
        for c in chunks:
            for token_bytes in c:
                # token_bytes 是字节串，如 b'low'
                # print(token_bytes)
                # 方法A：使用字节作为键
                self.dic_word2num[token_bytes] = self.dic_word2num.get(token_bytes, 0) + 1
        # 获取到每个单词对应的频率，单词前的空格不能省略
        # {b'low': 1, b' low': 4, b' lower': 2, b' widest': 3, b' newest': 6, b'\n': 1}
        # print(self.dic_word2num)
        
        # 将单词分解为字符，并添加结束标记</w>，同时保留前导空格信息
        for token_bytes, freq in self.dic_word2num.items():
            # print(token_bytes, freq)
            if token_bytes == b'\n':  # 跳过换行
                continue
            token = token_bytes.decode()  # 将字节串转换为字符串
            chars = []
    
            if token.startswith(' '):  # 有前导空格
                chars.append('▁')
                token = token.lstrip()  # 去掉前导空格
            
            # 添加字符和结束标记
            chars.extend(list(token))
            chars.append('</w>')
            
            # 保存为元组
            self.char_seq_freq[tuple(chars)] = freq
            # {('l', 'o', 'w', '</w>'): 1, ('▁', 'l', 'o', 'w', '</w>'): 4}
            # print(self.char_seq_freq)
        
        # 单词表
        word_list = []
        # 每个单词对应的频率，下标与单词表对应
        stats = []
        # merge对应的单词下标，优化效率，避免每次都要遍历整个词典
        indices = []
        for char_seq, freq in self.char_seq_freq.items():
            word_list.append(char_seq)
            stats.append(freq)

        # for char_seq, freq in self.char_seq_freq.items():
        #     # print(char_seq, freq)
        #     for i in range(len(char_seq) - 1):
        #         pair = (char_seq[i], char_seq[i + 1])
        #         self.pair_freq[pair] = self.pair_freq.get(pair, 0) + freq  # get是一个安全取值的函数
        
        for idx, char_seq in enumerate(word_list):
            # print(char_seq, freq)
            freq = stats[idx]
            for i in range(len(char_seq) - 1):
                pair = (char_seq[i], char_seq[i + 1])
                indices.append(idx)
                self.pair_freq[pair] = self.pair_freq.get(pair, 0) + freq  # get是一个安全取值的函数
        
        print(self.pair_freq)
        print(len(self.pair_freq))
        # print(indices)
        # print(stats)
        # print(max(self.pair_freq.items(), key=lambda x: x[1]))
        
        num_merges = 10
        merges = []
        while(len(self.pair_freq) < num_merges):
        # 统计哪两个字符对出现的频率最高，进行合并，更新char_seq_freq和pair_freq，直到达到预设的vocab_size
            
            best_pair = max(self.pair_freq.items(), key=lambda x: x[1])
            # print(best_pair)
            
            # 若所有频率都清空，已经合并完毕，则停止
            if(not stats):
                break
            
            
            

# print(pretokenize(text_example))

path = "../data/TinyStoriesV2-GPT4-valid.txt"
test_path = "./test.txt"
test = "low low low low low lower lower widest widest widest newest newest newest newest newest newest"
# chunks = Parallelizing_chunking(path)
# print(chunks)

tokenizer = BPETokenizer(vocab_size=1000, special_tokens=["<|endoftext|>", "<|pad|>"])
tokenizer.train(test_path)

# freq = {}
# dic_word2num = {}
# pre_test = pretokenize(test)
# print(test)
# words = test.split()
# print(words)
# dic_word2num = {}

# for word in words:
#     # 直接创建字节元组
#     # print(word)
#     # print(type(word))
#     chars_tuple = tuple(bytes([ord(ch)]) for ch in word)
#     print(chars_tuple)
#     dic_word2num[chars_tuple] = dic_word2num.get(chars_tuple, 0) + 1
# print(dic_word2num)
