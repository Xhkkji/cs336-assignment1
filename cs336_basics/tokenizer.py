import regex as re
# from pretokenization_example import find_chunk_boundaries
from tqdm import *

import os
from typing import BinaryIO
import heapq
from collections import defaultdict

class ComparablePair:
    """可比较的pair包装类"""
    def __init__(self, pair: tuple[str, str], count: int):
        self.pair = pair
        self.count = count
    
    def __lt__(self, other):
        """定义小于比较（用于堆排序）"""
        if self.count != other.count:
            # 频率高的优先（所以这里用 > 实现最大堆）
            return self.count > other.count
        else:
            # 频率相同时，字典序大的优先
            return (self.pair[0] + self.pair[1]) > (other.pair[0] + other.pair[1])
    
    def __eq__(self, other):
        return self.count == other.count and self.pair == other.pair


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

        # self.word_pos: dict[tuple[str, ...], list[tuple[str, int]]] = {}  # 储存所有单词的位置
        self.word_pos = {}  # 储存所有单词的位置
        self.freq = {}
        self.dic_word2num = {}
        self.char_seq_freq = {}  # 将单词拆解为单个字符，key为该单词的频率转换后：{('l','o','w','</w>'): 1, ...}
        self.pair_freq = {}  # 储存所有的字节对与频率
        # key:一个列表，[单词索引，字符索引]
        self.pair_positions:dict[tuple[str, str], list[tuple[int, int]]] = {}  # 记录字节对位置
        # self.pair_positions: dict[tuple[str, str], list[tuple[int, list[int]]]] = defaultdict(list)

        
        # 堆优化
        self.heap = []
        # 惰性删除，用于标记某字符对是否有效，避免在heap中搜索性删除
        self.heap_entries:dict[tuple[str, str], any] = {}
        
    
    def add_word_pos(self, word_bytes: bytes, position: int):
        """添加一个单词的出现位置"""
        word_chars = word_bytes.decode('UTF-8')
        if word_chars not in self.word_pos:
            self.word_pos[word_chars] = []
        self.word_pos[word_chars].append(position)
        
    def pretoken(self, input_path: str):
        """
        input_path: str Path to a text file with BPE tokenizer training data.
        """
        # print(input_path)
        chunks = Parallelizing_chunking(input_path)
        # print(f'chunks: {chunks}')
        for c in chunks:
            for pos, token_bytes in enumerate(c):
                # token_bytes 是字节串，如 b'low'
                # print(token_bytes)
                # 方法A：使用字节作为键
                self.dic_word2num[token_bytes] = self.dic_word2num.get(token_bytes, 0) + 1
                # 获取到每个单词对应的频率，单词前的空格不能省略
                # {b'low': 1, b' low': 4, b' lower': 2, b' widest': 3, b' newest': 6, b'\n': 1}
                # print(self.dic_word2num)
                # 添加位置信息
                self.add_word_pos(token_bytes, pos)
        # print(f'self.word_pos: {self.word_pos}')
        
        
        # 将单词分解为字符，并添加结束标记</w>，同时保留前导空格信息
        for token_bytes, freq in self.dic_word2num.items():
            # print(token_bytes, freq)
            chars = []
            if token_bytes == b'\n':  # 跳过换行
                chars.append('<nl>')  # 或 '¶' 或其他特殊标记
                chars.append('</w>')
                self.char_seq_freq[tuple(chars)] = freq
                continue
            
            token = token_bytes.decode()  # 将字节串转换为字符串
            chars = []
    
            if token.startswith(' '):  # 有前导空格
                chars.append('▁')
                token = token.lstrip()  # 去掉前导空格，替换为下划线
            
            # 添加字符和结束标记
            chars.extend(list(token))
            chars.append('</w>')
            
            # 保存为元组
            self.char_seq_freq[tuple(chars)] = freq
            # {('l', 'o', 'w', '</w>'): 1, ('▁', 'l', 'o', 'w', '</w>'): 4}
        # print(self.char_seq_freq)
        # print(f'self.dic_word2num_len:{len(self.dic_word2num)}')
        # print(f'char_seq_len:{len(self.char_seq_freq)}')
        
           
        # 统计字节对频率
        # for char_seq, freq in self.char_seq_freq.items():
        #     print(char_seq, freq)
        #     for i in range(len(char_seq) - 1):
        #         pair = (char_seq[i], char_seq[i + 1])
        #         self.pair_freq[pair] = self.pair_freq.get(pair, 0) + freq  # get是一个安全取值的函数
        #         if pair not in self.pair_positions:
        #             self.pair_positions[pair] = []
        #         self.pair_positions[pair].append((self.word_pos[char_seq], i))
        #     print(f'pair_position:{self.pair_positions}')
        for char_seq, freq in self.char_seq_freq.items():
            # print(f'char_seq:{char_seq}, freq:{freq}')
            char_list = list(char_seq)
            for i in range(len(char_list) - 1):
                pair = (char_list[i], char_list[i + 1])
                self.pair_freq[pair] = self.pair_freq.get(pair, 0) + freq  # get是一个安全取值的函数
                if pair not in self.pair_positions:
                    self.pair_positions[pair] = []
                # self.pair_positions[pair].append((self.word_pos[char_str], i))
                self.pair_positions[pair].append((char_seq, i))

        # print(f'pair_positions:{self.pair_positions}')
        # for key, value in self.pair_positions.items():
        #     print(key, value)
        
        # 添加到堆
        for pair, count in self.pair_freq.items():
            # 只添加计数大于1的对
            if count > 1:
                item = ComparablePair(pair, count)
                entry = [count, pair]
                # 默认最小堆,自定义为最大堆
                heapq.heappush(self.heap, item)
                self.heap_entries[pair] = entry
        
        
    def get_most_frequent(self) -> tuple[str, str]:
        """快速返回频率最高的字节对，跳过已被合并的无效条目"""
        while self.heap:
            item = self.heap[0]
            pair_count = item.count
            pair = item.pair
            if pair not in self.heap_entries:
                heapq.heappop(self.heap)
                continue
            
            current_count = self.pair_freq.get(pair, 0)
            if current_count == pair_count and current_count > 1:
                return pair
            else:
                # 移除该条目
                heapq.heappop(self.heap)
                if pair in self.heap_entries:
                    del self.heap_entries[pair]
            
    def merge_pair(self, pair: tuple[str, str], new_token: str) -> int:
        """合并字节对并更新索引"""
        """以下操作针对全体单词中指定的一个pair"""
        new_token = pair[0] + pair[1]
        print(new_token)
        if pair not in self.pair_freq or not self.pair_freq[pair]:
            return 0
        
        # 按序列和位置分组
        merge_count = 0

        # print(f'self.pair_freq[pair]:{self.pair_freq[pair]}')
        # seq_idx = self.pair_positions[pair]
        # print(f'seq_idx:{seq_idx}')
        # print(f'self.pair_positions:{self.pair_positions}')
        for seq, pair_pos in self.pair_positions[pair]:
            # seq:('▁', 'w', 'i', 'd', 'e', 's', 't', '</w>')
            # 修改seq，即对char_seq进行合并操作
            seq_list = list(seq)
            # 针对一个单词中重复出现同一个pair的情况
            
            # 执行合并
            seq_list[pair_pos] = new_token
            del seq_list[pair_pos + 1]
            
            # 更新pair_position需要考虑新元素的左右pair的值
        
        
        print(self.pair_positions[pair])

            
            
            
            

            
    

    def train(self, input_path: str):
        # 预分词
        self.pretoken(input_path)
        # print(self.pair_freq)
        # print(len(self.pair_freq))
        # print(max(self.pair_freq.items(), key=lambda x: x[1]))
        merged_pair = self.get_most_frequent()
        # print(f'pair_freq:{self.pair_freq}')
        # print(f"merged_pair:{merged_pair}")
        self.merge_pair(merged_pair, "2")
        
        
        
        
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
