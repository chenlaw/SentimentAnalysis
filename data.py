import pandas as pd
import jieba
import torch.nn.functional as F
import torch._tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) # 从1开始，0无意义
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class CustomChineseDataset(Dataset):
    def __init__(self, is_train=True, maxLen=50):
        self.data = None
        self.label = None
        self.dic = Dictionary()
        self.is_train = is_train
        self.maxLen = maxLen
        self.buildDataSet()

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        if idx >= len(self.data):
            raise IndexError(f"Index {idx} is out of range for dataset of size {len(self.data)}")
        return self.data[idx], self.label[idx]

    def getTokenLen(self):
        return len(self.dic.word2idx) + 1

    def buildDataSet(self):
        neg = pd.read_excel('neg.xls', header=None)
        pos = pd.read_excel('pos.xls', header=None)  # 读取训练语料完毕
        pos['mark'] = 1
        neg['mark'] = 0  # 给训练语料贴上标签
        pn = pd.concat([pos, neg], ignore_index=True)  # 合并语料
        neglen = len(neg)
        poslen = len(pos)  # 计算语料数目
        neg = torch.zeros(neglen)
        pos = torch.ones(poslen)
        label = torch.concat([neg, pos])
        pn[0] = pn[0].fillna('')
        cw = lambda x: list(jieba.cut(x))  # 定义分词函数
        pn['words'] = pn[0].apply(cw)

        comment = pd.read_excel('sum.xls')  # 读入评论内容
        # comment = pd.read_csv('a.csv', encoding='utf-8')
        comment = comment[comment['rateContent'].notnull()]  # 仅读取非空评论
        comment['words'] = comment['rateContent'].apply(cw)  # 评论分词

        d2v_train = pd.concat([pn['words'], comment['words']], ignore_index=True)

        for i in d2v_train:
            for j in i:
                self.dic.add_word(j)
        idss = []
        for words in pn['words']:
            ids = []
            for word in words:
                ids.append(self.dic.word2idx[word])
            idss.append(torch.tensor(ids).type(torch.int64))
        padded_sequences = []
        maxlen = 50
        for seq in idss:
            # 截断或填充序列
            if len(seq) > maxlen:
                seq = seq[:maxlen]  # 截断
            elif len(seq) < maxlen:
                padding = maxlen - len(seq)
                # 填充，确保填充到最大长度
                seq = F.pad(seq, (0, padding), value=0)  # 填充右侧  # 填充
            padded_sequences.append(seq)
        if self.is_train:
            self.data = padded_sequences[::2]
            self.label = label[::2]
        else:
            self.data = padded_sequences[1::2]
            self.label = label[1::2]
