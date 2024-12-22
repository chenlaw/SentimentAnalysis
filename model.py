import torch.nn as nn


class RNNModel(nn.Module):
    def __init__(self, ntoken, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(ntoken, 256)
        self.lstm = nn.LSTM(256, 128, batch_first=True)  # 使用 batch_first=True，方便处理输入
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.LazyLinear(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)  # 通过嵌入层
        output, _ = self.lstm(x)  # 只取 LSTM 的输出部分
        output = output[:, -1, :]
        output = self.dropout(output)  # 取最后一个时间步的输出，并应用 Dropout
        output = self.linear(output)  # 通过线性层
        output = self.sigmoid(output)  # 应用 Sigmoid 激活函数
        return output
