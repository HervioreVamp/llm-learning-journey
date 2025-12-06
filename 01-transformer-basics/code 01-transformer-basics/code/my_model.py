import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerLM(nn.Module):
    def __init__(self, ntokens, emsize, nhead, nhid, nlayers, dropout=0.1):
        super().__init__()
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(emsize, dropout)
        encoder_layers = nn.TransformerEncoderLayer(emsize, nhead, nhid, dropout, batch_first=False)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntokens, emsize)
        self.decoder = nn.Linear(emsize, ntokens)
        self.emsize = emsize
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def _generate_square_subsequent_mask(self, sz):
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)
    def forward(self, src):
        """
            如果之前没缓存过 mask，
            或者当前要处理的序列长度变了（比如生成时从 10 个词变成 11 个词），
            那就重新生成一个 刚好合适大小 的上三角因果掩码
        """
        if self.src_mask is None or self.src_mask.size(0) != src.size(0):
            self.src_mask = self._generate_square_subsequent_mask(src.size(0)).to(src.device)

        src = self.encoder(src) * math.sqrt(self.emsize)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return F.log_softmax(output, dim=-1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

'''
TransformerEncoderLayer:
self_attn → dropout1 → norm1 → linear1 → dropout → linear2 → dropout2 → norm2
当然 残差会被直接加上去 [残差连接]
超级黑箱这一块()

这里用encoder+掩码实现了decoder 上面的decoder实际上是输出投影层 output projection
'''
