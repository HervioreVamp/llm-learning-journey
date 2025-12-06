# coding: utf-8
import argparse
import time
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from my_model import PositionalEncoding, TransformerLM
# -------------------------- 1. 参数 --------------------------
parser = argparse.ArgumentParser(description='Modern Transformer LM on Wikitext-2')
parser.add_argument('--data', type=str, default='./data/wikitext-2')
parser.add_argument('--emsize', type=int, default=200, help='embedding dimension')
parser.add_argument('--nhead', type=int, default=2, help='num heads')
parser.add_argument('--nhid', type=int, default=2048, help='ffn hidden')
parser.add_argument('--nlayers', type=int, default=2, help='num layers')
parser.add_argument('--batch_size', type=int, default=160, help='batch size')
parser.add_argument('--seq_len', type=int, default=70, help='sequence length')
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--save', type=str, default='transformer_wikitext2.pt')
parser.add_argument('--seed', type=int, default=1111)
parser.add_argument('--log-interval', type=int, default=200, metavar='N',help='report interval')
parser.add_argument('--accel', action='store_true', default=True,help='Enables accelerated training')
args = parser.parse_args()

torch.manual_seed(args.seed)
if args.accel and torch.accelerator.is_available():
    device = torch.accelerator.current_accelerator()

else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# -------------------------- 2. 数据 --------------------------
from data import Corpus
import os
import requests
from pathlib import Path


def download_wikitext2(data_dir="./data/wikitext-2"):
    """自动下载数据集，如果本地没有的话"""
    os.makedirs(data_dir, exist_ok=True)
    base_url = "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/"
    files = ["train.txt", "valid.txt", "test.txt"]

    for fname in files:
        local_path = Path(data_dir) / fname
        if not local_path.exists():
            print(f"正在下载 {fname}...")
            url = base_url + fname
            response = requests.get(url)
            local_path.write_text(response.text, encoding="utf8")
    return str(data_dir)


# 使用方式
data_path = download_wikitext2()
corpus = Corpus(args.data)  # 语料库 读取txt那几个文件 训练/验证/测试,具体的类实现在data.py
ntokens = len(corpus.dictionary)


'''
把一段文本整除bsz(batch_size) 去除多余的 按列 一批批排布
坏处是每一列都是分开被喂给模型的 模型永远学不到被隔开的两个文本之前的关系
好处是可以并行处理 batch_size 倍的数据，极大提升训练速度。这就是典型的 “用空间（显存）换时间（速度）“
'''
def batchify(data, bsz):
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)


train_data = batchify(corpus.train, args.batch_size)
val_data = batchify(corpus.valid, 10)
test_data = batchify(corpus.test, 10)


def get_batch(source, i):
    seq_len = min(args.seq_len, source.size(0) - 1 - i)
    data = source[i:i + seq_len]
    target = source[i + 1:i + 1 + seq_len].reshape(-1)
    return data, target


# -------------------------- 4. 创建模型 --------------------------
model = TransformerLM(ntokens, args.emsize, args.nhead, args.nhid, args.nlayers, args.dropout).to(device)
criterion = nn.NLLLoss()  # 负对数似然 实际上是crossentrophy交叉熵的一个组成部分
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)


# -------------------------- 5. 训练循环 --------------------------
def train_epoch():
    model.train()
    total_loss = 0.
    start_time = time.time()
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.seq_len)):
        data, targets = get_batch(train_data, i)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        if batch % args.log_interval == 0 and batch > 0:
            ms_per_batch = (time.time() - start_time) * 1000 / args.log_interval
            cur_loss = total_loss / args.log_interval
            ppl = math.exp(cur_loss)
            print(
                f'| step | epoch {epoch:3d} | {batch:5d}/{len(train_data)//args.seq_len:5d} batches | | loss {cur_loss:5.2f} | ppl {ppl:6.1f} | ms/batch {ms_per_batch:5.2f}')
            total_loss = 0
            start_time = time.time()


def evaluate(data_source):
    model.eval()
    total_loss = 0.
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.seq_len):
            data, targets = get_batch(data_source, i)
            output = model(data)
            total_loss += len(data) * criterion(output.view(-1, ntokens), targets).item()
    return total_loss / (len(data_source) - 1)


# -------------------------- 6. 开始训练 --------------------------
best_val_loss = float('inf')
for epoch in range(1, args.epochs + 1):
    epoch_start_time = time.time()
    train_epoch()
    val_loss = evaluate(val_data)
    val_ppl = math.exp(val_loss)
    print('-' * 89)
    print(f'| end of epoch {epoch:3d} | time: {time.time() - epoch_start_time:5.1f}s | '
          f'valid loss {val_loss:5.2f} | valid ppl {val_ppl:6.1f}')
    print('-' * 89)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), args.save)

    scheduler.step()

# -------------------------- 7. 测试 --------------------------
model.load_state_dict(torch.load(args.save))
test_loss = evaluate(test_data)
print('=' * 89)
print(f'| End of training | test loss {test_loss:5.2f} | test ppl {math.exp(test_loss):6.1f}')
print('=' * 89)
