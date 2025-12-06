import torch
import torch.nn.functional as F
from data import Corpus
from my_model import TransformerLM

# 加载数据（创建词典）
print("加载词典...")
corpus = Corpus('./data/wikitext-2')
dictionary = corpus.dictionary
ntokens = len(dictionary)
print(f"词典大小: {ntokens}")

# 加载模型
print("加载模型...")
model = TransformerLM(
    ntokens=ntokens,
    emsize=200,
    nhead=2,
    nhid=2048,
    nlayers=2,
    dropout=0.1
)

# 加载权重
model.load_state_dict(torch.load('transformer_wikitext2.pt'))
model.eval()

# 放到GPU（如果有）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"使用设备: {device}")


def generate_simple(prompt='the', length=100, temperature=0.8):
    model.eval()

    # 将起始词转为token id
    if prompt in dictionary.word2idx:
        start_id = dictionary.word2idx[prompt]
    else:
        # 如果词不在词典，用'the'
        start_id = dictionary.word2idx['the'] if 'the' in dictionary.word2idx else 0

    # 初始输入：[1, 1] 形状
    input_seq = torch.tensor([[start_id]], dtype=torch.long).to(device)

    generated_ids = [start_id]

    with torch.no_grad():
        for i in range(length):
            # 前向传播
            output = model(input_seq)

            # 取最后一个预测
            logits = output[-1, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)

            # 采样下一个词
            next_token = torch.multinomial(probs, 1)

            # 添加到序列
            input_seq = torch.cat([input_seq, next_token.unsqueeze(0)], dim=0)
            generated_ids.append(next_token.item())

            # 打印进度
            if (i + 1) % 20 == 0:
                current_word = dictionary.idx2word[next_token.item()]
                print(f"生成进度: {i + 1}/{length}, 当前词: '{current_word}'")

    # 转换回文本
    words = [dictionary.idx2word[idx] for idx in generated_ids]
    text = ' '.join(words)

    # 简单清理
    import re
    text = re.sub(r'\s+([,.!?;:])', r'\1', text)

    return text


print("\n" + "=" * 50)
print("开始生成文本...")
print("=" * 50)

# 示例1：从"the"开始
print("\n示例1: 从 'the' 开始生成50个词")
text1 = generate_simple(prompt='the', length=50, temperature=0.7)
print(text1)

# 示例2：从"company"开始
print("\n\n示例2: 从 'company' 开始生成80个词")
text2 = generate_simple(prompt='company', length=80, temperature=0.8)
print(text2)

# 示例3：从"artificial"开始
print("\n\n示例3: 从 'artificial' 开始生成60个词")
text3 = generate_simple(prompt='artificial', length=60, temperature=0.9)
print(text3)
