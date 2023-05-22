import torch
from src.model.seq2seq_transformer import generate_square_subsequent_mask

def greedy_decode(model, src, src_mask, max_len, start_symbol, DEVICE, EOS_IDX):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len-1):
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0), DEVICE)
                    .type(torch.bool)).to(DEVICE)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break
    return ys


# actual function to translate input sentence into target language
def translate(model: torch.nn.Module, src_sentence: str, dm, BOS_IDX, DEVICE, EOS_IDX):
    model.eval()
    src = torch.Tensor(dm.source_tokenizer(src_sentence)).reshape(-1, 1) #text_transform[SRC_LANGUAGE](src_sentence).view(-1, 1)
    #print(src)
    num_tokens = len(src)
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model,  src, src_mask, num_tokens + 5, BOS_IDX, DEVICE, EOS_IDX).flatten().type(torch.int64).tolist()
    #print(tgt_tokens)
    return dm.target_tokenizer.decode(tgt_tokens)#" ".join(vocab_transform[TGT_LANGUAGE].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")