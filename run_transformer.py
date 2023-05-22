#!/usr/bin/env python3

import torch
import yaml
from src.data import datamodule
import torch
import torch.nn as nn
from tqdm import tqdm

from src.seq2seq_transformer import *

if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = 'cpu'

print("DEVICE: ", DEVICE)

data_config = yaml.load(open("configs/data_config.yaml", 'r'),   Loader=yaml.Loader)
dm = datamodule.DataManager(data_config, DEVICE)
train_dataloader, dev_dataloader = dm.prepare_data()

UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3


torch.manual_seed(0)

SRC_VOCAB_SIZE = len(dm.source_tokenizer.word2index)
TGT_VOCAB_SIZE = len(dm.target_tokenizer.word2index)
EMB_SIZE = 200
NHEAD = 8
FFN_HID_DIM = 200
BATCH_SIZE = 128
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3

transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)

for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

transformer = transformer.to(DEVICE)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0003, betas=(0.9, 0.98), eps=1e-9)

scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=500,
            gamma=0.9,
)

def train_epoch(model, optimizer):
    model.train()
    losses = 0
    #train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    #train_dataloader = DataLoader(train_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    i = 0
    for src, tgt in tqdm(train_dataloader):
        i += 1
        src = src.to(DEVICE).permute(1, 0)
        tgt = tgt.to(DEVICE).permute(1, 0)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, DEVICE, PAD_IDX)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]

        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))

        loss.backward()

        optimizer.step()

        losses += loss.item()

    return losses / len(list(train_dataloader))


def evaluate(model):
    model.eval()
    losses = 0

    
    val_dataloader = dev_dataloader#DataLoader(val_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    for src, tgt in val_dataloader:
        src = src.to(DEVICE).permute(1, 0)
        tgt = tgt.to(DEVICE).permute(1, 0)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

    return losses / len(list(val_dataloader))


from timeit import default_timer as timer

NUM_EPOCHS = 40

for epoch in tqdm(range(1, NUM_EPOCHS+1)):
    start_time = timer()
    train_loss = train_epoch(transformer, optimizer)
    end_time = timer()
    val_loss = evaluate(transformer)
    with open('training_logs.txt', 'a') as f:
      str_data = f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"
      f.write(f'{str(str_data)}\n')
    torch.save(transformer.state_dict(), 'transformer')
    print(str_data)
