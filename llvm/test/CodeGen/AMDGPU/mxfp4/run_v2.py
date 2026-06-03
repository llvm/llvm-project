#!/usr/bin/env python3
"""MXFP4 QAT v2: better quantization + larger model + longer training."""

import sys; sys.path.insert(0, '.')
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time, os
from dataclasses import dataclass
from train_mxfp4_qat import load_enwik8, get_batch

# ============================================================================
# Improved MXFP4 Quantized Linear with per-output-channel scaling
# ============================================================================
class MXFP4LinearV2(nn.Module):
    """MXFP4 linear with per-output-channel INT4 quantization and UE8M0 scale.
    Weights stored as FP32 master + quantized to INT4 for forward.
    Scale: per-row (output channel) computed dynamically, encoded as UE8M0.
    """
    def __init__(self, in_f, out_f):
        super().__init__()
        self.in_f, self.out_f = in_f, out_f
        self.master = nn.Parameter(torch.empty(out_f, in_f, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(out_f, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.master, a=np.sqrt(5))

    def get_quantized(self):
        """Quantize master weights to INT4 with per-row float scales.
        Returns (w_int4: float, scale: float[out_f])"""
        w = self.master
        # Per-row symmetric quantization: scale = max(|w_row|) / 3.5
        # INT4 signed range: [-8, 7], use [-3.5, 3.5] for better gradient behavior
        max_abs = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
        scale = (max_abs / 3.5)
        w_q = (w / scale).round().clamp(-8, 7)
        w_deq = w_q * scale
        return w_deq

    def forward(self, x):
        w_deq = self.get_quantized()
        return F.linear(x.to(torch.float32), w_deq, self.bias)

# ============================================================================
# Transformer
# ============================================================================
class Block(nn.Module):
    def __init__(self, d, h, mx=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, dropout=0.1, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        L = MXFP4LinearV2 if mx else nn.Linear
        self.mlp = nn.Sequential(L(d, 4*d), nn.GELU(), L(4*d, d))
        self.drop = nn.Dropout(0.1)
    def forward(self, x):
        x = x + self.drop(self.attn(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=False)[0])
        x = x + self.drop(self.mlp(self.ln2(x)))
        return x

class GPTv2(nn.Module):
    def __init__(self, d=512, h=8, L=8, block=256, vocab=256, mx=False):
        super().__init__()
        self.block = block
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(block, d)
        self.blocks = nn.ModuleList([Block(d, h, mx) for _ in range(L)])
        self.ln = nn.LayerNorm(d)
        LCls = MXFP4LinearV2 if mx else nn.Linear
        self.head = LCls(d, vocab)
        self.head.bias = nn.Parameter(torch.zeros(vocab))
    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        for blk in self.blocks: x = blk(x)
        x = self.ln(x)
        logits = self.head(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) if targets is not None else None
        return logits, loss

# ============================================================================
# Training
# ============================================================================
def train_one(model, cfg, train_data, val_data, steps, label, device):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    scaler = torch.amp.GradScaler('cuda')
    losses = []; best_val = float('inf'); t0 = time.time()

    for step in range(steps):
        x, y = get_batch(train_data, cfg['block'], cfg['bs'], device)
        with torch.amp.autocast('cuda'):
            _, loss = model(x, y)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        losses.append(loss.item())

        if (step+1) % 100 == 0:
            avg = sum(losses[-100:])/100
            tok_s = (step+1) * cfg['bs'] * cfg['block'] / (time.time()-t0)
            p = sum(p.numel() for p in model.parameters())
            print(f"  [{label}] step {step+1:5d} loss={avg:.4f} tok/s={tok_s:.0f} peak_mem={torch.cuda.max_memory_allocated()/1e9:.2f}GB")
            torch.cuda.reset_peak_memory_stats()

        if (step+1) % 500 == 0:
            model.eval(); vl=0
            with torch.no_grad():
                for _ in range(10):
                    x,y=get_batch(val_data,cfg['block'],cfg['bs'],device); _,l=model(x,y); vl+=l.item()
            vl/=10; model.train()
            print(f"  [{label}] --- VAL LOSS: {vl:.4f} ---")
            if vl<best_val: best_val=vl

    return losses, best_val

if __name__ == '__main__':
    device = torch.device('cuda')
    data = load_enwik8('/data/模型训练精度验证/enwik8')
    tr, va = data[:int(0.9*len(data))], data[int(0.9*len(data)):]
    cfg = {'block': 256, 'bs': 32}

    # --- Model configs ---
    # Baseline: FP16 @ ~25M params
    m_fp16 = GPTv2(d=512, h=8, L=8, mx=False).to(device)
    print(f"\nFP16 model: {sum(p.numel() for p in m_fp16.parameters())/1e6:.1f}M params")

    l_fp16, best_fp16 = train_one(m_fp16, cfg, tr, va, 2000, "FP16", device)

    torch.cuda.empty_cache()

    # MXFP4: same architecture with quantized linear layers
    m_mx = GPTv2(d=512, h=8, L=8, mx=True).to(device)
    print(f"\nMXFP4 model: {sum(p.numel() for p in m_mx.parameters())/1e6:.1f}M params")

    l_mx, best_mx = train_one(m_mx, cfg, tr, va, 2000, "MXFP4", device)

    print(f"\n{'='*60}")
    print(f"FP16:  best_val={best_fp16:.4f}")
    print(f"MXFP4: best_val={best_mx:.4f}")
    print(f"Ratio:  {best_mx/best_fp16:.3f} ({(best_mx/best_fp16-1)*100:+.1f}%)")
    print(f"{'='*60}")
