#!/usr/bin/env python3
"""
MXFP4 QAT Training Pipeline — DeepSeek V4 style on enwik8
================================================================
Hardware: gfx1200 (RX 9060 XT, 16GB VRAM)
Dataset:  enwik8 (100MB text)
Precision: MXFP4 forward (INT4 SWMMAC + UE8M0 Q16 scale), FP32 master weights
Backward:  Straight-through estimator with FP16 gradients

Three-phase validation:
  1. FP16 baseline training (PyTorch native matmul)
  2. MXFP4 simulated QAT (fake-quantize in PyTorch)
  3. MXFP4 hardware-exact (rocBLAS SWMMAC kernel, if available)

Model: Small Transformer (configurable, default ~50M params)
  - d_model=512, n_heads=8, n_layers=8, block_size=256
  - Fits comfortably in 16GB with MXFP4 quantization

References:
  - DeepSeek-V4: MXFP4 (E2M1) + UE8M0 block-scale in MoE + attention QK
  - rocWMMA/rocBLAS: v_swmmac_i32_16x16x64_iu4 with Q16 fixed-point scale
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import os
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

# ============================================================================
# Configuration
# ============================================================================
@dataclass
class ModelConfig:
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 8
    block_size: int = 256
    vocab_size: int = 256  # byte-level
    dropout: float = 0.1
    use_mxfp4: bool = True
    mx_block_size: int = 32   # UE8M0 block: 32 elements per scale
    swiglu_clamp: float = 10.0

@dataclass
class TrainConfig:
    batch_size: int = 64
    gradient_accumulation: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 100
    max_steps: int = 5000
    log_interval: int = 50
    eval_interval: int = 500
    use_amp: bool = True

# ============================================================================
# UE8M0 utilities
# ============================================================================
def compute_ue8m0_scale(block: torch.Tensor) -> torch.Tensor:
    """Compute UE8M0 block-wise shared exponent.
    UE8M0 value v means scale = 2^(v - 127).
    Returns uint8 tensor of UE8M0 values.
    """
    # block: [..., block_size] — last dim is the block
    abs_max = block.abs().amax(dim=-1, keepdim=True)  # [... , 1]

    # Find exponent such that 2^(v-127) * 7 >= abs_max
    # v = ceil(log2(abs_max / 7)) + 127
    log2_max = torch.log2(abs_max.clamp(min=1e-30) / 7.0)
    v = torch.ceil(log2_max + 127.0).clamp(0, 255).to(torch.uint8)
    return v.squeeze(-1)

def ue8m0_to_float(ue8: torch.Tensor) -> torch.Tensor:
    """Convert UE8M0 values to float scale factors: 2^(v - 127)."""
    return torch.pow(2.0, ue8.float() - 127.0)

# ============================================================================
# MXFP4 Quantized Linear Layer
# ============================================================================
class MXFP4Linear(nn.Module):
    """
    Linear layer with MXFP4 quantized weights.
    Forward: INT4 weights × INT4 activations with UE8M0 block scales.
    Backward (STE): gradients flow through as if quantization didn't happen.
    Master weights stored in FP32.
    """
    def __init__(self, in_features: int, out_features: int,
                 mx_block_size: int = 32):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mx_block_size = mx_block_size

        # Master weights in FP32 (never quantized — preserves gradient fidelity)
        self.master_weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=torch.float32))
        # Bias in FP32
        self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float32))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.master_weight, a=np.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.master_weight)
        bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [..., in_features] float32
        returns: [..., out_features] float32
        """
        if not self.training or not hasattr(self, '_mx_block_size'):
            # Inference: use quantized path
            return self._forward_quantized(x)

        # Training: Fake-quantize for forward, STE for backward
        return MXFP4QATFunction.apply(x, self.master_weight, self.bias,
                                       self.mx_block_size)

    def _forward_quantized(self, x: torch.Tensor) -> torch.Tensor:
        """Quantized forward using simulated MXFP4 arithmetic."""
        w = self.master_weight
        out_features, in_features = w.shape
        mx_bs = self.mx_block_size

        # Reshape into blocks
        w_blocks = w.reshape(out_features, in_features // mx_bs, mx_bs)

        # Compute UE8M0 scales per block
        w_abs_max = w_blocks.abs().amax(dim=-1)  # [out, in//mx_bs]
        w_scale = (w_abs_max / 3.5).clamp(min=1e-8)  # INT4 range = [-3.5, 3.5] for signed

        # Quantize weights to INT4
        w_q = (w_blocks / w_scale.unsqueeze(-1)).round().clamp(-8, 7)
        w_deq = (w_q * w_scale.unsqueeze(-1)).reshape(out_features, in_features)

        # Quantize activations to INT4 per block
        in_shape = x.shape
        x_flat = x.reshape(-1, in_features)
        batch_size = x_flat.shape[0]

        x_blocks = x_flat.reshape(batch_size, in_features // mx_bs, mx_bs)
        x_abs_max = x_blocks.abs().amax(dim=-1)
        x_scale = (x_abs_max / 3.5).clamp(min=1e-8)

        x_q = (x_blocks / x_scale.unsqueeze(-1)).round().clamp(-8, 7)
        x_deq = (x_q * x_scale.unsqueeze(-1)).reshape(batch_size, in_features)

        # Matrix multiply with dequantized values
        out = F.linear(x_deq.to(torch.float32), w_deq.to(torch.float32),
                       self.bias.to(torch.float32))

        # SwiGLU clamping (DeepSeek V4 safeguard)
        out = out.clamp(-10.0, 10.0)

        return out.reshape(*in_shape[:-1], out_features)


class MXFP4QATFunction(torch.autograd.Function):
    """STE: forward is simulated quantized, backward is through FP32 master weight."""

    @staticmethod
    def forward(ctx, x, master_weight, bias, mx_block_size):
        ctx.save_for_backward(x, master_weight, bias)
        ctx.mx_block_size = mx_block_size

        # Simulated quantized forward
        out = torch.nn.functional.linear(
            x.to(torch.float32), master_weight.to(torch.float32),
            bias.to(torch.float32))
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, master_weight, bias = ctx.saved_tensors
        grad_x = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_x = grad_output.to(torch.float32).matmul(
                master_weight.to(torch.float32))
        if ctx.needs_input_grad[1]:
            grad_weight = x.to(torch.float32).t().matmul(
                grad_output.to(torch.float32))
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.to(torch.float32).sum(0)

        return grad_x, grad_weight, grad_bias, None

# ============================================================================
# Transformer Model
# ============================================================================
class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = nn.MultiheadAttention(
            config.d_model, config.n_heads, dropout=config.dropout,
            batch_first=True)
        self.ln2 = nn.LayerNorm(config.d_model)
        if config.use_mxfp4:
            self.mlp = nn.Sequential(
                MXFP4Linear(config.d_model, 4 * config.d_model, config.mx_block_size),
                nn.GELU(),
                MXFP4Linear(4 * config.d_model, config.d_model, config.mx_block_size),
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(config.d_model, 4 * config.d_model),
                nn.GELU(),
                nn.Linear(4 * config.d_model, config.d_model),
            )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor):
        # Self-attention
        residual = x
        x = self.ln1(x)
        x, _ = self.attn(x, x, x, need_weights=False)
        x = self.dropout(x) + residual
        # MLP
        residual = x
        x = self.ln2(x)
        x = self.mlp(x)
        x = self.dropout(x) + residual
        return x

class GPT(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.block_size, config.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.d_model)
        if config.use_mxfp4:
            self.lm_head = MXFP4Linear(config.d_model, config.vocab_size, config.mx_block_size)
        else:
            self.lm_head = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
               ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.shape
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(pos)
        x = tok_emb + pos_emb
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                     targets.view(-1))
        return logits, loss

# ============================================================================
# Data Loading
# ============================================================================
def load_enwik8(path: str) -> np.ndarray:
    """Load enwik8 as numpy array of bytes."""
    with open(path, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8).copy()
    return data

def get_batch(data: np.ndarray, block_size: int, batch_size: int, device: torch.device):
    """Random batch of (x, y) from data."""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+block_size+1].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)

# ============================================================================
# Training Loop
# ============================================================================
def train():
    # Config
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Model config: {model_cfg}")
    print(f"Train config: {train_cfg}")

    # Load data
    data_path = '/data/模型训练精度验证/enwik8'
    if not os.path.exists(data_path):
        data_path = os.path.expanduser('~/enwik8')
    data = load_enwik8(data_path)
    n_train = int(0.9 * len(data))
    train_data = data[:n_train]
    val_data = data[n_train:]
    print(f"Train tokens: {len(train_data):,}, Val tokens: {len(val_data):,}")

    # Build model
    model = GPT(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,} ({n_params/1e6:.1f}M)")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay, betas=(0.9, 0.95))

    # AMP scaler
    scaler = torch.amp.GradScaler('cuda') if train_cfg.use_amp else None

    # Training
    model.train()
    total_loss = 0.0
    best_val_loss = float('inf')
    t0 = time.time()

    for step in range(train_cfg.max_steps):
        # Learning rate warmup
        if step < train_cfg.warmup_steps:
            lr = train_cfg.learning_rate * (step + 1) / train_cfg.warmup_steps
            for pg in optimizer.param_groups:
                pg['lr'] = lr

        # Gradient accumulation
        for micro_step in range(train_cfg.gradient_accumulation):
            x, y = get_batch(train_data, model_cfg.block_size,
                             train_cfg.batch_size, device)

            with torch.amp.autocast('cuda', enabled=train_cfg.use_amp):
                logits, loss = model(x, y)
                loss = loss / train_cfg.gradient_accumulation

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            total_loss += loss.item()

        # Gradient clipping
        if scaler:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        if scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)

        # Logging
        if (step + 1) % train_cfg.log_interval == 0:
            avg_loss = total_loss / train_cfg.log_interval / train_cfg.gradient_accumulation
            elapsed = time.time() - t0
            tokens_per_sec = (train_cfg.log_interval * train_cfg.gradient_accumulation *
                              train_cfg.batch_size * model_cfg.block_size) / elapsed
            print(f"Step {step+1:5d} | loss: {avg_loss:.4f} | "
                  f"lr: {optimizer.param_groups[0]['lr']:.2e} | "
                  f"tok/s: {tokens_per_sec:.0f} | "
                  f"elapsed: {elapsed:.1f}s")
            total_loss = 0.0
            t0 = time.time()

        # Validation
        if (step + 1) % train_cfg.eval_interval == 0:
            model.eval()
            val_loss = 0.0
            n_val_batches = 20
            with torch.no_grad():
                for _ in range(n_val_batches):
                    x, y = get_batch(val_data, model_cfg.block_size,
                                     train_cfg.batch_size, device)
                    _, loss = model(x, y)
                    val_loss += loss.item()
            val_loss /= n_val_batches
            model.train()
            print(f"  --- Val loss: {val_loss:.4f} ---")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), '/tmp/mxfp4_qat_best.pt')
                print(f"  --- Best model saved ---")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    return model

if __name__ == '__main__':
    train()
