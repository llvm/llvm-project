#!/usr/bin/env python3
"""Quick benchmark: FP16 baseline vs MXFP4 QAT on enwik8."""

import sys; sys.path.insert(0, '.')
from train_mxfp4_qat import *

def bench(model_cfg, train_cfg, n_steps=200, label="FP16"):
    device = torch.device('cuda')
    data = load_enwik8('/data/模型训练精度验证/enwik8')
    train_data = data[:int(0.9*len(data))]
    model = GPT(model_cfg).to(device)
    params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.learning_rate,
                                   weight_decay=train_cfg.weight_decay)
    scaler = torch.amp.GradScaler('cuda') if train_cfg.use_amp else None
    model.train()

    losses = []
    t0 = time.time()
    for step in range(n_steps):
        x, y = get_batch(train_data, model_cfg.block_size, train_cfg.batch_size, device)
        with torch.amp.autocast('cuda', enabled=train_cfg.use_amp):
            logits, loss = model(x, y)
        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if scaler:
            scaler.step(optimizer); scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(loss.item())
        if (step+1) % 50 == 0:
            avg = sum(losses[-50:]) / 50
            elapsed = time.time() - t0
            print(f"  [{label}] step {step+1:4d} avg_loss={avg:.4f} tok/s={n_steps*train_cfg.batch_size*model_cfg.block_size/elapsed:.0f}")

    elapsed = time.time() - t0
    avg_loss = sum(losses[-50:]) / 50
    print(f"[{label}] {params/1e6:.1f}M params | {n_steps} steps | "
          f"final_loss={avg_loss:.4f} | tok/s={n_steps*train_cfg.batch_size*model_cfg.block_size/elapsed:.0f} | "
          f"peak_mem={torch.cuda.max_memory_allocated()/1e9:.2f}GB")
    torch.cuda.reset_peak_memory_stats()
    return losses, avg_loss

if __name__ == '__main__':
    cfg = ModelConfig(d_model=384, n_heads=6, n_layers=6, block_size=256,
                      use_mxfp4=False)  # baseline FP16
    t_cfg = TrainConfig(batch_size=32, gradient_accumulation=1,
                        learning_rate=3e-4, max_steps=200,
                        use_amp=True)
    print("=== FP16 Baseline ===")
    l_fp16, avg_fp16 = bench(cfg, t_cfg, n_steps=200, label="FP16")

    torch.cuda.empty_cache()
    cfg.use_mxfp4 = True
    print("\n=== MXFP4 QAT ===")
    l_mx, avg_mx = bench(cfg, t_cfg, n_steps=200, label="MXFP4")

    print(f"\n=== Summary ===")
    print(f"FP16:  final_loss={avg_fp16:.4f}")
    print(f"MXFP4: final_loss={avg_mx:.4f}")
    loss_ratio = avg_mx / avg_fp16 if avg_fp16 > 0 else 0
    print(f"Loss ratio: {loss_ratio:.3f} ({'+' if loss_ratio>1 else ''}{(loss_ratio-1)*100:+.1f}%)")
