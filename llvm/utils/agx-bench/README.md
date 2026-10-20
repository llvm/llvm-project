# agx-bench

`agx-bench.py` measures the code the LLVM AGX backend generates for a fixed suite of kernels on G13 and G14:
- code bytes;
- GPRs and uniform halves (occupancy);
- `wait` count;
- `agx_cost` cycles of one SIMD-group, from the AGX Performance Simulator.

It compares the results with `baseline.json` and exits with 1 on a regression. Run it after every backend change:

```
python3 llvm/utils/agx-bench/agx-bench.py --llc build/bin/llc \
    --cost AGX-Performance-Simulator/rust/target/release/agx_cost
python3 ... --update                       # accept the new numbers
python3 ... --zoc ZLUDA/target/debug/zoc --ptx ZLUDA/ptx/src/test/*.ptx   # also PTX through ZLUDA
```

The kernels in `kernels/` cover the common shapes of CUDA code:

| Kernel | Shape |
| --- | --- |
| `saxpy` | streaming |
| `reduce` | threadgroup tree reduction + float atomic |
| `transpose` | tiled threadgroup memory |
| `stencil` | neighbour loads |
| `histogram` | threadgroup and device atomics |
| `scan` | SIMD prefix |
| `softmax_row` | SIMD reductions + `exp` |

`agx_cost` estimates come from the simulator's model calibrated on an M2 Max. Rebuild it from the simulator's
current `main` before comparing over time.
