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
| `matrixmul` | CUDA samples matrixMul through ZLUDA: threadgroup tiles, barriers, unrolled FMAs |

`agx_cost` estimates come from the simulator's model calibrated on an M2 Max. Rebuild it from the simulator's
current `main` before comparing over time. The baseline was recorded with simulator `main` f50bd88; that commit's
33-cycle hold on taken forward jumps adds 32 cycles to `saxpy`, `stencil` and `histogram`.

A kernel file may name launch inputs in a `; agx-cost: ARGS` line (for example `--uniform 5=1024 --wg 256`):
agx_cost then traces that launch, so bounds checks and loops on kernel arguments take their real path
(`saxpy`, `stencil`, `histogram`: `n = 1024`; `matrixmul`: `wA = wB = 64`, four K tiles). This needs an agx_cost
with launch inputs (AGX-Performance-Simulator#7). An older one rejects them and the kernel is costed with every
uniform zero, which skips a guarded body; each result records `launch_inputs`, and cycles are only compared
between runs of the same kind. The baseline was recorded with launch inputs.
