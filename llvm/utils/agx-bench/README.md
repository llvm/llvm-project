# agx-bench

`agx-bench.py` measures the code the LLVM AGX backend generates for a fixed suite of kernels on G13 and G14:
- code bytes;
- GPRs and uniform halves (occupancy);
- `wait` count;
- `agx_cost` cycles of one SIMD-group, from the AGX Performance Simulator;
- `tput`: agx_cost cycles per SIMD-group with as many groups sharing a core as the kernel's registers, workgroup
  size and threadgroup memory let reside (the simulator's measured occupancy model: min(96, 2 * floor(3328 /
  (2 * register halves))) groups, whole workgroups, 64 KiB of threadgroup memory). One group's cycles reward
  latency; `tput` also charges registers that cost occupancy (e.g. clustering threadgroup loads takes the tiled
  SGEMM from 70 to 77 GPRs: -16% cycles, +57% tput) and credits latency other groups hide.

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

`kernels/cuda_*.ll` are real-world CUDA kernels written for this suite (`cuda/*.cu`), compiled to PTX by clang's NVPTX
target (no CUDA toolkit: `cuda/bench.h` declares what they use) and to AGX IR by ZLUDA's zoc. `cuda/gen.py --zoc
ZLUDA/target/debug/zoc` regenerates them. Every one is costed with launch inputs (sizes, grid, memory contents where
control flow depends on them):

| Area | Kernels |
| --- | --- |
| LLM inference | `rmsnorm`, `softmax_online` (online max / sum), `rope`, `q8_matvec` / `q4_matvec` (llama.cpp q8_0 / q4_0 blocks: unaligned f16 scales, bytes, nibbles), `attn_decode` (f16 KV cache, online softmax, threadgroup merge), `swiglu`, `embedding`, `argmax` |
| Transformers, f16 | `layernorm_f16`, `gelu_f16` (half2 storage, tanh) |
| Frameworks | `permute4d` (PyTorch strided copy: 64-bit offsets by division / modulo by runtime sizes), `index_add` (float atomics) |
| Dense / CNN | `sgemm_reg4x4` (threadgroup tiles, 4 x 4 register blocks), `conv3x3` (direct NCHW) |
| Sparse | `spmv_csr` (thread per row), `spmv_csr_vector` (SIMD-group per row); row offsets mapped from `kernels/data/` |
| Stencil / image | `jacobi2d` (threadgroup tile with halo), `blur_rgba8` (bytes, clamped edges) |
| Parallel primitives | `reduce_shfl` (grid-stride, shuffles, atomics), `scan_blelloch`, `bitonic_smem`, `radix_hist` |
| Simulation / other | `nbody` (rsqrt, threadgroup tiles), `mandelbrot` (divergent loop trip counts), `philox` (curand's RNG: multiply-high), `ddot` (software f64) |

In an `; agx-cost:` line, `--map VA=@FILE` names a file relative to the kernel file. A grid-stride loop reads the
grid size from the `threadgroups_per_grid` uniform (named in llc's kernel header comment), which needs a
`--uniform` of its own besides `--grid`. agx_cost does not emulate `convert`, so a loop whose trip count depends on
an int-to-float conversion runs on wrong values (`mandelbrot` converts by the 2^23 exponent trick instead).

`agx_cost` estimates come from the simulator's model calibrated on an M2 Max. Rebuild it from the simulator's
current `main` before comparing over time. The baseline was recorded with simulator `main` f50bd88; that commit's
33-cycle hold on taken forward jumps adds 32 cycles to `saxpy`, `stencil` and `histogram`.

A kernel file may name launch inputs in a `; agx-cost: ARGS` line (for example `--uniform 5=1024 --wg 256`):
agx_cost then traces that launch, so bounds checks and loops on kernel arguments take their real path
(`saxpy`, `stencil`, `histogram`: `n = 1024`; `matrixmul`: `wA = wB = 64`, four K tiles). This needs an agx_cost
with launch inputs (AGX-Performance-Simulator#7). An older one rejects them and the kernel is costed with every
uniform zero, which skips a guarded body; each result records `launch_inputs`, and cycles are only compared
between runs of the same kind. The baseline was recorded with launch inputs.
