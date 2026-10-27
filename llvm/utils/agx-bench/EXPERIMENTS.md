# AGX backend experiments: results to verify on hardware

Every result below comes from the AGX Performance Simulator's cost model (`agx_cost`), not from a GPU. The
simulator is calibrated on an M2 Max, but several decisions rest on single measured parameters (named per
entry). This file lists each experiment so that it can be re-run on a real M2 (Asahi Linux, ZLUDA `zluda_agx`,
or any harness that loads the objects `llc` emits and times dispatches).

## How to verify an entry

1. Build `llc` from this branch. Each entry gives an **A/B switch**: an `llc` option (both settings in one
   binary) or, where there is none, the commit to revert.
2. Compile the named kernels both ways: `llc -mtriple=agx -mcpu=agx-g14g -filetype=obj K.ll` (`agx-g14x` for an
   M2 Max/Pro, `agx-g13g` for an M1). `kernels/cuda_*.ll` come from `cuda/*.cu`, whose first line gives the
   launch the model assumed (sizes, grid, workgroup).
3. Time both on the GPU with that launch, scaled up until the dispatch takes at least ~1 ms, median of many runs.
   Also time a **full-occupancy** launch (enough workgroups for every core), because the model reports two
   numbers:
   - **cycles**: one SIMD-group alone (latency; what a small grid sees);
   - **tput**: cycles per SIMD-group with as many groups per core as registers / threadgroup memory allow
     (throughput; what a large grid sees).
4. Check results are bit-identical between A and B (every change here should be semantics-preserving).

Kernel sets used: **rw** = `kernels/*.ll` (8 shapes + 27 real-world CUDA kernels), **co** = ZLUDA's PTX tests
(195 kernels compiled by zoc), **lb** = ZLUDA `agx_libs` MLIR libraries. Percentages are the change in summed
cycles over a set (negative = faster) unless stated.

## Adopted (in the default pipeline)

| # | Change (commit) | A/B switch | Model result | Rests on | Verify with |
|---|---|---|---|---|---|
| A1 | Store clustering in the machine scheduler, AGXMemCluster (73a21ff5a) | `-agx-mem-cluster=0` vs default; `-agx-mem-cluster-kinds=` mask (1 dev load, 2 dev store, 4 tg load, 8 tg store; default 11) | co -0.76% (28 better, 0 worse), lb -9.5% (tile GEMM epilogues), G14G sgemm tile at K=256 -0.1..-1.5% | `mem_quiet_device_store_ref` 33.6: a store right after an ALU op waits ~34 cycles, back-to-back stores ~3 | `cuda_add_extended`-like store tails, `agx_sgemm_tile64*` (agx_libs), `vector_operand` |
| A2 | Device-load clustering (in the mask since "Cluster device loads too") | `-agx-mem-cluster-kinds=10` (no loads) vs 11 | cycles: attn_decode -34%, blur_rgba8 -28%, q4_matvec -28%, q8_matvec -20%, co/lb -3.8%; **tput ~0%** (latency only); index_add on G13 +17% | `mem_quiet_load_ref` 20: a load right after ALU work completes ~20 cycles later | `cuda_attn_decode`, `cuda_q4_matvec`, `cuda_q8_matvec`, `cuda_blur_rgba8` with a **small** grid (e.g. 32 workgroups) and a full one |
| A3 | Post-RA MachineScheduler, top-down, with AGXMemCluster (transitive cluster edges) | `-enable-post-misched=false` vs default | rw -4.7%, co -4.2%, lb -4.8% cycles; tput +4.3% worse without it; llc +2% time | the scheduling model's latencies (AGXSchedule.td, from bench.json.gz) | whole suite; worst cases in the model: `stencil`, `reduce`, G14 hgemm `_u` tiles (+30..60%) |
| A4 | Scheduling model: matrix FMA latency 17 / issue 16, 3-source FP and cmpsel issue 2 (af6ddf4d9) | revert af6ddf4d9 | -0.10% over 428 kernels | bench.json.gz rows (G14C); G13 / Apple9 reuse them | tile GEMMs, fp-heavy kernels |
| A5 | iadd / isub / imadd with the second source (addend) shifted by 0-4; x*3/5/9/17 as iadd with shift | `-agx-decompose-mul=false` (multiply only); revert "Fold 32-bit shifts by 0-4" for the rest | 4 kernels -0.4..-1.8%, none worse | `iadd ... lsl s` = a + (b << s) (applegpu semantics, used by Mesa) | `cuda_rope`, index arithmetic; check results bit-exact |
| A6 | One bitop for a&~b, a\|~b, xnor, nand, nor; bfeil for single-use low masks wider than 8 bits | revert "One bitop for and-not..." | co -0.32% (16 better, 10 worse by <= 1.1%: bfeil latency 3 vs and 2) | bitop truth-table encoding; bfeil mask semantics | `cvt_*`, `mma_*bf16*`, `copysign` PTX tests |
| A7 | Software f64: shorter add / mul / fma / div / sqrt (9acacf2bd, 7fafed9d5, a64a6f0af) | revert | 100 operations x 2048 inputs bit-exact (`agx_f64_check.py`, both NaN modes) | integer ALU costs | `cuda_ddot`, PTX f64 tests |
| A8 | fatbin builds: one opt run shared by G13/G14, same-code chips compiled once (2e6011be4, 1a0d3648e) | revert | build 37 -> 15.9 s, objects 10x smaller | `AGXProcessors.td` same-code groups | build time only |
| A9 | ZLUDA agx_libs: in the `_ta` tile GEMMs, load each B fragment just before its four matrix products (ZLUDA "tile GEMM with A transposed loads each B fragment...") | revert that ZLUDA commit (`gen_tile_gemm.py`) | K = 256: sgemm _ta -19% G13G / -8% G14G, hgemm _ta -32..-35% G13G / -1..-4% G14G (one sgemm _u_ta +0.8%); other variants moved +-10% both ways and keep their order | register pressure / spills (G14 `_tb` still spills 6 values at 128 GPRs) | cuBLAS sgemm / hgemm with transa = T |
| A10 | ZLUDA agx_libs: in the 16-bit `_tb` tile GEMMs (C = A B^T: linear layers), load B first and each A fragment just before its products | revert that ZLUDA commit | K = 256: hgemm _tb -9..-13% on G14G (all four variants); G13G -12%..+21% (cf32_u_tb +21%), net ~+2% | register pressure | cuBLAS hgemm / gemmEx with transb = T on M2; check G13 cf32_u_tb |

## Measured and rejected (model)

| # | Change | Model result | Why rejected / note for hardware |
|---|---|---|---|
| R1 | Threadgroup-load clustering (`-agx-mem-cluster-kinds=15`) | cycles: sgemm_reg4x4 -16%, nbody -9%; tput +12.8% (sgemm +57%: 70 -> 77 GPRs drops resident groups); ddot +33% | occupancy. **Worth checking on hardware**: the occupancy formula decides it |
| R2 | Generic post-RA scheduling without clustering (`-misched-postra -enable-post-misched`, legacy `-post-RA-scheduler`) | +1.0..+3.0% | spreads stores apart (see A1) |
| R3 | Scheduler knobs: `-enable-misched=false` +43%, pre-RA top-down +17.7%, bottom-up +0.05%, `-misched-regpressure=false` +16.9%; `-misched-cluster`, `-misched-cyclicpath`, `-enable-aa-sched-mi` no change | | default bidirectional pre-RA stays |
| R4 | bfeil for every low mask (also masks with other uses), bfeil for fabs | co -0.37% but 30 kernels up to +2.5% | a shared mov_imm + and beats several bfeil |
| R5 | Load clustering with weak (cluster-only) edges | ~0% | edges must constrain to matter |
| R6 | Runs of 4 or 6 loads | +0.2..+0.7% | attention loop needs 8 loads together |
| R7 | SimplifyCFG sink / hoist common instructions, GVNSink (DARM-like melding of divergent branches) on top of -O3 | 0 kernels changed | -O3 already does it; no similar-but-different branch pairs in these kernels |
| R8 | 64-bit division bypass (addBypassSlowDiv) | no-op | already in AGXCodeGenPrepare |
| R9 | 4x8 matrices per SIMD-group tile GEMM | 13% fewer cycles per element but 128 GPRs, spills, 448 -> 384 threads | **check on hardware**: spill and occupancy costs are uncertain in the model |
| R10 | Software-pipelined tile GEMM loop; SIMD-group-per-row CSR SpMV | +19..125%; scalar kernel wins at all row lengths tried | |
| R11 | Speculating divergent branches harder: SpeculativeExecution cost limit 40 (default 7, already in -O3 for divergent targets), two-entry PHI folding 12 (default 4) | rw geomean -0.31% cycles (jacobi2d -7.4%), lb -0.08% tput (one conv kernel -12.9%), nothing worse | small; only a late run after -O3 finds these (the limits inside zoc's -O3 pipeline change none of 450 PTX-test or 124 CUDA kernel x chip results), so adopting needs a late SpeculativeExecution + SimplifyCFG in the AGX IR pipeline and a global cost option. **Hardware check**: jacobi2d, agx_conv2d_im2col_f16 with `opt -spec-exec-max-speculation-cost=40 -two-entry-phi-node-folding-threshold=12` |
| R12 | Unrolling single-block loops with device loads x2/x4/x8 (runtime trip counts) for memory-level parallelism | no suite kernel faster; lb tput +0.6..+1.1% geomean (more GPRs), worst +85% | |
| R13 | Occupancy-targeted scheduling: register-pressure limits scaled to 104 / 83 / 64 / 48 GPRs | mixed, +-20% on tile GEMMs; kernels that gained resident groups gained little tput (mma_m16n8k32_s32_s8: 72 -> 34 GPRs, 22 -> 48 groups, tput -1.9%; conv3x3 32 -> 40 groups, tput +5%) | the model's latency hiding saturates near 20-30 groups. **Hardware check**: mma_m16n8k32_s32_s8_s8_s32 (ZLUDA PTX test) with `llc` at the default vs a register-limited schedule: if occupancy matters more on the M2 than in the model, revisit |
| R14 | EarlyCSE or GVN after AGXCodeGenPrepare's division expansion (share the reciprocal work of divisions by the same divisor: permute4d's 12 div / rem pairs by 3 runtime sizes) | permute4d 2096 -> 2087 instructions, +0.6% cycles; elsewhere +-noise (G13/G14 tile GEMMs both ways) | the zero-divisor guards (select / umax per element, in separate structured blocks) keep the divisors distinct; not a backend problem |
| R15 | G13: register-blocked scalar SGEMM (4x4 / 8x4 / 4x8 / 8x8 per thread) instead of the tile GEMM with emulated SIMD-group matrices | 0.155-0.171 cycles per element-k per group (8 groups) against 0.072 for agx_sgemm_tile64 on G13G (FMA peak 0.031); 8x8 spills | the emulated-matrix tile kernel is already 2x better; the earlier "10x slower than G14" compared a 4-group makespan |
| R17 | Same A-late order for sgemm `_tb` / `_tatb`, B-early order for `_tb` | sgemm _tb +11% G14G, +17..21% G13G | |
| R16 | **Atomic optimizer** (`-agx-atomic-optimizer`, off by default; AGXCodeGenPrepare, after AMDGPUAtomicOptimizer): 32-bit int atomics to a uniform address -> one atomic by the first active thread (count x operand when the operand is uniform, else simd reduce), results rebuilt with an exclusive scan (add / sub) and a shuffle | **worse in the model**: cuda_compact 2530 -> 3576 cycles (1 group), 3710 -> 9069 (48 groups); PTX atom_add +68% | agx_cost does not serialize same-address atomics, which real GPUs do (the reason AMDGPU and CUDA compilers aggregate). **Key hardware check**: `kernels/cuda_compact.ll` (counter with result), `reduce_shfl`-style sums, PTX `atom_add`, with and without the option, at full occupancy. Correct on the simulator (PTX-formalization, agx_libs). Float add excluded (a reduction rounds and flushes denormals unlike the atomic) |

## Research sources (policy)

Experiments are based only on publicly published work: open-access papers (arXiv, ACM / IEEE papers with public
copies, university repositories), open-source compilers (LLVM, Mesa) and public documentation. **Patent documents
are not used as sources**: search results that are patents or patent applications are skipped and nothing is
implemented from them. This is not a freedom-to-operate review: a published technique can still be covered by a
patent, which only a legal review can establish. Techniques already shipped in open-source compilers are preferred.

| Round | Source | Kind | Used for |
|---|---|---|---|
| 1 | Saumya, Sundararajah, Kulkarni, "DARM: Control-Flow Melding for SIMT Thread Divergence Reduction", CGO 2022 (arXiv 2107.05681) | open-access paper | R7 |
| 1 | Lee et al., "Convergence and scalarization for data-parallel architectures", CGO 2013 | paper (public copy, Berkeley) | R1-E6 (deferred) |
| 1 | LLVM AMDGPU GCNSchedStrategy, UniformityAnalysis, SpeculativeExecution | open-source (Apache-2.0 with LLVM exception) | R11, R13, tput metric |
| 1 | Magni, Dubach, O'Boyle, thread coarsening studies (Edinburgh) | papers (university repository) | R1-E6 (deferred) |
| 1 | Lee et al., many-thread-aware prefetching for GPGPUs, MICRO 2010; NVIDIA developer blog on prefetching | paper, public blog | R12 |
| 1 | "Characterizing Warp Divergence from Pascal to Blackwell" (arXiv); NVIDIA Research, SPMD divergence management (2014) | open-access papers | R11 |
| 1 | Tillet, Kung, Cox, Triton (MAPL 2019); ML-Triton (arXiv 2503.14985) | open-access papers | background |

## Compile time (policy)

Every adopted change records its `llc` compile-time cost. A change that makes a real difference to the generated
code but costs significantly more compile time (over ~10% of `llc` time on the kernel sets, or more than a few
seconds on a single large kernel) is not enabled by default: it gets its own `llc` option, listed here, and
`zoc` / ZLUDA pass it only on request (`ZLUDA_AGX_LLVM_ARGS`). So far none qualifies:

| Change | llc time | Default |
|---|---|---|
| A3 post-RA MachineScheduler | +2% (4.45 -> 4.55 s over 209 files) | on |
| A1 / A2 memory clustering | not measurable | on |

## Model gaps that decide the entries above

1. Occupancy (`tput`): resident groups = min(96, 2 * floor(3328 / (2 * register halves))), whole workgroups,
   64 KiB threadgroup memory per core. A1-A3 and R1, R9 depend on it.
2. Memory after ALU (A1, A2): `mem_quiet_device_store_ref` 33.6, `mem_quiet_load_ref` 20. Does a `wait` or a
   load between stores reset the stall? Do 8-16 back-to-back loads issue as modelled?
3. G13 numbers reuse G14C rows; G13 tile GEMMs (matrix ops emulated, 128 GPRs, spills) move +-20% with any
   scheduling change.
4. Launch overhead, coalescing and cache behaviour are not in `cycles` / `tput`.

AGX-Performance-Simulator `tasks/zluda-agx-followups.md` lists the same gaps from the simulator's side.
