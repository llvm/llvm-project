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
| A11 | ZLUDA cuBLAS sgemm with transa = T (single product, n >= 256, m, k >= 64): A^T written to scratch by agx_sgeam, then the untransposed tile kernel | `ZLUDA_AGX_GEMM_PRETRANSPOSE=0` (runtime) | tile kernel 22.7k -> 17.0k cycles per SIMD-group at K = 256 (G14G) for ~1.5 cycles per element of A transposed: break-even near n = 70 in the model | the model ignores bandwidth; a strided transpose is bandwidth-bound, so the threshold (256) is conservative | `cublasSgemm(T, N)` at 512..4096, with and without the variable |
| A12 | ZLUDA cuBLAS sgemm split-K: single f32 products with <= 16 output tiles and K >= 1024 cut into S = min(16, 64 / tiles, K / 256) slices (one batched launch into scratch, then agx_ssum_slices sums into beta C in one pass) | `ZLUDA_AGX_GEMM_SPLITK=0` (runtime) | not measurable in agx_cost (one SIMD-group, not the spread over cores); expected ~S x on wall clock for 64 x 64 x 4096 minus one pass over S m x n partials | the tile counts per core; that scratch traffic is cheap | `cublasSgemm` at 64x64x4096, 128x128x8192, 64x512x4096 |
| A13 | MachineSink sinks preheader instructions into loops under register pressure (`-sink-insts-to-avoid-spills`, now on for AGX through a new `TargetSubtargetInfo::sinkInstsIntoCyclesToAvoidSpills` hook) | `-sink-insts-to-avoid-spills=false` | cycles: n-body -4.3%, ~-1.5% total on every set, none slower; tput: 26 lb kernels better, 4 worse by <= 2.2%; llc time unchanged | spill cost (stack accesses) vs. recomputing inside the loop | cuda_nbody, kernels that spill |
| A14 | 64-bit add of a constant that fits 32 bits as `iadd` with a zero-extended 32-bit register (one mov_imm and GPR instead of a materialized pair) | revert "[AGX] 64-bit add of a 32-bit constant..." | 4 suite kernels -0.1..-0.2%, none worse; lb -0.2% (tile GEMMs move both ways) | | specialized kernels (pointer steps become constants) |
| A15 | **Wide constants of kernels in uniform registers** (AGXUniformConstants, `-agx-uniform-constants`, off in `llc`): each 32-bit `mov_imm` of a kernel becomes a uniform preloaded after the arguments (one per value, at most 64), listed in `.agx.kernels` extension 6 for the launcher; AGXFoldOperands folds them into the users. **On in ZLUDA** (`compile_agx`, agx_libs builds; agx_runtime sets the uniforms) | `ZLUDA_AGX_UNIFORM_CONSTANTS=0` (ZLUDA, rebuild agx_libs too); `-agx-uniform-constants` in `llc` | tput geomean rw -2.4% (sgemm_reg4x4 -27.6%: 70 -> 66 GPRs, ddot -7%), co -2.0% (transcendental -29.9%), lb -1.0%; cycles ~0 (-0.1..-0.5% geomean; gelu_f16 +15% and softmax_online +4% from a different store / stall order, not from uniform reads); specialized CUDA kernels (R21) 2-5 GPRs fewer, tput -2..-4% (rmsnorm still above its unspecialized form); PTX tests and agx_libs simulator tests unchanged; llc time within noise | a uniform source costs the same as a GPR source (the model has no uniform penalty); preloading ~2 more uniform words per constant is free at dispatch | `cuda_sgemm_reg4x4`, `cuda_ddot`, `softmax_row`, `transcendental`, agx_libs dnn / blas; compare `ZLUDA_AGX_UNIFORM_CONSTANTS=0`; watch gelu_f16-like store tails |
| A16 | Post-RA strategy (AGXPostRASchedStrategy): the next access of a memory cluster ranks ahead of latency stalls, and the scheduler waits up to 8 cycles for a pending one instead of putting another instruction between two accesses | `-agx-postra-cluster-wait=0` vs default 8 (3 and 16 give the same) | cycles: gelu_f16 -13.9%, ldmatrix -15.1%, none worse (rw geomean -0.42%, co -0.19%); tput +-1% noise (attn_decode +1.0%: three moves reordered); agx_libs simulator tests unchanged; llc time not measurable | `mem_quiet_device_store_ref` (as A1): an ALU op between stores costs more than a few cycles' stall | `cuda_gelu_f16` (two 16-bit stores per iteration), `ldmatrix*` PTX tests |
| A17 | f32 add / mul / fma whose only use is a conversion to half precision write the 16-bit register themselves (FMUL32_F16_F32, FMADD32_F16_F32, FADD32_F16_F32), unless a source extends half precision (the FMADD32_M forms keep it) | revert "[AGX] Fold f32-to-f16 conversions..." | G14C hgemm tile64 cf16 -3.9..-5.6% cycles (all 8), gelu_f16 tput -2.1%, none worse (G13G hgemm -4..-50%: spilling tiles); PTX check and agx_libs hgemm / epilogue / dnn / conv simulator tests unchanged; llc time not measurable | applegpu: a 32-bit operation with a 16-bit destination rounds to f32 (flushing f32 denormals, which are 0 in f16 anyway) and converts, as fptrunc of it does | cuBLAS hgemm with f16 C, gelu_f16; check f16 results bit-exact against fptrunc of the f32 result |
| A18 | **Store vectorization limited by register pressure** (L3-E1): AGX TTI estimates each function's peak live 32-bit values (SSA liveness, backward dataflow over the IR) once, and when it reaches the subtarget's GPR count the LoadStoreVectorizer may merge stores only up to 4 bytes (a vector store needs its sources in consecutive registers, which forces copies and spills under pressure) | `-agx-store-vector-pressure=0` (off; default -1 = the GPR count); `-agx-max-store-vector-bytes` | both paths: sgemm_reg4x4 tput -27.7%; AIR path mma_m16n8k32_s32_s8_s8_s32_interleave cycles -75% (the L1-E1 +300% case); AIR vs direct co -5.18% cycles / -3.30% tput, rw +0.09% / -0.17%, lb ~0; llc time 4.42 vs 4.38 s over libll (noise) | an IR estimate tracks the final pressure closely enough to separate spilling kernels from the rest | `cuda_sgemm_reg4x4`, `mma_*_interleave` PTX tests, tile GEMM epilogues |

| A19 | Software f64: a float, half or bfloat widened to double exactly by rebiasing the exponent and shifting the fraction (a denormal normalized by one count of leading zeros) instead of the general unpack / round (L4-E3) | revert "[AGX] Exact widening to double..." | cvt_f64_f32 -16.6% cycles / -14.8% tput (direct path); nothing else changes | the conversion is exact, so no rounding step is needed | `agx_f64_check.py fpext` (8192 inputs bit-exact, both NaN modes), `cvt.f64.f32` PTX tests |
| A20 | **A 32-bit divisor's reciprocal computed once** (L5-E1, AGXCodeGenPrepare): the divisor-only part of the 32-bit division expansion (|y|, the f32 reciprocal estimate, its Newton step) is built right after the divisor's definition and shared by every division by it; a divisor that is a cheap pure expression (ZLUDA's `umax(y, 1)`, a trunc in each 64-bit division's 32-bit fast path) is rebuilt after its leaves, so the copies in unrolled iterations share it | `-agx-share-divisors=0` | both paths: permute4d -9.8% cycles / -8.1% tput (46 -> 60 GPRs, still faster); rw geomean -0.30%, co -0.05%; lb -0.31% cycles / -0.12% tput geomean with tile GEMMs moving both ways (sgemm_tile64_u -15.4%, hgemm_tile64_cf32_ta +11.9%: register allocation) | the estimate z depends on the divisor only; the corrections keep each quotient exact | `cuda_permute4d`, `divrem32.ll`, PTX div / rem tests |
| A21 | **64-bit induction variables counted in 32 bits** (L5-E2, AGXNarrowInductions, after loop strength reduction): a 64-bit header phi incremented by a loop-invariant amount whose value, increment and step scalar evolution bounds below 2^31 becomes a 32-bit phi, its 64-bit uses the zero extension (PTX's byte offsets over a threadgroup tile: one iadd and the loop test instead of a 64-bit add and three instructions comparing both halves) | `-agx-narrow-ivs=0` | both paths: nbody -6.1% cycles and tput; nothing else changes | | `cuda_nbody`, `narrow-inductions.ll` |
| A22 | **Bitfield extracts in one bfeil** (L6-E1): `(x >> s) & (2^k - 1)` (32 and 16 bits) is one `bfeil` with offset s and a k-bit mask when the shift has no other use (otherwise the shift stays and a cheaper `and` reads it: fma_bf16x2 +10% tput without that rule); `(x & (m << s)) >> s` becomes that form first (AGXISelLowering) | revert "[AGX] Bitfield extracts..." | both paths: ddot -9.7% cycles (AGXSoftF64), q4_matvec -6.7% (nibbles), rw geomean -0.54% / -0.39% tput, co -0.16% / -0.15%; lb -0.79% total cycles but G13 tile GEMMs at 128 GPRs move +3..+46% (spills: sgemm_tile64_u 1478 -> 1578 instructions), G14 unchanged | bfeil (latency 3) replaces a dependent shift and and (2 + 3) | `agx_f64_check.py` (all operations bit-exact, both NaN modes), `bitfield-extract.ll`, `cuda_q4_matvec` |
| A23 | **Branch conditions in 32 bits** (L6-E3, AGXISelLowering): the zero extension of 16-bit boolean logic (i1 promoted to i16: compares, 0 / 1 truncations, and / or / xor) to 32 bits for a branch is the same logic in 32 bits, where the compares produce it, and an if / loop on `b ^ 1` for a boolean b branches on `b == 0`, which the branch compares itself (bitonic sort's `(a > b) == up`: no 16-bit xor, no move, no xor with 1) | revert "[AGX] Branch conditions in 32 bits..." | both paths: bitonic_smem 818 -> 712 instructions, -2.4% cycles / -4.2% tput; subc_cc_s32 -16.7% tput; add_extended +4.7% tput (8.6 -> 9.0, same 29 instructions), agx_cger +2.0% tput | | `boolean-branch.ll`, `cuda_bitonic_smem` |

## Cumulative (model): session start (41842c90f) to A16 (ab8a3ff4c)

The same IR through the `llc` of 41842c90f (before A1-A16 and the earlier precise-math, wait and soft-f64 changes)
and the current one, G13G and G14C, agx-bench. Geomean / total over kernel x chip; negative is faster.

| Set | Metric | `llc` default | ZLUDA (+ `-agx-uniform-constants`) |
|---|---|---|---|
| CUDA suite (36 kernels) | cycles | -5.0% / -10.7% | -5.5% / -10.7% |
| | tput | -0.2% / -6.5% | -2.6% / -12.9% |
| | GPRs | -0.4% | -4.3% |
| PTX tests (200 files) | cycles | -1.2% / -1.2% | -1.7% / -2.3% |
| | tput | +0.7% / +0.3% | -1.3% / -2.3% |
| | GPRs | +1.1% | -2.1% |
| agx_libs (198 kernels) | cycles | -4.0% / -18.0% | -4.1% / -18.4% |
| | tput | -0.7% / -3.3% | -1.4% / -4.0% |
| | GPRs | -2.0% | -5.8% |

Largest gains (G14C cycles): hgemm tile64 variants -45..-50%, vector_operand -43%, gelu_f16 -37%, blur_rgba8 -37%,
attn_decode -34%; tput: sgemm_reg4x4 -39%, layernorm_f16 -26%, match_sync -24%. Slower by design: `transcendental`
+279% and softmax_row +23% (precise sin / cos / exp / log instead of the hardware approximations), div_ftz +19% and
min_nan_f16 +33% (IEEE results). Slower as a scheduling trade-off (A3): stencil +29%, reduce +28%, bitonic_smem +12%.

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
| R18 | TTI arithmetic costs at expansion sizes (f64 add / mul / div 180 / 190 / 280, 32-bit division by a variable 30, 64-bit 120), through the real -O3 pipelines (zoc for the CUDA suite and PTX tests, opt -O3 for agx_libs) | CUDA IR unchanged; agx_libs IR changed in blas / solver only, 0 instructions or cycles different | the libraries emulate f64 in MLIR before LLVM; no cost-driven decision flips |
| R19 | Other codegen options (task #14), all three sets: no machine LICM (cycles +2%, tput -3.5%, sgemm_reg4x4 tput -28% from fewer GPRs: mixed), no machine sink (lb geomean +10.5% / tput +19.5%), no branch folding (~0), no early tail duplication, no machine CSE (lb -1% but +18% tput on index_add), split-spill-mode=size, no post-RA machine LICM (~0) | mixed or worse | machine LICM is a latency-for-registers trade the model splits by metric: **hardware check** `-disable-machine-licm` on sgemm_reg4x4 / conv3x3 |
| R20 | R12 again with runtime unrolling allowed expensive trip counts (strided `i += blockDim` loops), x2 / x4 | spmv_csr -21% / -32% but q8_matvec +71%; lb tput geomean +2.7% / +3.2% | a generic unroll loses; the targeted rule (single-block loops, 1-2 device loads, <= 24 instructions) changes no suite kernel and costs lb tput +0.3% geomean (dnn_bn_stats +73%) |
| R21 | **ZLUDA launch-time specialization** (existing, opt-in `ZLUDA_AGX_SPECIALIZE`: scalar arguments and block size as constants), emulated on the CUDA suite (G14C) | permute4d -85% cycles / -94% tput (2096 -> 46 instructions), attn_decode -24% / -13%; but rmsnorm +81% tput, softmax_online +20%, conv3x3 +16%: constants too wide for immediates land in GPRs (14 -> 21 GPRs, fewer resident groups); not unrolling (an unroll limit changes nothing) | worth enabling only with large constants kept out of GPRs: a launcher-filled constant table in uniform registers (the open R1-E6 direction). **Hardware check**: permute-style and rmsnorm-style kernels with `ZLUDA_AGX_SPECIALIZE=1` |
| R22 | SimplifyCFG branch folding with more bonus instructions before llc (`simplifycfg<bonus-inst-threshold=2/4>`: argmax's `v > best \|\| (v == best && j < bi)` steps become selects, 195 -> 164 instructions) (L6-E2) | only argmax changes: cycles +1.8%, tput -2.3% (the reductions are not its hot loop) | too little to add a pass |
| R16 | **Atomic optimizer** (`-agx-atomic-optimizer`, off by default; AGXCodeGenPrepare, after AMDGPUAtomicOptimizer): 32-bit int atomics to a uniform address -> one atomic by the first active thread (count x operand when the operand is uniform, else simd reduce), results rebuilt with an exclusive scan (add / sub) and a shuffle | **worse in the model**: cuda_compact 2530 -> 3576 cycles (1 group), 3710 -> 9069 (48 groups); PTX atom_add +68% | agx_cost does not serialize same-address atomics, which real GPUs do (the reason AMDGPU and CUDA compilers aggregate). **Key hardware check**: `kernels/cuda_compact.ll` (counter with result), `reduce_shfl`-style sums, PTX `atom_add`, with and without the option, at full occupancy. Correct on the simulator (PTX-formalization, agx_libs). Float add excluded (a reduction rounds and flushes denormals unlike the atomic) |
| R22 | 64-bit constants (REG_SEQUENCE of two mov_imm) as aligned uniform pairs under A15, so AGXFoldOperands folds them into 64-bit sources | (reverted) | G14C cycles -0.01% geomean, tput -0.1% (spmv_csr -3.2%); G13G +0.57% (spilling tile GEMMs at 128 GPRs swing -15..+62%) | not worth the code; 64-bit constants are rare once A14 handles 32-bit adds |
| R23 | **Preamble (upper bound only)**: share of simulated cycles in instructions whose sources are only uniforms (or values computed from them), which a Mesa-style per-dispatch preamble could compute once | (not implemented) | CUDA suite 2.2% of cycles (philox 58%, permute4d 18.5%, rope 17%, mandelbrot 16%, ddot 5%, nbody 3%; most kernels < 1%) | too small on average for a new dispatch path (preshader USC words, unknown hardware behaviour); philox-style kernels benefit from ZLUDA specialization instead. **Hardware check**: none needed yet |

## Experiment: PTX -> upstream AIR -> AGX (`ZLUDA_AGX_VIA_AIR=1`)

A separate path (llvm/docs/AIR/AIRHints.rst): `agx-raise-to-air` writes ZLUDA's AGX IR as upstream AIR with
hints (operations AIR lacks, each with an AIR emulation; soft-f64 above AIR), AIR's portable passes run, and
`agx-lower-air` restores the AGX operations; the same AIR goes to the Metal pipeline. The direct path is
unchanged (asm identical, bench identical, d3 = d4).

| # | Change (llvm commit) | Model result, AIR path vs direct (same `llc` flags) | Correctness |
|---|---|---|---|
| X1 | Raise + hints + soft-f64 above AIR, by-value parameters restored (727052a5d and before) | cycles geomean rw -0.01%, co -0.00%, lb -0.01%; no kernel > 2% slower | PTX-formalization 408 pass = direct |
| X2 | Directed rounding / FP mode / exact min-max emulations (bb72df692, f534ac876) | AGX code unchanged (asm identical on the strict-FP tests) | every hint emulated: 381 -> 399 pass; left: f64 atomics, hardware approximations |
| X3 | System values read where the kernel extracts them (e4934c6dd); scalar ones stay at the entry (reading them where their uses meet: G14C +0.02%, rejected) | rw 72/72 identical (was 64), co 474/476, lb 434/468 (tile GEMMs reading the SIMD lane); G14C geomean -0.007% | 408 pass |
| X5 | Scalar system values (SIMD lane, SIMD-group index, thread index) through a `+value` hint at the read, whose emulation is AIR's parameter; AGX reads the register at the hint | **every kernel identical to direct**: rw 72/72, co 476/476, lb 468/468 | 407 pass native (tex-mip: the debug zoc's 121 s vs the checker's 120 s, both paths); emulated 404; Metal 246/248 |
| X6 | **Portable AIR**: every hint but the matrix FMA becomes its AIR emulation, inlined and folded at the use (`-agx-raise-hints`, default the matrix FMA); f64 atomics as 64-bit cmpxchg loops; `strictfp` dropped. AGX recovers what it can from plain AIR (`recoverAGXOperations`: afn 1/x, 1/sqrt, sin(x pi/2) as rcp / rsqrt / quadrant sin; the fmin-with-NaN-check idiom as minimum, in bfloat / half; `.ftz` flushes where the f32 ALU flushes). Measured first per kind (`-agx-air-emulate-hints`): system values +-0.01%, f64 -0.25% (faster), approximations rw +0.9% (nbody +36% tput), min/max up to +72%, FP mode co +0.9% (add_ftz +310%), matrix G14 GEMMs +15% (kept) | cycles geomean rw +0.29% (ddot +11%, its tput -1.2%), co -0.01% (call_rnd +73%, cvt_f64_f32 -36%), lb -0.01%; 2 of 248 modules keep a hint | 408 pass, with the matrix hint native and emulated; Metal 246/248 |
| L1-E1 | **Generic pointers as device pointers in AIR** when nothing in the module makes a generic pointer to other memory (cvta.shared / .local / .param) or calls through a pointer: AGX treats exactly those as device pointers (AGXLowerGenericAddresses), so the AIR is portable and Metal-correct, and AGX's alias analysis and load/store vectorizer now see device accesses (the direct path does not vectorize generic ones). 241 of 245 modules drop the generic hint. Store width limits tried (8 / 4 bytes, `-agx-max-store-vector-bytes`): lb +1.5% at 4, kept 16 | vs direct: co cycles -4.19% / tput -2.50% geomean (130 better, div_ftz -38%); rw / lb unchanged; mma_m16n8k32_s32_s8_s8_s32_interleave +300% (16 stores merged into 4 extend live ranges to 128 GPRs: open) | 408 pass; Metal 246/248 |
| L1-E2 | **Directed rounding of float add / sub / mul by error-free transformations** (TwoSum, the fma product error; Graillat, Lamotte, Nguyen Hong; Rump, Ogita, Oishi): the nearest result, the sign of its exact error, one unit in the direction; operands and result in [2^-70, 2^127) or zero (so the error is normal on a flushing ALU), exact zero sums by the IEEE sign rule; the rest soft-f64. The branch tests `ok or rounding is nearest`, so it folds away where the mode is known | call_rnd -60% cycles / -77% tput (305 cycles: below the direct path's 590, AGX's integer directed rounding); nothing else changes | 408 pass; host check (`tools/check_directed_eft.c`): 0 failures in 11.7M fast-path cases |
| L1-E4 | soft_f64.c add: denormal operands as exponent 1 without the implicit bit instead of normalized (two clz and shifts fewer; round_pack normalizes the sum) | fadd 195 -> ~171 instructions (378 cycles, direct 392), ddot +11% -> +3.4% against direct; rw geomean -0.19%, co -0.08% | soft_f64 host check 2M cases, f64 / testfloat PTX tests pass |
| L3-E1 | Store vectorization limited by register pressure in AGX TTI (A18; both paths) | co -5.18% cycles / -3.30% tput geomean vs direct (was -4.19% / -2.50%), interleave +300% -> -75% vs its L1-E1 form | 408 pass (tex-mip alone: under -j 4 the debug zoc passes the 120 s limit, as before) |
| L4-E1 | **Vector accesses shaped for AGX** (AGXVectorAccesses, around the load/store vectorizer): a vector load of which only some lanes are read (PTX `ld.v4` into `_`) loads only those lanes, so the vectorizer can merge it with its neighbours; a device vector store whose lanes are lanes of loads in another order (a swap, two loads) is split into the runs that need no register copy (a store right after the copy waits ~34 cycles). Threadgroup stores keep the vector: their split needs a second address register (bitonic sort +3.9%) | AIR path: ld_st_offset -36.8% cycles, sink -22.2% tput, co geomean -0.37% cycles / -0.19% tput, nothing worse; direct path unchanged (its generic accesses are not vectorized) | 408 pass on both paths (`-agx-narrow-vector-loads=0`, `-agx-split-permuted-stores=0` switch it off) |
| L4-E2 | **Rejected (flag, off): AIR's software f64 calls as AGX's double operations** (`-agx-air-native-f64`: agx-lower-air turns round-to-nearest `__air_f64_*` calls into fadd / fmul / fma / fpext / ... for AGXSoftF64). The portable library is as fast: ddot -3.3% cycles but +6.2% tput (52 -> 58 GPRs), fmul / fdiv faster, fadd / sqrt / fma tput / f64 atomics / cvt_f64_f32 slower (+4% .. +107%); co geomean +0.31% cycles / +0.50% tput | the AIR soft-f64 stays | -- |
| L2-E1 | **Rejected: InstCombine on the raised AIR** (inside the raise, target-neutral, kernels only). co -0.13% cycles / -0.34% tput, lb -0.11%, but stencil +77%: InstCombine sinks the loads the CUDA source hoists above its bounds check, and its code sinking cannot be switched off from a pass here (a per-context library option). **Measurement pitfall:** `opt -mtriple=agx -passes='agx-raise-to-air,function(instcombine)'` showed ldmatrix -69%: AGX's alias analysis takes AIR's address space 4 (the generic hint) for AGX's constant memory and InstCombine deleted the stores. Never run AGX-target IR passes over AIR. Kept: agx-lower-air also recognizes the `x == 0` form of the .ftz flush that InstCombine makes | no gain worth it | -- |
| X4 | Metal pipeline legalizes the hints: generic -> device, 65-128-bit integers as limbs, RMW loops (778b681ba) | n/a (Metal side) | Metal writer 241 -> 246 of 248 modules; limbs = LLVM on 20000 random i128 / i96 cases (lli) |

Compile time: `zoc` (debug build) over the 40 largest PTX tests 258.7 s direct vs 265.1 s through AIR (+2.5%);
`agx-raise-to-air` itself ~1.6 ms per module, the AGX side of the AIR (`agx-lower-air` + codegen) no slower than
direct.

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
| 2 | Osama, Merrill, Cecka, Garland, "Stream-K", PPoPP 2023 (arXiv 2301.03598) | open-access paper (motivation only; the implementation is plain split-K as in open-source BLAS libraries) | A12 |
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
| A15 uniform constants | within noise (3.28-3.41 s vs 3.41-3.45 s over the agx_libs kernels) | off in llc, on in ZLUDA |

## Model gaps that decide the entries above

1. Occupancy (`tput`): resident groups = min(96, 2 * floor(3328 / (2 * register halves))), whole workgroups,
   64 KiB threadgroup memory per core. A1-A3 and R1, R9 depend on it.
2. Memory after ALU (A1, A2): `mem_quiet_device_store_ref` 33.6, `mem_quiet_load_ref` 20. Does a `wait` or a
   load between stores reset the stall? Do 8-16 back-to-back loads issue as modelled?
3. G13 numbers reuse G14C rows; G13 tile GEMMs (matrix ops emulated, 128 GPRs, spills) move +-20% with any
   scheduling change.
4. Launch overhead, coalescing and cache behaviour are not in `cycles` / `tput`.
5. Uniform registers (A15): the model charges nothing for a uniform source or for more preloaded uniforms. Does a
   larger uniform count slow the dispatch, or limit resident workgroups?

AGX-Performance-Simulator `tasks/zluda-agx-followups.md` lists the same gaps from the simulator's side.
