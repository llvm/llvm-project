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

## Model gaps that decide the entries above

1. Occupancy (`tput`): resident groups = min(96, 2 * floor(3328 / (2 * register halves))), whole workgroups,
   64 KiB threadgroup memory per core. A1-A3 and R1, R9 depend on it.
2. Memory after ALU (A1, A2): `mem_quiet_device_store_ref` 33.6, `mem_quiet_load_ref` 20. Does a `wait` or a
   load between stores reset the stall? Do 8-16 back-to-back loads issue as modelled?
3. G13 numbers reuse G14C rows; G13 tile GEMMs (matrix ops emulated, 128 GPRs, spills) move +-20% with any
   scheduling change.
4. Launch overhead, coalescing and cache behaviour are not in `cycles` / `tput`.

AGX-Performance-Simulator `tasks/zluda-agx-followups.md` lists the same gaps from the simulator's side.
