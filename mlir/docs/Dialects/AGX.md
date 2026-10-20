# 'agx' Dialect: performance operations for Apple GPUs

[TOC]

The `agx` dialect exposes the parts of Apple GPUs (the AGX architecture) that make compute kernels faster than generic
`gpu`/`vector` code can. It sits next to the standard dialects in library kernels: a GEMM is `scf` loops, `vector` and
`memref` code that calls `agx.mma_tile` for its inner product and `agx.block_barrier` instead of a full barrier. The
`agxir` dialect below it is a 1:1 view of the LLVM AGX backend's intrinsics, so no `agx` operation encodes a
machine instruction. Each one describes an effect that every generation can provide.

## Design rules

1. **Version independence.** An operation's semantics never depend on the chip. Each operation has:
   - a *native lowering* for the generations whose hardware does it;
   - a *portable lowering* built from simpler operations that is correct everywhere.

   The chip is named only in the `#agxir.target<chip = ...>` attribute of the `gpu.module`, which the runtime sets
   from the device. The conversion to `agxir` reads the chip's capabilities from one table (`AGXChipInfo`, below) and
   picks the lowering. A kernel written today therefore compiles for G13 (M1) through Apple9 (M4, A18 Pro) and later
   chips, and gets faster on a new chip when that chip's row of the table changes, not its source.
2. **Measured, not guessed.** A native lowering is enabled for a generation only when the behaviour it relies on was
   observed on that hardware. The evidence is either the hardware-recorded suites of the AGX Performance Simulator and
   Apple-M2-Playground (G14C, M2 Max), or the clean-room agx-re documentation (Apple9 G16G/G17P, M5 G17G). Until then
   the portable lowering is used. Open questions are listed in the simulator's `tasks/ptx-re-blockers.md`.
3. **Standard dialects first.** If an effect can be reached by lowering a standard operation better, it is done in
   the backend and not given an `agx` operation. Examples: float atomic add, 64-bit atomic min/max, vector loads and
   fences. `agx` operations are for what the standard dialects cannot say: the cooperative matrix layout, lane-offset
   arithmetic, partial-threadgroup barriers, per-core uniform work, and launch hints.

## Capability table (`AGXChipInfo`)

`mlir/include/mlir/Dialect/AGX/Utils/ChipInfo.h` maps a chip name (`agx-g14c`, ...) to the facts below.

| Capability | G13 (M1) | G14 (M2) | G15 (M3) | G16/G17P (M4, A18 Pro) | G17G (M5) |
| --- | --- | --- | --- | --- | --- |
| SIMD width | 32 | 32 | 32 (assumed) | 32 | 32 |
| `matrix_unit` | emulated (shuffle + FMA) | emulated (operands of `simd_matrix_fmadd` not mapped) | emulated | native 8x8x8 MAC (`0xcf`), f16/bf16/f32, f32 accumulate, A·B−C mode | native tile load/store + MAC (`m5_matrix_mac`) |
| `lane_shift_fma` | no data | native (fmadd bit 53, within 16-lane halves) | no data | none known | none known |
| `block_barrier` (4 SIMD-groups) | no data | native (0x78: 2.9 ns vs 9.3 ns) | no data | not found | not found |
| `float_atomic_add` (device, f32) | no data | native (atomic bit 45 on `sub`) | no data | native (RMW op `fadd`) | native |
| `u64_atomic_minmax` (no return) | no data | native (bit 45 on `cmpxchg`) | no data | none (all 64-bit atomics absent) | none |
| `preamble` (per-core uniform program) | yes (Mesa) | yes: runs once per core per dispatch | yes | yes (per-stage uniform preamble) | yes |
| threadgroup memory per core | 64 KiB (Mesa) | 64 KiB, 32 banks x 4 B, 256 B units | dynamic caching | 32 KiB static, 64 KiB shared budget | 32 KiB static |
| GPRs before spilling | 128 | 128 (occupancy `2*floor(3328/(2*halves))`) | dynamic caching | 96 | 126 |

"no data" means no generation-specific measurement exists. The portable lowering is used there until one does.

## Operations

### Matrix: `agx.mma`, `agx.matrix_load`, `agx.matrix_store`, `agx.mma_tile`

A *fragment* is an 8x8 matrix spread over the 32 threads of a SIMD-group, two elements per thread
(`vector<2xT>`). Thread `t` holds `(r, c)` and `(r, c+1)` with `r = 4*(t/16) + (t/2)%4` and
`c = 4*((t/8)%2) + 2*(t%2)`. This layout is fixed by the operations, not by the chip. Each lowering converts if its
hardware layout differs.

- `agx.mma %a * %b + %c {negate_acc}` computes D = A·B ± C on fragments.
  - Element types: f16, bf16 and f32; mixed f16/bf16 inputs with an f32 accumulator are allowed.
  - Native on chips with `matrix_unit`. Apple9 has a native A·B−C mode (`negate_acc`); elsewhere C is negated first.
- `agx.matrix_load %mem[%i, %j] {transpose}` / `agx.matrix_store` move an 8x8 fragment to or from a 2-D `memref` in
  device or threadgroup memory. On Apple9 and M5 this is one 64-bit access per thread.
- `agx.mma_tile` is the SIMD-group tile product.
  - Signature: `%d = agx.mma_tile %A[%i, %k], %B[%k, %j], %acc : memref<...>, memref<...>, vector<MxN/32 x T>`, with
    M, N, K multiples of 8.
  - It accumulates an MxN tile over K, reading A and B straight from memory.
  - Its accumulator is the per-thread array of the MxN/64 fragments in row-major fragment order. This is the
    `cooperative_tensor` of Metal Performance Primitives.
  - It lowers to fragment loads and `agx.mma` everywhere, and to a hardware tile path where one is mapped (M5).
  - This is the operation library GEMM, convolution and attention kernels are written with. Apple's guidance for M5
    (Metal Performance Primitives guide) is used as the starting schedule: 32x32 SIMD-group tiles, 2x2 SIMD-groups
    per threadgroup, operands read from device memory without threadgroup staging, barriers every 128 along K, and
    Morton-ordered threadgroup walks.

### Lane-offset arithmetic: `agx.fma_lane_shift`

`%d = agx.fma_lane_shift %a, %b, %c, %k {width = 16 | 4 | 2} : f32` is `shuffle_up(a, k) * b + c` within segments of
`width` lanes, with 0 shifted in. It feeds prefix scans, stencils, 1-D convolutions and FFT butterflies.
- **Native lowering (G14):** fmadd32/fmadd16 bit 53 (10-byte form; bits 72/73 select segments of 2 or 4).
  - Only shifts that stay inside 16-lane halves are used. The cross-half read is timing-dependent and is never
    emitted.
  - An inactive source lane reads 0, which is also the portable semantics. A segment starting at an inactive lane
    therefore gives the same result on both lowerings.
- **Portable lowering:** `agx.shuffle up` with a segment mask, then `math.fma`.

### Synchronization: `agx.block_barrier`

`agx.block_barrier` waits for the threads of the caller's aligned group of 4 SIMD-groups (128 threads). It must be
reached by every thread of the threadgroup, in uniform control flow, like `agx.threadgroup_barrier`.
- **Native (G14):** the 0x78 barrier, about 3x cheaper than the full barrier.
- **Portable:** `agx.threadgroup_barrier`, which waits for a superset of those threads.

Whether 0x78 also orders threadgroup memory is open (ptx-re-blockers B16). Until it is answered, the native lowering
adds the threadgroup memory fence the full barrier implies.

### Per-core uniform work: `agx.uniform_region`

```mlir
%base, %stride = agx.uniform_region -> (i64, i32) {
  ...                       // depends only on kernel arguments and constants
  agx.yield %b, %s : i64, i32
}
```

The region computes values that are the same for every thread of the dispatch: argument-derived addresses, products
of grid sizes, constants.
- **Native lowering:** the region goes in the *preamble*. The preamble runs once per core per dispatch, with one
  active lane, and stores into uniform registers. About 2.8 µs buys 256 dependent instructions, versus once per
  thread.
- **Portable lowering:** inline the region. This is also what happens until the runtime installs preambles
  (agx_runtime, simulator task safe-09).

The verifier rejects regions that read thread-dependent values or memory.

### Introspection: `agx.core_id`, `agx.simd_slot`

`agx.core_id` (SR20: `cluster << 4 | core` on G14) and `agx.simd_slot` (the hardware SIMD slot) are for per-core
work distribution and debugging. Where a chip has no such register they lower to 0 / the SIMD-group index, and the
`agx.has_core_id` query folds accordingly.

### Launch hints: `#agx.launch_hints`

An attribute on `gpu.func`: `#agx.launch_hints<max_gprs = 64, threadgroup_bytes = 16384, occupancy = high>`.
- It selects the backend's register limit (`max-gprs-*`) and checks threadgroup memory against the chip's budget.
- On G15 and later, dynamic caching changes what occupancy means, so the hint is passed through and the chip table
  decides.
- Library schedules set it from the occupancy formula of the chip table.

### Existing operations

`agx.shuffle`, `agx.reduce` (with `prefix`), `agx.ballot`, `agx.threadgroup_barrier`, `agx.memory_barrier`,
`agx.set_fpmode`, `agx.get_fpmode` are unchanged. Collective costs on G14C:
- `simd_shuffle` / ballot: 4.8 cycles;
- `quad_shuffle`: 3.9 cycles;
- SIMD reductions: 22-23 cycles, which do not overlap within a group;
- prefix scan: about 17 cycles.

So library reductions over many values use shuffle trees and one `agx.reduce` at the end.

## Backend support used by standard operations

These are not `agx` operations. The LLVM AGX backend makes the standard operations below fast where the chip allows:

| Standard operation | G14 lowering | Fallback |
| --- | --- | --- |
| `memref.atomic_rmw addf` (f32, device) | native float atomic add | compare-and-swap loop |
| `memref.atomic_rmw maxu`/`minu` (i64, device, result unused) | native u64 min/max | lock table |
| `vector.load` of 2/4 elements | one vector load (vec4 = 4 core cycles per 512 B) | scalar loads |
| fences of every scope | measured barrier sequences (AGXUsage.rst, "Fences") | - |

## Library kernels

Library kernels (`mlir/lib/Dialect/AGX/Library`, used by ZLUDA's cuBLAS / cuDNN / cuFFT replacements) are MLIR
functions in `gpu.module`s that mix standard dialects and `agx` operations. Each is compiled per chip from one source
and tested in the simulator on at least two generations. The first one is a GEMM (`agx_gemm_f32`, `agx_gemm_f16`)
built on `agx.mma_tile`.
