# 'air' Dialect: Apple GPUs through AIR

[TOC]

The `air` dialect targets Apple GPUs through AIR, the LLVM IR form of Apple's Metal shading language that the AIR
target (`llvm/lib/Target/AIR`) writes. Kernels are written in the standard dialects (`gpu`, `scf`, `arith`, `math`,
`memref`, `vector`) plus the few `air` operations for what those cannot say. AIR abstracts over the chip: a kernel
names no chip, and the chip-specific code generation lives in the LLVM backends.

## Operations

### SIMD-group matrices

An 8x8 SIMD-group matrix is a `vector<64xT>` (T is f16, bf16 or f32) shared by the 32 threads of a SIMD-group. Its
elements are not individually addressable: only the operations below, element-wise arithmetic (`arith`, `math`),
`arith.extf` / `arith.truncf` and `arith.select` / `scf` values of matrices are allowed. Every thread of the
SIMD-group executes them with the same operands.

- `air.simdgroup_matrix_load %m[%i, %j] [transpose]`: the matrix `m[i + r, j + c]` (`m[i + c, j + r]` with
  `transpose`) of a 2-D memref with any strides.
- `air.simdgroup_matrix_store %x, %m[%i, %j] [transpose]`: the reverse.
- `air.simdgroup_matrix_multiply_accumulate %a, %b, %c`: `A * B + C`. A and B have one element type and C and the
  result another: the same, or f16 or bf16 A and B with f32 C (`: vector<64xf16>, vector<64xf32>`).
- `air.simdgroup_matrix_fill %v`: every element `v`.

## Target and pipeline

`#air.target<O, triple, chip, features>` on a `gpu.module` (`air-attach-target` attaches it):

- without a chip, `gpu-module-to-binary` produces a `.metallib` for Apple's Metal driver;
- with an AGX chip (an LLVM AGX processor such as `agx-g14g`), an AGX object: the module's AIR goes through the AIR
  target's portable passes (`air-portable`), then the AGX backend, whose `agx-lower-air` pass turns AIR into AGX IR
  at the start of its pipeline (`llvm/docs/AGXUsage.rst`, "AIR input"). `features` are AGX target features, such as
  `+agx-strict` or `+agx-checked`.

The conversion is `convert-gpu-to-air`, in each `gpu.module`. A memref without a memory space is device memory,
except a `memref.alloca`, which is the thread's (AIR has no generic address space). AIR has no double precision:
run `arith-emulate-f64` first, which computes f64 on i64 with a software library.

```mlir
mlir-opt kernel.mlir --pass-pipeline='builtin.module(air-attach-target{chip=agx-g14g},
    gpu.module(arith-emulate-f64,convert-gpu-to-air),gpu-module-to-binary)'
```

`mlir/utils/agx-kernels/agx-build-fatbin.py` compiles a library kernel source for several AGX chips and packs the
objects into one fat AGX object (`ZAGXFAT1`), which ZLUDA's `cuModuleLoadData` picks the device's entry from.

## Where the `agx` dialect went

The `agx` and `agxir` dialects (`convert-gpu-to-agxir`, `convert-agx-to-agxir`, `#agxir.target`) are gone. Their
chip-specific choices are the AGX backend's now, so that one AIR kernel serves every chip:

| `agx` operation | Now |
| --- | --- |
| `agx.mma`, `agx.matrix_load` / `store`, `agx.mma_tile` | `air` SIMD-group matrices (a tile is a loop of them) |
| `agx.shuffle`, `agx.reduce`, `agx.ballot` | `gpu.shuffle`, `gpu.subgroup_reduce`, `gpu.ballot` |
| `agx.threadgroup_barrier`, `agx.memory_barrier` | `gpu.barrier` (with `memfence` for the memory it orders) |
| `agx.block_barrier` | `gpu.barrier memfence []`; the backend is to use the 4-SIMD-group barrier when the threadgroup is that small |
| `agx.fma_lane_shift` | `gpu.shuffle up` in segments, then `math.fma`; the backend is to match the G14 lane-shift FMA |
| `agx.uniform_region` | inline code; the backend is to hoist uniform work into the preamble |
| `agx.core_id`, `agx.simd_slot` | none (machine specific); PTX's `%smid` keeps `llvm.agx.core.id` |
| `agx.set_fpmode`, `agx.get_fpmode` | none in MLIR; ZLUDA's PTX path keeps the `llvm.agx` intrinsics |
| `#agx.launch_hints` | to be kernel attributes the AGX backend reads and AIR ignores |

The capability table that decided the native lowerings (`AGXChipInfo`) moves into the AGX backend's subtarget.
