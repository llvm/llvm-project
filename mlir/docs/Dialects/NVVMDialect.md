# 'nvvm' Dialect

The NVVM dialect is MLIR's LLVM-IR-based, NVIDIA-specific backend dialect. It
models NVVM intrinsics and public ISA functionality and introduces NVIDIA
extensions to the MLIR/LLVM type system and address spaces (e.g., global,
shared, and cluster memory), enabling faithful lowering of GPU kernels to the
NVPTX toolchain. While a NVVM op usually maps to a single LLVM IR intrinsic,
the NVVM dialect uses type polymorphism and other attributes so that a single
NVVM op can map to different LLVM intrinsics.

[TOC]

## Scope and Capabilities

The dialect covers core GPU features such as thread/block builtins, barriers
and atomics, warp-level collectives (e.g., shuffle/vote), matrix/tensor core
operations (e.g., `mma.sync`, `wgmma`), tensor memory accelerator (TMA)
operations, asynchronous copies (`cp.async`, bulk/tensor variants) with memory
barriers, cache and prefetch controls, and NVVM-specific attributes and enums
(e.g., FP rounding modes, memory scopes, and MMA types/layouts).

## Placement in the Lowering Pipeline

NVVM sits below target-agnostic dialects like `gpu` and NVIDIA's `nvgpu`.
Typical pipelines convert `gpu`/`nvgpu` ops into NVVM using
`-convert-gpu-to-nvvm` and `-convert-nvgpu-to-nvvm`, then translate into LLVM
for final code generation via NVPTX backend.

## Target Configuration and Serialization

NVVM provides a `#nvvm.target` attribute to describe the GPU target (SM,
features, and flags). In conjunction with `gpu` serialization (e.g.,
`gpu-module-to-binary`), this enables producing architecture-specific GPU
binaries (such as CUBIN) from nested GPU modules.

## Inline PTX

When an intrinsic is unavailable or a performance-critical sequence must be
expressed directly, NVVM provides an `nvvm.inline_ptx` op to embed PTX inline
as a last-resort escape hatch, with explicit operands and results.

## Memory Spaces

The NVVM dialect introduces the following memory spaces, each with distinct
scopes and lifetimes:

| Memory Space      | Address Space | Scope                |
|-------------------|---------------|----------------------|
| `generic`         | 0             | All threads          |
| `global`          | 1             | All threads (device) |
| `shared`          | 3             | Thread block (CTA)   |
| `constant`        | 4             | All threads          |
| `local`           | 5             | Single thread        |
| `tensor`          | 6             | Thread block (CTA)   |
| `shared_cluster`  | 7             | Thread block cluster |

### Memory Space Details

- **generic**: Can point to any memory space; requires runtime resolution of
  actual address space. Use when pointer origin is unknown at compile time.
  Performance varies based on the underlying memory space. A pointer to this
  memory space is represented by `LLVM_PointerGeneric` in the NVVM Ops.
- **global**: Accessible by all threads across all blocks; persists across
  kernel launches. Highest latency but largest capacity (device memory). Best
  for large data and inter-kernel communication. A pointer to this memory space
  is represented by `LLVM_PointerGlobal` in the NVVM Ops.
- **shared**: Shared within a thread block (CTA); very fast on-chip memory for
  cooperation between threads in the same block. Limited capacity. Ideal for
  block-level collaboration, caching, and reducing global memory traffic.
  This memory is usually referred as `shared_cta` in the NVVMOps and as
  `shared::cta` in the PTX ISA. A pointer to this memory space is represented
  by the `LLVM_PointerShared` type in the NVVM Ops.
- **constant**: Read-only memory cached per SM. Size typically limited to 64KB.
  Best for read-only data and uniform values accessed by all threads. A pointer
  to this memory space is represented by `LLVM_PointerConst` type in NVVM Ops.
- **local**: Private to each thread. Use for per-thread private data and
  automatic variables that don't fit in registers. A pointer to this memory is
  represented by `LLVM_PointerLocal` type in NVVM Ops.
- **tensor**: Special memory space for tensor core operations. Used by
  `tcgen05` instructions on SM 100+ for tensor input/output operations.
  A pointer to this memory space is represented by the `LLVM_PointerTensor`
  type in the NVVM Ops.
- **shared_cluster**: Distributed shared memory across thread blocks within a
  cluster (SM 90+). Enables collaboration beyond single-block scope with fast
  access across cluster threads. This memory is usually referred as
  `shared_cluster` in the NVVMOps and as `shared::cluster` in the PTX ISA.
  A pointer to this memory space is represented by the `LLVM_PointerSharedCluster`
  type in the NVVM Ops.

## MBarrier objects

An ``mbarrier`` is a barrier created in shared memory that supports
synchronizing any subset of threads within a CTA. An *mbarrier object*
is an opaque object in shared memory with `.b64` type and an alignment of
8-bytes. Unlike ``nvvm.barrier`` Op which can access only a limited number
of barriers per CTA, the *mbarrier objects* are user-defined and are only
limited by the total shared memory size available. The list of operations
supported on an *mbarrier object* is exposed through the ``nvvm.mbarrier.*``
family of NVVM Ops.

## Non-Goals

NVVM is not a place for convenience or "wrapper" ops. It is not intended to
introduce high-level ops that expand into multiple unrelated NVVM intrinsics or
that lower to no intrinsic at all. Such abstractions belong in higher-level
dialects (e.g., `nvgpu`, `gpu`, or project-specific dialects). The design
intent is a thin, predictable, low-level surface with near-mechanical lowering
to NVVM/LLVM IR.


## Operations

All operations in the NVIDIA's instruction set have a custom form in MLIR. The mnemonic
of an operation is that used in LLVM IR prefixed with "`nvvm.`".

[include "Dialects/NVVMOps.md"]

