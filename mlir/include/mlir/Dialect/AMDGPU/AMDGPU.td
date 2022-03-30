//===-- AMDGPU.td - AMDGPU dialect definitions *- tablegen -*------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AMDGPU
#define AMDGPU

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

def AMDGPU_Dialect : Dialect {
  let name = "amdgpu";
  let cppNamespace = "::mlir::amdgpu";
  let description = [{
    The `AMDGPU` dialect provides wrappers around AMD-specific functionality
    and LLVM intrinsics. These wrappers should be used in conjunction with
    more generic dialects, such as `gpu` and `vector`, when generating LLVM IR
    that will eventually be executed on AMD hardware.
  }];
}

//===----------------------------------------------------------------------===//
// AMDGPU Op definitions
//===----------------------------------------------------------------------===//

class AMDGPU_Op<string mnemonic, list<Trait> traits = []> :
  Op<AMDGPU_Dialect, mnemonic, traits> {}

/// Raw buffer load
def AMDGPU_RawBufferLoadOp :
    AMDGPU_Op<"raw_buffer_load", [AllElementTypesMatch<["value", "memref"]>,
      AttrSizedOperandSegments]>,
    Arguments<(ins Arg<AnyMemRef, "buffer to load from", [MemRead]>:$memref,
                   BoolAttr:$targetIsRDNA,
                   Variadic<I32>:$indices,
                   DefaultValuedAttr<BoolAttr, "true">:$boundsCheck,
                   OptionalAttr<I32Attr>:$indexOffset,
                   Optional<I32>:$sgprOffset)>,
    Results<(outs AnyTypeOf<[BF16, F16, F32, I32, I8,
                              VectorOfLengthAndType<[2, 4], [F32, I32]>,
                              VectorOfLengthAndType<[2, 4, 8], [F16, BF16]>,
                              VectorOfLengthAndType<[2, 4, 8, 16], [I8]>]>:$value)> {

  let summary = "Raw Buffer load, exposing GCN features";
  let description = [{
    The `amdgpu.raw_buffer_load` op is a wrapper around the buffer load intrinsics
    available on AMD GPUs, including extensions in newer GPUs.

    The index into the buffer is computed as for `memref.load` with the additon
    of `indexOffset` and `sgprOffset` (which is added after bounds checks and
    includes any offset present on the memref type if it's non-zero).

    All indices and offsets are in units of the memref's data type and are
    converted to bytes during lowering.

    When a load is out of bounds, the instruction returns zero.
    Vector instructions bounds check each component's address.

    The memref struct is converted into a buffer resource (a V#) and the arguments
    are translated to intrinsic arguments as follows:
    - The base address of the buffer is the base address of the memref
    - The stride is 0 to enable raw mode
    - The number of records is the size of the memref, in bytes
      In the case of dynamically-shaped memrefs, this is computed at runtime
      as max_d (size(d) * stride(d)) * sizeof(elementType(memref))
    - The offset enable bit is 1, the index enable bit is 0.
    - The thread ID addition bit is off
    - If `boundsCheck` is false and the target is RDNA, OOB_SELECT is set to 2
      to disable bounds checks, otherwise it is 0
    - The cache coherency bits are off
    - `targetIsRDNA` controls the setting of some reserved values that differ
       between RDNA and CDNA cores
  }];
  let assemblyFormat = [{
    attr-dict $memref `[` $indices `]`
      (`sgprOffset` $sgprOffset^)? `:`
      type($memref) `,` type($indices) `->` type($value)
  }];
  let hasVerifier = 1;
}

/// Raw buffer store
def AMDGPU_RawBufferStoreOp :
    AMDGPU_Op<"raw_buffer_store", [AllElementTypesMatch<["value", "memref"]>,
      AttrSizedOperandSegments]>,
    Arguments<(ins AnyTypeOf<[BF16, F16, F32, I32, I8,
                              VectorOfLengthAndType<[2, 4], [F32, I32]>,
                              VectorOfLengthAndType<[2, 4, 8], [F16, BF16]>,
                              VectorOfLengthAndType<[2, 4, 8, 16], [I8]>]>:$value,
                   Arg<AnyMemRef, "buffer to store to", [MemWrite]>:$memref,
                   BoolAttr:$targetIsRDNA,
                   Variadic<I32>:$indices,
                   DefaultValuedAttr<BoolAttr, "true">:$boundsCheck,
                   OptionalAttr<I32Attr>:$indexOffset,
                   Optional<I32>:$sgprOffset)> {

  let summary = "Raw Buffer Store, exposing GCN features";
  let description = [{
    The `amdgpu.raw_buffer_store` op is a wrapper around the buffer store
    intrinsics available on AMD GPUs, including extensions in newer GPUs.

    The store index is computed as in `memref.store` with the addition of
    `indexOffset` (which is included for uniformity with atomics and may be useful
    when writing vectorized code) and `sgprOffset` (which is added after bounds
    checks and implicitly includes the offset of the memref type if non-zero).
    All index components are in terms of the elements of the memref, not bytes,
    and are scaled up appropriately.

    Out of bounds stores are ignored in hardware, including the out of bounds
    components of vector writes.

    See `amdgpu.raw_buffer_load` for a description of how the underlying
    instruction is constructed.
  }];
  let assemblyFormat = [{
    attr-dict $value `->` $memref `[` $indices `]`
      (`sgprOffset` $sgprOffset^)? `:`
      type($value) `->` type($memref) `,` type($indices)
  }];
  let hasVerifier = 1;
}

// Raw buffer atomic floating point add
def AMDGPU_RawBufferAtomicFaddOp :
    AMDGPU_Op<"raw_buffer_atomic_fadd", [AllElementTypesMatch<["value", "memref"]>,
      AttrSizedOperandSegments]>,
    Arguments<(ins F32:$value,
                   Arg<AnyMemRef, "buffer to operate on", [MemRead, MemWrite]>:$memref,
                   BoolAttr:$targetIsRDNA,
                   Variadic<I32>:$indices,
                   DefaultValuedAttr<BoolAttr, "true">:$boundsCheck,
                   OptionalAttr<I32Attr>:$indexOffset,
                   Optional<I32>:$sgprOffset)> {

  let summary = "Raw Buffer Floating-point Atomic Add (MI-* only)";
  let description = [{
    The `amdgpu.raw_buffer_atomic_fadd` op is a wrapper around the
    buffer-based atomic floating point addition available on the MI-* series
    of AMD GPUs.

    The index into the buffer is computed as for `memref.store` with the addition
    of `indexOffset` (which is used to aid in emitting vectorized code) and,
    if present `sgprOffset` (which is added after bounds checks and includes
    any non-zero offset on the memref type).

    All indexing components are given in terms of the memref's element size, not
    the byte lengths required by the intrinsic.

    Out of bounds atomic operations are ignored in hardware.

    See `amdgpu.raw_buffer_load` for a description of how the underlying
    instruction is constructed.
  }];
  let assemblyFormat = [{
    attr-dict $value `->` $memref `[` $indices `]`
      (`sgprOffset` $sgprOffset^)? `:`
      type($value) `->` type($memref) `,` type($indices)
  }];
  let hasVerifier = 1;
}

#endif // AMDGPU
