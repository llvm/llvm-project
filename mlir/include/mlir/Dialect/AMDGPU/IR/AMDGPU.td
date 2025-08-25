//===-- AMDGPU.td - AMDGPU dialect definitions *- tablegen -*------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AMDGPU
#define AMDGPU

include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/Properties.td"
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


  let dependentDialects = [
    "ROCDL::ROCDLDialect",
    "arith::ArithDialect",
    "gpu::GPUDialect"
  ];
  let useDefaultAttributePrinterParser = 1;
}

def AnyIntegerOrFloat : AnyTypeOf<[AnySignlessInteger, AnyFloat], "Integer or Float">;

def AnyIntegerOrFloatOr1DVector :
  AnyTypeOf<[AnyIntegerOrFloat, FixedVectorOfRankAndType<[1], [AnyIntegerOrFloat]>]>;

//===----------------------------------------------------------------------===//
// AMDGPU general attribute definitions
//===----------------------------------------------------------------------===//

def AMDGPU_AddressSpace : I32EnumAttr<"AddressSpace",
    "AMDGPU-specific address spaces",
    [
      I32EnumAttrCase<"FatRawBuffer",        0, "fat_raw_buffer">,
      I32EnumAttrCase<"BufferRsrc",          1, "buffer_rsrc">,
      I32EnumAttrCase<"FatStructuredBuffer", 2, "fat_structured_buffer">,
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::amdgpu";
}

def AMDGPU_AddressSpaceAttr : EnumAttr<AMDGPU_Dialect, AMDGPU_AddressSpace,
    "address_space"> {
  let description = [{
    AMDGPU-specific memory spaces that may not have exact analogues on other
    GPU targets or backends.

    - `fat_raw_buffer` is the memory space used when a memref is stored as
    as a "buffer fat pointer" - that is, a buffer resource (that is set up to
    use raw byte-level indexing) along with its offset. The AMDGPU backend
    implements `ptr addrspace(7)` to represent these fat pointers so that
    buffer resources (which allow advanced features like bounds checking or
    cache swizzling) can be used like ordinary LLVM pointers or memrefs.
    See also the `fat_raw_buffer_cast` operation
    - `buffer_rsrc` is the memory space for `ptr addrspace(8)`, representing a
    buffer resource. It should not be used for memrefs, since it does not support
    indexing
    - `fat_structured_buffer` represents `ptr addrspace(9)`, a buffer resource
    that carries both an index and offset field, which are used for complex
    structured indexing that is primarily seen in graphics applications. This
    is also incompatible with the simple indexing model supported by memref.
  }];
  let assemblyFormat = "`<` $value `>`";
}

//===----------------------------------------------------------------------===//
// AMDGPU Op definitions
//===----------------------------------------------------------------------===//

class AMDGPU_Op<string mnemonic, list<Trait> traits = []> :
  Op<AMDGPU_Dialect, mnemonic, traits> {}

def AMDGPU_ExtPackedFp8Op :
    AMDGPU_Op<"ext_packed_fp8", [Pure]>,
    Arguments<(ins AnyTypeOf<[F8E5M2FNUZ, F8E4M3FNUZ, F8E5M2, F8E4M3FN,
        VectorOfLengthAndType<[1, 2, 3, 4], [F8E5M2FNUZ, F8E4M3FNUZ, F8E5M2, F8E4M3FN]>]>:$source,
      ConfinedAttr<I32Attr, [IntNonNegative, IntMaxValue<3>]>:$index)>,
    Results<(outs AnyTypeOf<[F32, FixedVectorOfLengthAndType<[2], [F32]>]>:$res)> {
  let summary = "Extend a fp8 value to a float or a vector of packed fp8 values to two floats";

  let description = [{
    Extend one or two 8-bit floats in `source[index]` to a 32-bit float or
    two floats and return them.

    This rather unusual signature arises from the fact that AMD GPUs cannot
    easily work with sub 32-bit quantities, so the compiler intrinsics for
    extending 8-bit floats (which are, currently, the only way to work with
    this operation) take packed vectors of 4 such floats.

    If the passed-in vector has fewer than four elements, or the input is scalar,
    the remaining values in the <4 x i8> will be filled with
    undefined values as needed.
  }];
  let assemblyFormat = [{
    attr-dict $source `[` $index `]` `:` type($source) `to` type($res)
  }];
}

def AMDGPU_ScaledExtPackedOp
    : AMDGPU_Op<"scaled_ext_packed", [Pure]>,
      Arguments<(
          ins AnyTypeOf<[VectorOfLengthAndType<[1, 2, 3, 4], [F8E5M2, F8E4M3FN]>,
                         VectorOfLengthAndType<[1, 2, 3, 4, 5, 6, 7, 8],
                                               [F4E2M1FN]>]>:$source,
          F32:$scale,
          ConfinedAttr<I32Attr, [IntNonNegative, IntMaxValue<7>]>:$index)>,
      Results<(
          outs AnyTypeOf<[FixedVectorOfLengthAndType<[2], [F32]>,
                          FixedVectorOfLengthAndType<[2], [F16]>,
                          FixedVectorOfLengthAndType<[2], [BF16]>]>:$res)> {
  let summary = "Extend a vector of packed floating point values";

  let description = [{
    Extend and scale two packed floats in `source[index]` to two floats and
    return them.

    This rather unusual signature arises from the fact that AMD GPUs cannot
    easily work with sub 32-bit quantities, so the compiler intrinsics for
    extending 8-bit floats (which are, currently, the only way to work with
    this operation) take packed vectors of 2 such floats.

    If the passed-in vector has fewer than two elements, or the input is scalar,
    the remaining values in the <2 x i8> will be filled with
    undefined values as needed.
  }];
  let assemblyFormat = [{
    attr-dict $source `[` $index `]` `,` $scale `:` type($source) `to` type($res)
  }];
}

def AMDGPU_PackedTrunc2xFp8Op :
    AMDGPU_Op<"packed_trunc_2xfp8", [Pure, AttrSizedOperandSegments]>,
    Arguments<(ins F32:$sourceA,
      Optional<F32>:$sourceB,
      ConfinedAttr<I32Attr, [IntNonNegative, IntMaxValue<1>]>:$wordIndex,
      Optional<FixedVectorOfLengthAndType<[4], [F8E4M3FNUZ, F8E5M2FNUZ, F8E4M3FN, F8E5M2]>>:$existing)>,
    Results<(outs FixedVectorOfLengthAndType<[4], [F8E4M3FNUZ, F8E5M2FNUZ, F8E4M3FN, F8E5M2]>:$res)> {
  let summary = "Round two floats into a packed vector of 8-bit floats";
  let description = [{
    Round the inputs `sourceA` and `sourceB` (which is undefined if not
    specified) into the low or high word (bottom two or top two) elements
    of the returned vector, keeping the other two elements of `existing`
    unchanged if present (or undefined if it was not passed in).

    The reason for this odd signature is that AMD GPUs cannot easily work with
    sub-registers, and so the conversion intrinsics (which are currently the
    only way to work with 8-bit float types) take packed vectors of 4 8-bit
    values.
  }];
  let assemblyFormat = [{
    attr-dict $sourceA `,` ($sourceB^):(`undef`)?
    `into` ($existing^):(`undef`)? `[` `word` $wordIndex `]`
    `:` type($sourceA) `to` type($res) (`into` type($existing)^)?
  }];
  let hasVerifier = 1;
}

def AMDGPU_PackedScaledTruncOp
    : AMDGPU_Op<"packed_scaled_trunc", [Pure]>,
      Arguments<(ins VectorOfLengthAndType<[1, 2], [F32, F16, BF16]>:$source,
          F32:$scale,
          ConfinedAttr<I32Attr, [IntNonNegative, IntMaxValue<7>]>:$index,
          Optional<AnyTypeOf<
              [FixedVectorOfLengthAndType<[4], [F8E5M2, F8E4M3FN]>,
               FixedVectorOfLengthAndType<[8], [F4E2M1FN]>]>>:$existing)>,
      Results<(
          outs AnyTypeOf<[FixedVectorOfLengthAndType<[4], [F8E5M2, F8E4M3FN]>,
                          FixedVectorOfLengthAndType<[8], [F4E2M1FN]>]>:$res)> {
  let summary = "Round two floats into a packed vector of floats";
  let description = [{
    Scale and round the inputs `source` (which is undefined if not
    specified) into the low or high word (bottom two or top two) elements
    of the returned vector, keeping the other two elements of `existing`
    unchanged if present (or undefined if it was not passed in).

    The reason for this odd signature is that AMD GPUs cannot easily work with
    sub-registers, and so the conversion intrinsics take 32-bit wide
    packed vectors of float values.
  }];
  let assemblyFormat = [{
    attr-dict $source `into` ($existing^):(`undef`)? `[` $index `]`
    `,` $scale
    `:` type($source) `to` type($res) (`into` type($existing)^)?
  }];
  let hasVerifier = 1;
}

def AMDGPU_PackedStochRoundFp8Op :
    AMDGPU_Op<"packed_stoch_round_fp8", [Pure]>,
    Arguments<(ins F32:$source,
      I32:$stochiasticParam,
      ConfinedAttr<I32Attr, [IntNonNegative, IntMaxValue<3>]>:$storeIndex,
      Optional<FixedVectorOfLengthAndType<[4], [F8E4M3FNUZ, F8E5M2FNUZ, F8E4M3FN, F8E5M2]>>:$existing)>,
    Results<(outs FixedVectorOfLengthAndType<[4], [F8E4M3FNUZ, F8E5M2FNUZ, F8E4M3FN, F8E5M2]>:$res)> {
  let summary = "Round float stochiastically into a packed vector of 8-bit floats";
  let description = [{
    Round the input `source`, adding in `stochiasticParam`, and place it into
    the `storeIndex`th element of `res`.

    If `existing` is passed in, elements of `res` other than the one at `storeIndex`
    are copied from `existing`.

    The reason for this odd signature is that AMD GPUs cannot easily work with
    sub-registers, and so the conversion intrinsics (which are currently the
    only way to work with 8-bit float types) take packed vectors of 4 8-bit
    values.
  }];
  let assemblyFormat = [{
    attr-dict $source `+` $stochiasticParam
    `into` ($existing^):(`undef`)? `[` $storeIndex `]`
    `:` type($source) `to` type($res) (`into` type($existing)^)?
  }];
  let hasVerifier = 1;
}

def AMDGPU_FatRawBufferCastOp :
    AMDGPU_Op<"fat_raw_buffer_cast",
      [Pure,
       DeclareOpInterfaceMethods<InferTypeOpInterface>,
       ViewLikeOpInterface, AttrSizedOperandSegments]>,
    Arguments<(ins AnyMemRef:$source,
      Optional<I32>:$validBytes,
      Optional<I<14>>:$cacheSwizzleStride,
      DefaultValuedAttr<BoolAttr, "true">:$boundsCheck,
      UnitAttr:$resetOffset)>,
    Results<(outs AnyMemRef:$result)> {
  // TODO: Set `resetOffset` and `boundsCheck` to use `Property` once
  //       we implemented pythonic binding for `Property`.
  let summary = "Create a raw buffer fat pointer that matches `memref`";
  let description = [{
    Wraps the memory pointed to by `source` as a raw buffer fat pointer, or,
    in LLVM terms, a `ptr addrspace(7)`, returning a memref that has the same
    sizes and layout but the `#amdgpu.address_space<fat_raw_buffer>`
    address space.

    This memref can be used with standard memref operations like `memref.load`,
    `memref.store`, and `memref.atomicrmw`, which will be lowered to the relevant
    buffer intrinsics. (`vector.masked_load/store` will work once there's backend
    support for lowering them, and then this document will be updated)

    If `validBytes` is given, it is the number of bytes that will be valid as
    an offset to `out`. If it is not provided, this will be inferred from
    the size of the memref during lowering. This size is
    max_{d = 0 upto rank(source)} (sizes[d] * strides[d]) * sizeof(element type).

    The flags of the buffer descriptor will be set up to enable raw usage -
    for example, stride = 0, add_tid = 0, and so on. The `boundsCheck`
    property determines if bounds checking is enabled or not (on architectures
    where this can be controlled - that is, on RDNA chips).

    If `cacheSwizzleStride` is provided, L1 cache swizzling will be enabled
    on architectures that support it. This swizzling, unlike the main swizzling
    mode (whose usage makes a buffer non-raw) does not affect index calculation,
    but does affect cache behavior. Mixing access between cache-swizzled raw
    buffers and other forms of memory access, like ordinary pointer loads or
    unswizzled buffer pointers can cause incorrect behavior and must be avoided.

    This operation preserves the sizes, strides, and offset of the input
    memref - they'll be added in by `memref.load` later. However, if
    `resetOffset` is set, that offset will be added to the base pointer.
    If the value of the memref's offset is not uniform (independent of the lane/thread ID),
    this will lead to substantially decreased performance due to the need for
    a waterfall loop on the base address of the buffer resource.
  }];

  let extraClassDeclaration = [{
    Value getViewSource() { return getSource(); }
  }];

  let assemblyFormat = [{
    $source oilist (`validBytes` `(` $validBytes `)`
      | `cacheSwizzleStride` `(` $cacheSwizzleStride `)`
      | `boundsCheck` `(` $boundsCheck `)`
      | `resetOffset` $resetOffset )
    attr-dict `:` type($source) `to` type($result)
  }];

  let hasVerifier = 1;
}

/// Raw buffer load
def AMDGPU_RawBufferLoadOp :
    AMDGPU_Op<"raw_buffer_load", [AllElementTypesMatch<["value", "memref"]>,
      AttrSizedOperandSegments]>,
    Arguments<(ins Arg<AnyMemRef, "buffer to load from", [MemRead]>:$memref,
                   Variadic<I32>:$indices,
                   DefaultValuedAttr<BoolAttr, "true">:$boundsCheck,
                   OptionalAttr<I32Attr>:$indexOffset,
                   Optional<I32>:$sgprOffset)>,
    Results<(outs AnyType:$value)> {

  let summary = "Raw Buffer load, exposing GCN features";
  let description = [{
    The `amdgpu.raw_buffer_load` op is a wrapper around the buffer load intrinsics
    available on AMD GPUs, including extensions in newer GPUs.

    The index into the buffer is computed as for `memref.load` with the additon
    of `indexOffset` and `sgprOffset` (which **may or may not** be considered
    in bounds checks and includes any offset present on the memref type if it's
    non-zero).

    All indices and offsets are in units of the memref's data type and are
    converted to bytes during lowering.

    When a load is out of bounds, the instruction returns zero.
    Partially-out of bounds have chipset-dependent behavior: whether reading
    2 elements starting at index 7 of a `memref<8xf32>` returns the last element
    in the first vector component depends on the architecture.

    The memref struct is converted into a buffer resource (a V#) and the arguments
    are translated to intrinsic arguments as follows:
    - The base address of the buffer is the base address of the memref
    - The stride is 0 to enable raw mode
    - The number of records is the size of the memref, in bytes
      In the case of dynamically-shaped memrefs, this is computed at runtime
      as max_d (size(d) * stride(d)) * sizeof(elementType(memref))
    - The offset enable bit is 1, the index enable bit is 0.
    - The thread ID addition bit is off
    - If `boundsCheck` is false and the target chipset is RDNA, OOB_SELECT is set
      to 2 to disable bounds checks, otherwise it is 3
    - The cache coherency bits are off
  }];
  let assemblyFormat = [{
    attr-dict $memref `[` $indices `]`
      (`sgprOffset` $sgprOffset^)? `:`
      type($memref) (`,` type($indices)^)? `->` type($value)
  }];
  let hasCanonicalizer = 1;
  let hasVerifier = 1;
}

/// Raw buffer store
def AMDGPU_RawBufferStoreOp :
    AMDGPU_Op<"raw_buffer_store", [AllElementTypesMatch<["value", "memref"]>,
      AttrSizedOperandSegments]>,
    Arguments<(ins AnyType:$value,
                   Arg<AnyMemRef, "buffer to store to", [MemWrite]>:$memref,
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

    Out of bounds stores are ignored in hardware.
    Wthether a vector write that includes some in-bounds and soeme out-of-bounds
    components is partically completed is chipset-dependent.

    See `amdgpu.raw_buffer_load` for a description of how the underlying
    instruction is constructed.
  }];
  let assemblyFormat = [{
    attr-dict $value `->` $memref `[` $indices `]`
      (`sgprOffset` $sgprOffset^)? `:`
      type($value) `->` type($memref) (`,` type($indices)^)?
  }];
  let hasCanonicalizer = 1;
  let hasVerifier = 1;
}

// Raw buffer atomic compare-and-swap
def AMDGPU_RawBufferAtomicCmpswapOp :
    AMDGPU_Op<"raw_buffer_atomic_cmpswap", [
      AttrSizedOperandSegments,
      AllTypesMatch<["src", "cmp", "value"]>,
      AllElementTypesMatch<["value", "memref"]>]>,
    Arguments<(ins AnyType:$src,
                   AnyType:$cmp,
                   Arg<AnyMemRef, "buffer to operate on", [MemRead, MemWrite]>:$memref,
                   Variadic<I32>:$indices,
                   DefaultValuedAttr<BoolAttr, "true">:$boundsCheck,
                   OptionalAttr<I32Attr>:$indexOffset,
                   Optional<I32>:$sgprOffset)>,
    Results<(outs AnyType:$value)> {

  let summary = "Raw Buffer Atomic compare-and-swap";
  let description = [{
    The `amdgpu.raw_buffer_atomic_cmpswap` op is a wrapper around the
    buffer-based atomic compare-and-swap min available on AMD GPUs.

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
    attr-dict $src `,` $cmp `->` $memref `[` $indices `]`
      (`sgprOffset` $sgprOffset^)? `:`
      type($value) `->` type($memref) `,` type($indices)
  }];
  let hasCanonicalizer = 1;
  let hasVerifier = 1;
}

// Raw buffer atomic floating point add
def AMDGPU_RawBufferAtomicFaddOp :
    AMDGPU_Op<"raw_buffer_atomic_fadd", [AllElementTypesMatch<["value", "memref"]>,
      AttrSizedOperandSegments]>,
    Arguments<(ins AnyTypeOf<[F32, VectorOfLengthAndType<[2], [F16, BF16]>]>:$value,
                   Arg<AnyMemRef, "buffer to operate on", [MemRead, MemWrite]>:$memref,
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
  let hasCanonicalizer = 1;
  let hasVerifier = 1;
}

// Raw buffer atomic floating point max
def AMDGPU_RawBufferAtomicFmaxOp :
    AMDGPU_Op<"raw_buffer_atomic_fmax", [AllElementTypesMatch<["value", "memref"]>,
      AttrSizedOperandSegments]>,
    Arguments<(ins AnyTypeOf<[F32, F64]>:$value,
                   Arg<AnyMemRef, "buffer to operate on", [MemRead, MemWrite]>:$memref,
                   Variadic<I32>:$indices,
                   DefaultValuedAttr<BoolAttr, "true">:$boundsCheck,
                   OptionalAttr<I32Attr>:$indexOffset,
                   Optional<I32>:$sgprOffset)> {

  let summary = "Raw Buffer Floating-point Atomic Max (non-GFX9)";
  let description = [{
    The `amdgpu.raw_buffer_atomic_fmax` op is a wrapper around the
    buffer-based atomic floating point max available on AMD GPUs (except GFX9).

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
  let hasCanonicalizer = 1;
  let hasVerifier = 1;
}

// Raw buffer atomic signed integer max
def AMDGPU_RawBufferAtomicSmaxOp :
    AMDGPU_Op<"raw_buffer_atomic_smax", [
      AttrSizedOperandSegments]>,
    Arguments<(ins I32:$value,
                   Arg<AnyMemRef, "buffer to operate on", [MemRead, MemWrite]>:$memref,
                   Variadic<I32>:$indices,
                   DefaultValuedAttr<BoolAttr, "true">:$boundsCheck,
                   OptionalAttr<I32Attr>:$indexOffset,
                   Optional<I32>:$sgprOffset)> {

  let summary = "Raw Buffer Signed Integer Atomic Max";
  let description = [{
    The `amdgpu.raw_buffer_atomic_smax` op is a wrapper around the
    buffer-based atomic signed integer max available on AMD GPUs.

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
  let hasCanonicalizer = 1;
  let hasVerifier = 1;
}

// Raw buffer atomic unsigned integer min
def AMDGPU_RawBufferAtomicUminOp :
    AMDGPU_Op<"raw_buffer_atomic_umin", [
      AttrSizedOperandSegments]>,
    Arguments<(ins I32:$value,
                   Arg<AnyMemRef, "buffer to operate on", [MemRead, MemWrite]>:$memref,
                   Variadic<I32>:$indices,
                   DefaultValuedAttr<BoolAttr, "true">:$boundsCheck,
                   OptionalAttr<I32Attr>:$indexOffset,
                   Optional<I32>:$sgprOffset)> {

  let summary = "Raw Buffer Unsigned Integer Atomic Min";
  let description = [{
    The `amdgpu.raw_buffer_atomic_umin` op is a wrapper around the
    buffer-based atomic signed integer min available on AMD GPUs.

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
  let hasCanonicalizer = 1;
  let hasVerifier = 1;
}

def AMDGPU_DPPPerm : I32EnumAttr<"DPPPerm",
    "The possible permutations for a DPP operation",
    [
      I32EnumAttrCase<"quad_perm",  0>,
      I32EnumAttrCase<"row_shl",    1>,
      I32EnumAttrCase<"row_shr",    2>,
      I32EnumAttrCase<"row_ror",    3>,
      I32EnumAttrCase<"wave_shl",   4>,
      I32EnumAttrCase<"wave_shr",   5>,
      I32EnumAttrCase<"wave_ror",   6>,
      I32EnumAttrCase<"wave_rol",   7>,
      I32EnumAttrCase<"row_mirror", 8>,
      I32EnumAttrCase<"row_half_mirror", 9>,
      I32EnumAttrCase<"row_bcast_15", 10>,
      I32EnumAttrCase<"row_bcast_31", 11>
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::amdgpu";
}

def AMDGPU_DPPPermAttr : EnumAttr<AMDGPU_Dialect, AMDGPU_DPPPerm,
  "dpp_perm">;

def AMDGPU_DPPOp : AMDGPU_Op<"dpp",
    [Pure, SameTypeOperands, AllTypesMatch<["result", "old", "src"]>]>,
  Arguments<(ins AnyType:$old,
                 AnyType:$src,
                 AMDGPU_DPPPermAttr:$kind,
                 OptionalAttr<AnyAttrOf<[I32Attr, ArrayAttr, UnitAttr]>>:$permArgument,
                 DefaultValuedAttr<I32Attr, "0xf">:$row_mask,
                 DefaultValuedAttr<I32Attr, "0xf">:$bank_mask,
                 DefaultValuedAttr<BoolAttr, "false">:$bound_ctrl)> {
  let summary = "AMDGPU DPP operation";
  let description = [{
    This operation represents DPP functionality in a GPU program.
     DPP provides the following operations:
    - Full crossbar in a group of four (`quad_perm`)
    - Wavefront shift left by one lane (`wave_shl`)
    - Wavefront shift right by one lane (`wave_shr`)
    - Wavefront rotate right by one lane (`wave_ror`)
    - Wavefront rotate left by one lane (`wave_rol`)
    - Row shift left by 1–15 lanes (`row_shl`)
    - Row shift right by 1–15 lanes (`row_shr`)
    - Row rotate right by 1–15 lanes (`row_ror`)
    - Reverse within a row (`row_mirror`)
    - Reverse within a half-row (`row_half_mirror`)
    - Broadcast the 15th lane of each row to the next row (`row_bcast`)
    - Broadcast lane 31 to rows 2 and 3 (`row_bcast`)
  }];
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $old $src $kind (`(` $permArgument^ `)`)? attr-dict `:` type($result)
  }];
  let hasVerifier = 1;
}

def AMDGPU_SwizzleBitModeOp : AMDGPU_Op<"swizzle_bitmode",
    [Pure, AllTypesMatch<["result", "src"]>]>,
  Arguments<(ins AnyIntegerOrFloatOr1DVector:$src,
                 I32Attr:$and_mask,
                 I32Attr:$or_mask,
                 I32Attr:$xor_mask
             )> {
  let summary = "AMDGPU ds_swizzle op, bitmode variant";
  let description = [{
    High-level wrapper on bitmode `rocdl.ds_swizzle` op, masks are represented
    as separate fields so user won't need to do manual bitpacking.

    Supports arbitrary int/float/vector types, which will be repacked to i32 and
    one or more `rocdl.ds_swizzle` ops during lowering.
  }];
  let results = (outs AnyIntegerOrFloatOr1DVector:$result);
  let assemblyFormat = [{
    $src $and_mask $or_mask $xor_mask attr-dict `:` type($result)
  }];
}

def AMDGPU_PermlaneSwapOp : AMDGPU_Op<"permlane_swap", [Pure, AllTypesMatch<["result", "src"]>]> {
  let summary = "AMDGPU permlane swap op";
  let description = [{
    High-level wrapper on `rocdl.permlane{16,32}.swap` variants for permutations
    on rows of lanes in a subgroup.

    Supports arbitrary int/float/vector types, which will be repacked to i32 and
    one or more `rocdl.permlane_swap` ops during lowering.
    Supported lane permutations:
    - Swap the data between odd and even rows of 16 lanes
    - Swap the data between the first 32 lanes and the last 32 lanes

    Example:
    ```mlir
    %0 = amdgpu.permlane_swap %src 16 : f16
    %1 = amdgpu.permlane_swap %src 32 { fetch_inactive = true, bound_ctrl = true } : f16
    ```

    Operands:
    * `$src`: Vector register to permute across lanes of the subgroup.
    * `$row_length`: The length of a row to permute in number of lanes (valid values are 16 and 32).
    * `$fetch_inactive`: Optional. Used to dertermine behavior of a fetch from a disabled lane.
      `fetch_inactive = false`: If the source lane is disabled, use `bound_ctrl` to determine the source value.
      `fetch_inactive = true`: If the source lane is disabled, fetch the source value anyway (ignoring `bound_ctrl`).
    * `$bound_ctrl`: Optional. Used to determine what a thread should do if its source operand is from 
      a disabled lane: use the value zero, or disable the write. 
      `bound_ctrl = false`: Do not write when source is from a disabled lane
      `bound_ctrl = true`: Use zero as input if source is from a disabled lane

    Note: Lowering is only supported on gfx950 and up.
  }];
  let arguments = (ins AnyIntegerOrFloatOr1DVector:$src,
                       I32Attr:$row_length,
                       DefaultValuedAttr<BoolAttr, "false">:$fetch_inactive,
                       DefaultValuedAttr<BoolAttr, "false">:$bound_ctrl);
  let results = (outs AnyIntegerOrFloatOr1DVector:$result);
  let assemblyFormat = [{
    $src $row_length attr-dict `:` type($result)
  }];
  let hasVerifier = 1;
}

def AMDGPU_LDSBarrierOp : AMDGPU_Op<"lds_barrier"> {
  let summary = "Barrier that includes a wait for LDS memory operations.";
  let description = [{
    `amdgpu.lds_barrier` is both a barrier (all workitems in a workgroup must reach
    the barrier before any of them may proceed past it) and a wait for all
    operations that affect the Local Data Store (LDS) issued from that wrokgroup
    to complete before the workgroup may continue. Since the LDS is per-workgroup
    memory, this barrier may be used, for example, to ensure all workitems have
    written data to LDS before any workitem attempts to read from it.

    Note that `lds_barrier` does **not** force reads to or from global memory
    to complete before execution continues. Therefore, it should be used when
    operations on global memory can be issued far in advance of when their results
    are used (for example, by writing them to LDS).

    WARNING: On architectures that do not support the BackOffBarrier feature,
    (those which will implement this barrier by emitting inline assembly),
    use of this operation will impede the usabiliity of memory watches (including
    breakpoints set on variables) when debugging.
  }];
  let assemblyFormat = "attr-dict";
}

def AMDGPU_SchedBarrierOpOpt : I32BitEnumAttr<"sched_barrier_opt_enum",
    "The possible options for scheduling barriers",
    [
      I32BitEnumAttrCaseNone<"none">,
      I32BitEnumAttrCaseBit<"non_mem_non_sideffect", 0>,
      I32BitEnumAttrCaseBit<"valu", 1>,
      I32BitEnumAttrCaseBit<"salu", 2>,
      I32BitEnumAttrCaseBit<"mfma_wmma",  3>,
      I32BitEnumAttrCaseBit<"all_vmem",  4>,
      I32BitEnumAttrCaseBit<"vmem_read",  5>,
      I32BitEnumAttrCaseBit<"vmem_write", 6>,
      I32BitEnumAttrCaseBit<"all_ds", 7>,
      I32BitEnumAttrCaseBit<"ds_read", 8>,
      I32BitEnumAttrCaseBit<"ds_write", 9>,
      I32BitEnumAttrCaseBit<"transcendental", 10>
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::amdgpu";
}

def AMDGPU_SchedBarrierOpOptAttr : EnumAttr<AMDGPU_Dialect, AMDGPU_SchedBarrierOpOpt,
  "sched_barrier_opt">{
   let assemblyFormat = "`<` $value `>`";
}

def AMDGPU_SchedBarrierOp :
  AMDGPU_Op<"sched_barrier">,
  Arguments<(ins  AMDGPU_SchedBarrierOpOptAttr:$opts)>
  {
  let summary = "Barrier that limits the backend scheduler of instruction movement";
  let description = [{
    `amdgpu.sched_barrier` serves as a barrier that could be
    configured to restrict movements of instructions through it as
    defined by sched_barrier_opts.
  }];
  let assemblyFormat = [{
    `allow` `=` $opts attr-dict
  }];
}

def AMDGPU_MemoryCounterWaitOp :
  AMDGPU_Op<"memory_counter_wait">,
  Arguments<(ins
      OptionalAttr<I32Attr>:$load,
      OptionalAttr<I32Attr>:$store,
      OptionalAttr<I32Attr>:$ds,
      OptionalAttr<I32Attr>:$exp
    )>
  {
  let summary = "Wait for specified hardware counters";
  let description = [{
    Wait for the specified counters to be less-than or equal-to the provided
    values before continuing.

    Counters can lower to different instructions on different architectires,
    including clamping to the some HW supported max value or combining multiple
    counters into one.
  }];
  let assemblyFormat = [{
    oilist( `load` `(` $load `)` | `store` `(` $store `)` | `ds` `(` $ds `)` | `exp` `(` $exp `)` ) attr-dict
  }];
}

def AMDGPU_MFMAPermB : I32EnumAttr<"MFMAPermB",
    "The possible permutations of the lanes storing B available in an MFMA",
    [
      I32EnumAttrCase<"none",            0>,
      I32EnumAttrCase<"bcast_first_32",  1>,
      I32EnumAttrCase<"bcast_second_32", 2>,
      I32EnumAttrCase<"rotate_16_right", 3>,
      I32EnumAttrCase<"bcast_first_16",  4>,
      I32EnumAttrCase<"bcast_second_16", 5>,
      I32EnumAttrCase<"bcast_third_16",  6>,
      I32EnumAttrCase<"bcast_fourth_16", 7>
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::amdgpu";
}

def AMDGPU_MFMAPermBAttr : EnumAttr<AMDGPU_Dialect, AMDGPU_MFMAPermB,
  "mfma_perm_b">;

// mfma
def MFMAInTypes : AnyTypeOf<[F32, F64, I32, I64,
                             VectorOfLengthAndType<[2], [F32]>,
                             VectorOfLengthAndType<[4, 8], [F16]>,
                             VectorOfLengthAndType<[2, 4, 8], [BF16]>,
                             VectorOfLengthAndType<[4, 8, 16], [I8]>,
                             VectorOfLengthAndType<[8], [F8E5M2FNUZ, F8E4M3FNUZ]>,
                             VectorOfLengthAndType<[8, 32], [F8E5M2, F8E4M3FN]>,
                             VectorOfLengthAndType<[32], [F6E2M3FN, F6E3M2FN, F4E2M1FN]>]>;
def MFMAOutTypes : AnyTypeOf<[F64,
                              VectorOfLengthAndType<[4, 16, 32], [F32]>,
                              VectorOfLengthAndType<[4, 16, 32], [I32]>,
                              VectorOfLengthAndType<[4], [F64]>]>;
// scaled_mfma
def ScaledMFMAInTypes : AnyTypeOf<[VectorOfLengthAndType<[32], [F8E5M2, F8E4M3FN]>,
                                   VectorOfLengthAndType<[32], [F6E2M3FN, F6E3M2FN, F4E2M1FN]>]>;
def ScaledMFMAOutTypes : AnyTypeOf<[VectorOfLengthAndType<[4, 16], [F32]>]>;
// wmma
def WMMAInTypes : AnyTypeOf<[VectorOfLengthAndType<
                             [4, 8, 16],
                             [F16, BF16,
                              I8, SI8, UI8,
                              I<4>, SI<4>, UI<4>,
                              F8E4M3FN, F8E5M2]>]>;
def WMMAOutTypes : AnyTypeOf<[VectorOfLengthAndType<[4, 8], [F32, I32]>,
                              VectorOfLengthAndType<[4, 8, 16], [F16, BF16]>]>;

def AMDGPU_MFMAOp :
    AMDGPU_Op<"mfma", [AllTypesMatch<["destC", "destD"]>,
                        Pure]>,
    Arguments<(ins
                   I32Attr:$m,
                   I32Attr:$n,
                   I32Attr:$k,
                   I32Attr:$blocks,
                   MFMAInTypes:$sourceA,
                   MFMAInTypes:$sourceB,
                   MFMAOutTypes:$destC,
                   DefaultValuedAttr<I32Attr, "0">:$cbsz,
                   DefaultValuedAttr<I32Attr, "0">:$abid,
                   DefaultValuedAttr<AMDGPU_MFMAPermBAttr,
                    "::mlir::amdgpu::MFMAPermB::none">:$blgp,
                   UnitAttr:$reducePrecision,
                   UnitAttr:$negateA,
                   UnitAttr:$negateB,
                   UnitAttr:$negateC)>,
    Results<(outs MFMAOutTypes: $destD)> {
  let summary = "MLIR wrapper for CDNA mfma instructions";
  let description = [{
    The `amdgpu.mfma` op is an MLIR wrapper around intrinsics
    for various `mfma` instructions in the CDNA architecture, which perform
    multiple outer products in order to allow fast matrix multiplication.

    The wrapper will select an appropriate `mfma` instruction, if one is available,
    based on the provided `m`, `k`, `n`, and `nBlks` attributes, along with the
    types of the source and destination arguments.

    For information on the layouts of the input and output matrces (which are stored
    in `sourceA`, `sourceB`, `destC`, and `destD`), see the CDNA ISA documentation.

    The `cbsz`, `abid`, and `blgp` parameters control how the lanes of the wave
    are permuted when matrix data is being loaded: `blgp` can be any number of
    fixed permutations, `cbsz` specifies the log_2 of the number of chunks the lanes
    holding sourceA are split into, and `abid` selects one of those chunks.

    Note, this wrapper allows specifying `vector<4Kxi8>` arguments to MFMA
    intrinsics that take an integer type of width `4K`. For example,
    one can provide a vector<4xi8> as an argument to an MFMA instruction that
    logically takes 4 i8s but whose intrinsics are specified to take an i32.
    In these cases, the bytes in the vector will be concatenated in little-endian
    order (that is, v[0] will go to arg[7:0], v[1] to arg[15:8] and so on).

    The negateA, negateB, and negateC flags are only supported for double-precision
    operations on gfx94x.
  }];
  let assemblyFormat = [{
    $sourceA `*` $sourceB `+` $destC
    attr-dict
    `blgp` `=` $blgp
    `:` type($sourceA) `,` type($sourceB) `,` type($destC)
  }];
  let hasVerifier = 1;
}

def AMDGPU_WMMAOp :
    AMDGPU_Op<"wmma", [AllTypesMatch<["destC", "destD"]>,
                       Pure]>,
    Arguments<(ins
                   WMMAInTypes:$sourceA,
                   WMMAInTypes:$sourceB,
                   WMMAOutTypes:$destC,
                   DefaultValuedAttr<ConfinedAttr<I32Attr, [IntMinValue<0>, IntMaxValue<1>]>, "0">:$subwordOffset,
                   UnitAttr:$unsignedA,
                   UnitAttr:$unsignedB,
                   UnitAttr:$clamp)>,
    Results<(outs WMMAOutTypes: $destD)> {
  let summary = "MLIR wrapper for RDNA3 wmma instructions";
  let description = [{
    The `amdgpu.wmma` op is an MLIR wrapper around intrinsics
    for various `wmma` instructions in the RDNA3 or RDNA4 architecture, which
    perform a 16x16 * 16x16 matrix multiplication for different data types.
    Note that in gfx12/RDNA4, there is also a 16x32 * 32x16 instruction for 4-bit
    integer inputs.

    On gfx11/RDNA3, emitting f16->f16 (or bf16->bf16) wmma the output is a 16xf16
    (or 16xbf16) vector containing only 8 valid values:
      - If `subwordOffset` is 0, then the output is stored at indices 0, 2, 4, ..., 14.
      - If `subwordOffset` is 1, then the output is stored at indices 1, 3, 5, ..., 15.
    On gfx12/RDNA4, the result is instead returned as a vector<8 x f16/bf16> where
    all values are valid and the `subwordOffset` must be `0`, as it cannot be used.

    `unsignedA` and `unsignedB` flag that the `int8` LLVM inputs are unsigned.

    The `clamp` flag is used to saturate the output of type T to numeric_limits<T>::max()
    in case of overflow.
  }];
  let assemblyFormat = [{
    $sourceA `*` $sourceB `+` $destC
    attr-dict
    `:` type($sourceA) `,` type($sourceB) `,` type($destC)
  }];
  let hasVerifier = 1;
}

def AMDGPU_GatherToLDSOp :
    AMDGPU_Op<"gather_to_lds", [AttrSizedOperandSegments]>,
    Arguments<(ins
                   Arg<AnyMemRef, "buffer to gather from", [MemRead]>:$src,
                   Variadic<Index>:$srcIndices,
                   Arg<AnyMemRef, "buffer to write to", [MemWrite]>:$dst,
                   Variadic<Index>:$dstIndices,
                   TypeAttr:$transferType
                   )>,
    Results<(outs)> {
  let summary = "MLIR wrapper for CDNA Gather to LDS instructions";
  let description = [{
    The `amdgpu.gather_to_lds` op is a wrapper around the `global_load_lds` instructions.

    Operands:
    * `$src`: global memory (including fat buffer) memref to read from.
    * `$srcIndices`: indices into `$src` to read from for this thread.
    * `$dst`: LDS memory memref to write to.
    * `$dstIndices`: base indices into `$dst` to write to for the subgroup of this thread.
      The elements gathered by the subgroup will be written contiguously in order of lane ID
      starting at `$dst[$dstIndices]`. Byte-sized (ex. i8) or short-sized (ex. i16)
      types will be zero-padded/extended to 32 bits before being written. 96-bit types
      (ex. vector<3xf32>) will be zero-padded to 128 bits before being written. Only the
      offsets held by lane 0 are used.
    * `$transferType`: type of the data to be transferred by each thread. This is used to determine
      the size of the data to be transferred and the number of threads in the subgroup.
      The transfer type must be a scalar type or a vector type with a single element type.

    The `$dst`, along with its indices, points to the memory location the subgroup of this thread
    will write to.

    Note: only supported on gfx9 and gfx10.
  }];
  let assemblyFormat = [{
    $src `[` $srcIndices `]` `,` $dst `[` $dstIndices `]` attr-dict `:` $transferType `,` type($src) `,` type($dst)
  }];
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

def AMDGPU_TransposeLoadOp :
    AMDGPU_Op<"transpose_load", [SameVariadicOperandSize]>,
    Arguments<(ins Arg<AnyMemRef, "buffer to transpose load from", [MemRead]>:$src, Variadic<Index>:$srcIndices)>,
    Results<(outs AnyTypeOf<[AnyVectorOfNonZeroRank]>:$result)> {
  let summary = "MLIR wrapper for CDNA Transpose Load instructions";
  let description = [{
    The `amdgpu.transpose_load` op is a wrapper around the `ds_read_tr` instructions.
    The transpose load op represents a subgroup load from LDS memory,
    where the subgroup of threads collectively reads a matrix from the source
    memref, with each thread reading a vector of the matrix, and gets a transposed matrix
    in as the result. That is, each thread reads a vector of the col-major matrix at different
    indices, and the thread's read result is a vector of the corresponding row of the transposed
    matrix.

    This op is a direct wrapper around the ROCDL `ds_read_tr` family intrinsics. Please refer
    to the CDNA4 ISA documentation for more details about its exact semantics.

    Format example:
    ```
    %0 = amdgpu.transpose_load %src[%srcIndices] : memref<128x256xf16> -> vector<4xf16>
    ```
    Operands:
    * `$src`: LDS memref to read from.
    * `$srcIndices`: indices into `$src` to read from for this thread.
    * `$result`: target register this transpose load instruction will write to.

    Note: Lowering is only supported on gfx950 and up.
  }];
  let assemblyFormat = [{
    $src `[` $srcIndices `]` attr-dict `:` type($src) `->` type($result)
  }];
  let hasVerifier = 1;
}

def AMDGPU_ScaledMFMAOp :
    AMDGPU_Op<"scaled_mfma", [AllTypesMatch<["destC", "destD"]>,
                        Pure]>,
    Arguments<(ins
                   I32Attr:$m,
                   I32Attr:$n,
                   I32Attr:$k,
                   ScaledMFMAInTypes:$sourceA,
                   ScaledMFMAInTypes:$sourceB,
                   ScaledMFMAOutTypes:$destC,
                   AnyTypeOf<[F8E8M0FNU, FixedVectorOfLengthAndType<[4], [F8E8M0FNU]>]>:$scalesA,
                   AnyTypeOf<[F8E8M0FNU, FixedVectorOfLengthAndType<[4], [F8E8M0FNU]>]>:$scalesB,
                   ConfinedAttr<I32Attr, [IntNonNegative, IntMaxValue<3>]>:$scalesIdxA,
                   ConfinedAttr<I32Attr, [IntNonNegative, IntMaxValue<3>]>:$scalesIdxB
                   )>,
    Results<(outs ScaledMFMAOutTypes: $destD)> {
  let summary = "MLIR wrapper for CDNA scaled mfma instructions";
  let description = [{
    The `amdgpu.scaled_mfma` op is an MLIR wrapper around intrinsics
    for various scaled versions of `mfma` instructions in the CDNA architecture, which perform
    multiple outer products in order to allow fast matrix multiplication.

    The wrapper will select an appropriate `mfma` instruction, if one is available,
    based on the provided `m`, `k`, `n`, and `nBlks` attributes, along with the
    types of the source and destination arguments.

    Note, this wrapper allows specifying `vector<4Kxi8>` arguments to MFMA
    intrinsics that take an integer type of width `4K`. For example,
    one can provide a `vector<4xi8>` as an argument to an MFMA instruction that
    logically takes 4 i8s but whose intrinsics are specified to take an i32.
    In these cases, the bytes in the vector will be concatenated in little-endian
    order (that is, v[0] will go to arg[7:0], v[1] to arg[15:8] and so on).

    This wrapper takes inspiration from `amdgpu.mfma`, but has some key differences:
    - `amdgpu.scaled_mfma` operates on fp4 (f4E2M1FN), fp6 (f6E2M3FN and f6E3M2FN) and
    fp8 (f8E4M3FN and f8E5M2) types using either M=N=16, K=128 or M=N=32, K=64 as their tile
    size.
    - `amdgpu.scaled_mfma` does not support broadcasting. So, `cbsz`, `abid`, and `blgp`
    are omitted from this wrapper.
    - The `negateA`, `negateB`, and `negateC` flags in `amdgpu.mfma` are only supported for
    double-precision operations on gfx94x and so are not included here.
  }];
  let assemblyFormat = [{
    `(` $scalesA `[` $scalesIdxA `]` `*` $sourceA `)` `*` `(` $scalesB `[` $scalesIdxB `]` `*` $sourceB `)` `+` $destC
    attr-dict
    `:` type($scalesA) `,` type($sourceA) `,` type($scalesB) `,` type($sourceB) `,` type($destC)
  }];
}
#endif // AMDGPU
