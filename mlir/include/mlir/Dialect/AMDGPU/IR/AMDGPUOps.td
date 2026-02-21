//===-- AMDGPUOps.td - AMDGPU dialect operations *- tablegen -*------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_AMDGPU_IR_AMDGPUOPS_TD
#define MLIR_DIALECT_AMDGPU_IR_AMDGPUOPS_TD

include "mlir/Dialect/AMDGPU/IR/AMDGPUBase.td"
include "mlir/Dialect/AMDGPU/IR/AMDGPUAttrs.td"
include "mlir/Dialect/AMDGPU/IR/AMDGPUTypes.td"

include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// AMDGPU common type constraints
//===----------------------------------------------------------------------===//

class AMDGPU_ConcreteVector<Type elem, int length> :
  FixedVectorOfLengthAndType<[length], [elem]>,
  BuildableType<
    "::mlir::VectorType::get({" # length # "} ,"
      # elem.builderCall # ")">;

def AnyIntegerOrFloat : AnyTypeOf<[AnySignlessInteger, AnyFloat], "Integer or Float">;

def AnyIntegerOrFloatOr1DVector :
  AnyTypeOf<[AnyIntegerOrFloat, FixedVectorOfRankAndType<[1], [AnyIntegerOrFloat]>]>;

// Types with element width up to 64 bits, used to keep dpp operands legal.
def AMDGPU_IntOrFloatWidthLeq64 : Type<
  CPred<"$_self.isIntOrFloat() && $_self.getIntOrFloatBitWidth() <= 64">,
  "integer or float with element bitwidth <= 64">;

def AMDGPU_IntOrFloatOr1DVectorWidthLeq64 :
  AnyTypeOf<[
    AMDGPU_IntOrFloatWidthLeq64,
    FixedVectorOfRankAndType<[1], [AMDGPU_IntOrFloatWidthLeq64]>
  ]>;

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

def AMDGPU_ScaledExtPackedMatrixOp
    : AMDGPU_Op<"scaled_ext_packed_matrix", [Pure, AllShapesMatch<["source", "res"]>]>,
      Arguments<(
          ins AnyTypeOf<[FixedVectorOfShapeAndType<[8], F4E2M1FN>,
                         FixedVectorOfShapeAndType<[8], F8E4M3FN>,
                         FixedVectorOfShapeAndType<[8], F8E5M2>,
                         FixedVectorOfShapeAndType<[16], F6E2M3FN>,
                         FixedVectorOfShapeAndType<[16], F6E3M2FN>]>:$source,
          FixedVectorOfShapeAndType<[4], F8E8M0FNU>:$scale,
          ConfinedAttr<I32Attr, [IntIsOneOf<[16, 32]>]>:$blockSize,
          ConfinedAttr<I32Attr, [IntIsOneOf<[0, 16]>]>:$firstScaleLane,
          ConfinedAttr<I32Attr, [IntMinValue<0>, IntMaxValue<3>]>:$firstScaleByte)>,
      Results<(
          outs AnyTypeOf<[FixedVectorOfShapeAndType<[8], F32>,
                          FixedVectorOfShapeAndType<[8], F16>,
                          FixedVectorOfShapeAndType<[8], BF16>,
                          FixedVectorOfShapeAndType<[16], F32>,
                          FixedVectorOfShapeAndType<[16], F16>,
                          FixedVectorOfShapeAndType<[16], BF16>]>:$res)> {

  let summary = "Extend a wave-wide matrix of packed floating point values";

  let description = [{
    Extend matrix of microfloats (8 or 16 elements per lane) using a set of scales
    that may be stored on other lanes.

    The scales applied to the input microfloats are stored in bytes which
    come from the `scales` input provided in a *half* of the wave identified
    by `firstScaleLane`. The bytes used is selected by `firstScaleByte` and depends
    on the type of `source`. The 16 vectors in consecutive lanes starting from
    `firstScaleLane` (which we'll call the scale vectors) will be used by both
    halves of the wave (with lane L reading from L % 16'th scale vector).

    When `source` is either F4E2M1FN, F6E2M3FN, or F6E3M2FN each half of the
    wave will use a different byte. The first one being `firstScaleByte` and
    the second one being `firstScaleByte` + 1. When the block size is 32,
    `firstScaleByte` can be either 0 or 2, selecting halves of the scale vectors.
    Lanes 0-15 will read from `firstScaleByte` and lanes 16-31 will read
    from `firstScaleByte` + 1.


    For example:
    ```mlir
    // Input: 8-element vector of F8E4M3FN, converting to F32
    // Lanes 0-15 read from byte 0, lanes 16-31 read from byte 1
    %result = amdgpu.scaled_ext_packed_matrix %source scale(%scales)
      blockSize(32) firstScaleLane(0) firstScaleByte(0)
      : vector<8xf8E4M3FN>, vector<4xf8E8M0FNU> -> vector<8xf32>

    // Input: 16-element vector of F6E2M3FN, converting to F16
    // Lanes 0-15 read from byte 2, lanes 16-31 read from byte 3
    %result = amdgpu.scaled_ext_packed_matrix %source scale(%scales)
      blockSize(32) firstScaleLane(16) firstScaleByte(2)
      : vector<16xf6E2M3FN>, vector<4xf8E8M0FNU> -> vector<16xf16>
    ```

    When `source` is either F4E2M1FN, F6E2M3FN, or F6E3M2FN and
    the block size is 16, `firstScaleByte` can be 0 or 1.
    Lanes 0-15 read from the `firstScaleByte`th element of the scale vectors,
    while lanes 16-31 read from `firstScaleByte` + 2.
    For example:
    ```mlir
    // Input: 8-element vector of F8E5M2, converting to BF16
    // Lanes 0-15 read from byte 0, lanes 16-31 read from byte 2 (0+2)
    %result = amdgpu.scaled_ext_packed_matrix %source scale(%scales)
      blockSize(16) firstScaleLane(0) firstScaleByte(0)
      : vector<8xf8E5M2>, vector<4xf8E8M0FNU> -> vector<8xbf16>

    // Input: 16-element vector of F6E3M2FN, converting to F32
    // Lanes 0-15 read from byte 1, lanes 16-31 read from byte 3 (1+2)
    %result = amdgpu.scaled_ext_packed_matrix %source scale(%scales)
      blockSize(16) firstScaleLane(16) firstScaleByte(1)
      : vector<16xf6E3M2FN>, vector<4xf8E8M0FNU> -> vector<16xf32>
    ```

    Note: the layout for the scales generally mirrors how the WMMA
    instructions use for matrix scales. These selection operands allows
    one to choose portions of the matrix to convert.

    When `source` is either F8E4M3FN or F8E5M2 and `blockSize` is 32,
    then the same byte will be used by both halves of the wave.
    In this case, `firstScaleByte` can be any value from 0 to 3.

    When `source` is either F8E4M3FN or F8E5M2 and `blockSize` is 16,
    following combinations are allowed:
    * `firstScaleLane(0), firstScaleByte(0)`
    * `firstScaleLane(16), firstScaleByte(2)`
    all other combinations are reserved.

    Available on gfx1250+.
  }];

  let assemblyFormat = [{
    attr-dict $source
    `scale` `(` $scale `)`
    `blockSize` `(` $blockSize `)`
    `firstScaleLane` `(` $firstScaleLane`)`
    `firstScaleByte` `(` $firstScaleByte `)`
    `:` type($source) `,` type($scale) `->` type($res)
  }];

  let hasVerifier = 1;

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
       DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface,
                                 ["reifyDimOfResult"]>,
       ViewLikeOpInterface, AttrSizedOperandSegments]>,
    Arguments<(ins AnyMemRef:$source,
      Optional<I64>:$validBytes,
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


def AMDGPU_DPPOp : AMDGPU_Op<"dpp",
    [Pure, SameTypeOperands, AllTypesMatch<["result", "old", "src"]>]>,
  Arguments<(ins AMDGPU_IntOrFloatOr1DVectorWidthLeq64:$old,
                 AMDGPU_IntOrFloatOr1DVectorWidthLeq64:$src,
                 AMDGPU_DPPPermAttr:$kind,
                 OptionalAttr<AnyAttrOf<[I32Attr, ArrayAttr, UnitAttr]>>:$permArgument,
                 DefaultValuedAttr<I32Attr, "0xf">:$row_mask,
                 DefaultValuedAttr<I32Attr, "0xf">:$bank_mask,
                 DefaultValuedAttr<BoolAttr, "false">:$bound_ctrl)> {
  let summary = "AMDGPU DPP operation";
  let description = [{
    The `amdgpu.dpp` op performs a Data Parallel Primitives (DPP) lane
    permutation on a source value within a wavefront. Each lane reads its
    source data from another lane according to the permutation mode specified
    by `kind`. DPP operates at dword (32-bit) granularity: sub-32-bit types
    (e.g., f16, i16) are packed into an i32 during lowering, permuted, and
    extracted back.

    A Wave64 wavefront has 64 lanes (0-63) organized hierarchically:
    - 4 rows of 16 lanes each: row 0 = lanes 0-15, row 1 = lanes 16-31,
      row 2 = lanes 32-47, row 3 = lanes 48-63.
    - Each row is divided into 4 banks of 4 consecutive lanes: bank 0 =
      lanes 0-3, bank 1 = lanes 4-7, bank 2 = lanes 8-11, bank 3 =
      lanes 12-15 (lane numbers shown for row 0; add 16/32/48 for other rows).

    The `kind` attribute selects the permutation. Some modes require a
    `permArgument`; others take no argument.

    Quad permutation:
    - `quad_perm([a, b, c, d])`: Full crossbar within each group of 4
      consecutive lanes (a quad). Each element is in [0, 3] and selects which
      lane within the quad to read from. Lane 4k+i reads from lane 4k+perm[i].
      For example, `quad_perm([1, 0, 3, 2])` swaps adjacent pairs within
      every quad.

    Row shifts and rotates (operate within each 16-lane row independently):
    - `row_shl(N)`: Shift left by N (1-15) within the row. Lane n reads from
      lane (n % 16) + N in the same row. Lanes where the source index exceeds
      15 are out of bounds (see `bound_ctrl`).
    - `row_shr(N)`: Shift right by N (1-15) within the row. Lane n reads from
      lane (n % 16) - N in the same row. Lanes where the source index is
      negative are out of bounds.
    - `row_ror(N)`: Rotate right by N (1-15) within the row. Lane n reads from
      lane ((n % 16) - N) mod 16 in the same row. Always in bounds.

    Wavefront shifts and rotates (operate across all 64 lanes):
    - `wave_shl`: Shift left by 1. Lane n reads from lane n + 1. Lane 63 is
      out of bounds.
    - `wave_shr`: Shift right by 1. Lane n reads from lane n - 1. Lane 0 is
      out of bounds.
    - `wave_rol`: Rotate left by 1. Lane n reads from lane (n + 1) mod 64.
    - `wave_ror`: Rotate right by 1. Lane n reads from lane (n - 1) mod 64.

    Row mirrors:
    - `row_mirror`: Reverse lanes within each 16-lane row. Lane n reads from
      lane 15 - (n % 16) within its row.
    - `row_half_mirror`: Reverse within each 8-lane half-row. Lane n reads
      from lane 7 - (n % 8) within its half-row.

    Row broadcasts:
    - `row_bcast_15`: Lane 15 of each row broadcasts to all lanes of the next
      row. Lanes in row 0 are not affected (retain `old`).
    - `row_bcast_31`: Lane 31 broadcasts to all lanes in rows 2 and 3.
      Lanes in rows 0 and 1 are not affected (retain `old`).

    Example:
    ```mlir
    // Swap adjacent pairs within each quad (lanes 0<->1, 2<->3, etc.)
    %0 = amdgpu.dpp %old %src quad_perm([1, 0, 3, 2]) : i32

    // Shift right by 1 lane within each 16-lane row.
    // bound_ctrl=true -> lanes that would read past the row return 0.
    // row_mask=0x5 (0b0101) -> only rows 0 and 2 apply the shift;
    // rows 1 and 3 pass through %old unchanged.
    %1 = amdgpu.dpp %old %src row_shr(0x1 : i32)
      { row_mask = 0x5 : i32, bound_ctrl = true } : f32

    // Rotate left across the full wavefront by 1 lane
    %2 = amdgpu.dpp %old %src wave_rol : i32
    ```

    Operands:
    * `$old`: Fallback value. Lanes that are masked off by `row_mask` /
      `bank_mask` retain `old`. For lanes with an out-of-bounds source, behavior
      depends on `bound_ctrl`.
    * `$src`: Source value to be permuted across lanes.
    * `$kind`: A `#amdgpu.dpp_perm` enum selecting the permutation mode.
    * `$permArgument`: Mode-specific argument. Required for `quad_perm`
      (array of 4 integers in [0, 3]) and `row_shl`/`row_shr`/`row_ror`
      (integer in [1, 15]). Absent for all other modes.
    * `$row_mask` (default 0xf): 4-bit mask controlling which rows write
      results. Bit i enables row i (bit 0 = lanes 0-15, bit 1 = lanes
      16-31, etc.). Disabled lanes retain `old`.
    * `$bank_mask` (default 0xf): 4-bit mask controlling which banks write
      results. Bit i enables bank i (bit 0 = lanes 0-3, 16-19, 32-35, 48-51).
      Disabled lanes retain `old`.
    * `$bound_ctrl` (default false): When false, out of bounds lanes retain
      `old`. When true, out-of-bounds lanes receive zero.
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
    **DEPRECATION NOTICE**: Unless you need the inline-assembly-based workaround
    for gfx908/MI-100, you should represent this pattern with the equivalent

    ```mlir
    gpu.barrier memfence [#gpu.address_space<workgroup>]
    ```

    instead.

    `amdgpu.lds_barrier` is both a barrier (all workitems in a workgroup must reach
    the barrier before any of them may proceed past it) and a wait for all
    operations that affect the Local Data Store (LDS) issued from that workgroup
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
  let hasCanonicalizer = 1;
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
      OptionalAttr<I32Attr>:$exp,
      OptionalAttr<I32Attr>:$tensor
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
    oilist( `load` `(` $load `)` | `store` `(` $store `)` | `ds` `(` $ds `)` | `exp` `(` $exp `)` | `tensor` `(` $tensor `)` ) attr-dict
  }];

  let hasCanonicalizer = 1;
}


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

// sparse_mfma (smfmac)
def SMFMACSparseInTypes : AnyTypeOf<[
    VectorOfLengthAndType<[4, 8], [F16]>,
    VectorOfLengthAndType<[4, 8], [BF16]>,
    VectorOfLengthAndType<[8, 16], [I8]>,
    VectorOfLengthAndType<[8, 16], [F8E4M3FN, F8E5M2]>,
    VectorOfLengthAndType<[8, 16], [F8E4M3FNUZ, F8E5M2FNUZ]>
]>;

def SMFMACDenseInTypes : AnyTypeOf<[
    VectorOfLengthAndType<[8, 16], [F16]>,
    VectorOfLengthAndType<[8, 16], [BF16]>,
    VectorOfLengthAndType<[16, 32], [I8]>,
    VectorOfLengthAndType<[16, 32], [F8E4M3FN, F8E5M2]>,
    VectorOfLengthAndType<[16, 32], [F8E4M3FNUZ, F8E5M2FNUZ]>
]>;

def SMFMACOutTypes : AnyTypeOf<[
    VectorOfLengthAndType<[4, 16], [F32]>,
    VectorOfLengthAndType<[4, 16], [I32]>
]>;

def SMFMACIdxTypes : AnyTypeOf<[
    FixedVectorOfLengthAndType<[4], [I8]>,
    FixedVectorOfLengthAndType<[2], [I16]>
]>;

// scaled_mfma
def ScaledMFMAInTypes : AnyTypeOf<[VectorOfLengthAndType<[32], [F8E5M2, F8E4M3FN]>,
                                   VectorOfLengthAndType<[32], [F6E2M3FN, F6E3M2FN, F4E2M1FN]>]>;
def ScaledMFMAOutTypes : AnyTypeOf<[VectorOfLengthAndType<[4, 16], [F32]>]>;

// scaled_wmma
def ScaledWMMAInTypes
    : AnyTypeOf<[VectorOfLengthAndType<[64], [F8E5M2, F8E4M3FN]>,
                 VectorOfLengthAndType<[64], [F6E2M3FN, F6E3M2FN]>,
                 VectorOfLengthAndType<[64, 128], [F4E2M1FN]>]>;

def ScaledWMMAOutTypes : AnyTypeOf<[VectorOfLengthAndType<[8, 16], [F32]>]>;

// wmma
def WMMAInTypes : AnyTypeOf<[VectorOfLengthAndType<[2], [F32]>,
                             VectorOfLengthAndType<[4, 8, 16], [F16, BF16]>,
                             VectorOfLengthAndType<[4, 8, 16, 32], [I8, SI8, UI8]>,
                             VectorOfLengthAndType<[4, 8, 32, 64], [F8E4M3FN, F8E5M2]>,
                             VectorOfLengthAndType<[4, 8, 16], [I<4>, SI<4>, UI<4>]>]>;
def WMMAOutTypes : AnyTypeOf<[VectorOfLengthAndType<[4, 8], [F32, I32]>,
                              VectorOfLengthAndType<[4, 8, 16], [F16, BF16]>]>;

def AMDGPU_MFMAOp :
    AMDGPU_Op<"mfma", [AllTypesMatch<["destC", "destD"]>,
                        Pure]>,
    Arguments<(ins
                   ConfinedAttr<I32Attr, [IntIsOneOf<[4, 16, 32]>]>:$m,
                   ConfinedAttr<I32Attr, [IntIsOneOf<[4, 16, 32]>]>:$n,
                   ConfinedAttr<I32Attr, [IntIsOneOf<[1, 2, 4, 8, 16, 32, 64, 128]>]>:$k,
                   DefaultValuedAttr<ConfinedAttr<I32Attr, [IntIsOneOf<[1, 2, 4, 16]>]>, "1">:$blocks,
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

    For information on the layouts of the input and output matrices (which are stored
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

    Example:
    ```mlir
      %0 = amdgpu.mfma 16x16x16 %matA * %matB + %matC
        : vector<4xf16>, vector<4xf16>, vector<4xf32>

      %1 = amdgpu.mfma 32x32x1 %matD * %matE + %matF
        { abid = 1 : i32, cbsz = 1 : i32, blocks = 2 : i32 }
        blgp = bcast_second_32 : f32, f32, vector<32xf32>
    ```
  }];
  let assemblyFormat = [{
    custom<MNKDimensionList>($m, $n, $k) $sourceA `*` $sourceB `+` $destC
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
                   ConfinedAttr<I32Attr, [IntIsOneOf<[16]>]>:$m,
                   ConfinedAttr<I32Attr, [IntIsOneOf<[16]>]>:$n,
                   ConfinedAttr<I32Attr, [IntIsOneOf<[4, 16, 32, 64, 128]>]>:$k,
                   WMMAInTypes:$sourceA,
                   WMMAInTypes:$sourceB,
                   WMMAOutTypes:$destC,
                   DefaultValuedAttr<ConfinedAttr<I32Attr, [IntIsOneOf<[0, 1]>]>, "0">:$subwordOffset,
                   UnitAttr:$unsignedA,
                   UnitAttr:$unsignedB,
                   UnitAttr:$clamp)>,
    Results<(outs WMMAOutTypes: $destD)> {
  let summary = "MLIR wrapper for wmma instructions";
  let description = [{
    The `amdgpu.wmma` op is an MLIR wrapper around intrinsics for various `wmma`
    instructions in the AMDGPU architecture, which perform matrix multiplication.

    On gfx11/RDNA3, wmma intrinsics have M=N=K=16 dimensions.

    On gfx12/RDNA4, wmma intrinsics have M=N=16 dimensions and support K=16 for
    all element types, and K=32 for i4 sources.

    On gfx1250, wmma intrinsics have M=N=16 and K dimensions of 4, 32, 64, or 128,
    depending on the element types.

    On gfx11/RDNA3, emitting f16->f16 (or bf16->bf16) wmma the output is a 16xf16
    (or 16xbf16) vector containing only 8 valid values:
      - If `subwordOffset` is 0, then the output is stored at indices 0, 2, 4, ..., 14.
      - If `subwordOffset` is 1, then the output is stored at indices 1, 3, 5, ..., 15.
    On gfx12/RDNA4 and gfx1250, the result is instead returned as vector where all
    the values are valid and the `subwordOffset` must be `0`, as it cannot be used.

    `unsignedA` and `unsignedB` flag that the `int8` LLVM inputs are unsigned.

    The `clamp` flag is used to saturate the output of type T to `numeric_limits<T>::max()`
    in case of overflow.

    Example:
    ```mlir
      %0 = amdgpu.wmma 16x16x16 %matA * %matB + %matC : vector<8xf16>, vector<8xf16>, vector<8xf16>

      %1 = amdgpu.wmma 16x16x64 %matD * %matE + %matF : vector<32xi8>, vector<8xf32>, vector<8xf32>

      %2 = amdgpu.wmma 16x16x128 %matG * %matH + %matI : vector<64xf4E2M1FN>, vector<64xf4E2M1FN>, vector<8xf32>

      %3 = amdgpu.wmma 16x16x4 %matJ * %matK + %matL : vector<2xf32>, vector<2xf32>, vector<8xf32>
    ```
  }];
  let assemblyFormat = [{
    custom<MNKDimensionList>($m, $n, $k) $sourceA `*` $sourceB `+` $destC
    attr-dict
    `:` type($sourceA) `,` type($sourceB) `,` type($destC)
  }];
  let hasVerifier = 1;
}

def AMDGPU_SparseMFMAOp :
    AMDGPU_Op<"sparse_mfma", [AllTypesMatch<["destC", "destD"]>,
                              Pure]>,
    Arguments<(ins
                   ConfinedAttr<I32Attr, [IntIsOneOf<[16, 32]>]>:$m,
                   ConfinedAttr<I32Attr, [IntIsOneOf<[16, 32]>]>:$n,
                   ConfinedAttr<I32Attr, [IntIsOneOf<[16, 32, 64, 128]>]>:$k,
                   SMFMACSparseInTypes:$sourceA,
                   SMFMACDenseInTypes:$sourceB,
                   SMFMACOutTypes:$destC,
                   SMFMACIdxTypes:$sparseIdx,
                   DefaultValuedAttr<I32Attr, "0">:$cbsz,
                   DefaultValuedAttr<I32Attr, "0">:$abid)>,
    Results<(outs SMFMACOutTypes: $destD)> {
  let summary = "MLIR wrapper for CDNA sparse mfma (smfmac) instructions";
  let description = [{
    The `amdgpu.sparse_mfma` op is an MLIR wrapper around intrinsics for various
    `smfmac` instructions in the AMDGPU architecture, which perform matrix
    multiply-accumulate operations using 2:4 structured sparsity on matrix A
    with dense matrices B, C, and D.

    On gfx942, smfmac intrinsics support:
      - M=N=16, K=32 and M=N=32, K=16 for f16 and bf16 sources
      - M=N=16, K=64 and M=N=32, K=32 for i8 and fp8 sources

    On gfx950, smfmac intrinsics additionally support:
      - M=N=16, K=64 and M=N=32, K=32 for f16 and bf16 sources
      - M=N=16, K=128 and M=N=32, K=64 for i8 and fp8 sources

    The `sparseIdx` parameter contains packed indices identifying the positions
    of non-zero elements in the 2:4 sparse matrix A. For 16-bit source data,
    use `vector<4xi8>` (four 8-bit indices). For 8-bit source data, use
    `vector<2xi16>` (two 16-bit indices).

    The `cbsz` and `abid` parameters are repurposed to select the index set.
    If `cbsz == 0`, then `abid[1:0]` selects which index set to use.
    If `cbsz != 0`, then the very first is selected.

    Example:
    ```mlir
      %0 = amdgpu.sparse_mfma 16x16x32 %matA * %matB + %matC sparse(%idx : vector<4xi8>)
        : vector<4xf16>, vector<8xf16>, vector<4xf32>

      %1 = amdgpu.sparse_mfma 16x16x64 %matA * %matB + %matC sparse(%idx : vector<2xi16>)
        : vector<8xi8>, vector<16xi8>, vector<4xi32>

      %2 = amdgpu.sparse_mfma 16x16x64 %matA * %matB + %matC sparse(%idx : vector<2xi16>)
        { cbsz = 0 : i32, abid = 1 : i32 }
        : vector<8xf8E4M3FNUZ>, vector<16xf8E4M3FNUZ>, vector<4xf32>
    ```
  }];
  let assemblyFormat = [{
    custom<MNKDimensionList>($m, $n, $k) $sourceA `*` $sourceB `+` $destC
    `sparse` `(` $sparseIdx `:` type($sparseIdx) `)`
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
                   TypeAttr:$transferType,
                   UnitAttr:$async
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
    * If `$async` is set, the compiler will not attempt to infer the
      memory waits needed to ensure that the DMA operation has succeeded
      before a load that might access the stored-to LDS is performed.
      Instead, the `rocdl.asyncmark` and `rocdl.wait.asyncmark N`
      operations must be used to explicitly indicate the desired completion
      behavior. This enables more precise calculation of these waits at the
      cost of requiring user management of asynchrony.

    The `$dst`, along with its indices, points to the memory location the subgroup of this thread
    will write to.

    Note: only supported on gfx9 and gfx10.
  }];
  let assemblyFormat = [{
    (`async` $async^)? $src `[` $srcIndices `]` `,` $dst `[` $dstIndices `]` attr-dict `:` $transferType `,` type($src) `,` type($dst)
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
                   ConfinedAttr<I32Attr, [IntIsOneOf<[16, 32]>]>:$m,
                   ConfinedAttr<I32Attr, [IntIsOneOf<[16, 32]>]>:$n,
                   ConfinedAttr<I32Attr, [IntIsOneOf<[64, 128]>]>:$k,
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
    for various scaled versions of `mfma` instructions in the CDNA architecture, which
    perform multiple outer products in order to allow fast matrix multiplication.

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
      fp8 (f8E4M3FN and f8E5M2) types using either M=N=16, K=128 or M=N=32, K=64 as
      their tile size.
    - `amdgpu.scaled_mfma` does not support broadcasting. So, `cbsz`, `abid`, and `blgp`
      are omitted from this wrapper.
    - The `negateA`, `negateB`, and `negateC` flags in `amdgpu.mfma` are only supported
      for double-precision operations on gfx94x and so are not included here.

    Example:
    ```mlir
      %0 = amdgpu.scaled_mfma 32x32x64 (%arg0[0] * %arg1) * (%arg0[1] * %arg1) + %arg2
        : vector<4xf8E8M0FNU>, vector<32xf6E2M3FN>, f8E8M0FNU, vector<32xf6E2M3FN>, vector<16xf32>
    ```
  }];
  let assemblyFormat = [{
    custom<MNKDimensionList>($m, $n, $k) ` `
    `(` $scalesA `[` $scalesIdxA `]` `*` $sourceA `)` `*`
    `(` $scalesB `[` $scalesIdxB `]` `*` $sourceB `)` `+` $destC
    attr-dict
    `:` type($scalesA) `,` type($sourceA) `,` type($scalesB) `,` type($sourceB) `,` type($destC)
  }];
  let hasCanonicalizer = 1;
}

def AMDGPU_ScaledWMMAOp
    : AMDGPU_Op<"scaled_wmma", [AllTypesMatch<["destC", "destD"]>, Pure]>,
      Arguments<(ins ConfinedAttr<I32Attr, [IntIsOneOf<[16, 32]>]>:$m,
          ConfinedAttr<I32Attr, [IntIsOneOf<[16]>]>:$n,
          ConfinedAttr<I32Attr, [IntIsOneOf<[128]>]>:$k,
          ScaledWMMAInTypes:$sourceA, ScaledWMMAInTypes:$sourceB,
          ScaledWMMAOutTypes:$destC,
          VectorOfLengthAndType<[4, 8], [F8E8M0FNU, F8E4M3FN]>:$scaleA,
          ConfinedAttr<I32Attr, [IntIsOneOf<[0, 16]>]>:$a_first_scale_lane,
          VectorOfLengthAndType<[4, 8], [F8E8M0FNU, F8E4M3FN]>:$scaleB,
          ConfinedAttr<I32Attr, [IntIsOneOf<[0, 16]>]>:$b_first_scale_lane)>,
      Results<(outs ScaledWMMAOutTypes:$destD)> {
  // TODO: E5M3FNU scales are supported, but there is not yet MLIR support for
  // this datatype. Once we have support for that, update the scaleA and scaleB
  // types here.
  let summary = "MLIR wrapper for scaled wmma instructions";
  let description = [{
    The `amdgpu.scaled_wmma` op is an MLIR wrapper around intrinsics for scaled
    `wmma` instructions. These instructions perform matrix multiplication with
    per-block scaling of inputs, supporting fp4, fp6, and fp8 data formats.

    The scale instructions support a block size of 16 or 32 and two tile sizes:
    - 16x16x128 with mixed f8/f6/f4 formats (output: vector<8xf32>)
    - 32x16x128 with f4 format only (output: vector<16xf32>)

    Scale parameters (`scaleA`, `scaleB`) are small vectors of f8 scale values
    (either f8E8M0FNU, or f8E4M3FN) that are packed into i32/i64 values during
    lowering. Each lane can operate on 4 bytes (4 scale values), and the
    number of scales required for each matrix is determined by:
      num_scales_A = (M × K) / block_size
      num_scales_B = (N × K) / block_size

    The index attributes (`a_first_scale_lane`, `b_first_scale_lane`) select
    which lane to start reading scale values from (0 or 16):
    - For block size 32, 32 lanes across a single wave are used for the scale
    values. If the number of scales (num_scales_A or num_scales_B) can fit
    into half of the available lanes
    (i.e., num_scales / scales_per_lane == 16 (num_lanes)),
    then then first_scale_lane can be either 0 or 16. If all lanes are required
    for storing the scale values (num_scales / scales_per_lane == 32 (num_lanes)),
    then the first_scale_lane must be 0.
    - For block size 16, the same rules apply as above except that there are 64
    lanes across two waves that are used for the scale values. When
    num_scales / scales_per_lane == 32 (num lanes), then 16 lanes from each wave are used.
    first_scale_lane of 0 or 16 will decide which lanes are used for this. When
    num_scales / scales_per_lane == 64 (num_lanes), then first_scale_lane must
    be set to 0.

    Example:
    ```mlir
      // 16x16x128: fp8 inputs
      %0 = amdgpu.scaled_wmma 16x16x128 (%scaleVecA * %matA) * (%scaleVecB * %matB) + %matC
        {a_first_scale_lane = 0 : i32, b_first_scale_lane = 0 : i32}
        : vector<4xf8E8M0FNU>, vector<64xf8E4M3FN>,
        vector<4xf8E8M0FNU>, vector<64xf8E4M3FN>, vector<8xf32>

      // 32x16x128: fp4 inputs with different scale lanes
      %1 = amdgpu.scaled_wmma 32x16x128 (%scaleVecD * %matD) * (%scaleVecE * %matE) + %matF
        {a_first_scale_lane = 0 : i32, b_first_scale_lane = 16 : i32}
        : vector<8xf8E4M3FN>, vector<128xf4E2M1FN>,
        vector<8xf8E4M3FN>, vector<64xf4E2M1FN>, vector<16xf32>
    ```
  }];
  let assemblyFormat = [{
    custom<MNKDimensionList>($m, $n, $k) ` `
    `(` $scaleA `*` $sourceA `)` `*`
    `(` $scaleB `*` $sourceB `)` `+` $destC
    attr-dict
    `:` type($scaleA) `,` type($sourceA) `,` type($scaleB) `,` type($sourceB) `,` type($destC)
  }];
  let hasVerifier = 1;
}

class AMDGPU_DmaBaseOp<string mnemonic, Type outType> :
    AMDGPU_Op<mnemonic, [Pure, AttrSizedOperandSegments, AllElementTypesMatch<["global", "lds"]>]>,
    Arguments<(ins Arg<AnyMemRef>:$global,
                   Variadic<Index>:$global_indices,
                   Arg<AnyMemRef>:$lds,
                   Variadic<Index>:$lds_indices)>,
    Results<(outs outType: $base)> {

  // TODO:
  // * Add verifiers to make sure that the number of indices do not exceed the number of dimensions.

  let assemblyFormat = [{
    $global `[` $global_indices `]` `,` $lds `[` $lds_indices `]` attr-dict `:` type($global) `,` type($lds) `->` type(results)
  }];
}

def AMDGPU_MakeGatherDmaBaseOp : AMDGPU_DmaBaseOp<"make_gather_dma_base", AMDGPU_TDMGatherBaseType> {
  let summary = "Pair of based addresses used when moving tiles between LDS and global memory.";

  let description = [{
    This operation creates a pair of addresses that will be used by `tensor_load_to_lds`
    and `tensor_store_from_lds`.

    This operation creates a value corresponding to the tensor descriptor (D#) group 0
    found in TensorLoadToLDSOp and TensorStoreFromLDSOp in the rocdl dialect.

    Unlike `make_dma_base`, this operation returns `!amdgpu.tdm_gather_base<$element_type, $index_type>`
    which is only compatible with `make_gather_dma_descriptor`. Using the descriptor returned
    by `make_gather_dma_descriptor` will set the `tensor_load_to_lds` and `tensor_store_from_lds` to gather mode.

    ```mlir
      %base = amdgpu.make_gather_dma_base %global[%idx0, %idx1], %lds[%idx2, %idx3] : memref<64x64xi32>, memref<64x64xi32, #gpu.address_space<workgroup>> -> !amdgpu.tdm_gather_base<i32, i16>
      // %indices : i16
      %descriptor = amdgpu.make_gather_dma_descriptor %base[%indices] globalSize [2, 2] globalStride [2, 1] sharedSize [2, 2] : !amdgpu.tdm_gather_base<i32, i16>, i16 -> !amdgpu.tdm_descriptor
      amdgpu.tensor_load_to_lds %descriptor : !amdgpu.tdm_descriptor
    ```
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    static constexpr bool isGather() {
      return true;
    }
  }];
}


def AMDGPU_MakeDmaBaseOp : AMDGPU_DmaBaseOp<"make_dma_base", AMDGPU_TDMBaseType> {

  let summary = "Pair of based addresses used when moving tiles between LDS and global memory.";
  let description = [{
    This operation creates a pair of addresses that will be used by tensor_load_to_lds
    and tensor_store_from_lds.

    This operation creates a value corresponding to the tensor descriptor (D#) group 0
    found in TensorLoadToLDSOp and TensorStoreFromLDSOp in the rocdl dialect.

    For example:

    ```mlir
      %base = amdgpu.make_dma_base %global[%idx0, %idx1], %lds[%idx2, %idx3] : memref<64x64xi32>, memref<64x64xi32, #gpu.address_space<workgroup>> -> !amdgpu.tdm_base<i32>
      %descriptor = amdgpu.make_dma_descriptor %base globalSize [2, 2] globalStride [2, 1] sharedSize [2, 2] : !amdgpu.tdm_base<i32> -> !amdgpu.tdm_descriptor
      amdgpu.tensor_load_to_lds %descriptor : !amdgpu.tdm_descriptor
    ```

    to

    ```mlir
      // pseudo-code
      %global_base = llvm.extractvalue %global_memref[1]
      %global_address = llvm.get_element_ptr ...

      %lds_base = llvm.extractvalue %lds_memref[1]
      %lds_address = llvm.get_element_ptr ...

      // Definition of %base
      %undef = llvm.mlir.undef : vector<4xi32>
      %v0 = llvm.insertelement %15, %undef[0] : vector<4xi32>
      %v1 = llvm.insertelement %lds_address, %v0[1] : vector<4xi32>
      %v2 = llvm.insertelement %global_address_low, %v1[2] : vector<4xi32>
      %base = llvm.insertelement %global_address_high, %v2[3] : vector<4xi32>

      rocdl.tensor.load.to.lds %base, %dgroup1, %dgroup2, %dgroup3 cachepolicy 0 : vector<4xi32>, vector<8xi32>
    ```

    These tensor DMA operations were introduced in gfx1250.
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    static constexpr bool isGather() {
      return false;
    }
  }];
}

class AMDGPU_MakeDescriptorOp<string mnemonic> :
  AMDGPU_Op<mnemonic, [Pure, AttrSizedOperandSegments]>,
  Results<(outs AMDGPU_TDMDescriptorType: $desc)> {

  dag baseArgs = (ins
    Variadic<Index>: $global_dynamic_sizes,
    DenseI64ArrayAttr: $global_static_sizes,
    Variadic<Index>: $global_dynamic_strides,
    DenseI64ArrayAttr: $global_static_strides,
    Variadic<Index>: $shared_dynamic_sizes,
    DenseI64ArrayAttr: $shared_static_sizes,
    Optional<AMDGPU_ConcreteVector<I1, 16>>: $workgroup_mask,
    Optional<I1>: $early_timeout,
    Optional<I32>: $pad_amount,
    Optional<I32>: $pad_interval,
    Optional<MemRefOf<[AMDGPU_DsBarrierStateType]>>: $atomic_barrier_address,
    Variadic<Index>: $atomic_barrier_indices,
    Optional<Index>: $global_increment,
    Optional<I32>: $lds_increment,
    Optional<Index>: $iteration_count);

  code extraClassDeclarationBase = [{
    int64_t getRank() {
      return getGlobalStaticSizes().size();
    }

    unsigned getElementTypeWidth() {
      return getBase().getType().getElementType().getIntOrFloatBitWidth();
    }

    SmallVector<OpFoldResult> getMixedGlobalSizes() {
      return getMixedValues(getGlobalStaticSizes(), getGlobalDynamicSizes(), getContext());
    }

    SmallVector<OpFoldResult> getMixedGlobalStrides() {
      return getMixedValues(getGlobalStaticStrides(), getGlobalDynamicStrides(), getContext());
    }

    SmallVector<OpFoldResult> getMixedSharedSizes() {
      return getMixedValues(getSharedStaticSizes(), getSharedDynamicSizes(), getContext());
    }

  }];

}

def AMDGPU_MakeGatherDmaDescriptorOp : AMDGPU_MakeDescriptorOp<"make_gather_dma_descriptor"> {
  dag args = (ins AMDGPU_TDMGatherBaseType: $base,
                  AnyTypeOf<[VectorOfMinMaxLengthAndType<1, 8, [I32]>,
                             VectorOfMinMaxLengthAndType<1, 16, [I16]>]>: $indices);
  let arguments = !con(args, baseArgs);
  let summary = "Make all descriptor groups needed by TensorLoadToLDS/TensorStoreFromLDS.";

  let assemblyFormat = [{
    $base `[` $indices `]`
    `globalSize` custom<DynamicIndexList>($global_dynamic_sizes, $global_static_sizes)
    `globalStride` custom<DynamicIndexList>($global_dynamic_strides, $global_static_strides)
    `sharedSize` custom<DynamicIndexList>($shared_dynamic_sizes, $shared_static_sizes)
    ( `padShared` `(` $pad_amount^ `every` $pad_interval `)` )?
    ( `workgroupMask` $workgroup_mask^ ( `earlyTimeout` $early_timeout^)?)?
    ( `atomicBarrier` `(` $atomic_barrier_address^ `[` $atomic_barrier_indices `]`
                      `:` type($atomic_barrier_address) `)`)?
    ( `iterate` $global_increment^ `,` $lds_increment `,` $iteration_count )?
    attr-dict `:` qualified(type($base)) `,` type($indices) `->` type(results)
  }];

  let hasVerifier = 1;
  let hasFolder = 1;

  let extraClassDeclaration = extraClassDeclarationBase # [{
    static constexpr bool isGather() {
      return true;
    }
  }];
}

def AMDGPU_MakeDmaDescriptorOp : AMDGPU_MakeDescriptorOp<"make_dma_descriptor"> {
  dag args = (ins AMDGPU_TDMBaseType: $base);
  let arguments = !con(args, baseArgs);
  let summary = "Make all descriptor groups needed by TensorLoadToLDS/TensorStoreFromLDS.";
  let description = [{
     Make all descriptor groups needed by tensor memory operations.

     The $base operand corresponds to the base pair addresses, one must be an address in LDS
     while the other must be a global memory location.

     $global_{static/dynamic}_sizes determine the size of the tensor.
     $global_{static/dynamic}_strides determine the strides of the tensor.
     $shared_{static/dynamic}_sizes determines the size of the tile.

     $workgroup_mask broadcast load to workgroups inside of a workgroup cluster
     (0 = do not broadcast result to workgroup, 1 = broadcast result to workgroup). Ignored for stores.
     An all zeros mask is interpreted as a non-broadcasted load.

     $early_timeout return data to requesters as soon as cache supplies it.

     Padding can be applied to the LDS address when copying from memory to LDS,
     but not when copying from LDS to memory.
     The values in the padded target addresses remain the same as before the operation was applied.
     $pad_interval must be a power of two contained in [2, 256].
     $pad_amount must be a value contained in [1, 128].

     If an atomic barrier is provided, it will be arrived at **once** after
     each load/store using this descriptor is completed.

     2D and 3D tensors may be iterated over by setting $global_increment, $lds_increment, and $iteration_count.
     $global_increment determines how much to increment the starting global memory address per iteration in units of the $base's element type.
     $lds_increment determines how much to increment the starting LDS address per iteration in units of the $base's element type.
     $iterate_count determines how many times to iterate, it must be a value in the inclusive interval [1, 256].

     ```mlir
      // Example of moving a two-dimensional tensor to LDS.
      %base = amdgpu.make_dma_base %global[0, 0], %lds[0, 0] : memref<64x64xi32>, memref<64x64xi32, #gpu.address_space<workgroup>> -> !amdgpu.tdm_base<i32>
      %descriptor = amdgpu.make_dma_descriptor %base globalSize [64, 64] globalStride [64, 1] sharedSize [64, 64] : !amdgpu.tdm_base<i32> -> !amdgpu.tdm_descriptor
      amdgpu.tensor_load_to_lds %descriptor : !amdgpu.tdm_descriptor

      // Example of moving a two dimension tensor to LDS where padding is applied after every integer.
      %base = amdgpu.make_dma_base %global[0, 0], %lds[0, 0] : memref<32x32xi32>, memref<64x64xi32, #gpu.address_space<workgroup>> -> !amdgpu.tdm_base<i32>
      %descriptor = amdgpu.make_dma_descriptor %base globalSize [32, 32] globalStride [32, 1] sharedSize [64, 64] padShared(%pad_amount every %pad_interval) : !amdgpu.tdm_base<i32> -> !amdgpu.tdm_descriptor
      amdgpu.tensor_load_to_lds %descriptor : !amdgpu.tdm_descriptor
     ```
  }];

  let assemblyFormat = [{
    $base
    `globalSize` custom<DynamicIndexList>($global_dynamic_sizes, $global_static_sizes)
    `globalStride` custom<DynamicIndexList>($global_dynamic_strides, $global_static_strides)
    `sharedSize` custom<DynamicIndexList>($shared_dynamic_sizes, $shared_static_sizes)
    ( `padShared` `(` $pad_amount^ `every` $pad_interval `)` )?
    ( `workgroupMask` $workgroup_mask^ ( `earlyTimeout` $early_timeout^)?)?
    ( `atomicBarrier` `(` $atomic_barrier_address^ `[` $atomic_barrier_indices `]`
                      `:` type($atomic_barrier_address) `)`)?
    ( `iterate` $global_increment^ `,` $lds_increment `,` $iteration_count )?
    attr-dict `:` qualified(type($base)) `->` type(results)
  }];

  let hasVerifier = 1;
  let hasFolder = 1;

  let extraClassDeclaration = extraClassDeclarationBase # [{
    static constexpr bool isGather() {
      return false;
    }
  }];

}

def AMDGPU_TensorLoadToLDSOp :
  AMDGPU_Op<"tensor_load_to_lds", [MemoryEffects<[MemWrite, MemRead]>]>,
  Arguments<(ins AMDGPU_TDMDescriptorType: $desc)> {
  let summary = "Load tensors from global memory to LDS.";
  let description = [{
    Load tensors of up to five dimensions from global memory to LDS.

    This operation was introduced in gfx1250.
  }];

  let assemblyFormat = [{
    $desc attr-dict `:` qualified(type($desc))
  }];
}

def AMDGPU_TensorStoreFromLDSOp :
  AMDGPU_Op<"tensor_store_from_lds", [MemoryEffects<[MemWrite, MemRead]>]>,
  Arguments<(ins AMDGPU_TDMDescriptorType: $desc)> {

  let summary = "Store tensors from LDS to global memory.";
  let description = [{
    Store tensors of up to five dimensions from LDS to global memory.

    This operation was introduced in gfx1250.
  }];

  let assemblyFormat = [{
    $desc attr-dict `:` qualified(type($desc))
  }];
}

//===----------------------------------------------------------------------===//
// In-LDS Barrier Operations
//
// General note: any of these operations that impact memory have read and write
// effects as a crude model of their atomic nature - we don't want "reads"
// being hoisted out of loops.
//===----------------------------------------------------------------------===//

def AMDGPU_DsBarrierInitOp :
    AMDGPU_Op<"ds_barrier_init">,
    Arguments<(ins Arg<MemRefOf<[AMDGPU_DsBarrierStateType]>, "barrier(s)",
                       [MemRead, MemWrite]>:$base,
                   Variadic<Index>:$indices,
                   I32:$participants)> {
  let summary = "Initialize an in-LDS barrier.";
  let description = [{
    Given the location `!amdgpu.ds_barrier_state` in LDS (as specified by `base` and `indices`),
    initialize the barrier structure so that the pending and init counts are equal to
    `participants - 1`, which will have its high bits masked off, and its phase is equal to 0.

    Note that we subtract 1 from `participants` when constructing the barrier state
    to provide clearer high-level semantics.

    The subtraction means that, when the `participant`th arrival occurs, the phase will change.
    In practical terms, this means that you can use (for example) the number of subgroups or
    waves per workgroup as `participants`, instead of manually needing to remove one.

    While the write of the initial state will be performed atomically, no synchronization
    between waves will be performed by this operation.

    Example:
    ```mlir
    amdgpu.ds_barrier_init %barrier[], %c32 : memref<!amdgpu.ds_barrier_state, #gpu.address_space<workgroup>>, i32
    ```

    This operation is only available on gfx1250+.
  }];

  let assemblyFormat = [{
    $base `[` $indices `]` `,` $participants attr-dict `:` type($base) `,` type($participants)
  }];

  let hasVerifier = 1;
}

def AMDGPU_DsBarrierPollStateOp :
    AMDGPU_Op<"ds_barrier_poll_state">,
    Arguments<(ins Arg<MemRefOf<[AMDGPU_DsBarrierStateType]>, "barrier(s)",
                       [MemRead, MemWrite]>:$base,
                 Variadic<Index>:$indices)>,
    Results<(outs AMDGPU_DsBarrierStateType:$out)> {
  let summary = "Atomically read the state of an in-LDS barrier.";
  let description = [{
    Atomically read and return the state of the barrier at `base[indices...]`.

    This will ultimately act like a `memref.load`, but this operation will ensure
    that appropriate atomic orderings and syncscopes are set.

    Example:
    ```mlir
    %state = amdgpu.ds_barrier_poll_state %barrier[] : memref<!amdgpu.ds_barrier_state, #gpu.address_space<workgroup>> -> !amdgpu.ds_barrier_state
    ```

    This operation is only available on gfx1250+.
  }];

  let assemblyFormat = [{
    $base `[` $indices `]` attr-dict `:` type($base) `->` type($out)
  }];

  let hasVerifier = 1;
}

def AMDGPU_DsAsyncBarrierArriveOp :
    AMDGPU_Op<"ds_async_barrier_arrive">,
    Arguments<(ins Arg<MemRefOf<[AMDGPU_DsBarrierStateType]>, "barrier(s)",
                       [MemRead, MemWrite]>:$base,
                 Variadic<Index>:$indices)> {
  let summary = "Asynchronously arrive at an in-LDS barrier.";
  let description = [{
    Add a arrival to the LDS barrier at `base[indices]` to the sequence of pending
    asynchronous memory operations.

    This will add an "asynchronous memory operation" to the in-order list of pending
    asynchronous loads from global memory to LDS. When the queue of such operations
    issued before this operation is complete, the specified barrier will be arrived at,
    decrementing the pending count by 1 **per lane that executes it** and rolling
    over the phase if applicable.

    This operation does not return the old barrier state.

    Example:
    ```mlir
    amdgpu.ds_async_barrier_arrive %barrier[] : memref<!amdgpu.ds_barrier_state, #gpu.address_space<workgroup>>
    ```

    This operation is only available on gfx1250+.
  }];

  let assemblyFormat = [{
    $base `[` $indices `]` attr-dict `:` type($base)
  }];

  let hasVerifier = 1;
}

def AMDGPU_DsBarrierArriveOp :
    AMDGPU_Op<"ds_barrier_arrive">,
    Arguments<(ins Arg<MemRefOf<[AMDGPU_DsBarrierStateType]>, "barrier(s)",
                       [MemRead, MemWrite]>:$base,
                 Variadic<Index>:$indices,
                 I64:$count)>,
    Results<(outs AMDGPU_DsBarrierStateType:$out)> {
  let summary = "Arrive at an in-LDS barrier and return old state.";
  let description = [{
    Atomically arrive at the LDS barrier at `base[indices]` and decrement it by `count`,
    rolling over the phase if needed and returning the old barrier state.

    `count` is the number of participants that should be subtracted from the barrier's
    pending count **per lane that executes the operation**.

    Example:
    ```mlir
    %old_state = amdgpu.ds_barrier_arrive %barrier[], %c1 : memref<!amdgpu.ds_barrier_state, #gpu.address_space<workgroup>>, i64 -> !amdgpu.ds_barrier_state
    ```

    This operation is only available on gfx1250+.
  }];

  let assemblyFormat = [{
    $base `[` $indices `]` `,` $count attr-dict `:` type($base) `,` type($count) `->` type($out)
  }];

  let hasVerifier = 1;
}

def AMDGPU_DsBarrierStatePhaseOp :
    AMDGPU_Op<"ds_barrier_state_phase", [Pure]>,
    Arguments<(ins AMDGPU_DsBarrierStateType:$state)>,
    Results<(outs I32:$res)> {
  let summary = "Extract the phase of a barrier state.";
  let description = [{
    Extract the phase of the `!amdgpu.ds_barrier_state` `state` as a 32-bit value.

    Example:
    ```mlir
    %phase = amdgpu.ds_barrier_state_phase %state : !amdgpu.ds_barrier_state -> i32
    ```
  }];

  let assemblyFormat = [{
    $state attr-dict `:` type($state) `->` type($res)
  }];
}

def AMDGPU_DsBarrierStatePendingCountOp :
    AMDGPU_Op<"ds_barrier_state_pending_count", [Pure]>,
    Arguments<(ins AMDGPU_DsBarrierStateType:$state)>,
    Results<(outs I32:$res)> {
  let summary = "Extract the pending count of a barrier state.";
  let description = [{
    Extract the pending count of the `!amdgpu.ds_barrier_state` `state` as a 32-bit value.

    Example:
    ```mlir
    %pending = amdgpu.ds_barrier_state_pending_count %state : !amdgpu.ds_barrier_state -> i32
    ```
  }];

  let assemblyFormat = [{
    $state attr-dict `:` type($state) `->` type($res)
  }];
}

def AMDGPU_DsBarrierStateInitCountOp :
    AMDGPU_Op<"ds_barrier_state_init_count", [Pure]>,
    Arguments<(ins AMDGPU_DsBarrierStateType:$state)>,
    Results<(outs I32:$res)> {
  let summary = "Extract the init count of a barrier state.";
  let description = [{
    Extract the init count of the `!amdgpu.ds_barrier_state` `state` as a 32-bit value.

    Example:
    ```mlir
    %init = amdgpu.ds_barrier_state_init_count %state : !amdgpu.ds_barrier_state -> i32
    ```
  }];

  let assemblyFormat = [{
    $state attr-dict `:` type($state) `->` type($res)
  }];
}

def AMDGPU_DsBarrierStatePhaseParity :
    AMDGPU_Op<"ds_barrier_state_phase_parity", [Pure]>,
    Arguments<(ins AMDGPU_DsBarrierStateType:$state)>,
    Results<(outs I1:$res)> {
  let summary = "Extract the phase parity of a barrier state.";
  let description = [{
    Return the parity of the phase of the `!amdgpu.ds_barrier_state` `state`.

    This is intended to simplify the case where the barrier is being used to repeatedly
    track completion of a task where the precise value of the phase won't mater, only that
    it changed since (or as a result of) the arrival.

    Example:
    ```mlir
    %parity = amdgpu.ds_barrier_state_phase_parity %state : !amdgpu.ds_barrier_state -> i1
    ```
  }];

  let assemblyFormat = [{
    $state attr-dict `:` type($state) `->` type($res)
  }];
}

#endif // MLIR_DIALECT_AMDGPU_IR_AMDGPUOPS_TD
