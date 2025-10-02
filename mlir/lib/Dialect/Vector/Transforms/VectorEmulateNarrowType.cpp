//===- VectorEmulateNarrowType.cpp - Narrow type emulation ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements target-independent rewrites and utilities to emulate
// narrow types that are not supported by the target hardware, e.g. i4
// ("emulated type"), using wider types, e.g. i8 ("container type").
//
/// Currently, only power-of-two integer types are supported. These are
/// converted to wider integers that are either 8 bits wide or wider.
///
/// TODO: Support for non-powers-of-two.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/NarrowTypeEmulationConverter.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <optional>

#include "mlir/Dialect/MemRef/Transforms/Transforms.h"

using namespace mlir;

#define DEBUG_TYPE "vector-narrow-type-emulation"

using VectorValue = TypedValue<VectorType>;
using MemRefValue = TypedValue<MemRefType>;

//===----------------------------------------------------------------------===//
// Utils
//===----------------------------------------------------------------------===//

/// Returns a compressed mask for the emulated vector. For example, when
/// emulating an eight-element `i8` vector with `i32` (i.e. when the source
/// elements span two dest elements), this method compresses `vector<8xi1>`
/// into `vector<2xi1>`.
///
/// The compressed/output mask value is set iff any mask in the corresponding
/// `numSrcElemsPerDest` range of uncompressed/input masks is set. E.g., if
/// `numSrcElemsPerDest` equals to 2, and `numFrontPadElems` equals to 1, the
/// following mask:
///
///   %mask = [1, 1, 0, 0, 0, 0]
///
/// will first be padded in the front with `numFrontPadElems` zeros, and zeros
/// will be added in the back to make the number of elements a multiple of
/// `numSrcElemsPerDest` (for easier computation). The resulting mask will be:
///
///   %mask = [0, 1, 1, 0, 0, 0, 0, 0]
///
/// then it will return the following new compressed mask:
///
///   %mask = [1, 1, 0, 0]
///
/// NOTE: `numFrontPadElems` is assumed to be strictly smaller than
/// `numSrcElemsPerDest`.
static FailureOr<Operation *> getCompressedMaskOp(OpBuilder &rewriter,
                                                  Location loc, Value mask,
                                                  int numSrcElems,
                                                  int numSrcElemsPerDest,
                                                  int numFrontPadElems = 0) {

  assert(numFrontPadElems < numSrcElemsPerDest &&
         "numFrontPadElems must be less than numSrcElemsPerDest");

  auto numDestElems =
      (numFrontPadElems + numSrcElems + numSrcElemsPerDest - 1) /
      numSrcElemsPerDest;

  Operation *maskOp = mask.getDefiningOp();
  SmallVector<vector::ExtractOp, 2> extractOps;
  // TODO: add support to `vector.splat`.
  // Finding the mask creation operation.
  while (maskOp &&
         !isa<arith::ConstantOp, vector::CreateMaskOp, vector::ConstantMaskOp>(
             maskOp)) {
    if (auto extractOp = dyn_cast<vector::ExtractOp>(maskOp)) {
      maskOp = extractOp.getSource().getDefiningOp();
      extractOps.push_back(extractOp);
    }
  }

  if (!isa<arith::ConstantOp, vector::CreateMaskOp, vector::ConstantMaskOp>(
          maskOp))
    return failure();

  // Computing the "compressed" mask. All the emulation logic (i.e. computing
  // new mask index) only happens on the last dimension of the vectors.
  SmallVector<int64_t> maskShape(
      cast<VectorType>(maskOp->getResultTypes()[0]).getShape());
  maskShape.back() = numDestElems;
  auto newMaskType = VectorType::get(maskShape, rewriter.getI1Type());
  std::optional<Operation *> newMask =
      TypeSwitch<Operation *, std::optional<Operation *>>(maskOp)
          .Case<vector::CreateMaskOp>(
              [&](auto createMaskOp) -> std::optional<Operation *> {
                OperandRange maskOperands = createMaskOp.getOperands();
                // The `vector.create_mask` op creates a mask arrangement
                // without any zeros at the front. Also, because
                // `numFrontPadElems` is strictly smaller than
                // `numSrcElemsPerDest`, the compressed mask generated by
                // padding the original mask by `numFrontPadElems` will not
                // have any zeros at the front as well.
                AffineExpr s0;
                bindSymbols(rewriter.getContext(), s0);
                s0 = (s0 + numFrontPadElems).ceilDiv(numSrcElemsPerDest);
                OpFoldResult origIndex = getAsOpFoldResult(maskOperands.back());
                OpFoldResult maskIndex = affine::makeComposedFoldedAffineApply(
                    rewriter, loc, s0, origIndex);
                SmallVector<Value> newMaskOperands(maskOperands.drop_back());
                newMaskOperands.push_back(
                    getValueOrCreateConstantIndexOp(rewriter, loc, maskIndex));
                return vector::CreateMaskOp::create(rewriter, loc, newMaskType,
                                                    newMaskOperands);
              })
          .Case<vector::ConstantMaskOp>([&](auto constantMaskOp)
                                            -> std::optional<Operation *> {
            // Take the shape of mask, compress its trailing dimension:
            SmallVector<int64_t> maskDimSizes(constantMaskOp.getMaskDimSizes());
            int64_t &maskIndex = maskDimSizes.back();
            maskIndex = llvm::divideCeil(numFrontPadElems + maskIndex,
                                         numSrcElemsPerDest);
            return vector::ConstantMaskOp::create(rewriter, loc, newMaskType,
                                                  maskDimSizes);
          })
          .Case<arith::ConstantOp>([&](auto constantOp)
                                       -> std::optional<Operation *> {
            // TODO: Support multiple dimensions.
            if (maskShape.size() != 1)
              return std::nullopt;
            // Rearrange the original mask values to cover the whole potential
            // loading region. For example, in the case of using byte-size for
            // emulation, given the following mask:
            //
            // %mask = [0, 1, 0, 1, 0, 0]
            //
            // With front offset of 1, the mask will be padded 0s in the front
            // and back so that:
            // 1. It is aligned with the effective loading bits
            // 2. Its length is multiple of `numSrcElemPerDest` (and the total
            // coverage size is mulitiple of bytes). The new mask will be like
            // this before compressing:
            //
            // %new_mask = [0, 0, 1, 0, 1, 0, 0, 0]
            auto originalMask =
                cast<DenseIntElementsAttr>(constantOp.getValue());
            SmallVector<bool> paddedMaskValues(numFrontPadElems, false);
            paddedMaskValues.append(originalMask.template value_begin<bool>(),
                                    originalMask.template value_end<bool>());
            paddedMaskValues.resize(numDestElems * numSrcElemsPerDest, false);

            // Compressing by combining every `numSrcElemsPerDest` elements:
            SmallVector<bool> compressedMaskValues;
            for (size_t i = 0; i < paddedMaskValues.size();
                 i += numSrcElemsPerDest) {
              bool combinedValue = false;
              for (int j = 0; j < numSrcElemsPerDest; ++j) {
                combinedValue |= paddedMaskValues[i + j];
              }
              compressedMaskValues.push_back(combinedValue);
            }
            return arith::ConstantOp::create(
                rewriter, loc,
                DenseElementsAttr::get(newMaskType, compressedMaskValues));
          });

  if (!newMask)
    return failure();

  while (!extractOps.empty()) {
    newMask =
        vector::ExtractOp::create(rewriter, loc, (*newMask)->getResults()[0],
                                  extractOps.back().getMixedPosition());
    extractOps.pop_back();
  }

  return *newMask;
}

/// Extracts 1-D subvector from a 1-D vector.
///
/// Given the input rank-1 source vector, extracts `numElemsToExtract` elements
/// from `src`, starting at `offset`. The result is also a rank-1 vector:
///
///   vector<numElemsToExtract x !elemType>
///
/// (`!elType` is the element type of the source vector). As `offset` is a known
/// _static_ value, this helper hook emits `vector.extract_strided_slice`.
///
/// EXAMPLE:
///     %res = vector.extract_strided_slice %src
///       { offsets = [offset], sizes = [numElemsToExtract], strides = [1] }
static Value staticallyExtractSubvector(OpBuilder &rewriter, Location loc,
                                        Value src, int64_t offset,
                                        int64_t numElemsToExtract) {
  auto vectorType = cast<VectorType>(src.getType());
  assert(vectorType.getRank() == 1 && "expected source to be rank-1-D vector ");
  assert(offset + numElemsToExtract <= vectorType.getNumElements() &&
         "subvector out of bounds");

  // When extracting all available elements, just use the source vector as the
  // result.
  if (vectorType.getNumElements() == numElemsToExtract)
    return src;

  auto offsets = rewriter.getI64ArrayAttr({offset});
  auto sizes = rewriter.getI64ArrayAttr({numElemsToExtract});
  auto strides = rewriter.getI64ArrayAttr({1});

  auto resultVectorType =
      VectorType::get({numElemsToExtract}, vectorType.getElementType());
  return vector::ExtractStridedSliceOp::create(rewriter, loc, resultVectorType,
                                               src, offsets, sizes, strides)
      ->getResult(0);
}

/// Inserts 1-D subvector into a 1-D vector.
///
/// Inserts the input rank-1 source vector into the destination vector starting
/// at `offset`. As `offset` is a known _static_ value, this helper hook emits
/// `vector.insert_strided_slice`.
///
/// EXAMPLE:
///   %res = vector.insert_strided_slice %src, %dest
///     {offsets = [%offset], strides [1]}
static Value staticallyInsertSubvector(OpBuilder &rewriter, Location loc,
                                       Value src, Value dest, int64_t offset) {
  [[maybe_unused]] auto srcVecTy = cast<VectorType>(src.getType());
  [[maybe_unused]] auto destVecTy = cast<VectorType>(dest.getType());
  assert(srcVecTy.getRank() == 1 && destVecTy.getRank() == 1 &&
         "expected source and dest to be rank-1 vector types");

  // If overwritting the destination vector, just return the source.
  if (srcVecTy.getNumElements() == destVecTy.getNumElements() && offset == 0)
    return src;

  auto offsets = rewriter.getI64ArrayAttr({offset});
  auto strides = rewriter.getI64ArrayAttr({1});
  return vector::InsertStridedSliceOp::create(rewriter, loc, destVecTy, src,
                                              dest, offsets, strides);
}

/// Extracts 1-D subvector from a 1-D vector.
///
/// Given the input rank-1 source vector, extracts `numElemsToExtact` elements
/// from `src`, starting at `offset`. The result is also a rank-1 vector:
///
///   vector<numElemsToExtact x !elType>
///
/// (`!elType` is the element type of the source vector). As `offset` is assumed
/// to be a _dynamic_ SSA value, this helper method generates a sequence of
/// `vector.extract` + `vector.insert` pairs.
///
/// EXAMPLE:
///     %v1 = vector.extract %src[%offset] : i2 from vector<8xi2>
///     %r1 = vector.insert %v1, %dest[0] : i2 into vector<3xi2>
///     %c1 = arith.constant 1 : index
///     %idx2 = arith.addi %offset, %c1 : index
///     %v2 = vector.extract %src[%idx2] : i2 from vector<8xi2>
///     %r2 = vector.insert %v2, %r1 [1] : i2 into vector<3xi2>
///     (...)
static Value dynamicallyExtractSubVector(OpBuilder &rewriter, Location loc,
                                         Value src, Value dest,
                                         OpFoldResult offset,
                                         int64_t numElemsToExtract) {
  auto srcVecTy = cast<VectorType>(src.getType());
  assert(srcVecTy.getRank() == 1 && "expected source to be rank-1-D vector ");
  // NOTE: We are unable to take the offset into account in the following
  // assert, hence its still possible that the subvector is out-of-bounds even
  // if the condition is true.
  assert(numElemsToExtract <= srcVecTy.getNumElements() &&
         "subvector out of bounds");

  // When extracting all available elements, just use the source vector as the
  // result.
  if (srcVecTy.getNumElements() == numElemsToExtract)
    return src;

  for (int i = 0; i < numElemsToExtract; ++i) {
    Value extractLoc =
        (i == 0) ? dyn_cast<Value>(offset)
                 : arith::AddIOp::create(
                       rewriter, loc, rewriter.getIndexType(),
                       dyn_cast<Value>(offset),
                       arith::ConstantIndexOp::create(rewriter, loc, i));
    auto extractOp = vector::ExtractOp::create(rewriter, loc, src, extractLoc);
    dest = vector::InsertOp::create(rewriter, loc, extractOp, dest, i);
  }
  return dest;
}

/// Inserts 1-D subvector into a 1-D vector.
///
/// Inserts the input rank-1 source vector into the destination vector starting
/// at `offset`. As `offset` is assumed to be a _dynamic_ SSA value, this hook
/// uses a sequence of `vector.extract` + `vector.insert` pairs.
///
/// EXAMPLE:
///     %v1 = vector.extract %src[0] : i2 from vector<8xi2>
///     %r1 = vector.insert %v1, %dest[%offset] : i2 into vector<3xi2>
///     %c1 = arith.constant 1 : index
///     %idx2 = arith.addi %offset, %c1 : index
///     %v2 = vector.extract %src[1] : i2 from vector<8xi2>
///     %r2 = vector.insert %v2, %r1 [%idx2] : i2 into vector<3xi2>
///     (...)
static Value dynamicallyInsertSubVector(RewriterBase &rewriter, Location loc,
                                        Value src, Value dest,
                                        OpFoldResult offset,
                                        int64_t numElemsToInsert) {
  auto srcVecTy = cast<VectorType>(src.getType());
  auto destVecTy = cast<VectorType>(dest.getType());
  assert(srcVecTy.getRank() == 1 && destVecTy.getRank() == 1 &&
         "expected source and dest to be rank-1 vector types");
  (void)srcVecTy;
  (void)destVecTy;
  assert(numElemsToInsert > 0 &&
         "the number of elements to insert must be greater than 0");
  // NOTE: We are unable to take the offset into account in the following
  // assert, hence its still possible that the subvector is out-of-bounds even
  // if the condition is true.
  assert(numElemsToInsert <= destVecTy.getNumElements() &&
         "subvector out of bounds");

  Value destOffsetVal = getValueOrCreateConstantIndexOp(rewriter, loc, offset);
  for (int64_t i = 0; i < numElemsToInsert; ++i) {
    auto insertLoc =
        i == 0 ? destOffsetVal
               : arith::AddIOp::create(
                     rewriter, loc, rewriter.getIndexType(), destOffsetVal,
                     arith::ConstantIndexOp::create(rewriter, loc, i));
    auto extractOp = vector::ExtractOp::create(rewriter, loc, src, i);
    dest = vector::InsertOp::create(rewriter, loc, extractOp, dest, insertLoc);
  }
  return dest;
}

/// Emulate a vector load for `emulatedElemTy` using `containerElemTy`
///
/// Specifically, use `containerElemTy` for loading a vector of
/// `emulatedElemTy`. The load location is given by `base` and
/// `linearizedIndices`, and the load size is given by
/// `numEmulatedElementsToLoad`.
static VectorValue emulatedVectorLoad(OpBuilder &rewriter, Location loc,
                                      Value base,
                                      OpFoldResult linearizedIndices,
                                      int64_t numContainerElemsToLoad,
                                      Type emulatedElemTy,
                                      Type containerElemTy) {
  auto emulatedPerContainerElem = containerElemTy.getIntOrFloatBitWidth() /
                                  emulatedElemTy.getIntOrFloatBitWidth();
  auto newLoad = vector::LoadOp::create(
      rewriter, loc, VectorType::get(numContainerElemsToLoad, containerElemTy),
      base, getValueOrCreateConstantIndexOp(rewriter, loc, linearizedIndices));
  return vector::BitCastOp::create(
      rewriter, loc,
      VectorType::get(numContainerElemsToLoad * emulatedPerContainerElem,
                      emulatedElemTy),
      newLoad);
}

/// Downcast two values to `downcastType`, then select values
/// based on `mask`, and casts the result to `upcastType`.
static Value downcastSelectAndUpcast(OpBuilder &builder, Location loc,
                                     VectorType downcastType,
                                     VectorType upcastType, Value mask,
                                     Value trueValue, Value falseValue) {
  assert(
      downcastType.getNumElements() * downcastType.getElementTypeBitWidth() ==
          upcastType.getNumElements() * upcastType.getElementTypeBitWidth() &&
      "expected input and output number of bits to match");
  if (trueValue.getType() != downcastType) {
    trueValue =
        vector::BitCastOp::create(builder, loc, downcastType, trueValue);
  }
  if (falseValue.getType() != downcastType) {
    falseValue =
        vector::BitCastOp::create(builder, loc, downcastType, falseValue);
  }
  Value selectedType =
      arith::SelectOp::create(builder, loc, mask, trueValue, falseValue);
  // Upcast the selected value to the new type.
  return vector::BitCastOp::create(builder, loc, upcastType, selectedType);
}

/// Emits `memref.generic_atomic_rmw` op to store a subbyte-sized value to a
/// byte in `linearizedMemref`, with a mask. The `valueToStore` is a vector of
/// subbyte-sized elements, with size of 8 bits, and the mask is used to select
/// which elements to store.
///
/// Inputs:
///   linearizedMemref = |2|2|2|2| : <4xi2> (<1xi8>)
///   storeIdx = 2
///   valueToStore = |3|3|3|3| : vector<4xi2>
///   mask = |0|0|1|1| : vector<4xi1>
///
/// Result:
///   linearizedMemref = |2|2|3|3| : <4xi2> (<1xi8>)
static void atomicRMW(OpBuilder &builder, Location loc,
                      MemRefValue linearizedMemref, Value storeIdx,
                      VectorValue valueToStore, Value mask) {
  assert(valueToStore.getType().getRank() == 1 && "expected 1-D vector");

  // Create an atomic load-modify-write region using
  // `memref.generic_atomic_rmw`.
  auto atomicOp = memref::GenericAtomicRMWOp::create(
      builder, loc, linearizedMemref, ValueRange{storeIdx});
  Value origValue = atomicOp.getCurrentValue();

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(atomicOp.getBody());

  // Load the original value from memory, and cast it to the original element
  // type.
  auto oneElemVecType = VectorType::get({1}, origValue.getType());
  Value origVecValue = vector::FromElementsOp::create(
      builder, loc, oneElemVecType, ValueRange{origValue});

  // Construct the final masked value and yield it.
  Value maskedValue =
      downcastSelectAndUpcast(builder, loc, valueToStore.getType(),
                              oneElemVecType, mask, valueToStore, origVecValue);
  auto scalarMaskedValue =
      vector::ExtractOp::create(builder, loc, maskedValue, 0);
  memref::AtomicYieldOp::create(builder, loc, scalarMaskedValue);
}

/// Generate a non-atomic read-modify-write sequence for storing to the emulated
/// type. It has similar logic to `atomicRMWStore`, but without atomicity.
static void nonAtomicRMW(OpBuilder &builder, Location loc,
                         MemRefValue linearizedMemref, Value linearizedIndex,
                         VectorValue valueToStore, Value mask) {
  assert(valueToStore.getType().getRank() == 1 && "expected 1-D vector");

  auto oneElemVecType =
      VectorType::get({1}, linearizedMemref.getType().getElementType());
  Value origVecValue =
      vector::LoadOp::create(builder, loc, oneElemVecType, linearizedMemref,
                             ValueRange{linearizedIndex});
  origVecValue = vector::BitCastOp::create(builder, loc, valueToStore.getType(),
                                           origVecValue);

  Value maskedValue =
      downcastSelectAndUpcast(builder, loc, valueToStore.getType(),
                              oneElemVecType, mask, valueToStore, origVecValue);
  vector::StoreOp::create(builder, loc, maskedValue, linearizedMemref,
                          linearizedIndex);
}

/// Extract `sliceNumElements` from source `vector` at `extractOffset`,
/// and insert it into an empty vector at `insertOffset`.
/// Inputs:
///   vec_in  = |0|1|2|3| : vector<4xi2>
///   extractOffset = 1
///   sliceNumElements = 2
///   insertOffset = 2
/// Output:
///   vec_out = |0|0|1|2| : vector<4xi2>
static Value extractSliceIntoByte(ConversionPatternRewriter &rewriter,
                                  Location loc, VectorValue vector,
                                  int64_t extractOffset,
                                  int64_t sliceNumElements,
                                  int64_t insertOffset) {
  assert(vector.getType().getRank() == 1 && "expected 1-D vector");
  auto vectorElementType = vector.getType().getElementType();
  // TODO: update and use `alignedConversionPrecondition` in the place of
  // these asserts.
  assert(
      sliceNumElements * vectorElementType.getIntOrFloatBitWidth() <= 8 &&
      "sliceNumElements * vector element size must be less than or equal to 8");
  assert(8 % vectorElementType.getIntOrFloatBitWidth() == 0 &&
         "vector element must be a valid sub-byte type");
  auto emulatedPerContainerElem = 8 / vectorElementType.getIntOrFloatBitWidth();
  auto emptyByteVector = arith::ConstantOp::create(
      rewriter, loc,
      VectorType::get({emulatedPerContainerElem}, vectorElementType),
      rewriter.getZeroAttr(
          VectorType::get({emulatedPerContainerElem}, vectorElementType)));
  auto extracted = staticallyExtractSubvector(rewriter, loc, vector,
                                              extractOffset, sliceNumElements);
  return staticallyInsertSubvector(rewriter, loc, extracted, emptyByteVector,
                                   insertOffset);
}

namespace {

//===----------------------------------------------------------------------===//
// ConvertVectorStore
//===----------------------------------------------------------------------===//

// Emulate `vector.store` using a multi-byte container type.
//
// The container type is obtained through Op adaptor and would normally be
// generated via `NarrowTypeEmulationConverter`.
//
// EXAMPLE 1
// (aligned store of i4, emulated using i8 as the container type)
//
//      vector.store %src, %dest[%idx_1, %idx_2] : memref<4x8xi4>, vector<8xi4>
//
// is rewritten as:
//
//      %src_bitcast = vector.bitcast %src : vector<8xi4> to vector<4xi8>
//      vector.store %src_bitcast, %dest_bitcast[%idx]
//        : memref<16xi8>, vector<4xi8>
//
// EXAMPLE 2
// (unaligned store of i2, emulated using i8 as the container type)
//
//    vector.store %src, %dest[%c2, %c0] :memref<3x3xi2>, vector<3xi2>
//
// The i2 store is emulated through 2 x RMW sequences. The destination i2 memref
// is modelled using 3 bytes:
//
//    Byte 0     Byte 1     Byte 2
// +----------+----------+----------+
// | oooooooo | ooooNNNN | NNoooooo |
// +----------+----------+----------+
//
// N - (N)ew entries (i.e. to be overwritten by vector.store)
// o - (o)ld entries (to be preserved)
//
// For the generated output in the non-atomic case, see:
//  * @vector_store_i2_const_index_two_partial_stores`
// in:
//  * "vector-emulate-narrow-type-unaligned-non-atomic.mlir".
//
// NOTE: By default, all RMW sequences are atomic. Set `disableAtomicRMW` to
// `false` to generate non-atomic RMW sequences.
struct ConvertVectorStore final : OpConversionPattern<vector::StoreOp> {
  using Base::Base;

  ConvertVectorStore(MLIRContext *context, bool disableAtomicRMW)
      : OpConversionPattern<vector::StoreOp>(context),
        disableAtomicRMW(disableAtomicRMW) {}

  LogicalResult
  matchAndRewrite(vector::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    if (op.getValueToStore().getType().getRank() != 1)
      return rewriter.notifyMatchFailure(op,
                                         "only 1-D vectors are supported ATM");

    auto loc = op.getLoc();

    auto valueToStore = cast<VectorValue>(op.getValueToStore());
    auto containerElemTy =
        cast<MemRefType>(adaptor.getBase().getType()).getElementType();
    Type emulatedElemTy = op.getValueToStore().getType().getElementType();
    int emulatedBits = emulatedElemTy.getIntOrFloatBitWidth();
    int containerBits = containerElemTy.getIntOrFloatBitWidth();

    // Check per-element alignment.
    if (containerBits % emulatedBits != 0) {
      return rewriter.notifyMatchFailure(
          op, "impossible to pack emulated elements into container elements "
              "(bit-wise misalignment)");
    }
    int emulatedPerContainerElem = containerBits / emulatedBits;

    // Adjust the number of elements to store when emulating narrow types.
    // Here only the 1-D vector store is considered, and the N-D memref types
    // should be linearized.
    // For example, to emulate i4 to i8, the following op:
    //
    // vector.store %arg1, %0[%arg2, %arg3] : memref<4x8xi4>, vector<8xi4>
    //
    // can be replaced with
    //
    // %bitcast = vector.bitcast %arg1 : vector<8xi4> to vector<4xi8>
    // vector.store %bitcast, %alloc[%linear_index] : memref<16xi8>,
    // vector<4xi8>

    auto origElements = valueToStore.getType().getNumElements();
    // Note, per-element-alignment was already verified above.
    bool isDivisibleInSize = origElements % emulatedPerContainerElem == 0;
    // Do the trailing dim for source and destination match? If yes, then the
    // corresponding index must be 0.
    // FIXME: There's no way to tell for dynamic shapes, so we should bail out.
    // However, that makes some tests fail, so we need to audit first.
    auto trailingDim = op.getBase().getType().getShape().back();
    bool trailingDimsMatch =
        ShapedType::isDynamic(trailingDim) || trailingDim == origElements;

    auto stridedMetadata =
        memref::ExtractStridedMetadataOp::create(rewriter, loc, op.getBase());

    // FIXME: ATM, we do not test cases where offsets, sizes, or strides are
    // non-zero. As such, this is not needed.
    OpFoldResult linearizedIndices;
    memref::LinearizedMemRefInfo linearizedInfo;
    std::tie(linearizedInfo, linearizedIndices) =
        memref::getLinearizedMemRefOffsetAndSize(
            rewriter, loc, emulatedBits, containerBits,
            stridedMetadata.getConstifiedMixedOffset(),
            stridedMetadata.getConstifiedMixedSizes(),
            stridedMetadata.getConstifiedMixedStrides(),
            getAsOpFoldResult(adaptor.getIndices()));

    std::optional<int64_t> foldedNumFrontPadElems =
        (isDivisibleInSize && trailingDimsMatch)
            ? 0
            : getConstantIntValue(linearizedInfo.intraDataOffset);

    if (!foldedNumFrontPadElems) {
      return rewriter.notifyMatchFailure(
          op, "subbyte store emulation: dynamic front padding size is "
              "not yet implemented");
    }

    auto memrefBase = cast<MemRefValue>(adaptor.getBase());

    // RMWs are not needed when:
    //  * no _partial_ stores are required.
    // A partial store is defined as a store in which only a part of the
    // container element is overwritten, e.g.
    //
    //    Dest before (8 bits)
    //        +----------+
    //        | 11000000 |
    //        +----------+
    //
    //    Dest after storing 0xF at offset 4 (in bits)
    //        +----------+
    //        | 11001111 |
    //        +----------+
    //
    // At a higher level, this translats to:
    // 1. The source vector size (in bits) is a multiple of byte size.
    // 2. The address of the store is aligned to the container type width
    //    boundary.
    //
    // EXAMPLE 1:
    //  Requires partial store:
    //    vector.store %arg0, %0[%c3] : memref<13xi2>, vector<4xi2>
    //
    // EXAMPLE 2:
    //  Does not require a partial store:
    //    vector.store %arg0, %0[%c4] : memref<13xi2>, vector<4xi2>
    //
    // TODO: Take linearizedInfo.linearizedOffset into account. This is
    // currently not needed/used/exercised as all our tests set offset to 0.
    bool emulationRequiresPartialStores = *foldedNumFrontPadElems != 0;

    if (!emulationRequiresPartialStores) {
      // Basic case: storing full bytes.
      auto numElements = origElements / emulatedPerContainerElem;
      auto bitCast = vector::BitCastOp::create(
          rewriter, loc, VectorType::get(numElements, containerElemTy),
          op.getValueToStore());
      rewriter.replaceOpWithNewOp<vector::StoreOp>(
          op, bitCast.getResult(), memrefBase,
          getValueOrCreateConstantIndexOp(rewriter, loc, linearizedIndices));
      return success();
    }

    // Next, handle the case when sub-byte read-modify-write
    // sequences are needed to emulate a vector store.
    // Here is an example:
    //
    // Vector to store: vector<7xi2>
    // Value to store: 11 11 11 11 11 11 11 (all ones)
    //
    // Destination: memref<12xi2>
    // Store offset: 2 (i.e. 4 bits into the 1st emulated byte).
    //
    // Input MLIR: vector.store %val, %dest[%c2] : memref<12xi2>, vector<7xi2>
    //
    // Destination memref before:
    //
    //    Byte 0     Byte 1     Byte 2
    // +----------+----------+----------+
    // | 00000000 | 00000000 | 00000000 |
    // +----------+----------+----------+
    //
    // Destination memref after:
    //
    //    Byte 0     Byte 1     Byte 2
    // +----------+----------+----------+
    // | 00001111 | 11111111 | 11000000 |
    // +----------+----------+----------+
    //
    // Note, stores to Byte 1 are "full-width" and hence don't require RMW (no
    // need for atomicity). Stores to Bytes 0 and Byte 2 are "partial", hence
    // requiring RMW access (atomicity is required).

    // The index into the target memref we are storing to.
    Value currentDestIndex =
        getValueOrCreateConstantIndexOp(rewriter, loc, linearizedIndices);
    // The index into the source vector we are currently processing.
    auto currentSourceIndex = 0;

    // Build a mask used for rmw.
    auto subWidthStoreMaskType =
        VectorType::get({emulatedPerContainerElem}, rewriter.getI1Type());

    auto storeFunc = disableAtomicRMW ? nonAtomicRMW : atomicRMW;

    // 1. Partial width store for the leading byte.
    // When the store address is not aligned to emulated width boundary, deal
    // with the unaligned part so that the rest elements are aligned to width
    // boundary.
    auto frontSubWidthStoreElem =
        (emulatedPerContainerElem - *foldedNumFrontPadElems) %
        emulatedPerContainerElem;
    if (frontSubWidthStoreElem > 0) {
      SmallVector<bool> frontMaskValues(emulatedPerContainerElem, false);
      if (*foldedNumFrontPadElems + origElements < emulatedPerContainerElem) {
        std::fill_n(frontMaskValues.begin() + *foldedNumFrontPadElems,
                    origElements, true);
        frontSubWidthStoreElem = origElements;
      } else {
        std::fill_n(frontMaskValues.end() - frontSubWidthStoreElem,
                    *foldedNumFrontPadElems, true);
      }
      auto frontMask = arith::ConstantOp::create(
          rewriter, loc,
          DenseElementsAttr::get(subWidthStoreMaskType, frontMaskValues));

      currentSourceIndex = emulatedPerContainerElem - (*foldedNumFrontPadElems);
      auto value =
          extractSliceIntoByte(rewriter, loc, valueToStore, 0,
                               frontSubWidthStoreElem, *foldedNumFrontPadElems);

      storeFunc(rewriter, loc, memrefBase, currentDestIndex,
                cast<VectorValue>(value), frontMask.getResult());
    }

    if (currentSourceIndex >= origElements) {
      rewriter.eraseOp(op);
      return success();
    }

    // Increment the destination index by 1 to align to the emulated width
    // boundary.
    auto constantOne = arith::ConstantIndexOp::create(rewriter, loc, 1);
    currentDestIndex = arith::AddIOp::create(
        rewriter, loc, rewriter.getIndexType(), currentDestIndex, constantOne);

    // 2. Full width store for the inner output bytes.
    // After the previous step, the store address is aligned to the emulated
    // width boundary.
    int64_t fullWidthStoreSize =
        (origElements - currentSourceIndex) / emulatedPerContainerElem;
    int64_t numNonFullWidthElements =
        fullWidthStoreSize * emulatedPerContainerElem;
    if (fullWidthStoreSize > 0) {
      auto fullWidthStorePart = staticallyExtractSubvector(
          rewriter, loc, valueToStore, currentSourceIndex,
          numNonFullWidthElements);

      auto originType = cast<VectorType>(fullWidthStorePart.getType());
      auto memrefElemType = getElementTypeOrSelf(memrefBase.getType());
      auto storeType = VectorType::get(
          {originType.getNumElements() / emulatedPerContainerElem},
          memrefElemType);
      auto bitCast = vector::BitCastOp::create(rewriter, loc, storeType,
                                               fullWidthStorePart);
      vector::StoreOp::create(rewriter, loc, bitCast.getResult(), memrefBase,
                              currentDestIndex);

      currentSourceIndex += numNonFullWidthElements;
      currentDestIndex = arith::AddIOp::create(
          rewriter, loc, rewriter.getIndexType(), currentDestIndex,
          arith::ConstantIndexOp::create(rewriter, loc, fullWidthStoreSize));
    }

    // 3. Partial width store for the trailing output byte.
    // It is needed when the residual length is smaller than the emulated width,
    // which is not covered in step 2 above.
    auto remainingElements = origElements - currentSourceIndex;
    if (remainingElements != 0) {
      auto subWidthStorePart =
          extractSliceIntoByte(rewriter, loc, cast<VectorValue>(valueToStore),
                               currentSourceIndex, remainingElements, 0);

      // Generate back mask.
      auto maskValues = SmallVector<bool>(emulatedPerContainerElem, 0);
      std::fill_n(maskValues.begin(), remainingElements, 1);
      auto backMask = arith::ConstantOp::create(
          rewriter, loc,
          DenseElementsAttr::get(subWidthStoreMaskType, maskValues));

      storeFunc(rewriter, loc, memrefBase, currentDestIndex,
                cast<VectorValue>(subWidthStorePart), backMask.getResult());
    }

    rewriter.eraseOp(op);
    return success();
  }

private:
  const bool disableAtomicRMW;
};

//===----------------------------------------------------------------------===//
// ConvertVectorMaskedStore
//===----------------------------------------------------------------------===//

/// Converts `vector.maskedstore` operations on narrow element types to work
/// with wider, byte-aligned container types by adjusting the mask and using
/// bitcasting.
///
/// Example: Storing `vector<6xi4>` is emulated by bitcasting to `vector<3xi8>`
/// (each `i8` container element holds two `i4` values) and storing with an
/// adjusted mask .
struct ConvertVectorMaskedStore final
    : OpConversionPattern<vector::MaskedStoreOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(vector::MaskedStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // Prerequisite: memref in the vector.maskedstore op is flattened into 1-D.
    if (op.getValueToStore().getType().getRank() != 1)
      return rewriter.notifyMatchFailure(
          op, "Memref in vector.maskedstore op must be flattened beforehand.");

    auto loc = op.getLoc();
    auto containerElemTy =
        cast<MemRefType>(adaptor.getBase().getType()).getElementType();
    Type emulatedElemTy = op.getValueToStore().getType().getElementType();
    int emulatedBits = emulatedElemTy.getIntOrFloatBitWidth();
    int containerBits = containerElemTy.getIntOrFloatBitWidth();

    // Check per-element alignment.
    if (containerBits % emulatedBits != 0) {
      return rewriter.notifyMatchFailure(
          op, "impossible to pack emulated elements into container elements "
              "(bit-wise misalignment)");
    }

    int emulatedPerContainerElem = containerBits / emulatedBits;
    int origElements = op.getValueToStore().getType().getNumElements();
    if (origElements % emulatedPerContainerElem != 0)
      return failure();

    auto stridedMetadata =
        memref::ExtractStridedMetadataOp::create(rewriter, loc, op.getBase());
    OpFoldResult linearizedIndicesOfr;
    memref::LinearizedMemRefInfo linearizedInfo;
    std::tie(linearizedInfo, linearizedIndicesOfr) =
        memref::getLinearizedMemRefOffsetAndSize(
            rewriter, loc, emulatedBits, containerBits,
            stridedMetadata.getConstifiedMixedOffset(),
            stridedMetadata.getConstifiedMixedSizes(),
            stridedMetadata.getConstifiedMixedStrides(),
            getAsOpFoldResult(adaptor.getIndices()));
    Value linearizedIndices =
        getValueOrCreateConstantIndexOp(rewriter, loc, linearizedIndicesOfr);

    // Load the whole data and use arith.select to handle the corner cases.
    //
    // As an example, for this masked store of i4 values:
    //
    //   vector.maskedstore %0[%c0, %c0], %mask, %val_to_store
    //
    // and given these input values:
    //
    //   %mask = [0, 1, 1, 1, 1, 0, 0, 0]                     (8 * i1)
    //   %0[%c0, %c0] =
    //      [0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8]          (8 * i4)
    //   %val_to_store =
    //      [0x9, 0xA, 0xB, 0xC, 0xD, 0xE, 0xF, 0x0]          (8 * i4)
    //
    // we'll have the following i4 output:
    //
    //    expected output: [0x1, 0xA, 0xB, 0xC, 0xD, 0x6, 0x7, 0x8]
    //
    // Emulating the above using i8 will give:
    //
    //    %compressed_mask = [1, 1, 1, 0]                     (4 * i1)
    //    %maskedload = [0x12, 0x34, 0x56, 0x00]              (4 * i8)
    //    %bitcast = [0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x0, 0x0] (8 * i4)
    //    %select_using_shifted_mask =
    //      [0x1, 0xA, 0xB, 0xC, 0xD, 0x6, 0x0, 0x0]          (8 * i4)
    //    %packed_data = [0x1A, 0xBC, 0xD6, 0x00]             (4 * i8)
    //
    // Using the compressed mask to store %packed_data results in expected
    // output.
    //
    // FIXME: Make an example based on the comment above work (see #115460 for
    // reproducer).
    FailureOr<Operation *> newMask = getCompressedMaskOp(
        rewriter, loc, op.getMask(), origElements, emulatedPerContainerElem);
    if (failed(newMask))
      return failure();

    auto numElements = (origElements + emulatedPerContainerElem - 1) /
                       emulatedPerContainerElem;
    auto newType = VectorType::get(numElements, containerElemTy);
    auto passThru = arith::ConstantOp::create(rewriter, loc, newType,
                                              rewriter.getZeroAttr(newType));

    auto newLoad = vector::MaskedLoadOp::create(
        rewriter, loc, newType, adaptor.getBase(), linearizedIndices,
        newMask.value()->getResult(0), passThru);

    auto newBitCastType =
        VectorType::get(numElements * emulatedPerContainerElem, emulatedElemTy);
    Value valueToStore =
        vector::BitCastOp::create(rewriter, loc, newBitCastType, newLoad);
    valueToStore = arith::SelectOp::create(rewriter, loc, op.getMask(),
                                           op.getValueToStore(), valueToStore);
    valueToStore =
        vector::BitCastOp::create(rewriter, loc, newType, valueToStore);

    rewriter.replaceOpWithNewOp<vector::MaskedStoreOp>(
        op, adaptor.getBase(), linearizedIndices, newMask.value()->getResult(0),
        valueToStore);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertVectorLoad
//===----------------------------------------------------------------------===//

/// Converts `vector.load` on narrow element types to work with
/// wider, byte-aligned container types by adjusting load sizes and using
/// bitcasting.
///
/// Example: `vector.load` of `vector<4xi4>` from `memref<3x4xi4>` is emulated
/// by loading `vector<2xi8>` from the linearized `memref<6xi8>` (each `i8`
/// container holds two `i4` values) and bitcasting back.
///
/// There are cases where the number of elements to load is not byte-aligned. In
/// those cases, loads are converted to byte-aligned, byte-sized loads and the
/// target vector is extracted from the loaded vector.
struct ConvertVectorLoad final : OpConversionPattern<vector::LoadOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(vector::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Prerequisite:  memref in the vector.load op is flattened into 1-D.
    if (op.getVectorType().getRank() != 1)
      return rewriter.notifyMatchFailure(
          op, "Memref in emulated vector ops must be flattened beforehand.");

    auto loc = op.getLoc();
    auto containerElemTy =
        cast<MemRefType>(adaptor.getBase().getType()).getElementType();
    Type emulatedElemTy = op.getType().getElementType();
    int emulatedBits = emulatedElemTy.getIntOrFloatBitWidth();
    int containerBits = containerElemTy.getIntOrFloatBitWidth();

    // Check per-element alignment.
    if (containerBits % emulatedBits != 0) {
      return rewriter.notifyMatchFailure(
          op, "impossible to pack emulated elements into container elements "
              "(bit-wise misalignment)");
    }
    int emulatedPerContainerElem = containerBits / emulatedBits;

    // Adjust the number of elements to load when emulating narrow types,
    // and then cast back to the original type with vector.bitcast op.
    // For example, to emulate i4 to i8, the following op:
    //
    // %1 = vector.load %0[%c0, %c0] : memref<3x4xi4>, vector<4xi4>
    //
    // can be replaced with
    //
    // %1 = vector.load %0[%linear_index] : memref<6xi8>, vector<2xi8>
    // %2 = vector.bitcast %1 : vector<2xi8> to vector<4xi4>
    //
    // There are cases where the number of elements to load is not byte-aligned,
    // for example:
    //
    // %1 = vector.load %0[%c1, %c0] : memref<3x3xi2>, vector<3xi2>
    //
    // we will have to load extra bytes and extract the exact slice in between.
    //
    // %1 = vector.load %0[%c2] : memref<3xi8>, vector<2xi8>
    // %2 = vector.bitcast %1 : vector<2xi8> to vector<8xi2>
    // %3 = vector.extract_strided_slice %1 {offsets = [2], sizes = [3], strides
    // = [1]}
    //        : vector<8xi2> to vector<3xi2>
    //
    // TODO: Currently the extract_strided_slice's attributes must be known at
    // compile time as they must be constants.

    auto origElements = op.getVectorType().getNumElements();
    // Note, per-element-alignment was already verified above.
    bool isDivisibleInSize = origElements % emulatedPerContainerElem == 0;

    auto stridedMetadata =
        memref::ExtractStridedMetadataOp::create(rewriter, loc, op.getBase());

    OpFoldResult linearizedIndices;
    memref::LinearizedMemRefInfo linearizedInfo;
    std::tie(linearizedInfo, linearizedIndices) =
        memref::getLinearizedMemRefOffsetAndSize(
            rewriter, loc, emulatedBits, containerBits,
            stridedMetadata.getConstifiedMixedOffset(),
            stridedMetadata.getConstifiedMixedSizes(),
            stridedMetadata.getConstifiedMixedStrides(),
            getAsOpFoldResult(adaptor.getIndices()));

    std::optional<int64_t> foldedIntraVectorOffset =
        isDivisibleInSize ? 0
                          : getConstantIntValue(linearizedInfo.intraDataOffset);

    // Always load enough elements which can cover the original elements.
    int64_t maxintraDataOffset =
        foldedIntraVectorOffset.value_or(emulatedPerContainerElem - 1);
    auto numElements = llvm::divideCeil(maxintraDataOffset + origElements,
                                        emulatedPerContainerElem);
    Value result =
        emulatedVectorLoad(rewriter, loc, adaptor.getBase(), linearizedIndices,
                           numElements, emulatedElemTy, containerElemTy);

    if (!foldedIntraVectorOffset) {
      auto resultVector = arith::ConstantOp::create(
          rewriter, loc, op.getType(), rewriter.getZeroAttr(op.getType()));
      result = dynamicallyExtractSubVector(
          rewriter, loc, dyn_cast<TypedValue<VectorType>>(result), resultVector,
          linearizedInfo.intraDataOffset, origElements);
    } else if (!isDivisibleInSize) {
      result = staticallyExtractSubvector(
          rewriter, loc, result, *foldedIntraVectorOffset, origElements);
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertVectorMaskedLoad
//===----------------------------------------------------------------------===//

/// Converts `vector.maskedload` operations on narrow element types to work with
/// wider, byte-aligned container types by adjusting the mask and using
/// bitcasting.
///
/// Example: Loading `vector<6xi4>` is emulated by loading `vector<3xi8>` and
/// bitcasting, since each `i8` container element holds two `i4` values.
struct ConvertVectorMaskedLoad final
    : OpConversionPattern<vector::MaskedLoadOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(vector::MaskedLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getVectorType().getRank() != 1)
      return rewriter.notifyMatchFailure(
          op, "Memref in emulated vector ops must be flattened beforehand.");

    auto loc = op.getLoc();

    auto containerElemTy =
        cast<MemRefType>(adaptor.getBase().getType()).getElementType();
    Type emulatedElemTy = op.getType().getElementType();
    int emulatedBits = emulatedElemTy.getIntOrFloatBitWidth();
    int containerBits = containerElemTy.getIntOrFloatBitWidth();

    // Check per-element alignment.
    if (containerBits % emulatedBits != 0) {
      return rewriter.notifyMatchFailure(
          op, "impossible to pack emulated elements into container elements "
              "(bit-wise misalignment)");
    }
    int emulatedPerContainerElem = containerBits / emulatedBits;

    // Adjust the number of elements to load when emulating narrow types,
    // and then cast back to the original type with vector.bitcast op.
    // For example, to emulate i4 to i8, the following op:
    //
    //   %mask = vector.constant_mask [3] : vector<6xi1>
    //   %1 = vector.maskedload %0[%c0, %c0], %mask, %pass_thru :
    //        memref<3x6xi4>, vector<6xi1>, vector<6xi4> into vector<6xi4>
    //
    // can be replaced with
    //
    //   %new_mask = vector.constant_mask [2] : vector<3xi1>
    //   %new_pass_thru = vector.bitcast %pass_thru :
    //        vector<6xi4> to vector<3xi8>
    //   %1 = vector.maskedload %0[%linear_index], %new_mask, %new_pass_thru :
    //        memref<9xi8>, vector<3xi1>, vector<3xi8> into vector<3xi8>
    //   %2 = vector.bitcast %1 : vector<3xi8> to vector<6xi4>
    //
    // Since we are effectively loading 16 bits (2xi8) from the memref with the
    // new mask, while originally we only wanted to effectively load 12 bits
    // (3xi4) from the memref, we need to set the second half of the last i8
    // that was effectively loaded (i.e. the second i8) to %pass_thru.
    //
    //   %3 = arith.select %mask, %2, %pass_thru : vector<6xi1>, vector<6xi4>
    //
    // Given these input values:
    //   %mask = [1, 1, 1, 0, 0, 0]
    //   %0[%c0, %c0] contains [0x1, 0x2, 0x3, 0x4, 0x5, 0x6]
    //   %pass_thru = [0x7, 0x8, 0x9, 0xA, 0xB, 0xC]
    //
    // we'll have:
    //
    //   expected output: [0x1, 0x2, 0x3, 0xA, 0xB, 0xC]
    //
    //   %new_mask = [1, 1, 0]
    //   %new_pass_thru = [0x78, 0x9A, 0xBC]
    //   %1 = [0x12, 0x34, 0xBC]
    //   %2 = [0x1, 0x2, 0x3, 0x4, 0xB, 0xC]
    //   %3 = [0x1, 0x2, 0x3, 0xA, 0xB, 0xC]
    //
    // TODO: Currently, only the even number of elements loading is supported.
    // To deal with the odd number of elements, one has to extract the
    // subvector at the proper offset after bit-casting.
    auto origType = op.getVectorType();
    auto origElements = origType.getNumElements();
    // Note, per-element-alignment was already verified above.
    bool isDivisibleInSize = origElements % emulatedPerContainerElem == 0;

    auto stridedMetadata =
        memref::ExtractStridedMetadataOp::create(rewriter, loc, op.getBase());
    OpFoldResult linearizedIndices;
    memref::LinearizedMemRefInfo linearizedInfo;
    std::tie(linearizedInfo, linearizedIndices) =
        memref::getLinearizedMemRefOffsetAndSize(
            rewriter, loc, emulatedBits, containerBits,
            stridedMetadata.getConstifiedMixedOffset(),
            stridedMetadata.getConstifiedMixedSizes(),
            stridedMetadata.getConstifiedMixedStrides(),
            getAsOpFoldResult(adaptor.getIndices()));

    std::optional<int64_t> foldedIntraVectorOffset =
        isDivisibleInSize ? 0
                          : getConstantIntValue(linearizedInfo.intraDataOffset);

    int64_t maxIntraDataOffset =
        foldedIntraVectorOffset.value_or(emulatedPerContainerElem - 1);
    FailureOr<Operation *> newMask =
        getCompressedMaskOp(rewriter, loc, op.getMask(), origElements,
                            emulatedPerContainerElem, maxIntraDataOffset);
    if (failed(newMask))
      return failure();

    Value passthru = op.getPassThru();

    auto numElements = llvm::divideCeil(maxIntraDataOffset + origElements,
                                        emulatedPerContainerElem);
    auto loadType = VectorType::get(numElements, containerElemTy);
    auto newBitcastType =
        VectorType::get(numElements * emulatedPerContainerElem, emulatedElemTy);

    auto emptyVector = arith::ConstantOp::create(
        rewriter, loc, newBitcastType, rewriter.getZeroAttr(newBitcastType));
    if (!foldedIntraVectorOffset) {
      passthru = dynamicallyInsertSubVector(
          rewriter, loc, passthru, emptyVector, linearizedInfo.intraDataOffset,
          origElements);
    } else if (!isDivisibleInSize) {
      passthru = staticallyInsertSubvector(rewriter, loc, passthru, emptyVector,
                                           *foldedIntraVectorOffset);
    }
    auto newPassThru =
        vector::BitCastOp::create(rewriter, loc, loadType, passthru);

    // Generating the new masked load.
    auto newLoad = vector::MaskedLoadOp::create(
        rewriter, loc, loadType, adaptor.getBase(),
        getValueOrCreateConstantIndexOp(rewriter, loc, linearizedIndices),
        newMask.value()->getResult(0), newPassThru);

    // Setting the part that originally was not effectively loaded from memory
    // to pass through.
    auto bitCast =
        vector::BitCastOp::create(rewriter, loc, newBitcastType, newLoad);

    Value mask = op.getMask();
    auto newSelectMaskType = VectorType::get(
        numElements * emulatedPerContainerElem, rewriter.getI1Type());
    // TODO: try to fold if op's mask is constant
    auto emptyMask =
        arith::ConstantOp::create(rewriter, loc, newSelectMaskType,
                                  rewriter.getZeroAttr(newSelectMaskType));
    if (!foldedIntraVectorOffset) {
      mask = dynamicallyInsertSubVector(rewriter, loc, mask, emptyMask,
                                        linearizedInfo.intraDataOffset,
                                        origElements);
    } else if (!isDivisibleInSize) {
      mask = staticallyInsertSubvector(rewriter, loc, op.getMask(), emptyMask,
                                       *foldedIntraVectorOffset);
    }

    Value result =
        arith::SelectOp::create(rewriter, loc, mask, bitCast, passthru);
    if (!foldedIntraVectorOffset) {
      result = dynamicallyExtractSubVector(
          rewriter, loc, result, op.getPassThru(),
          linearizedInfo.intraDataOffset, origElements);
    } else if (!isDivisibleInSize) {
      result = staticallyExtractSubvector(
          rewriter, loc, result, *foldedIntraVectorOffset, origElements);
    }
    rewriter.replaceOp(op, result);

    return success();
  }
};

/// Check whether `subByteVecTy` fits wthin a vector of `multiByteScalarTy`
///
/// "Fitting" means that `subByteVecTy` (a vector of sub-byte elements, e.g.
/// vector<4xi4>), can fit within N scalar elements of type `multiByteScalarTy`
/// (a multi-byte scalar, e.g. i16), where N is some integer.
///
/// Put differently, this method checks whether this would be valid:
///
///   vector.bitcast subByteVecTy into vector<N x multiByteScalarTy>
///
/// EXAMPLES:
///   * vector<4xi4> -> i16 - yes (N = 1)
///   * vector<4xi4> -> i8 - yes (N = 2)
///   * vector<3xi4> -> i8 - no (N would have to be 1.5)
///   * vector<3xi2> -> i16 - no (N would have to be 0.5)
static bool fitsInMultiByteContainerTy(VectorType subByteVecTy,
                                       Type multiByteScalarTy) {
  assert((isa<IntegerType, FloatType>(multiByteScalarTy)) && "Not scalar!");

  int subByteBits = subByteVecTy.getElementType().getIntOrFloatBitWidth();
  int multiByteBits = multiByteScalarTy.getIntOrFloatBitWidth();

  assert(subByteBits < 8 && "Not a sub-byte scalar type!");
  assert(multiByteBits % 8 == 0 && "Not a multi-byte scalar type!");
  assert(multiByteBits % subByteBits == 0 && "Unalagined element types!");

  int elemsPerMultiByte = multiByteBits / subByteBits;

  return subByteVecTy.getShape().back() % elemsPerMultiByte == 0;
}

//===----------------------------------------------------------------------===//
// ConvertVectorTransferRead
//===----------------------------------------------------------------------===//

// TODO: Document-me
struct ConvertVectorTransferRead final
    : OpConversionPattern<vector::TransferReadOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(vector::TransferReadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // Prerequisites:  memref in the vector.transfer_read op is flattened into
    // 1-D.
    if (op.getVectorType().getRank() != 1)
      return rewriter.notifyMatchFailure(
          op, "Memref in emulated vector ops must be flattened beforehand.");

    auto loc = op.getLoc();
    auto containerElemTy =
        cast<MemRefType>(adaptor.getBase().getType()).getElementType();
    Type emulatedElemTy = op.getType().getElementType();
    int emulatedBits = emulatedElemTy.getIntOrFloatBitWidth();
    int containerBits = containerElemTy.getIntOrFloatBitWidth();

    // Check per-element alignment.
    if (containerBits % emulatedBits != 0) {
      return rewriter.notifyMatchFailure(
          op, "impossible to pack emulated elements into container elements "
              "(bit-wise misalignment)");
    }
    int emulatedPerContainerElem = containerBits / emulatedBits;

    auto origElements = op.getVectorType().getNumElements();

    // Note, per-element-alignment was already verified above.
    bool isDivisibleInSize =
        fitsInMultiByteContainerTy(op.getVectorType(), containerElemTy);

    // Pad the padding value with 0s on the left. These bits are discarded and
    // thus their values don't matter.
    Value padding = adaptor.getPadding();
    if (!padding.getType().isInteger()) {
      padding = arith::BitcastOp::create(
          rewriter, loc,
          IntegerType::get(rewriter.getContext(),
                           padding.getType().getIntOrFloatBitWidth()),
          padding);
    }
    auto newPadding =
        arith::ExtUIOp::create(rewriter, loc, containerElemTy, padding);

    auto stridedMetadata =
        memref::ExtractStridedMetadataOp::create(rewriter, loc, op.getBase());

    OpFoldResult linearizedIndices;
    memref::LinearizedMemRefInfo linearizedInfo;
    std::tie(linearizedInfo, linearizedIndices) =
        memref::getLinearizedMemRefOffsetAndSize(
            rewriter, loc, emulatedBits, containerBits,
            stridedMetadata.getConstifiedMixedOffset(),
            stridedMetadata.getConstifiedMixedSizes(),
            stridedMetadata.getConstifiedMixedStrides(),
            getAsOpFoldResult(adaptor.getIndices()));

    std::optional<int64_t> foldedIntraVectorOffset =
        isDivisibleInSize ? 0
                          : getConstantIntValue(linearizedInfo.intraDataOffset);

    int64_t maxIntraDataOffset =
        foldedIntraVectorOffset.value_or(emulatedPerContainerElem - 1);
    auto numElements = llvm::divideCeil(maxIntraDataOffset + origElements,
                                        emulatedPerContainerElem);

    auto newRead = vector::TransferReadOp::create(
        rewriter, loc, VectorType::get(numElements, containerElemTy),
        adaptor.getBase(),
        getValueOrCreateConstantIndexOp(rewriter, loc, linearizedIndices),
        newPadding);

    auto bitCast = vector::BitCastOp::create(
        rewriter, loc,
        VectorType::get(numElements * emulatedPerContainerElem, emulatedElemTy),
        newRead);

    Value result = bitCast->getResult(0);
    if (!foldedIntraVectorOffset) {
      auto zeros = arith::ConstantOp::create(
          rewriter, loc, op.getType(), rewriter.getZeroAttr(op.getType()));
      result = dynamicallyExtractSubVector(rewriter, loc, bitCast, zeros,
                                           linearizedInfo.intraDataOffset,
                                           origElements);
    } else if (!isDivisibleInSize) {
      result = staticallyExtractSubvector(
          rewriter, loc, result, *foldedIntraVectorOffset, origElements);
    }
    rewriter.replaceOp(op, result);

    return success();
  }
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// RewriteBitCastOfTruncI
//===----------------------------------------------------------------------===//

namespace {

/// Helper struct to keep track of the provenance of a contiguous set of bits
/// in a source vector.
struct SourceElementRange {
  /// The index of the source vector element that contributes bits to *this.
  int64_t sourceElementIdx;
  /// The range of bits in the source vector element that contribute to *this.
  int64_t sourceBitBegin;
  int64_t sourceBitEnd;
};

struct SourceElementRangeList : public SmallVector<SourceElementRange> {
  /// Given the index of a SourceElementRange in the SourceElementRangeList,
  /// compute the amount of bits that need to be shifted to the left to get the
  /// bits in their final location. This shift amount is simply the sum of the
  /// bits *before* `shuffleIdx` (i.e. the bits of `shuffleIdx = 0` are always
  /// the LSBs, the bits of `shuffleIdx = ` come next, etc).
  int64_t computeLeftShiftAmount(int64_t shuffleIdx) const {
    int64_t res = 0;
    for (int64_t i = 0; i < shuffleIdx; ++i)
      res += (*this)[i].sourceBitEnd - (*this)[i].sourceBitBegin;
    return res;
  }
};

/// Helper struct to enumerate the source elements and bit ranges that are
/// involved in a bitcast operation.
/// This allows rewriting a vector.bitcast into shuffles and bitwise ops for
/// any 1-D vector shape and any source/target bitwidths.
/// This creates and holds a mapping of the form:
/// [dstVectorElementJ] ==
///    [ {srcVectorElementX, bitRange}, {srcVectorElementY, bitRange}, ... ]
/// E.g. `vector.bitcast ... : vector<1xi24> to vector<3xi8>` is decomposed as:
///   [0] = {0, [0-8)}
///   [1] = {0, [8-16)}
///   [2] = {0, [16-24)}
/// and `vector.bitcast ... : vector<2xi15> to vector<3xi10>` is decomposed as:
///   [0] = {0, [0, 10)}, {1, [0, 5)}
///   [1] = {1, [5, 10)}, {2, [0, 10)}
struct BitCastBitsEnumerator {
  BitCastBitsEnumerator(VectorType sourceVectorType,
                        VectorType targetVectorType);

  int64_t getMaxNumberOfEntries() {
    int64_t numVectors = 0;
    for (const auto &l : sourceElementRanges)
      numVectors = std::max(numVectors, (int64_t)l.size());
    return numVectors;
  }

  VectorType sourceVectorType;
  VectorType targetVectorType;
  SmallVector<SourceElementRangeList> sourceElementRanges;
};

/// Rewrite vector.bitcast to a sequence of shuffles and bitwise ops that take
/// advantage of high-level information to avoid leaving LLVM to scramble with
/// peephole optimizations.
/// BitCastBitsEnumerator encodes for each element of the target vector the
/// provenance of the bits in the source vector. We can "transpose" this
/// information to build a sequence of shuffles and bitwise ops that will
/// produce the desired result.
//
/// Consider the following motivating example:
/// ```
///   %1 = vector.bitcast %0 : vector<32xi5> to vector<20xi8>
/// ```
//
/// BitCastBitsEnumerator contains the following information:
/// ```
///   { 0: b@[0..5) lshl: 0}{ 1: b@[0..3) lshl: 5}
///   { 1: b@[3..5) lshl: 0}{ 2: b@[0..5) lshl: 2}{ 3: b@[0..1) lshl: 7}
///   { 3: b@[1..5) lshl: 0}{ 4: b@[0..4) lshl: 4}
///   { 4: b@[4..5) lshl: 0}{ 5: b@[0..5) lshl: 1}{ 6: b@[0..2) lshl: 6}
///   { 6: b@[2..5) lshl: 0}{ 7: b@[0..5) lshl: 3}
///   { 8: b@[0..5) lshl: 0}{ 9: b@[0..3) lshl: 5}
///   { 9: b@[3..5) lshl: 0}{10: b@[0..5) lshl: 2}{11: b@[0..1) lshl: 7}
///   {11: b@[1..5) lshl: 0}{12: b@[0..4) lshl: 4}
///   {12: b@[4..5) lshl: 0}{13: b@[0..5) lshl: 1}{14: b@[0..2) lshl: 6}
///   {14: b@[2..5) lshl: 0}{15: b@[0..5) lshl: 3}
///   {16: b@[0..5) lshl: 0}{17: b@[0..3) lshl: 5}
///   {17: b@[3..5) lshl: 0}{18: b@[0..5) lshl: 2}{19: b@[0..1) lshl: 7}
///   {19: b@[1..5) lshl: 0}{20: b@[0..4) lshl: 4}
///   {20: b@[4..5) lshl: 0}{21: b@[0..5) lshl: 1}{22: b@[0..2) lshl: 6}
///   {22: b@[2..5) lshl: 0}{23: b@[0..5) lshl: 3}
///   {24: b@[0..5) lshl: 0}{25: b@[0..3) lshl: 5}
///   {25: b@[3..5) lshl: 0}{26: b@[0..5) lshl: 2}{27: b@[0..1) lshl: 7}
///   {27: b@[1..5) lshl: 0}{28: b@[0..4) lshl: 4}
///   {28: b@[4..5) lshl: 0}{29: b@[0..5) lshl: 1}{30: b@[0..2) lshl: 6}
///   {30: b@[2..5) lshl: 0}{31: b@[0..5) lshl: 3}
/// ```
///
/// In the above, each row represents one target vector element and each
/// column represents one bit contribution from a source vector element.
/// The algorithm creates vector.shuffle operations (in this case there are 3
/// shuffles (i.e. the max number of columns in BitCastBitsEnumerator). The
/// algorithm populates the bits as follows:
/// ```
///     src bits 0 ...
/// 1st shuffle |xxxxx   |xx      |...
/// 2nd shuffle |     xxx|  xxxxx |...
/// 3rd shuffle |        |       x|...
/// ```
//
/// The algorithm proceeds as follows:
///   1. for each vector.shuffle, collect the source vectors that participate in
///     this shuffle. One source vector per target element of the resulting
///     vector.shuffle. If there is no source element contributing bits for the
///     current vector.shuffle, take 0 (i.e. row 0 in the above example has only
///     2 columns).
///   2. represent the bitrange in the source vector as a mask. If there is no
///     source element contributing bits for the current vector.shuffle, take 0.
///   3. shift right by the proper amount to align the source bitrange at
///     position 0. This is exactly the low end of the bitrange. For instance,
///     the first element of row 2 is `{ 1: b@[3..5) lshl: 0}` and one needs to
///     shift right by 3 to get the bits contributed by the source element #1
///     into position 0.
///   4. shift left by the proper amount to to align to the desired position in
///     the result element vector.  For instance, the contribution of the second
///     source element for the first row needs to be shifted by `5` to form the
///     first i8 result element.
///
/// Eventually, we end up building  the sequence
/// `(shuffle -> and -> shiftright -> shiftleft -> or)` to iteratively update
/// the result vector (i.e. the `shiftright -> shiftleft -> or` part) with the
/// bits extracted from the source vector (i.e. the `shuffle -> and` part).
struct BitCastRewriter {
  /// Helper metadata struct to hold the static quantities for the rewrite.
  struct Metadata {
    SmallVector<int64_t> shuffles;
    SmallVector<Attribute> masks, shiftRightAmounts, shiftLeftAmounts;
  };

  BitCastRewriter(VectorType sourceVectorType, VectorType targetVectorType);

  /// Verify that general preconditions for the rewrite are met.
  LogicalResult commonPrecondition(PatternRewriter &rewriter,
                                   VectorType preconditionType, Operation *op);

  /// Precompute the metadata for the rewrite.
  SmallVector<BitCastRewriter::Metadata>
  precomputeMetadata(IntegerType shuffledElementType);

  /// Rewrite one step of the sequence:
  ///   `(shuffle -> and -> shiftright -> shiftleft -> or)`.
  Value genericRewriteStep(PatternRewriter &rewriter, Location loc,
                           Value initialValue, Value runningResult,
                           const BitCastRewriter::Metadata &metadata);

private:
  /// Underlying enumerator that encodes the provenance of the bits in the each
  /// element of the result vector.
  BitCastBitsEnumerator enumerator;
};

} // namespace

[[maybe_unused]] static raw_ostream &
operator<<(raw_ostream &os, const SmallVector<SourceElementRangeList> &vec) {
  for (const auto &l : vec) {
    for (auto it : llvm::enumerate(l)) {
      os << "{ " << it.value().sourceElementIdx << ": b@["
         << it.value().sourceBitBegin << ".." << it.value().sourceBitEnd
         << ") lshl: " << l.computeLeftShiftAmount(it.index()) << " } ";
    }
    os << "\n";
  }
  return os;
}

BitCastBitsEnumerator::BitCastBitsEnumerator(VectorType sourceVectorType,
                                             VectorType targetVectorType)
    : sourceVectorType(sourceVectorType), targetVectorType(targetVectorType) {

  assert(sourceVectorType.getRank() == 1 && !sourceVectorType.isScalable() &&
         "requires -D non-scalable vector type");
  assert(targetVectorType.getRank() == 1 && !targetVectorType.isScalable() &&
         "requires -D non-scalable vector type");
  int64_t sourceBitWidth = sourceVectorType.getElementTypeBitWidth();
  int64_t mostMinorSourceDim = sourceVectorType.getShape().back();
  LDBG() << "sourceVectorType: " << sourceVectorType;

  int64_t targetBitWidth = targetVectorType.getElementTypeBitWidth();
  int64_t mostMinorTargetDim = targetVectorType.getShape().back();
  LDBG() << "targetVectorType: " << targetVectorType;

  int64_t bitwidth = targetBitWidth * mostMinorTargetDim;
  (void)mostMinorSourceDim;
  assert(bitwidth == sourceBitWidth * mostMinorSourceDim &&
         "source and target bitwidths must match");

  // Prepopulate one source element range per target element.
  sourceElementRanges = SmallVector<SourceElementRangeList>(mostMinorTargetDim);
  for (int64_t resultBit = 0; resultBit < bitwidth;) {
    int64_t resultElement = resultBit / targetBitWidth;
    int64_t resultBitInElement = resultBit % targetBitWidth;
    int64_t sourceElementIdx = resultBit / sourceBitWidth;
    int64_t sourceBitInElement = resultBit % sourceBitWidth;
    int64_t step = std::min(sourceBitWidth - sourceBitInElement,
                            targetBitWidth - resultBitInElement);
    sourceElementRanges[resultElement].push_back(
        {sourceElementIdx, sourceBitInElement, sourceBitInElement + step});
    resultBit += step;
  }
}

BitCastRewriter::BitCastRewriter(VectorType sourceVectorType,
                                 VectorType targetVectorType)
    : enumerator(BitCastBitsEnumerator(sourceVectorType, targetVectorType)) {
  LDBG() << "\n" << enumerator.sourceElementRanges;
}

/// Verify that the precondition type meets the common preconditions for any
/// conversion.
static LogicalResult commonConversionPrecondition(PatternRewriter &rewriter,
                                                  VectorType preconditionType,
                                                  Operation *op) {
  if (!preconditionType || preconditionType.isScalable())
    return rewriter.notifyMatchFailure(op, "scalable vector");

  // TODO: consider relaxing this restriction in the future if we find ways
  // to really work with subbyte elements across the MLIR/LLVM boundary.
  unsigned bitwidth = preconditionType.getElementTypeBitWidth();
  if (bitwidth % 8 != 0)
    return rewriter.notifyMatchFailure(op, "bitwidth is not k * 8");

  return success();
}

LogicalResult BitCastRewriter::commonPrecondition(PatternRewriter &rewriter,
                                                  VectorType preconditionType,
                                                  Operation *op) {
  if (!enumerator.sourceVectorType || !enumerator.targetVectorType)
    return rewriter.notifyMatchFailure(op, "types are not vector");

  if (!preconditionType || preconditionType.getRank() != 1)
    return rewriter.notifyMatchFailure(op, "unsupported >1-D vector");

  return commonConversionPrecondition(rewriter, preconditionType, op);
}

/// Verify that `subByteVecTy` (vector) and `containerTy` (scalar) are aligned.
///
/// Alignment means that `subByteVecTy` can be packed into a vector of
/// `containerTy` elements. More specifically:
///   1. The bit-width of `containerTy` is a multiple of the
///      bit-width of `subByteVecTy` elements. For example, for `i4` and `i16`
///      this multiple is 4.
///   2. The multiple from 1. above divides evenly the number of the (trailing)
///      elements in `subByteVecTy`.
///
/// EXAMPLE 1:
///   `subByteVecTy = vector<2xi4>`, and
///   `containerTy = i16`
///
/// 2 divides evenly 4 ( = 16 / 4), hence both conditions are _met_.
///
/// EXAMPLE 2:
///   `subByteVecTy = vector<3xi4>`, and
///   `containerTy = i16`
///
/// 3 _does not_ divide evenly 4 (= 16/4), hence the conditions are _not met_.
///
/// EXAMPLE 3:
///   `subByteVecTy = vector<3xi3>`, and
///   `containerTy = i16`
///
/// 16 _is not_ a multiple of 3, hence the conditions are _not met_.
///
/// NOTE: This method assumes that common conversion preconditions are met. In
/// particular, `containerTy` is assumed to be a
/// multi-byte scalar type (e.g., i8, i16, i32).
static LogicalResult alignedConversionPrecondition(PatternRewriter &rewriter,
                                                   VectorType subByteVecTy,
                                                   Type containerTy,
                                                   Operation *op) {
  assert(containerTy.isIntOrFloat() &&
         "container element type is not a scalar");

  // TODO: This is validating the inputs rather than checking the conditions
  // documented above. Replace with an assert.
  if (!subByteVecTy)
    return rewriter.notifyMatchFailure(op, "not a vector!");

  unsigned subByteBits = subByteVecTy.getElementTypeBitWidth();
  unsigned containerBits = containerTy.getIntOrFloatBitWidth();

  // Enforced by the common pre-conditions.
  assert(containerBits % 8 == 0 && "Not a multi-byte scalar type!");

  // TODO: Add support other widths (when/if needed)
  if (subByteBits != 2 && subByteBits != 4)
    return rewriter.notifyMatchFailure(
        op, "only 2-bit and 4-bit sub-byte type is supported at this moment");

  // Condition 1 ("per-element" alignment)
  if (containerBits % subByteBits != 0)
    return rewriter.notifyMatchFailure(op, "unalagined element types");

  // Condition 2 ("full" alignment)
  if (!fitsInMultiByteContainerTy(subByteVecTy, containerTy))
    return rewriter.notifyMatchFailure(
        op, "not possible to fit this sub-byte vector type into a vector of "
            "the given multi-byte type");

  return success();
}

SmallVector<BitCastRewriter::Metadata>
BitCastRewriter::precomputeMetadata(IntegerType shuffledElementType) {
  SmallVector<BitCastRewriter::Metadata> result;
  for (int64_t shuffleIdx = 0, e = enumerator.getMaxNumberOfEntries();
       shuffleIdx < e; ++shuffleIdx) {
    SmallVector<int64_t> shuffles;
    SmallVector<Attribute> masks, shiftRightAmounts, shiftLeftAmounts;

    // Create the attribute quantities for the shuffle / mask / shift ops.
    for (auto &srcEltRangeList : enumerator.sourceElementRanges) {
      int64_t sourceElement = (shuffleIdx < (int64_t)srcEltRangeList.size())
                                  ? srcEltRangeList[shuffleIdx].sourceElementIdx
                                  : 0;
      shuffles.push_back(sourceElement);

      int64_t bitLo = (shuffleIdx < (int64_t)srcEltRangeList.size())
                          ? srcEltRangeList[shuffleIdx].sourceBitBegin
                          : 0;
      int64_t bitHi = (shuffleIdx < (int64_t)srcEltRangeList.size())
                          ? srcEltRangeList[shuffleIdx].sourceBitEnd
                          : 0;
      IntegerAttr mask = IntegerAttr::get(
          shuffledElementType,
          llvm::APInt::getBitsSet(shuffledElementType.getIntOrFloatBitWidth(),
                                  bitLo, bitHi));
      masks.push_back(mask);

      int64_t shiftRight = bitLo;
      shiftRightAmounts.push_back(
          IntegerAttr::get(shuffledElementType, shiftRight));

      int64_t shiftLeft = srcEltRangeList.computeLeftShiftAmount(shuffleIdx);
      shiftLeftAmounts.push_back(
          IntegerAttr::get(shuffledElementType, shiftLeft));
    }

    result.push_back({shuffles, masks, shiftRightAmounts, shiftLeftAmounts});
  }
  return result;
}

Value BitCastRewriter::genericRewriteStep(
    PatternRewriter &rewriter, Location loc, Value initialValue,
    Value runningResult, const BitCastRewriter::Metadata &metadata) {
  // Create vector.shuffle from the metadata.
  auto shuffleOp = vector::ShuffleOp::create(rewriter, loc, initialValue,
                                             initialValue, metadata.shuffles);

  // Intersect with the mask.
  VectorType shuffledVectorType = shuffleOp.getResultVectorType();
  auto constOp = arith::ConstantOp::create(
      rewriter, loc,
      DenseElementsAttr::get(shuffledVectorType, metadata.masks));
  Value andValue = arith::AndIOp::create(rewriter, loc, shuffleOp, constOp);

  // Align right on 0.
  auto shiftRightConstantOp = arith::ConstantOp::create(
      rewriter, loc,
      DenseElementsAttr::get(shuffledVectorType, metadata.shiftRightAmounts));
  Value shiftedRight =
      arith::ShRUIOp::create(rewriter, loc, andValue, shiftRightConstantOp);

  // Shift bits left into their final position.
  auto shiftLeftConstantOp = arith::ConstantOp::create(
      rewriter, loc,
      DenseElementsAttr::get(shuffledVectorType, metadata.shiftLeftAmounts));
  Value shiftedLeft =
      arith::ShLIOp::create(rewriter, loc, shiftedRight, shiftLeftConstantOp);

  runningResult =
      runningResult
          ? arith::OrIOp::create(rewriter, loc, runningResult, shiftedLeft)
          : shiftedLeft;

  return runningResult;
}

/// Bitcasts the aligned `subByteVec` vector to a vector of i8.
/// Where aligned means it satisfies the alignedConversionPreconditions.
///
/// Example:
/// vector<16x16xi2> -> vector<16x4xi8>
/// vector<16x16xi4> -> vector<16x8xi8>
static Value bitcastSubByteVectorToI8(PatternRewriter &rewriter, Location loc,
                                      Value subByteVec) {
  auto srcVecType = cast<VectorType>(subByteVec.getType());
  int64_t srcBitwidth = srcVecType.getElementType().getIntOrFloatBitWidth();
  assert(8 % srcBitwidth == 0 &&
         "Unsupported sub-byte type (not a divisor of i8)");
  int64_t numSrcElemsPerByte = 8 / srcBitwidth;
  SmallVector<int64_t> vecShape(srcVecType.getShape());
  // Adjust last dimension of the vector, so the total size remains the same.
  vecShape.back() = vecShape.back() / numSrcElemsPerByte;
  auto i8VecType = VectorType::get(vecShape, rewriter.getI8Type());
  return vector::BitCastOp::create(rewriter, loc, i8VecType, subByteVec);
}

/// Extracts a signed N-bit sequence from each element of a vector of bytes,
/// starting at the specified bit index.
/// The `bitIdx` starts at 0 from the LSB and moves to the left.
///
/// Example for a single element:
/// Extract numBits=2 starting at bitIdx=2
/// src     = [0 | 1 | 0 | 1 | 1 | 1 | 1 | 0]
/// indices = [7 | 6 | 5 | 4 | 3 | 2 | 1 | 0]
/// target  = [.   .   .   .   ^   ^   .   .]
///
/// The target sequence is [11](decimal=-1) as signed 2-bit integer.
/// So the result should be [11 11 11 11](decimal=-1) as signed 8-bit integer.
///
/// src     =                         [01 01 11 10]
/// shl     = arith.shl(src, 4)    -> [11 10 00 00]
/// result  = arith.shrsi(shl, 6)  -> [11 11 11 11]
static Value extractNBitsPerByteAndSignExtendToI8(PatternRewriter &rewriter,
                                                  Location loc, Value src,
                                                  int bitIdx, int numBits) {
  auto srcType = cast<VectorType>(src.getType());
  Value shl = src;
  int8_t bitsToShiftLeft = 8 - numBits - bitIdx;
  assert(bitIdx >= 0 && bitsToShiftLeft >= 0 && numBits > 0 && numBits <= 8 &&
         "Invalid bitIdx range");
  if (bitsToShiftLeft != 0) {
    Value shiftLeftValues = arith::ConstantOp::create(
        rewriter, loc, DenseElementsAttr::get(srcType, bitsToShiftLeft));
    shl = arith::ShLIOp::create(rewriter, loc, src, shiftLeftValues);
  }

  int8_t bitsToShiftRight = 8 - numBits;
  Value shiftRightValues = arith::ConstantOp::create(
      rewriter, loc, DenseElementsAttr::get(srcType, bitsToShiftRight));
  Value shr = arith::ShRSIOp::create(rewriter, loc, shl, shiftRightValues);
  return shr;
}

/// Extracts an unsigned N-bit sequence from each element of a vector of bytes,
/// starting at the specified bit index.
/// The `bitIdx` starts at 0 from the LSB and moves to the left.
///
/// Example for a single element:
/// Extract numBits=2 starting at bitIdx=2
/// src     = [0 | 1 | 0 | 1 | 1 | 0 | 1 | 0]
/// indices = [7 | 6 | 5 | 4 | 3 | 2 | 1 | 0]
/// target  = [.   .   .   .   ^   ^   .   .]
///
/// The target sequence is [10](decimal=2) as unsigned 2-bit integer.
/// So the result should be [00 00 00 10](decimal=2) as unsigned 8-bit integer.
///
/// src                            = [01 01 10 10]
/// mask                           = [00 00 00 11]
/// shr    = arith.shrui(src, 2)   = [00 01 01 10]
/// result = arith.andi(shr, mask) = [00 00 00 10]
/// NOTE: Similarly to extractNBitsPerByteAndSignExtendToI8, this could be
/// achieved by using arith::ShLIOp + arith::ShRUIOp instead of the masking.
/// However, by using arith::ShRUIOp + arith::AndIOp, we are eliminating shift
/// left when the index is 0.
static Value extractNBitsPerByteAndExtendToI8(PatternRewriter &rewriter,
                                              Location loc, Value src,
                                              int bitIdx, int numBits) {
  assert(bitIdx >= 0 && bitIdx <= 8 - numBits && numBits > 0 && numBits <= 8 &&
         "Invalid bitIdx range");
  auto srcType = cast<VectorType>(src.getType());
  int8_t bitsToShiftRight = bitIdx;
  Value shr = src;
  if (bitsToShiftRight != 0) {
    Value shiftRightValues = arith::ConstantOp::create(
        rewriter, loc, DenseElementsAttr::get(srcType, bitsToShiftRight));
    shr = arith::ShRUIOp::create(rewriter, loc, src, shiftRightValues);
  }
  if (bitIdx + numBits == 8) {
    return shr;
  }
  uint8_t lowBitsMask = (1 << numBits) - 1;
  Value lowBitsMaskValues = arith::ConstantOp::create(
      rewriter, loc, DenseElementsAttr::get(srcType, lowBitsMask));
  return arith::AndIOp::create(rewriter, loc, shr, lowBitsMaskValues);
}

using ExtractNBitsFn =
    std::function<Value(PatternRewriter &, Location, Value, int, int)>;

/// Rewrite the i4 -> i8  extension into a sequence of shuffles and
/// bitwise ops to avoid leaving LLVM to scramble with peephole optimizations.
static Value rewriteI4ToI8Ext(PatternRewriter &rewriter, Location loc,
                              Value srcValue, const ExtractNBitsFn &extFn) {
  [[maybe_unused]] auto srcVecType = cast<VectorType>(srcValue.getType());
  assert(srcVecType.getElementType().isSignlessInteger(4) &&
         "Expected i4 type");

  // 1. Generate a bitcast vector<Xxi4> -> vector<X/2xi8>.
  Value i8Vector = bitcastSubByteVectorToI8(rewriter, loc, srcValue);

  // 2. Extend i4 elements to i8 elements. Low i4 elemens of each
  // byte are place in one vector and the high i4 elements in another vector.
  Value low = extFn(rewriter, loc, i8Vector, 0, 4);
  Value high = extFn(rewriter, loc, i8Vector, 4, 4);

  // 3. Interleave low and high i8 elements.
  return vector::InterleaveOp::create(rewriter, loc, low, high);
}

/// Rewrite the i2 -> i8  extension into a sequence of shuffles and
/// bitwise ops to avoid leaving LLVM to scramble with peephole optimizations.
static Value rewriteI2ToI8Ext(PatternRewriter &rewriter, Location loc,
                              Value srcValue, const ExtractNBitsFn &extFn) {
  [[maybe_unused]] VectorType srcVecType = cast<VectorType>(srcValue.getType());
  assert(srcVecType.getElementType().isSignlessInteger(2) &&
         "Expected i2 type");

  // 1. Generate a bitcast vector<Xxi2> -> vector<X/2xi8>.
  Value i8Vector = bitcastSubByteVectorToI8(rewriter, loc, srcValue);

  // 2. Extract each i2 element
  // Positon 0 (bits 0-1)
  Value vec0 = extFn(rewriter, loc, i8Vector, 0, 2);
  // Position 1 (bits 2-3)
  Value vec1 = extFn(rewriter, loc, i8Vector, 2, 2);
  // Position 2 (bits 4-5)
  Value vec2 = extFn(rewriter, loc, i8Vector, 4, 2);
  // Position 3 (bits 6-7)
  Value vec3 = extFn(rewriter, loc, i8Vector, 6, 2);

  // 3. Interleave all 4 elements by first interleaving
  // even elements and then odd
  // vec0  = [0,0,0,0],...
  // vec1  = [1,1,1,1],...
  // vec2  = [2,2,2,2],...
  // vec3  = [3,3,3,3],...
  // 02    = [0,2,0,2,0,2,0,2],...
  // 13    = [1,3,1,3,1,3,1,3],...
  // 0213  = [0,1,2,3,...],...
  Value interleave02 = vector::InterleaveOp::create(rewriter, loc, vec0, vec2);
  Value interleave13 = vector::InterleaveOp::create(rewriter, loc, vec1, vec3);
  return vector::InterleaveOp::create(rewriter, loc, interleave02,
                                      interleave13);
}

/// Rewrite the i8 -> i4 truncation into a deinterleave and series of bitwise
/// ops to avoid leaving LLVM to scramble with peephole optimizations.
static Value rewriteI8ToI4Trunc(PatternRewriter &rewriter, Location loc,
                                Value srcValue) {
  VectorType srcVecType = cast<VectorType>(srcValue.getType());
  assert(srcVecType.getElementType().isSignlessInteger(8) &&
         "Expected i8 type");

  // 1. De-interleave low and high i8 elements.
  auto deinterleaveOp = vector::DeinterleaveOp::create(rewriter, loc, srcValue);

  // 2. Zero out the upper side of each low i8 element.
  constexpr int8_t i8LowBitMask = 0x0F;
  VectorType deinterI8VecType = deinterleaveOp.getResultVectorType();
  Value zeroOutMask = arith::ConstantOp::create(
      rewriter, loc, DenseElementsAttr::get(deinterI8VecType, i8LowBitMask));
  Value zeroOutLow = arith::AndIOp::create(
      rewriter, loc, deinterleaveOp.getRes1(), zeroOutMask);

  // 3. Move high i4 values to upper side of the byte.
  constexpr int8_t bitsToShift = 4;
  auto shiftValues = arith::ConstantOp::create(
      rewriter, loc, DenseElementsAttr::get(deinterI8VecType, bitsToShift));
  Value shlHigh = arith::ShLIOp::create(rewriter, loc, deinterleaveOp.getRes2(),
                                        shiftValues);

  // 4. Merge high and low i4 values.
  auto mergedHiLowOp = arith::OrIOp::create(rewriter, loc, zeroOutLow, shlHigh);

  // 5. Generate a bitcast vector<Xxi8> -> vector<2Xxi4>.
  auto i4VecType = srcVecType.cloneWith(std::nullopt, rewriter.getI4Type());
  return vector::BitCastOp::create(rewriter, loc, i4VecType, mergedHiLowOp);
}

namespace {
/// Rewrite bitcast(trunci) to a sequence of shuffles and bitwise ops that take
/// advantage of high-level information to avoid leaving LLVM to scramble with
/// peephole optimizations.
struct RewriteBitCastOfTruncI : OpRewritePattern<vector::BitCastOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(vector::BitCastOp bitCastOp,
                                PatternRewriter &rewriter) const override {
    // The source must be a trunc op.
    auto truncOp =
        bitCastOp.getSource().template getDefiningOp<arith::TruncIOp>();
    if (!truncOp)
      return rewriter.notifyMatchFailure(bitCastOp, "not a trunci source");

    // Set up the BitCastRewriter and verify the precondition.
    VectorType sourceVectorType = bitCastOp.getSourceVectorType();
    VectorType targetVectorType = bitCastOp.getResultVectorType();
    BitCastRewriter bcr(sourceVectorType, targetVectorType);
    if (failed(bcr.commonPrecondition(rewriter, targetVectorType, bitCastOp)))
      return failure();

    // Perform the rewrite.
    Value truncValue = truncOp.getIn();
    auto shuffledElementType =
        cast<IntegerType>(getElementTypeOrSelf(truncValue.getType()));
    Value runningResult;
    for (const BitCastRewriter ::Metadata &metadata :
         bcr.precomputeMetadata(shuffledElementType)) {
      runningResult = bcr.genericRewriteStep(
          rewriter, bitCastOp->getLoc(), truncValue, runningResult, metadata);
    }

    // Finalize the rewrite.
    bool narrowing = targetVectorType.getElementTypeBitWidth() <=
                     shuffledElementType.getIntOrFloatBitWidth();
    if (narrowing) {
      if (runningResult.getType() == bitCastOp.getResultVectorType()) {
        rewriter.replaceOp(bitCastOp, runningResult);
      } else {
        rewriter.replaceOpWithNewOp<arith::TruncIOp>(
            bitCastOp, bitCastOp.getResultVectorType(), runningResult);
      }
    } else {
      if (runningResult.getType() == bitCastOp.getResultVectorType()) {
        rewriter.replaceOp(bitCastOp, runningResult);
      } else {
        rewriter.replaceOpWithNewOp<arith::ExtUIOp>(
            bitCastOp, bitCastOp.getResultVectorType(), runningResult);
      }
    }

    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// RewriteExtOfBitCast
//===----------------------------------------------------------------------===//

namespace {
/// Rewrite ext{s,u}i(bitcast) to a sequence of shuffles and bitwise ops that
/// take advantage of high-level information to avoid leaving LLVM to scramble
/// with peephole optimizations.
template <typename ExtOpType>
struct RewriteExtOfBitCast : OpRewritePattern<ExtOpType> {
  using OpRewritePattern<ExtOpType>::OpRewritePattern;

  RewriteExtOfBitCast(MLIRContext *context, PatternBenefit benefit)
      : OpRewritePattern<ExtOpType>(context, benefit) {}

  LogicalResult matchAndRewrite(ExtOpType extOp,
                                PatternRewriter &rewriter) const override {
    // The source must be a bitcast op.
    auto bitCastOp = extOp.getIn().template getDefiningOp<vector::BitCastOp>();
    if (!bitCastOp)
      return rewriter.notifyMatchFailure(extOp, "not a bitcast source");

    // Set up the BitCastRewriter and verify the precondition.
    VectorType sourceVectorType = bitCastOp.getSourceVectorType();
    VectorType targetVectorType = bitCastOp.getResultVectorType();
    BitCastRewriter bcr(sourceVectorType, targetVectorType);
    if (failed(bcr.commonPrecondition(
            rewriter, cast<VectorType>(extOp.getOut().getType()), bitCastOp)))
      return failure();

    // Perform the rewrite.
    Value runningResult;
    Value sourceValue = bitCastOp.getSource();
    auto shuffledElementType =
        cast<IntegerType>(getElementTypeOrSelf(sourceValue.getType()));
    for (const BitCastRewriter::Metadata &metadata :
         bcr.precomputeMetadata(shuffledElementType)) {
      runningResult = bcr.genericRewriteStep(
          rewriter, bitCastOp->getLoc(), sourceValue, runningResult, metadata);
    }

    // Finalize the rewrite.
    bool narrowing =
        cast<VectorType>(extOp.getOut().getType()).getElementTypeBitWidth() <=
        shuffledElementType.getIntOrFloatBitWidth();
    if (narrowing) {
      rewriter.replaceOpWithNewOp<arith::TruncIOp>(
          extOp, cast<VectorType>(extOp.getOut().getType()), runningResult);
    } else {
      rewriter.replaceOpWithNewOp<ExtOpType>(
          extOp, cast<VectorType>(extOp.getOut().getType()), runningResult);
    }

    return success();
  }
};

/// Rewrite the i4 -> i8 part of any conversion into a sequence of shuffles and
/// bitwise ops that take advantage of high-level information to avoid leaving
/// LLVM to scramble with peephole optimizations. Templated to choose between
/// signed and unsigned conversions.
///
/// EXAMPLE 1 (signed):
///    arith.extsi %in : vector<8xi4> to vector<8xi32>
/// is rewriten as:
///    %0 = vector.bitcast %in : vector<8xi4> to vector<4xi8>
///    %1 = arith.shli %0, 4 : vector<4xi8>
///    %2 = arith.shrsi %1, 4 : vector<4xi8>
///    %3 = arith.shrsi %0, 4 : vector<4xi8>
///    %4 = vector.interleave %2, %3 : vector<4xi8> -> vector<8xi8>
///    %5 = arith.extsi %4 : vector<8xi8> to vector<8xi32>
///
/// EXAMPLE 2 (fp):
///    arith.sitofp %in : vector<8xi4> to vector<8xf32>
/// is rewriten as:
///    %0 = vector.bitcast %in : vector<8xi4> to vector<4xi8>
///    %1 = arith.shli %0, 4 : vector<4xi8>
///    %2 = arith.shrsi %1, 4 : vector<4xi8>
///    %3 = arith.shrsi %0, 4 : vector<4xi8>
///    %4 = vector.interleave %2, %3 : vector<4xi8> -> vector<8xi8>
///    %5 = arith.sitofp %4 : vector<8xi8> to vector<8xf32>
///
/// EXAMPLE 3 (unsigned):
///    arith.extui %in : vector<8xi4> to vector<8xi32>
///  is rewritten as:
///    %0 = vector.bitcast %in : vector<8xi4> to vector<4xi8>
///    %1 = arith.andi %0, 15 : vector<4xi8>
///    %2 = arith.shrui %0, 4 : vector<4xi8>
///    %3 = vector.interleave %1, %2 : vector<4xi8> -> vector<8xi8>
///    %4 = arith.extui %3 : vector<8xi8> to vector<8xi32>
///
template <typename ConversionOpType, bool isSigned>
struct RewriteAlignedSubByteIntExt : OpRewritePattern<ConversionOpType> {
  using OpRewritePattern<ConversionOpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(ConversionOpType conversionOp,
                                PatternRewriter &rewriter) const override {
    // Verify the preconditions.
    Value srcValue = conversionOp.getIn();
    VectorType srcVecType = dyn_cast<VectorType>(srcValue.getType());
    VectorType dstVecType = dyn_cast<VectorType>(conversionOp.getType());

    if (failed(
            commonConversionPrecondition(rewriter, dstVecType, conversionOp)))
      return failure();

    // Check general alignment preconditions.
    if (failed(alignedConversionPrecondition(
            rewriter, srcVecType,
            /*containerTy=*/rewriter.getI8Type(), conversionOp)))
      return failure();

    // Perform the rewrite.
    Location loc = conversionOp.getLoc();
    const auto &extFn = isSigned ? extractNBitsPerByteAndSignExtendToI8
                                 : extractNBitsPerByteAndExtendToI8;
    Value subByteExt;
    switch (srcVecType.getElementType().getIntOrFloatBitWidth()) {
    case 2:
      subByteExt = rewriteI2ToI8Ext(rewriter, loc, srcValue, extFn);
      break;
    case 4:
      subByteExt = rewriteI4ToI8Ext(rewriter, loc, srcValue, extFn);
      break;
    default:
      return failure();
    }

    // Finalize the rewrite.
    rewriter.replaceOpWithNewOp<ConversionOpType>(
        conversionOp, conversionOp.getType(), subByteExt);
    return success();
  }
};

/// Rewrite the i8 -> i4 part of any truncation into a deinterleave and
/// bitwise ops that take advantage of high-level information to avoid leaving
/// LLVM to scramble with peephole optimizations.
///
/// For example:
///    arith.trunci %in : vector<8xi32> to vector<8xi4>
///
/// is rewriten as:
///
///   %cst = arith.constant dense<15> : vector<4xi8>
///   %cst_0 = arith.constant dense<4> : vector<4xi8>
///   %0, %1 = vector.deinterleave %in : vector<8xi8>, vector<8xi8>
///   %2 = arith.andi %0, %cst : vector<4xi8>
///   %3 = arith.shli %1, %cst_0 : vector<4xi8>
///   %4 = arith.ori %2, %3 : vector<4xi8>
///   %5 = vector.bitcast %4 : vector<4xi8> to vector<8xi4>
///
struct RewriteAlignedSubByteIntTrunc : OpRewritePattern<arith::TruncIOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(arith::TruncIOp truncOp,
                                PatternRewriter &rewriter) const override {
    // Verify the preconditions.
    Value srcValue = truncOp.getIn();
    auto srcVecType = dyn_cast<VectorType>(srcValue.getType());
    auto dstVecType = dyn_cast<VectorType>(truncOp.getType());
    if (!srcVecType || !dstVecType)
      return failure();

    if (failed(commonConversionPrecondition(rewriter, srcVecType, truncOp)))
      return failure();

    // TODO: Add support for truncating to i2.
    if (dstVecType.getElementType().getIntOrFloatBitWidth() == 2)
      return failure();

    // Check general alignment preconditions. We invert the src/dst type order
    // to reuse the existing precondition logic.
    if (failed(alignedConversionPrecondition(
            rewriter, dstVecType,
            /*containerTy=*/rewriter.getI8Type(), truncOp)))
      return failure();

    // Create a new iX -> i8 truncation op.
    Location loc = truncOp.getLoc();
    auto i8VecType = srcVecType.cloneWith(std::nullopt, rewriter.getI8Type());
    Value i8TruncVal =
        arith::TruncIOp::create(rewriter, loc, i8VecType, srcValue);

    // Rewrite the i8 -> i4 truncation part.
    Value subByteTrunc = rewriteI8ToI4Trunc(rewriter, loc, i8TruncVal);

    // Finalize the rewrite.
    rewriter.replaceOp(truncOp, subByteTrunc);
    return success();
  }
};

/// Rewrite a sub-byte vector transpose into a sequence of instructions that
/// perform the transpose on wider (byte) element types.
///
/// EXAMPLE:
///   %0 = vector.transpose %a, [1, 0] : vector<8x16xi4> to vector<16x8xi4>
///
/// is rewritten as:
///
///   %0 = arith.extsi %arg0 : vector<8x16xi4> to vector<8x16xi8>
///   %1 = vector.transpose %0, [1, 0] : vector<8x16xi8> to vector<16x8xi8>
///   %2 = arith.trunci %1 : vector<16x8xi8> to vector<16x8xi4>
///
struct RewriteVectorTranspose : OpRewritePattern<vector::TransposeOp> {
  using Base::Base;

  RewriteVectorTranspose(MLIRContext *context, PatternBenefit benefit)
      : OpRewritePattern<vector::TransposeOp>(context, benefit) {}

  LogicalResult matchAndRewrite(vector::TransposeOp transposeOp,
                                PatternRewriter &rewriter) const override {
    // Precondition: sub-byte integer transpose.
    constexpr unsigned minNativeBitwidth = 8;
    VectorType srcSubByteVecType = transposeOp.getSourceVectorType();
    if (!srcSubByteVecType.getElementType().isSignlessInteger() ||
        srcSubByteVecType.getElementTypeBitWidth() >= minNativeBitwidth) {
      return rewriter.notifyMatchFailure(transposeOp,
                                         "not a sub-byte transpose");
    }

    // Perform the rewrite.
    Location loc = transposeOp.getLoc();
    // Signed/unsigned interpretation shouldn't matter here as we are just
    // transposing the elements and truncating them back to the original size.
    // TODO: Use unsigned extension (more efficient) when emulation or backend
    // support is available.
    auto srcNativeVecType = srcSubByteVecType.cloneWith(
        std::nullopt, rewriter.getIntegerType(minNativeBitwidth));
    Value extOp = arith::ExtSIOp::create(rewriter, loc, srcNativeVecType,
                                         transposeOp.getVector());
    Value newTranspose = vector::TransposeOp::create(
        rewriter, loc, extOp, transposeOp.getPermutation());
    VectorType dstSubByteVecType = transposeOp.getResultVectorType();
    rewriter.replaceOpWithNewOp<arith::TruncIOp>(transposeOp, dstSubByteVecType,
                                                 newTranspose);
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Public Interface Definition
//===----------------------------------------------------------------------===//

// The emulated type is inferred from the converted memref type.
void vector::populateVectorNarrowTypeEmulationPatterns(
    const arith::NarrowTypeEmulationConverter &typeConverter,
    RewritePatternSet &patterns, bool disableAtomicRMW) {
  // Populate `vector.*` conversion patterns.
  // TODO: #119553 support atomicity
  patterns.add<ConvertVectorLoad, ConvertVectorMaskedLoad,
               ConvertVectorMaskedStore, ConvertVectorTransferRead>(
      typeConverter, patterns.getContext());

  // Populate `vector.*` store conversion patterns. The caller can choose
  // to avoid emitting atomic operations and reduce it to read-modify-write
  // sequence for stores if it is known there are no thread contentions.
  patterns.insert<ConvertVectorStore>(patterns.getContext(), disableAtomicRMW);
}

void vector::populateVectorNarrowTypeRewritePatterns(
    RewritePatternSet &patterns, PatternBenefit benefit) {
  // TODO: Document what the emulated type is.
  patterns.add<RewriteBitCastOfTruncI, RewriteExtOfBitCast<arith::ExtUIOp>,
               RewriteExtOfBitCast<arith::ExtSIOp>>(patterns.getContext(),
                                                    benefit);

  // Patterns for aligned cases. We set higher priority as they are expected to
  // generate better performance for aligned cases.
  // The container type is always i8.
  patterns.add<RewriteAlignedSubByteIntExt<arith::ExtSIOp, /*isSigned=*/true>,
               RewriteAlignedSubByteIntExt<arith::SIToFPOp, /*isSigned=*/true>,
               RewriteAlignedSubByteIntTrunc>(patterns.getContext(),
                                              benefit.getBenefit() + 1);
  // The container type is always i8.
  patterns
      .add<RewriteAlignedSubByteIntExt<arith::ExtUIOp, /*isSigned=*/false>,
           RewriteAlignedSubByteIntExt<arith::UIToFPOp, /*isSigned=*/false>>(
          patterns.getContext(), benefit.getBenefit() + 1);
}

// The container type is always i8.
void vector::populateVectorTransposeNarrowTypeRewritePatterns(
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<RewriteVectorTranspose>(patterns.getContext(), benefit);
}

void vector::populateMemRefFlattenAndVectorNarrowTypeEmulationPatterns(
    arith::NarrowTypeEmulationConverter &typeConverter,
    RewritePatternSet &patterns) {
  memref::populateFlattenVectorOpsOnMemrefPatterns(patterns);
  vector::populateVectorNarrowTypeEmulationPatterns(typeConverter, patterns);
}
