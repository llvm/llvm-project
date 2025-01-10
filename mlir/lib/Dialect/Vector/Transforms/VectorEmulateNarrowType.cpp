//===- VectorEmulateNarrowType.cpp - Narrow type emulation ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements target-independent rewrites and utilities to emulate
// narrow types that are not supported by the target hardware, e.g. i4, using
// wider types, e.g. i8.
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
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <optional>

using namespace mlir;

#define DEBUG_TYPE "vector-narrow-type-emulation"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

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
      maskOp = extractOp.getVector().getDefiningOp();
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
                return rewriter.create<vector::CreateMaskOp>(loc, newMaskType,
                                                             newMaskOperands);
              })
          .Case<vector::ConstantMaskOp>(
              [&](auto constantMaskOp) -> std::optional<Operation *> {
                // Take the shape of mask, compress its trailing dimension:
                SmallVector<int64_t> maskDimSizes(
                    constantMaskOp.getMaskDimSizes());
                int64_t &maskIndex = maskDimSizes.back();
                maskIndex = llvm::divideCeil(numFrontPadElems + maskIndex,
                                             numSrcElemsPerDest);
                return rewriter.create<vector::ConstantMaskOp>(loc, newMaskType,
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
            return rewriter.create<arith::ConstantOp>(
                loc, DenseElementsAttr::get(newMaskType, compressedMaskValues));
          });

  if (!newMask)
    return failure();

  while (!extractOps.empty()) {
    newMask = rewriter.create<vector::ExtractOp>(
        loc, (*newMask)->getResults()[0], extractOps.back().getMixedPosition());
    extractOps.pop_back();
  }

  return *newMask;
}

/// Extracts 1-D subvector from a 1-D vector. It is a wrapper function for
/// emitting `vector.extract_strided_slice`.
static Value staticallyExtractSubvector(OpBuilder &rewriter, Location loc,
                                        VectorType extractType, Value source,
                                        int64_t frontOffset,
                                        int64_t subvecSize) {
  auto vectorType = cast<VectorType>(source.getType());
  assert((vectorType.getRank() == 1 && extractType.getRank() == 1) &&
         "expected 1-D source and destination types");
  (void)vectorType;
  assert(frontOffset + subvecSize <= vectorType.getNumElements() &&
         "subvector out of bounds");

  // do not need extraction if the subvector size is the same as the source
  if (vectorType.getNumElements() == subvecSize)
    return source;

  auto offsets = rewriter.getI64ArrayAttr({frontOffset});
  auto sizes = rewriter.getI64ArrayAttr({subvecSize});
  auto strides = rewriter.getI64ArrayAttr({1});
  return rewriter
      .create<vector::ExtractStridedSliceOp>(loc, extractType, source, offsets,
                                             sizes, strides)
      ->getResult(0);
}

/// Inserts 1-D subvector into a 1-D vector by overwriting the elements starting
/// at `offset`. it is a wrapper function for emitting
/// `vector.insert_strided_slice`.
static Value staticallyInsertSubvector(OpBuilder &rewriter, Location loc,
                                       Value src, Value dest, int64_t offset) {
  [[maybe_unused]] auto srcType = cast<VectorType>(src.getType());
  [[maybe_unused]] auto destType = cast<VectorType>(dest.getType());
  assert(srcType.getRank() == 1 && destType.getRank() == 1 &&
         "expected source and dest to be vector type");
  auto offsets = rewriter.getI64ArrayAttr({offset});
  auto strides = rewriter.getI64ArrayAttr({1});
  return rewriter.create<vector::InsertStridedSliceOp>(loc, dest.getType(), src,
                                                       dest, offsets, strides);
}

/// Extracts a 1-D subvector from a 1-D `source` vector, with index at `offset`
/// and size `numElementsToExtract`, and inserts into the `dest` vector. This
/// function emits multiple `vector.extract` and `vector.insert` ops, so only
/// use it when `offset` cannot be folded into a constant value.
static Value dynamicallyExtractSubVector(OpBuilder &rewriter, Location loc,
                                         TypedValue<VectorType> source,
                                         Value dest, OpFoldResult offset,
                                         int64_t numElementsToExtract) {
  for (int i = 0; i < numElementsToExtract; ++i) {
    Value extractLoc =
        (i == 0) ? offset.dyn_cast<Value>()
                 : rewriter.create<arith::AddIOp>(
                       loc, rewriter.getIndexType(), offset.dyn_cast<Value>(),
                       rewriter.create<arith::ConstantIndexOp>(loc, i));
    auto extractOp =
        rewriter.create<vector::ExtractOp>(loc, source, extractLoc);
    dest = rewriter.create<vector::InsertOp>(loc, extractOp, dest, i);
  }
  return dest;
}

/// Inserts a 1-D subvector into a 1-D `dest` vector at index `destOffsetVar`.
static Value dynamicallyInsertSubVector(RewriterBase &rewriter, Location loc,
                                        TypedValue<VectorType> source,
                                        Value dest, OpFoldResult destOffsetVar,
                                        size_t length) {
  assert(length > 0 && "length must be greater than 0");
  Value destOffsetVal =
      getValueOrCreateConstantIndexOp(rewriter, loc, destOffsetVar);
  for (size_t i = 0; i < length; ++i) {
    auto insertLoc = i == 0
                         ? destOffsetVal
                         : rewriter.create<arith::AddIOp>(
                               loc, rewriter.getIndexType(), destOffsetVal,
                               rewriter.create<arith::ConstantIndexOp>(loc, i));
    auto extractOp = rewriter.create<vector::ExtractOp>(loc, source, i);
    dest = rewriter.create<vector::InsertOp>(loc, extractOp, dest, insertLoc);
  }
  return dest;
}

/// Returns the op sequence for an emulated sub-byte data type vector load.
/// specifically, use `emulatedElemType` for loading a vector of `origElemType`.
/// The load location is given by `base` and `linearizedIndices`, and the
/// load size is given by `numEmulatedElementsToLoad`.
static TypedValue<VectorType>
emulatedVectorLoad(OpBuilder &rewriter, Location loc, Value base,
                   OpFoldResult linearizedIndices,
                   int64_t numEmultedElementsToLoad, Type origElemType,
                   Type emulatedElemType) {
  auto scale = emulatedElemType.getIntOrFloatBitWidth() /
               origElemType.getIntOrFloatBitWidth();
  auto newLoad = rewriter.create<vector::LoadOp>(
      loc, VectorType::get(numEmultedElementsToLoad, emulatedElemType), base,
      getValueOrCreateConstantIndexOp(rewriter, loc, linearizedIndices));
  return rewriter.create<vector::BitCastOp>(
      loc, VectorType::get(numEmultedElementsToLoad * scale, origElemType),
      newLoad);
}

namespace {

//===----------------------------------------------------------------------===//
// ConvertVectorStore
//===----------------------------------------------------------------------===//

struct ConvertVectorStore final : OpConversionPattern<vector::StoreOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // See #115653
    if (op.getValueToStore().getType().getRank() != 1)
      return rewriter.notifyMatchFailure(op,
                                         "only 1-D vectors are supported ATM");

    auto loc = op.getLoc();
    auto convertedType = cast<MemRefType>(adaptor.getBase().getType());
    Type oldElementType = op.getValueToStore().getType().getElementType();
    Type newElementType = convertedType.getElementType();
    int srcBits = oldElementType.getIntOrFloatBitWidth();
    int dstBits = newElementType.getIntOrFloatBitWidth();

    if (dstBits % srcBits != 0) {
      return rewriter.notifyMatchFailure(
          op, "only dstBits % srcBits == 0 supported");
    }
    int scale = dstBits / srcBits;

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

    auto origElements = op.getValueToStore().getType().getNumElements();
    if (origElements % scale != 0)
      return failure();

    auto stridedMetadata =
        rewriter.create<memref::ExtractStridedMetadataOp>(loc, op.getBase());

    OpFoldResult linearizedIndices;
    std::tie(std::ignore, linearizedIndices) =
        memref::getLinearizedMemRefOffsetAndSize(
            rewriter, loc, srcBits, dstBits,
            stridedMetadata.getConstifiedMixedOffset(),
            stridedMetadata.getConstifiedMixedSizes(),
            stridedMetadata.getConstifiedMixedStrides(),
            getAsOpFoldResult(adaptor.getIndices()));

    auto numElements = origElements / scale;
    auto bitCast = rewriter.create<vector::BitCastOp>(
        loc, VectorType::get(numElements, newElementType),
        op.getValueToStore());

    rewriter.replaceOpWithNewOp<vector::StoreOp>(
        op, bitCast.getResult(), adaptor.getBase(),
        getValueOrCreateConstantIndexOp(rewriter, loc, linearizedIndices));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertVectorMaskedStore
//===----------------------------------------------------------------------===//

struct ConvertVectorMaskedStore final
    : OpConversionPattern<vector::MaskedStoreOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::MaskedStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // See #115653
    if (op.getValueToStore().getType().getRank() != 1)
      return rewriter.notifyMatchFailure(op,
                                         "only 1-D vectors are supported ATM");

    auto loc = op.getLoc();
    auto convertedType = cast<MemRefType>(adaptor.getBase().getType());
    Type oldElementType = op.getValueToStore().getType().getElementType();
    Type newElementType = convertedType.getElementType();
    int srcBits = oldElementType.getIntOrFloatBitWidth();
    int dstBits = newElementType.getIntOrFloatBitWidth();

    if (dstBits % srcBits != 0) {
      return rewriter.notifyMatchFailure(
          op, "only dstBits % srcBits == 0 supported");
    }

    int scale = dstBits / srcBits;
    int origElements = op.getValueToStore().getType().getNumElements();
    if (origElements % scale != 0)
      return failure();

    auto stridedMetadata =
        rewriter.create<memref::ExtractStridedMetadataOp>(loc, op.getBase());
    OpFoldResult linearizedIndicesOfr;
    memref::LinearizedMemRefInfo linearizedInfo;
    std::tie(linearizedInfo, linearizedIndicesOfr) =
        memref::getLinearizedMemRefOffsetAndSize(
            rewriter, loc, srcBits, dstBits,
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
    FailureOr<Operation *> newMask =
        getCompressedMaskOp(rewriter, loc, op.getMask(), origElements, scale);
    if (failed(newMask))
      return failure();

    auto numElements = (origElements + scale - 1) / scale;
    auto newType = VectorType::get(numElements, newElementType);
    auto passThru = rewriter.create<arith::ConstantOp>(
        loc, newType, rewriter.getZeroAttr(newType));

    auto newLoad = rewriter.create<vector::MaskedLoadOp>(
        loc, newType, adaptor.getBase(), linearizedIndices,
        newMask.value()->getResult(0), passThru);

    auto newBitCastType = VectorType::get(numElements * scale, oldElementType);
    Value valueToStore =
        rewriter.create<vector::BitCastOp>(loc, newBitCastType, newLoad);
    valueToStore = rewriter.create<arith::SelectOp>(
        loc, op.getMask(), op.getValueToStore(), valueToStore);
    valueToStore =
        rewriter.create<vector::BitCastOp>(loc, newType, valueToStore);

    rewriter.replaceOpWithNewOp<vector::MaskedStoreOp>(
        op, adaptor.getBase(), linearizedIndices, newMask.value()->getResult(0),
        valueToStore);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertVectorLoad
//===----------------------------------------------------------------------===//

struct ConvertVectorLoad final : OpConversionPattern<vector::LoadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // See #115653
    if (op.getVectorType().getRank() != 1)
      return rewriter.notifyMatchFailure(op,
                                         "only 1-D vectors are supported ATM");

    auto loc = op.getLoc();
    auto convertedType = cast<MemRefType>(adaptor.getBase().getType());
    Type oldElementType = op.getType().getElementType();
    Type newElementType = convertedType.getElementType();
    int srcBits = oldElementType.getIntOrFloatBitWidth();
    int dstBits = newElementType.getIntOrFloatBitWidth();

    if (dstBits % srcBits != 0) {
      return rewriter.notifyMatchFailure(
          op, "only dstBits % srcBits == 0 supported");
    }
    int scale = dstBits / srcBits;

    // Adjust the number of elements to load when emulating narrow types,
    // and then cast back to the original type with vector.bitcast op.
    // Here only the 1-D vector load is considered, and the N-D memref types
    // should be linearized.
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
    bool isUnalignedEmulation = origElements % scale != 0;

    auto stridedMetadata =
        rewriter.create<memref::ExtractStridedMetadataOp>(loc, op.getBase());

    OpFoldResult linearizedIndices;
    memref::LinearizedMemRefInfo linearizedInfo;
    std::tie(linearizedInfo, linearizedIndices) =
        memref::getLinearizedMemRefOffsetAndSize(
            rewriter, loc, srcBits, dstBits,
            stridedMetadata.getConstifiedMixedOffset(),
            stridedMetadata.getConstifiedMixedSizes(),
            stridedMetadata.getConstifiedMixedStrides(),
            getAsOpFoldResult(adaptor.getIndices()));

    std::optional<int64_t> foldedIntraVectorOffset =
        isUnalignedEmulation
            ? getConstantIntValue(linearizedInfo.intraDataOffset)
            : 0;

    // Always load enough elements which can cover the original elements.
    int64_t maxintraDataOffset = foldedIntraVectorOffset.value_or(scale - 1);
    auto numElements =
        llvm::divideCeil(maxintraDataOffset + origElements, scale);
    Value result =
        emulatedVectorLoad(rewriter, loc, adaptor.getBase(), linearizedIndices,
                           numElements, oldElementType, newElementType);

    if (!foldedIntraVectorOffset) {
      auto resultVector = rewriter.create<arith::ConstantOp>(
          loc, op.getType(), rewriter.getZeroAttr(op.getType()));
      result = dynamicallyExtractSubVector(
          rewriter, loc, dyn_cast<TypedValue<VectorType>>(result), resultVector,
          linearizedInfo.intraDataOffset, origElements);
    } else if (isUnalignedEmulation) {
      result =
          staticallyExtractSubvector(rewriter, loc, op.getType(), result,
                                     *foldedIntraVectorOffset, origElements);
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertVectorMaskedLoad
//===----------------------------------------------------------------------===//

struct ConvertVectorMaskedLoad final
    : OpConversionPattern<vector::MaskedLoadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::MaskedLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // See #115653
    if (op.getVectorType().getRank() != 1)
      return rewriter.notifyMatchFailure(op,
                                         "only 1-D vectors are supported ATM");

    auto loc = op.getLoc();
    auto convertedType = cast<MemRefType>(adaptor.getBase().getType());
    Type oldElementType = op.getType().getElementType();
    Type newElementType = convertedType.getElementType();
    int srcBits = oldElementType.getIntOrFloatBitWidth();
    int dstBits = newElementType.getIntOrFloatBitWidth();

    if (dstBits % srcBits != 0) {
      return rewriter.notifyMatchFailure(
          op, "only dstBits % srcBits == 0 supported");
    }
    int scale = dstBits / srcBits;

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
    bool isUnalignedEmulation = origElements % scale != 0;

    auto stridedMetadata =
        rewriter.create<memref::ExtractStridedMetadataOp>(loc, op.getBase());
    OpFoldResult linearizedIndices;
    memref::LinearizedMemRefInfo linearizedInfo;
    std::tie(linearizedInfo, linearizedIndices) =
        memref::getLinearizedMemRefOffsetAndSize(
            rewriter, loc, srcBits, dstBits,
            stridedMetadata.getConstifiedMixedOffset(),
            stridedMetadata.getConstifiedMixedSizes(),
            stridedMetadata.getConstifiedMixedStrides(),
            getAsOpFoldResult(adaptor.getIndices()));

    std::optional<int64_t> foldedIntraVectorOffset =
        isUnalignedEmulation
            ? getConstantIntValue(linearizedInfo.intraDataOffset)
            : 0;

    int64_t maxIntraDataOffset = foldedIntraVectorOffset.value_or(scale - 1);
    FailureOr<Operation *> newMask = getCompressedMaskOp(
        rewriter, loc, op.getMask(), origElements, scale, maxIntraDataOffset);
    if (failed(newMask))
      return failure();

    Value passthru = op.getPassThru();

    auto numElements =
        llvm::divideCeil(maxIntraDataOffset + origElements, scale);
    auto loadType = VectorType::get(numElements, newElementType);
    auto newBitcastType = VectorType::get(numElements * scale, oldElementType);

    auto emptyVector = rewriter.create<arith::ConstantOp>(
        loc, newBitcastType, rewriter.getZeroAttr(newBitcastType));
    if (!foldedIntraVectorOffset) {
      passthru = dynamicallyInsertSubVector(
          rewriter, loc, dyn_cast<TypedValue<VectorType>>(passthru),
          emptyVector, linearizedInfo.intraDataOffset, origElements);
    } else if (isUnalignedEmulation) {
      passthru = staticallyInsertSubvector(rewriter, loc, passthru, emptyVector,
                                           *foldedIntraVectorOffset);
    }
    auto newPassThru =
        rewriter.create<vector::BitCastOp>(loc, loadType, passthru);

    // Generating the new masked load.
    auto newLoad = rewriter.create<vector::MaskedLoadOp>(
        loc, loadType, adaptor.getBase(),
        getValueOrCreateConstantIndexOp(rewriter, loc, linearizedIndices),
        newMask.value()->getResult(0), newPassThru);

    // Setting the part that originally was not effectively loaded from memory
    // to pass through.
    auto bitCast =
        rewriter.create<vector::BitCastOp>(loc, newBitcastType, newLoad);

    Value mask = op.getMask();
    auto newSelectMaskType =
        VectorType::get(numElements * scale, rewriter.getI1Type());
    // TODO: try to fold if op's mask is constant
    auto emptyMask = rewriter.create<arith::ConstantOp>(
        loc, newSelectMaskType, rewriter.getZeroAttr(newSelectMaskType));
    if (!foldedIntraVectorOffset) {
      mask = dynamicallyInsertSubVector(
          rewriter, loc, dyn_cast<TypedValue<VectorType>>(mask), emptyMask,
          linearizedInfo.intraDataOffset, origElements);
    } else if (isUnalignedEmulation) {
      mask = staticallyInsertSubvector(rewriter, loc, op.getMask(), emptyMask,
                                       *foldedIntraVectorOffset);
    }

    Value result =
        rewriter.create<arith::SelectOp>(loc, mask, bitCast, passthru);
    if (!foldedIntraVectorOffset) {
      result = dynamicallyExtractSubVector(
          rewriter, loc, dyn_cast<TypedValue<VectorType>>(result),
          op.getPassThru(), linearizedInfo.intraDataOffset, origElements);
    } else if (isUnalignedEmulation) {
      result =
          staticallyExtractSubvector(rewriter, loc, op.getType(), result,
                                     *foldedIntraVectorOffset, origElements);
    }
    rewriter.replaceOp(op, result);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertVectorTransferRead
//===----------------------------------------------------------------------===//

struct ConvertVectorTransferRead final
    : OpConversionPattern<vector::TransferReadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::TransferReadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // See #115653
    if (op.getVectorType().getRank() != 1)
      return rewriter.notifyMatchFailure(op,
                                         "only 1-D vectors are supported ATM");

    auto loc = op.getLoc();
    auto convertedType = cast<MemRefType>(adaptor.getSource().getType());
    Type oldElementType = op.getType().getElementType();
    Type newElementType = convertedType.getElementType();
    int srcBits = oldElementType.getIntOrFloatBitWidth();
    int dstBits = newElementType.getIntOrFloatBitWidth();

    if (dstBits % srcBits != 0) {
      return rewriter.notifyMatchFailure(
          op, "only dstBits % srcBits == 0 supported");
    }
    int scale = dstBits / srcBits;

    auto origElements = op.getVectorType().getNumElements();

    bool isUnalignedEmulation = origElements % scale != 0;

    auto newPadding = rewriter.create<arith::ExtUIOp>(loc, newElementType,
                                                      adaptor.getPadding());

    auto stridedMetadata =
        rewriter.create<memref::ExtractStridedMetadataOp>(loc, op.getSource());

    OpFoldResult linearizedIndices;
    memref::LinearizedMemRefInfo linearizedInfo;
    std::tie(linearizedInfo, linearizedIndices) =
        memref::getLinearizedMemRefOffsetAndSize(
            rewriter, loc, srcBits, dstBits,
            stridedMetadata.getConstifiedMixedOffset(),
            stridedMetadata.getConstifiedMixedSizes(),
            stridedMetadata.getConstifiedMixedStrides(),
            getAsOpFoldResult(adaptor.getIndices()));

    std::optional<int64_t> foldedIntraVectorOffset =
        isUnalignedEmulation
            ? getConstantIntValue(linearizedInfo.intraDataOffset)
            : 0;

    int64_t maxIntraDataOffset = foldedIntraVectorOffset.value_or(scale - 1);
    auto numElements =
        llvm::divideCeil(maxIntraDataOffset + origElements, scale);

    auto newRead = rewriter.create<vector::TransferReadOp>(
        loc, VectorType::get(numElements, newElementType), adaptor.getSource(),
        getValueOrCreateConstantIndexOp(rewriter, loc, linearizedIndices),
        newPadding);

    auto bitCast = rewriter.create<vector::BitCastOp>(
        loc, VectorType::get(numElements * scale, oldElementType), newRead);

    Value result = bitCast->getResult(0);
    if (!foldedIntraVectorOffset) {
      auto zeros = rewriter.create<arith::ConstantOp>(
          loc, op.getType(), rewriter.getZeroAttr(op.getType()));
      result = dynamicallyExtractSubVector(rewriter, loc, bitCast, zeros,
                                           linearizedInfo.intraDataOffset,
                                           origElements);
    } else if (isUnalignedEmulation) {
      result =
          staticallyExtractSubvector(rewriter, loc, op.getType(), result,
                                     *foldedIntraVectorOffset, origElements);
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
  LDBG("sourceVectorType: " << sourceVectorType);

  int64_t targetBitWidth = targetVectorType.getElementTypeBitWidth();
  int64_t mostMinorTargetDim = targetVectorType.getShape().back();
  LDBG("targetVectorType: " << targetVectorType);

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
  LDBG("\n" << enumerator.sourceElementRanges);
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

/// Verify that `subByteVecType` and `dstType` are aligned. Alignment
/// means that:
///   1. The `dstType` element type is a multiple of the
///   `srcVectorOfSubByteType` element type (e.g. i4 vs i8 is OK, but i3 vs i8
///   is not supported). Let this multiple be `N`.
///   2. The number of the (trailing) elements in `srcVectorOfSubByteType` is a
///   multiple of `N` from 1. (e.g., when targetting i8, 2xi4 is OK, but 3xi4 is
///   not supported).
///
/// NOTE: This method assumes that common conversion preconditions are met. In
/// particular, the element type of `dstType` is assumed to be a multi-byte
/// type (e.g. i8, i16, i32).
static LogicalResult alignedConversionPrecondition(PatternRewriter &rewriter,
                                                   VectorType subByteVecType,
                                                   VectorType dstType,
                                                   Operation *op) {
  if (!subByteVecType || !dstType)
    return rewriter.notifyMatchFailure(op, "Not a supported aligned case");
  unsigned srcElemBitwidth = subByteVecType.getElementTypeBitWidth();
  unsigned dstElemBitwidth = dstType.getElementTypeBitWidth();

  // Only {s}i4 -> (size_of({{s}i/f}) >= 8) are supported for now.
  if (srcElemBitwidth != 4 || dstElemBitwidth < 8 ||
      (dstElemBitwidth % srcElemBitwidth) != 0)
    return rewriter.notifyMatchFailure(op, "Not a supported aligned case");

  const int numSrcElemsPerDestElem = dstElemBitwidth / srcElemBitwidth;
  if ((subByteVecType.getShape().back() % numSrcElemsPerDestElem) != 0)
    return rewriter.notifyMatchFailure(
        op, "Not an even number of i4 elements in trailing dim");

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
  auto shuffleOp = rewriter.create<vector::ShuffleOp>(
      loc, initialValue, initialValue, metadata.shuffles);

  // Intersect with the mask.
  VectorType shuffledVectorType = shuffleOp.getResultVectorType();
  auto constOp = rewriter.create<arith::ConstantOp>(
      loc, DenseElementsAttr::get(shuffledVectorType, metadata.masks));
  Value andValue = rewriter.create<arith::AndIOp>(loc, shuffleOp, constOp);

  // Align right on 0.
  auto shiftRightConstantOp = rewriter.create<arith::ConstantOp>(
      loc,
      DenseElementsAttr::get(shuffledVectorType, metadata.shiftRightAmounts));
  Value shiftedRight =
      rewriter.create<arith::ShRUIOp>(loc, andValue, shiftRightConstantOp);

  // Shift bits left into their final position.
  auto shiftLeftConstantOp = rewriter.create<arith::ConstantOp>(
      loc,
      DenseElementsAttr::get(shuffledVectorType, metadata.shiftLeftAmounts));
  Value shiftedLeft =
      rewriter.create<arith::ShLIOp>(loc, shiftedRight, shiftLeftConstantOp);

  runningResult =
      runningResult
          ? rewriter.create<arith::OrIOp>(loc, runningResult, shiftedLeft)
          : shiftedLeft;

  return runningResult;
}

/// Rewrite the i4 -> i8 signed extension into a sequence of shuffles and
/// bitwise ops that take advantage of high-level information to avoid leaving
/// LLVM to scramble with peephole optimizations.
static Value rewriteI4ToI8SignedExt(PatternRewriter &rewriter, Location loc,
                                    Value srcValue) {
  VectorType srcVecType = cast<VectorType>(srcValue.getType());
  assert(srcVecType.getElementType().isSignlessInteger(4) &&
         "Expected i4 type");

  // 1. Generate a bitcast vector<Xxi4> -> vector<X/2xi8>.
  SmallVector<int64_t> i8VecShape = llvm::to_vector(srcVecType.getShape());
  constexpr int64_t i4Toi8BitwidthFactor = 2;
  i8VecShape.back() = i8VecShape.back() / i4Toi8BitwidthFactor;
  auto i8VecType = VectorType::get(i8VecShape, rewriter.getI8Type());
  Value i8Vector = rewriter.create<vector::BitCastOp>(loc, i8VecType, srcValue);

  // 2. Extend i4 elements to i8 elements using shifts. Low i4 elemens of each
  // byte are place in one vector and the high i4 elements in another vector.
  constexpr int8_t bitsToShift = 4;
  auto shiftValues = rewriter.create<arith::ConstantOp>(
      loc, DenseElementsAttr::get(i8VecType, bitsToShift));
  Value shl = rewriter.create<arith::ShLIOp>(loc, i8Vector, shiftValues);
  Value low = rewriter.create<arith::ShRSIOp>(loc, shl, shiftValues);
  Value high = rewriter.create<arith::ShRSIOp>(loc, i8Vector, shiftValues);

  // 3. Interleave low and high i8 elements.
  return rewriter.create<vector::InterleaveOp>(loc, low, high);
}

/// Rewrite the i4 -> i8 unsigned extension into a sequence of shuffles and
/// bitwise ops that take advantage of high-level information to avoid leaving
/// LLVM to scramble with peephole optimizations.
static Value rewriteI4ToI8UnsignedExt(PatternRewriter &rewriter, Location loc,
                                      Value srcValue) {
  VectorType srcVecType = cast<VectorType>(srcValue.getType());
  assert(srcVecType.getElementType().isSignlessInteger(4) &&
         "Expected i4 type");

  // 1. Generate a bitcast vector<Xxi4> -> vector<X/2xi8>.
  SmallVector<int64_t> i8VecShape = llvm::to_vector(srcVecType.getShape());
  constexpr int64_t i4Toi8BitwidthFactor = 2;
  i8VecShape.back() = i8VecShape.back() / i4Toi8BitwidthFactor;
  auto i8VecType = VectorType::get(i8VecShape, rewriter.getI8Type());
  Value i8Vector = rewriter.create<vector::BitCastOp>(loc, i8VecType, srcValue);

  // 2 Extend the i4 elements using shifts & masking. Low i4 elements of each
  //  byte are placed in one vector and the high i4 elements in another vector.
  constexpr uint8_t lowBitsMask = 15; // Equivalent to [00001111] bit mask
  auto lowBitsMaskValues = rewriter.create<arith::ConstantOp>(
      loc, DenseElementsAttr::get(i8VecType, lowBitsMask));
  Value low = rewriter.create<arith::AndIOp>(loc, i8VecType, i8Vector,
                                             lowBitsMaskValues);
  constexpr int8_t highBitsToShift = 4;
  auto highShiftValues = rewriter.create<arith::ConstantOp>(
      loc, DenseElementsAttr::get(i8VecType, highBitsToShift));
  Value high = rewriter.create<arith::ShRUIOp>(loc, i8Vector, highShiftValues);

  // 3. Interleave low and high i8 elements.
  return rewriter.create<vector::InterleaveOp>(loc, low, high);
}

/// Rewrite the i8 -> i4 truncation into a deinterleave and series of bitwise
/// ops that take advantage of high-level information to avoid leaving LLVM to
/// scramble with peephole optimizations.
static Value rewriteI8ToI4Trunc(PatternRewriter &rewriter, Location loc,
                                Value srcValue) {
  VectorType srcVecType = cast<VectorType>(srcValue.getType());
  assert(srcVecType.getElementType().isSignlessInteger(8) &&
         "Expected i8 type");

  // 1. De-interleave low and high i8 elements.
  auto deinterleaveOp = rewriter.create<vector::DeinterleaveOp>(loc, srcValue);

  // 2. Zero out the upper side of each low i8 element.
  constexpr int8_t i8LowBitMask = 0x0F;
  VectorType deinterI8VecType = deinterleaveOp.getResultVectorType();
  Value zeroOutMask = rewriter.create<arith::ConstantOp>(
      loc, DenseElementsAttr::get(deinterI8VecType, i8LowBitMask));
  Value zeroOutLow = rewriter.create<arith::AndIOp>(
      loc, deinterleaveOp.getRes1(), zeroOutMask);

  // 3. Move high i4 values to upper side of the byte.
  constexpr int8_t bitsToShift = 4;
  auto shiftValues = rewriter.create<arith::ConstantOp>(
      loc, DenseElementsAttr::get(deinterI8VecType, bitsToShift));
  Value shlHigh = rewriter.create<arith::ShLIOp>(loc, deinterleaveOp.getRes2(),
                                                 shiftValues);

  // 4. Merge high and low i4 values.
  auto mergedHiLowOp = rewriter.create<arith::OrIOp>(loc, zeroOutLow, shlHigh);

  // 5. Generate a bitcast vector<Xxi8> -> vector<2Xxi4>.
  auto i4VecType = srcVecType.cloneWith(std::nullopt, rewriter.getI4Type());
  return rewriter.create<vector::BitCastOp>(loc, i4VecType, mergedHiLowOp);
}

namespace {
/// Rewrite bitcast(trunci) to a sequence of shuffles and bitwise ops that take
/// advantage of high-level information to avoid leaving LLVM to scramble with
/// peephole optimizations.
struct RewriteBitCastOfTruncI : OpRewritePattern<vector::BitCastOp> {
  using OpRewritePattern::OpRewritePattern;

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
/// For example (signed):
///    arith.extsi %in : vector<8xi4> to vector<8xi32>
///      is rewriten as
///        %0 = vector.bitcast %in : vector<8xi4> to vector<4xi8>
///        %1 = arith.shli %0, 4 : vector<4xi8>
///        %2 = arith.shrsi %1, 4 : vector<4xi8>
///        %3 = arith.shrsi %0, 4 : vector<4xi8>
///        %4 = vector.interleave %2, %3 : vector<4xi8> -> vector<8xi8>
///        %5 = arith.extsi %4 : vector<8xi8> to vector<8xi32>
///
///    arith.sitofp %in : vector<8xi4> to vector<8xf32>
///      is rewriten as
///        %0 = vector.bitcast %in : vector<8xi4> to vector<4xi8>
///        %1 = arith.shli %0, 4 : vector<4xi8>
///        %2 = arith.shrsi %1, 4 : vector<4xi8>
///        %3 = arith.shrsi %0, 4 : vector<4xi8>
///        %4 = vector.interleave %2, %3 : vector<4xi8> -> vector<8xi8>
///        %5 = arith.sitofp %4 : vector<8xi8> to vector<8xf32>
///
/// Example (unsigned):
///    arith.extui %in : vector<8xi4> to vector<8xi32>
///      is rewritten as
///        %0 = vector.bitcast %in : vector<8xi4> to vector<4xi8>
///        %1 = arith.andi %0, 15 : vector<4xi8>
///        %2 = arith.shrui %0, 4 : vector<4xi8>
///        %3 = vector.interleave %1, %2 : vector<4xi8> -> vector<8xi8>
///        %4 = arith.extui %3 : vector<8xi8> to vector<8xi32>
///
template <typename ConversionOpType, bool isSigned>
struct RewriteAlignedSubByteIntExt : OpRewritePattern<ConversionOpType> {
  using OpRewritePattern<ConversionOpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(ConversionOpType conversionOp,
                                PatternRewriter &rewriter) const override {
    // Verify the preconditions.
    Value srcValue = conversionOp.getIn();
    auto srcVecType = dyn_cast<VectorType>(srcValue.getType());
    auto dstVecType = dyn_cast<VectorType>(conversionOp.getType());

    if (failed(
            commonConversionPrecondition(rewriter, dstVecType, conversionOp)))
      return failure();

    // Check general alignment preconditions.
    if (failed(alignedConversionPrecondition(rewriter, srcVecType, dstVecType,
                                             conversionOp)))
      return failure();

    // Perform the rewrite.
    Value subByteExt;
    if (isSigned) {
      subByteExt =
          rewriteI4ToI8SignedExt(rewriter, conversionOp.getLoc(), srcValue);
    } else {
      subByteExt =
          rewriteI4ToI8UnsignedExt(rewriter, conversionOp.getLoc(), srcValue);
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
///      is rewriten as
///
///        %cst = arith.constant dense<15> : vector<4xi8>
///        %cst_0 = arith.constant dense<4> : vector<4xi8>
///        %0, %1 = vector.deinterleave %in : vector<8xi8>, vector<8xi8>
///        %2 = arith.andi %0, %cst : vector<4xi8>
///        %3 = arith.shli %1, %cst_0 : vector<4xi8>
///        %4 = arith.ori %2, %3 : vector<4xi8>
///        %5 = vector.bitcast %4 : vector<4xi8> to vector<8xi4>
///
struct RewriteAlignedSubByteIntTrunc : OpRewritePattern<arith::TruncIOp> {
  using OpRewritePattern<arith::TruncIOp>::OpRewritePattern;

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

    // Check general alignment preconditions. We invert the src/dst type order
    // to reuse the existing precondition logic.
    if (failed(alignedConversionPrecondition(rewriter, dstVecType, srcVecType,
                                             truncOp)))
      return failure();

    // Create a new iX -> i8 truncation op.
    Location loc = truncOp.getLoc();
    auto i8VecType = srcVecType.cloneWith(std::nullopt, rewriter.getI8Type());
    Value i8TruncVal =
        rewriter.create<arith::TruncIOp>(loc, i8VecType, srcValue);

    // Rewrite the i8 -> i4 truncation part.
    Value subByteTrunc = rewriteI8ToI4Trunc(rewriter, loc, i8TruncVal);

    // Finalize the rewrite.
    rewriter.replaceOp(truncOp, subByteTrunc);
    return success();
  }
};

/// Rewrite a sub-byte vector transpose into a sequence of instructions that
/// perform the transpose on wider (byte) element types.
/// For example:
///   %0 = vector.transpose %a, [1, 0] : vector<8x16xi4> to vector<16x8xi4>
///
///   is rewritten as:
///
///   %0 = arith.extsi %arg0 : vector<8x16xi4> to vector<8x16xi8>
///   %1 = vector.transpose %0, [1, 0] : vector<8x16xi8> to vector<16x8xi8>
///   %2 = arith.trunci %1 : vector<16x8xi8> to vector<16x8xi4>
///
struct RewriteVectorTranspose : OpRewritePattern<vector::TransposeOp> {
  using OpRewritePattern<vector::TransposeOp>::OpRewritePattern;

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
    Value extOp = rewriter.create<arith::ExtSIOp>(loc, srcNativeVecType,
                                                  transposeOp.getVector());
    Value newTranspose = rewriter.create<vector::TransposeOp>(
        loc, extOp, transposeOp.getPermutation());
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

void vector::populateVectorNarrowTypeEmulationPatterns(
    const arith::NarrowTypeEmulationConverter &typeConverter,
    RewritePatternSet &patterns) {

  // Populate `vector.*` conversion patterns.
  patterns.add<ConvertVectorLoad, ConvertVectorMaskedLoad, ConvertVectorStore,
               ConvertVectorMaskedStore, ConvertVectorTransferRead>(
      typeConverter, patterns.getContext());
}

void vector::populateVectorNarrowTypeRewritePatterns(
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<RewriteBitCastOfTruncI, RewriteExtOfBitCast<arith::ExtUIOp>,
               RewriteExtOfBitCast<arith::ExtSIOp>>(patterns.getContext(),
                                                    benefit);

  // Patterns for aligned cases. We set higher priority as they are expected to
  // generate better performance for aligned cases.
  patterns.add<RewriteAlignedSubByteIntExt<arith::ExtSIOp, /*isSigned=*/true>,
               RewriteAlignedSubByteIntExt<arith::SIToFPOp, /*isSigned=*/true>,
               RewriteAlignedSubByteIntTrunc>(patterns.getContext(),
                                              benefit.getBenefit() + 1);
  patterns
      .add<RewriteAlignedSubByteIntExt<arith::ExtUIOp, /*isSigned=*/false>,
           RewriteAlignedSubByteIntExt<arith::UIToFPOp, /*isSigned=*/false>>(
          patterns.getContext(), benefit.getBenefit() + 1);
}

void vector::populateVectorTransposeNarrowTypeRewritePatterns(
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<RewriteVectorTranspose>(patterns.getContext(), benefit);
}
