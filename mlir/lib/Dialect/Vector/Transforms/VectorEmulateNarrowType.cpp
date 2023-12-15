//===- VectorEmulateNarrowType.cpp - Narrow type emulation ----*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/NarrowTypeEmulationConverter.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>

using namespace mlir;

#define DEBUG_TYPE "vector-narrow-type-emulation"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

/// Returns a compressed mask. The mask value is set only if any mask is present
/// in the scale range. E.g., if `scale` equals to 2, the following mask:
///
///   %mask = [1, 1, 1, 0, 0, 0]
///
/// will return the following new compressed mask:
///
///   %mask = [1, 1, 0]
static FailureOr<Operation *> getCompressedMaskOp(OpBuilder &rewriter,
                                                  Location loc, Value mask,
                                                  int origElements, int scale) {
  auto numElements = (origElements + scale - 1) / scale;

  Operation *maskOp = mask.getDefiningOp();
  SmallVector<vector::ExtractOp, 2> extractOps;
  // Finding the mask creation operation.
  while (maskOp && !isa<vector::CreateMaskOp, vector::ConstantMaskOp>(maskOp)) {
    if (auto extractOp = dyn_cast<vector::ExtractOp>(maskOp)) {
      maskOp = extractOp.getVector().getDefiningOp();
      extractOps.push_back(extractOp);
    }
  }
  auto createMaskOp = dyn_cast_or_null<vector::CreateMaskOp>(maskOp);
  auto constantMaskOp = dyn_cast_or_null<vector::ConstantMaskOp>(maskOp);
  if (!createMaskOp && !constantMaskOp)
    return failure();

  // Computing the "compressed" mask. All the emulation logic (i.e. computing
  // new mask index) only happens on the last dimension of the vectors.
  Operation *newMask = nullptr;
  SmallVector<int64_t> shape(
      maskOp->getResultTypes()[0].cast<VectorType>().getShape());
  shape.back() = numElements;
  auto newMaskType = VectorType::get(shape, rewriter.getI1Type());
  if (createMaskOp) {
    OperandRange maskOperands = createMaskOp.getOperands();
    size_t numMaskOperands = maskOperands.size();
    AffineExpr s0;
    bindSymbols(rewriter.getContext(), s0);
    s0 = s0 + scale - 1;
    s0 = s0.floorDiv(scale);
    OpFoldResult origIndex =
        getAsOpFoldResult(maskOperands[numMaskOperands - 1]);
    OpFoldResult maskIndex =
        affine::makeComposedFoldedAffineApply(rewriter, loc, s0, origIndex);
    SmallVector<Value> newMaskOperands(maskOperands.drop_back());
    newMaskOperands.push_back(
        getValueOrCreateConstantIndexOp(rewriter, loc, maskIndex));
    newMask = rewriter.create<vector::CreateMaskOp>(loc, newMaskType,
                                                    newMaskOperands);
  } else if (constantMaskOp) {
    ArrayRef<Attribute> maskDimSizes =
        constantMaskOp.getMaskDimSizes().getValue();
    size_t numMaskOperands = maskDimSizes.size();
    auto origIndex =
        cast<IntegerAttr>(maskDimSizes[numMaskOperands - 1]).getInt();
    IntegerAttr maskIndexAttr =
        rewriter.getI64IntegerAttr((origIndex + scale - 1) / scale);
    SmallVector<Attribute> newMaskDimSizes(maskDimSizes.drop_back());
    newMaskDimSizes.push_back(maskIndexAttr);
    newMask = rewriter.create<vector::ConstantMaskOp>(
        loc, newMaskType, rewriter.getArrayAttr(newMaskDimSizes));
  }

  while (!extractOps.empty()) {
    newMask = rewriter.create<vector::ExtractOp>(
        loc, newMask->getResults()[0], extractOps.back().getMixedPosition());
    extractOps.pop_back();
  }

  return newMask;
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
    std::tie(std::ignore, linearizedIndicesOfr) =
        memref::getLinearizedMemRefOffsetAndSize(
            rewriter, loc, srcBits, dstBits,
            stridedMetadata.getConstifiedMixedOffset(),
            stridedMetadata.getConstifiedMixedSizes(),
            stridedMetadata.getConstifiedMixedStrides(),
            getAsOpFoldResult(adaptor.getIndices()));
    Value linearizedIndices =
        getValueOrCreateConstantIndexOp(rewriter, loc, linearizedIndicesOfr);

    // Load the whole data and use arith.select to handle the corner cases.
    // E.g., given these input values:
    //
    //   %mask = [1, 1, 1, 0, 0, 0]
    //   %0[%c0, %c0] contains [0x1, 0x2, 0x3, 0x4, 0x5, 0x6]
    //   %value_to_store = [0x7, 0x8, 0x9, 0xA, 0xB, 0xC]
    //
    // we'll have
    //
    //    expected output: [0x7, 0x8, 0x9, 0x4, 0x5, 0x6]
    //
    //    %new_mask = [1, 1, 0]
    //    %maskedload = [0x12, 0x34, 0x0]
    //    %bitcast = [0x1, 0x2, 0x3, 0x4, 0x0, 0x0]
    //    %select_using_original_mask = [0x7, 0x8, 0x9, 0x4, 0x0, 0x0]
    //    %packed_data = [0x78, 0x94, 0x00]
    //
    // Using the new mask to store %packed_data results in expected output.
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

    Value valueToStore = rewriter.create<vector::BitCastOp>(
        loc, op.getValueToStore().getType(), newLoad);
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
    // TODO: Currently, only the even number of elements loading is supported.
    // To deal with the odd number of elements, one has to extract the
    // subvector at the proper offset after bit-casting.

    auto origElements = op.getVectorType().getNumElements();
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

    auto numElements = (origElements + scale - 1) / scale;
    auto newLoad = rewriter.create<vector::LoadOp>(
        loc, VectorType::get(numElements, newElementType), adaptor.getBase(),
        getValueOrCreateConstantIndexOp(rewriter, loc, linearizedIndices));

    auto bitCast =
        rewriter.create<vector::BitCastOp>(loc, op.getType(), newLoad);

    rewriter.replaceOp(op, bitCast->getResult(0));
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

    FailureOr<Operation *> newMask =
        getCompressedMaskOp(rewriter, loc, op.getMask(), origElements, scale);
    if (failed(newMask))
      return failure();

    auto numElements = (origElements + scale - 1) / scale;
    auto newType = VectorType::get(numElements, newElementType);
    auto newPassThru =
        rewriter.create<vector::BitCastOp>(loc, newType, op.getPassThru());

    // Generating the new masked load.
    auto newLoad = rewriter.create<vector::MaskedLoadOp>(
        loc, newType, adaptor.getBase(),
        getValueOrCreateConstantIndexOp(rewriter, loc, linearizedIndices),
        newMask.value()->getResult(0), newPassThru);

    // Setting the part that originally was not effectively loaded from memory
    // to pass through.
    auto bitCast =
        rewriter.create<vector::BitCastOp>(loc, op.getType(), newLoad);
    auto select = rewriter.create<arith::SelectOp>(loc, op.getMask(), bitCast,
                                                   op.getPassThru());
    rewriter.replaceOp(op, select->getResult(0));

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
    if (origElements % scale != 0)
      return failure();

    auto newPadding = rewriter.create<arith::ExtUIOp>(loc, newElementType,
                                                      adaptor.getPadding());

    auto stridedMetadata =
        rewriter.create<memref::ExtractStridedMetadataOp>(loc, op.getSource());

    OpFoldResult linearizedIndices;
    std::tie(std::ignore, linearizedIndices) =
        memref::getLinearizedMemRefOffsetAndSize(
            rewriter, loc, srcBits, dstBits,
            stridedMetadata.getConstifiedMixedOffset(),
            stridedMetadata.getConstifiedMixedSizes(),
            stridedMetadata.getConstifiedMixedStrides(),
            getAsOpFoldResult(adaptor.getIndices()));

    auto numElements = (origElements + scale - 1) / scale;
    auto newReadType = VectorType::get(numElements, newElementType);

    auto newRead = rewriter.create<vector::TransferReadOp>(
        loc, newReadType, adaptor.getSource(),
        getValueOrCreateConstantIndexOp(rewriter, loc, linearizedIndices),
        newPadding);

    auto bitCast =
        rewriter.create<vector::BitCastOp>(loc, op.getType(), newRead);

    rewriter.replaceOp(op, bitCast->getResult(0));
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

  /// Verify that the preconditions for the rewrite are met.
  LogicalResult precondition(PatternRewriter &rewriter,
                             VectorType preconditionVectorType, Operation *op);

  /// Precompute the metadata for the rewrite.
  SmallVector<BitCastRewriter::Metadata>
  precomputeMetadata(IntegerType shuffledElementType);

  /// Rewrite one step of the sequence:
  ///   `(shuffle -> and -> shiftright -> shiftleft -> or)`.
  Value rewriteStep(PatternRewriter &rewriter, Location loc, Value initialValue,
                    Value runningResult,
                    const BitCastRewriter::Metadata &metadata);

private:
  /// Underlying enumerator that encodes the provenance of the bits in the each
  /// element of the result vector.
  BitCastBitsEnumerator enumerator;
};

} // namespace

[[maybe_unused]] static raw_ostream &operator<<(raw_ostream &os,
                               const SmallVector<SourceElementRangeList> &vec) {
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

LogicalResult BitCastRewriter::precondition(PatternRewriter &rewriter,
                                            VectorType precondition,
                                            Operation *op) {
  if (precondition.getRank() != 1 || precondition.isScalable())
    return rewriter.notifyMatchFailure(op, "scalable or >1-D vector");

  // TODO: consider relaxing this restriction in the future if we find ways
  // to really work with subbyte elements across the MLIR/LLVM boundary.
  int64_t resultBitwidth = precondition.getElementTypeBitWidth();
  if (resultBitwidth % 8 != 0)
    return rewriter.notifyMatchFailure(op, "bitwidth is not k * 8");

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

Value BitCastRewriter::rewriteStep(PatternRewriter &rewriter, Location loc,
                                   Value initialValue, Value runningResult,
                                   const BitCastRewriter::Metadata &metadata) {
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
    if (failed(bcr.precondition(rewriter, targetVectorType, bitCastOp)))
      return failure();

    // Perform the rewrite.
    Value truncValue = truncOp.getIn();
    auto shuffledElementType =
        cast<IntegerType>(getElementTypeOrSelf(truncValue.getType()));
    Value runningResult;
    for (const BitCastRewriter ::Metadata &metadata :
         bcr.precomputeMetadata(shuffledElementType)) {
      runningResult = bcr.rewriteStep(rewriter, bitCastOp->getLoc(), truncValue,
                                      runningResult, metadata);
    }

    // Finalize the rewrite.
    bool narrowing = targetVectorType.getElementTypeBitWidth() <=
                     shuffledElementType.getIntOrFloatBitWidth();
    if (narrowing) {
      rewriter.replaceOpWithNewOp<arith::TruncIOp>(
          bitCastOp, bitCastOp.getResultVectorType(), runningResult);
    } else {
      rewriter.replaceOpWithNewOp<arith::ExtUIOp>(
          bitCastOp, bitCastOp.getResultVectorType(), runningResult);
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
    if (failed(bcr.precondition(
            rewriter, cast<VectorType>(extOp.getOut().getType()), bitCastOp)))
      return failure();

    // Perform the rewrite.
    Value runningResult;
    Value sourceValue = bitCastOp.getSource();
    auto shuffledElementType =
        cast<IntegerType>(getElementTypeOrSelf(sourceValue.getType()));
    for (const BitCastRewriter::Metadata &metadata :
         bcr.precomputeMetadata(shuffledElementType)) {
      runningResult = bcr.rewriteStep(rewriter, bitCastOp->getLoc(),
                                      sourceValue, runningResult, metadata);
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
} // namespace

//===----------------------------------------------------------------------===//
// Public Interface Definition
//===----------------------------------------------------------------------===//

void vector::populateVectorNarrowTypeEmulationPatterns(
    arith::NarrowTypeEmulationConverter &typeConverter,
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
}
