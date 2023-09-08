//===- VectorEmulateNarrowType.cpp - Narrow type emulation ----*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

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

namespace {

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
    // %1 = vector.load %0[%linear_index] : memref<12xi8>, vector<2xi8>
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
  int64_t sourceElement;
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

} // namespace

static raw_ostream &operator<<(raw_ostream &os,
                               const SmallVector<SourceElementRangeList> &vec) {
  for (const auto &l : vec) {
    for (auto it : llvm::enumerate(l)) {
      os << "{ " << it.value().sourceElement << ": b@["
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

  assert(targetVectorType.getRank() == 1 && !targetVectorType.isScalable() &&
         "requires -D non-scalable vector type");
  int64_t sourceBitWidth = sourceVectorType.getElementTypeBitWidth();
  int64_t mostMinorSourceDim = sourceVectorType.getShape().back();
  LDBG("sourceVectorType: " << sourceVectorType);

  int64_t targetBitWidth = targetVectorType.getElementTypeBitWidth();
  int64_t mostMinorTargetDim = targetVectorType.getShape().back();
  LDBG("targetVectorType: " << targetVectorType);

  int64_t bitwidth = targetBitWidth * mostMinorTargetDim;
  assert(bitwidth == sourceBitWidth * mostMinorSourceDim &&
         "source and target bitwidths must match");

  // Prepopulate one source element range per target element.
  sourceElementRanges = SmallVector<SourceElementRangeList>(mostMinorTargetDim);
  for (int64_t resultBit = 0; resultBit < bitwidth;) {
    int64_t resultElement = resultBit / targetBitWidth;
    int64_t resultBitInElement = resultBit % targetBitWidth;
    int64_t sourceElement = resultBit / sourceBitWidth;
    int64_t sourceBitInElement = resultBit % sourceBitWidth;
    int64_t step = std::min(sourceBitWidth - sourceBitInElement,
                            targetBitWidth - resultBitInElement);
    sourceElementRanges[resultElement].push_back(
        {sourceElement, sourceBitInElement, sourceBitInElement + step});
    resultBit += step;
  }
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

    VectorType targetVectorType = bitCastOp.getResultVectorType();
    if (targetVectorType.getRank() != 1 || targetVectorType.isScalable())
      return rewriter.notifyMatchFailure(bitCastOp, "scalable or >1-D vector");
    // TODO: consider relaxing this restriction in the future if we find ways to
    // really work with subbyte elements across the MLIR/LLVM boundary.
    int64_t resultBitwidth = targetVectorType.getElementTypeBitWidth();
    if (resultBitwidth % 8 != 0)
      return rewriter.notifyMatchFailure(bitCastOp, "bitwidth is not k * 8");

    VectorType sourceVectorType = bitCastOp.getSourceVectorType();
    BitCastBitsEnumerator be(sourceVectorType, targetVectorType);
    LDBG("\n" << be.sourceElementRanges);

    Value initialValue = truncOp.getIn();
    auto initalVectorType = initialValue.getType().cast<VectorType>();
    auto initalElementType = initalVectorType.getElementType();
    auto initalElementBitWidth = initalElementType.getIntOrFloatBitWidth();

    // BitCastBitsEnumerator encodes for each element of the target vector the
    // provenance of the bits in the source vector. We can "transpose" this
    // information to build a sequence of shuffles and bitwise ops that will
    // produce the desired result.
    // The algorithm proceeds as follows:
    //   1. there are as many shuffles as max entries in BitCastBitsEnumerator
    //   2. for each shuffle:
    //     a. collect the source vectors that participate in this shuffle. One
    //     source vector per target element of the shuffle. If overflow, take 0.
    //     b. the bitrange in the source vector as a mask. If overflow, take 0.
    //     c. the number of bits to shift right to align the source bitrange at
    //     position 0. This is exactly the low end of the bitrange.
    //     d. number of bits to shift left to align to the desired position in
    //     the result element vector.
    // Then build the sequence:
    //   (shuffle -> and -> shiftright -> shiftleft -> or) to iteratively update
    // the result vector (i.e. the "shiftright -> shiftleft -> or" part) with
    // the bits extracted from the source vector (i.e. the "shuffle -> and"
    // part).
    Value res;
    for (int64_t shuffleIdx = 0, e = be.getMaxNumberOfEntries(); shuffleIdx < e;
         ++shuffleIdx) {
      SmallVector<int64_t> shuffles;
      SmallVector<Attribute> masks, shiftRightAmounts, shiftLeftAmounts;
      for (auto &l : be.sourceElementRanges) {
        int64_t sourceElement =
            (shuffleIdx < (int64_t)l.size()) ? l[shuffleIdx].sourceElement : 0;
        shuffles.push_back(sourceElement);

        int64_t bitLo =
            (shuffleIdx < (int64_t)l.size()) ? l[shuffleIdx].sourceBitBegin : 0;
        int64_t bitHi =
            (shuffleIdx < (int64_t)l.size()) ? l[shuffleIdx].sourceBitEnd : 0;
        IntegerAttr mask = IntegerAttr::get(
            rewriter.getIntegerType(initalElementBitWidth),
            llvm::APInt::getBitsSet(initalElementBitWidth, bitLo, bitHi));
        masks.push_back(mask);

        int64_t shiftRight = bitLo;
        shiftRightAmounts.push_back(IntegerAttr::get(
            rewriter.getIntegerType(initalElementBitWidth), shiftRight));

        int64_t shiftLeft = l.computeLeftShiftAmount(shuffleIdx);
        shiftLeftAmounts.push_back(IntegerAttr::get(
            rewriter.getIntegerType(initalElementBitWidth), shiftLeft));
      }

      //
      auto shuffleOp = rewriter.create<vector::ShuffleOp>(
          bitCastOp.getLoc(), initialValue, initialValue, shuffles);

      VectorType vt = VectorType::Builder(initalVectorType)
                          .setDim(initalVectorType.getRank() - 1, masks.size());
      auto constOp = rewriter.create<arith::ConstantOp>(
          bitCastOp.getLoc(), DenseElementsAttr::get(vt, masks));
      Value andValue = rewriter.create<arith::AndIOp>(bitCastOp.getLoc(),
                                                      shuffleOp, constOp);

      auto shiftRightConstantOp = rewriter.create<arith::ConstantOp>(
          bitCastOp.getLoc(), DenseElementsAttr::get(vt, shiftRightAmounts));
      Value shiftedRight = rewriter.create<arith::ShRUIOp>(
          bitCastOp.getLoc(), andValue, shiftRightConstantOp);

      auto shiftLeftConstantOp = rewriter.create<arith::ConstantOp>(
          bitCastOp.getLoc(), DenseElementsAttr::get(vt, shiftLeftAmounts));
      Value shiftedLeft = rewriter.create<arith::ShLIOp>(
          bitCastOp.getLoc(), shiftedRight, shiftLeftConstantOp);

      res = res ? rewriter.create<arith::OrIOp>(bitCastOp.getLoc(), res,
                                                shiftedLeft)
                : shiftedLeft;
    }

    bool narrowing = resultBitwidth <= initalElementBitWidth;
    if (narrowing) {
      rewriter.replaceOpWithNewOp<arith::TruncIOp>(
          bitCastOp, bitCastOp.getResultVectorType(), res);
    } else {
      rewriter.replaceOpWithNewOp<arith::ExtUIOp>(
          bitCastOp, bitCastOp.getResultVectorType(), res);
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
  patterns.add<ConvertVectorLoad, ConvertVectorTransferRead>(
      typeConverter, patterns.getContext());
}

void vector::populateVectorNarrowTypeRewritePatterns(
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<RewriteBitCastOfTruncI>(patterns.getContext(), benefit);
}
