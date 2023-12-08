//===- FoldIntoPackAndUnpackPatterns.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace tensor {
namespace {

static bool areAllConstantIntValue(ArrayRef<OpFoldResult> ofrs, int64_t value) {
  return llvm::all_of(
      ofrs, [&](OpFoldResult ofr) { return isConstantIntValue(ofr, value); });
}

/// Fold a `pad` -> `pack` into `pack` if they have the same padding values and
/// the pad op has zero low paddings, or if `pack` has no padding values.
struct FoldPadWithPackOp : public OpRewritePattern<PackOp> {
  using OpRewritePattern<PackOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(PackOp packOp,
                                PatternRewriter &rewriter) const override {
    auto padOp = packOp.getSource().getDefiningOp<PadOp>();

    if (!padOp || padOp.getNofold() || !padOp.hasZeroLowPad())
      return failure();

    Value constantPaddingValue = padOp.getConstantPaddingValue();
    if (!constantPaddingValue)
      return failure();

    if (auto paddingValue = packOp.getPaddingValue())
      if (!isEqualConstantIntOrValue(paddingValue, constantPaddingValue))
        return failure();

    rewriter.replaceOpWithNewOp<PackOp>(
        packOp, padOp.getSource(), packOp.getDest(), packOp.getInnerDimsPos(),
        packOp.getMixedTiles(), constantPaddingValue,
        packOp.getOuterDimsPerm());
    return success();
  }
};

/// Fold a `unpack` -> `extract_slice` into the `unpack` since it already
/// has extract_slice semantics.
struct FoldUnpackWithExtractSliceOp : public OpRewritePattern<ExtractSliceOp> {
  using OpRewritePattern<ExtractSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ExtractSliceOp sliceOp,
                                PatternRewriter &rewriter) const override {
    auto unpackOp = sliceOp.getSource().getDefiningOp<UnPackOp>();
    if (!unpackOp)
      return failure();

    if (sliceOp.getResultType().getRank() != unpackOp.getDestType().getRank()) {
      return rewriter.notifyMatchFailure(
          sliceOp, "rank-reduced folding is not supported");
    }

    // Check all offsets are zeros, and all strides are ones.
    if (!areAllConstantIntValue(sliceOp.getMixedOffsets(), 0) ||
        !areAllConstantIntValue(sliceOp.getMixedStrides(), 1)) {
      return rewriter.notifyMatchFailure(
          sliceOp, "expects offsets to be 0s and strides to be 1s");
    }

    // Create a new empty output tensor.
    Type elementType = unpackOp.getDestType().getElementType();
    Value output = rewriter.create<EmptyOp>(
        sliceOp.getLoc(), sliceOp.getMixedSizes(), elementType);
    rewriter.replaceOpWithNewOp<UnPackOp>(
        sliceOp, unpackOp.getSource(), output, unpackOp.getInnerDimsPos(),
        unpackOp.getMixedTiles(), unpackOp.getOuterDimsPerm());
    return success();
  }
};

/// Fold 'pack' -> 'transpose' into 'pack' since 'pack' already has transpose
/// semantics.
struct FoldProducerPackWithConsumerLinalgTransposeOp
    : public OpRewritePattern<linalg::TransposeOp> {
  using OpRewritePattern<linalg::TransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::TransposeOp transposeOp,
                                PatternRewriter &rewriter) const override {
    auto transposeInputTensor = transposeOp.getOperand(0);
    auto packOp = transposeInputTensor.getDefiningOp<PackOp>();

    if (!packOp)
      return failure();

    auto packInnerDimsPos = packOp.getInnerDimsPos();
    auto packInnerTiles = packOp.getStaticInnerTiles();
    auto packOuterDimsPerm = packOp.getOuterDimsPerm();
    auto transposePerm = transposeOp.getPermutation();
    SmallVector<int64_t> newPackOuterDimsPermVec;
    SmallVector<int64_t> newPackInnerDimsPosVec;
    SmallVector<int64_t> newPackInnerTilesVec;

    // Variable for storing translated position after considering original
    // outer_dims_perm and permutation attributes of tensor.pack and
    // linalg.transpose.
    int64_t translatedPosition;

    // Process transpose operation for non-tiled outer dimensions of the tensor.
    for (unsigned int i = 0; i < transposePerm.size() - packInnerTiles.size();
         ++i) {
      // If tensor.pack has outer_dims_perm attribute, then consider it during
      // index translation.
      if (packOuterDimsPerm.size())
        translatedPosition = packOuterDimsPerm[transposePerm[i]];
      else
        translatedPosition = transposePerm[i];

      // Note: static_cast was added around translatedPosition to suppress the
      // compiler warning of comparison between variables of different types.
      if (static_cast<unsigned long>(translatedPosition) >=
          transposePerm.size() - packInnerTiles.size())
        return rewriter.notifyMatchFailure(
            transposeOp,
            "Cannot fold in tensor.pack if a tile dimension was transposed "
            "with a non-tile dimension in linalg.transpose.");

      newPackOuterDimsPermVec.push_back(translatedPosition);
    }

    // Process transpose operation for tiled inner dimensions of the tensor.
    for (unsigned int i = transposePerm.size() - packInnerTiles.size();
         i < transposePerm.size(); ++i) {
      translatedPosition =
          transposePerm[i] - (transposePerm.size() - packInnerTiles.size());

      newPackInnerTilesVec.push_back(packInnerTiles[translatedPosition]);
      newPackInnerDimsPosVec.push_back(packInnerDimsPos[translatedPosition]);
    }

    SmallVector<OpFoldResult> opFoldResultsTiles;
    opFoldResultsTiles.reserve(newPackInnerTilesVec.size());

    transform(newPackInnerTilesVec, std::back_inserter(opFoldResultsTiles),
              [&rewriter](int64_t value) {
                return IntegerAttr::get(IndexType::get(rewriter.getContext()),
                                        value);
              });

    ArrayRef<OpFoldResult> newPackInnerTilesArrayRef(opFoldResultsTiles);

    Value output = packOp.createDestinationTensor(
        rewriter, transposeOp.getLoc(), packOp.getSource(),
        newPackInnerTilesArrayRef,
        static_cast<ArrayRef<int64_t>>(newPackInnerDimsPosVec),
        static_cast<ArrayRef<int64_t>>(newPackOuterDimsPermVec));

    rewriter.replaceOpWithNewOp<PackOp>(
        transposeOp, packOp.getSource(), output,
        static_cast<ArrayRef<int64_t>>(newPackInnerDimsPosVec),
        newPackInnerTilesArrayRef, packOp.getPaddingValue(),
        static_cast<ArrayRef<int64_t>>(newPackOuterDimsPermVec));

    return success();
  }
};

/// Fold 'transpose' -> 'pack' into 'pack' since 'pack' already has transpose
/// semantics.
struct FoldConsumerPackWithProducerLinalgTransposeOp
    : public OpRewritePattern<PackOp> {
  using OpRewritePattern<PackOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(PackOp packOp,
                                PatternRewriter &rewriter) const override {
    auto packInputTensor = packOp.getOperand(0);
    auto transposeOp = packInputTensor.getDefiningOp<linalg::TransposeOp>();

    if (!transposeOp)
      return failure();

    auto packInnerDimsPos = packOp.getInnerDimsPos();
    auto packInnerTiles = packOp.getStaticInnerTiles();
    auto packOuterDimsPerm = packOp.getOuterDimsPerm();
    auto transposePerm = transposeOp.getPermutation();
    SmallVector<int64_t> newPackOuterDimsPermVec;
    SmallVector<int64_t> newPackInnerDimsPosVec;
    SmallVector<int64_t> newPackInnerTilesVec;

    // Variable for storing translated position after considering original
    // outer_dims_perm and permutation attributes of tensor.pack and
    // linalg.transpose.
    int64_t translatedPosition;

    // Process transpose operation for non-tiled outer dimensions of the tensor.
    for (unsigned int i = 0; i < transposePerm.size() - packInnerTiles.size();
         ++i) {
      // If tensor.pack has outer_dims_perm attribute, then consider it during
      // index translation.
      if (packOuterDimsPerm.size())
        translatedPosition = packOuterDimsPerm[transposePerm[i]];
      else
        translatedPosition = transposePerm[i];

      // Note: static_cast was added around translatedPosition to suppress the
      // compiler warning of comparison between variables of different types.
      if (static_cast<unsigned long>(translatedPosition) >=
          transposePerm.size() - packInnerTiles.size())
        return rewriter.notifyMatchFailure(
            packOp,
            "Cannot fold in tensor.pack if a tile dimension was transposed "
            "with a non-tile dimension in linalg.transpose.");

      newPackOuterDimsPermVec.push_back(translatedPosition);
    }

    // Process transpose operation for tiled inner dimensions of the tensor.
    for (unsigned int i = transposePerm.size() - packInnerTiles.size();
         i < transposePerm.size(); ++i) {
      translatedPosition =
          transposePerm[i] - (transposePerm.size() - packInnerTiles.size());

      newPackInnerTilesVec.push_back(packInnerTiles[translatedPosition]);
      newPackInnerDimsPosVec.push_back(packInnerDimsPos[translatedPosition]);
    }

    SmallVector<OpFoldResult> opFoldResultsTiles;
    opFoldResultsTiles.reserve(newPackInnerTilesVec.size());

    transform(newPackInnerTilesVec, std::back_inserter(opFoldResultsTiles),
              [&rewriter](int64_t value) {
                return IntegerAttr::get(IndexType::get(rewriter.getContext()),
                                        value);
              });

    ArrayRef<OpFoldResult> newPackInnerTilesArrayRef(opFoldResultsTiles);

    Value output = packOp.createDestinationTensor(
        rewriter, packOp.getLoc(), transposeOp.getOperand(0),
        newPackInnerTilesArrayRef, 
        static_cast<ArrayRef<int64_t>>(newPackInnerDimsPosVec),
        static_cast<ArrayRef<int64_t>>(newPackOuterDimsPermVec));

    output.dump();

    rewriter.replaceOpWithNewOp<PackOp>(
        packOp, transposeOp.getOperand(0), output, 
        static_cast<ArrayRef<int64_t>>(newPackInnerDimsPosVec),
        newPackInnerTilesArrayRef, packOp.getPaddingValue(),
        static_cast<ArrayRef<int64_t>>(newPackOuterDimsPermVec));

    return success();
  }
};
} // namespace

void populateFoldIntoPackAndUnpackPatterns(RewritePatternSet &patterns) {
  patterns.insert<FoldUnpackWithExtractSliceOp, FoldPadWithPackOp,
                  FoldProducerPackWithConsumerLinalgTransposeOp,
                  FoldConsumerPackWithProducerLinalgTransposeOp>(
      patterns.getContext());
}

} // namespace tensor
} // namespace mlir
