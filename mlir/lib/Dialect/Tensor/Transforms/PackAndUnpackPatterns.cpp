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
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace tensor {
namespace {

static bool areAllConstantIntValue(ArrayRef<OpFoldResult> ofrs, int64_t value) {
  return llvm::all_of(
      ofrs, [&](OpFoldResult ofr) { return isConstantIntValue(ofr, value); });
}

/// Returns the number of shape sizes that is either dynamic or greater than 1.
static int64_t getNumGtOneDims(ArrayRef<int64_t> shape) {
  return llvm::count_if(
      shape, [](int64_t v) { return ShapedType::isDynamic(v) || v > 1; });
}

/// Returns success() if there is only 1 dimension size in non-packed domain
/// being greater than 1 and packing only happens on the dimension.
/// Note: this method should only be used by pack/unpack to reshape conversion.
/// It assumes that non-unit inner tile size must be used by the non-unit
/// dimension.
static LogicalResult isPackOn1D(RewriterBase &rewriter, Operation *op,
                                ArrayRef<int64_t> srcShape,
                                ArrayRef<int64_t> innerPackTileSize) {
  if (getNumGtOneDims(srcShape) > 1) {
    return rewriter.notifyMatchFailure(
        op, "expects non-packed domain to have at most one non-unit dims");
  }
  // Non-unit inner tile size must be used by the non-unit dimension. If not, it
  // will faill on getting reassociation maps.
  if (getNumGtOneDims(innerPackTileSize) > 1) {
    return rewriter.notifyMatchFailure(
        op, "expects at most one non-unit inner tiles");
  }
  return success();
}

// If the `linalgOp` represents a transpose, return the permutation vector for
// the transpose. Otherwise, return failure.
static FailureOr<SmallVector<int64_t>>
getTransposeOpPermutation(linalg::LinalgOp linalgOp) {
  if (auto transposeOp = dyn_cast<linalg::TransposeOp>(linalgOp.getOperation()))
    return SmallVector<int64_t>(transposeOp.getPermutation());
  if (linalgOp.getNumParallelLoops() != linalgOp.getNumLoops())
    return failure();

  if (linalgOp.getNumDpsInputs() != 1 || linalgOp.getNumDpsInits() != 1)
    return failure();
  auto mapRange = linalgOp.getIndexingMapsArray();
  if (!mapRange.front().isPermutation() || !mapRange.back().isPermutation() ||
      mapRange.front() == mapRange.back()) {
    return failure();
  }
  if (!llvm::hasSingleElement(linalgOp.getBlock()->getOperations()))
    return failure();
  AffineMap outMap = mapRange.back();
  AffineMap inMap = mapRange.front();
  // To get the permutation, look at each output index and find which
  // dimension in the input we're reading from for that index.
  return llvm::map_to_vector(outMap.getResults(),
                             [&](AffineExpr expr) -> int64_t {
                               return *inMap.getResultPosition(expr);
                             });
}

/// Packing one-dimensional tensor can be expressed as an expand shape op.
struct SimplifyPackToExpandShape : public OpRewritePattern<PackOp> {
  using OpRewritePattern<PackOp>::OpRewritePattern;

  FailureOr<Value>
  insertExpand(RewriterBase &rewriter, Location loc, Value operand,
               Type newOperandType,
               ArrayRef<ReassociationIndices> reassociation) const {
    if (operand.getType() == newOperandType)
      return operand;
    return rewriter
        .create<tensor::ExpandShapeOp>(loc, newOperandType, operand,
                                       reassociation)
        .getResult();
  }

  /// Returns success() if it is only packing on the innermost dimension.
  LogicalResult isPackOnInnerMostDim(RewriterBase &rewriter,
                                     PackOp packOp) const {
    auto outerDimsPerm = packOp.getOuterDimsPerm();
    if (!outerDimsPerm.empty() && !isIdentityPermutation(outerDimsPerm)) {
      return rewriter.notifyMatchFailure(
          packOp,
          "expects outer_dims_perm is empty or an identity permutation");
    }

    int64_t srcRank = packOp.getSourceRank();
    ArrayRef<int64_t> dimsPos = packOp.getInnerDimsPos();
    if (dimsPos.size() != 1 || (dimsPos[0] + 1 != srcRank)) {
      return rewriter.notifyMatchFailure(
          packOp, "expects packing at the innermost dimension");
    }
    return success();
  }

  LogicalResult matchAndRewrite(PackOp packOp,
                                PatternRewriter &rewriter) const override {
    if (packOp.getPaddingValue())
      return rewriter.notifyMatchFailure(packOp, "expects no padding value");

    RankedTensorType sourceType = packOp.getSourceType();
    if (failed(isPackOnInnerMostDim(rewriter, packOp)) &&
        failed(isPackOn1D(rewriter, packOp, sourceType.getShape(),
                          packOp.getStaticTiles())) &&
        !packOp.isLikePad()) {
      return failure();
    }

    RankedTensorType destType = packOp.getDestType();
    auto reassociation =
        getReassociationIndicesForReshape(sourceType, destType);
    if (!reassociation)
      return failure();
    FailureOr<Value> expanded =
        insertExpand(rewriter, packOp.getLoc(), packOp.getSource(), destType,
                     *reassociation);
    if (failed(expanded)) {
      return rewriter.notifyMatchFailure(
          packOp, "unable to expand source of tensor.pack");
    }
    rewriter.replaceOp(packOp, *expanded);
    return success();
  }
};

struct SimplifyUnPackToCollapseShape : public OpRewritePattern<UnPackOp> {
  using OpRewritePattern<UnPackOp>::OpRewritePattern;

  Value insertCollapse(RewriterBase &rewriter, Location loc, Value operand,
                       Type newOperandType, ArrayAttr reassociation) const {
    if (operand.getType() == newOperandType)
      return operand;
    return rewriter.create<tensor::CollapseShapeOp>(loc, newOperandType,
                                                    operand, reassociation);
  }

  /// Returns success() if it is unpacking on the innermost dimension.
  LogicalResult isUnpackOnInnerMostDim(RewriterBase &rewriter,
                                       UnPackOp unpackOp) const {
    auto outerDimsPerm = unpackOp.getOuterDimsPerm();
    if (!outerDimsPerm.empty() && !isIdentityPermutation(outerDimsPerm)) {
      return rewriter.notifyMatchFailure(
          unpackOp,
          "expects outer_dims_perm is empty or an identity permutation");
    }

    RankedTensorType sourceType = unpackOp.getSourceType();
    RankedTensorType destType = unpackOp.getDestType();
    if (!sourceType.hasStaticShape() || !destType.hasStaticShape())
      return rewriter.notifyMatchFailure(unpackOp, "expects static shapes");

    ArrayRef<int64_t> dimsPos = unpackOp.getInnerDimsPos();
    if (dimsPos.size() != 1 || (dimsPos[0] + 1 != destType.getRank())) {
      return rewriter.notifyMatchFailure(
          unpackOp, "expects unpacking on the innermost dimension");
    }

    return success();
  }

  LogicalResult matchAndRewrite(UnPackOp unpackOp,
                                PatternRewriter &rewriter) const override {
    RankedTensorType destType = unpackOp.getDestType();
    if (failed(isUnpackOnInnerMostDim(rewriter, unpackOp)) &&
        failed(isPackOn1D(rewriter, unpackOp, destType.getShape(),
                          unpackOp.getStaticTiles())) &&
        !unpackOp.isLikeUnPad()) {
      return failure();
    }

    RankedTensorType sourceType = unpackOp.getSourceType();
    auto reassociation =
        getReassociationIndicesForReshape(sourceType, destType);
    if (!reassociation)
      return failure();
    Value collapsed = insertCollapse(
        rewriter, unpackOp.getLoc(), unpackOp.getSource(), destType,
        getReassociationIndicesAttribute(rewriter, *reassociation));
    rewriter.replaceOp(unpackOp, collapsed);
    return success();
  }
};

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

// Applies 'permutation' on 'inVec' and stores the result in resVec.
// 'inVec' may be empty, in that case it's one-to-one mapping with permutation.
// `rank` sets the boundary for permutation i.e., the permutation dim can't be
// greater than the rank specified. If it's so then return false.
// For e.g., permutation {1, 0, 3, 2} with rank 2 is allowed since the values in
// permutation[:rank] doesn't exceed rank, whereas, permutation {1, 3, 0, 2} is
// not allowed since `3` exceeds the value of the rank in the given range.
static bool checkAndPermute(ArrayRef<int64_t> permutation,
                            ArrayRef<int64_t> inVec,
                            SmallVectorImpl<int64_t> &resVec, int64_t rank) {

  for (unsigned int i = 0; i < rank; ++i) {
    int64_t remappedPosition = permutation[i];
    if (remappedPosition >= rank)
      return false;
    if (!inVec.empty())
      remappedPosition = inVec[remappedPosition];
    resVec.push_back(remappedPosition);
  }

  return true;
}

/// Fold 'pack' -> 'transpose' into 'pack' since 'pack' already has transpose
/// semantics.
struct FoldProducerPackWithConsumerLinalgTransposeOp
    : public OpInterfaceRewritePattern<linalg::LinalgOp> {
  using OpInterfaceRewritePattern<linalg::LinalgOp>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(linalg::LinalgOp linalgOp,
                                PatternRewriter &rewriter) const override {
    auto packOp = linalgOp->getOperand(0).getDefiningOp<PackOp>();

    if (!packOp)
      return failure();

    FailureOr<SmallVector<int64_t>> maybePerm =
        getTransposeOpPermutation(linalgOp);
    if (failed(maybePerm))
      return failure();

    auto innerDimsPos = packOp.getInnerDimsPos();
    auto mixedInnerTiles = packOp.getMixedTiles();
    auto outerDimsPerm = packOp.getOuterDimsPerm();
    auto transposePerm = maybePerm.value();
    SmallVector<int64_t> newOuterDimsPermVec;
    SmallVector<int64_t> newInnerDimsPosVec;
    SmallVector<OpFoldResult> newMixedInnerTilesVec;
    int64_t srcRank = packOp.getSourceRank();

    if (!checkAndPermute(transposePerm, outerDimsPerm, newOuterDimsPermVec,
                         srcRank))
      return rewriter.notifyMatchFailure(
          linalgOp,
          "Cannot fold in tensor.pack if a tile dimension was transposed "
          "with a non-tile dimension in linalg.transpose.");

    // Process transpose operation for tiled inner dimensions
    for (unsigned int i = srcRank; i < transposePerm.size(); ++i) {
      int64_t remappedPosition = transposePerm[i] - srcRank;
      newMixedInnerTilesVec.push_back(mixedInnerTiles[remappedPosition]);
      newInnerDimsPosVec.push_back(innerDimsPos[remappedPosition]);
    }

    Value output = packOp.createDestinationTensor(
        rewriter, linalgOp.getLoc(), packOp.getSource(), newMixedInnerTilesVec,
        newInnerDimsPosVec, newOuterDimsPermVec);

    rewriter.replaceOpWithNewOp<PackOp>(
        linalgOp, packOp.getSource(), output, newInnerDimsPosVec,
        newMixedInnerTilesVec, packOp.getPaddingValue(), newOuterDimsPermVec);

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
    auto linalgOp = packOp.getSource().getDefiningOp<linalg::LinalgOp>();
    if (!linalgOp)
      return failure();

    FailureOr<SmallVector<int64_t>> maybePerm =
        getTransposeOpPermutation(linalgOp);
    if (failed(maybePerm))
      return failure();

    auto transposePermutation = maybePerm.value();
    auto outerDimsPerm = packOp.getOuterDimsPerm();
    auto innerDimsPos = packOp.getInnerDimsPos();
    SmallVector<int64_t> newInnerDimsPosVec;
    SmallVector<int64_t> newOuterDimsPermVec =
        llvm::to_vector(transposePermutation);

    if (!outerDimsPerm.empty())
      applyPermutationToVector(newOuterDimsPermVec, outerDimsPerm);

    // Can't use applyPermutationToVector for newInnerDimsPosVec since input and
    // permutation rank won't necessarily be equal in all cases.
    for (auto dim : innerDimsPos)
      newInnerDimsPosVec.push_back(transposePermutation[dim]);

    Value output = packOp.createDestinationTensor(
        rewriter, packOp.getLoc(), linalgOp->getOperand(0),
        packOp.getMixedTiles(), newInnerDimsPosVec, newOuterDimsPermVec);

    rewriter.replaceOpWithNewOp<PackOp>(
        packOp, linalgOp->getOperand(0), output, newInnerDimsPosVec,
        packOp.getMixedTiles(), packOp.getPaddingValue(), newOuterDimsPermVec);

    return success();
  }
};

/// Fold 'unpack' -> 'transpose' into 'unpack' since 'unpack' already has
/// transpose semantics.
struct FoldProducerUnPackWithConsumerLinalgTransposeOp
    : public OpInterfaceRewritePattern<linalg::LinalgOp> {
  using OpInterfaceRewritePattern<linalg::LinalgOp>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(linalg::LinalgOp linalgOp,
                                PatternRewriter &rewriter) const override {
    auto unPackOp = linalgOp->getOperand(0).getDefiningOp<UnPackOp>();

    if (!unPackOp)
      return failure();

    FailureOr<SmallVector<int64_t>> maybePerm =
        getTransposeOpPermutation(linalgOp);
    if (failed(maybePerm))
      return failure();

    auto outerDimsPerm = unPackOp.getOuterDimsPerm();
    auto innerDimsPos = unPackOp.getInnerDimsPos();
    SmallVector<int64_t> newInnerDimsPosVec;
    SmallVector<int64_t> newOuterDimsPermVec =
        invertPermutationVector(maybePerm.value());

    // Can't use applyPermutationToVector for newInnerDimsPosVec since input and
    // permutation rank won't necessarily be equal in all cases.
    for (auto dim : innerDimsPos)
      newInnerDimsPosVec.push_back(newOuterDimsPermVec[dim]);

    if (!outerDimsPerm.empty())
      applyPermutationToVector(newOuterDimsPermVec, outerDimsPerm);

    // Reuse the destination of the transpose op.
    rewriter.replaceOpWithNewOp<UnPackOp>(
        linalgOp, unPackOp.getSource(), linalgOp.getDpsInits()[0],
        newInnerDimsPosVec, unPackOp.getMixedTiles(), newOuterDimsPermVec);

    return success();
  }
};

/// Fold 'transpose' -> 'unpack' into 'unpack' since 'unpack' already has
/// transpose semantics.
struct FoldConsumerUnPackWithProducerLinalgTransposeOp
    : public OpRewritePattern<UnPackOp> {
  using OpRewritePattern<UnPackOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(UnPackOp unPackOp,
                                PatternRewriter &rewriter) const override {
    auto linalgOp = unPackOp.getSource().getDefiningOp<linalg::LinalgOp>();
    if (!linalgOp)
      return failure();

    FailureOr<SmallVector<int64_t>> maybePerm =
        getTransposeOpPermutation(linalgOp);
    if (failed(maybePerm))
      return failure();

    SmallVector<SmallVector<OpFoldResult>> unpackOpResultDims;
    if (failed(reifyResultShapes(rewriter, unPackOp, unpackOpResultDims))) {
      return failure();
    }

    SmallVector<int64_t> inverseTransposePerm =
        invertPermutationVector(maybePerm.value());
    auto outerDimsPerm = unPackOp.getOuterDimsPerm();
    auto innerDimsPos = unPackOp.getInnerDimsPos();
    int64_t destRank = unPackOp.getSourceRank() - innerDimsPos.size();
    auto mixedInnerTilesVec = unPackOp.getMixedTiles();
    SmallVector<int64_t> newOuterDimsPermVec;
    SmallVector<int64_t> newInnerDimsPosVec;
    SmallVector<OpFoldResult> newMixedInnerTilesVec;
    if (!checkAndPermute(inverseTransposePerm, outerDimsPerm,
                         newOuterDimsPermVec, destRank))
      return rewriter.notifyMatchFailure(
          unPackOp,
          "Cannot fold in tensor.unpack if a tile dimension was transposed "
          "with a non-tile dimension in linalg.transpose.");

    // Process transpose operation for tiled inner dimensions
    for (unsigned int i = destRank; i < inverseTransposePerm.size(); ++i) {
      int64_t remappedPosition = inverseTransposePerm[i] - destRank;
      newMixedInnerTilesVec.push_back(mixedInnerTilesVec[remappedPosition]);
      newInnerDimsPosVec.push_back(innerDimsPos[remappedPosition]);
    }

    auto elemType =
        cast<ShapedType>(unPackOp->getResultTypes()[0]).getElementType();
    Value output = rewriter.create<tensor::EmptyOp>(
        unPackOp->getLoc(), unpackOpResultDims[0], elemType);

    rewriter.replaceOpWithNewOp<UnPackOp>(
        unPackOp, linalgOp->getOperand(0), output, newInnerDimsPosVec,
        newMixedInnerTilesVec, newOuterDimsPermVec);

    return success();
  }
};
} // namespace

void populateFoldIntoPackAndUnpackPatterns(RewritePatternSet &patterns) {
  patterns.insert<FoldUnpackWithExtractSliceOp, FoldPadWithPackOp,
                  FoldProducerPackWithConsumerLinalgTransposeOp,
                  FoldConsumerPackWithProducerLinalgTransposeOp,
                  FoldConsumerUnPackWithProducerLinalgTransposeOp,
                  FoldProducerUnPackWithConsumerLinalgTransposeOp>(
      patterns.getContext());
}

void populateSimplifyPackAndUnpackPatterns(RewritePatternSet &patterns) {
  patterns.add<SimplifyPackToExpandShape, SimplifyUnPackToCollapseShape>(
      patterns.getContext());
}

} // namespace tensor
} // namespace mlir
