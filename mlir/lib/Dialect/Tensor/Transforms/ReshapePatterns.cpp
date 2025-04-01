//===- RankReductionPatterns.cpp - Patterns related to rank reductions ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"

using namespace mlir;
using namespace mlir::tensor;

namespace {
/// Fold expand_shape(extract_slice) ops that cancel itself out.
struct FoldExpandOfRankReducingExtract
    : public OpRewritePattern<ExpandShapeOp> {
  using OpRewritePattern<ExpandShapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ExpandShapeOp expandShapeOp,
                                PatternRewriter &rewriter) const override {
    RankedTensorType resultType = expandShapeOp.getResultType();
    auto extractSliceOp =
        expandShapeOp.getSrc().getDefiningOp<ExtractSliceOp>();
    if (!extractSliceOp)
      return failure();
    RankedTensorType srcType = extractSliceOp.getSourceType();

    // Only cases where the ExpandShapeOp can be folded away entirely are
    // supported. Moreover, only simple cases where the resulting ExtractSliceOp
    // has no rank-reduction anymore are supported at the moment.
    RankedTensorType nonReducingExtractType = ExtractSliceOp::inferResultType(
        srcType, extractSliceOp.getStaticOffsets(),
        extractSliceOp.getStaticSizes(), extractSliceOp.getStaticStrides());
    if (nonReducingExtractType != resultType)
      return failure();

    SmallVector<OpFoldResult> mixedOffsets = extractSliceOp.getMixedOffsets();
    SmallVector<OpFoldResult> mixedSizes = extractSliceOp.getMixedSizes();
    SmallVector<OpFoldResult> mixedStrides = extractSliceOp.getMixedStrides();
    rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(
        expandShapeOp, extractSliceOp.getSource(), mixedOffsets, mixedSizes,
        mixedStrides);
    return success();
  }
};

/// Fold collapse_shape which only removes static dimensions of size `1`
/// into extract_slice.
struct FoldUnPaddingCollapseIntoExtract
    : public OpRewritePattern<tensor::CollapseShapeOp> {
  using OpRewritePattern<tensor::CollapseShapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::CollapseShapeOp collapseShapeOp,
                                PatternRewriter &rewriter) const override {
    auto extractSliceOp =
        collapseShapeOp.getSrc().getDefiningOp<tensor::ExtractSliceOp>();
    // Collapse cannot be folded away with multiple users of the extract slice
    // and it is not necessarily beneficial to only convert the collapse into
    // another extract slice.
    if (!extractSliceOp || !extractSliceOp->hasOneUse())
      return failure();

    // Only fold away simple collapse where all removed dimensions have static
    // size `1`.
    SliceVerificationResult res = isRankReducedType(
        collapseShapeOp.getSrcType(), collapseShapeOp.getResultType());
    if (res != SliceVerificationResult::Success)
      return rewriter.notifyMatchFailure(collapseShapeOp,
                                         "expected unpadding collapse");

    Value unPaddedExtractSlice = rewriter.create<tensor::ExtractSliceOp>(
        extractSliceOp.getLoc(), collapseShapeOp.getResultType(),
        extractSliceOp.getSource(), extractSliceOp.getMixedOffsets(),
        extractSliceOp.getMixedSizes(), extractSliceOp.getMixedStrides());
    rewriter.replaceOp(collapseShapeOp, unPaddedExtractSlice);
    return success();
  }
};

/// Fold insert_slice(collapse_shape) ops that cancel itself out.
template <typename OpTy>
struct FoldInsertOfRankReducingInsert : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy insertSliceOp,
                                PatternRewriter &rewriter) const override {
    auto collapseShapeOp =
        insertSliceOp.getSource().template getDefiningOp<CollapseShapeOp>();
    if (!collapseShapeOp)
      return failure();
    RankedTensorType srcType = collapseShapeOp.getSrcType();

    // Only cases where the CollapseShapeOp can be folded away entirely are
    // supported. Moreover, only simple cases where the resulting InsertSliceOp
    // has no rank-reduction anymore are supported at the moment.
    RankedTensorType nonReducingInsertType =
        RankedTensorType::get(insertSliceOp.getStaticSizes(),
                              insertSliceOp.getDestType().getElementType());
    if (nonReducingInsertType != srcType)
      return failure();

    SmallVector<OpFoldResult> mixedOffsets = insertSliceOp.getMixedOffsets();
    SmallVector<OpFoldResult> mixedSizes = insertSliceOp.getMixedSizes();
    SmallVector<OpFoldResult> mixedStrides = insertSliceOp.getMixedStrides();
    rewriter.replaceOpWithNewOp<OpTy>(insertSliceOp, collapseShapeOp.getSrc(),
                                      insertSliceOp.getDest(), mixedOffsets,
                                      mixedSizes, mixedStrides);
    return success();
  }
};

/// Fold expand_shape which only adds static dimensions of size `1`
/// into insert_slice.
template <typename OpTy>
struct FoldPaddingExpandIntoInsert : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy insertSliceOp,
                                PatternRewriter &rewriter) const override {
    auto expandShapeOp = insertSliceOp.getSource()
                             .template getDefiningOp<tensor::ExpandShapeOp>();
    if (!expandShapeOp)
      return failure();

    // Only fold away simple expansion where all added dimensions have static
    // size `1`.
    SliceVerificationResult res = isRankReducedType(
        expandShapeOp.getResultType(), expandShapeOp.getSrcType());
    if (res != SliceVerificationResult::Success)
      return rewriter.notifyMatchFailure(insertSliceOp,
                                         "expected rank increasing expansion");

    rewriter.modifyOpInPlace(insertSliceOp, [&]() {
      insertSliceOp.getSourceMutable().assign(expandShapeOp.getSrc());
    });
    return success();
  }
};

/// Pattern to bubble up a tensor.expand_shape op through a producer
/// tensor.collapse_shape op that has non intersecting reassociations.
struct BubbleUpExpandThroughParallelCollapse
    : public OpRewritePattern<tensor::ExpandShapeOp> {
  using OpRewritePattern<tensor::ExpandShapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExpandShapeOp expandOp,
                                PatternRewriter &rewriter) const override {
    auto collapseOp =
        expandOp.getSrc().getDefiningOp<tensor::CollapseShapeOp>();
    if (!collapseOp)
      return failure();
    auto expandReInds = expandOp.getReassociationIndices();
    auto collapseReInds = collapseOp.getReassociationIndices();

    // Special case where the collapsed tensor to expand is a 0-D tensor,
    // then the reassociation maps will be empty and not produce valid results.
    if (expandReInds.size() == 0) {
      return failure();
    }

    // Reshapes are parallel to each other if none of the reassociation indices
    // have greater than 1 index for both reshapes.
    for (auto [expandReassociation, collapseReassociation] :
         llvm::zip_equal(expandReInds, collapseReInds)) {
      if (collapseReassociation.size() != 1 && expandReassociation.size() != 1)
        return failure();
    }

    // Compute new reassociation indices and expanded/collaped shapes.
    SmallVector<ReassociationIndices> newExpandReInds, newCollapseReInds;
    Location loc = expandOp->getLoc();
    SmallVector<OpFoldResult> collapseSizes =
        tensor::getMixedSizes(rewriter, loc, collapseOp.getSrc());
    SmallVector<OpFoldResult> expandSizes(getMixedValues(
        expandOp.getStaticOutputShape(), expandOp.getOutputShape(), rewriter));
    SmallVector<OpFoldResult> newExpandSizes;
    int64_t index = 0, expandIndex = 0, collapseIndex = 0;
    for (auto [idx, collapseReassociation] : llvm::enumerate(collapseReInds)) {
      if (collapseReassociation.size() != 1) {
        ReassociationIndices newCollapseReassociation;
        for (size_t i = 0; i < collapseReassociation.size(); ++i) {
          newCollapseReassociation.push_back(index);
          newExpandReInds.push_back({index++});
          newExpandSizes.push_back(collapseSizes[collapseIndex++]);
        }
        newCollapseReInds.push_back(newCollapseReassociation);
        expandIndex++;
        continue;
      }
      ReassociationIndices newExpandReassociation;
      auto expandReassociation = expandReInds[idx];
      for (size_t i = 0; i < expandReassociation.size(); ++i) {
        newExpandReassociation.push_back(index);
        newCollapseReInds.push_back({index++});
        newExpandSizes.push_back(expandSizes[expandIndex++]);
      }
      newExpandReInds.push_back(newExpandReassociation);
      collapseIndex++;
    }

    // Swap reshape order.
    SmallVector<Value> dynamicSizes;
    SmallVector<int64_t> staticSizes;
    dispatchIndexOpFoldResults(newExpandSizes, dynamicSizes, staticSizes);
    auto expandResultType = expandOp.getResultType().clone(staticSizes);
    auto newExpand = rewriter.create<tensor::ExpandShapeOp>(
        loc, expandResultType, collapseOp.getSrc(), newExpandReInds,
        newExpandSizes);
    rewriter.replaceOpWithNewOp<tensor::CollapseShapeOp>(
        expandOp, newExpand.getResult(), newCollapseReInds);
    return success();
  }
};

/// Converts `tensor.extract_slice(tensor.expand_shape)` to
/// `tensor.expand_shape(tensor.extract_slice)`.
///
/// For this transformation to be possible, the slice must be fully contiguous
/// within each reassociation group of the expand_shape. A slice is defined as
/// fully contiguous within a reassociation group if after flattening the
/// reassociation group to a single 1D range, then the slice taken out of the
/// group could be defined as a single contiguous subrange within that range.
///
/// Rank reducing slices are not supported.
///
/// Example:
/// The transformation is possible because each reassociation group has a
/// contiguous slice (i.e., [2x4->2x4], [2x8->1x5], [4x2x4->1x1x4]).
/// ```
/// BEFORE:
/// %reshape = tensor.expand_shape %in [[0, 1], [2, 3], [4, 5, 6]]
///     tensor<8x16x32xf32> to tensor<2x4x2x8x4x2x4xf32>
/// %slice = tensor.extract_slice %reshape ...
///     tensor<2x4x2x8x4x2x4xf32> to tensor<2x4x1x5x1x1x4xf32>
///
/// AFTER:
/// %slice = tensor.extract_slice %in ...
///     tensor<8x16x32xf32> to tensor<8x5x4xf32>
/// %reshape = tensor.expand_shape %slice [[0, 1], [2, 3], [4, 5, 6]]
///     tensor<8x5x4xf32> to tensor<2x4x1x5x1x1x4xf32>
/// ```
///
/// Note - this pattern could be extended to be a swap pattern between
/// `tensor.expand_shape` and `tensor.extract_slice`, but is currently
/// implemented only as a bubble up pattern for `tensor.extract_slice`.
struct BubbleUpExpandShapeThroughExtractSlice
    : public OpRewritePattern<tensor::ExtractSliceOp> {
  using OpRewritePattern<tensor::ExtractSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractSliceOp sliceOp,
                                PatternRewriter &rewriter) const override {
    auto expandShapeOp =
        sliceOp.getSource().getDefiningOp<tensor::ExpandShapeOp>();

    if (checkPreconditionForBubbleUpExtractSlice(sliceOp, expandShapeOp,
                                                 rewriter)
            .failed())
      return failure();

    // The tensor.extract_slice before applying the pattern works on the result
    // of the tensor.expand_shape, so variables (i.e. inputs for ExtractSliceOp)
    // referring to the state before applying the pattern are named with the
    // prefix "expanded", and ones referring to the state after applying the
    // pattern are named with the prefix "collapsed".
    SmallVector<OpFoldResult> expandedOffsets = sliceOp.getMixedOffsets();
    SmallVector<OpFoldResult> expandedSizes = sliceOp.getMixedSizes();
    SmallVector<OpFoldResult> expandedShape =
        getMixedValues(expandShapeOp.getStaticOutputShape(),
                       expandShapeOp.getOutputShape(), rewriter);

    // Helper variables and function for accumulating the size values.
    Location loc = expandShapeOp->getLoc();
    AffineExpr d0, d1, d2;
    bindDims(rewriter.getContext(), d0, d1, d2);
    // Multiply two integers.
    auto mul = [&](OpFoldResult v1, OpFoldResult v2) {
      auto mulMap = AffineMap::get(2, 0, {d0 * d1});
      return affine::makeComposedFoldedAffineApply(rewriter, loc, mulMap,
                                                   {v1, v2});
    };

    // Compute new offsets, sizes, and strides for tensor.extract_slice.
    // The new tensor.extract_slice will work on a tensor that has has a rank of
    // ReassociationIndices.size(). In the loop a single offset, size, and
    // stride value is computed per reassociation group.
    SmallVector<OpFoldResult> collapsedOffsets, collapsedSizes,
        collapsedStrides;
    for (const ReassociationIndices &indices :
         expandShapeOp.getReassociationIndices()) {
      // collapsedSize will hold the size of the single dim that represents the
      // reassociation group in the non expanded tensor.
      OpFoldResult collapsedSize = rewriter.getIndexAttr(1);
      // The reassocGroupSizes and reassocGroupOffsets are used to create an
      // affine.linearize_index op to linearize the single offset value required
      // for this reassociation group.
      SmallVector<OpFoldResult> reassocGroupSizes, reassocGroupOffsets;

      for (long expandedDim : indices) {
        // reassocGroupSizes and reassocGroupOffsets can be obtained directly
        // from the expanded state, but the collapsed size requires calculation
        // as it did not previously exist.
        reassocGroupSizes.push_back(expandedShape[expandedDim]);
        reassocGroupOffsets.push_back(expandedOffsets[expandedDim]);
        collapsedSize = mul(collapsedSize, expandedSizes[expandedDim]);
      }

      SmallVector<Value> offsetVals =
          llvm::map_to_vector(reassocGroupOffsets, [&](OpFoldResult ofr) {
            return getValueOrCreateConstantIndexOp(rewriter, loc, ofr);
          });
      OpFoldResult collapsedOffset =
          rewriter
              .create<affine::AffineLinearizeIndexOp>(loc, offsetVals,
                                                      reassocGroupSizes,
                                                      /*disjoint=*/true)
              .getResult();
      collapsedOffsets.push_back(collapsedOffset);
      collapsedSizes.push_back(collapsedSize);

      // Only unit stride is supported.
      collapsedStrides.push_back(rewriter.getIndexAttr(1));
    }

    // The shape of the result can be obtained from the sizes passed in.
    SmallVector<Value> dynDims;
    SmallVector<int64_t> shape;
    dispatchIndexOpFoldResults(expandedSizes, dynDims, shape);
    RankedTensorType resultType = RankedTensorType::get(
        shape, expandShapeOp.getResultType().getElementType());

    // Create a new ExtractSliceOp and ExpandShapeOp.
    Value newSliceOp = rewriter.create<tensor::ExtractSliceOp>(
        loc, expandShapeOp.getSrc(), collapsedOffsets, collapsedSizes,
        collapsedStrides);
    rewriter.replaceOpWithNewOp<tensor::ExpandShapeOp>(
        sliceOp, resultType, newSliceOp,
        expandShapeOp.getReassociationIndices(), expandedSizes);
    return success();
  }

  // Helper function to check if all the required conditions for the
  // tensor.extract_slice to be bubbled up through the tensor.expand_shape are
  // met.
  LogicalResult
  checkPreconditionForBubbleUpExtractSlice(tensor::ExtractSliceOp sliceOp,
                                           tensor::ExpandShapeOp expandShapeOp,
                                           PatternRewriter &rewriter) const {

    if (!expandShapeOp) {
      return rewriter.notifyMatchFailure(
          sliceOp, "tensor.extract_slice source not produced by expand_shape");
    }

    if (!sliceOp.hasUnitStride()) {
      return rewriter.notifyMatchFailure(
          sliceOp, "unsupported: non-unit stride. Only contiguous slices can "
                   "be supported in this transformation.");
    }

    SmallVector<OpFoldResult> offsets = sliceOp.getMixedOffsets();
    SmallVector<OpFoldResult> sizes = sliceOp.getMixedSizes();

    if (static_cast<size_t>(sliceOp.getResultType().getRank()) !=
        sizes.size()) {
      return rewriter.notifyMatchFailure(sliceOp,
                                         "unimplemented: rank reducing slice");
    }

    SmallVector<OpFoldResult> outputShape =
        getMixedValues(expandShapeOp.getStaticOutputShape(),
                       expandShapeOp.getOutputShape(), rewriter);

    std::function<bool(OpFoldResult, OpFoldResult, OpFoldResult)>
        isZeroOffsetAndFullSize =
            [](OpFoldResult offset, OpFoldResult sliceSize, OpFoldResult size) {
              if (!isConstantIntValue(offset, 0))
                return false;
              FailureOr<bool> maybeEqual =
                  ValueBoundsConstraintSet::areEqual(sliceSize, size);
              return llvm::succeeded(maybeEqual) && maybeEqual.value();
            };

    // Check that the slice is contiguous within each reassociation group.
    // The slice is contiguous only if after the first dimension where a non
    // unit slice is taken, the slice size on all subsequent dimensions of the
    // group is equal to the entire size of the dimension.
    // Examples of contiguous slices:
    //   full sizes: [8, 8, 10] slice offsets: [0, 0, 0] slice sizes: [1, 1, 10]
    //   full sizes: [5, 10] slice offsets: [3, 0] slice sizes: [2, 10]
    // Examples of non contiguous slices:
    //   full sizes: [8, 8, 10] slice offsets: [0, 0, 0] slice sizes: [1, 2, 5]
    //   full sizes: [5, 10] slice offsets: [0, 4] slice sizes: [2, 5]
    for (const ReassociationIndices &indices :
         expandShapeOp.getReassociationIndices()) {
      int64_t i = 0;
      int64_t e = indices.size();
      // Find the first expanded dim after the first dim with non-unit extracted
      // size.
      for (; i < e; ++i) {
        if (!isConstantIntValue(sizes[indices[i]], 1)) {
          // +1 to skip the first non-unit size dim.
          i++;
          break;
        }
      }

      // Verify that all subsequent dimensions extract the full size of the
      // source tensor.
      for (; i < e; ++i) {
        int64_t expandedDim = indices[i];
        if (!isZeroOffsetAndFullSize(offsets[expandedDim], sizes[expandedDim],
                                     outputShape[expandedDim])) {
          return rewriter.notifyMatchFailure(
              sliceOp, "Not a contiguous slice of the expanded tensor.");
        }
      }
    }

    return success();
  }
};

} // namespace

void mlir::tensor::populateReassociativeReshapeFoldingPatterns(
    RewritePatternSet &patterns) {
  patterns
      .add<FoldExpandOfRankReducingExtract, FoldUnPaddingCollapseIntoExtract,
           FoldInsertOfRankReducingInsert<tensor::InsertSliceOp>,
           FoldInsertOfRankReducingInsert<tensor::ParallelInsertSliceOp>,
           FoldPaddingExpandIntoInsert<tensor::InsertSliceOp>,
           FoldPaddingExpandIntoInsert<tensor::ParallelInsertSliceOp>>(
          patterns.getContext());
}

void mlir::tensor::populateBubbleUpExpandShapePatterns(
    RewritePatternSet &patterns) {
  patterns.add<BubbleUpExpandThroughParallelCollapse>(patterns.getContext());
}

void mlir::tensor::populateBubbleUpExtractSliceOpPatterns(
    RewritePatternSet &patterns) {
  patterns.add<BubbleUpExpandShapeThroughExtractSlice>(patterns.getContext());
}
