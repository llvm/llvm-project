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
#include "llvm/ADT/STLExtras.h"
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

    Value unPaddedExtractSlice = tensor::ExtractSliceOp::create(
        rewriter, extractSliceOp.getLoc(), collapseShapeOp.getResultType(),
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

    // Reshapes are parallel to each other (by construction the number of
    // reassociations specified in the collapse and expand are the same), if at
    // any position
    // 1. either the reassociation indices are of the same size, or
    // 2. either the reassociation in the collapse or the expand is of size 1.
    ArrayRef<int64_t> staticSourceSize = collapseOp.getSrcType().getShape();
    ArrayRef<int64_t> staticResultSize = expandOp.getStaticOutputShape();
    for (auto [expandReassociation, collapseReassociation] :
         llvm::zip_equal(expandReInds, collapseReInds)) {
      if (collapseReassociation.size() == expandReassociation.size()) {
        // Even if the reassociations are the same, the collapse/expand should
        // result in the same dimensions. i.e  4x8x2 into 64 should be expanded
        // into 4x8x2 again. In presense of dynamic dimensions one can only
        // verify "equality" when there is only one dynamic dimension present,
        // and all other static dimensions are equal.
        ArrayRef<int64_t> collapsedStaticShapes = staticSourceSize.slice(
            collapseReassociation.front(), collapseReassociation.size());
        int64_t numCollapsedDynamic =
            llvm::count_if(collapsedStaticShapes, ShapedType::isDynamic);
        ArrayRef<int64_t> expandedStaticShapes = staticResultSize.slice(
            expandReassociation.front(), expandReassociation.size());
        int64_t numExpandedDynamic =
            llvm::count_if(expandedStaticShapes, ShapedType::isDynamic);
        if (numCollapsedDynamic > 1 || numExpandedDynamic > 1 ||
            collapsedStaticShapes != expandedStaticShapes) {
          return failure();
        }
        continue;
      }
      // If the reassociations are not same, one or the other needs to be of
      // size one.
      if (collapseReassociation.size() != 1 && expandReassociation.size() != 1)
        return failure();
    }

    // Compute new reassociation indices and expanded/collaped shapes.
    SmallVector<ReassociationIndices> newExpandReInds, newCollapseReInds;
    Location loc = expandOp->getLoc();
    SmallVector<OpFoldResult> sourceSizes =
        tensor::getMixedSizes(rewriter, loc, collapseOp.getSrc());
    SmallVector<OpFoldResult> resultSizes = expandOp.getMixedOutputShape();
    SmallVector<OpFoldResult> newExpandSizes;

    int64_t newExpandIndex = 0, newCollapseIndex = 0, sourceSizeIndex = 0,
            resultSizeIndex = 0;

    for (size_t idx = 0, idxEnd = collapseReInds.size(); idx < idxEnd; idx++) {
      auto &collapseReassociation = collapseReInds[idx];
      auto &expandReassociation = expandReInds[idx];

      // Case 1. The reassociations are same in the collapse producer
      // and expand consumer. In the swapped expand, each of the final
      // dimensions are kept as is in the expand and the collapse. So,
      // for every element in the `ReassocationIndices` vector add a new
      // `ReassociationIndices` vector for the swapped expand and collapse
      // (of size 1).
      if (collapseReassociation.size() == expandReassociation.size()) {
        for (size_t i = 0; i < collapseReassociation.size(); ++i) {
          newCollapseReInds.push_back({newCollapseIndex++});
          newExpandReInds.push_back({newExpandIndex++});
          newExpandSizes.push_back(resultSizes[resultSizeIndex++]);
          sourceSizeIndex++;
        }
        continue;
      }

      // Case 2. The `ReassociationIndices` in the collapse is of size > 1 (and
      // in the expand is of size == 1). In this case, the original dimensions
      // are preserved on expansion and collapsed subsequently.
      if (collapseReassociation.size() != 1) {
        ReassociationIndices newCollapseReassociation;
        for (size_t i = 0; i < collapseReassociation.size(); ++i) {
          newCollapseReassociation.push_back(newCollapseIndex++);
          newExpandReInds.push_back({newExpandIndex++});
          newExpandSizes.push_back(sourceSizes[sourceSizeIndex++]);
        }
        resultSizeIndex++;
        newCollapseReInds.push_back(newCollapseReassociation);
        continue;
      }

      // Case 3. The `ReassociationIndices` in the expand is of size > 1 (and
      // in the collapse is of size == 1). In this case, the expansion happens
      // first and the expanded dimensions are preserved on collapse.
      ReassociationIndices newExpandReassociation;
      for (size_t i = 0; i < expandReassociation.size(); ++i) {
        newExpandReassociation.push_back(newExpandIndex++);
        newCollapseReInds.push_back({newCollapseIndex++});
        newExpandSizes.push_back(resultSizes[resultSizeIndex++]);
      }
      newExpandReInds.push_back(newExpandReassociation);
      sourceSizeIndex++;
    }

    // Swap reshape order.
    SmallVector<Value> dynamicSizes;
    SmallVector<int64_t> staticSizes;
    dispatchIndexOpFoldResults(newExpandSizes, dynamicSizes, staticSizes);
    auto expandResultType = expandOp.getResultType().clone(staticSizes);
    Value newCollapseSrc = collapseOp.getSrc();
    // If the number of reassociation indices in the new `expand_shape` op
    // matches the number of dimensions of the result, then the expand_shape
    // is a no-op.
    if (newExpandReInds.size() != newExpandSizes.size()) {
      newCollapseSrc = tensor::ExpandShapeOp::create(
          rewriter, loc, expandResultType, newCollapseSrc, newExpandReInds,
          newExpandSizes);
    }

    // If the number of reassociation indices in the new `collapse_shape` op
    // matches the number of dimensions of the source, then the collapse_shape
    // is a no-op.
    Value replacement = newCollapseSrc;
    if (newCollapseReInds.size() != newExpandSizes.size()) {
      replacement = tensor::CollapseShapeOp::create(
          rewriter, loc, newCollapseSrc, newCollapseReInds);
    }
    rewriter.replaceOp(expandOp, replacement);
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
struct BubbleUpExtractSliceThroughExpandShape
    : public OpRewritePattern<tensor::ExtractSliceOp> {
  using OpRewritePattern<tensor::ExtractSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractSliceOp sliceOp,
                                PatternRewriter &rewriter) const override {
    auto expandShapeOp =
        sliceOp.getSource().getDefiningOp<tensor::ExpandShapeOp>();
    if (!expandShapeOp) {
      return rewriter.notifyMatchFailure(
          sliceOp, "tensor.extract_slice source not produced by expand_shape");
    }
    SmallVector<ReassociationIndices> reassociation =
        expandShapeOp.getReassociationIndices();

    SmallVector<OpFoldResult> offsets, sizes, strides;
    if (failed(getCollapsedExtractSliceInfo(rewriter, sliceOp, reassociation,
                                            offsets, sizes, strides)))
      return failure();

    // The shape of the result can be obtained from the sizes passed in.
    SmallVector<OpFoldResult> expandedSizes = sliceOp.getMixedSizes();
    RankedTensorType resultType = sliceOp.getResultType();

    // Create a new ExtractSliceOp and ExpandShapeOp.
    Location loc = sliceOp.getLoc();
    Value newSliceOp = tensor::ExtractSliceOp::create(
        rewriter, loc, expandShapeOp.getSrc(), offsets, sizes, strides);
    rewriter.replaceOpWithNewOp<tensor::ExpandShapeOp>(
        sliceOp, resultType, newSliceOp,
        expandShapeOp.getReassociationIndices(), expandedSizes);
    return success();
  }
};

/// Converts `tensor.extract_slice(tensor.collapse_shape)` to
///          `tensor.collapse_shape(tensor.extract_slice)`.
///
/// For this transformation to be possible - after bubbling up, the extraction
/// of the contiguous slice must be representable as a single slice obtained via
/// tensor.extract_slice within each reassociation group of the src.
///
/// In case the size and offset extracted are static then this is possible if
/// the following conditions are met within each reassociation group:
/// Let T be a tensor of shape [A0, A1, ..., An] (these are the sizes of the
/// dimensions in the reassociation group), and let S = [S0, S1, ..., Sn] be the
/// shape of a desired slice. A slice of shape S can be extracted as a
/// contiguous span of elements if and only if there exists an index k in {0, 1,
/// ..., n} such that:
///      S_i = 1 for all i < k (that is, all leading dimensions are singleton),
///      1 <= S_k <= A_k (that is, non trivial slicing occurs along exactly
///                       one dimension),
///      S_i = A_i for all i > k (that is, all trailing dimensions are preserved
///      in full).
/// In other words, the slice shape S must be of the form:
/// [ 1, 1, ..., 1, Sk, Ak + 1, Ak + 2, ...,An ]
///
/// In case the size and/or offset extracted are dynamic then this is possible
/// only if there is single dimension in the reassociation group that has a size
/// not equal to 1.
/// In other words, the tensor shape must be of the form:
/// [ 1, 1, ..., 1, A, 1, ...,1 ]
/// Note - it might be possible to enable this pattern for more cases when the
/// size/offset are dynamic via performing an analysis of the possible values
/// that could be given to the size/offset.
///
/// Example:
/// The transformation is possible because each reassociation group can be
/// represented as a contiguous slice (i.e., [8x16->2x16], [1x7->1x?],
/// [20->10]).
/// ```
/// BEFORE:
/// %collapse = tensor.collapse_shape %src [[0, 1], [2, 3], [4]] ...
///     tensor<8x16x1x7x20f32> to tensor<128x7x20xf32>
/// %slice = tensor.extract_slice %slice [0, 0, 0][32, %size, 10][1, 1, 1]
///     tensor<128x7x20xf32> to tensor<32x?x10xf32>
///
/// AFTER:
/// %slice = tensor.extract_slice %src [0, 0, 0, 0, 0][2, 16, 1, %size, 10]
//           [1, 1, 1, 1, 1] : tensor<8x16x1x7x20f32> to tensor<2x16x1x?x10xf32>
/// %collapse = tensor.collapse_shape %slice [[0, 1], [2, 3], [4]] ...
///     tensor<2x16x1x?x10xf32> to tensor<32x?x10xf32>
/// ```
///
/// Negative example:
/// The transformation is not possible because we cannot use a single slice to
/// represent the reassociation group [2x3x10->???]. If we would want the
/// collapse to be after the extraction, we would need to extract multiple
/// slices and concat them together.
/// ```
/// %collapse = tensor.collapse_shape %src [[0, 1, 2]] : tensor<2x3x10xf32> into
/// tensor<60xf32> %extract = tensor.extract_slice %collapse[0][15][1] :
///                                      tensor<60xf32> to tensor<15xf32>
/// ```
/// If we would want the collapse to be after the extraction, a possible
/// alternate transformation could be to extract multiple slices and concat them
/// together:
/// ```
/// %extract_1 = tensor.extract_slice %src[0, 0, 0][1, 1, 10] :
///                               tensor<2x3x10xf32> to tensor <1x1x10xf32>
/// %extract_2 = tensor.extract_slice %src[0, 1, 0][1, 1, 5] :
///                               tensor<2x3x10xf32> to tensor <1x1x5xf32>
/// %concat = tosa.concat %extract_1, %extract_2 {axis = 0 : i32} :
///                    (<1x1x10xf32>, <1x1x5xf32>) -> <1x1x15xf32>
/// %collapse = tensor.collapse_shape %concat [[0, 1, 2]] : tensor<1x1x15xf32>
///                                                       to tensor<15xf32>
/// ```
/// But this is not the intended purpose of the transformation.
struct BubbleUpExtractSliceThroughCollapseShape
    : public OpRewritePattern<tensor::ExtractSliceOp> {
  using OpRewritePattern<tensor::ExtractSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractSliceOp sliceOp,
                                PatternRewriter &rewriter) const override {
    auto collapseShapeOp =
        sliceOp.getSource().getDefiningOp<tensor::CollapseShapeOp>();
    if (!collapseShapeOp) {
      return rewriter.notifyMatchFailure(
          sliceOp,
          "tensor.extract_slice source not produced by tensor.collapse_shape");
    }

    SmallVector<OpFoldResult> offsets, sizes, strides;
    if (failed(getExpandedExtractSliceInfo(
            rewriter, sliceOp, collapseShapeOp.getReassociationIndices(),
            collapseShapeOp.getSrcType().getShape(), offsets, sizes, strides)))
      return failure();

    Value newSliceOp = tensor::ExtractSliceOp::create(
        rewriter, collapseShapeOp->getLoc(), collapseShapeOp.getSrc(), offsets,
        sizes, strides);
    rewriter.replaceOpWithNewOp<tensor::CollapseShapeOp>(
        sliceOp, sliceOp.getResultType(), newSliceOp,
        collapseShapeOp.getReassociationIndices());

    return success();
  }
};

} // namespace

LogicalResult mlir::tensor::getCollapsedExtractSliceInfo(
    OpBuilder &b, tensor::ExtractSliceOp sliceOp,
    ArrayRef<ReassociationIndices> reassociation,
    SmallVectorImpl<OpFoldResult> &collapsedOffsets,
    SmallVectorImpl<OpFoldResult> &collapsedSizes,
    SmallVectorImpl<OpFoldResult> &collapsedStrides) {
  if (!sliceOp.hasUnitStride()) {
    return failure();
  }

  SmallVector<OpFoldResult> offsets = sliceOp.getMixedOffsets();
  SmallVector<OpFoldResult> sizes = sliceOp.getMixedSizes();

  if (static_cast<size_t>(sliceOp.getResultType().getRank()) != sizes.size()) {
    return failure();
  }

  auto isZeroOffsetAndFullSize = [&](OpFoldResult offset,
                                     OpFoldResult sliceSize, int64_t inputDim) {
    if (!isZeroInteger(offset))
      return false;
    ValueBoundsConstraintSet::Variable inputSize(sliceOp.getSource(), inputDim);
    FailureOr<bool> maybeEqual =
        ValueBoundsConstraintSet::areEqual(sliceSize, inputSize);
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
  for (const ReassociationIndices &indices : reassociation) {
    int64_t i = 0;
    int64_t e = indices.size();
    // Find the first expanded dim after the first dim with non-unit extracted
    // size.
    for (; i < e; ++i) {
      if (!isOneInteger(sizes[indices[i]])) {
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
                                   expandedDim)) {
        return failure();
      }
    }
  }

  // The tensor.extract_slice before applying the pattern works on the result
  // of the tensor.expand_shape, so variables (i.e. inputs for ExtractSliceOp)
  // referring to the state before applying the pattern are named with the
  // prefix "expanded", and ones referring to the state after applying the
  // pattern are named with the prefix "collapsed".
  Location loc = sliceOp.getLoc();
  SmallVector<OpFoldResult> expandedOffsets = sliceOp.getMixedOffsets();
  SmallVector<OpFoldResult> expandedSizes = sliceOp.getMixedSizes();
  SmallVector<OpFoldResult> expandedShape =
      getMixedSizes(b, loc, sliceOp.getSource());

  // Helper variables and function for accumulating the size values.
  AffineExpr d0, d1, d2;
  bindDims(b.getContext(), d0, d1, d2);
  // Multiply two integers.
  auto mul = [&](OpFoldResult v1, OpFoldResult v2) {
    auto mulMap = AffineMap::get(2, 0, {d0 * d1});
    return affine::makeComposedFoldedAffineApply(b, loc, mulMap, {v1, v2});
  };

  // Compute new offsets, sizes, and strides for tensor.extract_slice.
  // The new tensor.extract_slice will work on a tensor that has has a rank of
  // ReassociationIndices.size(). In the loop a single offset, size, and
  // stride value is computed per reassociation group.
  for (const ReassociationIndices &indices : reassociation) {
    // collapsedSize will hold the size of the single dim that represents the
    // reassociation group in the non expanded tensor.
    OpFoldResult collapsedSize = b.getIndexAttr(1);
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
          return getValueOrCreateConstantIndexOp(b, loc, ofr);
        });
    OpFoldResult collapsedOffset = affine::AffineLinearizeIndexOp::create(
                                       b, loc, offsetVals, reassocGroupSizes,
                                       /*disjoint=*/true)
                                       .getResult();
    collapsedOffsets.push_back(collapsedOffset);
    collapsedSizes.push_back(collapsedSize);

    // Only unit stride is supported.
    collapsedStrides.push_back(b.getIndexAttr(1));
  }
  return success();
}

LogicalResult mlir::tensor::getExpandedExtractSliceInfo(
    OpBuilder &b, tensor::ExtractSliceOp sliceOp,
    ArrayRef<ReassociationIndices> reassociation,
    ArrayRef<int64_t> expandedShape,
    SmallVectorImpl<OpFoldResult> &expandedOffsets,
    SmallVectorImpl<OpFoldResult> &expandedSizes,
    SmallVectorImpl<OpFoldResult> &expandedStrides) {
  if (!sliceOp.hasUnitStride()) {
    return failure();
  }

  // The tensor.extract_slice before applying the pattern works on the result
  // of the tensor.collapse_shape, so variables (i.e. inputs for
  // ExtractSliceOp) referring to the state before applying the pattern are
  // named with the prefix "collapsed", and ones referring to the state after
  // applying the pattern are named with the prefix "expanded".
  SmallVector<OpFoldResult> collapsedOffsets = sliceOp.getMixedOffsets();
  SmallVector<OpFoldResult> collapsedSizes = sliceOp.getMixedSizes();
  if (static_cast<size_t>(sliceOp.getResultType().getRank()) !=
      collapsedSizes.size()) {
    return failure();
  }

  // Compute new offsets, sizes, and strides for tensor.extract_slice.
  // The new tensor.extract_slice will work on a tensor that has has a rank
  // equal to the rank of the src of the collapse_shape. In each iteration of
  // the loop, the offsets and sizes will be computed per reassociation group.
  expandedStrides.resize(expandedShape.size(), b.getIndexAttr(1));
  for (auto [collapsedSize, collapsedOffset, reassocIndices] :
       llvm::zip_equal(collapsedSizes, collapsedOffsets, reassociation)) {
    // CASE #1 - size and/or offset are dynamic.
    // In this case, the slice can be represented as a contiguous slice only
    // if there is a single dimension in the reassociation group that has a
    // size not equal to 1.
    if (isa<Value>(collapsedSize) || isa<Value>(collapsedOffset)) {
      int nonUnitSizeCount = 0;
      for (int64_t expandedShapeIdx : reassocIndices) {
        if (expandedShape[expandedShapeIdx] != 1) {
          nonUnitSizeCount++;
          expandedSizes.push_back(collapsedSize);
          expandedOffsets.push_back(collapsedOffset);
          continue;
        }

        expandedSizes.push_back(b.getIndexAttr(1));
        expandedOffsets.push_back(b.getIndexAttr(0));
      }

      if (nonUnitSizeCount != 1) {
        return failure();
      }
      continue;
    }

    // CASE #2 = size and offset are static.
    // Verify that the slice can be represented as a contiguous slice of the
    // src of the collapse_shape.
    // Checking this is done on order of most internal dimensions first,
    // so traversal is done in reverse order of the reassociation group.
    // If the expected slice shape is [1, 1, ..., 1, Sk, Ak + 1, Ak + 2,
    // ...,An] then we first find the size and offset for n...k+1 then for k
    // and then for k-1...0.

    // currentCollapsedsize and currentCollapsedOffset are initialized with
    // the original collapsed size and offset and divided by the expanded
    // shape size in each dimension as we go along the reassociation group.
    // In essence we are spreading the original collapsed size and offset over
    // the various expanded slice dimensions.
    // The variables are used both to check the validity of the slice and to
    // compute the expanded sizes and offsets.
    int64_t currentCollapsedsize = getConstantIntValue(collapsedSize).value();
    int64_t currentCollapsedOffset =
        getConstantIntValue(collapsedOffset).value();
    SmallVector<OpFoldResult> groupExpandedSizes, groupExpandedOffsets;
    ReassociationIndices reversedReassocIndices(reassocIndices.rbegin(),
                                                reassocIndices.rend());
    int64_t idx = 0;
    int64_t reassocGroupSize = reassocIndices.size();

    // First handle the trailing dimensions where the slice size should be
    // equal to the tensor shape and the offset should be 0 (n...k+1).
    for (; idx < reassocGroupSize; ++idx) {
      int64_t expandedShapeSize = expandedShape[reversedReassocIndices[idx]];

      if (currentCollapsedsize < expandedShapeSize)
        break;

      // We need to make sure that the slice size can be set to the shape size
      // and the offset to 0.
      if ((currentCollapsedsize % expandedShapeSize) != 0 ||
          (currentCollapsedOffset % expandedShapeSize) != 0) {
        return failure();
      }

      groupExpandedSizes.push_back(b.getIndexAttr(expandedShapeSize));
      groupExpandedOffsets.push_back(b.getIndexAttr(0));

      currentCollapsedsize /= expandedShapeSize;
      currentCollapsedOffset /= expandedShapeSize;
    }

    // Now handle the first dim where slicing occurs on (k).
    if (idx < reassocGroupSize) {
      int64_t expandedShapeSize = expandedShape[reversedReassocIndices[idx]];
      int64_t offsetInDim = currentCollapsedOffset % expandedShapeSize;
      // We need to make sure that the slice size in this dim + offset will
      // not exceed the shape size.
      if ((currentCollapsedsize + offsetInDim) >= expandedShapeSize) {
        return failure();
      }
      groupExpandedSizes.push_back(b.getIndexAttr(currentCollapsedsize));
      groupExpandedOffsets.push_back(b.getIndexAttr(offsetInDim));
      currentCollapsedOffset /= expandedShapeSize;
    }

    // Now handle the leading dimensions where the slice size is equal to 1
    // (k-1...0).
    // The size for these dimensions must be 1 because of how we constructed
    // the slice size of the expanded shape. We spread the original collapsed
    // size over the expanded shape sizes until we reached dimension k where
    // the remaining size was smaller than the expanded shape size, and spread
    // the remaining size on it. So, now we are left with only 1s.
    for (idx++; idx < reassocGroupSize; ++idx) {
      int64_t expandedShapeSize = expandedShape[reversedReassocIndices[idx]];
      int64_t offsetInDim = currentCollapsedOffset % expandedShapeSize;
      groupExpandedSizes.push_back(b.getIndexAttr(1));
      groupExpandedOffsets.push_back(b.getIndexAttr(offsetInDim));
      currentCollapsedOffset /= expandedShapeSize;
    }
    expandedSizes.append(groupExpandedSizes.rbegin(),
                         groupExpandedSizes.rend());
    expandedOffsets.append(groupExpandedOffsets.rbegin(),
                           groupExpandedOffsets.rend());
  }
  return success();
}

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
  patterns.add<BubbleUpExtractSliceThroughExpandShape,
               BubbleUpExtractSliceThroughCollapseShape>(patterns.getContext());
}
