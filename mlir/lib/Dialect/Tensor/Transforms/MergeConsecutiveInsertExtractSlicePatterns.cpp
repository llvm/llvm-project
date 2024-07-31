//===- MergeConsecutiveInsertExtractSlicePatterns.cpp ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/ViewLikeInterfaceUtils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::tensor;

namespace {
/// Merges consecutive tensor.extract_slice ops into one.
// TODO: move to FoldTensorSubsetOps and unify APIs with FoldMemRefAliasOps.
struct MergeConsecutiveExtractSlice : public OpRewritePattern<ExtractSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ExtractSliceOp nextOp,
                                PatternRewriter &rewriter) const override {
    auto prevOp = nextOp.getSource().getDefiningOp<ExtractSliceOp>();
    if (!prevOp)
      return failure();

    SmallVector<OpFoldResult> newOffsets, newSizes, newStrides;
    if (failed(affine::mergeOffsetsSizesAndStrides(
            rewriter, nextOp.getLoc(), prevOp, nextOp, prevOp.getDroppedDims(),
            newOffsets, newSizes, newStrides)))
      return failure();

    rewriter.replaceOpWithNewOp<ExtractSliceOp>(nextOp, nextOp.getType(),
                                                prevOp.getSource(), newOffsets,
                                                newSizes, newStrides);
    return success();
  }
};

/// Merges consecutive tensor.insert_slice ops into one.
// TODO: move to FoldTensorSubsetOps and unify APIs with FoldMemRefAliasOps.
template <typename OpTy>
struct MergeConsecutiveInsertSlice : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy nextOp,
                                PatternRewriter &rewriter) const override {
    auto prevOp = nextOp.getSource().template getDefiningOp<InsertSliceOp>();
    if (!prevOp)
      return failure();

    if (!prevOp.hasUnitStride() || !nextOp.hasUnitStride())
      return failure();

    // The first insert_slice op should be rank reducing to make sure we cover
    // the full source tensor to be inserted in the second insert_slice op.
    SliceVerificationResult result =
        isRankReducedType(prevOp.getDestType(), prevOp.getSourceType());
    if (result != SliceVerificationResult::Success)
      return failure();

    // Dynamic dimensions can pass rank reducing check in the above, e.g,
    // inserting <?xf32> into <1x?x1xf32>. For such cases we cannot be certain
    // the dynamic size covers the full tensor.
    if (!prevOp.getSourceType().hasStaticShape() ||
        !prevOp.getDestType().hasStaticShape())
      return failure();

    rewriter.replaceOpWithNewOp<OpTy>(
        nextOp, prevOp.getSource(), nextOp.getDest(), nextOp.getMixedOffsets(),
        nextOp.getMixedSizes(), nextOp.getMixedStrides());
    return success();
  }
};

/// Drop redundant rank expansion of insert_slice that are directly followed
/// by extract_slice. E.g.:
/// %0 = tensor.insert_slice ... : tensor<5x10xf32> into tensor<1x1x5x10xf32>
/// %1 = tensor.extract_slice %0[0, 0, 2, 3] [1, 1, 2, 2] [1, 1, 1, 1]
///     : tensor<1x1x5x10xf32> to tensor<2x2xf32>
struct DropRedundantRankExpansionOnExtractSliceOfInsertSlice
    : public OpRewritePattern<ExtractSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ExtractSliceOp extractSliceOp,
                                PatternRewriter &rewriter) const override {
    // Nothing to do if no dims are dropped.
    llvm::SmallBitVector droppedDims = extractSliceOp.getDroppedDims();
    if (droppedDims.none())
      return failure();

    // Look for tensor.insert_slice op that has an inverse rank expansion.
    auto insertSliceOp =
        extractSliceOp.getSource().getDefiningOp<InsertSliceOp>();
    if (!insertSliceOp)
      return failure();
    llvm::SmallBitVector expandedDims = insertSliceOp.getDroppedDims();

    // TODO: This could be extended to support cases where the dropped dims are
    // a subset of the expanded dims.
    if (expandedDims != droppedDims)
      return failure();

    // The tensor.insert_slice may not be redundant if it has multiple users.
    if (!insertSliceOp->hasOneUse())
      return failure();

    // Only consider tensor.insert_slice ops that are pure rank-reductions.
    // I.e., no elements are taken from the destination.
    if (!isCastLikeInsertSliceOp(insertSliceOp))
      return failure();

    // Extract directly from the source.
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(extractSliceOp);
    SmallVector<OpFoldResult> newOffsets, newSizes, newStrides;
    for (int64_t i = 0, e = extractSliceOp.getSourceType().getRank(); i < e;
         ++i) {
      if (droppedDims.test(i))
        continue;
      newOffsets.push_back(extractSliceOp.getMixedOffsets()[i]);
      newSizes.push_back(extractSliceOp.getMixedSizes()[i]);
      newStrides.push_back(extractSliceOp.getMixedStrides()[i]);
    }
    rewriter.replaceOpWithNewOp<ExtractSliceOp>(
        extractSliceOp, /*source=*/insertSliceOp.getSource(), newOffsets,
        newSizes, newStrides);
    rewriter.eraseOp(insertSliceOp);
    return success();
  }
};

/// Drop redundant rank expansion of insert_slice that direclty follows
/// extract_slice.
///
/// This can be done when the insert_slice op purely expands ranks (adds unit
/// dims) and the extrace_slice drops corresponding unit dims. For example:
///
/// %extracted_slice = tensor.extract_slice %in[0, 0] [1, 8] [1, 1]
///     : tensor<2x8xf32> to tensor<8xf32>
/// %inserted_slice = tensor.insert_slice %extracted_slice
///     into %dest[0, 0] [1, 8] [1, 1]
///     : tensor<8xf32> into tensor<1x8xf32>
///
/// can be folded into:
///
/// %extracted_slice = tensor.extract_slice %in[0, 0] [1, 8] [1, 1]
///     : tensor<2x8xf32> to tensor<1x8xf32>
struct DropRedundantRankExpansionOnInsertSliceOfExtractSlice final
    : public OpRewritePattern<tensor::InsertSliceOp> {
  using OpRewritePattern<tensor::InsertSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::InsertSliceOp insertSliceOp,
                                PatternRewriter &rewriter) const override {
    auto extractSliceOp =
        insertSliceOp.getSource().getDefiningOp<tensor::ExtractSliceOp>();
    if (!extractSliceOp) {
      return rewriter.notifyMatchFailure(insertSliceOp,
                                         "source is not extract_slice");
    }

    // Can't fold if the extract_slice op has other users.
    if (!extractSliceOp->hasOneUse()) {
      return rewriter.notifyMatchFailure(insertSliceOp,
                                         "source has multi-uses");
    }

    // Check if the insert_slice op purely expands ranks (add unit dims).
    if (!isCastLikeInsertSliceOp(insertSliceOp)) {
      return rewriter.notifyMatchFailure(insertSliceOp,
                                         "insert_slice is not cast-like");
    }

    llvm::SmallBitVector extractDroppedDims = extractSliceOp.getDroppedDims();
    llvm::SmallBitVector insertDroppedDims = insertSliceOp.getDroppedDims();
    // Can't fold if the insert_slice op expands to more dims.
    if (extractDroppedDims.size() < insertDroppedDims.size()) {
      return rewriter.notifyMatchFailure(insertSliceOp,
                                         "insert_slice expands more dims");
    }

    // Try to match the extract dropped dims to the insert dropped dims. This is
    // done by scanning the dims of extract_slice and find the left-most one can
    // match the dim of insert_slice. If a match is found, advance the dim of
    // insert_slice to match the next one.
    unsigned insertDimPos = 0;
    for (unsigned extractDimPos = 0; extractDimPos < extractDroppedDims.size();
         ++extractDimPos) {
      // Matched all dims.
      if (insertDimPos == insertDroppedDims.size())
        break;

      bool isExtractDropped = extractDroppedDims[extractDimPos];
      bool isInsertDropped = insertDroppedDims[insertDimPos];
      // Match if both sides drop/keep the dim. Advance and match the next dim
      // of insert_slice.
      if (isExtractDropped == isInsertDropped) {
        insertDimPos += 1;
      } else if (!isExtractDropped && isInsertDropped) {
        // Not enough extract dropped dims to match the insert dropped dims.
        return rewriter.notifyMatchFailure(insertSliceOp,
                                           "insert_slice drops more unit dims");
      }
      // If the dim is dropped by extract_slice and not by insert_slice, look
      // the next dim of extract_slice to see if it can match the current dim of
      // insert_slice.
    }
    // Can't match some insert dims.
    if (insertDimPos != insertDroppedDims.size()) {
      return rewriter.notifyMatchFailure(insertSliceOp,
                                         "insert_slice has unmatched dims");
    }

    rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(
        insertSliceOp, insertSliceOp.getType(), extractSliceOp.getSource(),
        extractSliceOp.getMixedOffsets(), extractSliceOp.getMixedSizes(),
        extractSliceOp.getMixedStrides());
    rewriter.eraseOp(extractSliceOp);

    return success();
  }
};
} // namespace

void mlir::tensor::populateMergeConsecutiveInsertExtractSlicePatterns(
    RewritePatternSet &patterns) {
  patterns.add<MergeConsecutiveExtractSlice,
               MergeConsecutiveInsertSlice<InsertSliceOp>,
               MergeConsecutiveInsertSlice<ParallelInsertSliceOp>>(
      patterns.getContext());
}

void mlir::tensor::populateDropRedundantInsertSliceRankExpansionPatterns(
    RewritePatternSet &patterns) {
  patterns.add<DropRedundantRankExpansionOnExtractSliceOfInsertSlice,
               DropRedundantRankExpansionOnInsertSliceOfExtractSlice>(
      patterns.getContext());
}
