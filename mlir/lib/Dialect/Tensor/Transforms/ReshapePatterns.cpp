//===- RankReductionPatterns.cpp - Patterns related to rank reductions ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/Debug.h"

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
} // namespace

void mlir::tensor::populateReassociativeReshapeFoldingPatterns(
    RewritePatternSet &patterns) {
  patterns.add<FoldExpandOfRankReducingExtract,
               FoldInsertOfRankReducingInsert<tensor::InsertSliceOp>,
               FoldInsertOfRankReducingInsert<tensor::ParallelInsertSliceOp>,
               FoldPaddingExpandIntoInsert<tensor::InsertSliceOp>,
               FoldPaddingExpandIntoInsert<tensor::ParallelInsertSliceOp>>(
      patterns.getContext());
}
