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

#define DEBUG_TYPE "mlir-tensor-split-padding"

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
struct FoldInsertOfRankReducingInsert : public OpRewritePattern<InsertSliceOp> {
  using OpRewritePattern<InsertSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(InsertSliceOp insertSliceOp,
                                PatternRewriter &rewriter) const override {
    auto collapseShapeOp =
        insertSliceOp.getSource().getDefiningOp<CollapseShapeOp>();
    if (!collapseShapeOp)
      return failure();
    RankedTensorType srcType = collapseShapeOp.getSrcType();

    // Only cases where the CollapseShapeOp can be folded away entirely are
    // supported. Moreover, only simple cases where the resulting InsertSliceOp
    // has no rank-reduction anymore are supported at the moment.
    RankedTensorType nonReducingInsertType =
        RankedTensorType::get(insertSliceOp.getStaticSizes(),
                              insertSliceOp.getType().getElementType());
    if (nonReducingInsertType != srcType)
      return failure();

    SmallVector<OpFoldResult> mixedOffsets = insertSliceOp.getMixedOffsets();
    SmallVector<OpFoldResult> mixedSizes = insertSliceOp.getMixedSizes();
    SmallVector<OpFoldResult> mixedStrides = insertSliceOp.getMixedStrides();
    rewriter.replaceOpWithNewOp<tensor::InsertSliceOp>(
        insertSliceOp, collapseShapeOp.getSrc(), insertSliceOp.getDest(),
        mixedOffsets, mixedSizes, mixedStrides);
    return success();
  }
};
} // namespace

void mlir::tensor::populateReassociativeReshapeFoldingPatterns(
    RewritePatternSet &patterns) {
  patterns.add<FoldExpandOfRankReducingExtract, FoldInsertOfRankReducingInsert>(
      patterns.getContext());
}
