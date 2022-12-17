//===- EmptyOpPatterns.cpp - Patterns related to tensor.empty folding ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::tensor;

namespace {

template <typename ReshapeOp>
struct FoldEmptyTensorWithReshapeOp : public OpRewritePattern<ReshapeOp> {
  using OpRewritePattern<ReshapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReshapeOp reshapeOp,
                                PatternRewriter &rewriter) const override {
    if (!reshapeOp.getSrc().template getDefiningOp<EmptyOp>())
      return failure();
    Location loc = reshapeOp.getLoc();
    ReifiedRankedShapedTypeDims resultShapes;
    ReifyRankedShapedTypeOpInterface reifyShapedTypeInterface =
        cast<ReifyRankedShapedTypeOpInterface>(reshapeOp.getOperation());
    if (failed(reifyShapedTypeInterface.reifyResultShapes(rewriter,
                                                          resultShapes)) ||
        !llvm::hasSingleElement(resultShapes))
      return failure();
    // TODO: Do not drop tensor type encoding.
    Value emptyTensor =
        rewriter.create<EmptyOp>(loc, getAsOpFoldResult(resultShapes[0]),
                                 reshapeOp.getResultType().getElementType());
    if (emptyTensor.getType() != reshapeOp.getResultType()) {
      rewriter.replaceOpWithNewOp<tensor::CastOp>(
          reshapeOp, reshapeOp.getResultType(), emptyTensor);
    } else {
      rewriter.replaceOp(reshapeOp, emptyTensor);
    }
    return success();
  }
};

/// `tensor.empty` does not define any tensor contents, so a slice of a
/// `tensor.empty` can be canonicalized to a smaller `tensor.empty`.
struct FoldEmptyTensorWithExtractSliceOp
    : public OpRewritePattern<ExtractSliceOp> {
  using OpRewritePattern<ExtractSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ExtractSliceOp sliceOp,
                                PatternRewriter &rewriter) const override {
    if (!sliceOp.getSource().getDefiningOp<EmptyOp>())
      return failure();

    // ExtractSliceOp may be rank-reducing; its dynamic sizes must be
    // preserved as well as its result type.
    auto tensorType = RankedTensorType::get(sliceOp.getType().getShape(),
                                            sliceOp.getType().getElementType(),
                                            sliceOp.getType().getEncoding());
    rewriter.replaceOpWithNewOp<EmptyOp>(sliceOp, tensorType,
                                         sliceOp.getSizes());
    return success();
  }
};

} // namespace

void mlir::tensor::populateFoldTensorEmptyPatterns(
    RewritePatternSet &patterns) {
  patterns.add<FoldEmptyTensorWithExtractSliceOp,
               FoldEmptyTensorWithReshapeOp<tensor::ExpandShapeOp>,
               FoldEmptyTensorWithReshapeOp<tensor::CollapseShapeOp>>(
      patterns.getContext());
}
