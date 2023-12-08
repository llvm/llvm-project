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
  FoldEmptyTensorWithReshapeOp(MLIRContext *ctx, PatternBenefit benefit = 1,
                               bool foldSingleUseOnly = false)
      : OpRewritePattern<ReshapeOp>(ctx, benefit),
        foldSingleUseOnly(foldSingleUseOnly) {}

  LogicalResult matchAndRewrite(ReshapeOp reshapeOp,
                                PatternRewriter &rewriter) const override {
    // Check for tensor.empty source.
    auto emptyOp = reshapeOp.getSrc().template getDefiningOp<EmptyOp>();
    if (!emptyOp)
      return failure();

    // Check for single use.
    if (foldSingleUseOnly && !llvm::hasSingleElement(emptyOp->getUses()))
      return failure();

    // Reify result shape.
    Location loc = reshapeOp.getLoc();
    ReifiedRankedShapedTypeDims resultShapes;
    if (failed(reifyResultShapes(rewriter, reshapeOp, resultShapes)) ||
        !llvm::hasSingleElement(resultShapes))
      return failure();

    // Create new tensor.empty op.
    // TODO: Do not drop tensor type encoding.
    Value emptyTensor = rewriter.create<EmptyOp>(
        loc, resultShapes[0], reshapeOp.getResultType().getElementType());
    if (emptyTensor.getType() != reshapeOp.getResultType()) {
      rewriter.replaceOpWithNewOp<tensor::CastOp>(
          reshapeOp, reshapeOp.getResultType(), emptyTensor);
    } else {
      rewriter.replaceOp(reshapeOp, emptyTensor);
    }
    return success();
  }

private:
  bool foldSingleUseOnly = false;
};

/// tensor.empty does not define any tensor contents, so a slice of a
/// tensor.empty can be folded to a smaller tensor.empty.
struct FoldEmptyTensorWithExtractSliceOp
    : public OpRewritePattern<ExtractSliceOp> {
  FoldEmptyTensorWithExtractSliceOp(MLIRContext *ctx,
                                    PatternBenefit benefit = 1,
                                    bool foldSingleUseOnly = false)
      : OpRewritePattern<ExtractSliceOp>(ctx, benefit),
        foldSingleUseOnly(foldSingleUseOnly) {}

  LogicalResult matchAndRewrite(ExtractSliceOp sliceOp,
                                PatternRewriter &rewriter) const override {
    // Check for tensor.empty source.
    auto emptyOp = sliceOp.getSource().template getDefiningOp<EmptyOp>();
    if (!emptyOp)
      return failure();

    // Check for single use.
    if (foldSingleUseOnly && !llvm::hasSingleElement(emptyOp->getUses()))
      return failure();

    // Create new tensor.empty op. tensor.extract_slice may be rank-reducing;
    // its dynamic sizes must be preserved as well as its result type.
    auto tensorType = RankedTensorType::get(sliceOp.getType().getShape(),
                                            sliceOp.getType().getElementType(),
                                            sliceOp.getType().getEncoding());
    rewriter.replaceOpWithNewOp<EmptyOp>(sliceOp, tensorType,
                                         sliceOp.getSizes());
    return success();
  }

private:
  bool foldSingleUseOnly = false;
};

} // namespace

void mlir::tensor::populateFoldTensorEmptyPatterns(RewritePatternSet &patterns,
                                                   bool foldSingleUseOnly) {
  patterns.add<FoldEmptyTensorWithExtractSliceOp,
               FoldEmptyTensorWithReshapeOp<tensor::ExpandShapeOp>,
               FoldEmptyTensorWithReshapeOp<tensor::CollapseShapeOp>>(
      patterns.getContext(), /*benefit=*/1, foldSingleUseOnly);
}
