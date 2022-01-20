//===- ResolveShapedTypeResultDims.cpp - Resolve dim ops of result values -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass resolves `memref.dim` operations of result values in terms of
// shapes of their operands using the `InferShapedTypeOpInterface`.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {
/// Fold dim of an operation that implements the InferShapedTypeOpInterface
template <typename OpTy>
struct DimOfShapedTypeOpInterface : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy dimOp,
                                PatternRewriter &rewriter) const override {
    OpResult dimValue = dimOp.source().template dyn_cast<OpResult>();
    if (!dimValue)
      return failure();
    auto shapedTypeOp =
        dyn_cast<InferShapedTypeOpInterface>(dimValue.getOwner());
    if (!shapedTypeOp)
      return failure();

    Optional<int64_t> dimIndex = dimOp.getConstantIndex();
    if (!dimIndex)
      return failure();

    SmallVector<Value> reifiedResultShapes;
    if (failed(shapedTypeOp.reifyReturnTypeShapes(
            rewriter, shapedTypeOp->getOperands(), reifiedResultShapes)))
      return failure();

    if (reifiedResultShapes.size() != shapedTypeOp->getNumResults())
      return failure();

    Value resultShape = reifiedResultShapes[dimValue.getResultNumber()];
    auto resultShapeType = resultShape.getType().dyn_cast<RankedTensorType>();
    if (!resultShapeType || !resultShapeType.getElementType().isa<IndexType>())
      return failure();

    Location loc = dimOp->getLoc();
    rewriter.replaceOpWithNewOp<tensor::ExtractOp>(
        dimOp, resultShape,
        rewriter.createOrFold<arith::ConstantIndexOp>(loc, *dimIndex));
    return success();
  }
};

/// Fold dim of an operation that implements the InferShapedTypeOpInterface
template <typename OpTy>
struct DimOfReifyRankedShapedTypeOpInterface : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy dimOp,
                                PatternRewriter &rewriter) const override {
    OpResult dimValue = dimOp.source().template dyn_cast<OpResult>();
    if (!dimValue)
      return failure();
    auto rankedShapeTypeOp =
        dyn_cast<ReifyRankedShapedTypeOpInterface>(dimValue.getOwner());
    if (!rankedShapeTypeOp)
      return failure();

    Optional<int64_t> dimIndex = dimOp.getConstantIndex();
    if (!dimIndex)
      return failure();

    SmallVector<SmallVector<Value>> reifiedResultShapes;
    if (failed(
            rankedShapeTypeOp.reifyResultShapes(rewriter, reifiedResultShapes)))
      return failure();

    if (reifiedResultShapes.size() != rankedShapeTypeOp->getNumResults())
      return failure();

    unsigned resultNumber = dimValue.getResultNumber();
    auto sourceType = dimValue.getType().dyn_cast<RankedTensorType>();
    if (reifiedResultShapes[resultNumber].size() !=
        static_cast<size_t>(sourceType.getRank()))
      return failure();

    rewriter.replaceOp(dimOp, reifiedResultShapes[resultNumber][*dimIndex]);
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Pass registration
//===----------------------------------------------------------------------===//

namespace {
struct ResolveRankedShapeTypeResultDimsPass final
    : public ResolveRankedShapeTypeResultDimsBase<
          ResolveRankedShapeTypeResultDimsPass> {
  void runOnOperation() override;
};

struct ResolveShapedTypeResultDimsPass final
    : public ResolveShapedTypeResultDimsBase<ResolveShapedTypeResultDimsPass> {
  void runOnOperation() override;
};

} // namespace

void memref::populateResolveRankedShapeTypeResultDimsPatterns(
    RewritePatternSet &patterns) {
  patterns.add<DimOfReifyRankedShapedTypeOpInterface<memref::DimOp>,
               DimOfReifyRankedShapedTypeOpInterface<tensor::DimOp>>(
      patterns.getContext());
}

void memref::populateResolveShapedTypeResultDimsPatterns(
    RewritePatternSet &patterns) {
  // TODO: Move tensor::DimOp pattern to the Tensor dialect.
  patterns.add<DimOfShapedTypeOpInterface<memref::DimOp>,
               DimOfShapedTypeOpInterface<tensor::DimOp>>(
      patterns.getContext());
}

void ResolveRankedShapeTypeResultDimsPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  memref::populateResolveRankedShapeTypeResultDimsPatterns(patterns);
  if (failed(applyPatternsAndFoldGreedily(getOperation()->getRegions(),
                                          std::move(patterns))))
    return signalPassFailure();
}

void ResolveShapedTypeResultDimsPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  memref::populateResolveRankedShapeTypeResultDimsPatterns(patterns);
  memref::populateResolveShapedTypeResultDimsPatterns(patterns);
  if (failed(applyPatternsAndFoldGreedily(getOperation()->getRegions(),
                                          std::move(patterns))))
    return signalPassFailure();
}

std::unique_ptr<Pass> memref::createResolveShapedTypeResultDimsPass() {
  return std::make_unique<ResolveShapedTypeResultDimsPass>();
}

std::unique_ptr<Pass> memref::createResolveRankedShapeTypeResultDimsPass() {
  return std::make_unique<ResolveRankedShapeTypeResultDimsPass>();
}
