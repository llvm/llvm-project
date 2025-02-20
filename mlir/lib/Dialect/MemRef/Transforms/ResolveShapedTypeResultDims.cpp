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

#include "mlir/Dialect/MemRef/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace memref {
#define GEN_PASS_DEF_RESOLVERANKEDSHAPETYPERESULTDIMS
#define GEN_PASS_DEF_RESOLVESHAPEDTYPERESULTDIMS
#include "mlir/Dialect/MemRef/Transforms/Passes.h.inc"
} // namespace memref
} // namespace mlir

using namespace mlir;

namespace {
/// Fold dim of an operation that implements the InferShapedTypeOpInterface
template <typename OpTy>
struct DimOfShapedTypeOpInterface : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy dimOp,
                                PatternRewriter &rewriter) const override {
    OpResult dimValue = dyn_cast<OpResult>(dimOp.getSource());
    if (!dimValue)
      return failure();
    auto shapedTypeOp =
        dyn_cast<InferShapedTypeOpInterface>(dimValue.getOwner());
    if (!shapedTypeOp)
      return failure();

    std::optional<int64_t> dimIndex = dimOp.getConstantIndex();
    if (!dimIndex)
      return failure();

    SmallVector<Value> reifiedResultShapes;
    if (failed(shapedTypeOp.reifyReturnTypeShapes(
            rewriter, shapedTypeOp->getOperands(), reifiedResultShapes)))
      return failure();

    if (reifiedResultShapes.size() != shapedTypeOp->getNumResults())
      return failure();

    Value resultShape = reifiedResultShapes[dimValue.getResultNumber()];
    auto resultShapeType = dyn_cast<RankedTensorType>(resultShape.getType());
    if (!resultShapeType || !isa<IndexType>(resultShapeType.getElementType()))
      return failure();

    Location loc = dimOp->getLoc();
    rewriter.replaceOpWithNewOp<tensor::ExtractOp>(
        dimOp, resultShape,
        rewriter.create<arith::ConstantIndexOp>(loc, *dimIndex).getResult());
    return success();
  }
};

/// Fold dim of an operation that implements the InferShapedTypeOpInterface
template <typename OpTy>
struct DimOfReifyRankedShapedTypeOpInterface : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  void initialize() { OpRewritePattern<OpTy>::setHasBoundedRewriteRecursion(); }

  LogicalResult matchAndRewrite(OpTy dimOp,
                                PatternRewriter &rewriter) const override {
    OpResult dimValue = dyn_cast<OpResult>(dimOp.getSource());
    if (!dimValue)
      return failure();
    std::optional<int64_t> dimIndex = dimOp.getConstantIndex();
    if (!dimIndex)
      return failure();

    ReifiedRankedShapedTypeDims reifiedResultShapes;
    if (failed(reifyResultShapes(rewriter, dimValue.getOwner(),
                                 reifiedResultShapes)))
      return failure();
    unsigned resultNumber = dimValue.getResultNumber();
    // Do not apply pattern if the IR is invalid (dim out of bounds).
    if ((size_t)(*dimIndex) >= reifiedResultShapes[resultNumber].size())
      return rewriter.notifyMatchFailure(dimOp, "dimension is out of bounds");
    Value replacement = getValueOrCreateConstantIndexOp(
        rewriter, dimOp.getLoc(), reifiedResultShapes[resultNumber][*dimIndex]);
    rewriter.replaceOp(dimOp, replacement);
    return success();
  }
};

/// Fold dim ops of iter_args to dim ops of their respective init args. E.g.:
///
/// ```
/// %0 = ... : tensor<?x?xf32>
/// scf.forall ... shared_outs(%arg0 = %0) -> (tensor<?x?xf32>) {
///   %1 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
///   ...
/// }
/// ```
///
/// is folded to:
///
/// ```
/// %0 = ... : tensor<?x?xf32>
/// scf.forall ... shared_outs(%arg0 = %0) -> (tensor<?x?xf32>) {
///   %1 = tensor.dim %0, %c0 : tensor<?x?xf32>
///   ...
/// }
/// ```
struct IterArgsToInitArgs : public OpRewritePattern<tensor::DimOp> {
  using OpRewritePattern<tensor::DimOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::DimOp dimOp,
                                PatternRewriter &rewriter) const final {
    auto blockArg = dyn_cast<BlockArgument>(dimOp.getSource());
    if (!blockArg)
      return failure();
    // TODO: Enable this for loopLikeInterface. Restricting for scf.for
    // because the init args shape might change in the loop body.
    // For e.g.:
    // ```
    //  %0 = tensor.empty(%c1) : tensor<?xf32>
    //  %r = scf.for %iv = %c0 to %c10 step %c1 iter_args(%arg0 = %0) ->
    //  tensor<?xf32> {
    //    %1 = tensor.dim %arg0, %c0 : tensor<?xf32>
    //    %2 = arith.addi %c1, %1 : index
    //    %3 = tensor.empty(%2) : tensor<?xf32>
    //    scf.yield %3 : tensor<?xf32>
    //  }
    //
    // ```
    auto forAllOp =
        dyn_cast<scf::ForallOp>(blockArg.getParentBlock()->getParentOp());
    if (!forAllOp)
      return failure();
    Value initArg = forAllOp.getTiedLoopInit(blockArg)->get();
    rewriter.modifyOpInPlace(
        dimOp, [&]() { dimOp.getSourceMutable().assign(initArg); });
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Pass registration
//===----------------------------------------------------------------------===//

namespace {
struct ResolveRankedShapeTypeResultDimsPass final
    : public memref::impl::ResolveRankedShapeTypeResultDimsBase<
          ResolveRankedShapeTypeResultDimsPass> {
  void runOnOperation() override;
};

struct ResolveShapedTypeResultDimsPass final
    : public memref::impl::ResolveShapedTypeResultDimsBase<
          ResolveShapedTypeResultDimsPass> {
  void runOnOperation() override;
};

} // namespace

void memref::populateResolveRankedShapedTypeResultDimsPatterns(
    RewritePatternSet &patterns) {
  patterns.add<DimOfReifyRankedShapedTypeOpInterface<memref::DimOp>,
               DimOfReifyRankedShapedTypeOpInterface<tensor::DimOp>,
               IterArgsToInitArgs>(patterns.getContext());
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
  memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
    return signalPassFailure();
}

void ResolveShapedTypeResultDimsPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
  memref::populateResolveShapedTypeResultDimsPatterns(patterns);
  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
    return signalPassFailure();
}

std::unique_ptr<Pass> memref::createResolveShapedTypeResultDimsPass() {
  return std::make_unique<ResolveShapedTypeResultDimsPass>();
}

std::unique_ptr<Pass> memref::createResolveRankedShapeTypeResultDimsPass() {
  return std::make_unique<ResolveRankedShapeTypeResultDimsPass>();
}
