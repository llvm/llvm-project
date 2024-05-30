// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "mlir/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Utils/LoopUtils.h"
#include "mlir/Interfaces/LoopLikeInterface.h"

namespace mlir {
#define GEN_PASS_DEF_NORMALIZELOOPBOUNDS
#include "mlir/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

/// Normalize a loop-like operation with induction variables, i.e. calculate
/// new normalized upper bounds for lower bounds equal to zero and step sizes
/// equal to one. Then, insert new `affine.apply` operations to calculate the
/// denormalized index values and update all usage from the original induction
/// variables to the results of the `affine.apply` operations.
///
/// Example:
/// Transform a `scf.forall` loop with a strictly positive steps
///   forall (%i, %j) = (%lb0, %lb1) to (%ub0, %ub1) step (%s0, %s1)
/// into a 0-based loop with step 1
///   forall (%i, %j) in (ceildiv(%ub0 - %lb0, %s0), ceildiv(%ub1 - %lb1, %s1))
LogicalResult
normalizeLoopBounds(RewriterBase &rewriter,
                    LoopLikeWithInductionVarsOpInterface loopLikeOp) {
  OpBuilder::InsertionGuard g(rewriter);
  if (loopLikeOp.isNormalized())
    return success();

  SmallVector<Value> newLbs;
  SmallVector<Value> newUbs;
  SmallVector<Value> newSteps;
  rewriter.setInsertionPoint(loopLikeOp);
  for (auto &&[iv, lb, ub, step] : llvm::zip(
           loopLikeOp.getInductionVars(), loopLikeOp.getLowerBound(rewriter),
           loopLikeOp.getUpperBound(rewriter), loopLikeOp.getStep(rewriter))) {
    std::optional<int64_t> lbInt = getConstantIntValue(lb);
    std::optional<int64_t> stepInt = getConstantIntValue(step);

    rewriter.setInsertionPoint(loopLikeOp);
    auto newLoopParams =
        emitNormalizedLoopBounds(rewriter, loopLikeOp.getLoc(), lb, ub, step);

    newLbs.push_back(newLoopParams.lowerBound);
    newUbs.push_back(newLoopParams.upperBound);
    newSteps.push_back(newLoopParams.step);

    Region &region = loopLikeOp.getOperation()->getRegion(0);
    rewriter.setInsertionPointToStart(&region.front());
    SmallVector<Value> operands = {iv};
    AffineExpr idxExpr, stepExpr, offsetExpr, res;
    if (!lbInt && !stepInt) {
      bindDims(loopLikeOp.getContext(), idxExpr, stepExpr, offsetExpr);
      res = idxExpr * stepExpr + offsetExpr;
      operands.push_back(step);
      operands.push_back(lb);
    } else if (!lbInt) {
      bindDims(loopLikeOp.getContext(), idxExpr, offsetExpr);
      res = idxExpr * stepInt.value() + offsetExpr;
      operands.push_back(lb);
    } else if (!stepInt) {
      bindDims(loopLikeOp.getContext(), idxExpr, stepExpr);
      res = idxExpr * stepExpr + lbInt.value();
      operands.push_back(step);
    } else {
      bindDims(loopLikeOp.getContext(), idxExpr);
      res = idxExpr * stepInt.value() + lbInt.value();
    }

    auto affineApply = rewriter.create<affine::AffineApplyOp>(
        loopLikeOp.getLoc(), res, operands);
    SmallPtrSet<Operation *, 2> preserve(
        {iv.getDefiningOp(), affineApply.getOperation()});
    rewriter.replaceAllUsesExcept(iv, affineApply.getResult(), preserve);
  }

  rewriter.setInsertionPoint(loopLikeOp);
  rewriter.modifyOpInPlace(loopLikeOp, [&]() {
    loopLikeOp.setLowerBounds(newLbs);
    loopLikeOp.setUpperBounds(newUbs);
    loopLikeOp.setSteps(newSteps);
  });
  return success();
}

namespace {

/// Pass which normalizes the loop bounds of operations implementing
/// `LoopLikeWithInductionVarsOpInterface`.
struct NormalizeLoopBounds
    : public impl::NormalizeLoopBoundsBase<NormalizeLoopBounds> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect>();
  }

  void runOnOperation() override {
    Operation *parentOp = getOperation();
    IRRewriter rewriter(parentOp->getContext());

    parentOp->walk([&](LoopLikeWithInductionVarsOpInterface loopLikeOp) {
      (void)normalizeLoopBounds(rewriter, loopLikeOp);
    });
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createNormalizeLoopBoundsPass() {
  return std::make_unique<NormalizeLoopBounds>();
}
