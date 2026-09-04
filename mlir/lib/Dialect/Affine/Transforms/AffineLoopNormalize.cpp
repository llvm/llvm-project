//===- AffineLoopNormalize.cpp - AffineLoopNormalize Pass -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a normalizer for affine loop-like ops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace affine {
#define GEN_PASS_DEF_AFFINELOOPNORMALIZE
#include "mlir/Dialect/Affine/Transforms/Passes.h.inc"
} // namespace affine
} // namespace mlir

using namespace mlir;
using namespace mlir::affine;

namespace {

/// Computes the constant upper or lower bound for a given affine map expression
/// and its operands, constrained by the specified type.
static FailureOr<int64_t> computeConstantBound(AffineMap map,
                                               ValueRange operands,
                                               presburger::BoundType type) {
  ValueBoundsConstraintSet::Variable var(map, operands);
  return ValueBoundsConstraintSet::computeConstantBound(
      type, var, nullptr, {/*closedUb=*/true, /*allowIntegerType=*/true});
}

/// Attempts to infer a static constant upper bound for the given normalized
/// `affine.for` loop using Value Bounds Analysis. If the dynamic upper bound's
/// range [upperMin, upperMax] is proven to be a single constant value (upperMin
/// == upperMax), the upper bound is directly replaced with this constant.
/// Otherwise, if upperMin > 0, the loop is split (peeled) into a static main
/// loop with a constant upper bound (`upperMin`) and a residual tail loop
/// iterating from `upperMin` to the original dynamic bound.
static LogicalResult
inferAffineLoopUpperConstantBound(RewriterBase &b, AffineForOp forOp,
                                  bool promoteSingleIter = true) {
  // The loop is normalized so we can expect its lower bound to be 0 and step to
  // be 1
  if (!forOp.hasConstantLowerBound() || forOp.getConstantLowerBound() != 0)
    return failure();
  if (forOp.getStepAsInt() != 1)
    return failure();
  if (forOp.getUpperBoundMap().getNumResults() > 1)
    return failure();

  // Infer the range [upperMin, upperMax] for the upper bound. We require a
  // strictly positive minimum bound (upperMin > 0) to guarantee a safe,
  // non-empty static trip count for the main loop.
  FailureOr<int64_t> upperMin = computeConstantBound(
      forOp.getUpperBoundMap(), forOp.getUpperBoundOperands(),
      presburger::BoundType::LB);
  FailureOr<int64_t> upperMax = computeConstantBound(
      forOp.getUpperBoundMap(), forOp.getUpperBoundOperands(),
      presburger::BoundType::UB);
  if (failed(upperMin) || *upperMin <= 0)
    return failure();

  // The upper bound is dynamic within [upperMin, upperMax]. Split the loop into
  // a static main loop (0 to upperMin) and a residual tail loop (upperMin to
  // dynamic bound).
  if (failed(upperMax) || *upperMax > *upperMin) {
    b.setInsertionPoint(forOp);
    AffineForOp clonedForOp = cast<AffineForOp>(b.clone(*forOp));
    clonedForOp.setConstantUpperBound(*upperMin);
    forOp.setConstantLowerBound(*upperMin);
    forOp.getInitsMutable().assign(clonedForOp->getResults());
    if (promoteSingleIter)
      (void)promoteIfSingleIteration(clonedForOp);

    return success();
  }

  // If upperMin == upperMax. The upper bound is proven to be a strict constant
  // at compile time. Directly constantize the bound without peeling a tail
  // loop.
  forOp.setConstantUpperBound(*upperMin);
  if (promoteSingleIter)
    (void)promoteIfSingleIteration(forOp);
  return success();
}

/// Normalize affine.parallel ops so that lower bounds are 0 and steps are 1.
/// As currently implemented, this pass cannot fail, but it might skip over ops
/// that are already in a normalized form.
struct AffineLoopNormalizePass
    : public affine::impl::AffineLoopNormalizeBase<AffineLoopNormalizePass> {
  explicit AffineLoopNormalizePass(bool promoteSingleIter,
                                   bool useExpensiveMath) {
    this->promoteSingleIter = promoteSingleIter;
    this->useExpensiveMath = useExpensiveMath;
  }

  void runOnOperation() override {
    getOperation().walk([&](Operation *op) {
      if (auto affineParallel = dyn_cast<AffineParallelOp>(op))
        normalizeAffineParallel(affineParallel);
      else if (auto affineFor = dyn_cast<AffineForOp>(op))
        (void)normalizeAffineFor(affineFor, promoteSingleIter);
    });

    // Infer and rewrite the upper bound into a compile-time constant for each
    // loop.
    if (useExpensiveMath) {
      IRRewriter b(&getContext());
      SmallVector<AffineForOp> loops;

      // Collect target loops because `inferAffineLoopUpperConstantBound` may
      // create new loops during processing.
      // TODO: When running `normalizeAffineFor` with `promoteSingleIter=true`,
      // there is currently no clean way to know if the loop was promoted. We
      // can improve this in the future to avoid calling `walk` to pre-collect
      // loops.
      getOperation()->walk([&](AffineForOp forOp) { loops.push_back(forOp); });
      for (AffineForOp forOp : loops)
        (void)inferAffineLoopUpperConstantBound(b, forOp, promoteSingleIter);
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::affine::createAffineLoopNormalizePass(bool promoteSingleIter,
                                            bool useExpensiveMath) {
  return std::make_unique<AffineLoopNormalizePass>(promoteSingleIter,
                                                   useExpensiveMath);
}
