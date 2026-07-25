//===- AffineLoopConstantizeBounds.cpp - Constantize loop bounds pass ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to transform affine loops with symbolic bounds
// into a static-trip-count main loop and a residual tail loop using range
// analysis.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Transforms/Passes.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include <cstdint>

namespace mlir {
namespace affine {
#define GEN_PASS_DEF_AFFINELOOPCONSTANTIZEBOUNDS
#include "mlir/Dialect/Affine/Transforms/Passes.h.inc"
} // namespace affine
} // namespace mlir

#define DEBUG_TYPE "affine-loop-constantize-bounds"

using namespace mlir;
using namespace mlir::affine;

namespace {

/// Computes the constant upper or lower bound for a given affine map expression
/// \p map and its operands \p operands, constrained by the specified \p type.
static FailureOr<int64_t> computeConstantBound(AffineMap map,
                                               ValueRange operands,
                                               presburger::BoundType type) {
  ValueBoundsConstraintSet::Variable var(map, operands);
  return ValueBoundsConstraintSet::computeConstantBound(type, var, nullptr,
                                                        {true, true});
}

static LogicalResult inferAffineLoopUpperConstantBound(AffineForOp forOp) {
  // Ensure the loop is normalized (lower bound is strictly 0 and step is 1).
  if (!forOp.hasConstantLowerBound() || forOp.getConstantLowerBound() != 0)
    return failure();
  if (forOp.getStepAsInt() != 1)
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

  IRRewriter b(forOp->getContext());

  // The upper bound is dynamic within [upperMin, upperMax]. Split the loop into
  // a static main loop (0 to upperMin) and a residual tail loop (upperMin to
  // dynamic bound).
  if (failed(upperMax) || *upperMax > *upperMin) {
    b.setInsertionPoint(forOp);
    AffineForOp clonedForOp = cast<AffineForOp>(b.clone(*forOp));
    clonedForOp.setConstantUpperBound(*upperMin);
    forOp.setConstantLowerBound(*upperMin);
    forOp.getInitsMutable().assign(clonedForOp->getResults());
    return success();
  }

  // If upperMin == upperMax. The upper bound is proven to be a strict constant
  // at compile time. Directly constantize the bound without peeling a tail
  // loop.
  forOp.setConstantUpperBound(*upperMin);
  return success();
}

struct AffineLoopConstantizeBounds
    : public affine::impl::AffineLoopConstantizeBoundsBase<
          AffineLoopConstantizeBounds> {
  void runOnOperation() override;
};
} // namespace

void AffineLoopConstantizeBounds::runOnOperation() {
  SmallVector<AffineForOp, 4> loops;
  getOperation()->walk([&](AffineForOp forOp) {
    // First collect eligible loops and normalize them so that their lower bound
    // is locked to 0 and step to 1.
    if (succeeded(normalizeAffineFor(forOp, false)))
      loops.push_back(forOp);
  });
  // Infer and rewrite the upper bound into a compile-time constant for each
  // loop.
  for (AffineForOp loop : loops)
    (void)inferAffineLoopUpperConstantBound(loop);
}
