//===- LoopUnroll.cpp - Code to perform loop unrolling --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements loop unrolling.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/Passes.h"

#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include <optional>

namespace mlir {
#define GEN_PASS_DEF_AFFINELOOPUNROLL
#include "mlir/Dialect/Affine/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "affine-loop-unroll"

using namespace mlir;

namespace {

// TODO: this is really a test pass and should be moved out of dialect
// transforms.

/// Loop unrolling pass. Unrolls all innermost loops unless full unrolling and a
/// full unroll threshold was specified, in which case, fully unrolls all loops
/// with trip count less than the specified threshold. The latter is for testing
/// purposes, especially for testing outer loop unrolling.
struct LoopUnroll : public impl::AffineLoopUnrollBase<LoopUnroll> {
  // Callback to obtain unroll factors; if this has a callable target, takes
  // precedence over command-line argument or passed argument.
  const std::function<unsigned(AffineForOp)> getUnrollFactor;

  LoopUnroll() : getUnrollFactor(nullptr) {}
  LoopUnroll(const LoopUnroll &other)

      = default;
  explicit LoopUnroll(
      std::optional<unsigned> unrollFactor = std::nullopt,
      bool unrollUpToFactor = false, bool unrollFull = false,
      const std::function<unsigned(AffineForOp)> &getUnrollFactor = nullptr)
      : getUnrollFactor(getUnrollFactor) {
    if (unrollFactor)
      this->unrollFactor = *unrollFactor;
    this->unrollUpToFactor = unrollUpToFactor;
    this->unrollFull = unrollFull;
  }

  void runOnOperation() override;

  /// Unroll this for op. Returns failure if nothing was done.
  LogicalResult runOnAffineForOp(AffineForOp forOp);
};
} // namespace

/// Returns true if no other affine.for ops are nested within `op`.
static bool isInnermostAffineForOp(AffineForOp op) {
  return !op.getBody()
              ->walk([&](AffineForOp nestedForOp) {
                return WalkResult::interrupt();
              })
              .wasInterrupted();
}

/// Gathers loops that have no affine.for's nested within.
static void gatherInnermostLoops(func::FuncOp f,
                                 SmallVectorImpl<AffineForOp> &loops) {
  f.walk([&](AffineForOp forOp) {
    if (isInnermostAffineForOp(forOp))
      loops.push_back(forOp);
  });
}

void LoopUnroll::runOnOperation() {
  func::FuncOp func = getOperation();
  if (func.isExternal())
    return;

  if (unrollFull && unrollFullThreshold.hasValue()) {
    // Store short loops as we walk.
    SmallVector<AffineForOp, 4> loops;

    // Gathers all loops with trip count <= minTripCount. Do a post order walk
    // so that loops are gathered from innermost to outermost (or else unrolling
    // an outer one may delete gathered inner ones).
    getOperation().walk([&](AffineForOp forOp) {
      std::optional<uint64_t> tripCount = getConstantTripCount(forOp);
      if (tripCount && *tripCount <= unrollFullThreshold)
        loops.push_back(forOp);
    });
    for (auto forOp : loops)
      (void)loopUnrollFull(forOp);
    return;
  }

  // If the call back is provided, we will recurse until no loops are found.
  SmallVector<AffineForOp, 4> loops;
  for (unsigned i = 0; i < numRepetitions || getUnrollFactor; i++) {
    loops.clear();
    gatherInnermostLoops(func, loops);
    if (loops.empty())
      break;
    bool unrolled = false;
    for (auto forOp : loops)
      unrolled |= succeeded(runOnAffineForOp(forOp));
    if (!unrolled)
      // Break out if nothing was unrolled.
      break;
  }
}

/// Unrolls a 'affine.for' op. Returns success if the loop was unrolled,
/// failure otherwise. The default unroll factor is 4.
LogicalResult LoopUnroll::runOnAffineForOp(AffineForOp forOp) {
  // Use the function callback if one was provided.
  if (getUnrollFactor)
    return loopUnrollByFactor(forOp, getUnrollFactor(forOp),
                              /*annotateFn=*/nullptr, cleanUpUnroll);
  // Unroll completely if full loop unroll was specified.
  if (unrollFull)
    return loopUnrollFull(forOp);
  // Otherwise, unroll by the given unroll factor.
  if (unrollUpToFactor)
    return loopUnrollUpToFactor(forOp, unrollFactor);
  return loopUnrollByFactor(forOp, unrollFactor, /*annotateFn=*/nullptr,
                            cleanUpUnroll);
}

std::unique_ptr<OperationPass<func::FuncOp>> mlir::createLoopUnrollPass(
    int unrollFactor, bool unrollUpToFactor, bool unrollFull,
    const std::function<unsigned(AffineForOp)> &getUnrollFactor) {
  return std::make_unique<LoopUnroll>(
      unrollFactor == -1 ? std::nullopt : std::optional<unsigned>(unrollFactor),
      unrollUpToFactor, unrollFull, getUnrollFactor);
}
