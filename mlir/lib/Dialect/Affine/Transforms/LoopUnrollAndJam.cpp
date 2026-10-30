//===- LoopUnrollAndJam.cpp - Code to perform loop unroll and jam ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements loop unroll and jam. Unroll and jam is a transformation
// that improves locality, in particular, register reuse, while also improving
// operation level parallelism. The example below shows what it does in nearly
// the general case. Loop unroll and jam currently works if the bounds of the
// loops inner to the loop being unroll-jammed do not depend on the latter.
//
// Before      After unroll and jam of i by factor 2:
//
//             for i, step = 2
// for i         S1(i);
//   S1;         S2(i);
//   S2;         S1(i+1);
//   for j       S2(i+1);
//     S3;       for j
//     S4;         S3(i, j);
//   S5;           S4(i, j);
//   S6;           S3(i+1, j)
//                 S4(i+1, j)
//               S5(i);
//               S6(i);
//               S5(i+1);
//               S6(i+1);
//
// Note: 'if/else' blocks are not jammed. So, if there are loops inside if
// op's, bodies of those loops will not be jammed.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/Transforms/Passes.h"

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CommandLine.h"
#include <optional>

namespace mlir {
namespace affine {
#define GEN_PASS_DEF_AFFINELOOPUNROLLANDJAM
#include "mlir/Dialect/Affine/Transforms/Passes.h.inc"
} // namespace affine
} // namespace mlir

#define DEBUG_TYPE "affine-loop-unroll-jam"

using namespace mlir;
using namespace mlir::affine;

/// Iteration independence is a sufficient condition for unroll-and-jam.
/// Conservatively reject operations and aliases that the dependence analysis
/// cannot model. The utility still checks the factor and loop structure.
static bool isUnrollJamProvenSafe(AffineForOp loop,
                                  AliasAnalysis &aliasAnalysis) {
  SmallVector<Operation *> accesses;
  auto walkResult = loop.walk([&](Operation *op) {
    if (isa<AffineForOp, AffineYieldOp>(op))
      return WalkResult::advance();
    if (isa<AffineLoadOp, AffineStoreOp>(op)) {
      // Access relations require a memref base fixed across iterations.
      if (!loop.isDefinedOutsideOfLoop(MemRefAccess(op).memref))
        return WalkResult::interrupt();
      accesses.push_back(op);
      return WalkResult::advance();
    }
    // Memory-effect freedom alone does not rule out nontermination or other
    // implicit behavior. Initially accept only pure, region-free operations.
    if (op->getNumRegions() == 0 && isPure(op))
      return WalkResult::advance();
    return WalkResult::interrupt();
  });
  if (walkResult.wasInterrupted())
    return false;

  // Affine dependence analysis assumes different memref SSA values do not
  // alias. Establish that precondition for each pair involving a write.
  for (Operation *src : accesses) {
    for (Operation *dst : accesses) {
      if (!isa<AffineStoreOp>(src) && !isa<AffineStoreOp>(dst))
        continue;
      Value srcMemref = MemRefAccess(src).memref;
      Value dstMemref = MemRefAccess(dst).memref;
      if (srcMemref != dstMemref &&
          !aliasAnalysis.alias(srcMemref, dstMemref).isNo())
        return false;
    }
  }

  // Reject outer SSA recurrences and memory dependences. Inner-loop iter_args
  // retain their per-iteration order when the containing loop is jammed.
  return isLoopParallel(loop);
}

namespace {
/// Unroll and jam the affine loops in the function entry block.
struct LoopUnrollAndJam
    : public affine::impl::AffineLoopUnrollAndJamBase<LoopUnrollAndJam> {
  explicit LoopUnrollAndJam(
      std::optional<unsigned> unrollJamFactor = std::nullopt) {
    if (unrollJamFactor)
      this->unrollJamFactor = *unrollJamFactor;
  }

  void runOnOperation() override;
};
} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>>
mlir::affine::createLoopUnrollAndJamPass(int unrollJamFactor) {
  return std::make_unique<LoopUnrollAndJam>(
      unrollJamFactor == -1 ? std::nullopt
                            : std::optional<unsigned>(unrollJamFactor));
}

void LoopUnrollAndJam::runOnOperation() {
  if (getOperation().isExternal())
    return;

  // Keep the legacy target and prove legality for additional loops before
  // rewriting. The utility can erase selected loops or create cleanup loops,
  // which must not be revisited.
  auto &entryBlock = getOperation().front();
  SmallVector<AffineForOp> loops;
  if (auto forOp = dyn_cast<AffineForOp>(entryBlock.front()))
    loops.push_back(forOp);

  if (unrollJamAllLoops) {
    for (Operation &op : llvm::drop_begin(entryBlock)) {
      if (auto forOp = dyn_cast<AffineForOp>(op))
        if (isUnrollJamProvenSafe(forOp, getAnalysis<AliasAnalysis>()))
          loops.push_back(forOp);
    }
  }
  for (auto forOp : loops)
    (void)loopUnrollJamByFactor(forOp, unrollJamFactor);
}
