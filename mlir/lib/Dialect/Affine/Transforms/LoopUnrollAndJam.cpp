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

#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
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

  // Collect entry-block loops before rewriting: the utility can erase a
  // selected loop or create cleanup loops, which must not be revisited.
  SmallVector<AffineForOp> loops;
  for (Operation &op : getOperation().front()) {
    if (auto forOp = dyn_cast<AffineForOp>(op))
      loops.push_back(forOp);
    if (!unrollJamAllLoops)
      break;
  }
  for (auto forOp : loops)
    (void)loopUnrollJamByFactor(forOp, unrollJamFactor);
}
