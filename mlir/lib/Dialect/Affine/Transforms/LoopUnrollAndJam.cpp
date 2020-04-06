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
#include "mlir/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/LoopUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/CommandLine.h"

using namespace mlir;

#define DEBUG_TYPE "affine-loop-unroll-jam"

static llvm::cl::OptionCategory clOptionsCategory(DEBUG_TYPE " options");

// Loop unroll and jam factor.
static llvm::cl::opt<unsigned>
    clUnrollJamFactor("unroll-jam-factor", llvm::cl::Hidden,
                      llvm::cl::desc("Use this unroll jam factor for all loops"
                                     " (default 4)"),
                      llvm::cl::cat(clOptionsCategory));

namespace {
/// Loop unroll jam pass. Currently, this just unroll jams the first
/// outer loop in a Function.
struct LoopUnrollAndJam : public FunctionPass<LoopUnrollAndJam> {
/// Include the generated pass utilities.
#define GEN_PASS_AffineLoopUnrollAndJam
#include "mlir/Dialect/Affine/Passes.h.inc"

  Optional<unsigned> unrollJamFactor;
  static const unsigned kDefaultUnrollJamFactor = 4;

  explicit LoopUnrollAndJam(Optional<unsigned> unrollJamFactor = None)
      : unrollJamFactor(unrollJamFactor) {}

  void runOnFunction() override;
  LogicalResult runOnAffineForOp(AffineForOp forOp);
};
} // end anonymous namespace

std::unique_ptr<OpPassBase<FuncOp>>
mlir::createLoopUnrollAndJamPass(int unrollJamFactor) {
  return std::make_unique<LoopUnrollAndJam>(
      unrollJamFactor == -1 ? None : Optional<unsigned>(unrollJamFactor));
}

void LoopUnrollAndJam::runOnFunction() {
  // Currently, just the outermost loop from the first loop nest is
  // unroll-and-jammed by this pass. However, runOnAffineForOp can be called on
  // any for operation.
  auto &entryBlock = getFunction().front();
  if (auto forOp = dyn_cast<AffineForOp>(entryBlock.front()))
    runOnAffineForOp(forOp);
}

/// Unroll and jam a 'affine.for' op. Default unroll jam factor is
/// kDefaultUnrollJamFactor. Return failure if nothing was done.
LogicalResult LoopUnrollAndJam::runOnAffineForOp(AffineForOp forOp) {
  // Unroll and jam by the factor that was passed if any.
  if (unrollJamFactor.hasValue())
    return loopUnrollJamByFactor(forOp, unrollJamFactor.getValue());
  // Otherwise, unroll jam by the command-line factor if one was specified.
  if (clUnrollJamFactor.getNumOccurrences() > 0)
    return loopUnrollJamByFactor(forOp, clUnrollJamFactor);

  // Unroll and jam by four otherwise.
  return loopUnrollJamByFactor(forOp, kDefaultUnrollJamFactor);
}
