//===- TrivialDeadCodeElimination.cpp - Trivial DCE -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"

namespace mlir {
#define GEN_PASS_DEF_TRIVIALDEADCODEELIMINATIONPASS
#include "mlir/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {
struct TrivialDeadCodeElimination
    : public impl::TrivialDeadCodeEliminationPassBase<
          TrivialDeadCodeElimination> {
  using impl::TrivialDeadCodeEliminationPassBase<
      TrivialDeadCodeElimination>::TrivialDeadCodeEliminationPassBase;

  void runOnOperation() override {
    Operation *target = getOperation();
    IRRewriter rewriter(target->getContext());
    if (removeBlocks)
      (void)eraseUnreachableBlocks(rewriter, target->getRegions(), recursive);
    for (Region &region : target->getRegions())
      eliminateTriviallyDeadOps(rewriter, region, recursive);
  }
};
} // namespace
