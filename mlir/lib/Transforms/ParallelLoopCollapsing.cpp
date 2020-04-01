//===- ParallelLoopCollapsing.cpp - Pass collapsing parallel loop indices -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LoopOps/LoopOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/LoopUtils.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

#define PASS_NAME "parallel-loop-collapsing"
#define DEBUG_TYPE PASS_NAME

using namespace mlir;

namespace {
struct ParallelLoopCollapsing : public OperationPass<ParallelLoopCollapsing> {
  ParallelLoopCollapsing() = default;
  ParallelLoopCollapsing(const ParallelLoopCollapsing &) {}
  void runOnOperation() override {
    Operation *module = getOperation();

    module->walk([&](loop::ParallelOp op) {
      // The common case for GPU dialect will be simplifying the ParallelOp to 3
      // arguments, so we do that here to simplify things.
      llvm::SmallVector<std::vector<unsigned>, 3> combinedLoops;
      if (clCollapsedIndices0.size())
        combinedLoops.push_back(clCollapsedIndices0);
      if (clCollapsedIndices1.size())
        combinedLoops.push_back(clCollapsedIndices1);
      if (clCollapsedIndices2.size())
        combinedLoops.push_back(clCollapsedIndices2);
      collapseParallelLoops(op, combinedLoops);
    });
  }

  ListOption<unsigned> clCollapsedIndices0{
      *this, "collapsed-indices-0",
      llvm::cl::desc("Which loop indices to combine 0th loop index"),
      llvm::cl::MiscFlags::CommaSeparated};

  ListOption<unsigned> clCollapsedIndices1{
      *this, "collapsed-indices-1",
      llvm::cl::desc(
          "Which loop indices to combine into the position 1 loop index"),
      llvm::cl::MiscFlags::CommaSeparated};

  ListOption<unsigned> clCollapsedIndices2{
      *this, "collapsed-indices-2",
      llvm::cl::desc(
          "Which loop indices to combine into the position 2 loop index"),
      llvm::cl::MiscFlags::CommaSeparated};
};

} // namespace

std::unique_ptr<Pass> mlir::createParallelLoopCollapsingPass() {
  return std::make_unique<ParallelLoopCollapsing>();
}

static PassRegistration<ParallelLoopCollapsing>
    reg(PASS_NAME, "collapse parallel loops to use less induction variables.");
