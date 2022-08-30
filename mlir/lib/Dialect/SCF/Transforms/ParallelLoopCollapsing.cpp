//===- ParallelLoopCollapsing.cpp - Pass collapsing parallel loop indices -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SCF/Transforms/Passes.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

namespace mlir {
#define GEN_PASS_DEF_SCFPARALLELLOOPCOLLAPSINGPASS
#include "mlir/Dialect/SCF/Transforms/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "parallel-loop-collapsing"

using namespace mlir;

namespace {
struct SCFParallelLoopCollapsingPass
    : public impl::SCFParallelLoopCollapsingPassBase<
          SCFParallelLoopCollapsingPass> {
  using SCFParallelLoopCollapsingPassBase::SCFParallelLoopCollapsingPassBase;

  void runOnOperation() override {
    Operation *module = getOperation();

    module->walk([&](scf::ParallelOp op) {
      // The common case for GPU dialect will be simplifying the ParallelOp to 3
      // arguments, so we do that here to simplify things.
      llvm::SmallVector<std::vector<unsigned>, 3> combinedLoops;
      if (!clCollapsedIndices0.empty())
        combinedLoops.push_back(clCollapsedIndices0);
      if (!clCollapsedIndices1.empty())
        combinedLoops.push_back(clCollapsedIndices1);
      if (!clCollapsedIndices2.empty())
        combinedLoops.push_back(clCollapsedIndices2);
      collapseParallelLoops(op, combinedLoops);
    });
  }
};
} // namespace
