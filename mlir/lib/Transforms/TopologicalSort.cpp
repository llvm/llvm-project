//===- TopologicalSort.cpp - Topological sort pass ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/Passes.h"

#include "mlir/IR/RegionKindInterface.h"
#include "mlir/Transforms/TopologicalSortUtils.h"

namespace mlir {
#define GEN_PASS_DEF_TOPOLOGICALSORT
#include "mlir/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {
struct TopologicalSortPass
    : public impl::TopologicalSortBase<TopologicalSortPass> {
  void runOnOperation() override {
    // Topologically sort the regions of the operation without SSA dominance.
    getOperation()->walk([](RegionKindInterface op) {
      for (auto &it : llvm::enumerate(op->getRegions())) {
        if (op.hasSSADominance(it.index()))
          continue;
        for (Block &block : it.value())
          sortTopologically(&block);
      }
    });
  }
};
} // end anonymous namespace

std::unique_ptr<Pass> mlir::createTopologicalSortPass() {
  return std::make_unique<TopologicalSortPass>();
}
