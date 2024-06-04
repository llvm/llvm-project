//===- ParallelLoopFusion.cpp - Code to perform loop fusion ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements loop fusion on parallel loops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SCF/Transforms/Passes.h"

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
#define GEN_PASS_DEF_SCFPARALLELLOOPFUSION
#include "mlir/Dialect/SCF/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::scf;

void mlir::scf::naivelyFuseParallelOps(
    Region &region, llvm::function_ref<bool(Value, Value)> mayAlias) {
  OpBuilder b(region);
  // Consider every single block and attempt to fuse adjacent loops.
  SmallVector<SmallVector<ParallelOp>, 1> ploopChains;
  for (auto &block : region) {
    ploopChains.clear();
    ploopChains.push_back({});

    // Not using `walk()` to traverse only top-level parallel loops and also
    // make sure that there are no side-effecting ops between the parallel
    // loops.
    bool noSideEffects = true;
    for (auto &op : block) {
      if (auto ploop = dyn_cast<ParallelOp>(op)) {
        if (noSideEffects) {
          ploopChains.back().push_back(ploop);
        } else {
          ploopChains.push_back({ploop});
          noSideEffects = true;
        }
        continue;
      }
      // TODO: Handle region side effects properly.
      noSideEffects &= isMemoryEffectFree(&op) && op.getNumRegions() == 0;
    }
    for (MutableArrayRef<ParallelOp> ploops : ploopChains) {
      for (int i = 0, e = ploops.size(); i + 1 < e; ++i)
        mlir::fuseIfLegal(ploops[i], ploops[i + 1], b, mayAlias);
    }
  }
}

namespace {
struct ParallelLoopFusion
    : public impl::SCFParallelLoopFusionBase<ParallelLoopFusion> {
  void runOnOperation() override {
    auto &AA = getAnalysis<AliasAnalysis>();

    auto mayAlias = [&](Value val1, Value val2) -> bool {
      return !AA.alias(val1, val2).isNo();
    };

    getOperation()->walk([&](Operation *child) {
      for (Region &region : child->getRegions())
        naivelyFuseParallelOps(region, mayAlias);
    });
  }
};
} // namespace

std::unique_ptr<Pass> mlir::createParallelLoopFusionPass() {
  return std::make_unique<ParallelLoopFusion>();
}
