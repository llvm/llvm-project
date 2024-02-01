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
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
#define GEN_PASS_DEF_SCFPARALLELLOOPFUSION
#include "mlir/Dialect/SCF/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::scf;

/// Verify there are no nested ParallelOps.
static bool hasNestedParallelOp(ParallelOp ploop) {
  auto walkResult =
      ploop.getBody()->walk([](ParallelOp) { return WalkResult::interrupt(); });
  return walkResult.wasInterrupted();
}

/// Verify equal iteration spaces.
static bool equalIterationSpaces(ParallelOp firstPloop,
                                 ParallelOp secondPloop) {
  if (firstPloop.getNumLoops() != secondPloop.getNumLoops())
    return false;

  auto matchOperands = [&](const OperandRange &lhs,
                           const OperandRange &rhs) -> bool {
    // TODO: Extend this to support aliases and equal constants.
    return std::equal(lhs.begin(), lhs.end(), rhs.begin());
  };
  return matchOperands(firstPloop.getLowerBound(),
                       secondPloop.getLowerBound()) &&
         matchOperands(firstPloop.getUpperBound(),
                       secondPloop.getUpperBound()) &&
         matchOperands(firstPloop.getStep(), secondPloop.getStep());
}

/// Checks if the parallel loops have mixed access to the same buffers. Returns
/// `true` if the first parallel loop writes to the same indices that the second
/// loop reads.
static bool haveNoReadsAfterWriteExceptSameIndex(
    ParallelOp firstPloop, ParallelOp secondPloop,
    const IRMapping &firstToSecondPloopIndices,
    llvm::function_ref<bool(Value, Value)> mayAlias) {
  DenseMap<Value, SmallVector<ValueRange, 1>> bufferStores;
  SmallVector<Value> bufferStoresVec;
  firstPloop.getBody()->walk([&](memref::StoreOp store) {
    bufferStores[store.getMemRef()].push_back(store.getIndices());
    bufferStoresVec.emplace_back(store.getMemRef());
  });
  auto walkResult = secondPloop.getBody()->walk([&](memref::LoadOp load) {
    Value loadMem = load.getMemRef();
    // Stop if the memref is defined in secondPloop body. Careful alias analysis
    // is needed.
    auto *memrefDef = loadMem.getDefiningOp();
    if (memrefDef && memrefDef->getBlock() == load->getBlock())
      return WalkResult::interrupt();

    for (Value store : bufferStoresVec)
      if (store != loadMem && mayAlias(store, loadMem))
        return WalkResult::interrupt();

    auto write = bufferStores.find(loadMem);
    if (write == bufferStores.end())
      return WalkResult::advance();

    // Check that at last one store was retrieved
    if (!write->second.size())
      return WalkResult::interrupt();

    auto storeIndices = write->second.front();

    // Multiple writes to the same memref are allowed only on the same indices
    for (const auto &othStoreIndices : write->second) {
      if (othStoreIndices != storeIndices)
        return WalkResult::interrupt();
    }

    // Check that the load indices of secondPloop coincide with store indices of
    // firstPloop for the same memrefs.
    auto loadIndices = load.getIndices();
    if (storeIndices.size() != loadIndices.size())
      return WalkResult::interrupt();
    for (int i = 0, e = storeIndices.size(); i < e; ++i) {
      if (firstToSecondPloopIndices.lookupOrDefault(storeIndices[i]) !=
          loadIndices[i])
        return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return !walkResult.wasInterrupted();
}

/// Analyzes dependencies in the most primitive way by checking simple read and
/// write patterns.
static LogicalResult
verifyDependencies(ParallelOp firstPloop, ParallelOp secondPloop,
                   const IRMapping &firstToSecondPloopIndices,
                   llvm::function_ref<bool(Value, Value)> mayAlias) {
  if (!haveNoReadsAfterWriteExceptSameIndex(
          firstPloop, secondPloop, firstToSecondPloopIndices, mayAlias))
    return failure();

  IRMapping secondToFirstPloopIndices;
  secondToFirstPloopIndices.map(secondPloop.getBody()->getArguments(),
                                firstPloop.getBody()->getArguments());
  return success(haveNoReadsAfterWriteExceptSameIndex(
      secondPloop, firstPloop, secondToFirstPloopIndices, mayAlias));
}

static bool isFusionLegal(ParallelOp firstPloop, ParallelOp secondPloop,
                          const IRMapping &firstToSecondPloopIndices,
                          llvm::function_ref<bool(Value, Value)> mayAlias) {
  return !hasNestedParallelOp(firstPloop) &&
         !hasNestedParallelOp(secondPloop) &&
         equalIterationSpaces(firstPloop, secondPloop) &&
         succeeded(verifyDependencies(firstPloop, secondPloop,
                                      firstToSecondPloopIndices, mayAlias));
}

/// Prepends operations of firstPloop's body into secondPloop's body.
static void fuseIfLegal(ParallelOp firstPloop, ParallelOp secondPloop,
                        OpBuilder b,
                        llvm::function_ref<bool(Value, Value)> mayAlias) {
  IRMapping firstToSecondPloopIndices;
  firstToSecondPloopIndices.map(firstPloop.getBody()->getArguments(),
                                secondPloop.getBody()->getArguments());

  if (!isFusionLegal(firstPloop, secondPloop, firstToSecondPloopIndices,
                     mayAlias))
    return;

  b.setInsertionPointToStart(secondPloop.getBody());
  for (auto &op : firstPloop.getBody()->without_terminator())
    b.clone(op, firstToSecondPloopIndices);
  firstPloop.erase();
}

void mlir::scf::naivelyFuseParallelOps(
    Region &region, llvm::function_ref<bool(Value, Value)> mayAlias) {
  OpBuilder b(region);
  // Consider every single block and attempt to fuse adjacent loops.
  for (auto &block : region) {
    SmallVector<SmallVector<ParallelOp, 8>, 1> ploopChains{{}};
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
    for (ArrayRef<ParallelOp> ploops : ploopChains) {
      for (int i = 0, e = ploops.size(); i + 1 < e; ++i)
        fuseIfLegal(ploops[i], ploops[i + 1], b, mayAlias);
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
