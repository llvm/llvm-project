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
#include "mlir/IR/OperationSupport.h"
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
          loadIndices[i]) {
        auto *storeIndexDefOp = storeIndices[i].getDefiningOp();
        auto *loadIndexDefOp = loadIndices[i].getDefiningOp();
        if (storeIndexDefOp && loadIndexDefOp) {
          if (!isMemoryEffectFree(storeIndexDefOp))
            return WalkResult::interrupt();
          if (!isMemoryEffectFree(loadIndexDefOp))
            return WalkResult::interrupt();
          if (!OperationEquivalence::isEquivalentTo(
                  storeIndexDefOp, loadIndexDefOp,
                  [&](Value storeIndex, Value loadIndex) {
                    if (firstToSecondPloopIndices.lookupOrDefault(storeIndex) !=
                        firstToSecondPloopIndices.lookupOrDefault(loadIndex))
                      return failure();
                    else
                      return success();
                  },
                  /*markEquivalent=*/nullptr,
                  OperationEquivalence::Flags::IgnoreLocations)) {
            return WalkResult::interrupt();
          }
        } else
          return WalkResult::interrupt();
      }
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
/// Updates secondPloop with new loop.
static void fuseIfLegal(ParallelOp firstPloop, ParallelOp &secondPloop,
                        OpBuilder builder,
                        llvm::function_ref<bool(Value, Value)> mayAlias) {
  Block *block1 = firstPloop.getBody();
  Block *block2 = secondPloop.getBody();
  IRMapping firstToSecondPloopIndices;
  firstToSecondPloopIndices.map(block1->getArguments(), block2->getArguments());

  if (!isFusionLegal(firstPloop, secondPloop, firstToSecondPloopIndices,
                     mayAlias))
    return;

  DominanceInfo dom;
  // We are fusing first loop into second, make sure there are no users of the
  // first loop results between loops.
  for (Operation *user : firstPloop->getUsers())
    if (!dom.properlyDominates(secondPloop, user, /*enclosingOpOk*/ false))
      return;

  ValueRange inits1 = firstPloop.getInitVals();
  ValueRange inits2 = secondPloop.getInitVals();

  SmallVector<Value> newInitVars(inits1.begin(), inits1.end());
  newInitVars.append(inits2.begin(), inits2.end());

  IRRewriter b(builder);
  b.setInsertionPoint(secondPloop);
  auto newSecondPloop = b.create<ParallelOp>(
      secondPloop.getLoc(), secondPloop.getLowerBound(),
      secondPloop.getUpperBound(), secondPloop.getStep(), newInitVars);

  Block *newBlock = newSecondPloop.getBody();
  auto term1 = cast<ReduceOp>(block1->getTerminator());
  auto term2 = cast<ReduceOp>(block2->getTerminator());

  b.inlineBlockBefore(block2, newBlock, newBlock->begin(),
                      newBlock->getArguments());
  b.inlineBlockBefore(block1, newBlock, newBlock->begin(),
                      newBlock->getArguments());

  ValueRange results = newSecondPloop.getResults();
  if (!results.empty()) {
    b.setInsertionPointToEnd(newBlock);

    ValueRange reduceArgs1 = term1.getOperands();
    ValueRange reduceArgs2 = term2.getOperands();
    SmallVector<Value> newReduceArgs(reduceArgs1.begin(), reduceArgs1.end());
    newReduceArgs.append(reduceArgs2.begin(), reduceArgs2.end());

    auto newReduceOp = b.create<scf::ReduceOp>(term2.getLoc(), newReduceArgs);

    for (auto &&[i, reg] : llvm::enumerate(llvm::concat<Region>(
             term1.getReductions(), term2.getReductions()))) {
      Block &oldRedBlock = reg.front();
      Block &newRedBlock = newReduceOp.getReductions()[i].front();
      b.inlineBlockBefore(&oldRedBlock, &newRedBlock, newRedBlock.begin(),
                          newRedBlock.getArguments());
    }

    firstPloop.replaceAllUsesWith(results.take_front(inits1.size()));
    secondPloop.replaceAllUsesWith(results.take_back(inits2.size()));
  }
  term1->erase();
  term2->erase();
  firstPloop.erase();
  secondPloop.erase();
  secondPloop = newSecondPloop;
}

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
