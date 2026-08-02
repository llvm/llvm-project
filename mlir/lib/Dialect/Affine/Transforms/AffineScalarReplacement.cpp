//===- AffineScalarReplacement.cpp - Affine scalar replacement pass -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to forward affine memref stores to loads, thereby
// potentially getting rid of intermediate memrefs entirely. It also removes
// redundant loads.
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Affine/Transforms/Passes.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IntegerSet.h"

namespace mlir {
namespace affine {
#define GEN_PASS_DEF_AFFINESCALARREPLACEMENT
#include "mlir/Dialect/Affine/Transforms/Passes.h.inc"
} // namespace affine
} // namespace mlir

#define DEBUG_TYPE "affine-scalrep"

using namespace mlir;
using namespace mlir::affine;

namespace {

// Stores matched pairs of AffineStoreOps to be sinked outside the AffineIfOp.
struct IfStorePairToSink {
  AffineStoreOp thenStore;
  AffineStoreOp elseStore;
  AffineStoreOp parentStore;
};

// Transformation plan for a AffineIfOp.
struct IfSinkPlan {
  AffineIfOp ifOp;
  llvm::SmallVector<IfStorePairToSink, 4> pairs;
};

// Returns true if `store` is the last store accessing its memory location
// within its block.
static bool isLastStoreInBlock(AffineStoreOp store) {
  MemRefAccess curAccess(store);
  for (Operation &op : llvm::reverse(*store->getBlock())) {
    if (auto otherStore = dyn_cast<AffineStoreOp>(&op)) {
      if (MemRefAccess(otherStore) == curAccess)
        return otherStore == store;
    }
  }
  return false;
}

/// Analyzes a single block (either 'then' or 'else') of an AffineIfOp to
/// identify store operations that can be sinked outside the if statement.
///
/// This function identifies store operations across the following memory access
/// scenarios by tracing users of the target memref:
///   1. Parent + Both Branches: The parentStore (located before the ifOp in the
///   same outer block), the thenStore, and the elseStore all access the exact
///   same memory location.
///   2. Parent + Single Branch: The parentStore (located before the ifOp in the
///   same outer block) and a store in either the 'then' or 'else' block access
///   the exact same memory location.
///
/// Found store candidates are recorded into 'plan' for transformation, and
/// matched stores in the 'else' block are added to 'visited' to prevent
/// duplicate processing.
static void analyzeIfBlock(AffineIfOp ifOp, Block *block, IfSinkPlan &plan,
                           SmallVectorImpl<AffineStoreOp> &visited) {
  // Check whether the target block is the 'then' block of the ifOp (and an
  // 'else' block exists).
  bool curThenBlock = ifOp.hasElse() && block == ifOp.getThenBlock();

  // Iterate through operations in the block in reverse order.
  for (Operation &op : llvm::reverse(*block)) {
    auto store = dyn_cast<AffineStoreOp>(&op);

    // Skip if it is not an AffineStoreOp or not the last store targeting the
    // memory location in this.
    if (!store || !isLastStoreInBlock(store))
      continue;

    // Skip if this store has already been processed and visited.
    if (llvm::is_contained(visited, store))
      continue;

    // Verify index operands are valid outside the ifOp scope.
    MemRefAccess access(store);
    if (llvm::any_of(store.getIndices(), [&](Value operand) {
          Operation *defineOp = operand.getDefiningOp();
          return !defineOp && defineOp->getBlock() == block;
        }))
      continue;

    Value memref = store.getMemRef();
    AffineStoreOp parentStore = nullptr;
    AffineStoreOp elseStore = nullptr;

    // Trace all users of the memref to locate matching stores.
    for (Operation *user : memref.getUsers()) {
      auto userStore = dyn_cast<AffineStoreOp>(user);
      if (!userStore || userStore == store)
        continue;

      if (MemRefAccess(userStore) != access)
        continue;

      // Ensure the `parentStore` is in the same block as ifOp and precedes it,
      // update parentStore to keep the closest one preceding ifOp.
      if (userStore->getBlock() == ifOp->getBlock() &&
          userStore->isBeforeInBlock(ifOp))
        if (!parentStore || parentStore->isBeforeInBlock(userStore))
          parentStore = userStore;

      // Identify matching stores in the 'else' block when analyzing the 'then'
      // block.
      bool userInElseBlock =
          ifOp.hasElse() && userStore->getBlock() == ifOp.getElseBlock();
      if (curThenBlock && userInElseBlock &&
          (!elseStore || isLastStoreInBlock(userStore)))
        elseStore = userStore;
    }

    // Skip if no matching parent store was found.
    if (!parentStore)
      continue;

    // Record the candidate store pair into the sink plan based on the current
    // branch being analyzed.
    if (curThenBlock) {
      plan.pairs.push_back({store, elseStore, parentStore});
      if (elseStore)
        visited.push_back(elseStore);
    } else {
      plan.pairs.push_back({nullptr, store, parentStore});
    }
  }
}

/// Analyzes an AffineIfOp to build a store sinking plan.
static void analyzeIfOp(AffineIfOp ifOp, IfSinkPlan &plan) {
  /// maintaining a `visited` tracking vector across both blocks to prevent
  /// store operations in the `else` block from being processed twice.
  SmallVector<AffineStoreOp> visited;
  analyzeIfBlock(ifOp, ifOp.getThenBlock(), plan, visited);
  if (ifOp.hasElse())
    analyzeIfBlock(ifOp, ifOp.getElseBlock(), plan, visited);
}

/// Applies the store sinking plan to rewrite the target AffineIfOp. This
/// transformation replaces the existing `ifOp` with a new `AffineIfOp` that
/// yields the values to be stored across branches. It updates the
/// `AffineYieldOp` terminators in both 'then' and 'else' blocks, erases
/// internal store operations, and emits a single unified `AffineStoreOp`
/// immediately following the new `ifOp` for each sinked pair.
static void applySinkPlan(IfSinkPlan &plan) {
  if (plan.pairs.empty())
    return;

  AffineIfOp ifOp = plan.ifOp;
  IRRewriter rewriter(ifOp);

  // Collect new result types for the new AffineIfOp.
  SmallVector<Type, 4> newTypes(ifOp.getResultTypes());
  for (IfStorePairToSink &pair : plan.pairs)
    newTypes.push_back(pair.parentStore.getValue().getType());

  // Create a new AffineIfOp with 'withElse = true' because yielded values must
  // be passed through both branches (e.g., propagating parentStore value in the
  // else branch).
  auto newIf =
      AffineIfOp::create(rewriter, ifOp->getLoc(), newTypes,
                         ifOp.getIntegerSet(), ifOp->getOperands(), true);

  // Take blocks from oldIf into newIf.
  newIf.getThenRegion().takeBody(ifOp.getThenRegion());
  if (ifOp.hasElse()) {
    newIf.getElseRegion().takeBody(ifOp.getElseRegion());
  } else {
    // If we create an else block, we need to explicitly insert a yield.
    rewriter.setInsertionPointToEnd(newIf.getElseBlock());
    AffineYieldOp::create(rewriter, newIf->getLoc());
  }

  // Update yield operands.
  Block *thenBlock = newIf.getThenBlock();
  Block *elseBlock = newIf.getElseBlock();

  AffineYieldOp thenYield = cast<AffineYieldOp>(thenBlock->getTerminator());
  AffineYieldOp elseYield = cast<AffineYieldOp>(elseBlock->getTerminator());

  // Preserve existing yield values from both branches.
  SmallVector<Value, 4> thenYieldVals(thenYield.getOperands());
  SmallVector<Value, 4> elseYieldVals(elseYield.getOperands());

  for (auto [thenStore, elseStore, parentStore] : plan.pairs) {
    // 1. Both branches have stores: yield their respective store values.
    // 2 & 3. Only one branch has a store: yield that store value in its branch,
    //        and propagate parentStore's value through the other branch.
    if (thenStore && elseStore) {
      thenYieldVals.push_back(thenStore.getValue());
      elseYieldVals.push_back(elseStore.getValue());
    } else if (thenStore) {
      thenYieldVals.push_back(thenStore.getValue());
      elseYieldVals.push_back(parentStore.getValue());
    } else {
      thenYieldVals.push_back(parentStore.getValue());
      elseYieldVals.push_back(elseStore.getValue());
    }
  }

  // Update Then block terminator.
  rewriter.setInsertionPoint(thenYield);
  AffineYieldOp::create(rewriter, thenYield->getLoc(), thenYieldVals);
  rewriter.eraseOp(thenYield);

  // Update Else block terminator.
  rewriter.setInsertionPoint(elseYield);
  AffineYieldOp::create(rewriter, elseYield->getLoc(), elseYieldVals);
  rewriter.eraseOp(elseYield);

  // Emit unified AffineStoreOps after newIf using appended results.
  rewriter.setInsertionPointAfter(newIf);
  unsigned baseIdx = ifOp.getNumResults();
  for (auto it : llvm::enumerate(plan.pairs)) {
    AffineStoreOp store = it.value().parentStore;
    Value valueToStore = newIf.getResult(baseIdx + it.index());
    AffineStoreOp::create(rewriter, store->getLoc(), valueToStore,
                          store.getMemRef(), store.getIndices());
    if (it.value().elseStore)
      rewriter.eraseOp(it.value().elseStore);
    if (it.value().thenStore)
      rewriter.eraseOp(it.value().thenStore);
    rewriter.eraseOp(store);
  }

  // Replace original ifOp results and erase stale oldIf.
  for (auto [oldValue, newValue] : llvm::zip_equal(
           ifOp.getResults(), newIf.getResults().take_front(baseIdx)))
    rewriter.replaceAllUsesWith(oldValue, newValue);
  rewriter.eraseOp(ifOp);
}

struct AffineScalarReplacement
    : public affine::impl::AffineScalarReplacementBase<
          AffineScalarReplacement> {
  void runOnOperation() override;
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::affine::createAffineScalarReplacementPass() {
  return std::make_unique<AffineScalarReplacement>();
}

void AffineScalarReplacement::runOnOperation() {
  SmallVector<IfSinkPlan> plans;
  getOperation()->walk([&](AffineIfOp ifOp) {
    IfSinkPlan plan{/*ifOp=*/ifOp, /*pairs=*/{}};
    analyzeIfOp(ifOp, plan);
    if (!plan.pairs.empty()) {
      plans.push_back(plan);
    }
  });
  for (IfSinkPlan &plan : plans)
    applySinkPlan(plan);

  affineScalarReplace(getOperation(), getAnalysis<DominanceInfo>(),
                      getAnalysis<PostDominanceInfo>(),
                      getAnalysis<AliasAnalysis>());
}
