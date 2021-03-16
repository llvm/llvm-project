//===- GreedyPatternRewriteDriver.cpp - A greedy rewriter -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements mlir::applyPatternsAndFoldGreedily.
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Rewrite/PatternApplicator.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

#define DEBUG_TYPE "pattern-matcher"

/// The max number of iterations scanning for pattern match.
static unsigned maxPatternMatchIterations = 10;

//===----------------------------------------------------------------------===//
// GreedyPatternRewriteDriver
//===----------------------------------------------------------------------===//

namespace {
/// This is a worklist-driven driver for the PatternMatcher, which repeatedly
/// applies the locally optimal patterns in a roughly "bottom up" way.
class GreedyPatternRewriteDriver : public PatternRewriter {
public:
  explicit GreedyPatternRewriteDriver(MLIRContext *ctx,
                                      const FrozenRewritePatternList &patterns)
      : PatternRewriter(ctx), matcher(patterns), folder(ctx) {
    worklist.reserve(64);

    // Apply a simple cost model based solely on pattern benefit.
    matcher.applyDefaultCostModel();
  }

  bool simplify(MutableArrayRef<Region> regions, int maxIterations);

  void addToWorklist(Operation *op) {
    // Check to see if the worklist already contains this op.
    if (worklistMap.count(op))
      return;

    worklistMap[op] = worklist.size();
    worklist.push_back(op);
  }

  Operation *popFromWorklist() {
    auto *op = worklist.back();
    worklist.pop_back();

    // This operation is no longer in the worklist, keep worklistMap up to date.
    if (op)
      worklistMap.erase(op);
    return op;
  }

  /// If the specified operation is in the worklist, remove it.  If not, this is
  /// a no-op.
  void removeFromWorklist(Operation *op) {
    auto it = worklistMap.find(op);
    if (it != worklistMap.end()) {
      assert(worklist[it->second] == op && "malformed worklist data structure");
      worklist[it->second] = nullptr;
      worklistMap.erase(it);
    }
  }

  // These are hooks implemented for PatternRewriter.
protected:
  // Implement the hook for inserting operations, and make sure that newly
  // inserted ops are added to the worklist for processing.
  void notifyOperationInserted(Operation *op) override { addToWorklist(op); }

  // If an operation is about to be removed, make sure it is not in our
  // worklist anymore because we'd get dangling references to it.
  void notifyOperationRemoved(Operation *op) override {
    addToWorklist(op->getOperands());
    op->walk([this](Operation *operation) {
      removeFromWorklist(operation);
      folder.notifyRemoval(operation);
    });
  }

  // When the root of a pattern is about to be replaced, it can trigger
  // simplifications to its users - make sure to add them to the worklist
  // before the root is changed.
  void notifyRootReplaced(Operation *op) override {
    for (auto result : op->getResults())
      for (auto *user : result.getUsers())
        addToWorklist(user);
  }

private:
  // Look over the provided operands for any defining operations that should
  // be re-added to the worklist. This function should be called when an
  // operation is modified or removed, as it may trigger further
  // simplifications.
  template <typename Operands> void addToWorklist(Operands &&operands) {
    for (Value operand : operands) {
      // If the use count of this operand is now < 2, we re-add the defining
      // operation to the worklist.
      // TODO: This is based on the fact that zero use operations
      // may be deleted, and that single use values often have more
      // canonicalization opportunities.
      if (!operand.use_empty() && !operand.hasOneUse())
        continue;
      if (auto *defInst = operand.getDefiningOp())
        addToWorklist(defInst);
    }
  }

  /// The low-level pattern applicator.
  PatternApplicator matcher;

  /// The worklist for this transformation keeps track of the operations that
  /// need to be revisited, plus their index in the worklist.  This allows us to
  /// efficiently remove operations from the worklist when they are erased, even
  /// if they aren't the root of a pattern.
  std::vector<Operation *> worklist;
  DenseMap<Operation *, unsigned> worklistMap;

  /// Non-pattern based folder for operations.
  OperationFolder folder;
};
} // end anonymous namespace

/// Performs the rewrites while folding and erasing any dead ops. Returns true
/// if the rewrite converges in `maxIterations`.
bool GreedyPatternRewriteDriver::simplify(MutableArrayRef<Region> regions,
                                          int maxIterations) {
  // Add the given operation to the worklist.
  auto collectOps = [this](Operation *op) { addToWorklist(op); };

  bool changed = false;
  int i = 0;
  do {
    // Add all nested operations to the worklist.
    for (auto &region : regions)
      region.walk(collectOps);

    // These are scratch vectors used in the folding loop below.
    SmallVector<Value, 8> originalOperands, resultValues;

    changed = false;
    while (!worklist.empty()) {
      auto *op = popFromWorklist();

      // Nulls get added to the worklist when operations are removed, ignore
      // them.
      if (op == nullptr)
        continue;

      // If the operation is trivially dead - remove it.
      if (isOpTriviallyDead(op)) {
        notifyOperationRemoved(op);
        op->erase();
        changed = true;
        continue;
      }

      // Collects all the operands and result uses of the given `op` into work
      // list. Also remove `op` and nested ops from worklist.
      originalOperands.assign(op->operand_begin(), op->operand_end());
      auto preReplaceAction = [&](Operation *op) {
        // Add the operands to the worklist for visitation.
        addToWorklist(originalOperands);

        // Add all the users of the result to the worklist so we make sure
        // to revisit them.
        for (auto result : op->getResults())
          for (auto *userOp : result.getUsers())
            addToWorklist(userOp);

        notifyOperationRemoved(op);
      };

      // Try to fold this op.
      bool inPlaceUpdate;
      if ((succeeded(folder.tryToFold(op, collectOps, preReplaceAction,
                                      &inPlaceUpdate)))) {
        changed = true;
        if (!inPlaceUpdate)
          continue;
      }

      // Try to match one of the patterns. The rewriter is automatically
      // notified of any necessary changes, so there is nothing else to do here.
      changed |= succeeded(matcher.matchAndRewrite(op, *this));
    }

    // After applying patterns, make sure that the CFG of each of the regions is
    // kept up to date.
    if (succeeded(simplifyRegions(regions))) {
      folder.clear();
      changed = true;
    }
  } while (changed && ++i < maxIterations);
  // Whether the rewrite converges, i.e. wasn't changed in the last iteration.
  return !changed;
}

/// Rewrite the regions of the specified operation, which must be isolated from
/// above, by repeatedly applying the highest benefit patterns in a greedy
/// work-list driven manner. Return success if no more patterns can be matched
/// in the result operation regions. Note: This does not apply patterns to the
/// top-level operation itself.
///
LogicalResult
mlir::applyPatternsAndFoldGreedily(Operation *op,
                                   const FrozenRewritePatternList &patterns) {
  return applyPatternsAndFoldGreedily(op, patterns, maxPatternMatchIterations);
}
LogicalResult
mlir::applyPatternsAndFoldGreedily(Operation *op,
                                   const FrozenRewritePatternList &patterns,
                                   unsigned maxIterations) {
  return applyPatternsAndFoldGreedily(op->getRegions(), patterns,
                                      maxIterations);
}
/// Rewrite the given regions, which must be isolated from above.
LogicalResult
mlir::applyPatternsAndFoldGreedily(MutableArrayRef<Region> regions,
                                   const FrozenRewritePatternList &patterns) {
  return applyPatternsAndFoldGreedily(regions, patterns,
                                      maxPatternMatchIterations);
}
LogicalResult
mlir::applyPatternsAndFoldGreedily(MutableArrayRef<Region> regions,
                                   const FrozenRewritePatternList &patterns,
                                   unsigned maxIterations) {
  if (regions.empty())
    return success();

  // The top-level operation must be known to be isolated from above to
  // prevent performing canonicalizations on operations defined at or above
  // the region containing 'op'.
  auto regionIsIsolated = [](Region &region) {
    return region.getParentOp()->hasTrait<OpTrait::IsIsolatedFromAbove>();
  };
  (void)regionIsIsolated;
  assert(llvm::all_of(regions, regionIsIsolated) &&
         "patterns can only be applied to operations IsolatedFromAbove");

  // Start the pattern driver.
  GreedyPatternRewriteDriver driver(regions[0].getContext(), patterns);
  bool converged = driver.simplify(regions, maxIterations);
  LLVM_DEBUG(if (!converged) {
    llvm::dbgs() << "The pattern rewrite doesn't converge after scanning "
                 << maxIterations << " times\n";
  });
  return success(converged);
}

//===----------------------------------------------------------------------===//
// OpPatternRewriteDriver
//===----------------------------------------------------------------------===//

namespace {
/// This is a simple driver for the PatternMatcher to apply patterns and perform
/// folding on a single op. It repeatedly applies locally optimal patterns.
class OpPatternRewriteDriver : public PatternRewriter {
public:
  explicit OpPatternRewriteDriver(MLIRContext *ctx,
                                  const FrozenRewritePatternList &patterns)
      : PatternRewriter(ctx), matcher(patterns), folder(ctx) {
    // Apply a simple cost model based solely on pattern benefit.
    matcher.applyDefaultCostModel();
  }

  /// Performs the rewrites and folding only on `op`. The simplification
  /// converges if the op is erased as a result of being folded, replaced, or
  /// dead, or no more changes happen in an iteration. Returns success if the
  /// rewrite converges in `maxIterations`. `erased` is set to true if `op` gets
  /// erased.
  LogicalResult simplifyLocally(Operation *op, int maxIterations, bool &erased);

  // These are hooks implemented for PatternRewriter.
protected:
  /// If an operation is about to be removed, mark it so that we can let clients
  /// know.
  void notifyOperationRemoved(Operation *op) override {
    opErasedViaPatternRewrites = true;
  }

  // When a root is going to be replaced, its removal will be notified as well.
  // So there is nothing to do here.
  void notifyRootReplaced(Operation *op) override {}

private:
  /// The low-level pattern applicator.
  PatternApplicator matcher;

  /// Non-pattern based folder for operations.
  OperationFolder folder;

  /// Set to true if the operation has been erased via pattern rewrites.
  bool opErasedViaPatternRewrites = false;
};

} // anonymous namespace

LogicalResult OpPatternRewriteDriver::simplifyLocally(Operation *op,
                                                      int maxIterations,
                                                      bool &erased) {
  bool changed = false;
  erased = false;
  opErasedViaPatternRewrites = false;
  int i = 0;
  // Iterate until convergence or until maxIterations. Deletion of the op as
  // a result of being dead or folded is convergence.
  do {
    changed = false;

    // If the operation is trivially dead - remove it.
    if (isOpTriviallyDead(op)) {
      op->erase();
      erased = true;
      return success();
    }

    // Try to fold this op.
    bool inPlaceUpdate;
    if (succeeded(folder.tryToFold(op, /*processGeneratedConstants=*/nullptr,
                                   /*preReplaceAction=*/nullptr,
                                   &inPlaceUpdate))) {
      changed = true;
      if (!inPlaceUpdate) {
        erased = true;
        return success();
      }
    }

    // Try to match one of the patterns. The rewriter is automatically
    // notified of any necessary changes, so there is nothing else to do here.
    changed |= succeeded(matcher.matchAndRewrite(op, *this));
    if ((erased = opErasedViaPatternRewrites))
      return success();
  } while (changed && ++i < maxIterations);

  // Whether the rewrite converges, i.e. wasn't changed in the last iteration.
  return failure(changed);
}

/// Rewrites only `op` using the supplied canonicalization patterns and
/// folding. `erased` is set to true if the op is erased as a result of being
/// folded, replaced, or dead.
LogicalResult mlir::applyOpPatternsAndFold(
    Operation *op, const FrozenRewritePatternList &patterns, bool *erased) {
  // Start the pattern driver.
  OpPatternRewriteDriver driver(op->getContext(), patterns);
  bool opErased;
  LogicalResult converged =
      driver.simplifyLocally(op, maxPatternMatchIterations, opErased);
  if (erased)
    *erased = opErased;
  LLVM_DEBUG(if (failed(converged)) {
    llvm::dbgs() << "The pattern rewrite doesn't converge after scanning "
                 << maxPatternMatchIterations << " times";
  });
  return converged;
}
