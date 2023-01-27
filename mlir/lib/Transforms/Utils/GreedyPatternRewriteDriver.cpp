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
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Rewrite/PatternApplicator.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ScopedPrinter.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

#define DEBUG_TYPE "greedy-rewriter"

//===----------------------------------------------------------------------===//
// GreedyPatternRewriteDriver
//===----------------------------------------------------------------------===//

namespace {
/// This is a worklist-driven driver for the PatternMatcher, which repeatedly
/// applies the locally optimal patterns in a roughly "bottom up" way.
class GreedyPatternRewriteDriver : public PatternRewriter {
public:
  explicit GreedyPatternRewriteDriver(MLIRContext *ctx,
                                      const FrozenRewritePatternSet &patterns,
                                      const GreedyRewriteConfig &config);

  /// Simplify the operations within the given regions.
  bool simplify(MutableArrayRef<Region> regions);

  /// Add the given operation and its ancestors to the worklist.
  void addToWorklist(Operation *op);

  /// Pop the next operation from the worklist.
  Operation *popFromWorklist();

  /// If the specified operation is in the worklist, remove it.
  void removeFromWorklist(Operation *op);

  /// Notifies the driver that the specified operation may have been modified
  /// in-place.
  void finalizeRootUpdate(Operation *op) override;

protected:
  /// Add the given operation to the worklist.
  virtual void addSingleOpToWorklist(Operation *op);

  // Implement the hook for inserting operations, and make sure that newly
  // inserted ops are added to the worklist for processing.
  void notifyOperationInserted(Operation *op) override;

  // Look over the provided operands for any defining operations that should
  // be re-added to the worklist. This function should be called when an
  // operation is modified or removed, as it may trigger further
  // simplifications.
  void addOperandsToWorklist(ValueRange operands);

  // If an operation is about to be removed, make sure it is not in our
  // worklist anymore because we'd get dangling references to it.
  void notifyOperationRemoved(Operation *op) override;

  // When the root of a pattern is about to be replaced, it can trigger
  // simplifications to its users - make sure to add them to the worklist
  // before the root is changed.
  void notifyRootReplaced(Operation *op, ValueRange replacement) override;

  /// PatternRewriter hook for notifying match failure reasons.
  LogicalResult
  notifyMatchFailure(Location loc,
                     function_ref<void(Diagnostic &)> reasonCallback) override;

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

protected:
  /// Configuration information for how to simplify.
  GreedyRewriteConfig config;

  /// Only ops within this scope are simplified. This is set at the beginning
  /// of `simplify()` and `simplifyLocally()` to the current scope the rewriter
  /// operates on.
  DenseSet<Region *> scope;

private:
#ifndef NDEBUG
  /// A logger used to emit information during the application process.
  llvm::ScopedPrinter logger{llvm::dbgs()};
#endif
};
} // namespace

GreedyPatternRewriteDriver::GreedyPatternRewriteDriver(
    MLIRContext *ctx, const FrozenRewritePatternSet &patterns,
    const GreedyRewriteConfig &config)
    : PatternRewriter(ctx), matcher(patterns), folder(ctx), config(config) {
  worklist.reserve(64);

  // Apply a simple cost model based solely on pattern benefit.
  matcher.applyDefaultCostModel();
}

bool GreedyPatternRewriteDriver::simplify(MutableArrayRef<Region> regions) {
  scope.clear();
  for (Region &r : regions)
    scope.insert(&r);

#ifndef NDEBUG
  const char *logLineComment =
      "//===-------------------------------------------===//\n";

  /// A utility function to log a process result for the given reason.
  auto logResult = [&](StringRef result, const llvm::Twine &msg = {}) {
    logger.unindent();
    logger.startLine() << "} -> " << result;
    if (!msg.isTriviallyEmpty())
      logger.getOStream() << " : " << msg;
    logger.getOStream() << "\n";
  };
  auto logResultWithLine = [&](StringRef result, const llvm::Twine &msg = {}) {
    logResult(result, msg);
    logger.startLine() << logLineComment;
  };
#endif

  auto insertKnownConstant = [&](Operation *op) {
    // Check for existing constants when populating the worklist. This avoids
    // accidentally reversing the constant order during processing.
    Attribute constValue;
    if (matchPattern(op, m_Constant(&constValue)))
      if (!folder.insertKnownConstant(op, constValue))
        return true;
    return false;
  };

  bool changed = false;
  int64_t iteration = 0;
  do {
    // Check if the iteration limit was reached.
    if (iteration++ >= config.maxIterations &&
        config.maxIterations != GreedyRewriteConfig::kNoLimit)
      break;

    worklist.clear();
    worklistMap.clear();

    if (!config.useTopDownTraversal) {
      // Add operations to the worklist in postorder.
      for (auto &region : regions) {
        region.walk([&](Operation *op) {
          if (!insertKnownConstant(op))
            addToWorklist(op);
        });
      }
    } else {
      // Add all nested operations to the worklist in preorder.
      for (auto &region : regions) {
        region.walk<WalkOrder::PreOrder>([&](Operation *op) {
          if (!insertKnownConstant(op)) {
            worklist.push_back(op);
            return WalkResult::advance();
          }
          return WalkResult::skip();
        });
      }

      // Reverse the list so our pop-back loop processes them in-order.
      std::reverse(worklist.begin(), worklist.end());
      // Remember the reverse index.
      for (size_t i = 0, e = worklist.size(); i != e; ++i)
        worklistMap[worklist[i]] = i;
    }

    // These are scratch vectors used in the folding loop below.
    SmallVector<Value, 8> originalOperands, resultValues;

    changed = false;
    int64_t numRewrites = 0;
    while (!worklist.empty() &&
           (numRewrites < config.maxNumRewrites ||
            config.maxNumRewrites == GreedyRewriteConfig::kNoLimit)) {
      auto *op = popFromWorklist();

      // Nulls get added to the worklist when operations are removed, ignore
      // them.
      if (op == nullptr)
        continue;

      LLVM_DEBUG({
        logger.getOStream() << "\n";
        logger.startLine() << logLineComment;
        logger.startLine() << "Processing operation : '" << op->getName()
                           << "'(" << op << ") {\n";
        logger.indent();

        // If the operation has no regions, just print it here.
        if (op->getNumRegions() == 0) {
          op->print(
              logger.startLine(),
              OpPrintingFlags().printGenericOpForm().elideLargeElementsAttrs());
          logger.getOStream() << "\n\n";
        }
      });

      // If the operation is trivially dead - remove it.
      if (isOpTriviallyDead(op)) {
        notifyOperationRemoved(op);
        op->erase();
        changed = true;

        LLVM_DEBUG(logResultWithLine("success", "operation is trivially dead"));
        continue;
      }

      // Collects all the operands and result uses of the given `op` into work
      // list. Also remove `op` and nested ops from worklist.
      originalOperands.assign(op->operand_begin(), op->operand_end());
      auto preReplaceAction = [&](Operation *op) {
        // Add the operands to the worklist for visitation.
        addOperandsToWorklist(originalOperands);

        // Add all the users of the result to the worklist so we make sure
        // to revisit them.
        for (auto result : op->getResults())
          for (auto *userOp : result.getUsers())
            addToWorklist(userOp);

        notifyOperationRemoved(op);
      };

      // Add the given operation to the worklist.
      auto collectOps = [this](Operation *op) { addToWorklist(op); };

      // Try to fold this op.
      bool inPlaceUpdate;
      if ((succeeded(folder.tryToFold(op, collectOps, preReplaceAction,
                                      &inPlaceUpdate)))) {
        LLVM_DEBUG(logResultWithLine("success", "operation was folded"));

        changed = true;
        if (!inPlaceUpdate)
          continue;
      }

      // Try to match one of the patterns. The rewriter is automatically
      // notified of any necessary changes, so there is nothing else to do
      // here.
#ifndef NDEBUG
      auto canApply = [&](const Pattern &pattern) {
        LLVM_DEBUG({
          logger.getOStream() << "\n";
          logger.startLine() << "* Pattern " << pattern.getDebugName() << " : '"
                             << op->getName() << " -> (";
          llvm::interleaveComma(pattern.getGeneratedOps(), logger.getOStream());
          logger.getOStream() << ")' {\n";
          logger.indent();
        });
        return true;
      };
      auto onFailure = [&](const Pattern &pattern) {
        LLVM_DEBUG(logResult("failure", "pattern failed to match"));
      };
      auto onSuccess = [&](const Pattern &pattern) {
        LLVM_DEBUG(logResult("success", "pattern applied successfully"));
        return success();
      };

      LogicalResult matchResult =
          matcher.matchAndRewrite(op, *this, canApply, onFailure, onSuccess);
      if (succeeded(matchResult))
        LLVM_DEBUG(logResultWithLine("success", "pattern matched"));
      else
        LLVM_DEBUG(logResultWithLine("failure", "pattern failed to match"));
#else
      LogicalResult matchResult = matcher.matchAndRewrite(op, *this);
#endif

      if (succeeded(matchResult)) {
        changed = true;
        ++numRewrites;
      }
    }

    // After applying patterns, make sure that the CFG of each of the regions
    // is kept up to date.
    if (config.enableRegionSimplification)
      changed |= succeeded(simplifyRegions(*this, regions));
  } while (changed);

  // Whether the rewrite converges, i.e. wasn't changed in the last iteration.
  return !changed;
}

void GreedyPatternRewriteDriver::addToWorklist(Operation *op) {
  // Gather potential ancestors while looking for a "scope" parent region.
  SmallVector<Operation *, 8> ancestors;
  ancestors.push_back(op);
  while (Region *region = op->getParentRegion()) {
    if (scope.contains(region)) {
      // All gathered ops are in fact ancestors.
      for (Operation *op : ancestors)
        addSingleOpToWorklist(op);
      break;
    }
    op = region->getParentOp();
    if (!op)
      break;
    ancestors.push_back(op);
  }
}

void GreedyPatternRewriteDriver::addSingleOpToWorklist(Operation *op) {
  // Check to see if the worklist already contains this op.
  if (worklistMap.count(op))
    return;

  worklistMap[op] = worklist.size();
  worklist.push_back(op);
}

Operation *GreedyPatternRewriteDriver::popFromWorklist() {
  auto *op = worklist.back();
  worklist.pop_back();

  // This operation is no longer in the worklist, keep worklistMap up to date.
  if (op)
    worklistMap.erase(op);
  return op;
}

void GreedyPatternRewriteDriver::removeFromWorklist(Operation *op) {
  auto it = worklistMap.find(op);
  if (it != worklistMap.end()) {
    assert(worklist[it->second] == op && "malformed worklist data structure");
    worklist[it->second] = nullptr;
    worklistMap.erase(it);
  }
}

void GreedyPatternRewriteDriver::notifyOperationInserted(Operation *op) {
  LLVM_DEBUG({
    logger.startLine() << "** Insert  : '" << op->getName() << "'(" << op
                       << ")\n";
  });
  addToWorklist(op);
}

void GreedyPatternRewriteDriver::finalizeRootUpdate(Operation *op) {
  LLVM_DEBUG({
    logger.startLine() << "** Modified: '" << op->getName() << "'(" << op
                       << ")\n";
  });
  addToWorklist(op);
}

void GreedyPatternRewriteDriver::addOperandsToWorklist(ValueRange operands) {
  for (Value operand : operands) {
    // If the use count of this operand is now < 2, we re-add the defining
    // operation to the worklist.
    // TODO: This is based on the fact that zero use operations
    // may be deleted, and that single use values often have more
    // canonicalization opportunities.
    if (!operand || (!operand.use_empty() && !operand.hasOneUse()))
      continue;
    if (auto *defOp = operand.getDefiningOp())
      addToWorklist(defOp);
  }
}

void GreedyPatternRewriteDriver::notifyOperationRemoved(Operation *op) {
  LLVM_DEBUG({
    logger.startLine() << "** Erase   : '" << op->getName() << "'(" << op
                       << ")\n";
  });

  addOperandsToWorklist(op->getOperands());
  op->walk([this](Operation *operation) {
    removeFromWorklist(operation);
    folder.notifyRemoval(operation);
  });
}

void GreedyPatternRewriteDriver::notifyRootReplaced(Operation *op,
                                                    ValueRange replacement) {
  LLVM_DEBUG({
    logger.startLine() << "** Replace : '" << op->getName() << "'(" << op
                       << ")\n";
  });
  for (auto result : op->getResults())
    for (auto *user : result.getUsers())
      addToWorklist(user);
}

LogicalResult GreedyPatternRewriteDriver::notifyMatchFailure(
    Location loc, function_ref<void(Diagnostic &)> reasonCallback) {
  LLVM_DEBUG({
    Diagnostic diag(loc, DiagnosticSeverity::Remark);
    reasonCallback(diag);
    logger.startLine() << "** Failure : " << diag.str() << "\n";
  });
  return failure();
}

/// Rewrite the regions of the specified operation, which must be isolated from
/// above, by repeatedly applying the highest benefit patterns in a greedy
/// work-list driven manner. Return success if no more patterns can be matched
/// in the result operation regions. Note: This does not apply patterns to the
/// top-level operation itself.
///
LogicalResult
mlir::applyPatternsAndFoldGreedily(MutableArrayRef<Region> regions,
                                   const FrozenRewritePatternSet &patterns,
                                   GreedyRewriteConfig config) {
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
  GreedyPatternRewriteDriver driver(regions[0].getContext(), patterns, config);
  bool converged = driver.simplify(regions);
  LLVM_DEBUG(if (!converged) {
    llvm::dbgs() << "The pattern rewrite did not converge after scanning "
                 << config.maxIterations << " times\n";
  });
  return success(converged);
}

//===----------------------------------------------------------------------===//
// MultiOpPatternRewriteDriver
//===----------------------------------------------------------------------===//

namespace {

/// This is a specialized GreedyPatternRewriteDriver to apply patterns and
/// perform folding for a supplied set of ops. It repeatedly simplifies while
/// restricting the rewrites to only the provided set of ops or optionally
/// to those directly affected by it (result users or operand providers). Parent
/// ops are not considered.
class MultiOpPatternRewriteDriver : public GreedyPatternRewriteDriver {
public:
  explicit MultiOpPatternRewriteDriver(MLIRContext *ctx,
                                       const FrozenRewritePatternSet &patterns,
                                       GreedyRewriteStrictness strictMode)
      : GreedyPatternRewriteDriver(ctx, patterns, GreedyRewriteConfig()),
        strictMode(strictMode) {}

  /// Performs the specified rewrites on `ops` while also trying to fold these
  /// ops. `strictMode` controls which other ops are simplified. Only ops
  /// within the given scope region are added to the worklist. If no scope is
  /// specified, it assumed to be closest common region of all `ops`.
  ///
  /// Note that ops in `ops` could be erased as a result of folding, becoming
  /// dead, or via pattern rewrites. The return value indicates convergence.
  ///
  /// All `ops` that survived the rewrite are stored in `surviving`.
  LogicalResult
  simplifyLocally(ArrayRef<Operation *> ops, bool *changed = nullptr,
                  llvm::SmallDenseSet<Operation *, 4> *surviving = nullptr,
                  Region *scope = nullptr);

protected:
  void addSingleOpToWorklist(Operation *op) override {
    if (strictMode == GreedyRewriteStrictness::AnyOp ||
        strictModeFilteredOps.contains(op))
      GreedyPatternRewriteDriver::addSingleOpToWorklist(op);
  }

private:
  void notifyOperationInserted(Operation *op) override {
    if (strictMode == GreedyRewriteStrictness::ExistingAndNewOps)
      strictModeFilteredOps.insert(op);
    GreedyPatternRewriteDriver::notifyOperationInserted(op);
  }

  void notifyOperationRemoved(Operation *op) override {
    GreedyPatternRewriteDriver::notifyOperationRemoved(op);
    if (survivingOps)
      survivingOps->erase(op);
    if (strictMode != GreedyRewriteStrictness::AnyOp)
      strictModeFilteredOps.erase(op);
  }

  /// `strictMode` control which ops are added to the worklist during
  /// simplification.
  GreedyRewriteStrictness strictMode = GreedyRewriteStrictness::AnyOp;

  /// The list of ops we are restricting our rewrites to. These include the
  /// supplied set of ops as well as new ops created while rewriting those ops
  /// depending on `strictMode`. This set is not maintained when `strictMode`
  /// is GreedyRewriteStrictness::AnyOp.
  llvm::SmallDenseSet<Operation *, 4> strictModeFilteredOps;

  /// An optional set of ops that survived the rewrite. This set is populated
  /// at the beginning of `simplifyLocally` with the inititally provided list
  /// of ops.
  llvm::SmallDenseSet<Operation *, 4> *survivingOps = nullptr;
};

} // namespace

LogicalResult MultiOpPatternRewriteDriver::simplifyLocally(
    ArrayRef<Operation *> ops, bool *changed,
    llvm::SmallDenseSet<Operation *, 4> *surviving, Region *scope) {
  auto cleanup = llvm::make_scope_exit([&]() { survivingOps = nullptr; });
  if (surviving) {
    survivingOps = surviving;
    survivingOps->clear();
    survivingOps->insert(ops.begin(), ops.end());
  }

  if (strictMode != GreedyRewriteStrictness::AnyOp) {
    strictModeFilteredOps.clear();
    strictModeFilteredOps.insert(ops.begin(), ops.end());
  }

  assert(scope && "scope is mandatory");
  this->scope.clear();
  this->scope.insert(scope);

  if (changed)
    *changed = false;
  worklist.clear();
  worklistMap.clear();
  for (Operation *op : ops)
    addSingleOpToWorklist(op);

  // These are scratch vectors used in the folding loop below.
  SmallVector<Value, 8> originalOperands, resultValues;
  int64_t numRewrites = 0;
  while (!worklist.empty() &&
         (numRewrites < config.maxNumRewrites ||
          config.maxNumRewrites == GreedyRewriteConfig::kNoLimit)) {
    Operation *op = popFromWorklist();

    // Nulls get added to the worklist when operations are removed, ignore
    // them.
    if (op == nullptr)
      continue;

    assert((strictMode == GreedyRewriteStrictness::AnyOp ||
            strictModeFilteredOps.contains(op)) &&
           "unexpected op was inserted under strict mode");

    // If the operation is trivially dead - remove it.
    if (isOpTriviallyDead(op)) {
      notifyOperationRemoved(op);
      op->erase();
      if (changed)
        *changed = true;
      continue;
    }

    // Collects all the operands and result uses of the given `op` into work
    // list. Also remove `op` and nested ops from worklist.
    originalOperands.assign(op->operand_begin(), op->operand_end());
    auto preReplaceAction = [&](Operation *op) {
      // Add the operands to the worklist for visitation.
      addOperandsToWorklist(originalOperands);

      // Add all the users of the result to the worklist so we make sure
      // to revisit them.
      for (Value result : op->getResults()) {
        for (Operation *userOp : result.getUsers())
          addToWorklist(userOp);
      }

      notifyOperationRemoved(op);
    };

    // Add the given operation generated by the folder to the worklist.
    auto processGeneratedConstants = [this](Operation *op) {
      notifyOperationInserted(op);
    };

    // Try to fold this op.
    bool inPlaceUpdate;
    if (succeeded(folder.tryToFold(op, processGeneratedConstants,
                                   preReplaceAction, &inPlaceUpdate))) {
      if (changed)
        *changed = true;
      if (!inPlaceUpdate) {
        // Op has been erased.
        continue;
      }
    }

    // Try to match one of the patterns. The rewriter is automatically
    // notified of any necessary changes, so there is nothing else to do
    // here.
    if (succeeded(matcher.matchAndRewrite(op, *this))) {
      if (changed)
        *changed = true;
      ++numRewrites;
    }
  }

  return success(worklist.empty());
}

/// Find the region that is the closest common ancestor of all given ops.
static Region *findCommonAncestor(ArrayRef<Operation *> ops) {
  assert(!ops.empty() && "expected at least one op");
  // Fast path in case there is only one op.
  if (ops.size() == 1)
    return ops.front()->getParentRegion();

  Region *region = ops.front()->getParentRegion();
  ops = ops.drop_front();
  int sz = ops.size();
  llvm::BitVector remainingOps(sz, true);
  do {
    int pos = -1;
    // Iterate over all remaining ops.
    while ((pos = remainingOps.find_first_in(pos + 1, sz)) != -1) {
      // Is this op contained in `region`?
      if (region->findAncestorOpInRegion(*ops[pos]))
        remainingOps.reset(pos);
    }
    if (remainingOps.none())
      break;
  } while ((region = region->getParentRegion()));
  assert(region && "could not find common parent region");
  return region;
}

LogicalResult
mlir::applyOpPatternsAndFold(ArrayRef<Operation *> ops,
                             const FrozenRewritePatternSet &patterns,
                             GreedyRewriteStrictness strictMode, bool *changed,
                             bool *allErased, Region *scope) {
  if (ops.empty()) {
    if (changed)
      *changed = false;
    if (allErased)
      *allErased = true;
    return success();
  }

  if (!scope) {
    // Compute scope if none was provided.
    scope = findCommonAncestor(ops);
  } else {
    // If a scope was provided, make sure that all ops are in scope.
#ifndef NDEBUG
    bool allOpsInScope = llvm::all_of(ops, [&](Operation *op) {
      return static_cast<bool>(scope->findAncestorOpInRegion(*op));
    });
    assert(allOpsInScope && "ops must be within the specified scope");
#endif // NDEBUG
  }

  // Start the pattern driver.
  MultiOpPatternRewriteDriver driver(ops.front()->getContext(), patterns,
                                     strictMode);
  llvm::SmallDenseSet<Operation *, 4> surviving;
  LogicalResult converged = driver.simplifyLocally(
      ops, changed, allErased ? &surviving : nullptr, /*scope=*/scope);
  if (allErased)
    *allErased = surviving.empty();
  LLVM_DEBUG(if (failed(converged)) {
    llvm::dbgs() << "The pattern rewrite did not converge after "
                 << GreedyRewriteConfig().maxNumRewrites << " rewrites";
  });
  return converged;
}
