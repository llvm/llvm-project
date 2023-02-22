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
/// applies the locally optimal patterns.
///
/// This abstract class manages the worklist and contains helper methods for
/// rewriting ops on the worklist. Derived classes specify how ops are added
/// to the worklist in the beginning.
class GreedyPatternRewriteDriver : public PatternRewriter,
                                   public RewriterBase::Listener {
protected:
  explicit GreedyPatternRewriteDriver(MLIRContext *ctx,
                                      const FrozenRewritePatternSet &patterns,
                                      const GreedyRewriteConfig &config);

  /// Add the given operation to the worklist.
  void addSingleOpToWorklist(Operation *op);

  /// Add the given operation and its ancestors to the worklist.
  void addToWorklist(Operation *op);

  /// Notify the driver that the specified operation may have been modified
  /// in-place. The operation is added to the worklist.
  void finalizeRootUpdate(Operation *op) override;

  /// Notify the driver that the specified operation was inserted. Update the
  /// worklist as needed: The operation is enqueued depending on scope and
  /// strict mode.
  void notifyOperationInserted(Operation *op) override;

  /// Notify the driver that the specified operation was removed. Update the
  /// worklist as needed: The operation and its children are removed from the
  /// worklist.
  void notifyOperationRemoved(Operation *op) override;

  /// Notify the driver that the specified operation was replaced. Update the
  /// worklist as needed: New users are added enqueued.
  void notifyOperationReplaced(Operation *op, ValueRange replacement) override;

  /// Process ops until the worklist is empty or `config.maxNumRewrites` is
  /// reached. Return `true` if any IR was changed.
  bool processWorklist();

  /// The worklist for this transformation keeps track of the operations that
  /// need to be revisited, plus their index in the worklist.  This allows us to
  /// efficiently remove operations from the worklist when they are erased, even
  /// if they aren't the root of a pattern.
  std::vector<Operation *> worklist;
  DenseMap<Operation *, unsigned> worklistMap;

  /// Non-pattern based folder for operations.
  OperationFolder folder;

  /// Configuration information for how to simplify.
  const GreedyRewriteConfig config;

  /// The list of ops we are restricting our rewrites to. These include the
  /// supplied set of ops as well as new ops created while rewriting those ops
  /// depending on `strictMode`. This set is not maintained when
  /// `config.strictMode` is GreedyRewriteStrictness::AnyOp.
  llvm::SmallDenseSet<Operation *, 4> strictModeFilteredOps;

private:
  /// Look over the provided operands for any defining operations that should
  /// be re-added to the worklist. This function should be called when an
  /// operation is modified or removed, as it may trigger further
  /// simplifications.
  void addOperandsToWorklist(ValueRange operands);

  /// Pop the next operation from the worklist.
  Operation *popFromWorklist();

  /// Notify the driver that the given block was created.
  void notifyBlockCreated(Block *block) override;

  /// For debugging only: Notify the driver of a pattern match failure.
  LogicalResult
  notifyMatchFailure(Location loc,
                     function_ref<void(Diagnostic &)> reasonCallback) override;

  /// If the specified operation is in the worklist, remove it.
  void removeFromWorklist(Operation *op);

#ifndef NDEBUG
  /// A logger used to emit information during the application process.
  llvm::ScopedPrinter logger{llvm::dbgs()};
#endif

  /// The low-level pattern applicator.
  PatternApplicator matcher;
};
} // namespace

GreedyPatternRewriteDriver::GreedyPatternRewriteDriver(
    MLIRContext *ctx, const FrozenRewritePatternSet &patterns,
    const GreedyRewriteConfig &config)
    : PatternRewriter(ctx), folder(ctx), config(config), matcher(patterns) {
  worklist.reserve(64);

  // Apply a simple cost model based solely on pattern benefit.
  matcher.applyDefaultCostModel();

  // Set up listener.
  setListener(this);
}

bool GreedyPatternRewriteDriver::processWorklist() {
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

  // These are scratch vectors used in the folding loop below.
  SmallVector<Value, 8> originalOperands;

  bool changed = false;
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
      logger.startLine() << "Processing operation : '" << op->getName() << "'("
                         << op << ") {\n";
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

  return changed;
}

void GreedyPatternRewriteDriver::addToWorklist(Operation *op) {
  // Gather potential ancestors while looking for a "scope" parent region.
  SmallVector<Operation *, 8> ancestors;
  Region *region = nullptr;
  do {
    ancestors.push_back(op);
    region = op->getParentRegion();
    if (config.scope == region) {
      // Scope (can be `nullptr`) was reached. Stop traveral and enqueue ops.
      for (Operation *op : ancestors)
        addSingleOpToWorklist(op);
      return;
    }
    if (region == nullptr)
      return;
  } while ((op = region->getParentOp()));
}

void GreedyPatternRewriteDriver::addSingleOpToWorklist(Operation *op) {
  if (config.strictMode == GreedyRewriteStrictness::AnyOp ||
      strictModeFilteredOps.contains(op)) {
    // Check to see if the worklist already contains this op.
    if (worklistMap.count(op))
      return;

    worklistMap[op] = worklist.size();
    worklist.push_back(op);
  }
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

void GreedyPatternRewriteDriver::notifyBlockCreated(Block *block) {
  if (config.listener)
    config.listener->notifyBlockCreated(block);
}

void GreedyPatternRewriteDriver::notifyOperationInserted(Operation *op) {
  LLVM_DEBUG({
    logger.startLine() << "** Insert  : '" << op->getName() << "'(" << op
                       << ")\n";
  });
  if (config.listener)
    config.listener->notifyOperationInserted(op);
  if (config.strictMode == GreedyRewriteStrictness::ExistingAndNewOps)
    strictModeFilteredOps.insert(op);
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
  if (config.listener)
    config.listener->notifyOperationRemoved(op);

  addOperandsToWorklist(op->getOperands());
  op->walk([this](Operation *operation) {
    removeFromWorklist(operation);
    folder.notifyRemoval(operation);
  });

  if (config.strictMode != GreedyRewriteStrictness::AnyOp)
    strictModeFilteredOps.erase(op);
}

void GreedyPatternRewriteDriver::notifyOperationReplaced(
    Operation *op, ValueRange replacement) {
  LLVM_DEBUG({
    logger.startLine() << "** Replace : '" << op->getName() << "'(" << op
                       << ")\n";
  });
  if (config.listener)
    config.listener->notifyOperationReplaced(op, replacement);
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
  if (config.listener)
    return config.listener->notifyMatchFailure(loc, reasonCallback);
  return failure();
}

//===----------------------------------------------------------------------===//
// RegionPatternRewriteDriver
//===----------------------------------------------------------------------===//

namespace {
/// This driver simplfies all ops in a region.
class RegionPatternRewriteDriver : public GreedyPatternRewriteDriver {
public:
  explicit RegionPatternRewriteDriver(MLIRContext *ctx,
                                      const FrozenRewritePatternSet &patterns,
                                      const GreedyRewriteConfig &config,
                                      Region &regions);

  /// Simplify ops inside `region` and simplify the region itself. Return
  /// success if the transformation converged.
  LogicalResult simplify() &&;

private:
  /// The region that is simplified.
  Region &region;
};
} // namespace

RegionPatternRewriteDriver::RegionPatternRewriteDriver(
    MLIRContext *ctx, const FrozenRewritePatternSet &patterns,
    const GreedyRewriteConfig &config, Region &region)
    : GreedyPatternRewriteDriver(ctx, patterns, config), region(region) {
  // Populate strict mode ops.
  if (config.strictMode != GreedyRewriteStrictness::AnyOp) {
    region.walk([&](Operation *op) { strictModeFilteredOps.insert(op); });
  }
}

LogicalResult RegionPatternRewriteDriver::simplify() && {
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
      region.walk([&](Operation *op) {
        if (!insertKnownConstant(op))
          addToWorklist(op);
      });
    } else {
      // Add all nested operations to the worklist in preorder.
      region.walk<WalkOrder::PreOrder>([&](Operation *op) {
        if (!insertKnownConstant(op)) {
          worklist.push_back(op);
          return WalkResult::advance();
        }
        return WalkResult::skip();
      });

      // Reverse the list so our pop-back loop processes them in-order.
      std::reverse(worklist.begin(), worklist.end());
      // Remember the reverse index.
      for (size_t i = 0, e = worklist.size(); i != e; ++i)
        worklistMap[worklist[i]] = i;
    }

    changed = processWorklist();

    // After applying patterns, make sure that the CFG of each of the regions
    // is kept up to date.
    if (config.enableRegionSimplification)
      changed |= succeeded(simplifyRegions(*this, region));
  } while (changed);

  // Whether the rewrite converges, i.e. wasn't changed in the last iteration.
  return success(!changed);
}

LogicalResult
mlir::applyPatternsAndFoldGreedily(Region &region,
                                   const FrozenRewritePatternSet &patterns,
                                   GreedyRewriteConfig config) {
  // The top-level operation must be known to be isolated from above to
  // prevent performing canonicalizations on operations defined at or above
  // the region containing 'op'.
  assert(region.getParentOp()->hasTrait<OpTrait::IsIsolatedFromAbove>() &&
         "patterns can only be applied to operations IsolatedFromAbove");

  // Set scope if not specified.
  if (!config.scope)
    config.scope = &region;

  // Start the pattern driver.
  RegionPatternRewriteDriver driver(region.getContext(), patterns, config,
                                    region);
  LogicalResult converged = std::move(driver).simplify();
  LLVM_DEBUG(if (failed(converged)) {
    llvm::dbgs() << "The pattern rewrite did not converge after scanning "
                 << config.maxIterations << " times\n";
  });
  return converged;
}

//===----------------------------------------------------------------------===//
// MultiOpPatternRewriteDriver
//===----------------------------------------------------------------------===//

namespace {
/// This driver simplfies a list of ops.
class MultiOpPatternRewriteDriver : public GreedyPatternRewriteDriver {
public:
  explicit MultiOpPatternRewriteDriver(
      MLIRContext *ctx, const FrozenRewritePatternSet &patterns,
      const GreedyRewriteConfig &config, ArrayRef<Operation *> ops,
      llvm::SmallDenseSet<Operation *, 4> *survivingOps = nullptr);

  /// Simplify `ops`. Return `success` if the transformation converged.
  LogicalResult simplify(ArrayRef<Operation *> ops, bool *changed = nullptr) &&;

private:
  void notifyOperationRemoved(Operation *op) override {
    GreedyPatternRewriteDriver::notifyOperationRemoved(op);
    if (survivingOps)
      survivingOps->erase(op);
  }

  /// An optional set of ops that survived the rewrite. This set is populated
  /// at the beginning of `simplifyLocally` with the inititally provided list
  /// of ops.
  llvm::SmallDenseSet<Operation *, 4> *const survivingOps = nullptr;
};
} // namespace

MultiOpPatternRewriteDriver::MultiOpPatternRewriteDriver(
    MLIRContext *ctx, const FrozenRewritePatternSet &patterns,
    const GreedyRewriteConfig &config, ArrayRef<Operation *> ops,
    llvm::SmallDenseSet<Operation *, 4> *survivingOps)
    : GreedyPatternRewriteDriver(ctx, patterns, config),
      survivingOps(survivingOps) {
  if (config.strictMode != GreedyRewriteStrictness::AnyOp)
    strictModeFilteredOps.insert(ops.begin(), ops.end());

  if (survivingOps) {
    survivingOps->clear();
    survivingOps->insert(ops.begin(), ops.end());
  }
}

LogicalResult MultiOpPatternRewriteDriver::simplify(ArrayRef<Operation *> ops,
                                                    bool *changed) && {
  // Populate the initial worklist.
  for (Operation *op : ops)
    addSingleOpToWorklist(op);

  // Process ops on the worklist.
  bool result = processWorklist();
  if (changed)
    *changed = result;

  return success(worklist.empty());
}

/// Find the region that is the closest common ancestor of all given ops.
///
/// Note: This function returns `nullptr` if there is a top-level op among the
/// given list of ops.
static Region *findCommonAncestor(ArrayRef<Operation *> ops) {
  assert(!ops.empty() && "expected at least one op");
  // Fast path in case there is only one op.
  if (ops.size() == 1)
    return ops.front()->getParentRegion();

  Region *region = ops.front()->getParentRegion();
  ops = ops.drop_front();
  int sz = ops.size();
  llvm::BitVector remainingOps(sz, true);
  while (region) {
    int pos = -1;
    // Iterate over all remaining ops.
    while ((pos = remainingOps.find_first_in(pos + 1, sz)) != -1) {
      // Is this op contained in `region`?
      if (region->findAncestorOpInRegion(*ops[pos]))
        remainingOps.reset(pos);
    }
    if (remainingOps.none())
      break;
    region = region->getParentRegion();
  }
  return region;
}

LogicalResult mlir::applyOpPatternsAndFold(
    ArrayRef<Operation *> ops, const FrozenRewritePatternSet &patterns,
    GreedyRewriteConfig config, bool *changed, bool *allErased) {
  if (ops.empty()) {
    if (changed)
      *changed = false;
    if (allErased)
      *allErased = true;
    return success();
  }

  // Determine scope of rewrite.
  if (!config.scope) {
    // Compute scope if none was provided. The scope will remain `nullptr` if
    // there is a top-level op among `ops`.
    config.scope = findCommonAncestor(ops);
  } else {
    // If a scope was provided, make sure that all ops are in scope.
#ifndef NDEBUG
    bool allOpsInScope = llvm::all_of(ops, [&](Operation *op) {
      return static_cast<bool>(config.scope->findAncestorOpInRegion(*op));
    });
    assert(allOpsInScope && "ops must be within the specified scope");
#endif // NDEBUG
  }

  // Start the pattern driver.
  llvm::SmallDenseSet<Operation *, 4> surviving;
  MultiOpPatternRewriteDriver driver(ops.front()->getContext(), patterns,
                                     config, ops,
                                     allErased ? &surviving : nullptr);
  LogicalResult converged = std::move(driver).simplify(ops, changed);
  if (allErased)
    *allErased = surviving.empty();
  LLVM_DEBUG(if (failed(converged)) {
    llvm::dbgs() << "The pattern rewrite did not converge after "
                 << config.maxNumRewrites << " rewrites";
  });
  return converged;
}
