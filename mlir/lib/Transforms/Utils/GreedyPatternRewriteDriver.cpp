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
#include "llvm/ADT/DenseMap.h"
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

  /// Add the given operation to the worklist.
  void addToWorklist(Operation *op);

  /// Pop the next operation from the worklist.
  Operation *popFromWorklist();

  /// If the specified operation is in the worklist, remove it.
  void removeFromWorklist(Operation *op);

protected:
  // Implement the hook for inserting operations, and make sure that newly
  // inserted ops are added to the worklist for processing.
  void notifyOperationInserted(Operation *op) override;

  // Look over the provided operands for any defining operations that should
  // be re-added to the worklist. This function should be called when an
  // operation is modified or removed, as it may trigger further
  // simplifications.
  template <typename Operands>
  void addToWorklist(Operands &&operands);

  // If an operation is about to be removed, make sure it is not in our
  // worklist anymore because we'd get dangling references to it.
  void notifyOperationRemoved(Operation *op) override;

  // When the root of a pattern is about to be replaced, it can trigger
  // simplifications to its users - make sure to add them to the worklist
  // before the root is changed.
  void notifyRootReplaced(Operation *op) override;

  /// PatternRewriter hook for erasing a dead operation.
  void eraseOp(Operation *op) override;

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

private:
  /// Configuration information for how to simplify.
  GreedyRewriteConfig config;

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
  unsigned iteration = 0;
  do {
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
      for (auto &region : regions)
        region.walk<WalkOrder::PreOrder>([&](Operation *op) {
          if (!insertKnownConstant(op))
            worklist.push_back(op);
        });

      // Reverse the list so our pop-back loop processes them in-order.
      std::reverse(worklist.begin(), worklist.end());
      // Remember the reverse index.
      for (size_t i = 0, e = worklist.size(); i != e; ++i)
        worklistMap[worklist[i]] = i;
    }

    // These are scratch vectors used in the folding loop below.
    SmallVector<Value, 8> originalOperands, resultValues;

    changed = false;
    while (!worklist.empty()) {
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
        addToWorklist(originalOperands);

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
      changed |= succeeded(matchResult);
    }

    // After applying patterns, make sure that the CFG of each of the regions
    // is kept up to date.
    if (config.enableRegionSimplification)
      changed |= succeeded(simplifyRegions(*this, regions));
  } while (changed &&
           (iteration++ < config.maxIterations ||
            config.maxIterations == GreedyRewriteConfig::kNoIterationLimit));

  // Whether the rewrite converges, i.e. wasn't changed in the last iteration.
  return !changed;
}

void GreedyPatternRewriteDriver::addToWorklist(Operation *op) {
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

template <typename Operands>
void GreedyPatternRewriteDriver::addToWorklist(Operands &&operands) {
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
  addToWorklist(op->getOperands());
  op->walk([this](Operation *operation) {
    removeFromWorklist(operation);
    folder.notifyRemoval(operation);
  });
}

void GreedyPatternRewriteDriver::notifyRootReplaced(Operation *op) {
  LLVM_DEBUG({
    logger.startLine() << "** Replace : '" << op->getName() << "'(" << op
                       << ")\n";
  });
  for (auto result : op->getResults())
    for (auto *user : result.getUsers())
      addToWorklist(user);
}

void GreedyPatternRewriteDriver::eraseOp(Operation *op) {
  LLVM_DEBUG({
    logger.startLine() << "** Erase   : '" << op->getName() << "'(" << op
                       << ")\n";
  });
  PatternRewriter::eraseOp(op);
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
    llvm::dbgs() << "The pattern rewrite doesn't converge after scanning "
                 << config.maxIterations << " times\n";
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
                                  const FrozenRewritePatternSet &patterns)
      : PatternRewriter(ctx), matcher(patterns), folder(ctx) {
    // Apply a simple cost model based solely on pattern benefit.
    matcher.applyDefaultCostModel();
  }

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

} // namespace

/// Performs the rewrites and folding only on `op`. The simplification
/// converges if the op is erased as a result of being folded, replaced, or
/// becoming dead, or no more changes happen in an iteration. Returns success if
/// the rewrite converges in `maxIterations`. `erased` is set to true if `op`
/// gets erased.
LogicalResult OpPatternRewriteDriver::simplifyLocally(Operation *op,
                                                      int maxIterations,
                                                      bool &erased) {
  bool changed = false;
  erased = false;
  opErasedViaPatternRewrites = false;
  int iterations = 0;
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
  } while (changed &&
           (++iterations < maxIterations ||
            maxIterations == GreedyRewriteConfig::kNoIterationLimit));

  // Whether the rewrite converges, i.e. wasn't changed in the last iteration.
  return failure(changed);
}

//===----------------------------------------------------------------------===//
// MultiOpPatternRewriteDriver
//===----------------------------------------------------------------------===//

namespace {

/// This is a specialized GreedyPatternRewriteDriver to apply patterns and
/// perform folding for a supplied set of ops. It repeatedly simplifies while
/// restricting the rewrites to only the provided set of ops or optionally
/// to those directly affected by it (result users or operand providers).
class MultiOpPatternRewriteDriver : public GreedyPatternRewriteDriver {
public:
  explicit MultiOpPatternRewriteDriver(MLIRContext *ctx,
                                       const FrozenRewritePatternSet &patterns,
                                       bool strict)
      : GreedyPatternRewriteDriver(ctx, patterns, GreedyRewriteConfig()),
        strictMode(strict) {}

  bool simplifyLocally(ArrayRef<Operation *> op);

private:
  // Look over the provided operands for any defining operations that should
  // be re-added to the worklist. This function should be called when an
  // operation is modified or removed, as it may trigger further
  // simplifications. If `strict` is set to true, only ops in
  // `strictModeFilteredOps` are considered.
  template <typename Operands>
  void addOperandsToWorklist(Operands &&operands) {
    for (Value operand : operands) {
      if (auto *defOp = operand.getDefiningOp()) {
        if (!strictMode || strictModeFilteredOps.contains(defOp))
          addToWorklist(defOp);
      }
    }
  }

  void notifyOperationRemoved(Operation *op) override {
    GreedyPatternRewriteDriver::notifyOperationRemoved(op);
    if (strictMode)
      strictModeFilteredOps.erase(op);
  }

  /// If `strictMode` is true, any pre-existing ops outside of
  /// `strictModeFilteredOps` remain completely untouched by the rewrite driver.
  /// If `strictMode` is false, operations that use results of (or supply
  /// operands to) any rewritten ops stemming from the simplification of the
  /// provided ops are in turn simplified; any other ops still remain untouched
  /// (i.e., regardless of `strictMode`).
  bool strictMode = false;

  /// The list of ops we are restricting our rewrites to if `strictMode` is on.
  /// These include the supplied set of ops as well as new ops created while
  /// rewriting those ops. This set is not maintained when strictMode is off.
  llvm::SmallDenseSet<Operation *, 4> strictModeFilteredOps;
};

} // namespace

/// Performs the specified rewrites on `ops` while also trying to fold these ops
/// as well as any other ops that were in turn created due to these rewrite
/// patterns. Any pre-existing ops outside of `ops` remain completely
/// unmodified if `strictMode` is true. If `strictMode` is false, other
/// operations that use results of rewritten ops or supply operands to such ops
/// are in turn simplified; any other ops still remain unmodified (i.e.,
/// regardless of `strictMode`). Note that ops in `ops` could be erased as a
/// result of folding, becoming dead, or via pattern rewrites. Returns true if
/// at all any changes happened.
// Unlike `OpPatternRewriteDriver::simplifyLocally` which works on a single op
// or GreedyPatternRewriteDriver::simplify, this method just iterates until
// the worklist is empty. As our objective is to keep simplification "local",
// there is no strong rationale to re-add all operations into the worklist and
// rerun until an iteration changes nothing. If more widereaching simplification
// is desired, GreedyPatternRewriteDriver should be used.
bool MultiOpPatternRewriteDriver::simplifyLocally(ArrayRef<Operation *> ops) {
  if (strictMode) {
    strictModeFilteredOps.clear();
    strictModeFilteredOps.insert(ops.begin(), ops.end());
  }

  bool changed = false;
  worklist.clear();
  worklistMap.clear();
  for (Operation *op : ops)
    addToWorklist(op);

  // These are scratch vectors used in the folding loop below.
  SmallVector<Value, 8> originalOperands, resultValues;
  while (!worklist.empty()) {
    Operation *op = popFromWorklist();

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
      addOperandsToWorklist(originalOperands);

      // Add all the users of the result to the worklist so we make sure
      // to revisit them.
      for (Value result : op->getResults())
        for (Operation *userOp : result.getUsers()) {
          if (!strictMode || strictModeFilteredOps.contains(userOp))
            addToWorklist(userOp);
        }
      notifyOperationRemoved(op);
    };

    // Add the given operation generated by the folder to the worklist.
    auto processGeneratedConstants = [this](Operation *op) {
      // Newly created ops are also simplified -- these are also "local".
      addToWorklist(op);
      // When strict mode is off, we don't need to maintain
      // strictModeFilteredOps.
      if (strictMode)
        strictModeFilteredOps.insert(op);
    };

    // Try to fold this op.
    bool inPlaceUpdate;
    if (succeeded(folder.tryToFold(op, processGeneratedConstants,
                                   preReplaceAction, &inPlaceUpdate))) {
      changed = true;
      if (!inPlaceUpdate) {
        // Op has been erased.
        continue;
      }
    }

    // Try to match one of the patterns. The rewriter is automatically
    // notified of any necessary changes, so there is nothing else to do
    // here.
    changed |= succeeded(matcher.matchAndRewrite(op, *this));
  }

  return changed;
}

/// Rewrites only `op` using the supplied canonicalization patterns and
/// folding. `erased` is set to true if the op is erased as a result of being
/// folded, replaced, or dead.
LogicalResult mlir::applyOpPatternsAndFold(
    Operation *op, const FrozenRewritePatternSet &patterns, bool *erased) {
  // Start the pattern driver.
  GreedyRewriteConfig config;
  OpPatternRewriteDriver driver(op->getContext(), patterns);
  bool opErased;
  LogicalResult converged =
      driver.simplifyLocally(op, config.maxIterations, opErased);
  if (erased)
    *erased = opErased;
  LLVM_DEBUG(if (failed(converged)) {
    llvm::dbgs() << "The pattern rewrite doesn't converge after scanning "
                 << config.maxIterations << " times";
  });
  return converged;
}

bool mlir::applyOpPatternsAndFold(ArrayRef<Operation *> ops,
                                  const FrozenRewritePatternSet &patterns,
                                  bool strict) {
  if (ops.empty())
    return false;

  // Start the pattern driver.
  MultiOpPatternRewriteDriver driver(ops.front()->getContext(), patterns,
                                     strict);
  return driver.simplifyLocally(ops);
}
