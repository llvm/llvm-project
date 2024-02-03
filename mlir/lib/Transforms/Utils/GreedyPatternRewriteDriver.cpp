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

#include "mlir/Config/mlir-config.h"
#include "mlir/IR/Action.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Verifier.h"
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

#ifdef MLIR_GREEDY_REWRITE_RANDOMIZER_SEED
#include <random>
#endif // MLIR_GREEDY_REWRITE_RANDOMIZER_SEED

using namespace mlir;

#define DEBUG_TYPE "greedy-rewriter"

namespace {

//===----------------------------------------------------------------------===//
// Debugging Infrastructure
//===----------------------------------------------------------------------===//

#if MLIR_ENABLE_EXPENSIVE_PATTERN_API_CHECKS
/// A helper struct that performs various "expensive checks" to detect broken
/// rewrite patterns use the rewriter API incorrectly. A rewrite pattern is
/// broken if:
/// * IR does not verify after pattern application / folding.
/// * Pattern returns "failure" but the IR has changed.
/// * Pattern returns "success" but the IR has not changed.
///
/// This struct stores finger prints of ops to determine whether the IR has
/// changed or not.
struct ExpensiveChecks : public RewriterBase::ForwardingListener {
  ExpensiveChecks(RewriterBase::Listener *driver, Operation *topLevel)
      : RewriterBase::ForwardingListener(driver), topLevel(topLevel) {}

  /// Compute finger prints of the given op and its nested ops.
  void computeFingerPrints(Operation *topLevel) {
    this->topLevel = topLevel;
    this->topLevelFingerPrint.emplace(topLevel);
    topLevel->walk([&](Operation *op) {
      fingerprints.try_emplace(op, op, /*includeNested=*/false);
    });
  }

  /// Clear all finger prints.
  void clear() {
    topLevel = nullptr;
    topLevelFingerPrint.reset();
    fingerprints.clear();
  }

  void notifyRewriteSuccess() {
    if (!topLevel)
      return;

    // Make sure that the IR still verifies.
    if (failed(verify(topLevel)))
      llvm::report_fatal_error("IR failed to verify after pattern application");

    // Pattern application success => IR must have changed.
    OperationFingerPrint afterFingerPrint(topLevel);
    if (*topLevelFingerPrint == afterFingerPrint) {
      // Note: Run "mlir-opt -debug" to see which pattern is broken.
      llvm::report_fatal_error(
          "pattern returned success but IR did not change");
    }
    for (const auto &it : fingerprints) {
      // Skip top-level op, its finger print is never invalidated.
      if (it.first == topLevel)
        continue;
      // Note: Finger print computation may crash when an op was erased
      // without notifying the rewriter. (Run with ASAN to see where the op was
      // erased; the op was probably erased directly, bypassing the rewriter
      // API.) Finger print computation does may not crash if a new op was
      // created at the same memory location. (But then the finger print should
      // have changed.)
      if (it.second !=
          OperationFingerPrint(it.first, /*includeNested=*/false)) {
        // Note: Run "mlir-opt -debug" to see which pattern is broken.
        llvm::report_fatal_error("operation finger print changed");
      }
    }
  }

  void notifyRewriteFailure() {
    if (!topLevel)
      return;

    // Pattern application failure => IR must not have changed.
    OperationFingerPrint afterFingerPrint(topLevel);
    if (*topLevelFingerPrint != afterFingerPrint) {
      // Note: Run "mlir-opt -debug" to see which pattern is broken.
      llvm::report_fatal_error("pattern returned failure but IR did change");
    }
  }

  void notifyFoldingSuccess() {
    if (!topLevel)
      return;

    // Make sure that the IR still verifies.
    if (failed(verify(topLevel)))
      llvm::report_fatal_error("IR failed to verify after folding");
  }

protected:
  /// Invalidate the finger print of the given op, i.e., remove it from the map.
  void invalidateFingerPrint(Operation *op) { fingerprints.erase(op); }

  void notifyBlockRemoved(Block *block) override {
    RewriterBase::ForwardingListener::notifyBlockRemoved(block);

    // The block structure (number of blocks, types of block arguments, etc.)
    // is part of the fingerprint of the parent op.
    // TODO: The parent op fingerprint should also be invalidated when modifying
    // the block arguments of a block, but we do not have a
    // `notifyBlockModified` callback yet.
    invalidateFingerPrint(block->getParentOp());
  }

  void notifyOperationInserted(Operation *op,
                               OpBuilder::InsertPoint previous) override {
    RewriterBase::ForwardingListener::notifyOperationInserted(op, previous);
    invalidateFingerPrint(op->getParentOp());
  }

  void notifyOperationModified(Operation *op) override {
    RewriterBase::ForwardingListener::notifyOperationModified(op);
    invalidateFingerPrint(op);
  }

  void notifyOperationRemoved(Operation *op) override {
    RewriterBase::ForwardingListener::notifyOperationRemoved(op);
    op->walk([this](Operation *op) { invalidateFingerPrint(op); });
  }

  /// Operation finger prints to detect invalid pattern API usage. IR is checked
  /// against these finger prints after pattern application to detect cases
  /// where IR was modified directly, bypassing the rewriter API.
  DenseMap<Operation *, OperationFingerPrint> fingerprints;

  /// Top-level operation of the current greedy rewrite.
  Operation *topLevel = nullptr;

  /// Finger print of the top-level operation.
  std::optional<OperationFingerPrint> topLevelFingerPrint;
};
#endif // MLIR_ENABLE_EXPENSIVE_PATTERN_API_CHECKS

#ifndef NDEBUG
static Operation *getDumpRootOp(Operation *op) {
  // Dump the parent op so that materialized constants are visible. If the op
  // is a top-level op, dump it directly.
  if (Operation *parentOp = op->getParentOp())
    return parentOp;
  return op;
}
static void logSuccessfulFolding(Operation *op) {
  llvm::dbgs() << "// *** IR Dump After Successful Folding ***\n";
  op->dump();
  llvm::dbgs() << "\n\n";
}
#endif // NDEBUG

//===----------------------------------------------------------------------===//
// Worklist
//===----------------------------------------------------------------------===//

/// A LIFO worklist of operations with efficient removal and set semantics.
///
/// This class maintains a vector of operations and a mapping of operations to
/// positions in the vector, so that operations can be removed efficiently at
/// random. When an operation is removed, it is replaced with nullptr. Such
/// nullptr are skipped when pop'ing elements.
class Worklist {
public:
  Worklist();

  /// Clear the worklist.
  void clear();

  /// Return whether the worklist is empty.
  bool empty() const;

  /// Push an operation to the end of the worklist, unless the operation is
  /// already on the worklist.
  void push(Operation *op);

  /// Pop the an operation from the end of the worklist. Only allowed on
  /// non-empty worklists.
  Operation *pop();

  /// Remove an operation from the worklist.
  void remove(Operation *op);

  /// Reverse the worklist.
  void reverse();

protected:
  /// The worklist of operations.
  std::vector<Operation *> list;

  /// A mapping of operations to positions in `list`.
  DenseMap<Operation *, unsigned> map;
};

Worklist::Worklist() { list.reserve(64); }

void Worklist::clear() {
  list.clear();
  map.clear();
}

bool Worklist::empty() const {
  // Skip all nullptr.
  return !llvm::any_of(list,
                       [](Operation *op) { return static_cast<bool>(op); });
}

void Worklist::push(Operation *op) {
  assert(op && "cannot push nullptr to worklist");
  // Check to see if the worklist already contains this op.
  if (map.count(op))
    return;
  map[op] = list.size();
  list.push_back(op);
}

Operation *Worklist::pop() {
  assert(!empty() && "cannot pop from empty worklist");
  // Skip and remove all trailing nullptr.
  while (!list.back())
    list.pop_back();
  Operation *op = list.back();
  list.pop_back();
  map.erase(op);
  // Cleanup: Remove all trailing nullptr.
  while (!list.empty() && !list.back())
    list.pop_back();
  return op;
}

void Worklist::remove(Operation *op) {
  assert(op && "cannot remove nullptr from worklist");
  auto it = map.find(op);
  if (it != map.end()) {
    assert(list[it->second] == op && "malformed worklist data structure");
    list[it->second] = nullptr;
    map.erase(it);
  }
}

void Worklist::reverse() {
  std::reverse(list.begin(), list.end());
  for (size_t i = 0, e = list.size(); i != e; ++i)
    map[list[i]] = i;
}

#ifdef MLIR_GREEDY_REWRITE_RANDOMIZER_SEED
/// A worklist that pops elements at a random position. This worklist is for
/// testing/debugging purposes only. It can be used to ensure that lowering
/// pipelines work correctly regardless of the order in which ops are processed
/// by the GreedyPatternRewriteDriver.
class RandomizedWorklist : public Worklist {
public:
  RandomizedWorklist() : Worklist() {
    generator.seed(MLIR_GREEDY_REWRITE_RANDOMIZER_SEED);
  }

  /// Pop a random non-empty op from the worklist.
  Operation *pop() {
    Operation *op = nullptr;
    do {
      assert(!list.empty() && "cannot pop from empty worklist");
      int64_t pos = generator() % list.size();
      op = list[pos];
      list.erase(list.begin() + pos);
      for (int64_t i = pos, e = list.size(); i < e; ++i)
        map[list[i]] = i;
      map.erase(op);
    } while (!op);
    return op;
  }

private:
  std::minstd_rand0 generator;
};
#endif // MLIR_GREEDY_REWRITE_RANDOMIZER_SEED

//===----------------------------------------------------------------------===//
// GreedyPatternRewriteDriver
//===----------------------------------------------------------------------===//

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
  void notifyOperationModified(Operation *op) override;

  /// Notify the driver that the specified operation was inserted. Update the
  /// worklist as needed: The operation is enqueued depending on scope and
  /// strict mode.
  void notifyOperationInserted(Operation *op, InsertPoint previous) override;

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
  /// need to be (re)visited.
#ifdef MLIR_GREEDY_REWRITE_RANDOMIZER_SEED
  RandomizedWorklist worklist;
#else
  Worklist worklist;
#endif // MLIR_GREEDY_REWRITE_RANDOMIZER_SEED

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

  /// Notify the driver that the given block was inserted.
  void notifyBlockInserted(Block *block, Region *previous,
                           Region::iterator previousIt) override;

  /// Notify the driver that the given block is about to be removed.
  void notifyBlockRemoved(Block *block) override;

  /// For debugging only: Notify the driver of a pattern match failure.
  LogicalResult
  notifyMatchFailure(Location loc,
                     function_ref<void(Diagnostic &)> reasonCallback) override;

#ifndef NDEBUG
  /// A logger used to emit information during the application process.
  llvm::ScopedPrinter logger{llvm::dbgs()};
#endif

  /// The low-level pattern applicator.
  PatternApplicator matcher;

#if MLIR_ENABLE_EXPENSIVE_PATTERN_API_CHECKS
  ExpensiveChecks expensiveChecks;
#endif // MLIR_ENABLE_EXPENSIVE_PATTERN_API_CHECKS
};
} // namespace

GreedyPatternRewriteDriver::GreedyPatternRewriteDriver(
    MLIRContext *ctx, const FrozenRewritePatternSet &patterns,
    const GreedyRewriteConfig &config)
    : PatternRewriter(ctx), config(config), matcher(patterns)
#if MLIR_ENABLE_EXPENSIVE_PATTERN_API_CHECKS
      // clang-format off
      , expensiveChecks(
          /*driver=*/this,
          /*topLevel=*/config.scope ? config.scope->getParentOp() : nullptr)
// clang-format on
#endif // MLIR_ENABLE_EXPENSIVE_PATTERN_API_CHECKS
{
  // Apply a simple cost model based solely on pattern benefit.
  matcher.applyDefaultCostModel();

  // Set up listener.
#if MLIR_ENABLE_EXPENSIVE_PATTERN_API_CHECKS
  // Send IR notifications to the debug handler. This handler will then forward
  // all notifications to this GreedyPatternRewriteDriver.
  setListener(&expensiveChecks);
#else
  setListener(this);
#endif // MLIR_ENABLE_EXPENSIVE_PATTERN_API_CHECKS
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

  bool changed = false;
  int64_t numRewrites = 0;
  while (!worklist.empty() &&
         (numRewrites < config.maxNumRewrites ||
          config.maxNumRewrites == GreedyRewriteConfig::kNoLimit)) {
    auto *op = worklist.pop();

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
      eraseOp(op);
      changed = true;

      LLVM_DEBUG(logResultWithLine("success", "operation is trivially dead"));
      continue;
    }

    // Try to fold this op. Do not fold constant ops. That would lead to an
    // infinite folding loop, as every constant op would be folded to an
    // Attribute and then immediately be rematerialized as a constant op, which
    // is then put on the worklist.
    if (!op->hasTrait<OpTrait::ConstantLike>()) {
      SmallVector<OpFoldResult> foldResults;
      if (succeeded(op->fold(foldResults))) {
        LLVM_DEBUG(logResultWithLine("success", "operation was folded"));
#ifndef NDEBUG
        Operation *dumpRootOp = getDumpRootOp(op);
#endif // NDEBUG
        if (foldResults.empty()) {
          // Op was modified in-place.
          notifyOperationModified(op);
          changed = true;
          LLVM_DEBUG(logSuccessfulFolding(dumpRootOp));
#if MLIR_ENABLE_EXPENSIVE_PATTERN_API_CHECKS
          expensiveChecks.notifyFoldingSuccess();
#endif // MLIR_ENABLE_EXPENSIVE_PATTERN_API_CHECKS
          continue;
        }

        // Op results can be replaced with `foldResults`.
        assert(foldResults.size() == op->getNumResults() &&
               "folder produced incorrect number of results");
        OpBuilder::InsertionGuard g(*this);
        setInsertionPoint(op);
        SmallVector<Value> replacements;
        bool materializationSucceeded = true;
        for (auto [ofr, resultType] :
             llvm::zip_equal(foldResults, op->getResultTypes())) {
          if (auto value = ofr.dyn_cast<Value>()) {
            assert(value.getType() == resultType &&
                   "folder produced value of incorrect type");
            replacements.push_back(value);
            continue;
          }
          // Materialize Attributes as SSA values.
          Operation *constOp = op->getDialect()->materializeConstant(
              *this, ofr.get<Attribute>(), resultType, op->getLoc());

          if (!constOp) {
            // If materialization fails, cleanup any operations generated for
            // the previous results.
            llvm::SmallDenseSet<Operation *> replacementOps;
            for (Value replacement : replacements) {
              assert(replacement.use_empty() &&
                     "folder reused existing op for one result but constant "
                     "materialization failed for another result");
              replacementOps.insert(replacement.getDefiningOp());
            }
            for (Operation *op : replacementOps) {
              eraseOp(op);
            }

            materializationSucceeded = false;
            break;
          }

          assert(constOp->hasTrait<OpTrait::ConstantLike>() &&
                 "materializeConstant produced op that is not a ConstantLike");
          assert(constOp->getResultTypes()[0] == resultType &&
                 "materializeConstant produced incorrect result type");
          replacements.push_back(constOp->getResult(0));
        }

        if (materializationSucceeded) {
          replaceOp(op, replacements);
          changed = true;
          LLVM_DEBUG(logSuccessfulFolding(dumpRootOp));
#if MLIR_ENABLE_EXPENSIVE_PATTERN_API_CHECKS
          expensiveChecks.notifyFoldingSuccess();
#endif // MLIR_ENABLE_EXPENSIVE_PATTERN_API_CHECKS
          continue;
        }
      }
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
#else
    function_ref<bool(const Pattern &)> canApply = {};
    function_ref<void(const Pattern &)> onFailure = {};
    function_ref<LogicalResult(const Pattern &)> onSuccess = {};
#endif

#if MLIR_ENABLE_EXPENSIVE_PATTERN_API_CHECKS
    if (config.scope) {
      expensiveChecks.computeFingerPrints(config.scope->getParentOp());
    }
    auto clearFingerprints =
        llvm::make_scope_exit([&]() { expensiveChecks.clear(); });
#endif // MLIR_ENABLE_EXPENSIVE_PATTERN_API_CHECKS

    LogicalResult matchResult =
        matcher.matchAndRewrite(op, *this, canApply, onFailure, onSuccess);

    if (succeeded(matchResult)) {
      LLVM_DEBUG(logResultWithLine("success", "pattern matched"));
#if MLIR_ENABLE_EXPENSIVE_PATTERN_API_CHECKS
      expensiveChecks.notifyRewriteSuccess();
#endif // MLIR_ENABLE_EXPENSIVE_PATTERN_API_CHECKS
      changed = true;
      ++numRewrites;
    } else {
      LLVM_DEBUG(logResultWithLine("failure", "pattern failed to match"));
#if MLIR_ENABLE_EXPENSIVE_PATTERN_API_CHECKS
      expensiveChecks.notifyRewriteFailure();
#endif // MLIR_ENABLE_EXPENSIVE_PATTERN_API_CHECKS
    }
  }

  return changed;
}

void GreedyPatternRewriteDriver::addToWorklist(Operation *op) {
  assert(op && "expected valid op");
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
      strictModeFilteredOps.contains(op))
    worklist.push(op);
}

void GreedyPatternRewriteDriver::notifyBlockInserted(
    Block *block, Region *previous, Region::iterator previousIt) {
  if (config.listener)
    config.listener->notifyBlockInserted(block, previous, previousIt);
}

void GreedyPatternRewriteDriver::notifyBlockRemoved(Block *block) {
  if (config.listener)
    config.listener->notifyBlockRemoved(block);
}

void GreedyPatternRewriteDriver::notifyOperationInserted(Operation *op,
                                                         InsertPoint previous) {
  LLVM_DEBUG({
    logger.startLine() << "** Insert  : '" << op->getName() << "'(" << op
                       << ")\n";
  });
  if (config.listener)
    config.listener->notifyOperationInserted(op, previous);
  if (config.strictMode == GreedyRewriteStrictness::ExistingAndNewOps)
    strictModeFilteredOps.insert(op);
  addToWorklist(op);
}

void GreedyPatternRewriteDriver::notifyOperationModified(Operation *op) {
  LLVM_DEBUG({
    logger.startLine() << "** Modified: '" << op->getName() << "'(" << op
                       << ")\n";
  });
  if (config.listener)
    config.listener->notifyOperationModified(op);
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

#ifndef NDEBUG
  // Only ops that are within the configured scope are added to the worklist of
  // the greedy pattern rewriter. Moreover, the parent op of the scope region is
  // the part of the IR that is taken into account for the "expensive checks".
  // A greedy pattern rewrite is not allowed to erase the parent op of the scope
  // region, as that would break the worklist handling and the expensive checks.
  if (config.scope && config.scope->getParentOp() == op)
    llvm_unreachable(
        "scope region must not be erased during greedy pattern rewrite");
#endif // NDEBUG

  if (config.listener)
    config.listener->notifyOperationRemoved(op);

  addOperandsToWorklist(op->getOperands());
  worklist.remove(op);

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
  LogicalResult simplify(bool *changed) &&;

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

namespace {
class GreedyPatternRewriteIteration
    : public tracing::ActionImpl<GreedyPatternRewriteIteration> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GreedyPatternRewriteIteration)
  GreedyPatternRewriteIteration(ArrayRef<IRUnit> units, int64_t iteration)
      : tracing::ActionImpl<GreedyPatternRewriteIteration>(units),
        iteration(iteration) {}
  static constexpr StringLiteral tag = "GreedyPatternRewriteIteration";
  void print(raw_ostream &os) const override {
    os << "GreedyPatternRewriteIteration(" << iteration << ")";
  }

private:
  int64_t iteration = 0;
};
} // namespace

LogicalResult RegionPatternRewriteDriver::simplify(bool *changed) && {
  bool continueRewrites = false;
  int64_t iteration = 0;
  MLIRContext *ctx = getContext();
  do {
    // Check if the iteration limit was reached.
    if (++iteration > config.maxIterations &&
        config.maxIterations != GreedyRewriteConfig::kNoLimit)
      break;

    // New iteration: start with an empty worklist.
    worklist.clear();

    // `OperationFolder` CSE's constant ops (and may move them into parents
    // regions to enable more aggressive CSE'ing).
    OperationFolder folder(getContext(), this);
    auto insertKnownConstant = [&](Operation *op) {
      // Check for existing constants when populating the worklist. This avoids
      // accidentally reversing the constant order during processing.
      Attribute constValue;
      if (matchPattern(op, m_Constant(&constValue)))
        if (!folder.insertKnownConstant(op, constValue))
          return true;
      return false;
    };

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
          addToWorklist(op);
          return WalkResult::advance();
        }
        return WalkResult::skip();
      });

      // Reverse the list so our pop-back loop processes them in-order.
      worklist.reverse();
    }

    ctx->executeAction<GreedyPatternRewriteIteration>(
        [&] {
          continueRewrites = processWorklist();

          // After applying patterns, make sure that the CFG of each of the
          // regions is kept up to date.
          if (config.enableRegionSimplification)
            continueRewrites |= succeeded(simplifyRegions(*this, region));
        },
        {&region}, iteration);
  } while (continueRewrites);

  if (changed)
    *changed = iteration > 1;

  // Whether the rewrite converges, i.e. wasn't changed in the last iteration.
  return success(!continueRewrites);
}

LogicalResult
mlir::applyPatternsAndFoldGreedily(Region &region,
                                   const FrozenRewritePatternSet &patterns,
                                   GreedyRewriteConfig config, bool *changed) {
  // The top-level operation must be known to be isolated from above to
  // prevent performing canonicalizations on operations defined at or above
  // the region containing 'op'.
  assert(region.getParentOp()->hasTrait<OpTrait::IsIsolatedFromAbove>() &&
         "patterns can only be applied to operations IsolatedFromAbove");

  // Set scope if not specified.
  if (!config.scope)
    config.scope = &region;

#if MLIR_ENABLE_EXPENSIVE_PATTERN_API_CHECKS
  if (failed(verify(config.scope->getParentOp())))
    llvm::report_fatal_error(
        "greedy pattern rewriter input IR failed to verify");
#endif // MLIR_ENABLE_EXPENSIVE_PATTERN_API_CHECKS

  // Start the pattern driver.
  RegionPatternRewriteDriver driver(region.getContext(), patterns, config,
                                    region);
  LogicalResult converged = std::move(driver).simplify(changed);
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

#if MLIR_ENABLE_EXPENSIVE_PATTERN_API_CHECKS
  if (config.scope && failed(verify(config.scope->getParentOp())))
    llvm::report_fatal_error(
        "greedy pattern rewriter input IR failed to verify");
#endif // MLIR_ENABLE_EXPENSIVE_PATTERN_API_CHECKS

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
