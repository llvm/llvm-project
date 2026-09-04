//===- GreedyPatternRewriteDriver.cpp - A greedy rewriter -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements mlir::applyPatternsGreedily.
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "mlir/Config/mlir-config.h"
#include "mlir/IR/Action.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Rewrite/PatternApplicator.h"
#include "mlir/Transforms/CSE.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/DebugLog.h"
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
  ExpensiveChecks(RewriterBase::Listener *listener, Operation *topLevel)
      : RewriterBase::ForwardingListener(listener), topLevel(topLevel) {}

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

  void notifyBlockErased(Block *block) override {
    RewriterBase::ForwardingListener::notifyBlockErased(block);

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

  void notifyOperationErased(Operation *op) override {
    RewriterBase::ForwardingListener::notifyOperationErased(op);
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
  LDBG() << "// *** IR Dump After Successful Folding ***\n"
         << OpWithFlags(op, OpPrintingFlags().elideLargeElementsAttrs());
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
  if (!map.insert({op, list.size()}).second)
    return;
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

/// Tracks which blocks are reachable from their region's entry block while
/// patterns rewrite the CFG, so that the driver can skip operations in blocks
/// that a rewrite disconnected. Notifications only record what changed; the
/// queries, which the driver issues between rewrites, bring the cache up to
/// date. `Block::isReachable` is not used because it traverses the region on
/// every call, while the driver queries once per processed operation. All
/// notifications are forwarded to the wrapped listener.
class ReachabilityListener : public RewriterBase::ForwardingListener {
public:
  ReachabilityListener(RewriterBase::Listener *listener, Region *scope)
      : RewriterBase::ForwardingListener(listener), scope(scope) {}

  /// Return whether `op` is reachable: its block is reachable from the entry
  /// block of its region, and so is every containing block up to `scope`.
  /// Must be called between rewrites, when the IR is in a consistent state.
  bool isReachable(Operation *op);

  /// Drop all cached reachability information.
  void clear() { regionStates.clear(); }

  void notifyBlockInserted(Block *block, Region *previous,
                           Region::iterator previousIt) override;
  void notifyBlockErased(Block *block) override;
  void notifyOperationInserted(Operation *op,
                               OpBuilder::InsertPoint previous) override;
  void notifyOperationModified(Operation *op) override;
  void notifyOperationErased(Operation *op) override;

private:
  /// The reachability cache of one region.
  struct RegionState {
    /// The entry block at the time of the traversal. The traversal starts
    /// from the entry block, so a different entry block invalidates the cache.
    Block *entry = nullptr;
    /// The reachable blocks, mapped to their successors within the region as
    /// of the last query. The snapshots let the next query tell lost edges
    /// from added ones.
    DenseMap<Block *, SmallVector<Block *, 2>> reachable;
    /// Blocks of `reachable` whose reachability could not be decided
    /// incrementally. A query for one of them traverses the region. Marking
    /// and traversing both cost O(blocks), so a lost edge into a loop whose
    /// target keeps other predecessors costs O(blocks) per rewrite if one of
    /// the marked blocks is queried before the next change. Bottom-up
    /// processing has usually popped the operations of these blocks already;
    /// top-down processing pops them next.
    llvm::SmallDenseSet<Block *, 4> uncertain;
    /// Blocks whose terminator changed since the last query, in notification
    /// order, so that the cache evolves deterministically.
    llvm::SmallSetVector<Block *, 4> changedBlocks;
    /// Blocks erased from or moved out of the region since the last query.
    /// Their pointers are only compared, never dereferenced.
    llvm::SmallDenseSet<Block *, 4> removedBlocks;

    /// Rebuild the cache from the entry block of `region`.
    void traverse(Region *region);
    /// Add `roots` and every block reachable from them to `reachable`. The
    /// added blocks are uncertain if `isUncertain` is set.
    void addReachable(SmallVector<Block *> roots, bool isUncertain);
    /// Mark `block` and every block behind it in the snapshots uncertain.
    void markUncertain(Block *block);
    /// Apply the pending changes, if any. On failure, the state is
    /// inconsistent and the region must be traversed again.
    LogicalResult applyChanges();
  };

  /// Record that the terminator of `block` changed. Only tracked regions
  /// record changes, so no pointer to a block of an untracked region is kept.
  void markChanged(Block *block) {
    if (!block)
      return;
    auto it = regionStates.find(block->getParent());
    if (it != regionStates.end())
      it->second.changedBlocks.insert(block);
  }
  /// Record that `block` left `region`.
  void markRemoved(Region *region, Block *block) {
    auto it = regionStates.find(region);
    if (it != regionStates.end())
      it->second.removedBlocks.insert(block);
  }
  bool isBlockReachable(Block *block);

  /// The tracked regions. A region without an entry is traversed by the next
  /// query that needs it.
  DenseMap<Region *, RegionState> regionStates;

  Region *const scope;
};

/// This is a worklist-driven driver for the PatternMatcher, which repeatedly
/// applies the locally optimal patterns.
///
/// This abstract class manages the worklist and contains helper methods for
/// rewriting ops on the worklist. Derived classes specify how ops are added
/// to the worklist in the beginning.
class GreedyPatternRewriteDriver : public RewriterBase::Listener {
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
  void notifyOperationInserted(Operation *op,
                               OpBuilder::InsertPoint previous) override;

  /// Notify the driver that the specified operation was removed. Update the
  /// worklist as needed: The operation and its children are removed from the
  /// worklist.
  void notifyOperationErased(Operation *op) override;

  /// Notify the driver that the specified operation was replaced. Update the
  /// worklist as needed: New users are added enqueued.
  void notifyOperationReplaced(Operation *op, ValueRange replacement) override;

  /// Process ops until the worklist is empty or `config.maxNumRewrites` is
  /// reached. Skip operations in unreachable blocks. Return `true` if any IR
  /// was changed.
  bool processWorklist();

  /// The pattern rewriter that is used for making IR modifications and is
  /// passed to rewrite patterns.
  PatternRewriter rewriter;

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

  /// Tracks block reachability while the worklist is processed. The rewriter
  /// notifies this listener, which forwards to the driver.
  ReachabilityListener reachabilityListener;

private:
  /// Look over the provided operands for any defining operations that should
  /// be re-added to the worklist. This function should be called when an
  /// operation is modified or removed, as it may trigger further
  /// simplifications.
  void addOperandsToWorklist(Operation *op);

  /// Notify the driver that the given block was inserted.
  void notifyBlockInserted(Block *block, Region *previous,
                           Region::iterator previousIt) override;

  /// Notify the driver that the given block is about to be removed.
  void notifyBlockErased(Block *block) override;

  /// For debugging only: Notify the driver of a pattern match failure.
  void
  notifyMatchFailure(Location loc,
                     function_ref<void(Diagnostic &)> reasonCallback) override;

#ifndef NDEBUG
  /// A raw output stream used to prefix the debug log.

  llvm::impl::raw_ldbg_ostream os{(Twine("[") + DEBUG_TYPE + ":1] ").str(),
                                  llvm::dbgs()};
  /// A logger used to emit information during the application process.
  llvm::ScopedPrinter logger{os};
#endif

  /// The low-level pattern applicator.
  PatternApplicator matcher;

#if MLIR_ENABLE_EXPENSIVE_PATTERN_API_CHECKS
  ExpensiveChecks expensiveChecks;
#endif // MLIR_ENABLE_EXPENSIVE_PATTERN_API_CHECKS
};
} // namespace

/// Return whether `op` may define successors of its block.
static bool mayDefineSuccessors(Operation *op) {
  return op->mightHaveTrait<OpTrait::IsTerminator>() ||
         op->getNumSuccessors() != 0;
}

/// Append the distinct successors of `block` that are in the same region to
/// `successors`. A successor in another region is invalid IR that cannot make
/// a block of this region reachable.
static void appendRegionSuccessors(Block *block,
                                   SmallVectorImpl<Block *> &successors) {
  for (Block *successor : block->getSuccessors())
    if (successor->getParent() == block->getParent() &&
        !llvm::is_contained(successors, successor))
      successors.push_back(successor);
}

void ReachabilityListener::RegionState::traverse(Region *region) {
  // Clear rather than reassign, to keep the allocations for the next
  // traversal.
  reachable.clear();
  uncertain.clear();
  changedBlocks.clear();
  removedBlocks.clear();
  entry = &region->front();
  addReachable({entry}, /*isUncertain=*/false);
}

void ReachabilityListener::RegionState::addReachable(SmallVector<Block *> roots,
                                                     bool isUncertain) {
  while (!roots.empty()) {
    auto [it, inserted] = reachable.try_emplace(roots.pop_back_val());
    if (!inserted)
      continue;
    appendRegionSuccessors(it->first, it->second);
    if (isUncertain)
      uncertain.insert(it->first);
    roots.append(it->second);
  }
}

void ReachabilityListener::RegionState::markUncertain(Block *block) {
  SmallVector<Block *> worklist{block};
  while (!worklist.empty()) {
    Block *current = worklist.pop_back_val();
    auto it = reachable.find(current);
    if (it == reachable.end() || !uncertain.insert(current).second)
      continue;
    worklist.append(it->second);
  }
}

LogicalResult ReachabilityListener::RegionState::applyChanges() {
  if (changedBlocks.empty() && removedBlocks.empty())
    return success();
  // Blocks are dropped after the loop, so that the snapshot of a lost block
  // remains available to every changed block that lost an edge to it.
  SmallVector<Block *> unreachable;
  for (Block *changed : changedBlocks) {
    // A removed block is no longer part of the region, and a change in an
    // unreachable block cannot make any block reachable. Neither can a change
    // in a block that has no predecessors left, unless it is the entry. Its
    // added edges must not extend the reachable set; the block that lost the
    // edge to it drops it or marks it uncertain, using its snapshot.
    if (removedBlocks.contains(changed) ||
        (changed != entry && changed->hasNoPredecessors()))
      continue;
    auto it = reachable.find(changed);
    if (it == reachable.end())
      continue;
    SmallVector<Block *, 2> old = std::move(it->second);
    SmallVector<Block *, 2> current;
    appendRegionSuccessors(changed, current);
    for (Block *lost : old) {
      if (llvm::is_contained(current, lost))
        continue;
      // The successors of a reachable block are reachable, so `lost` has a
      // snapshot unless the cache is inconsistent. This is not expected to
      // happen; the failure is a safety net that forces a traversal.
      auto lostIt = reachable.find(lost);
      if (lostIt == reachable.end())
        return failure();
      // `lost` and every block behind it may have become unreachable. When
      // `lost` is gone or has no predecessors left, it is unreachable; if all
      // of its cached successors are still successors of `changed`, nothing
      // else changed. This is the shape left by folding a constant conditional
      // branch and by merging a block into its predecessor. Any other lost
      // edge leaves the blocks behind `lost` uncertain until a query for one
      // of them traverses the region.
      bool isGone = removedBlocks.contains(lost) || lost->hasNoPredecessors();
      if (isGone && llvm::all_of(lostIt->second, [&](Block *successor) {
            return llvm::is_contained(current, successor);
          })) {
        unreachable.push_back(lost);
        continue;
      }
      markUncertain(lost);
    }
    // Added edges can only make more blocks reachable. Blocks reached from an
    // uncertain block are uncertain as well.
    SmallVector<Block *> roots;
    for (Block *successor : current)
      if (!reachable.contains(successor))
        roots.push_back(successor);
    // Nothing was inserted into `reachable` since `it` was looked up.
    it->second = std::move(current);
    addReachable(std::move(roots), uncertain.contains(changed));
  }
  // Drop every block that left the region, so that no stale key can match a
  // block inserted later at the same address.
  unreachable.append(removedBlocks.begin(), removedBlocks.end());
  for (Block *block : unreachable) {
    reachable.erase(block);
    uncertain.erase(block);
  }
  changedBlocks.clear();
  removedBlocks.clear();
  return success();
}

bool ReachabilityListener::isBlockReachable(Block *block) {
  Region *region = block->getParent();
  // A detached block is treated as reachable, as is the entry block.
  if (!region || block == &region->front())
    return true;
  if (block->hasNoPredecessors())
    return false;
  auto [it, inserted] = regionStates.try_emplace(region);
  RegionState &state = it->second;
  bool isStale = inserted || state.entry != &region->front();
  if (isStale || failed(state.applyChanges()) ||
      state.uncertain.contains(block))
    state.traverse(region);
  bool result = state.reachable.contains(block);
#if MLIR_ENABLE_EXPENSIVE_PATTERN_API_CHECKS
  assert(result == region->front().isReachable(block) &&
         "reachability cache disagrees with a traversal");
#endif // MLIR_ENABLE_EXPENSIVE_PATTERN_API_CHECKS
  return result;
}

bool ReachabilityListener::isReachable(Operation *op) {
  // A nested region can be locally reachable while its owning operation is in
  // an unreachable block, so check every containing block up to the scope.
  while (Block *block = op->getBlock()) {
    if (!isBlockReachable(block))
      return false;
    Region *region = block->getParent();
    if (!region || region == scope)
      break;
    op = region->getParentOp();
  }
  return true;
}

void ReachabilityListener::notifyBlockInserted(Block *block, Region *previous,
                                               Region::iterator previousIt) {
  Region *region = block->getParent();
  // A block moved within its region keeps its reachability, unless it becomes
  // the entry block, which the next query detects. A block that is new to the
  // region is unreachable until a recorded terminator change adds an edge to
  // it, but a stale entry from an earlier removal cannot be trusted.
  if (previous != region) {
    if (previous)
      markRemoved(previous, block);
    auto it = regionStates.find(region);
    if (it != regionStates.end() && (it->second.removedBlocks.contains(block) ||
                                     it->second.reachable.contains(block)))
      regionStates.erase(it);
  }
  RewriterBase::ForwardingListener::notifyBlockInserted(block, previous,
                                                        previousIt);
}

void ReachabilityListener::notifyBlockErased(Block *block) {
  markRemoved(block->getParent(), block);
  RewriterBase::ForwardingListener::notifyBlockErased(block);
}

void ReachabilityListener::notifyOperationInserted(
    Operation *op, OpBuilder::InsertPoint previous) {
  if (mayDefineSuccessors(op)) {
    markChanged(op->getBlock());
    if (previous.isSet())
      markChanged(previous.getBlock());
  }
  RewriterBase::ForwardingListener::notifyOperationInserted(op, previous);
}

void ReachabilityListener::notifyOperationModified(Operation *op) {
  if (mayDefineSuccessors(op))
    markChanged(op->getBlock());
  RewriterBase::ForwardingListener::notifyOperationModified(op);
}

void ReachabilityListener::notifyOperationErased(Operation *op) {
  if (mayDefineSuccessors(op))
    markChanged(op->getBlock());
  // Do not retain keys for regions that are about to be destroyed.
  for (Region &region : op->getRegions())
    regionStates.erase(&region);
  RewriterBase::ForwardingListener::notifyOperationErased(op);
}

GreedyPatternRewriteDriver::GreedyPatternRewriteDriver(
    MLIRContext *ctx, const FrozenRewritePatternSet &patterns,
    const GreedyRewriteConfig &config)
    : rewriter(ctx), config(config),
      reachabilityListener(this, config.getScope()), matcher(patterns)
#if MLIR_ENABLE_EXPENSIVE_PATTERN_API_CHECKS
      // clang-format off
      , expensiveChecks(
          /*listener=*/&reachabilityListener,
          /*topLevel=*/config.getScope() ? config.getScope()->getParentOp()
                                         : nullptr)
// clang-format on
#endif // MLIR_ENABLE_EXPENSIVE_PATTERN_API_CHECKS
{
  // Apply a simple cost model based solely on pattern benefit.
  matcher.applyDefaultCostModel();

  // Set up listener.
#if MLIR_ENABLE_EXPENSIVE_PATTERN_API_CHECKS
  // Chain notifications through the debug handler, reachability listener, and
  // worklist driver, in that order.
  rewriter.setListener(&expensiveChecks);
#else
  rewriter.setListener(&reachabilityListener);
#endif // MLIR_ENABLE_EXPENSIVE_PATTERN_API_CHECKS
}

bool GreedyPatternRewriteDriver::processWorklist() {
  // The cache is only maintained while this worklist pass runs. Between
  // passes, the IR changes without notifications, e.g. in region
  // simplification and in the unreachable-block sweep.
  llvm::scope_exit clearReachability([&] { reachabilityListener.clear(); });

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
         (numRewrites < config.getMaxNumRewrites() ||
          config.getMaxNumRewrites() == GreedyRewriteConfig::kNoLimit)) {
    auto *op = worklist.pop();
    if (!reachabilityListener.isReachable(op)) {
      LLVM_DEBUG(logger.startLine() << "Skipping unreachable operation : '"
                                    << op->getName() << "'(" << op << ")\n");
      continue;
    }

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
      rewriter.eraseOp(op);
      changed = true;

      LLVM_DEBUG(logResultWithLine("success", "operation is trivially dead"));
      continue;
    }

    // Try to fold this op. Do not fold constant ops. That would lead to an
    // infinite folding loop, as every constant op would be folded to an
    // Attribute and then immediately be rematerialized as a constant op, which
    // is then put on the worklist.
    if (config.isFoldingEnabled() && !op->hasTrait<OpTrait::ConstantLike>()) {
      SmallVector<OpFoldResult> foldResults;
      if (succeeded(op->fold(foldResults))) {
        LLVM_DEBUG(logResultWithLine("success", "operation was folded"));
#ifndef NDEBUG
        Operation *dumpRootOp = getDumpRootOp(op);
#endif // NDEBUG
        if (foldResults.empty()) {
          // Folding modified the op in place without going through the
          // rewriter.
          // Notify both the reachability listener and the worklist driver.
          reachabilityListener.notifyOperationModified(op);
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
        OpBuilder::InsertionGuard g(rewriter);
        rewriter.setInsertionPoint(op);
        SmallVector<Value> replacements;
        bool materializationSucceeded = true;
        for (auto [ofr, resultType] :
             llvm::zip_equal(foldResults, op->getResultTypes())) {
          if (auto value = dyn_cast<Value>(ofr)) {
            assert(value.getType() == resultType &&
                   "folder produced value of incorrect type");
            replacements.push_back(value);
            continue;
          }
          // Materialize Attributes as SSA values.
          Operation *constOp = op->getDialect()->materializeConstant(
              rewriter, cast<Attribute>(ofr), resultType, op->getLoc());

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
              rewriter.eraseOp(op);
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
          rewriter.replaceOp(op, replacements);
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
    auto canApplyCallback = [&](const Pattern &pattern) {
      LLVM_DEBUG({
        logger.getOStream() << "\n";
        logger.startLine() << "* Pattern " << pattern.getDebugName() << " : '"
                           << op->getName() << " -> (";
        llvm::interleaveComma(pattern.getGeneratedOps(), logger.getOStream());
        logger.getOStream() << ")' {\n";
        logger.indent();
      });
      if (RewriterBase::Listener *listener = config.getListener())
        listener->notifyPatternBegin(pattern, op);
      return true;
    };
    function_ref<bool(const Pattern &)> canApply = canApplyCallback;
    auto onFailureCallback = [&](const Pattern &pattern) {
      LLVM_DEBUG(logResult("failure", "pattern failed to match"));
      if (RewriterBase::Listener *listener = config.getListener())
        listener->notifyPatternEnd(pattern, failure());
    };
    function_ref<void(const Pattern &)> onFailure = onFailureCallback;
    auto onSuccessCallback = [&](const Pattern &pattern) {
      LLVM_DEBUG(logResult("success", "pattern applied successfully"));
      if (RewriterBase::Listener *listener = config.getListener())
        listener->notifyPatternEnd(pattern, success());
      return success();
    };
    function_ref<LogicalResult(const Pattern &)> onSuccess = onSuccessCallback;

#ifdef NDEBUG
    // Optimization: PatternApplicator callbacks are not needed when running in
    // optimized mode and without a listener.
    if (!config.getListener()) {
      canApply = nullptr;
      onFailure = nullptr;
      onSuccess = nullptr;
    }
#endif // NDEBUG

#if MLIR_ENABLE_EXPENSIVE_PATTERN_API_CHECKS
    if (config.getScope()) {
      expensiveChecks.computeFingerPrints(config.getScope()->getParentOp());
    }
    llvm::scope_exit clearFingerprints([&]() { expensiveChecks.clear(); });
#endif // MLIR_ENABLE_EXPENSIVE_PATTERN_API_CHECKS

    LogicalResult matchResult =
        matcher.matchAndRewrite(op, rewriter, canApply, onFailure, onSuccess);

    if (succeeded(matchResult)) {
      LLVM_DEBUG(logResultWithLine("success", "at least one pattern matched"));
#if MLIR_ENABLE_EXPENSIVE_PATTERN_API_CHECKS
      expensiveChecks.notifyRewriteSuccess();
#endif // MLIR_ENABLE_EXPENSIVE_PATTERN_API_CHECKS
      changed = true;
      ++numRewrites;
    } else {
      LLVM_DEBUG(logResultWithLine("failure", "all patterns failed to match"));
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
    if (config.getScope() == region) {
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
  if (config.getStrictness() == GreedyRewriteStrictness::AnyOp ||
      strictModeFilteredOps.contains(op))
    worklist.push(op);
}

void GreedyPatternRewriteDriver::notifyBlockInserted(
    Block *block, Region *previous, Region::iterator previousIt) {
  if (RewriterBase::Listener *listener = config.getListener())
    listener->notifyBlockInserted(block, previous, previousIt);
}

void GreedyPatternRewriteDriver::notifyBlockErased(Block *block) {
  if (RewriterBase::Listener *listener = config.getListener())
    listener->notifyBlockErased(block);
}

void GreedyPatternRewriteDriver::notifyOperationInserted(
    Operation *op, OpBuilder::InsertPoint previous) {
  LLVM_DEBUG({
    logger.startLine() << "** Insert  : '" << op->getName() << "'(" << op
                       << ")\n";
  });
  if (RewriterBase::Listener *listener = config.getListener())
    listener->notifyOperationInserted(op, previous);
  if (config.getStrictness() == GreedyRewriteStrictness::ExistingAndNewOps)
    strictModeFilteredOps.insert(op);
  addToWorklist(op);
}

void GreedyPatternRewriteDriver::notifyOperationModified(Operation *op) {
  LLVM_DEBUG({
    logger.startLine() << "** Modified: '" << op->getName() << "'(" << op
                       << ")\n";
  });
  if (RewriterBase::Listener *listener = config.getListener())
    listener->notifyOperationModified(op);
  addToWorklist(op);
}

void GreedyPatternRewriteDriver::addOperandsToWorklist(Operation *op) {
  for (Value operand : op->getOperands()) {
    // If this operand currently has at most 2 users, add its defining op to the
    // worklist. Indeed, after the op is deleted, then the operand will have at
    // most 1 user left. If it has 0 users left, it can be deleted too,
    // and if it has 1 user left, there may be further canonicalization
    // opportunities.
    if (!operand)
      continue;

    auto *defOp = operand.getDefiningOp();
    if (!defOp)
      continue;

    Operation *otherUser = nullptr;
    bool hasMoreThanTwoUses = false;
    for (auto *user : operand.getUsers()) {
      if (user == op || user == otherUser)
        continue;
      if (!otherUser) {
        otherUser = user;
        continue;
      }
      hasMoreThanTwoUses = true;
      break;
    }
    if (hasMoreThanTwoUses)
      continue;

    addToWorklist(defOp);
  }
}

void GreedyPatternRewriteDriver::notifyOperationErased(Operation *op) {
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
  if (Region *scope = config.getScope(); scope->getParentOp() == op)
    llvm_unreachable(
        "scope region must not be erased during greedy pattern rewrite");
#endif // NDEBUG

  if (RewriterBase::Listener *listener = config.getListener())
    listener->notifyOperationErased(op);

  addOperandsToWorklist(op);
  worklist.remove(op);

  if (config.getStrictness() != GreedyRewriteStrictness::AnyOp)
    strictModeFilteredOps.erase(op);
}

void GreedyPatternRewriteDriver::notifyOperationReplaced(
    Operation *op, ValueRange replacement) {
  LLVM_DEBUG({
    logger.startLine() << "** Replace : '" << op->getName() << "'(" << op
                       << ")\n";
  });
  if (RewriterBase::Listener *listener = config.getListener())
    listener->notifyOperationReplaced(op, replacement);
}

void GreedyPatternRewriteDriver::notifyMatchFailure(
    Location loc, function_ref<void(Diagnostic &)> reasonCallback) {
  LLVM_DEBUG({
    Diagnostic diag(loc, DiagnosticSeverity::Remark);
    reasonCallback(diag);
    logger.startLine() << "** Match Failure : " << diag.str() << "\n";
  });
  if (RewriterBase::Listener *listener = config.getListener())
    listener->notifyMatchFailure(loc, reasonCallback);
}

//===----------------------------------------------------------------------===//
// RegionPatternRewriteDriver
//===----------------------------------------------------------------------===//

namespace {
/// This driver simplfies all ops in a region. If a scope is set in the
/// config, the provided region must be within that scope.
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
  if (config.getStrictness() != GreedyRewriteStrictness::AnyOp) {
    region.walk([&](Operation *op) { strictModeFilteredOps.insert(op); });
  }
#ifndef NDEBUG
  // Verify that the region is within the configured scope (if any).
  if (Region *scope = config.getScope()) {
    Region *r = &region;
    while (r && r != scope)
      r = r->getParentRegion();
    assert(r && "provided region is not within the config scope");
  }
#endif
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
  MLIRContext *ctx = rewriter.getContext();
  do {
    // Check if the iteration limit was reached.
    if (++iteration > config.getMaxIterations() &&
        config.getMaxIterations() != GreedyRewriteConfig::kNoLimit)
      break;

    // New iteration: start with an empty worklist.
    worklist.clear();

    // `OperationFolder` CSE's constant ops (and may move them into parents
    // regions to enable more aggressive CSE'ing). It must notify the
    // reachability listener, which forwards to the driver, so that the
    // reachability cache sees every CFG change.
    OperationFolder folder(ctx, &reachabilityListener);
    auto insertKnownConstant = [&](Operation *op) {
      // Check for existing constants when populating the worklist. This avoids
      // accidentally reversing the constant order during processing.
      Attribute constValue;
      if (matchPattern(op, m_Constant(&constValue)))
        if (!folder.insertKnownConstant(op, constValue))
          return true;
      return false;
    };

    if (!config.getUseTopDownTraversal()) {
      // Add operations to the worklist in postorder.
      region.walk([&](Operation *op) {
        if (!config.isConstantCSEEnabled() || !insertKnownConstant(op))
          addToWorklist(op);
      });
    } else {
      // Add all nested operations to the worklist in preorder.
      region.walk<WalkOrder::PreOrder>([&](Operation *op) {
        if (!config.isConstantCSEEnabled() || !insertKnownConstant(op)) {
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
          continueRewrites = false;

          // Erase unreachable blocks
          // Operations like:
          //   %add = arith.addi %add, %add : i64
          // are legal in unreachable code. Unfortunately many patterns would be
          // unsafe to apply on such IR and can lead to crashes or infinite
          // loops. A rewrite can disconnect more blocks while processWorklist()
          // runs; processWorklist() skips their queued operations until the
          // next sweep, since repeated rewrites on such IR can prevent the
          // worklist pass from returning.
          continueRewrites |=
              succeeded(eraseUnreachableBlocks(rewriter, region));

          continueRewrites |= processWorklist();

          // After applying patterns, make sure that the CFG of each of the
          // regions is kept up to date.
          if (config.getRegionSimplificationLevel() !=
              GreedySimplifyRegionLevel::Disabled) {
            continueRewrites |= succeeded(simplifyRegions(
                rewriter, region,
                /*mergeBlocks=*/config.getRegionSimplificationLevel() ==
                    GreedySimplifyRegionLevel::Aggressive));
          }

          // Optionally run full CSE. If CSE changes the IR we iterate again so
          // that patterns can fire on the deduplicated operations.
          if (config.isCSEBetweenIterationsEnabled()) {
            DominanceInfo domInfo;
            bool cseChanged = false;
            eliminateCommonSubExpressions(rewriter, domInfo, region,
                                          &cseChanged);
            continueRewrites |= cseChanged;
          }
        },
        {&region}, iteration);
  } while (continueRewrites);

  if (changed)
    *changed = iteration > 1;

  // Whether the rewrite converges, i.e. wasn't changed in the last iteration.
  return success(!continueRewrites);
}

LogicalResult
mlir::applyPatternsGreedily(Region &region,
                            const FrozenRewritePatternSet &patterns,
                            GreedyRewriteConfig config, bool *changed) {
  // The top-level operation must be known to be isolated from above to
  // prevent performing canonicalizations on operations defined at or above
  // the region containing 'op'.
  assert(region.getParentOp()->hasTrait<OpTrait::IsIsolatedFromAbove>() &&
         "patterns can only be applied to operations IsolatedFromAbove");

  // Set scope if not specified.
  if (!config.getScope())
    config.setScope(&region);

#if MLIR_ENABLE_EXPENSIVE_PATTERN_API_CHECKS
  if (failed(verify(config.getScope()->getParentOp())))
    llvm::report_fatal_error(
        "greedy pattern rewriter input IR failed to verify");
#endif // MLIR_ENABLE_EXPENSIVE_PATTERN_API_CHECKS

  // Start the pattern driver.
  RegionPatternRewriteDriver driver(region.getContext(), patterns, config,
                                    region);
  LogicalResult converged = std::move(driver).simplify(changed);
  if (failed(converged))
    LDBG() << "The pattern rewrite did not converge after scanning "
           << config.getMaxIterations() << " times";
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
  void notifyOperationErased(Operation *op) override {
    GreedyPatternRewriteDriver::notifyOperationErased(op);
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
  if (config.getStrictness() != GreedyRewriteStrictness::AnyOp)
    strictModeFilteredOps.insert_range(ops);

  if (survivingOps) {
    survivingOps->clear();
    survivingOps->insert_range(ops);
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

LogicalResult mlir::applyOpPatternsGreedily(
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
  if (!config.getScope()) {
    // Compute scope if none was provided. The scope will remain `nullptr` if
    // there is a top-level op among `ops`.
    config.setScope(findCommonAncestor(ops));
  } else {
    // If a scope was provided, make sure that all ops are in scope.
#ifndef NDEBUG
    bool allOpsInScope = llvm::all_of(ops, [&](Operation *op) {
      return static_cast<bool>(config.getScope()->findAncestorOpInRegion(*op));
    });
    assert(allOpsInScope && "ops must be within the specified scope");
#endif // NDEBUG
  }

#if MLIR_ENABLE_EXPENSIVE_PATTERN_API_CHECKS
  if (config.getScope() && failed(verify(config.getScope()->getParentOp())))
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
  if (failed(converged))
    LDBG() << "The pattern rewrite did not converge after "
           << config.getMaxNumRewrites() << " rewrites";
  return converged;
}
