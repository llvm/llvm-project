//===- CSE.cpp - Common Sub-expression Elimination ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements common sub-expression elimination as a library utility.
// The matching CSE pass is a thin wrapper over the APIs declared here.
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/CSE.h"

#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/RecyclingAllocator.h"
#include <deque>

using namespace mlir;

namespace {
struct SimpleOperationInfo : public llvm::DenseMapInfo<Operation *> {
  static unsigned getHashValue(const Operation *opC) {
    return OperationEquivalence::computeHash(
        const_cast<Operation *>(opC),
        /*hashOperands=*/OperationEquivalence::directHashValue,
        /*hashResults=*/OperationEquivalence::ignoreHashValue,
        OperationEquivalence::IgnoreLocations);
  }
  static bool isEqual(const Operation *lhsC, const Operation *rhsC) {
    auto *lhs = const_cast<Operation *>(lhsC);
    auto *rhs = const_cast<Operation *>(rhsC);
    if (lhs == rhs)
      return true;
    if (lhs == getTombstoneKey() || lhs == getEmptyKey() ||
        rhs == getTombstoneKey() || rhs == getEmptyKey())
      return false;
    return OperationEquivalence::isEquivalentTo(
        const_cast<Operation *>(lhsC), const_cast<Operation *>(rhsC),
        OperationEquivalence::IgnoreLocations);
  }
};
} // namespace

namespace {
/// Simple common sub-expression elimination.
class CSEDriver {
public:
  CSEDriver(RewriterBase &rewriter, DominanceInfo *domInfo)
      : rewriter(rewriter), domInfo(domInfo) {}

  /// Simplify all operations within the given op.
  ///
  /// When simplifying nested regions, \p markOpForDeletionInParent is used to
  /// notify the enclosing region that one of its directly nested ops may have
  /// become trivially dead.
  void
  simplify(Operation *op, bool *changed = nullptr,
           function_ref<void(Operation *)> markOpForDeletionInParent = nullptr);

  /// Simplify operations within the given region.
  ///
  /// When simplifying a nested region, \p markOpForDeletionInParent is used to
  /// notify the enclosing region that one of its directly nested ops may have
  /// become trivially dead.
  void
  simplify(Region &region, bool *changed = nullptr,
           function_ref<void(Operation *)> markOpForDeletionInParent = nullptr);

  int64_t getNumCSE() const { return numCSE; }
  int64_t getNumDCE() const { return numDCE; }

private:
  /// Shared implementation of operation elimination and scoped map definitions.
  using AllocatorTy = llvm::RecyclingAllocator<
      llvm::BumpPtrAllocator,
      llvm::ScopedHashTableVal<Operation *, Operation *>>;
  using ScopedMapTy = llvm::ScopedHashTable<Operation *, Operation *,
                                            SimpleOperationInfo, AllocatorTy>;

  /// Cache holding MemoryEffects information between two operations. The first
  /// operation is stored has the key. The second operation is stored inside a
  /// pair in the value. The pair also hold the MemoryEffects between those
  /// two operations. If the MemoryEffects is nullptr then we assume there is
  /// no operation with MemoryEffects::Write between the two operations.
  using MemEffectsCache =
      DenseMap<Operation *, std::pair<Operation *, MemoryEffects::Effect *>>;

  /// Represents a single entry in the depth first traversal of a CFG.
  struct CFGStackNode {
    CFGStackNode(ScopedMapTy &knownValues, DominanceInfoNode *node)
        : scope(knownValues), node(node), childIterator(node->begin()) {}

    /// Scope for the known values.
    ScopedMapTy::ScopeTy scope;

    DominanceInfoNode *node;
    DominanceInfoNode::const_iterator childIterator;

    /// If this node has been fully processed yet or not.
    bool processed = false;
  };

  /// Attempt to CSE \p op against an equivalent known operation.
  ///
  /// If CSE succeeds, uses of \p op that are valid to rewrite are replaced with
  /// the existing operation's results. If \p op then has no remaining uses,
  /// \p markCSEDuplicateForDeletion records it as a CSE duplicate that should
  /// be erased after the current traversal scope unwinds.
  LogicalResult
  simplifyOperation(ScopedMapTy &knownValues, Operation *op,
                    function_ref<void(Operation *)> markCSEDuplicateForDeletion,
                    bool hasSSADominance);
  /// Simplify operations in \p bb.
  ///
  /// \p markKnownDeadOpForDeletion records operations that are already
  /// trivially dead before CSE inspects them. \p markCSEDuplicateForDeletion
  /// records operations whose uses were replaced by CSE.
  bool
  simplifyBlock(ScopedMapTy &knownValues, Block *bb, bool hasSSADominance,
                function_ref<void(Operation *)> markKnownDeadOpForDeletion,
                function_ref<void(Operation *)> markCSEDuplicateForDeletion);
  /// Simplify operations in \p region.
  ///
  /// \p markKnownDeadOpForDeletionInParent is forwarded to nested regions so
  /// they can notify this region when deleting nested uses makes a parent
  /// operation trivially dead.
  bool simplifyRegion(
      ScopedMapTy &knownValues, Region &region,
      function_ref<void(Operation *)> markKnownDeadOpForDeletionInParent);

  /// Replace uses of \p op with \p existing and schedule \p op for deletion
  /// when all remaining uses can be removed safely.
  ///
  /// In regions with SSA dominance, all uses can be replaced immediately. In
  /// graph regions, only uses owned by operations that have not already been
  /// visited are rewritten. If no uses remain, \p markCSEDuplicateForDeletion
  /// records \p op as a duplicate produced by CSE.
  void replaceUsesAndDelete(
      ScopedMapTy &knownValues, Operation *op, Operation *existing,
      function_ref<void(Operation *)> markCSEDuplicateForDeletion,
      bool hasSSADominance);

  /// Check if there is side-effecting operations other than the given effect
  /// between the two operations.
  bool hasOtherSideEffectingOpInBetween(Operation *fromOp, Operation *toOp);

  /// A rewriter for modifying the IR.
  RewriterBase &rewriter;

  DominanceInfo *domInfo = nullptr;
  MemEffectsCache memEffectsCache;

  // Various statistics.
  int64_t numCSE = 0;
  int64_t numDCE = 0;
};
} // namespace

void CSEDriver::replaceUsesAndDelete(
    ScopedMapTy &knownValues, Operation *op, Operation *existing,
    function_ref<void(Operation *)> markCSEDuplicateForDeletion,
    bool hasSSADominance) {
  // If we find one then replace all uses of the current operation with the
  // existing one and mark it for deletion. We can only replace an operand in
  // an operation if it has not been visited yet.
  if (hasSSADominance) {
    // If the region has SSA dominance, then we are guaranteed to have not
    // visited any use of the current operation.
    // Replace all uses, but do not remove the operation yet.
    rewriter.replaceAllOpUsesWith(op, existing->getResults());
    markCSEDuplicateForDeletion(op);
  } else {
    // When the region does not have SSA dominance, we need to check if we
    // have visited a use before replacing any use.
    auto wasVisited = [&](OpOperand &operand) {
      return !knownValues.count(operand.getOwner());
    };
    if (auto *rewriteListener =
            dyn_cast_if_present<RewriterBase::Listener>(rewriter.getListener()))
      for (Value v : op->getResults())
        if (all_of(v.getUses(), wasVisited))
          rewriteListener->notifyOperationReplaced(op, existing);

    // Replace all uses, but do not remove the operation yet. This does not
    // notify the listener because the original op is not erased.
    rewriter.replaceUsesWithIf(op->getResults(), existing->getResults(),
                               wasVisited);

    // There may be some remaining uses of the operation.
    if (op->use_empty())
      markCSEDuplicateForDeletion(op);
  }

  // If the existing operation has an unknown location and the current
  // operation doesn't, then set the existing op's location to that of the
  // current op.
  if (isa<UnknownLoc>(existing->getLoc()) && !isa<UnknownLoc>(op->getLoc()))
    existing->setLoc(op->getLoc());

  ++numCSE;
}

bool CSEDriver::hasOtherSideEffectingOpInBetween(Operation *fromOp,
                                                 Operation *toOp) {
  assert(fromOp->getBlock() == toOp->getBlock());
  assert(hasEffect<MemoryEffects::Read>(fromOp) &&
         "expected read effect on fromOp");
  assert(hasEffect<MemoryEffects::Read>(toOp) &&
         "expected read effect on toOp");

  // Collect the read effects of fromOp. A write can only block CSE if it
  // can conflict with one of these reads.
  SmallVector<MemoryEffects::EffectInstance> readEffects;
  if (auto memOp = dyn_cast<MemoryEffectOpInterface>(fromOp)) {
    SmallVector<MemoryEffects::EffectInstance> fromEffects;
    memOp.getEffects(fromEffects);
    for (MemoryEffects::EffectInstance &e : fromEffects)
      if (isa<MemoryEffects::Read>(e.getEffect()))
        readEffects.push_back(e);
  }

  Operation *nextOp = fromOp->getNextNode();
  auto result =
      memEffectsCache.try_emplace(fromOp, std::make_pair(fromOp, nullptr));
  if (!result.second) {
    auto memEffectsCachePair = result.first->second;
    if (memEffectsCachePair.second == nullptr) {
      // No MemoryEffects::Write has been detected until the cached operation.
      // Continue looking from the cached operation to toOp.
      nextOp = memEffectsCachePair.first;
    } else {
      // MemoryEffects::Write has been detected before so there is no need to
      // check further.
      return true;
    }
  }
  while (nextOp && nextOp != toOp) {
    std::optional<SmallVector<MemoryEffects::EffectInstance>> effects =
        getEffectsRecursively(nextOp);
    if (!effects) {
      // TODO: Do we need to handle other effects generically?
      // If the operation does not implement the MemoryEffectOpInterface we
      // conservatively assume it writes.
      result.first->second =
          std::make_pair(nextOp, MemoryEffects::Write::get());
      return true;
    }

    for (const MemoryEffects::EffectInstance &effect : *effects) {
      if (isa<MemoryEffects::Write>(effect.getEffect())) {
        // A write on a resource disjoint from all read resources cannot
        // conflict with the reads being CSE'd.
        SideEffects::Resource *writeResource = effect.getResource();
        bool canConflict =
            llvm::any_of(readEffects, [&](const auto &readEffect) {
              SideEffects::Resource *readResource = readEffect.getResource();
              if (writeResource->isDisjointFrom(readResource))
                return false;
              // A pointer-based access to an addressable resource cannot
              // conflict with a non-addressable resource.
              if (readEffect.getValue() && !writeResource->isAddressable())
                return false;
              if (effect.getValue() && !readResource->isAddressable())
                return false;
              return true;
            });
        if (canConflict) {
          result.first->second = {nextOp, MemoryEffects::Write::get()};
          return true;
        }
      }
    }
    nextOp = nextOp->getNextNode();
  }
  result.first->second = std::make_pair(toOp, nullptr);
  return false;
}

/// Attempt to eliminate a redundant operation.
LogicalResult CSEDriver::simplifyOperation(
    ScopedMapTy &knownValues, Operation *op,
    function_ref<void(Operation *)> markCSEDuplicateForDeletion,
    bool hasSSADominance) {
  // Don't simplify terminator operations.
  if (op->hasTrait<OpTrait::IsTerminator>())
    return failure();

  // Don't simplify operations with regions that have multiple blocks.
  // TODO: We need additional tests to verify that we handle such IR correctly.
  if (!llvm::all_of(op->getRegions(),
                    [](Region &r) { return r.empty() || r.hasOneBlock(); }))
    return failure();

  // Some simple use case of operation with memory side-effect are dealt with
  // here. Operations with no side-effect are done after.
  if (!isMemoryEffectFree(op)) {
    // TODO: Only basic use case for operations with MemoryEffects::Read can be
    // eleminated now. More work needs to be done for more complicated patterns
    // and other side-effects.
    if (!hasSingleEffect<MemoryEffects::Read>(op))
      return failure();

    // Look for an existing definition for the operation.
    if (auto *existing = knownValues.lookup(op)) {
      if (existing->getBlock() == op->getBlock() &&
          !hasOtherSideEffectingOpInBetween(existing, op)) {
        // The operation that can be deleted has been reach with no
        // side-effecting operations in between the existing operation and
        // this one so we can remove the duplicate.
        replaceUsesAndDelete(knownValues, op, existing,
                             markCSEDuplicateForDeletion, hasSSADominance);
        return success();
      }
    }
    knownValues.insert(op, op);
    return failure();
  }

  // Look for an existing definition for the operation.
  if (auto *existing = knownValues.lookup(op)) {
    replaceUsesAndDelete(knownValues, op, existing, markCSEDuplicateForDeletion,
                         hasSSADominance);
    return success();
  }

  // Otherwise, we add this operation to the known values map.
  knownValues.insert(op, op);
  return failure();
}

bool CSEDriver::simplifyBlock(
    ScopedMapTy &knownValues, Block *bb, bool hasSSADominance,
    function_ref<void(Operation *)> markKnownDeadOpForDeletion,
    function_ref<void(Operation *)> markCSEDuplicateForDeletion) {
  bool changed = false;
  for (auto &op : llvm::make_early_inc_range(*bb)) {
    // If the operation is already trivially dead just add it to the erase list.
    // This also avoids calling `simplifyRegion` on dead region ops
    // unnecessarily.
    if (isOpTriviallyDead(&op)) {
      markKnownDeadOpForDeletion(&op);
      continue;
    }

    // Most operations don't have regions, so fast path that case.
    if (op.getNumRegions() != 0) {
      // If this operation is isolated above, we can't process nested regions
      // with the given 'knownValues' map. This would cause the insertion of
      // implicit captures in explicit capture only regions.
      if (op.mightHaveTrait<OpTrait::IsIsolatedFromAbove>()) {
        ScopedMapTy nestedKnownValues;
        for (auto &region : op.getRegions())
          changed |= simplifyRegion(nestedKnownValues, region,
                                    markKnownDeadOpForDeletion);
      } else {
        // Otherwise, process nested regions normally.
        for (auto &region : op.getRegions())
          changed |=
              simplifyRegion(knownValues, region, markKnownDeadOpForDeletion);
      }
    }

    if (succeeded(simplifyOperation(
            knownValues, &op, markCSEDuplicateForDeletion, hasSSADominance)))
      continue;
  }
  // Clear the MemoryEffects cache since its usage is by block only.
  memEffectsCache.clear();
  return changed;
}

bool CSEDriver::simplifyRegion(
    ScopedMapTy &knownValues, Region &region,
    function_ref<void(Operation *)> markKnownDeadOpForDeletionInParent) {
  // If the region is empty there is nothing to do.
  if (region.empty())
    return false;

  bool hasSSADominance = domInfo->hasSSADominance(&region);
  bool changed = false;

  // Operations that were already trivially dead when encountered during CSE.
  // These go through the DCE helper so deleting them can propagate deadness to
  // their producers, including producers in parent regions.
  SmallVector<Operation *> triviallyDeadOps;

  // Operations whose uses were replaced by CSE. Keep these separate from
  // triviallyDeadOps so CSE erasures are not counted as DCE erasures.
  SmallVector<Operation *> deadOpsAfterCSE;

  // Install a listener while erasing deferred ops so dominance caches for
  // nested regions owned by erased ops are invalidated before the operation
  // storage is destroyed.
  struct CSEErasureListener final : RewriterBase::ForwardingListener {
    CSEErasureListener(OpBuilder::Listener *listener, DominanceInfo &domInfo)
        : RewriterBase::ForwardingListener(listener), domInfo(domInfo) {}

    void notifyOperationErased(Operation *op) override {
      for (Region &region : op->getRegions())
        domInfo.invalidate(&region);
      RewriterBase::ForwardingListener::notifyOperationErased(op);
    }

    DominanceInfo &domInfo;
  };

  // Record an operation that is known to be trivially dead independently of
  // CSE. If the op belongs to the parent region, notify the parent traversal
  // instead of queuing it in this region.
  auto markKnownDeadOpForDeletion = [&](Operation *op) {
    LDBG(2) << "Marking operation for deletion: "
            << OpWithFlags(op, OpPrintingFlags().skipRegions());
    if (op->getParentRegion() != &region) {
      LDBG(2) << "Operation is not in the current region";
      if (markKnownDeadOpForDeletionInParent)
        markKnownDeadOpForDeletionInParent(op);
      return;
    }
    if (isOpTriviallyDead(op))
      triviallyDeadOps.push_back(op);
  };

  // Record a duplicate operation whose uses have been replaced by CSE. These
  // ops are erased after the current scoped known-values table has unwound.
  auto markCSEDuplicateForDeletion = [&](Operation *op) {
    LDBG(2) << "Marking CSE duplicate for deletion: "
            << OpWithFlags(op, OpPrintingFlags().skipRegions());
    deadOpsAfterCSE.push_back(op);
  };

  // Erase queued ops after the traversal scope has unwound. Known-dead ops use
  // the DCE helper so deadness propagates through operands; CSE duplicates are
  // erased separately so they remain classified as CSE erasures.
  auto eraseOpsToDelete = [&]() {
    OpBuilder::Listener *previousListener = rewriter.getListener();
    CSEErasureListener listener(previousListener, *domInfo);
    rewriter.setListener(&listener);
    int64_t erasedCount = eliminateTriviallyDeadOps(
        rewriter, region, triviallyDeadOps, markKnownDeadOpForDeletionInParent);
    numDCE += erasedCount;
    int64_t cseErasedCount = deadOpsAfterCSE.size();
    for (Operation *op : deadOpsAfterCSE)
      rewriter.eraseOp(op);
    rewriter.setListener(previousListener);
    return erasedCount != 0 || cseErasedCount != 0;
  };

  // If the region only contains one block, then simplify it directly.
  if (region.hasOneBlock()) {
    {
      ScopedMapTy::ScopeTy scope(knownValues);
      changed |= simplifyBlock(knownValues, &region.front(), hasSSADominance,
                               markKnownDeadOpForDeletion,
                               markCSEDuplicateForDeletion);
    }
    changed |= eraseOpsToDelete();
    return changed;
  }

  // If the region does not have dominanceInfo, then skip it.
  // TODO: Regions without SSA dominance should define a different
  // traversal order which is appropriate and can be used here.
  if (!hasSSADominance)
    return false;

  // Note, deque is being used here because there was significant performance
  // gains over vector when the container becomes very large due to the
  // specific access patterns. If/when these performance issues are no
  // longer a problem we can change this to vector. For more information see
  // the llvm mailing list discussion on this:
  // http://lists.llvm.org/pipermail/llvm-commits/Week-of-Mon-20120116/135228.html
  std::deque<std::unique_ptr<CFGStackNode>> stack;

  // Process the nodes of the dom tree for this region.
  stack.emplace_back(std::make_unique<CFGStackNode>(
      knownValues, domInfo->getRootNode(&region)));

  while (!stack.empty()) {
    auto &currentNode = stack.back();

    // Check to see if we need to process this node.
    if (!currentNode->processed) {
      currentNode->processed = true;
      changed |= simplifyBlock(knownValues, currentNode->node->getBlock(),
                               hasSSADominance, markKnownDeadOpForDeletion,
                               markCSEDuplicateForDeletion);
    }

    // Otherwise, check to see if we need to process a child node.
    if (currentNode->childIterator != currentNode->node->end()) {
      auto *childNode = *(currentNode->childIterator++);
      stack.emplace_back(
          std::make_unique<CFGStackNode>(knownValues, childNode));
    } else {
      // Finally, if the node and all of its children have been processed
      // then we delete the node.
      stack.pop_back();
    }
  }
  changed |= eraseOpsToDelete();
  return changed;
}

void CSEDriver::simplify(
    Operation *op, bool *changed,
    function_ref<void(Operation *)> markOpForDeletionInParent) {
  // Simplify all regions.
  ScopedMapTy knownValues;
  bool anyChanged = false;
  for (auto &region : op->getRegions())
    anyChanged |=
        simplifyRegion(knownValues, region, markOpForDeletionInParent);
  if (changed)
    *changed = anyChanged;
}

void CSEDriver::simplify(
    Region &region, bool *changed,
    function_ref<void(Operation *)> markOpForDeletionInParent) {
  ScopedMapTy knownValues;
  bool anyChanged =
      simplifyRegion(knownValues, region, markOpForDeletionInParent);
  if (changed)
    *changed = anyChanged;
}

void mlir::eliminateCommonSubExpressions(RewriterBase &rewriter,
                                         DominanceInfo &domInfo, Operation *op,
                                         bool *changed, int64_t *numCSE,
                                         int64_t *numDCE) {
  CSEDriver driver(rewriter, &domInfo);
  driver.simplify(op, changed);
  if (numCSE)
    *numCSE = driver.getNumCSE();
  if (numDCE)
    *numDCE = driver.getNumDCE();
}

void mlir::eliminateCommonSubExpressions(RewriterBase &rewriter,
                                         DominanceInfo &domInfo, Region &region,
                                         bool *changed) {
  CSEDriver driver(rewriter, &domInfo);
  driver.simplify(region, changed);
}
