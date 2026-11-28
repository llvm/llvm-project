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
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/RecyclingAllocator.h"
#include <deque>

using namespace mlir;

#define DEBUG_TYPE "cse"

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
  void simplify(Operation *op, bool *changed = nullptr);

  /// Simplify operations within the given region.
  void simplify(Region &region, bool *changed = nullptr);

  int64_t getNumCSE() const { return numCSE; }
  int64_t getNumDCE() const { return numDCE; }

private:
  /// Shared implementation of operation elimination and scoped map definitions.
  using AllocatorTy = llvm::RecyclingAllocator<
      llvm::BumpPtrAllocator,
      llvm::ScopedHashTableVal<Operation *, Operation *>>;
  using ScopedMapTy = llvm::ScopedHashTable<Operation *, Operation *,
                                            SimpleOperationInfo, AllocatorTy>;

  /// Index writes and unknown effects in block order, retaining each read's
  /// progress through the index for resource-specific queries.
  struct MemEffectsCache {
    SmallVector<Operation *> writes;
    struct ReadState {
      Operation *lastScanned = nullptr;
      unsigned nextWrite = 0;
      bool blocked = false;
    };
    DenseMap<Operation *, ReadState> reads;
    Operation *lastScanned = nullptr;
  };

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

  /// Attempt to eliminate a redundant operation. Returns success if the
  /// operation was marked for removal, failure otherwise.
  LogicalResult simplifyOperation(ScopedMapTy &knownValues, Operation *op,
                                  bool hasSSADominance,
                                  MemEffectsCache &memEffectsCache);
  void simplifyBlock(ScopedMapTy &knownValues, Block *bb, bool hasSSADominance);
  void simplifyRegion(ScopedMapTy &knownValues, Region &region);

  /// Erase opertion that were marked as dead during simplification, and remove
  /// their associated dominator trees.
  void eraseDeadOp(Operation *op);

  void replaceUsesAndDelete(ScopedMapTy &knownValues, Operation *op,
                            Operation *existing, bool hasSSADominance);

  /// Check if there is side-effecting operations other than the given effect
  /// between the two operations.
  bool hasOtherSideEffectingOpInBetween(Operation *fromOp, Operation *toOp,
                                        MemEffectsCache &memEffectsCache);

  /// A rewriter for modifying the IR.
  RewriterBase &rewriter;

  DominanceInfo *domInfo = nullptr;

  // Various statistics.
  int64_t numCSE = 0;
  int64_t numDCE = 0;
};
} // namespace

void CSEDriver::replaceUsesAndDelete(ScopedMapTy &knownValues, Operation *op,
                                     Operation *existing,
                                     bool hasSSADominance) {
  // If the existing operation has an unknown location and the current
  // operation doesn't, then set the existing op's location to that of the
  // current op.
  if (isa<UnknownLoc>(existing->getLoc()) && !isa<UnknownLoc>(op->getLoc()))
    existing->setLoc(op->getLoc());

  ++numCSE;

  // If we find one then replace all uses of the current operation with the
  // existing one and delete it.
  if (hasSSADominance) {
    // If the region has SSA dominance, then we are guaranteed to have not
    // visited any use of the current operation.
    // Replace all uses, but do not remove the operation yet.
    rewriter.replaceAllOpUsesWith(op, existing->getResults());
    eraseDeadOp(op);
    return;
  }

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
    eraseDeadOp(op);
}

bool CSEDriver::hasOtherSideEffectingOpInBetween(Operation *fromOp,
                                                 Operation *toOp,
                                                 MemEffectsCache &cache) {
  assert(fromOp->getBlock() == toOp->getBlock());
  assert(hasEffect<MemoryEffects::Read>(fromOp) &&
         "expected read effect on fromOp");
  assert(hasEffect<MemoryEffects::Read>(toOp) &&
         "expected read effect on toOp");
  if (fromOp->getNextNode() == toOp)
    return false;

  SmallVector<MemoryEffects::EffectInstance> readEffects;
  cast<MemoryEffectOpInterface>(fromOp).getEffects(readEffects);
  auto getEffects = [](Operation *op) {
    LDBG() << "CSE checking effects: " << op->getName();
    return getEffectsRecursively(op);
  };
  auto canConflict = [&](const auto &effects) {
    if (!effects)
      return true;
    return llvm::any_of(*effects, [&](const auto &effect) {
      if (!isa<MemoryEffects::Write>(effect.getEffect()))
        return false;
      SideEffects::Resource *writeResource = effect.getResource();
      return llvm::any_of(readEffects, [&](const auto &readEffect) {
        SideEffects::Resource *readResource = readEffect.getResource();
        if (writeResource->isDisjointFrom(readResource))
          return false;
        if (readEffect.getValue() && !writeResource->isAddressable())
          return false;
        if (effect.getValue() && !readResource->isAddressable())
          return false;
        return true;
      });
    });
  };

  auto &read = cache.reads[fromOp];
  if (rewriter.getListener()) {
    if (read.blocked)
      return true;
    Operation *nextOp =
        read.lastScanned ? read.lastScanned : fromOp->getNextNode();
    for (Operation *op = nextOp; op != toOp; op = op->getNextNode()) {
      if (canConflict(getEffects(op))) {
        read.blocked = true;
        return true;
      }
    }
    read.lastScanned = toOp->getPrevNode();
    return false;
  }

  auto isBefore = [](Operation *a, Operation *b) {
    return a->isBeforeInBlock(b);
  };
  auto first = llvm::upper_bound(cache.writes, fromOp, isBefore);
  unsigned &nextWrite = read.nextWrite;
  nextWrite = std::max(nextWrite, unsigned(first - cache.writes.begin()));
  for (; nextWrite < cache.writes.size(); ++nextWrite)
    if (canConflict(getEffects(cache.writes[nextWrite])))
      return true;

  // Index only already simplified operations: the query endpoint may be erased.
  for (Operation *op = cache.lastScanned->getNextNode(); op != toOp;
       op = op->getNextNode()) {
    cache.lastScanned = op;
    auto effects = getEffects(op);
    if (effects && llvm::none_of(*effects, [](const auto &effect) {
          return isa<MemoryEffects::Write>(effect.getEffect());
        }))
      continue;
    cache.writes.push_back(op);
    if (fromOp->isBeforeInBlock(op) && canConflict(effects))
      return true;
    nextWrite = cache.writes.size();
  }
  return false;
}

/// Attempt to eliminate a redundant operation.
LogicalResult CSEDriver::simplifyOperation(ScopedMapTy &knownValues,
                                           Operation *op, bool hasSSADominance,
                                           MemEffectsCache &memEffectsCache) {
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

    if (!memEffectsCache.lastScanned)
      memEffectsCache.lastScanned = op;

    // Look for an existing definition for the operation.
    if (auto *existing = knownValues.lookup(op)) {
      if (existing->getBlock() == op->getBlock() &&
          !hasOtherSideEffectingOpInBetween(existing, op, memEffectsCache)) {
        // The operation that can be deleted has been reach with no
        // side-effecting operations in between the existing operation and
        // this one so we can remove the duplicate.
        replaceUsesAndDelete(knownValues, op, existing, hasSSADominance);
        return success();
      }
    }
    knownValues.insert(op, op);
    return failure();
  }

  // Look for an existing definition for the operation.
  if (auto *existing = knownValues.lookup(op)) {
    replaceUsesAndDelete(knownValues, op, existing, hasSSADominance);
    return success();
  }

  // Otherwise, we add this operation to the known values map.
  knownValues.insert(op, op);
  return failure();
}

void CSEDriver::simplifyBlock(ScopedMapTy &knownValues, Block *bb,
                              bool hasSSADominance) {
  MemEffectsCache memEffectsCache;
  for (auto &op : llvm::make_early_inc_range(*bb)) {
    // If the operation is already trivially dead just add it to the erase list.
    // This also avoids calling `simplifyRegion` on dead region ops
    // unnecessarily.
    if (isOpTriviallyDead(&op)) {
      eraseDeadOp(&op);
      ++numDCE;
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
          simplifyRegion(nestedKnownValues, region);
      } else {
        // Otherwise, process nested regions normally.
        for (auto &region : op.getRegions())
          simplifyRegion(knownValues, region);
      }
    }

    // If the operation is simplified, we don't process any held regions.
    if (succeeded(simplifyOperation(knownValues, &op, hasSSADominance,
                                    memEffectsCache)))
      continue;
  }
}

void CSEDriver::simplifyRegion(ScopedMapTy &knownValues, Region &region) {
  // If the region is empty there is nothing to do.
  if (region.empty())
    return;

  bool hasSSADominance = domInfo->hasSSADominance(&region);

  // If the region only contains one block, then simplify it directly.
  if (region.hasOneBlock()) {
    ScopedMapTy::ScopeTy scope(knownValues);
    simplifyBlock(knownValues, &region.front(), hasSSADominance);
    return;
  }

  // If the region does not have dominanceInfo, then skip it.
  // TODO: Regions without SSA dominance should define a different
  // traversal order which is appropriate and can be used here.
  if (!hasSSADominance)
    return;

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
      simplifyBlock(knownValues, currentNode->node->getBlock(),
                    hasSSADominance);
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
}

void CSEDriver::eraseDeadOp(Operation *op) {
  for (Region &region : op->getRegions())
    domInfo->invalidate(&region);
  rewriter.eraseOp(op);

  // Note: CSE only removes ops within blocks, without adding or removing
  // blocks themselves. Since DominanceInfo captures relationships between
  // the direct blocks of the region being analyzed, not the blocks inside
  // any nested regions of those ops, it remains valid after CSE.
}

void CSEDriver::simplify(Operation *op, bool *changed) {
  // Simplify all regions.
  ScopedMapTy knownValues;
  for (auto &region : op->getRegions())
    simplifyRegion(knownValues, region);
  if (changed)
    *changed = numCSE || numDCE;
}

void CSEDriver::simplify(Region &region, bool *changed) {
  ScopedMapTy knownValues;
  simplifyRegion(knownValues, region);
  if (changed)
    *changed = numCSE || numDCE;
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
