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
  CSEDriver(RewriterBase &rewriter, DominanceInfo *domInfo,
            bool hoistPureOps = true)
      : rewriter(rewriter), domInfo(domInfo), hoistPureOps(hoistPureOps) {}

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

  /// Attempt to eliminate a redundant operation. Returns success if the
  /// operation was marked for removal, failure otherwise.
  LogicalResult simplifyOperation(ScopedMapTy &knownValues,
                                  ScopedMapTy &knownPureOps, Operation *op,
                                  bool hasSSADominance);
  void simplifyBlock(ScopedMapTy &knownValues, ScopedMapTy &knownPureOps,
                     Block *bb, bool hasSSADominance);
  void simplifyRegion(ScopedMapTy &knownValues, ScopedMapTy &knownPureOps,
                      Region &region);

  /// Erase all operations queued for deletion by the simplification routines.
  void eraseDeadOps(bool *changed);

  void replaceUsesAndDelete(ScopedMapTy &knownValues, Operation *op,
                            Operation *existing, bool hasSSADominance);

  /// Check if there is side-effecting operations other than the given effect
  /// between the two operations.
  bool hasOtherSideEffectingOpInBetween(Operation *fromOp, Operation *toOp);

  LogicalResult hoistPureOp(Operation *existing, Operation *op);

  /// A rewriter for modifying the IR.
  RewriterBase &rewriter;

  /// Operations marked as dead and to be erased.
  std::vector<Operation *> opsToErase;

  DominanceInfo *domInfo = nullptr;
  MemEffectsCache memEffectsCache;

  // Various statistics.
  int64_t numCSE = 0;
  int64_t numDCE = 0;

  bool hoistPureOps = true;

  // The keys of this map are existing ops, and the values are lists of
  // operations. Each entry describes an existing op that has been hoisted along
  // with its associated operations. When an existing op and its equivalent
  // operations are hoisted for the first time, they are added to this map.
  // During subsequent hoistings of the same existing op(hoisting exiting op
  // multiple times), the operations stored in the map's value will also be
  // hoisted together with it.
  DenseMap<Operation *, SmallVector<Operation *>> hoistOpsSet;
};
} // namespace

/// Returns true if the path between block 'a' and block 'b' in the region
/// hierarchy crosses an operation with the 'IsIsolatedFromAbove' trait.
static bool isBlockCrossIsIsolatedFromAbove(DominanceInfo *dominate, Block *a,
                                            Block *b) {
  if (a == b)
    return false;
  if (a->getParent() == b->getParent())
    return false;
  if (dominate->dominates(b, a))
    std::swap(b, a);
  while (b && b->getParentOp()) {
    Operation *parentOp = b->getParentOp();
    if (parentOp->mightHaveTrait<OpTrait::IsIsolatedFromAbove>())
      return true;
    b = parentOp->getBlock();
    if (b == a)
      return false;
  }
  return false;
}

/// Hoist the pure ops to the location of the Nearest Common Dominator.
LogicalResult CSEDriver::hoistPureOp(Operation *existing, Operation *op) {
  Block *ancestorBlock =
      domInfo->findNearestCommonDominator(existing->getBlock(), op->getBlock());
  if (!ancestorBlock) {
    LDBG() << "hoist " << OpWithFlags(existing, OpPrintingFlags().skipRegions())
           << " and " << OpWithFlags(op, OpPrintingFlags().skipRegions())
           << " failed";
    return failure();
  }

  if (isBlockCrossIsIsolatedFromAbove(domInfo, ancestorBlock,
                                      existing->getBlock()) ||
      isBlockCrossIsIsolatedFromAbove(domInfo, ancestorBlock, op->getBlock()))
    return failure();

  // Find the insertion point based on dominance relationships. When hoisting a
  // region op, we must consider not only its operands but also the dominance
  // relationships of the operations within the region when determining the
  // insertion point
  Operation *insertPoint = nullptr;
  SetVector<Value> dependentOperands;
  dependentOperands.insert_range(existing->getOperands());
  mlir::getUsedValuesDefinedAbove(existing->getRegions(), dependentOperands);
  for (Value operand : dependentOperands) {
    if (domInfo->properlyDominates(operand, &ancestorBlock->front()))
      continue;
    if (!insertPoint) {
      insertPoint = operand.getDefiningOp();
    } else {
      insertPoint = domInfo->dominates(insertPoint, operand.getDefiningOp())
                        ? operand.getDefiningOp()
                        : insertPoint;
    }
  }

  // We hoist both `op` and `existing` here because if they are identical
  // regionOps and we only hoist existing, the two would no longer be congruent.
  // This would lead to a missed optimization opportunity in subsequent CSE
  // passes. The test @cse_multiple_regions's `%r2` tests it.
  if (!insertPoint) {
    rewriter.moveOpBefore(existing, ancestorBlock, ancestorBlock->begin());
    rewriter.moveOpAfter(op, existing);
  } else {
    rewriter.moveOpAfter(existing, insertPoint);
    rewriter.moveOpAfter(op, existing);
  }

  // When hoisting an exiting op multiple times, we must also hoist the
  // operations that were previously hoisted alongside it.
  if (hoistOpsSet.contains(existing) && !hoistOpsSet[existing].empty())
    for (Operation *op : hoistOpsSet[existing])
      rewriter.moveOpAfter(op, existing);
  hoistOpsSet[existing].push_back(op);
  LDBG() << "hoist " << OpWithFlags(existing, OpPrintingFlags().skipRegions())
         << " and " << OpWithFlags(op, OpPrintingFlags().skipRegions())
         << " success";
  return success();
}

void CSEDriver::replaceUsesAndDelete(ScopedMapTy &knownValues, Operation *op,
                                     Operation *existing,
                                     bool hasSSADominance) {
  // If we find one then replace all uses of the current operation with the
  // existing one and mark it for deletion. We can only replace an operand in
  // an operation if it has not been visited yet.
  if (hasSSADominance) {
    // If the region has SSA dominance, then we are guaranteed to have not
    // visited any use of the current operation.
    // Replace all uses, but do not remove the operation yet.
    if (!domInfo->properlyDominates(existing, op)) {
      if (!hoistPureOps || failed(hoistPureOp(existing, op)))
        return;
    } else {
      // Hoist `op` even though `existing` already dominates it, because
      // hoisting op may create further CSE optimization opportunities for
      // subsequent region operations. The test @cse_multiple_regions's `%r3`
      // tests it.
      rewriter.moveOpAfter(op, existing);
    }
    LDBG() << "replace " << OpWithFlags(op, OpPrintingFlags().skipRegions())
           << " with "
           << OpWithFlags(existing, OpPrintingFlags().skipRegions());
    LDBG() << "add " << OpWithFlags(op, OpPrintingFlags().skipRegions())
           << " to opsToErase";
    rewriter.replaceAllOpUsesWith(op, existing->getResults());
    opsToErase.push_back(op);
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

    if (!domInfo->properlyDominates(existing, op)) {
      if (!hoistPureOps || failed(hoistPureOp(existing, op)))
        return;
    }
    // Replace all uses, but do not remove the operation yet. This does not
    // notify the listener because the original op is not erased.
    LDBG() << "replace " << OpWithFlags(op, OpPrintingFlags().skipRegions())
           << " with "
           << OpWithFlags(existing, OpPrintingFlags().skipRegions());
    // Replace all uses, but do not remove the operation yet. This does not
    // notify the listener because the original op is not erased.
    rewriter.replaceUsesWithIf(op->getResults(), existing->getResults(),
                               wasVisited);
    // There may be some remaining uses of the operation.
    if (op->use_empty())
      opsToErase.push_back(op);
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
LogicalResult CSEDriver::simplifyOperation(ScopedMapTy &knownValues,
                                           ScopedMapTy &knownPureOps,
                                           Operation *op,
                                           bool hasSSADominance) {
  LDBG() << "visit operation: "
         << OpWithFlags(op, OpPrintingFlags().skipRegions());
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
        replaceUsesAndDelete(knownValues, op, existing, hasSSADominance);
        return success();
      }
    }
    LDBG() << "insert op: " << OpWithFlags(op, OpPrintingFlags().skipRegions())
           << " to map";
    knownValues.insert(op, op);
    return failure();
  }

  // Look for an existing definition for the operation.
  if (auto *existing = knownValues.lookup(op)) {
    replaceUsesAndDelete(knownValues, op, existing, hasSSADominance);
    return success();
  }

  if (auto *existing = knownPureOps.lookup(op)) {
    replaceUsesAndDelete(knownPureOps, op, existing, hasSSADominance);
    return success();
  }

  if (mlir::isPure(op)) {
    LDBG() << "insert op: " << OpWithFlags(op, OpPrintingFlags().skipRegions())
           << " to pureMap";
    knownPureOps.insert(op, op);
  } else {
    // Otherwise, we add this operation to the known values map.
    LDBG() << "insert op: " << OpWithFlags(op, OpPrintingFlags().skipRegions())
           << " to map";
    knownValues.insert(op, op);
  }
  return failure();
}

void CSEDriver::simplifyBlock(ScopedMapTy &knownValues,
                              ScopedMapTy &knownPureOps, Block *bb,
                              bool hasSSADominance) {
  LDBG() << "visit block #" << bb->computeBlockNumber() << " of "
         << OpWithFlags(bb->getParentOp(), OpPrintingFlags().skipRegions());
  for (auto &op : llvm::make_early_inc_range(*bb)) {
    // If the operation is already trivially dead just add it to the erase list.
    // This also avoids calling `simplifyRegion` on dead region ops
    // unnecessarily.
    if (isOpTriviallyDead(&op)) {
      opsToErase.push_back(&op);
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
        ScopedMapTy nestedKnownPureOps;
        ScopedMapTy::ScopeTy scope(nestedKnownValues);
        ScopedMapTy::ScopeTy pureScope(nestedKnownPureOps);
        for (auto &region : op.getRegions())
          simplifyRegion(nestedKnownValues, nestedKnownPureOps, region);
      } else {
        // Otherwise, process nested regions normally.
        for (auto &region : op.getRegions())
          simplifyRegion(knownValues, knownPureOps, region);
      }
    }

    // If the operation is simplified, we don't process any held regions.
    if (succeeded(
            simplifyOperation(knownValues, knownPureOps, &op, hasSSADominance)))
      continue;
  }
  // Clear the MemoryEffects cache since its usage is by block only.
  memEffectsCache.clear();
}

void CSEDriver::simplifyRegion(ScopedMapTy &knownValues,
                               ScopedMapTy &knownPureOps, Region &region) {
  // If the region is empty there is nothing to do.
  if (region.empty())
    return;

  LDBG() << "visit region #" << region.getRegionNumber() << " of "
         << OpWithFlags(region.getParentOp(), OpPrintingFlags().skipRegions());

  bool hasSSADominance = domInfo->hasSSADominance(&region);

  // If the region only contains one block, then simplify it directly.
  if (region.hasOneBlock()) {
    ScopedMapTy::ScopeTy scope(knownValues);
    simplifyBlock(knownValues, knownPureOps, &region.front(), hasSSADominance);
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
      simplifyBlock(knownValues, knownPureOps, currentNode->node->getBlock(),
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

void CSEDriver::eraseDeadOps(bool *changed) {
  // Erase any operations that were marked as dead during simplification, and
  // remove their associated dominator trees.
  for (auto *op : opsToErase) {
    for (Region &region : op->getRegions())
      domInfo->invalidate(&region);
    rewriter.eraseOp(op);
  }
  if (changed)
    *changed = !opsToErase.empty();
  opsToErase.clear();

  // Note: CSE does currently not remove ops with regions, so DominanceInfo
  // does not have to be invalidated.
}

void CSEDriver::simplify(Operation *op, bool *changed) {
  /// Simplify all regions. Added a new scope using curly braces to release the
  /// knownPureOps scope before deleting the operation.
  {
    /// The entry point for CSE simplification. A top-level scope is added for
    /// 'knownPureOps' to track pure operations across the entire operation's
    /// regions, enabling potential hoisting opportunities. Since only pure
    /// operations are candidates for hoisting, 'knownValues' does not require
    /// a corresponding top-level scope here.
    ScopedMapTy knownValues;
    ScopedMapTy knownPureOps;
    ScopedMapTy::ScopeTy scope(knownPureOps);
    for (auto &region : op->getRegions())
      simplifyRegion(knownValues, knownPureOps, region);
  }
  eraseDeadOps(changed);
}

void CSEDriver::simplify(Region &region, bool *changed) {
  {
    ScopedMapTy knownValues;
    ScopedMapTy knownPureOps;
    ScopedMapTy::ScopeTy scope(knownPureOps);
    simplifyRegion(knownValues, knownPureOps, region);
  }
  eraseDeadOps(changed);
}

void mlir::eliminateCommonSubExpressions(RewriterBase &rewriter,
                                         DominanceInfo &domInfo, Operation *op,
                                         bool *changed, int64_t *numCSE,
                                         int64_t *numDCE, bool hoistPureOps) {
  CSEDriver driver(rewriter, &domInfo, hoistPureOps);
  driver.simplify(op, changed);
  if (numCSE)
    *numCSE = driver.getNumCSE();
  if (numDCE)
    *numDCE = driver.getNumDCE();
}

void mlir::eliminateCommonSubExpressions(RewriterBase &rewriter,
                                         DominanceInfo &domInfo, Region &region,
                                         bool *changed, bool hoistPureOps) {
  CSEDriver driver(rewriter, &domInfo, hoistPureOps);
  driver.simplify(region, changed);
}
