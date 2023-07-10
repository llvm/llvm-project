//===- CSE.cpp - Common Sub-expression Elimination ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transformation pass performs a simple common sub-expression elimination
// algorithm on operations within a region.
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/CSE.h"

#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/RecyclingAllocator.h"
#include <deque>

namespace mlir {
#define GEN_PASS_DEF_CSE
#include "mlir/Transforms/Passes.h.inc"
} // namespace mlir

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
  void simplify(Operation *op, bool *changed = nullptr);

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
  LogicalResult simplifyOperation(ScopedMapTy &knownValues, Operation *op,
                                  bool hasSSADominance);
  void simplifyBlock(ScopedMapTy &knownValues, Block *bb, bool hasSSADominance);
  void simplifyRegion(ScopedMapTy &knownValues, Region &region);

  void replaceUsesAndDelete(ScopedMapTy &knownValues, Operation *op,
                            Operation *existing, bool hasSSADominance);

  /// Check if there is side-effecting operations other than the given effect
  /// between the two operations.
  bool hasOtherSideEffectingOpInBetween(Operation *fromOp, Operation *toOp);

  /// A rewriter for modifying the IR.
  RewriterBase &rewriter;

  /// Operations marked as dead and to be erased.
  std::vector<Operation *> opsToErase;
  DominanceInfo *domInfo = nullptr;
  MemEffectsCache memEffectsCache;

  // Various statistics.
  int64_t numCSE = 0;
  int64_t numDCE = 0;
};
} // namespace

void CSEDriver::replaceUsesAndDelete(ScopedMapTy &knownValues, Operation *op,
                                     Operation *existing,
                                     bool hasSSADominance) {
  // If we find one then replace all uses of the current operation with the
  // existing one and mark it for deletion. We can only replace an operand in
  // an operation if it has not been visited yet.
  if (hasSSADominance) {
    // If the region has SSA dominance, then we are guaranteed to have not
    // visited any use of the current operation.
    if (auto *rewriteListener =
            dyn_cast_if_present<RewriterBase::Listener>(rewriter.getListener()))
      rewriteListener->notifyOperationReplaced(op, existing);
    // Replace all uses, but do not remote the operation yet. This does not
    // notify the listener because the original op is not erased.
    rewriter.replaceAllUsesWith(op->getResults(), existing->getResults());
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

    // Replace all uses, but do not remote the operation yet. This does not
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
  assert(
      isa<MemoryEffectOpInterface>(fromOp) &&
      cast<MemoryEffectOpInterface>(fromOp).hasEffect<MemoryEffects::Read>() &&
      isa<MemoryEffectOpInterface>(toOp) &&
      cast<MemoryEffectOpInterface>(toOp).hasEffect<MemoryEffects::Read>());
  Operation *nextOp = fromOp->getNextNode();
  auto result =
      memEffectsCache.try_emplace(fromOp, std::make_pair(fromOp, nullptr));
  if (result.second) {
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
    auto nextOpMemEffects = dyn_cast<MemoryEffectOpInterface>(nextOp);
    // TODO: Do we need to handle other effects generically?
    // If the operation does not implement the MemoryEffectOpInterface we
    // conservatively assumes it writes.
    if ((nextOpMemEffects &&
         nextOpMemEffects.hasEffect<MemoryEffects::Write>()) ||
        !nextOpMemEffects) {
      result.first->second =
          std::make_pair(nextOp, MemoryEffects::Write::get());
      return true;
    }
    nextOp = nextOp->getNextNode();
  }
  result.first->second = std::make_pair(toOp, nullptr);
  return false;
}

/// Attempt to eliminate a redundant operation.
LogicalResult CSEDriver::simplifyOperation(ScopedMapTy &knownValues,
                                           Operation *op,
                                           bool hasSSADominance) {
  // Don't simplify terminator operations.
  if (op->hasTrait<OpTrait::IsTerminator>())
    return failure();

  // If the operation is already trivially dead just add it to the erase list.
  if (isOpTriviallyDead(op)) {
    opsToErase.push_back(op);
    ++numDCE;
    return success();
  }

  // Don't simplify operations with regions that have multiple blocks.
  // TODO: We need additional tests to verify that we handle such IR correctly.
  if (!llvm::all_of(op->getRegions(), [](Region &r) {
        return r.getBlocks().empty() || llvm::hasSingleElement(r.getBlocks());
      }))
    return failure();

  // Some simple use case of operation with memory side-effect are dealt with
  // here. Operations with no side-effect are done after.
  if (!isMemoryEffectFree(op)) {
    auto memEffects = dyn_cast<MemoryEffectOpInterface>(op);
    // TODO: Only basic use case for operations with MemoryEffects::Read can be
    // eleminated now. More work needs to be done for more complicated patterns
    // and other side-effects.
    if (!memEffects || !memEffects.onlyHasEffect<MemoryEffects::Read>())
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
    knownValues.insert(op, op);
    return failure();
  }

  // Look for an existing definition for the operation.
  if (auto *existing = knownValues.lookup(op)) {
    replaceUsesAndDelete(knownValues, op, existing, hasSSADominance);
    ++numCSE;
    return success();
  }

  // Otherwise, we add this operation to the known values map.
  knownValues.insert(op, op);
  return failure();
}

void CSEDriver::simplifyBlock(ScopedMapTy &knownValues, Block *bb,
                              bool hasSSADominance) {
  for (auto &op : *bb) {
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
    if (succeeded(simplifyOperation(knownValues, &op, hasSSADominance)))
      continue;
  }
  // Clear the MemoryEffects cache since its usage is by block only.
  memEffectsCache.clear();
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

void CSEDriver::simplify(Operation *op, bool *changed) {
  /// Simplify all regions.
  ScopedMapTy knownValues;
  for (auto &region : op->getRegions())
    simplifyRegion(knownValues, region);

  /// Erase any operations that were marked as dead during simplification.
  for (auto *op : opsToErase)
    rewriter.eraseOp(op);
  if (changed)
    *changed = !opsToErase.empty();

  // Note: CSE does currently not remove ops with regions, so DominanceInfo
  // does not have to be invalidated.
}

void mlir::eliminateCommonSubExpressions(RewriterBase &rewriter,
                                         DominanceInfo &domInfo, Operation *op,
                                         bool *changed) {
  CSEDriver driver(rewriter, &domInfo);
  driver.simplify(op, changed);
}

namespace {
/// CSE pass.
struct CSE : public impl::CSEBase<CSE> {
  void runOnOperation() override;
};
} // namespace

void CSE::runOnOperation() {
  // Simplify the IR.
  IRRewriter rewriter(&getContext());
  CSEDriver driver(rewriter, &getAnalysis<DominanceInfo>());
  bool changed = false;
  driver.simplify(getOperation(), &changed);

  // Set statistics.
  numCSE = driver.getNumCSE();
  numDCE = driver.getNumDCE();

  // If there was no change to the IR, we mark all analyses as preserved.
  if (!changed)
    return markAllAnalysesPreserved();

  // We currently don't remove region operations, so mark dominance as
  // preserved.
  markAnalysesPreserved<DominanceInfo, PostDominanceInfo>();
}

std::unique_ptr<Pass> mlir::createCSEPass() { return std::make_unique<CSE>(); }
