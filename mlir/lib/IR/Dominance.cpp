//===- Dominance.cpp - Dominator analysis for CFGs ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of dominance related classes and instantiations of extern
// templates.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Dominance.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/RegionKindInterface.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/GenericDomTreeConstruction.h"

using namespace mlir;
using namespace mlir::detail;

template class llvm::DominatorTreeBase<Block, /*IsPostDom=*/false>;
template class llvm::DominatorTreeBase<Block, /*IsPostDom=*/true>;
template class llvm::DomTreeNodeBase<Block>;

//===----------------------------------------------------------------------===//
// DominanceInfoBase
//===----------------------------------------------------------------------===//

template <bool IsPostDom>
DominanceInfoBase<IsPostDom>::~DominanceInfoBase() {
  for (auto entry : dominanceInfos)
    delete entry.second.getPointer();
}

/// Return the dom tree and "hasSSADominance" bit for the given region.  The
/// DomTree will be null for single-block regions.  This lazily constructs the
/// DomTree on demand when needsDomTree=true.
template <bool IsPostDom>
auto DominanceInfoBase<IsPostDom>::getDominanceInfo(Region *region,
                                                    bool needsDomTree) const
    -> llvm::PointerIntPair<DomTree *, 1, bool> {
  // Check to see if we already have this information.
  auto itAndInserted = dominanceInfos.insert({region, {nullptr, true}});
  auto &entry = itAndInserted.first->second;

  // This method builds on knowledge that multi-block regions always have
  // SSADominance.  Graph regions are only allowed to be single-block regions,
  // but of course single-block regions may also have SSA dominance.
  if (!itAndInserted.second) {
    // We do have it, so we know the 'hasSSADominance' bit is correct, but we
    // may not have constructed a DominatorTree yet.  If we need it, build it.
    if (needsDomTree && !entry.getPointer() && !region->hasOneBlock()) {
      auto *domTree = new DomTree();
      domTree->recalculate(*region);
      entry.setPointer(domTree);
    }
    return entry;
  }

  // Nope, lazily construct it.  Create a DomTree if this is a multi-block
  // region.
  if (!region->hasOneBlock()) {
    auto *domTree = new DomTree();
    domTree->recalculate(*region);
    entry.setPointer(domTree);
    // Multiblock regions always have SSA dominance, leave `second` set to true.
    return entry;
  }

  // Single block regions have a more complicated predicate.
  if (Operation *parentOp = region->getParentOp()) {
    if (!parentOp->isRegistered()) { // We don't know about unregistered ops.
      entry.setInt(false);
    } else if (auto regionKindItf = dyn_cast<RegionKindInterface>(parentOp)) {
      // Registered ops can opt-out of SSA dominance with
      // RegionKindInterface.
      entry.setInt(regionKindItf.hasSSADominance(region->getRegionNumber()));
    }
  }

  return entry;
}

/// Return the ancestor block enclosing the specified block.  This returns null
/// if we reach the top of the hierarchy.
static Block *getAncestorBlock(Block *block) {
  if (Operation *ancestorOp = block->getParentOp())
    return ancestorOp->getBlock();
  return nullptr;
}

/// Walks up the list of containers of the given block and calls the
/// user-defined traversal function for every pair of a region and block that
/// could be found during traversal. If the user-defined function returns true
/// for a given pair, traverseAncestors will return the current block. Nullptr
/// otherwise.
template <typename FuncT>
static Block *traverseAncestors(Block *block, const FuncT &func) {
  do {
    // Invoke the user-defined traversal function for each block.
    if (func(block))
      return block;
  } while ((block = getAncestorBlock(block)));
  return nullptr;
}

/// Tries to update the given block references to live in the same region by
/// exploring the relationship of both blocks with respect to their regions.
static bool tryGetBlocksInSameRegion(Block *&a, Block *&b) {
  // If both block do not live in the same region, we will have to check their
  // parent operations.
  Region *aRegion = a->getParent();
  Region *bRegion = b->getParent();
  if (aRegion == bRegion)
    return true;

  // Iterate over all ancestors of `a`, counting the depth of `a`. If one of
  // `a`s ancestors are in the same region as `b`, then we stop early because we
  // found our NCA.
  size_t aRegionDepth = 0;
  if (Block *aResult = traverseAncestors(a, [&](Block *block) {
        ++aRegionDepth;
        return block->getParent() == bRegion;
      })) {
    a = aResult;
    return true;
  }

  // Iterate over all ancestors of `b`, counting the depth of `b`. If one of
  // `b`s ancestors are in the same region as `a`, then we stop early because
  // we found our NCA.
  size_t bRegionDepth = 0;
  if (Block *bResult = traverseAncestors(b, [&](Block *block) {
        ++bRegionDepth;
        return block->getParent() == aRegion;
      })) {
    b = bResult;
    return true;
  }

  // Otherwise we found two blocks that are siblings at some level.  Walk the
  // deepest one up until we reach the top or find an NCA.
  while (true) {
    if (aRegionDepth > bRegionDepth) {
      a = getAncestorBlock(a);
      --aRegionDepth;
    } else if (aRegionDepth < bRegionDepth) {
      b = getAncestorBlock(b);
      --bRegionDepth;
    } else {
      break;
    }
  }

  // If we found something with the same level, then we can march both up at the
  // same time from here on out.
  while (a) {
    // If they are at the same level, and have the same parent region then we
    // succeeded.
    if (a->getParent() == b->getParent())
      return true;

    a = getAncestorBlock(a);
    b = getAncestorBlock(b);
  }

  // They don't share an NCA, perhaps they are in different modules or
  // something.
  return false;
}

template <bool IsPostDom>
Block *
DominanceInfoBase<IsPostDom>::findNearestCommonDominator(Block *a,
                                                         Block *b) const {
  // If either a or b are null, then conservatively return nullptr.
  if (!a || !b)
    return nullptr;

  // If they are the same block, then we are done.
  if (a == b)
    return a;

  // Try to find blocks that are in the same region.
  if (!tryGetBlocksInSameRegion(a, b))
    return nullptr;

  // If the common ancestor in a common region is the same block, then return
  // it.
  if (a == b)
    return a;

  // Otherwise, there must be multiple blocks in the region, check the
  // DomTree.
  return getDomTree(a->getParent()).findNearestCommonDominator(a, b);
}

/// Return true if the specified block A properly dominates block B.
template <bool IsPostDom>
bool DominanceInfoBase<IsPostDom>::properlyDominates(Block *a, Block *b) const {
  assert(a && b && "null blocks not allowed");

  // A block dominates itself but does not properly dominate itself.
  if (a == b)
    return false;

  // If both blocks are not in the same region, `a` properly dominates `b` if
  // `b` is defined in an operation region that (recursively) ends up being
  // dominated by `a`. Walk up the list of containers enclosing B.
  Region *regionA = a->getParent();
  if (regionA != b->getParent()) {
    b = regionA ? regionA->findAncestorBlockInRegion(*b) : nullptr;
    // If we could not find a valid block b then it is a not a dominator.
    if (b == nullptr)
      return false;

    // Check to see if the ancestor of `b` is the same block as `a`.  A properly
    // dominates B if it contains an op that contains the B block.
    if (a == b)
      return true;
  }

  // Otherwise, they are two different blocks in the same region, use DomTree.
  return getDomTree(regionA).properlyDominates(a, b);
}

/// Return true if the specified block is reachable from the entry block of
/// its region.
template <bool IsPostDom>
bool DominanceInfoBase<IsPostDom>::isReachableFromEntry(Block *a) const {
  // If this is the first block in its region, then it is obviously reachable.
  Region *region = a->getParent();
  if (&region->front() == a)
    return true;

  // Otherwise this is some block in a multi-block region.  Check DomTree.
  return getDomTree(region).isReachableFromEntry(a);
}

template class detail::DominanceInfoBase</*IsPostDom=*/true>;
template class detail::DominanceInfoBase</*IsPostDom=*/false>;

//===----------------------------------------------------------------------===//
// DominanceInfo
//===----------------------------------------------------------------------===//

/// Return true if operation `a` properly dominates operation `b`.  The
/// 'enclosingOpOk' flag says whether we should return true if the `b` op is
/// enclosed by a region on 'a'.
bool DominanceInfo::properlyDominatesImpl(Operation *a, Operation *b,
                                          bool enclosingOpOk) const {
  Block *aBlock = a->getBlock(), *bBlock = b->getBlock();
  assert(aBlock && bBlock && "operations must be in a block");

  // An instruction dominates, but does not properlyDominate, itself unless this
  // is a graph region.
  if (a == b)
    return !hasSSADominance(aBlock);

  // If these ops are in different regions, then normalize one into the other.
  Region *aRegion = aBlock->getParent();
  if (aRegion != bBlock->getParent()) {
    // Scoot up b's region tree until we find an operation in A's region that
    // encloses it.  If this fails, then we know there is no post-dom relation.
    b = aRegion ? aRegion->findAncestorOpInRegion(*b) : nullptr;
    if (!b)
      return false;
    bBlock = b->getBlock();
    assert(bBlock->getParent() == aRegion);

    // If 'a' encloses 'b', then we consider it to dominate.
    if (a == b && enclosingOpOk)
      return true;
  }

  // Ok, they are in the same region now.
  if (aBlock == bBlock) {
    // Dominance changes based on the region type. In a region with SSA
    // dominance, uses inside the same block must follow defs. In other
    // regions kinds, uses and defs can come in any order inside a block.
    if (hasSSADominance(aBlock)) {
      // If the blocks are the same, then check if b is before a in the block.
      return a->isBeforeInBlock(b);
    }
    return true;
  }

  // If the blocks are different, use DomTree to resolve the query.
  return getDomTree(aRegion).properlyDominates(aBlock, bBlock);
}

/// Return true if the `a` value properly dominates operation `b`, i.e if the
/// operation that defines `a` properlyDominates `b` and the operation that
/// defines `a` does not contain `b`.
bool DominanceInfo::properlyDominates(Value a, Operation *b) const {
  // block arguments properly dominate all operations in their own block, so
  // we use a dominates check here, not a properlyDominates check.
  if (auto blockArg = a.dyn_cast<BlockArgument>())
    return dominates(blockArg.getOwner(), b->getBlock());

  // `a` properlyDominates `b` if the operation defining `a` properlyDominates
  // `b`, but `a` does not itself enclose `b` in one of its regions.
  return properlyDominatesImpl(a.getDefiningOp(), b, /*enclosingOpOk=*/false);
}

//===----------------------------------------------------------------------===//
// PostDominanceInfo
//===----------------------------------------------------------------------===//

/// Returns true if statement 'a' properly postdominates statement b.
bool PostDominanceInfo::properlyPostDominates(Operation *a, Operation *b) {
  auto *aBlock = a->getBlock(), *bBlock = b->getBlock();
  assert(aBlock && bBlock && "operations must be in a block");

  // An instruction postDominates, but does not properlyPostDominate, itself
  // unless this is a graph region.
  if (a == b)
    return !hasSSADominance(aBlock);

  // If these ops are in different regions, then normalize one into the other.
  Region *aRegion = aBlock->getParent();
  if (aRegion != bBlock->getParent()) {
    // Scoot up b's region tree until we find an operation in A's region that
    // encloses it.  If this fails, then we know there is no post-dom relation.
    b = aRegion ? aRegion->findAncestorOpInRegion(*b) : nullptr;
    if (!b)
      return false;
    bBlock = b->getBlock();
    assert(bBlock->getParent() == aRegion);

    // If 'a' encloses 'b', then we consider it to postdominate.
    if (a == b)
      return true;
  }

  // Ok, they are in the same region.  If they are in the same block, check if b
  // is before a in the block.
  if (aBlock == bBlock) {
    // Dominance changes based on the region type.
    if (hasSSADominance(aBlock)) {
      // If the blocks are the same, then check if b is before a in the block.
      return b->isBeforeInBlock(a);
    }
    return true;
  }

  // If the blocks are different, check if a's block post dominates b's.
  return getDomTree(aRegion).properlyDominates(aBlock, bBlock);
}
