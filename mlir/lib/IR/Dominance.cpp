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

template <bool IsPostDom>
void DominanceInfoBase<IsPostDom>::invalidate() {
  for (auto entry : dominanceInfos)
    delete entry.second.getPointer();
  dominanceInfos.clear();
}

template <bool IsPostDom>
void DominanceInfoBase<IsPostDom>::invalidate(Region *region) {
  auto it = dominanceInfos.find(region);
  if (it != dominanceInfos.end()) {
    delete it->second.getPointer();
    dominanceInfos.erase(it);
  }
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

/// Returns the given block iterator if it lies within the region region.
/// Otherwise, otherwise finds the ancestor of the given block iterator that
/// lies within the given region. Returns and "empty" iterator if the latter
/// fails.
///
/// Note: This is a variant of Region::findAncestorOpInRegion that operates on
/// block iterators instead of ops.
static std::pair<Block *, Block::iterator>
findAncestorIteratorInRegion(Region *r, Block *b, Block::iterator it) {
  // Case 1: The iterator lies within the region region.
  if (b->getParent() == r)
    return std::make_pair(b, it);

  // Otherwise: Find ancestor iterator. Bail if we run out of parent ops.
  Operation *parentOp = b->getParentOp();
  if (!parentOp)
    return std::make_pair(static_cast<Block *>(nullptr), Block::iterator());
  Operation *op = r->findAncestorOpInRegion(*parentOp);
  if (!op)
    return std::make_pair(static_cast<Block *>(nullptr), Block::iterator());
  return std::make_pair(op->getBlock(), op->getIterator());
}

/// Given two iterators into the same block, return "true" if `a` is before `b.
/// Note: This is a variant of Operation::isBeforeInBlock that operates on
/// block iterators instead of ops.
static bool isBeforeInBlock(Block *block, Block::iterator a,
                            Block::iterator b) {
  if (a == b)
    return false;
  if (a == block->end())
    return false;
  if (b == block->end())
    return true;
  return a->isBeforeInBlock(&*b);
}

template <bool IsPostDom>
bool DominanceInfoBase<IsPostDom>::properlyDominatesImpl(
    Block *aBlock, Block::iterator aIt, Block *bBlock, Block::iterator bIt,
    bool enclosingOk) const {
  assert(aBlock && bBlock && "expected non-null blocks");

  // A block iterator (post)dominates, but does not properly (post)dominate,
  // itself unless this is a graph region.
  if (aBlock == bBlock && aIt == bIt)
    return !hasSSADominance(aBlock);

  // If the iterators are in different regions, then normalize one into the
  // other.
  Region *aRegion = aBlock->getParent();
  if (aRegion != bBlock->getParent()) {
    // Scoot up b's region tree until we find a location in A's region that
    // encloses it.  If this fails, then we know there is no (post)dom relation.
    if (!aRegion) {
      bBlock = nullptr;
      bIt = Block::iterator();
    } else {
      std::tie(bBlock, bIt) =
          findAncestorIteratorInRegion(aRegion, bBlock, bIt);
    }
    if (!bBlock)
      return false;
    assert(bBlock->getParent() == aRegion && "expected block in regionA");

    // If 'a' encloses 'b', then we consider it to (post)dominate.
    if (aBlock == bBlock && aIt == bIt && enclosingOk)
      return true;
  }

  // Ok, they are in the same region now.
  if (aBlock == bBlock) {
    // Dominance changes based on the region type. In a region with SSA
    // dominance, uses inside the same block must follow defs. In other
    // regions kinds, uses and defs can come in any order inside a block.
    if (!hasSSADominance(aBlock))
      return true;
    if constexpr (IsPostDom) {
      return isBeforeInBlock(aBlock, bIt, aIt);
    } else {
      return isBeforeInBlock(aBlock, aIt, bIt);
    }
  }

  // If the blocks are different, use DomTree to resolve the query.
  return getDomTree(aRegion).properlyDominates(aBlock, bBlock);
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

bool DominanceInfo::properlyDominates(Operation *a, Operation *b,
                                      bool enclosingOpOk) const {
  return super::properlyDominatesImpl(a->getBlock(), a->getIterator(),
                                      b->getBlock(), b->getIterator(),
                                      enclosingOpOk);
}

bool DominanceInfo::properlyDominates(Block *a, Block *b) const {
  return super::properlyDominatesImpl(a, a->begin(), b, b->begin(),
                                      /*enclosingOk=*/true);
}

/// Return true if the `a` value properly dominates operation `b`, i.e if the
/// operation that defines `a` properlyDominates `b` and the operation that
/// defines `a` does not contain `b`.
bool DominanceInfo::properlyDominates(Value a, Operation *b) const {
  // block arguments properly dominate all operations in their own block, so
  // we use a dominates check here, not a properlyDominates check.
  if (auto blockArg = dyn_cast<BlockArgument>(a))
    return dominates(blockArg.getOwner(), b->getBlock());

  // `a` properlyDominates `b` if the operation defining `a` properlyDominates
  // `b`, but `a` does not itself enclose `b` in one of its regions.
  return properlyDominates(a.getDefiningOp(), b, /*enclosingOpOk=*/false);
}

//===----------------------------------------------------------------------===//
// PostDominanceInfo
//===----------------------------------------------------------------------===//

bool PostDominanceInfo::properlyPostDominates(Operation *a, Operation *b,
                                              bool enclosingOpOk) const {
  return super::properlyDominatesImpl(a->getBlock(), a->getIterator(),
                                      b->getBlock(), b->getIterator(),
                                      enclosingOpOk);
}

bool PostDominanceInfo::properlyPostDominates(Block *a, Block *b) const {
  return super::properlyDominatesImpl(a, a->end(), b, b->end(),
                                      /*enclosingOk=*/true);
}
