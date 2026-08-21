//===- GenericLoopInfoImp.h - Generic Loop Info Implementation --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This fle contains the implementation of GenericLoopInfo. It should only be
// included in files that explicitly instantiate a GenericLoopInfo.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_GENERICLOOPINFOIMPL_H
#define LLVM_SUPPORT_GENERICLOOPINFOIMPL_H

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/GenericLoopInfo.h"

namespace llvm {

//===----------------------------------------------------------------------===//
// APIs for simple analysis of the loop. See header notes.

/// getExitingBlocks - Return all blocks inside the loop that have successors
/// outside of the loop.  These are the blocks _inside of the current loop_
/// which branch out.  The returned list is always unique.
///
template <class BlockT, class LoopT>
void LoopBase<BlockT, LoopT>::getExitingBlocks(
    SmallVectorImpl<BlockT *> &ExitingBlocks) const {
  assert(!isInvalid() && "Loop not in a valid state!");
  for (const auto BB : blocks())
    for (auto *Succ : children<BlockT *>(BB))
      if (!contains(Succ)) {
        // Not in current loop? It must be an exit block.
        ExitingBlocks.push_back(BB);
        break;
      }
}

/// getExitingBlock - If getExitingBlocks would return exactly one block,
/// return that block. Otherwise return null.
template <class BlockT, class LoopT>
BlockT *LoopBase<BlockT, LoopT>::getExitingBlock() const {
  assert(!isInvalid() && "Loop not in a valid state!");
  auto notInLoop = [&](BlockT *BB) { return !contains(BB); };
  auto isExitBlock = [&](BlockT *BB, bool AllowRepeats) -> BlockT * {
    assert(!AllowRepeats && "Unexpected parameter value.");
    // Child not in current loop?  It must be an exit block.
    return any_of(children<BlockT *>(BB), notInLoop) ? BB : nullptr;
  };

  return find_singleton<BlockT>(blocks(), isExitBlock);
}

/// getExitBlocks - Return all of the successor blocks of this loop.  These
/// are the blocks _outside of the current loop_ which are branched to.
///
template <class BlockT, class LoopT>
void LoopBase<BlockT, LoopT>::getExitBlocks(
    SmallVectorImpl<BlockT *> &ExitBlocks) const {
  assert(!isInvalid() && "Loop not in a valid state!");
  for (const auto BB : blocks())
    for (auto *Succ : children<BlockT *>(BB))
      if (!contains(Succ))
        // Not in current loop? It must be an exit block.
        ExitBlocks.push_back(Succ);
}

/// getExitBlock - If getExitBlocks would return exactly one block,
/// return that block. Otherwise return null.
template <class BlockT, class LoopT>
std::pair<BlockT *, bool> getExitBlockHelper(const LoopBase<BlockT, LoopT> *L,
                                             bool Unique) {
  assert(!L->isInvalid() && "Loop not in a valid state!");
  auto notInLoop = [&](BlockT *BB,
                       bool AllowRepeats) -> std::pair<BlockT *, bool> {
    assert(AllowRepeats == Unique && "Unexpected parameter value.");
    return {!L->contains(BB) ? BB : nullptr, false};
  };
  auto singleExitBlock = [&](BlockT *BB,
                             bool AllowRepeats) -> std::pair<BlockT *, bool> {
    assert(AllowRepeats == Unique && "Unexpected parameter value.");
    return find_singleton_nested<BlockT>(children<BlockT *>(BB), notInLoop,
                                         AllowRepeats);
  };
  return find_singleton_nested<BlockT>(L->blocks(), singleExitBlock, Unique);
}

template <class BlockT, class LoopT>
bool LoopInfoBase<BlockT, LoopT>::hasNoExitBlocks(const LoopT &L) const {
  auto RC = getExitBlockHelper(&L, false);
  if (RC.second)
    // found multiple exit blocks
    return false;
  // return true if there is no exit block
  return !RC.first;
}

/// getExitBlock - If getExitBlocks would return exactly one block,
/// return that block. Otherwise return null.
template <class BlockT, class LoopT>
BlockT *LoopBase<BlockT, LoopT>::getExitBlock() const {
  return getExitBlockHelper(this, false).first;
}

template <class BlockT, class LoopT>
bool LoopBase<BlockT, LoopT>::hasDedicatedExits() const {
  // Each predecessor of each exit block of a normal loop is contained
  // within the loop.
  SmallVector<BlockT *, 4> UniqueExitBlocks;
  getUniqueExitBlocks(UniqueExitBlocks);
  for (BlockT *EB : UniqueExitBlocks)
    for (BlockT *Predecessor : inverse_children<BlockT *>(EB))
      if (!contains(Predecessor))
        return false;
  // All the requirements are met.
  return true;
}

// Helper function to get unique loop exits. Pred is a predicate pointing to
// BasicBlocks in a loop which should be considered to find loop exits.
template <class BlockT, class LoopT, typename PredicateT>
void getUniqueExitBlocksHelper(const LoopT *L,
                               SmallVectorImpl<BlockT *> &ExitBlocks,
                               PredicateT Pred) {
  assert(!L->isInvalid() && "Loop not in a valid state!");
  SmallPtrSet<BlockT *, 32> Visited;
  auto Filtered = make_filter_range(L->blocks(), Pred);
  for (BlockT *BB : Filtered)
    for (BlockT *Successor : children<BlockT *>(BB))
      if (!L->contains(Successor))
        if (Visited.insert(Successor).second)
          ExitBlocks.push_back(Successor);
}

template <class BlockT, class LoopT>
void LoopBase<BlockT, LoopT>::getUniqueExitBlocks(
    SmallVectorImpl<BlockT *> &ExitBlocks) const {
  getUniqueExitBlocksHelper(this, ExitBlocks,
                            [](const BlockT *BB) { return true; });
}

template <class BlockT, class LoopT>
void LoopBase<BlockT, LoopT>::getUniqueNonLatchExitBlocks(
    SmallVectorImpl<BlockT *> &ExitBlocks) const {
  const BlockT *Latch = getLoopLatch();
  assert(Latch && "Latch block must exists");
  getUniqueExitBlocksHelper(this, ExitBlocks,
                            [Latch](const BlockT *BB) { return BB != Latch; });
}

template <class BlockT, class LoopT>
BlockT *LoopBase<BlockT, LoopT>::getUniqueExitBlock() const {
  return getExitBlockHelper(this, true).first;
}

template <class BlockT, class LoopT>
BlockT *
LoopInfoBase<BlockT, LoopT>::getUniqueLatchExitBlock(const LoopT &L) const {
  BlockT *Latch = L.getLoopLatch();
  assert(Latch && "Latch block must exists");
  auto IsExitBlock = [&L](BlockT *BB, bool AllowRepeats) -> BlockT * {
    assert(!AllowRepeats && "Unexpected parameter value.");
    return !L.contains(BB) ? BB : nullptr;
  };
  return find_singleton<BlockT>(children<BlockT *>(Latch), IsExitBlock);
}

/// getExitEdges - Return all pairs of (_inside_block_,_outside_block_).
template <class BlockT, class LoopT>
void LoopInfoBase<BlockT, LoopT>::getExitEdges(
    const LoopT &L, SmallVectorImpl<Edge> &ExitEdges) const {
  for (const auto BB : L.blocks())
    for (auto *Succ : children<BlockT *>(BB))
      if (!L.contains(Succ))
        // Not in current loop? It must be an exit block.
        ExitEdges.emplace_back(BB, Succ);
}

namespace detail {
template <class BlockT>
using has_hoist_check = decltype(&BlockT::isLegalToHoistInto);

template <class BlockT>
using detect_has_hoist_check = llvm::is_detected<has_hoist_check, BlockT>;

/// SFINAE functions that dispatch to the isLegalToHoistInto member function or
/// return false, if it doesn't exist.
template <class BlockT> bool isLegalToHoistInto(BlockT *Block) {
  if constexpr (detect_has_hoist_check<BlockT>::value)
    return Block->isLegalToHoistInto();
  return false;
}
} // namespace detail

/// getLoopPreheader - If there is a preheader for this loop, return it.  A
/// loop has a preheader if there is only one edge to the header of the loop
/// from outside of the loop and it is legal to hoist instructions into the
/// predecessor. If this is the case, the block branching to the header of the
/// loop is the preheader node.
///
/// This method returns null if there is no preheader for the loop.
///
template <class BlockT, class LoopT>
BlockT *LoopBase<BlockT, LoopT>::getLoopPreheader() const {
  assert(!isInvalid() && "Loop not in a valid state!");
  // Keep track of nodes outside the loop branching to the header...
  BlockT *Out = getLoopPredecessor();
  if (!Out)
    return nullptr;

  // Make sure we are allowed to hoist instructions into the predecessor.
  if (!detail::isLegalToHoistInto(Out))
    return nullptr;

  // Make sure there is only one exit out of the preheader.
  if (!llvm::hasSingleElement(llvm::children<BlockT *>(Out)))
    return nullptr; // Multiple exits from the block, must not be a preheader.

  // The predecessor has exactly one successor, so it is a preheader.
  return Out;
}

/// getLoopPredecessor - If the given loop's header has exactly one unique
/// predecessor outside the loop, return it. Otherwise return null.
/// This is less strict that the loop "preheader" concept, which requires
/// the predecessor to have exactly one successor.
///
template <class BlockT, class LoopT>
BlockT *LoopBase<BlockT, LoopT>::getLoopPredecessor() const {
  assert(!isInvalid() && "Loop not in a valid state!");
  // Keep track of nodes outside the loop branching to the header...
  BlockT *Out = nullptr;

  // Loop over the predecessors of the header node...
  BlockT *Header = getHeader();
  for (const auto Pred : inverse_children<BlockT *>(Header)) {
    if (!contains(Pred)) { // If the block is not in the loop...
      if (Out && Out != Pred)
        return nullptr; // Multiple predecessors outside the loop
      Out = Pred;
    }
  }

  return Out;
}

/// getLoopLatch - If there is a single latch block for this loop, return it.
/// A latch block is a block that contains a branch back to the header.
template <class BlockT, class LoopT>
BlockT *LoopBase<BlockT, LoopT>::getLoopLatch() const {
  assert(!isInvalid() && "Loop not in a valid state!");
  BlockT *Header = getHeader();
  BlockT *Latch = nullptr;
  for (const auto Pred : inverse_children<BlockT *>(Header)) {
    if (contains(Pred)) {
      if (Latch)
        return nullptr;
      Latch = Pred;
    }
  }

  return Latch;
}

//===----------------------------------------------------------------------===//
// APIs for updating loop information after changing the CFG
//

/// addBasicBlockToLoop - This method is used by other analyses to update loop
/// information.  NewBB is set to be a new member of the current loop.
/// Because of this, it is added as a member of all parent loops, and is added
/// to the specified LoopInfo object as being in the current basic block.  It
/// is not valid to replace the loop header with this method.
///
template <class BlockT, class LoopT>
void LoopBase<BlockT, LoopT>::addBasicBlockToLoop(
    BlockT *NewBB, LoopInfoBase<BlockT, LoopT> &LIB) {
  assert(!isInvalid() && "Loop not in a valid state!");
#ifndef NDEBUG
  if (!getBlocks().empty()) {
    auto SameHeader = LIB[getHeader()];
    assert(contains(SameHeader) && getHeader() == SameHeader->getHeader() &&
           "Incorrect LI specified for this loop!");
  }
#endif
  assert(NewBB && "Cannot add a null basic block to the loop!");
  assert(!LIB[NewBB] && "BasicBlock already in the loop!");

  LoopT *L = static_cast<LoopT *>(this);

  // Add the loop mapping to the LoopInfo object...
  LIB.changeLoopFor(NewBB, L);

  // Add the basic block to this loop and all parent loops...
  while (L) {
    L->addBlockEntry(NewBB);
    L = L->getParentLoop();
  }
}

/// replaceChildLoopWith - This is used when splitting loops up.  It replaces
/// the OldChild entry in our children list with NewChild, and updates the
/// parent pointer of OldChild to be null and the NewChild to be this loop.
/// This updates the loop depth of the new child.
template <class BlockT, class LoopT>
void LoopBase<BlockT, LoopT>::replaceChildLoopWith(LoopT *OldChild,
                                                   LoopT *NewChild) {
  assert(!isInvalid() && "Loop not in a valid state!");
  assert(OldChild->ParentLoop == this && "This loop is already broken!");
  assert(!NewChild->ParentLoop && "NewChild already has a parent!");
  typename std::vector<LoopT *>::iterator I = find(SubLoops, OldChild);
  assert(I != SubLoops.end() && "OldChild not in loop!");
  *I = NewChild;
  OldChild->ParentLoop = nullptr;
  NewChild->ParentLoop = static_cast<LoopT *>(this);
}

/// verifyLoop - Verify loop structure
template <class BlockT, class LoopT>
void LoopBase<BlockT, LoopT>::verifyLoop() const {
  assert(!isInvalid() && "Loop not in a valid state!");
#ifndef NDEBUG
  assert(!getBlocks().empty() && "Loop header is missing");

  // Setup for using a depth-first iterator to visit every block in the loop.
  SmallVector<BlockT *, 8> ExitBBs;
  getExitBlocks(ExitBBs);
  df_iterator_default_set<BlockT *> VisitSet;
  VisitSet.insert(ExitBBs.begin(), ExitBBs.end());

  // Keep track of the BBs visited.
  SmallPtrSet<BlockT *, 8> VisitedBBs;

  // Check the individual blocks.
  for (BlockT *BB : depth_first_ext(getHeader(), VisitSet)) {
    assert(llvm::any_of(children<BlockT *>(BB),
                        [&](BlockT *B) { return contains(B); }) &&
           "Loop block has no in-loop successors!");

    assert(llvm::any_of(inverse_children<BlockT *>(BB),
                        [&](BlockT *B) { return contains(B); }) &&
           "Loop block has no in-loop predecessors!");

    SmallVector<BlockT *, 2> OutsideLoopPreds;
    for (BlockT *B : inverse_children<BlockT *>(BB))
      if (!contains(B))
        OutsideLoopPreds.push_back(B);

    if (BB == getHeader()) {
      assert(!OutsideLoopPreds.empty() && "Loop is unreachable!");
    } else if (!OutsideLoopPreds.empty()) {
      // A non-header loop block shouldn't be reachable from outside the loop,
      // though it is permitted if the predecessor is not itself actually
      // reachable.
      BlockT *EntryBB = &BB->getParent()->front();
      for (BlockT *CB : depth_first(EntryBB))
        for (unsigned i = 0, e = OutsideLoopPreds.size(); i != e; ++i)
          assert(CB != OutsideLoopPreds[i] &&
                 "Loop has multiple entry points!");
    }
    assert(BB != &getHeader()->getParent()->front() &&
           "Loop contains function entry block!");

    VisitedBBs.insert(BB);
  }

  if (VisitedBBs.size() != getNumBlocks()) {
    dbgs() << "The following blocks are unreachable in the loop: ";
    for (auto *BB : getBlocks()) {
      if (!VisitedBBs.count(BB)) {
        dbgs() << *BB << "\n";
      }
    }
    assert(false && "Unreachable block in loop");
  }

  // Check the subloops.
  for (iterator I = begin(), E = end(); I != E; ++I)
    // Each block in each subloop should be contained within this loop.
    for (block_iterator BI = (*I)->block_begin(), BE = (*I)->block_end();
         BI != BE; ++BI) {
      assert(contains(*BI) &&
             "Loop does not contain all the blocks of a subloop!");
    }

  // Check the parent loop pointer.
  if (ParentLoop) {
    assert(is_contained(ParentLoop->getSubLoops(), this) &&
           "Loop is not a subloop of its parent!");
  }
#endif
}

/// verifyLoop - Verify loop structure of this loop and all nested loops.
template <class BlockT, class LoopT>
void LoopBase<BlockT, LoopT>::verifyLoopNest(
    DenseSet<const LoopT *> *Loops) const {
  assert(!isInvalid() && "Loop not in a valid state!");
  Loops->insert(static_cast<const LoopT *>(this));
  // Verify this loop.
  verifyLoop();
  // Verify the subloops.
  for (iterator I = begin(), E = end(); I != E; ++I)
    (*I)->verifyLoopNest(Loops);
}

template <class BlockT, class LoopT>
void LoopBase<BlockT, LoopT>::print(raw_ostream &OS, bool Verbose,
                                    bool PrintNested, unsigned Depth) const {
  OS.indent(Depth * 2);
  if (static_cast<const LoopT *>(this)->isAnnotatedParallel())
    OS << "Parallel ";
  OS << "Loop at depth " << getLoopDepth() << " containing: ";

  BlockT *H = getHeader();
  for (unsigned i = 0; i < getBlocks().size(); ++i) {
    BlockT *BB = getBlocks()[i];
    if (!Verbose) {
      if (i)
        OS << ",";
      BB->printAsOperand(OS, false);
    } else {
      OS << '\n';
    }

    if (BB == H)
      OS << "<header>";
    if (isLoopLatch(BB))
      OS << "<latch>";
    if (isLoopExiting(BB))
      OS << "<exiting>";
    if (Verbose)
      BB->print(OS);
  }

  if (PrintNested) {
    OS << "\n";

    for (iterator I = begin(), E = end(); I != E; ++I)
      (*I)->print(OS, /*Verbose*/ false, PrintNested, Depth + 2);
  }
}

//===----------------------------------------------------------------------===//
/// Stable LoopInfo Analysis - Build a loop tree using stable iterators so the
/// result does / not depend on use list (block predecessor) order.
///

/// Analyze LoopInfo identifies the loops during a single forward depth-first
/// search of the CFG.
///
/// Then build a loop-contiguous reverse postorder for in-loops blocks. Lists
/// are header-first with each subloop's blocks contiguous, ordered by first
/// appearance in RPO; SubLoops keep program order, TopLevelLoops reverse
/// program order.
template <class BlockT, class LoopT>
void LoopInfoBase<BlockT, LoopT>::analyze(const DomTreeBase<BlockT> &DomTree) {
  analyze(DomTree.getRootNode()->getBlock()->getParent(),
          [&]() -> const DomTreeBase<BlockT> & { return DomTree; });
}

template <class BlockT, class LoopT>
void LoopInfoBase<BlockT, LoopT>::analyze(ParentT F) {
  DomTreeBase<BlockT> DomTree;
  analyze(F, [&]() -> const DomTreeBase<BlockT> & {
    DomTree.recalculate(*F);
    return DomTree;
  });
}

template <class BlockT, class LoopT>
void LoopInfoBase<BlockT, LoopT>::analyze(
    ParentT F, function_ref<const DomTreeBase<BlockT> &()> GetDomTree) {
  using BlockTraits = GraphTraits<BlockT *>;
  auto num = [](const BlockT *BB) {
    return GraphTraits<const BlockT *>::getNumber(BB);
  };

  ParentPtr = F;
  BlockNumberEpoch = GraphTraits<ParentT>::getNumberEpoch(ParentPtr);
  unsigned MaxNumber = GraphTraits<ParentT>::getMaxNumber(ParentPtr);

  // Sentinel block number meaning "no block".
  constexpr unsigned NoBlock = ~0u;
  // States during DFS (Unvisited, OffPath, >=FirstOnPath) and post-DFS
  // (IsHeader, IsReentered).
  constexpr unsigned Unvisited = 0;
  constexpr unsigned OffPath = 1;
  constexpr unsigned IsHeader = 2;
  constexpr unsigned IsReentered = 3;
  constexpr unsigned FirstOnPath = IsReentered + 1;

  // Per-block search state, indexed by block number.
  struct BlockInfo {
    // Unvisited. Spelled 0 to work around GCC 11 ICE.
    unsigned Pos = 0;
    // Block number of the innermost enclosing header; NoBlock if none. Set to
    // NoBlock when the block is visited, then woven by tagLoopHeader.
    unsigned LoopHeader = 0;
  };
  SmallVector<BlockInfo, 32> Info(MaxNumber);
  // The loop headers, repeated once per backedge.
  SmallVector<unsigned, 4> Headers;
  // The headers of the loops that an edge re-enters. They mark irreducible
  // loops that need to be reduced to natural loop subsets.
  DenseSet<unsigned> Reentries;

  // Weave loop header \p H (and its own header chain) into the loop header
  // chain of \p B, keeping the chain ordered from innermost to outermost by
  // search path position. Building this chain on the fly is why the algorithm
  // needs no union-find (used in the Havlak algorithm) at all.
  auto tagLoopHeader = [&](unsigned B, unsigned H) {
    assert(H != NoBlock);
    // Invariant: Info[B].Pos >= Info[H].Pos.
    while (B != H) {
      unsigned IH = Info[B].LoopHeader;
      if (IH == NoBlock) {
        // B's chain ended: append the rest of H's chain.
        Info[B].LoopHeader = H;
        return;
      }
      // Keep whichever candidate header is inner (larger search path position).
      if (Info[IH].Pos >= Info[H].Pos) {
        B = IH;
      } else {
        Info[B].LoopHeader = H;
        B = H;
        H = IH;
      }
    }
  };

  // Identify loops with the algorithm of Wei et al., "A New Algorithm for
  // Identifying Loops in Decompilation" (SAS 2007): tag each block with its
  // innermost enclosing header. It also records the postorder the layout below
  // needs.
  SmallVector<BlockT *, 32> Postorder;
  Postorder.reserve(MaxNumber);
  struct Frame {
    BlockT *Block;
    typename BlockTraits::ChildIteratorType Cur, End;
  };
  SmallVector<Frame, 8> Stack;
  unsigned Counter = FirstOnPath;
  auto open = [&](BlockT *BB) {
    unsigned B = num(BB);
    Info[B].Pos = Counter++;
    Info[B].LoopHeader = NoBlock;
    Stack.push_back(
        {BB, BlockTraits::child_begin(BB), BlockTraits::child_end(BB)});
  };

  open(GraphTraits<ParentT>::getEntryNode(ParentPtr));
  while (!Stack.empty()) {
    Frame &Top = Stack.back();
    if (Top.Cur == Top.End) {
      // Leave the search path, and weave into the parent's chain.
      unsigned B0 = num(Top.Block);
      Info[B0].Pos = OffPath;
      Postorder.push_back(Top.Block);
      Stack.pop_back();
      if (!Stack.empty() && Info[B0].LoopHeader != NoBlock)
        tagLoopHeader(num(Stack.back().Block), Info[B0].LoopHeader);
      continue;
    }
    BlockT *B0P = Top.Block;
    BlockT *B1P = *Top.Cur++;
    unsigned B1 = num(B1P);
    if (Info[B1].Pos == Unvisited) {
      // Tree edge; the weaving happens when B1's frame is popped.
      open(B1P);
    } else if (Info[B1].Pos >= FirstOnPath) {
      // Retreating edge, including a self edge: B1 heads a loop.
      Headers.push_back(B1);
      tagLoopHeader(num(B0P), B1);
    } else {
      // Climb B1's header chain: each enclosing header still off the DFS path
      // heads a closed cycle this edge re-enters, so B1 is a non-header entry
      // of it (and it is irreducible). Stop at the first on-path header and
      // attribute B0 to it.
      for (unsigned H = Info[B1].LoopHeader; H != NoBlock;
           H = Info[H].LoopHeader) {
        if (Info[H].Pos >= FirstOnPath) {
          tagLoopHeader(num(B0P), H);
          break;
        }
        Reentries.insert(H);
      }
    }
  }
  // Most functions have no loops; skip the layout construction.
  if (Headers.empty())
    return;
  // Every block is off the search path now, so marking the headers cannot be
  // mistaken for a position on it.
  for (unsigned H : Headers)
    Info[H].Pos = IsHeader;

  if (!Reentries.empty()) {
    // A re-entered loop has more than one entry, so it is not a natural loop.
    // Reduce it, innermost first, to the natural loop of its header's
    // backedges: a backward search from the latches finds the blocks to keep;
    // splice the header out of the chain of every other block.
    for (unsigned H : Reentries)
      Info[H].Pos = IsReentered;
    const DomTreeBase<BlockT> &DomTree = GetDomTree();
    assert(DomTree.getRootNode()->getBlock() ==
           GraphTraits<ParentT>::getEntryNode(ParentPtr));
    DomTree.updateDFSNumbers();
    SmallVector<unsigned, 0> Mark(MaxNumber, NoBlock);
    SmallVector<BlockT *, 8> Worklist;
    // Invert the chains into the loop forest, so that a header visits only its
    // own blocks.
    SmallVector<unsigned, 0> FirstChild(MaxNumber, NoBlock);
    SmallVector<unsigned, 0> NextSibling(MaxNumber, NoBlock);
    SmallVector<BlockT *, 0> Blocks(MaxNumber);
    for (BlockT *BB : Postorder) {
      unsigned B = num(BB);
      Blocks[B] = BB;
      if (unsigned P = Info[B].LoopHeader; P != NoBlock) {
        NextSibling[B] = FirstChild[P];
        FirstChild[P] = B;
      }
    }
    for (BlockT *Header : Postorder) {
      unsigned H = num(Header);
      if (Info[H].Pos != IsReentered)
        continue;
      Mark[H] = H;
      Worklist.clear();
      auto enqueue = [&](BlockT *Pred) {
        unsigned P = num(Pred);
        // If Pred is in a natural loop, mark its header and skip interior
        // blocks.
        for (unsigned A = P; A != NoBlock; A = Info[A].LoopHeader)
          if (Info[A].LoopHeader == H) {
            P = A;
            Pred = Blocks[A];
            break;
          }
        if (Mark[P] == H)
          return;
        Mark[P] = H;
        Worklist.push_back(Pred);
      };
      // Place the latches, the predecessors the header dominates, into a
      // worklist.
      const DomTreeNodeBase<BlockT> *DomNode = DomTree.getNode(Header);
      assert(DomNode && "header missing from the dominator tree");
      bool HasBackedge = false;
      for (BlockT *Pred : inverse_children<BlockT *>(Header)) {
        const DomTreeNodeBase<BlockT> *PredNode = DomTree.getNode(Pred);
        if (PredNode && DomTree.dominates(DomNode, PredNode)) {
          HasBackedge = true;
          enqueue(Pred);
        }
      }
      // Whatever reaches a latch without passing the header is in the loop.
      for (unsigned I = 0; I != Worklist.size(); ++I)
        for (BlockT *Pred : inverse_children<BlockT *>(Worklist[I]))
          enqueue(Pred);
      // Without a backedge the header forms no loop at all.
      Info[H].Pos = HasBackedge ? IsHeader : OffPath;
      // Partition the header's blocks: the loop keeps the ones the traversal
      // reached, and the enclosing header takes the rest, which its own turn
      // then tests. Both arms relink the block, so step first.
      unsigned Parent = Info[H].LoopHeader;
      unsigned Kept = NoBlock;
      for (unsigned B = FirstChild[H], Next; B != NoBlock; B = Next) {
        Next = NextSibling[B];
        if (Mark[B] == H) {
          NextSibling[B] = Kept;
          Kept = B;
        } else {
          // Leaving the loop; the block is top level if it had no other header.
          Info[B].LoopHeader = Parent;
          if (Parent != NoBlock) {
            NextSibling[B] = FirstChild[Parent];
            FirstChild[Parent] = B;
          }
        }
      }
      FirstChild[H] = Kept;
    }
    if (none_of(Headers, [&](unsigned H) { return Info[H].Pos == IsHeader; }))
      return;
  }

  // Resolve the chains in reverse postorder: a block's innermost header is
  // one of its search tree ancestors, so it is mapped to its loop first.
  BBMap.resize(MaxNumber);
  for (BlockT *BB : llvm::reverse(Postorder)) {
    unsigned B = num(BB);
    unsigned H = Info[B].LoopHeader;
    LoopT *Enclosing = H == NoBlock ? nullptr : BBMap[H];
    LoopT *L = Enclosing;
    if (Info[B].Pos == IsHeader) {
      L = allocateLoop(BB);
      L->setParentLoop(Enclosing);
    }
    BBMap[B] = L;
  }

  // Record each in-loop block with its innermost loop in forward CFG postorder,
  // and build the loop list in PO.
  SmallVector<std::pair<BlockT *, LoopT *>, 32> PO;
  SmallVector<LoopT *, 4> LoopsPO;
  PO.reserve(Postorder.size());
  for (BlockT *BB : Postorder) {
    LoopT *L = lookupLoopFor(BB);
    if (!L)
      continue;
    PO.emplace_back(BB, L);
    ++L->BlockLen;
    if (BB != pendingHeader(L))
      continue;
    LoopsPO.push_back(L);
    if (LoopT *Parent = L->getParentLoop())
      Parent->BlockLen += L->BlockLen;
    else
      TopLevelLoops.push_back(L);
  }
  // Headers are dominator-tree nodes, hence reachable and in the postorder.
  assert(!LoopsPO.empty() && "discovered loops but found no header");

  BlockLayout.reset(new BlockT *[PO.size()]);
  BlockT **RootCursor = BlockLayout.get();
  for (auto &[BB, L] : llvm::reverse(PO)) {
    if (L->BlockCapacity == 0) {
      // The first block of a L is its the header. Carve its slice from the
      // parent (already visited)'s cursor.
      if (LoopT *Parent = L->getParentLoop()) {
        assert(Parent->BlockCapacity != 0 &&
               "parent slice not carved before child");
        L->BlockData = Parent->BlockData + Parent->BlockCapacity;
        Parent->BlockCapacity += L->BlockLen;
        Parent->SubLoops.push_back(L);
      } else {
        L->BlockData = RootCursor;
        RootCursor += L->BlockLen;
      }
    }
    // Each block lands once, at its innermost loop's cursor.
    L->BlockData[L->BlockCapacity++] = BB;
  }

  // Mark every slice as borrowed from BlockLayout; a later mutation copies it
  // into private storage (see materializeBlocks).
  for (LoopT *L : LoopsPO) {
    assert(L->BlockCapacity == L->BlockLen && "layout slice not fully used");
    L->BlockCapacity = LoopT::BorrowedCapacity;
  }
}

template <class BlockT, class LoopT>
SmallVector<LoopT *, 4>
LoopInfoBase<BlockT, LoopT>::getLoopsInPreorder() const {
  SmallVector<LoopT *, 4> PreOrderLoops;
  // The outer-most loop actually goes into the result in the same relative
  // order as we walk it. But LoopInfo stores the top level loops in reverse
  // program order so for here we reverse it to get forward program order.
  // FIXME: If we change the order of LoopInfo we will want to remove the
  // reverse here.
  for (LoopT *RootL : reverse(*this)) {
    PreOrderLoops.push_back(RootL);
    LoopT::getInnerLoopsInPreorder(*RootL, PreOrderLoops);
  }

  return PreOrderLoops;
}

template <class BlockT, class LoopT>
SmallVector<LoopT *, 4>
LoopInfoBase<BlockT, LoopT>::getLoopsInReverseSiblingPreorder() const {
  SmallVector<LoopT *, 4> PreOrderLoops, PreOrderWorklist;
  // The outer-most loop actually goes into the result in the same relative
  // order as we walk it. LoopInfo stores the top level loops in reverse
  // program order so we walk in order here.
  // FIXME: If we change the order of LoopInfo we will want to add a reverse
  // here.
  for (LoopT *RootL : *this) {
    assert(PreOrderWorklist.empty() &&
           "Must start with an empty preorder walk worklist.");
    PreOrderWorklist.push_back(RootL);
    do {
      LoopT *L = PreOrderWorklist.pop_back_val();
      // Sub-loops are stored in forward program order, but will process the
      // worklist backwards so we can just append them in order.
      PreOrderWorklist.append(L->begin(), L->end());
      PreOrderLoops.push_back(L);
    } while (!PreOrderWorklist.empty());
  }

  return PreOrderLoops;
}

template <class BlockT, class LoopT>
LoopT *LoopInfoBase<BlockT, LoopT>::getSmallestCommonLoop(LoopT *A,
                                                          LoopT *B) const {
  if (!A || !B)
    return nullptr;

  // If loops A and B have different depth replace them with parent loop
  // until they have the same depth.
  unsigned DepthA = A->getLoopDepth(), DepthB = B->getLoopDepth();
  for (; DepthA > DepthB; --DepthA)
    A = A->getParentLoop();
  for (; DepthB > DepthA; --DepthB)
    B = B->getParentLoop();

  // Loops A and B are at same depth but may be disjoint, replace them with
  // parent loops until we find loop that contains both or we run out of
  // parent loops.
  while (A != B) {
    A = A->getParentLoop();
    B = B->getParentLoop();
  }

  return A;
}

template <class BlockT, class LoopT>
LoopT *LoopInfoBase<BlockT, LoopT>::getSmallestCommonLoop(BlockT *A,
                                                          BlockT *B) const {
  return getSmallestCommonLoop(getLoopFor(A), getLoopFor(B));
}

// Debugging
template <class BlockT, class LoopT>
void LoopInfoBase<BlockT, LoopT>::print(raw_ostream &OS) const {
  for (unsigned i = 0; i < TopLevelLoops.size(); ++i)
    TopLevelLoops[i]->print(OS);
}

template <typename T>
bool compareVectors(std::vector<T> &BB1, std::vector<T> &BB2) {
  llvm::sort(BB1);
  llvm::sort(BB2);
  return BB1 == BB2;
}

template <class BlockT, class LoopT>
void addInnerLoopsToHeadersMap(DenseMap<BlockT *, const LoopT *> &LoopHeaders,
                               const LoopInfoBase<BlockT, LoopT> &LI,
                               const LoopT &L) {
  LoopHeaders[L.getHeader()] = &L;
  for (LoopT *SL : L)
    addInnerLoopsToHeadersMap(LoopHeaders, LI, *SL);
}

#ifndef NDEBUG
template <class BlockT, class LoopT>
static void compareLoops(const LoopT *L, const LoopT *OtherL,
                         DenseMap<BlockT *, const LoopT *> &OtherLoopHeaders) {
  BlockT *H = L->getHeader();
  BlockT *OtherH = OtherL->getHeader();
  assert(H == OtherH &&
         "Mismatched headers even though found in the same map entry!");

  assert(L->getLoopDepth() == OtherL->getLoopDepth() &&
         "Mismatched loop depth!");
  const LoopT *ParentL = L, *OtherParentL = OtherL;
  do {
    assert(ParentL->getHeader() == OtherParentL->getHeader() &&
           "Mismatched parent loop headers!");
    ParentL = ParentL->getParentLoop();
    OtherParentL = OtherParentL->getParentLoop();
  } while (ParentL);

  for (const LoopT *SubL : *L) {
    BlockT *SubH = SubL->getHeader();
    const LoopT *OtherSubL = OtherLoopHeaders.lookup(SubH);
    assert(OtherSubL && "Inner loop is missing in computed loop info!");
    OtherLoopHeaders.erase(SubH);
    compareLoops(SubL, OtherSubL, OtherLoopHeaders);
  }

  std::vector<BlockT *> BBs = L->getBlocks();
  std::vector<BlockT *> OtherBBs = OtherL->getBlocks();
  assert(compareVectors(BBs, OtherBBs) &&
         "Mismatched basic blocks in the loops!");
}
#endif

template <class BlockT, class LoopT>
void LoopInfoBase<BlockT, LoopT>::verify() const {
  DenseSet<const LoopT *> Loops;
  for (iterator I = begin(), E = end(); I != E; ++I) {
    assert((*I)->isOutermost() && "Top-level loop has a parent!");
    (*I)->verifyLoopNest(&Loops);
  }

// Verify that blocks are mapped to valid loops.
#ifndef NDEBUG
  // Every loop must point back at this LoopInfo (see resetLoopInfoOwners).
  for (const LoopT *L : Loops)
    assert(L->LI == this && "Loop has a stale owning-LoopInfo back-pointer");

  // Recompute the innermost loop of each block from the loops' block lists,
  // which are maintained independently of BBMap. Using contains() here would
  // derive from BBMap itself and check nothing.
  SmallVector<const LoopT *> Innermost(BBMap.size());
  SmallVector<const LoopT *, 8> Worklist(begin(), end());
  while (!Worklist.empty()) {
    const LoopT *L = Worklist.pop_back_val();
    // A loop is visited before its children, so a child's blocks overwrite the
    // entries written by its ancestors.
    for (const BlockT *BB : L->getBlocks()) {
      unsigned Number = GraphTraits<const BlockT *>::getNumber(BB);
      assert(Number < Innermost.size() && "block missing from BBMap");
      Innermost[Number] = L;
    }
    Worklist.append(L->begin(), L->end());
  }

  for (auto [Number, L] : enumerate(BBMap)) {
    assert((!L || Loops.count(L)) && "orphaned loop");
    assert(L == Innermost[Number] &&
           "BBMap should point to the innermost loop containing the block");
  }

  // Recompute LoopInfo to verify loops structure.
  LoopInfoBase<BlockT, LoopT> OtherLI;
  OtherLI.analyze(ParentPtr);

  // Build a map we can use to move from our LI to the computed one. This
  // allows us to ignore the particular order in any layer of the loop forest
  // while still comparing the structure.
  DenseMap<BlockT *, const LoopT *> OtherLoopHeaders;
  for (LoopT *L : OtherLI)
    addInnerLoopsToHeadersMap(OtherLoopHeaders, OtherLI, *L);

  // Walk the top level loops and ensure there is a corresponding top-level
  // loop in the computed version and then recursively compare those loop
  // nests.
  for (LoopT *L : *this) {
    BlockT *Header = L->getHeader();
    const LoopT *OtherL = OtherLoopHeaders.lookup(Header);
    assert(OtherL && "Top level loop is missing in computed loop info!");
    // Now that we've matched this loop, erase its header from the map.
    OtherLoopHeaders.erase(Header);
    // And recursively compare these loops.
    compareLoops(L, OtherL, OtherLoopHeaders);
  }

  // Any remaining entries in the map are loops which were found when computing
  // a fresh LoopInfo but not present in the current one.
  if (!OtherLoopHeaders.empty()) {
    for (const auto &HeaderAndLoop : OtherLoopHeaders)
      dbgs() << "Found new loop: " << *HeaderAndLoop.second << "\n";
    llvm_unreachable("Found new loops when recomputing LoopInfo!");
  }
#endif
}

} // namespace llvm

#endif // LLVM_SUPPORT_GENERICLOOPINFOIMPL_H
