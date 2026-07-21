//===- GenericCycleImpl.h -------------------------------------*- C++ -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This template implementation resides in a separate file so that it
/// does not get injected into every .cpp file that includes the
/// generic header.
///
/// DO NOT INCLUDE THIS FILE WHEN MERELY USING CYCLEINFO.
///
/// This file should only be included by files that implement a
/// specialization of the relevant templates. Currently these are:
/// - llvm/lib/IR/CycleInfo.cpp
/// - llvm/lib/CodeGen/MachineCycleAnalysis.cpp
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_GENERICCYCLEIMPL_H
#define LLVM_ADT_GENERICCYCLEIMPL_H

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/GenericCycleInfo.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include <iterator>

#define DEBUG_TYPE "generic-cycle-impl"

namespace llvm {

template <typename ContextT>
void GenericCycleInfo<ContextT>::getExitBlocks(
    CycleRef C, SmallVectorImpl<BlockT *> &TmpStorage) const {
  if (ExitBlocksCaches.empty())
    ExitBlocksCaches.resize(NumCycles);
  auto &Cache = ExitBlocksCaches[C.Index];
  if (Cache.empty()) {
    SmallPtrSet<BlockT *, 4> Seen;
    for (BlockT *Block : getBlocks(C))
      for (BlockT *Succ : successors(Block))
        if (!contains(C, Succ) && Seen.insert(Succ).second)
          Cache.push_back(Succ);
  }
  TmpStorage.append(Cache.begin(), Cache.end());
}

template <typename ContextT>
void GenericCycleInfo<ContextT>::getExitingBlocks(
    CycleRef C, SmallVectorImpl<BlockT *> &TmpStorage) const {
  for (BlockT *Block : getBlocks(C)) {
    for (BlockT *Succ : successors(Block)) {
      if (!contains(C, Succ)) {
        TmpStorage.push_back(Block);
        break;
      }
    }
  }
}

template <typename ContextT>
auto GenericCycleInfo<ContextT>::getCyclePreheader(CycleRef C) const
    -> BlockT * {
  BlockT *Predecessor = getCyclePredecessor(C);
  if (!Predecessor)
    return nullptr;

  assert(isReducible(C) && "Cycle Predecessor must be in a reducible cycle!");

  if (succ_size(Predecessor) != 1)
    return nullptr;

  // Make sure we are allowed to hoist instructions into the predecessor.
  if (!Predecessor->isLegalToHoistInto())
    return nullptr;

  return Predecessor;
}

template <typename ContextT>
auto GenericCycleInfo<ContextT>::getCyclePredecessor(CycleRef C) const
    -> BlockT * {
  if (!isReducible(C))
    return nullptr;

  BlockT *Out = nullptr;

  // Loop over the predecessors of the header node...
  BlockT *Header = getHeader(C);
  for (const auto Pred : predecessors(Header)) {
    if (!contains(C, Pred)) {
      if (Out && Out != Pred)
        return nullptr;
      Out = Pred;
    }
  }

  return Out;
}

template <typename ContextT>
void GenericCycleInfo<ContextT>::verifyCycle(CycleRef C) const {
#ifndef NDEBUG
  assert(getNumBlocks(C) != 0 && "Cycle cannot be empty.");
  DenseSet<BlockT *> Blocks;
  for (BlockT *BB : getBlocks(C)) {
    assert(Blocks.insert(BB).second); // duplicates in block list?
  }
  assert(!getEntries(C).empty() && "Cycle must have one or more entries.");

  DenseSet<BlockT *> Entries;
  for (BlockT *Entry : getEntries(C)) {
    assert(Entries.insert(Entry).second); // duplicate entry?
    assert(contains(C, Entry));
  }

  // Setup for using a depth-first iterator to visit every block in the cycle.
  SmallVector<BlockT *, 8> ExitBBs;
  getExitBlocks(C, ExitBBs);
  df_iterator_default_set<BlockT *> VisitSet;
  VisitSet.insert(ExitBBs.begin(), ExitBBs.end());

  // Keep track of the BBs visited.
  SmallPtrSet<BlockT *, 8> VisitedBBs;

  // Check the individual blocks.
  for (BlockT *BB : depth_first_ext(getHeader(C), VisitSet)) {
    assert(llvm::any_of(llvm::children<BlockT *>(BB),
                        [&](BlockT *B) { return contains(C, B); }) &&
           "Cycle block has no in-cycle successors!");

    assert(llvm::any_of(llvm::inverse_children<BlockT *>(BB),
                        [&](BlockT *B) { return contains(C, B); }) &&
           "Cycle block has no in-cycle predecessors!");

    DenseSet<BlockT *> OutsideCyclePreds;
    for (BlockT *B : llvm::inverse_children<BlockT *>(BB))
      if (!contains(C, B))
        OutsideCyclePreds.insert(B);

    if (Entries.contains(BB)) {
      assert(!OutsideCyclePreds.empty() && "Entry is unreachable!");
    } else if (!OutsideCyclePreds.empty()) {
      // A non-entry block shouldn't be reachable from outside the cycle,
      // though it is permitted if the predecessor is not itself actually
      // reachable.
      BlockT *EntryBB = &BB->getParent()->front();
      for (BlockT *CB : depth_first(EntryBB))
        assert(!OutsideCyclePreds.contains(CB) &&
               "Non-entry block reachable from outside!");
    }
    assert(BB != &getHeader(C)->getParent()->front() &&
           "Cycle contains function entry block!");

    VisitedBBs.insert(BB);
  }

  if (VisitedBBs.size() != getNumBlocks(C)) {
    dbgs() << "The following blocks are unreachable in the cycle:\n  ";
    ListSeparator LS;
    for (auto *BB : Blocks) {
      if (!VisitedBBs.count(BB)) {
        dbgs() << LS;
        BB->printAsOperand(dbgs());
      }
    }
    dbgs() << "\n";
    llvm_unreachable("Unreachable block in cycle");
  }

  verifyCycleNest(C);
#endif
}

template <typename ContextT>
void GenericCycleInfo<ContextT>::verifyCycleNest(CycleRef C) const {
#ifndef NDEBUG
  const CycleT &Cyc = deref(C);
  // Check the subcycles.
  for (auto Child : children(C)) {
    // Each block in each subcycle should be contained within this cycle.
    for (BlockT *BB : getBlocks(Child)) {
      assert(contains(C, BB) &&
             "Cycle does not contain all the blocks of a subcycle!");
    }
    assert(deref(Child).Depth == Cyc.Depth + 1);
  }

  // Check the parent cycle.
  if (Cyc.hasParent()) {
    assert(is_contained(children(Cyc.Parent), C) &&
           "Cycle is not a subcycle of its parent!");
  }
#endif
}

/// \brief Helper class for computing cycle information.
template <typename ContextT> class GenericCycleInfoCompute {
  using BlockT = typename ContextT::BlockT;
  using FunctionT = typename ContextT::FunctionT;
  using CycleInfoT = GenericCycleInfo<ContextT>;
  using CycleT = typename CycleInfoT::CycleT;

  CycleInfoT &Info;

  // Sentinel header-preorder rank meaning "no cycle".
  static constexpr unsigned NoCycle = ~0u;
  // Sentinel block number meaning "no block".
  static constexpr unsigned NoBlock = ~0u;

  // Per-block state indexed by block number. All fields default to zero.
  struct BlockInfo {
    // The block this entry describes (non-null once visited by DFS), packed
    // with a bit for whether it heads a loop.
    PointerIntPair<BlockT *, 1, bool> BlockAndHeader;
    union {
      // (Live only during DFS) 1-based position on the current DFS path; 0 if
      // off path.
      unsigned Pos = 0;
      // (Live after DFS) Header-preorder rank of the innermost
      // loop containing this block (the loop it heads if IsHeader); NoCycle if
      // none.
      unsigned LoopIdx;
    };
    // Block number of the innermost loop header; NoBlock if none. Set to
    // NoBlock by open() on first visit, then woven by tagLoopHeader.
    unsigned LoopHeader = 0;

    BlockT *getBlock() const { return BlockAndHeader.getPointer(); }
    bool isHeader() const { return BlockAndHeader.getInt(); }
    void setHeader() { BlockAndHeader.setInt(true); }
    // An unvisited entry is all-zero,
    bool visited() const { return BlockAndHeader.getOpaqueValue() != nullptr; }
  };

  // Per-cycle scratch built in run() and consumed by flatten(), keyed by
  // header-preorder rank.
  struct CycleBuild {
    BlockT *Header;
    unsigned ChildHead;
    unsigned NextSibling;
    unsigned OwnCount;
  };

  SmallVector<BlockInfo, 8> BlockInfos;
  // Reachable block numbers in DFS preorder.
  SmallVector<unsigned, 8> Preorder;
  // Number of loop headers found by dfs().
  unsigned NumHeaders = 0;
  // Records (header H, block B): an edge from outside re-enters the closed
  // cycle headed by H at B, making B a non-header entry of it.
  SmallVector<std::pair<unsigned, unsigned>, 8> Reentries;

  GenericCycleInfoCompute(const GenericCycleInfoCompute &) = delete;
  GenericCycleInfoCompute &operator=(const GenericCycleInfoCompute &) = delete;

  static unsigned num(const BlockT *B) {
    return GraphTraits<const BlockT *>::getNumber(B);
  }

  BlockInfo &info(unsigned Number) { return BlockInfos[Number]; }

  // Weave loop header \p H (and its own header chain) into the loop header
  // chain of \p B, keeping the chain ordered from innermost to outermost by
  // DFS-path position. Building this chain on the fly is why the algorithm
  // needs no union-find (used in the Havlak algorithm) at all.
  void tagLoopHeader(unsigned B, unsigned H) {
    assert(H != NoBlock);
    // Invariant: info(B).Pos >= info(H).Pos.
    while (B != H) {
      unsigned IH = info(B).LoopHeader;
      if (IH == NoBlock) {
        // B's chain ended: append the rest of H's chain.
        info(B).LoopHeader = H;
        return;
      }
      // Keep whichever candidate header is inner (larger DFS-path position).
      if (info(IH).Pos >= info(H).Pos)
        B = IH;
      else {
        info(B).LoopHeader = H;
        B = H;
        H = IH;
      }
    }
  }

  void dfs(BlockT *EntryBlock);
  void flatten(ArrayRef<CycleBuild> Build, unsigned TopHead);

public:
  GenericCycleInfoCompute(CycleInfoT &Info) : Info(Info) {}

  void run(FunctionT *F);
};

template <typename ContextT>
void GenericCycleInfo<ContextT>::addToBlockMap(BlockT *Block, CycleRef C) {
  // The caller should ensure that BlockMap is large enough. C is a flat
  // cycle, so its preorder index is well-defined.
  verifyBlockNumberEpoch(Block->getParent());
  unsigned Number = GraphTraits<BlockT *>::getNumber(Block);
  BlockMap[Number] = C;
}

template <typename ContextT>
void GenericCycleInfo<ContextT>::addBlockToCycle(BlockT *Block, CycleRef C) {
  CycleT &Cyc = deref(C);
  // Make sure BlockMap is large enough for the new block.
  unsigned Number = GraphTraits<BlockT *>::getNumber(Block);
  if (Number >= BlockMap.size())
    BlockMap.resize(GraphTraits<FunctionT *>::getMaxNumber(Block->getParent()),
                    CycleRef());

  // Insert Block at the end of Cyc's slice and shift every later cycle's
  // range right. Ranges straddling Pos belong to Cyc's ancestors and are
  // extended below.
  unsigned Pos = Cyc.IdxEnd;
  BlockLayout.insert(BlockLayout.begin() + Pos, Block);
  for (unsigned I = 0; I != NumCycles; ++I) {
    CycleT &X = Cycles[I];
    if (X.IdxBegin >= Pos) {
      ++X.IdxBegin;
      ++X.IdxEnd;
    }
  }
  addToBlockMap(Block, C);
  // Cyc and its ancestors gain the new block: extend each one's slice and
  // invalidate its exit-block cache in a single walk up the tree.
  for (CycleRef I = C; I; I = deref(I).Parent) {
    ++deref(I).IdxEnd;
    if (!ExitBlocksCaches.empty())
      ExitBlocksCaches[I.Index].clear();
  }
}

/// Lay the discovered cycle forest out into Info's flat preorder array: number
/// the cycles in Euler-tour order, set each one's parent, depth and descendant
/// count, place every block into its innermost cycle's region of BlockLayout,
/// and fill BlockMap. \p Build is keyed by header-preorder rank.
template <typename ContextT>
void GenericCycleInfoCompute<ContextT>::flatten(ArrayRef<CycleBuild> Build,
                                                unsigned TopHead) {
  unsigned N = Build.size();
  Info.NumCycles = N;
  Info.Cycles = std::make_unique<CycleT[]>(N);

  // Walk the cycle forest as an Euler tour. On entry a cycle reserves [Cursor,
  // Cursor + OwnCount) for its own blocks (IdxBegin temporarily holds that
  // region's end; the fill loop below walks it back down); its descendants take
  // the following slots, so on exit Cursor is its IdxEnd.
  SmallVector<unsigned, 8> FlatIdx(N);
  struct Frame {
    unsigned Flat;
    unsigned Child; // Next child to enter, NoCycle once exhausted.
  };
  SmallVector<Frame, 8> Stack;
  unsigned Cursor = 0, NextID = 0;
  auto enter = [&](unsigned C, CycleRef Parent) {
    unsigned ID = NextID++;
    FlatIdx[C] = ID;
    CycleT &Flat = Info.Cycles[ID];
    Flat.Parent = Parent;
    Flat.Depth = Parent ? Info.deref(Parent).Depth + 1 : 1;
    Flat.appendEntry(Build[C].Header);
    Cursor += Build[C].OwnCount;
    Flat.IdxBegin = Cursor;
    Stack.push_back({ID, Build[C].ChildHead});
  };
  for (auto TLC = TopHead; TLC != NoCycle; TLC = Build[TLC].NextSibling) {
    enter(TLC, CycleRef());
    while (!Stack.empty()) {
      Frame &F = Stack.back();
      if (F.Child != NoCycle) {
        unsigned C = F.Child;
        F.Child = Build[C].NextSibling;
        enter(C, CycleRef(F.Flat));
      } else {
        CycleT &Flat = Info.Cycles[F.Flat];
        Flat.IdxEnd = Cursor;
        Flat.NumDescendants = NextID - F.Flat - 1;
        Stack.pop_back();
      }
    }
  }

  // Place every block into its innermost cycle's own region.
  Info.BlockLayout.resize_for_overwrite(Cursor);
  for (unsigned N : llvm::reverse(Preorder)) {
    BlockInfo &BI = info(N);
    if (BI.LoopIdx == NoCycle)
      continue;
    unsigned Flat = FlatIdx[BI.LoopIdx];
    Info.BlockMap[N] = CycleRef(Flat);
    Info.BlockLayout[--Info.Cycles[Flat].IdxBegin] = BI.getBlock();
  }
}

/// \brief Main function of the cycle info computations.
template <typename ContextT>
void GenericCycleInfoCompute<ContextT>::run(FunctionT *F) {
  BlockT *EntryBlock = GraphTraits<FunctionT *>::getEntryNode(F);
  BlockInfos.assign(GraphTraits<FunctionT *>::getMaxNumber(F), BlockInfo{});

  dfs(EntryBlock);
  if (!NumHeaders)
    return;

  // Number the cycles by their header's preorder rank and resolve every
  // block's innermost cycle in one pass: a block's LoopHeader is a DFS
  // ancestor and so already numbered, and parents get smaller ranks than
  // their children.
  SmallVector<CycleBuild, 8> Build;
  // Exact reserve so the Head reference below survives each push_back.
  Build.reserve(NumHeaders);
  unsigned TopHead = NoCycle;
  for (unsigned N : Preorder) {
    BlockInfo &BI = info(N);
    if (BI.isHeader()) {
      unsigned I = Build.size();
      BI.LoopIdx = I;
      unsigned &Head = BI.LoopHeader != NoBlock
                           ? Build[info(BI.LoopHeader).LoopIdx].ChildHead
                           : TopHead;
      Build.push_back({BI.getBlock(), NoCycle, Head, 1}); // OwnCount 1: header.
      Head = I;
      LLVM_DEBUG(dbgs() << "Found cycle for header: "
                        << Info.Context.print(BI.getBlock()) << "\n");
    } else if (BI.LoopHeader != NoBlock) {
      BI.LoopIdx = info(BI.LoopHeader).LoopIdx;
      ++Build[BI.LoopIdx].OwnCount;
    } else {
      BI.LoopIdx = NoCycle;
    }
  }
  flatten(Build, TopHead);
  if (Reentries.empty())
    return;

  // Add the non-header entries recorded during the DFS. Sorting by (header,
  // block) groups each cycle's entries together and in block preorder; a block
  // may re-enter a cycle via several edges, so skip duplicates.
  SmallVector<unsigned, 8> Rank(BlockInfos.size());
  for (auto [R, N] : enumerate(Preorder))
    Rank[N] = R;
  for (auto &[H, B] : Reentries)
    B = Rank[B];
  llvm::sort(Reentries);
  for (unsigned I = 0, E = Reentries.size(); I != E; ++I) {
    if (I && Reentries[I] == Reentries[I - 1])
      continue;
    auto [H, R] = Reentries[I];
    Info.deref(Info.BlockMap[H]).appendEntry(info(Preorder[R]).getBlock());
  }
}

/// Identify (possibly irreducible) loops using a single-pass DFS algorithm of
/// "A New Algorithm for Identifying Loops in Decompilation" (SAS 2007). The
/// cycle forest is then reconstructed from the per-block header tags.
template <typename ContextT>
void GenericCycleInfoCompute<ContextT>::dfs(BlockT *EntryBlock) {
  // Successors are visited in reverse order to match the legacy
  // single-LIFO-stack traversal, keeping cycle identification and block order
  // unchanged.
  using SuccIt = decltype(successors(EntryBlock).begin());
  struct Frame {
    unsigned Block;
    std::reverse_iterator<SuccIt> Cur, End;
  };
  SmallVector<Frame, 8> Stack;
  unsigned Counter = 0;
  Preorder.resize_for_overwrite(BlockInfos.size());

  auto open = [&](BlockT *Block) {
    unsigned N = num(Block);
    Preorder[Counter] = N;
    BlockInfo &BI = info(N);
    BI.BlockAndHeader.setPointerAndInt(Block, false);
    BI.Pos = ++Counter;
    BI.LoopHeader = NoBlock;
    auto Succs = successors(Block);
    Stack.push_back({N, std::make_reverse_iterator(Succs.end()),
                     std::make_reverse_iterator(Succs.begin())});
  };

  open(EntryBlock);
  while (!Stack.empty()) {
    Frame &Top = Stack.back();
    if (Top.Cur != Top.End) {
      unsigned B0 = Top.Block;
      BlockT *B1P = *Top.Cur++;
      unsigned B1 = num(B1P);
      BlockInfo &B1Info = info(B1);
      if (!B1Info.visited()) {
        // Tree edge; the weaving happens when B1's frame is popped.
        open(B1P);
      } else if (B1Info.Pos > 0) {
        // B1 is a loop header (including self-edge).
        if (!B1Info.isHeader()) {
          B1Info.setHeader();
          ++NumHeaders;
        }
        tagLoopHeader(B0, B1);
      } else {
        // Climb B1's header chain: each enclosing header still off the DFS path
        // heads a closed cycle this edge re-enters, so B1 is a non-header entry
        // of it (and it is irreducible). Stop at the first on-path header and
        // attribute B0 to it.
        for (unsigned H = B1Info.LoopHeader; H != NoBlock;
             H = info(H).LoopHeader) {
          if (info(H).Pos > 0) {
            tagLoopHeader(B0, H);
            break;
          }
          Reentries.push_back({H, B1});
        }
      }
    } else {
      // Leave the DFS path.
      unsigned B0 = Top.Block;
      info(B0).Pos = 0;
      Stack.pop_back();
      // And weave into the parent's chain (continue the "Tree edge" case).
      if (!Stack.empty() && info(B0).LoopHeader != NoBlock)
        tagLoopHeader(Stack.back().Block, info(B0).LoopHeader);
    }
  }
  Preorder.truncate(Counter);
}

/// \brief Reset the object to its initial state.
template <typename ContextT> void GenericCycleInfo<ContextT>::clear() {
  BlockMap.clear();
  BlockLayout.clear();
  Cycles.reset();
  NumCycles = 0;
  ExitBlocksCaches.clear();
}

/// \brief Compute the cycle info for a function.
template <typename ContextT>
void GenericCycleInfo<ContextT>::compute(FunctionT &F) {
  GenericCycleInfoCompute<ContextT> Compute(*this);
  Context = ContextT(&F);
  BlockNumberEpoch = GraphTraits<FunctionT *>::getNumberEpoch(&F);
  BlockMap.assign(GraphTraits<FunctionT *>::getMaxNumber(&F), CycleRef());

  LLVM_DEBUG(dbgs() << "Computing cycles for function: " << F.getName()
                    << "\n");
  Compute.run(&F);
}

template <typename ContextT>
void GenericCycleInfo<ContextT>::splitCriticalEdge(BlockT *Pred, BlockT *Succ,
                                                   BlockT *NewBlock) {
  // Edge Pred-Succ is replaced by edges Pred-NewBlock and NewBlock-Succ, all
  // cycles that had blocks Pred and Succ also get NewBlock.
  CycleRef C = getSmallestCommonCycle(getCycle(Pred), getCycle(Succ));
  if (!C)
    return;

  addBlockToCycle(NewBlock, C);
  verifyCycleNest();
}

/// \brief Find the innermost cycle containing both given cycles.
///
/// \returns the innermost cycle containing both \p A and \p B
///          or nullptr if there is no such cycle.
template <typename ContextT>
auto GenericCycleInfo<ContextT>::getSmallestCommonCycle(CycleRef A,
                                                        CycleRef B) const
    -> CycleRef {
  if (!A || !B)
    return CycleRef();

  // If cycles A and B have different depth replace them with parent cycle
  // until they have the same depth.
  while (getDepth(A) > getDepth(B))
    A = getParentCycle(A);
  while (getDepth(B) > getDepth(A))
    B = getParentCycle(B);

  // Cycles A and B are at same depth but may be disjoint, replace them with
  // parent cycles until we find cycle that contains both or we run out of
  // parent cycles.
  while (A != B) {
    A = getParentCycle(A);
    B = getParentCycle(B);
  }

  return A;
}

/// \brief Find the innermost cycle containing both given blocks.
///
/// \returns the innermost cycle containing both \p A and \p B
///          or nullptr if there is no such cycle.
template <typename ContextT>
auto GenericCycleInfo<ContextT>::getSmallestCommonCycle(BlockT *A,
                                                        BlockT *B) const
    -> CycleRef {
  return getSmallestCommonCycle(getCycle(A), getCycle(B));
}

/// \brief Verify the internal consistency of the cycle tree.
///
/// Note that this does \em not check that cycles are really cycles in the CFG,
/// or that the right set of cycles in the CFG were found.
template <typename ContextT>
void GenericCycleInfo<ContextT>::verifyCycleNest(bool VerifyFull) const {
#ifndef NDEBUG
  DenseSet<BlockT *> CycleHeaders;

  for (auto C : cycles()) {
    BlockT *Header = getHeader(C);
    assert(CycleHeaders.insert(Header).second);
    if (VerifyFull)
      verifyCycle(C);
    else
      verifyCycleNest(C);
    // Check the block map entries for blocks contained in this cycle.
    for (BlockT *BB : getBlocks(C)) {
      CycleRef InBlockMap = getCycle(BB);
      assert(InBlockMap.isValid());
      assert(contains(C, InBlockMap));
    }
  }
#endif
}

/// \brief Verify that the entire cycle tree well-formed.
template <typename ContextT> void GenericCycleInfo<ContextT>::verify() const {
  verifyCycleNest(/*VerifyFull=*/true);
}

/// \brief Print the cycle info.
template <typename ContextT>
void GenericCycleInfo<ContextT>::print(raw_ostream &Out) const {
  for (auto C : cycles()) {
    for (unsigned I = 0, Depth = getDepth(C); I < Depth; ++I)
      Out << "    ";

    Out << print(C) << '\n';
  }
}

/// \brief Print a single cycle: its depth, entries, and remaining blocks.
template <typename ContextT>
Printable GenericCycleInfo<ContextT>::print(CycleRef C) const {
  return Printable([this, C](raw_ostream &Out) {
    Out << "depth=" << getDepth(C) << ": entries(" << printEntries(C, Context)
        << ')';

    for (auto *Block : getBlocks(C)) {
      if (isEntry(C, Block))
        continue;

      Out << ' ' << Context.print(Block);
    }
  });
}

} // namespace llvm

#undef DEBUG_TYPE

#endif // LLVM_ADT_GENERICCYCLEIMPL_H
