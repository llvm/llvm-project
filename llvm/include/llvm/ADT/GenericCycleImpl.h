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
#include "llvm/ADT/StringExtras.h"
#include <deque>
#include <iterator>

#define DEBUG_TYPE "generic-cycle-impl"

namespace llvm {

template <typename ContextT>
void GenericCycleInfo<ContextT>::getExitBlocks(
    CycleRef C, SmallVectorImpl<BlockT *> &TmpStorage) const {
  if (ExitBlocksCaches.empty())
    ExitBlocksCaches.resize(NumCycles);
  auto &Cache = ExitBlocksCaches[C.Index];
  if (!Cache.empty()) {
    TmpStorage.append(Cache.begin(), Cache.end());
    return;
  }

  size_t NumExitBlocks = 0;
  for (BlockT *Block : getBlocks(C)) {
    llvm::append_range(Cache, successors(Block));

    for (size_t Idx = NumExitBlocks, End = Cache.size(); Idx < End; ++Idx) {
      BlockT *Succ = Cache[Idx];
      if (!contains(C, Succ)) {
        auto ExitEndIt = Cache.begin() + NumExitBlocks;
        if (std::find(Cache.begin(), ExitEndIt, Succ) == ExitEndIt)
          Cache[NumExitBlocks++] = Succ;
      }
    }

    Cache.resize(NumExitBlocks);
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
    assert(is_contained(children(CycleRef(Cyc.ParentIndex)), C) &&
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

  struct DFSInfo {
    unsigned Start = 0; // DFS start; positive if block is found
    unsigned End = 0;   // DFS end

    DFSInfo() = default;
    explicit DFSInfo(unsigned Start) : Start(Start) {}

    explicit operator bool() const { return Start; }

    /// Whether this node is an ancestor (or equal to) the node \p Other
    /// in the DFS tree.
    bool isAncestorOf(const DFSInfo &Other) const {
      return Start <= Other.Start && Other.End <= End;
    }
  };

  // Indexed by block number.
  SmallVector<DFSInfo, 8> BlockDFSInfo;
  SmallVector<BlockT *, 8> BlockPreorder;

  /// Append-only cycles discovered so far, in creation order.
  std::deque<CycleT> AllCycles;

  /// Flat log of child attachments, in attach order. Cycles are only ever
  /// attached to the newest cycle, so each cycle's children occupy the
  /// contiguous slice [IdxBegin, Depth) of this log.
  SmallVector<CycleT *, 8> AttachedChildren;

  SmallVector<CycleT *, 8> TopLevelCycles;

  GenericCycleInfoCompute(const GenericCycleInfoCompute &) = delete;
  GenericCycleInfoCompute &operator=(const GenericCycleInfoCompute &) = delete;

  DFSInfo getDFSInfo(BlockT *B) const {
    unsigned Number = GraphTraits<BlockT *>::getNumber(B);
    return BlockDFSInfo[Number];
  }

  DFSInfo &getOrInsertDFSInfo(BlockT *B) {
    unsigned Number = GraphTraits<BlockT *>::getNumber(B);
    return BlockDFSInfo[Number];
  }

  /// Make top-level cycle \p Child a child of \p NewParent.
  void moveTopLevelCycleToNewParent(CycleT *NewParent, CycleT *Child) {
    assert((!Child->hasParent() && !NewParent->hasParent()) &&
           "NewParent and Child must be both top level cycle!\n");
    auto Pos = llvm::find(TopLevelCycles, Child);
    assert(Pos != TopLevelCycles.end());
    *Pos = TopLevelCycles.back();
    TopLevelCycles.pop_back();
    AttachedChildren.push_back(Child);
    // NewParent is the newest cycle and its creation-order index is
    // AllCycles.size() - 1.
    assert(NewParent == &AllCycles.back() &&
           "attach slices in AttachedChildren must stay contiguous");
    Child->ParentIndex = AllCycles.size() - 1;
  }

  /// Record that \p Block's innermost cycle is the one currently being built
  /// (always AllCycles.back()), storing its creation-order index. flatten()
  /// remaps it to the preorder index.
  void recordInnermostCycle(BlockT *Block) {
    Info.BlockMap[GraphTraits<BlockT *>::getNumber(Block)] =
        AllCycles.size() - 1;
  }

public:
  GenericCycleInfoCompute(CycleInfoT &Info) : Info(Info) {}

  void run(FunctionT *F);

private:
  void dfs(FunctionT *F, BlockT *EntryBlock);
  void flatten(ArrayRef<BlockT *> Order);
};

template <typename ContextT>
void GenericCycleInfo<ContextT>::addToBlockMap(BlockT *Block, CycleT *Cycle) {
  // The caller should ensure that BlockMap is large enough. \p Cycle is a flat
  // cycle, so its preorder index is well-defined.
  verifyBlockNumberEpoch(Block->getParent());
  unsigned Number = GraphTraits<BlockT *>::getNumber(Block);
  BlockMap[Number] = getCycleIndex(*Cycle);
}

template <typename ContextT>
void GenericCycleInfo<ContextT>::addBlockToCycle(BlockT *Block, CycleRef C) {
  CycleT &Cyc = deref(C);
  // Make sure BlockMap is large enough for the new block.
  unsigned Number = GraphTraits<BlockT *>::getNumber(Block);
  if (Number >= BlockMap.size())
    BlockMap.resize(GraphTraits<FunctionT *>::getMaxNumber(Block->getParent()),
                    NoCycle);

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
  addToBlockMap(Block, &Cyc);
  // Cyc and its ancestors gain the new block: extend each one's slice and
  // invalidate its exit-block cache in a single walk up the tree.
  for (unsigned I = getCycleIndex(Cyc); I != NoCycle;
       I = Cycles[I].ParentIndex) {
    ++Cycles[I].IdxEnd;
    if (!ExitBlocksCaches.empty())
      ExitBlocksCaches[I].clear();
  }
}

/// Move the discovered forest into Info's flat preorder array. Assigns preorder
/// IDs, depths and descendant counts, remaps BlockMap from creation-order to
/// preorder indices, and lays out every cycle's blocks in BlockLayout.
template <typename ContextT>
void GenericCycleInfoCompute<ContextT>::flatten(ArrayRef<BlockT *> Order) {
  unsigned N = AllCycles.size();
  Info.NumCycles = N;
  if (!N)
    return;
  Info.Cycles = std::make_unique<CycleT[]>(N);

  // Walk the cycle forest as an Euler tour. On entry, a cycle's IdxEnd still
  // holds its own-block count (accumulated during run()); reserve that many
  // slots for its own region [Cursor, Cursor + count). Its children take the
  // following slots, so on leaving, Cursor is the cycle's real range end, which
  // overwrites the now-consumed count in IdxEnd.
  struct Frame {
    CycleT *Flat;
    unsigned ChildCur, ChildEnd;
    unsigned ID;
  };
  SmallVector<Frame, 8> Stack;
  unsigned Cursor = 0;
  unsigned NextID = 0;
  auto enter = [&](CycleT *Temp, CycleT *Parent) {
    unsigned ID = NextID++;
    CycleT &Flat = Info.Cycles[ID];
    Flat.ParentIndex =
        Parent ? Info.getCycleIndex(*Parent) : CycleInfoT::NoCycle;
    Flat.Depth = Parent ? Parent->Depth + 1 : 1;
    Flat.Entries = std::move(Temp->Entries);
    Cursor += Temp->IdxEnd; // IdxEnd currently holds Temp's own-block count.
    Flat.IdxBegin = Cursor; // Real begin restored by the fill loop below.
    Stack.push_back({&Flat, Temp->IdxBegin, Temp->Depth, ID});
    // Temp's IdxBegin now holds the flat index, for the BlockMap remap below.
    Temp->IdxBegin = ID;
  };
  for (CycleT *TLC : TopLevelCycles) {
    enter(TLC, nullptr);
    while (!Stack.empty()) {
      Frame &F = Stack.back();
      if (F.ChildCur != F.ChildEnd) {
        enter(AttachedChildren[F.ChildCur++], F.Flat);
      } else {
        F.Flat->IdxEnd = Cursor;
        F.Flat->NumDescendants = NextID - F.ID - 1;
        Stack.pop_back();
      }
    }
  }

  // Place every block into its innermost cycle's own region, remapping its
  // BlockMap entry from a creation-order index to the flat preorder index.
  Info.BlockLayout.resize_for_overwrite(Cursor);
  for (BlockT *B : llvm::reverse(Order)) {
    unsigned Number = GraphTraits<const BlockT *>::getNumber(B);
    unsigned Created = Info.BlockMap[Number];
    if (Created != CycleInfoT::NoCycle) {
      // Created indexes AllCycles; enter() stashed the flat preorder index in
      // that temporary node's IdxBegin.
      unsigned Flat = AllCycles[Created].IdxBegin;
      Info.BlockMap[Number] = Flat;
      CycleT &FlatCycle = Info.Cycles[Flat];
      Info.BlockLayout[--FlatCycle.IdxBegin] = B;
    }
  }
}

/// \brief Main function of the cycle info computations.
template <typename ContextT>
void GenericCycleInfoCompute<ContextT>::run(FunctionT *F) {
  BlockT *EntryBlock = GraphTraits<FunctionT *>::getEntryNode(F);
  LLVM_DEBUG(errs() << "Entry block: " << Info.Context.print(EntryBlock)
                    << "\n");
  dfs(F, EntryBlock);

  SmallVector<BlockT *, 8> Worklist;

  for (BlockT *HeaderCandidate : llvm::reverse(BlockPreorder)) {
    const DFSInfo CandidateInfo = getDFSInfo(HeaderCandidate);

    for (BlockT *Pred : predecessors(HeaderCandidate)) {
      const DFSInfo PredDFSInfo = getDFSInfo(Pred);
      // This automatically ignores unreachable predecessors since they have
      // zeros in their DFSInfo.
      if (CandidateInfo.isAncestorOf(PredDFSInfo))
        Worklist.push_back(Pred);
    }
    if (Worklist.empty()) {
      continue;
    }

    // Found a cycle with the candidate as its header.
    LLVM_DEBUG(errs() << "Found cycle for header: "
                      << Info.Context.print(HeaderCandidate) << "\n");
    CycleT *NewCycle = &AllCycles.emplace_back();
    NewCycle->IdxBegin = AttachedChildren.size(); // Attach-log slice start.
    NewCycle->appendEntry(HeaderCandidate);
    recordInnermostCycle(HeaderCandidate);
    // The header is this cycle's first own block. Until flatten() runs,
    // IdxEnd accumulates this cycle's own-block count (see the IdxBegin/
    // IdxEnd doc comment), so flatten() needs no separate counting pass.
    ++NewCycle->IdxEnd;

    // Helper function to process (non-back-edge) predecessors of a discovered
    // block and either add them to the worklist or recognize that the given
    // block is an additional cycle entry.
    auto ProcessPredecessors = [&](BlockT *Block) {
      LLVM_DEBUG(errs() << "  block " << Info.Context.print(Block) << ": ");

      bool IsEntry = false;
      for (BlockT *Pred : predecessors(Block)) {
        const DFSInfo PredDFSInfo = getDFSInfo(Pred);
        if (CandidateInfo.isAncestorOf(PredDFSInfo)) {
          Worklist.push_back(Pred);
        } else if (!PredDFSInfo) {
          // Ignore an unreachable predecessor. It will will incorrectly cause
          // Block to be treated as a cycle entry.
          LLVM_DEBUG(errs() << " skipped unreachable predecessor.\n");
        } else {
          IsEntry = true;
        }
      }
      if (IsEntry) {
        assert(!is_contained(NewCycle->Entries, Block));
        LLVM_DEBUG(errs() << "append as entry\n");
        NewCycle->appendEntry(Block);
      } else {
        LLVM_DEBUG(errs() << "append as child\n");
      }
    };

    do {
      BlockT *Block = Worklist.pop_back_val();
      if (Block == HeaderCandidate)
        continue;

      // If the block has already been discovered by some cycle
      // (possibly by ourself), then the outermost cycle containing it
      // should become our child. Walk the temporary forest directly:
      // handles are not meaningful until flatten() builds the flat array, so
      // BlockMap still holds creation-order indices into AllCycles.
      unsigned Created = Info.BlockMap[GraphTraits<BlockT *>::getNumber(Block)];
      CycleT *BlockParent =
          Created == CycleInfoT::NoCycle ? nullptr : &AllCycles[Created];
      while (BlockParent && BlockParent->hasParent())
        BlockParent = &AllCycles[BlockParent->ParentIndex];
      if (BlockParent) {
        LLVM_DEBUG(errs() << "  block " << Info.Context.print(Block) << ": ");

        if (BlockParent != NewCycle) {
          LLVM_DEBUG(errs()
                     << "discovered child cycle "
                     << Info.Context.print(BlockParent->Entries[0]) << "\n");
          // Make BlockParent the child of NewCycle.
          moveTopLevelCycleToNewParent(NewCycle, BlockParent);

          for (auto *ChildEntry : BlockParent->Entries)
            ProcessPredecessors(ChildEntry);
        } else {
          LLVM_DEBUG(errs()
                     << "known child cycle "
                     << Info.Context.print(BlockParent->Entries[0]) << "\n");
        }
      } else {
        recordInnermostCycle(Block);
        ++NewCycle->IdxEnd; // Block's innermost cycle is NewCycle.
        ProcessPredecessors(Block);
      }
    } while (!Worklist.empty());

    NewCycle->Depth = AttachedChildren.size(); // Attach-log slice end.
    TopLevelCycles.push_back(NewCycle);
  }

  // The cycle forest and the block-to-innermost-cycle map are complete; move
  // the forest into Info's flat preorder array and lay out every cycle's
  // blocks into the shared contiguous BlockLayout.
  flatten(BlockPreorder);
}

/// \brief Compute a DFS of basic blocks starting at the function entry.
///
/// Fills BlockDFSInfo with start/end counters and BlockPreorder.
template <typename ContextT>
void GenericCycleInfoCompute<ContextT>::dfs(FunctionT *F, BlockT *EntryBlock) {
  BlockDFSInfo.resize(GraphTraits<FunctionT *>::getMaxNumber(F));

  // Successors are visited in reverse order to match the legacy
  // single-LIFO-stack traversal, keeping cycle identification and block order
  // unchanged.
  using SuccIt = decltype(successors(EntryBlock).begin());
  struct Frame {
    BlockT *Block;
    std::reverse_iterator<SuccIt> Cur, End;
  };
  SmallVector<Frame, 8> Stack;
  unsigned Counter = 0;

  auto open = [&](BlockT *Block) {
    getOrInsertDFSInfo(Block).Start = ++Counter;
    BlockPreorder.push_back(Block);
    LLVM_DEBUG(errs() << "DFS visiting block: " << Info.Context.print(Block)
                      << ", preorder number: " << Counter << "\n");
    auto Succs = successors(Block);
    Stack.push_back({Block, std::make_reverse_iterator(Succs.end()),
                     std::make_reverse_iterator(Succs.begin())});
  };

  open(EntryBlock);
  while (!Stack.empty()) {
    Frame &Top = Stack.back();
    BlockT *Next = nullptr;
    while (Top.Cur != Top.End) {
      BlockT *Succ = *Top.Cur++;
      if (getOrInsertDFSInfo(Succ).Start == 0) {
        Next = Succ;
        break;
      }
      LLVM_DEBUG(errs() << "  already visited successor: "
                        << Info.Context.print(Succ) << "\n");
    }
    if (Next) {
      open(Next);
    } else {
      // Top's subtree is complete. Its end counter is the largest preorder
      // number in the subtree.
      getOrInsertDFSInfo(Top.Block).End = Counter;
      LLVM_DEBUG(errs() << "DFS block " << Info.Context.print(Top.Block)
                        << " ended at " << Counter << "\n");
      Stack.pop_back();
    }
  }

  LLVM_DEBUG({
    errs() << "Preorder:\n";
    for (int I = 0, E = BlockPreorder.size(); I != E; ++I)
      errs() << "  " << Info.Context.print(BlockPreorder[I]) << ": " << I
             << "\n";
  });
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
  BlockMap.assign(GraphTraits<FunctionT *>::getMaxNumber(&F), NoCycle);

  LLVM_DEBUG(errs() << "Computing cycles for function: " << F.getName()
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
