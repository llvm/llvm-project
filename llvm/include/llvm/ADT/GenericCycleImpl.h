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

#define DEBUG_TYPE "generic-cycle-impl"

namespace llvm {

template <typename ContextT>
bool GenericCycle<ContextT>::contains(const GenericCycle *C) const {
  if (!C)
    return false;

  if (Depth > C->Depth)
    return false;
  while (Depth < C->Depth)
    C = C->ParentCycle;
  return this == C;
}

template <typename ContextT>
void GenericCycle<ContextT>::getExitBlocks(
    SmallVectorImpl<BlockT *> &TmpStorage) const {
  TmpStorage.clear();

  size_t NumExitBlocks = 0;
  for (BlockT *Block : blocks()) {
    llvm::append_range(TmpStorage, successors(Block));

    for (size_t Idx = NumExitBlocks, End = TmpStorage.size(); Idx < End;
         ++Idx) {
      BlockT *Succ = TmpStorage[Idx];
      if (!contains(Succ)) {
        auto ExitEndIt = TmpStorage.begin() + NumExitBlocks;
        if (std::find(TmpStorage.begin(), ExitEndIt, Succ) == ExitEndIt)
          TmpStorage[NumExitBlocks++] = Succ;
      }
    }

    TmpStorage.resize(NumExitBlocks);
  }
}

template <typename ContextT>
void GenericCycle<ContextT>::getExitingBlocks(
    SmallVectorImpl<BlockT *> &TmpStorage) const {
  TmpStorage.clear();

  for (BlockT *Block : blocks()) {
    for (BlockT *Succ : successors(Block)) {
      if (!contains(Succ)) {
        TmpStorage.push_back(Block);
        break;
      }
    }
  }
}

template <typename ContextT>
auto GenericCycle<ContextT>::getCyclePreheader() const -> BlockT * {
  BlockT *Predecessor = getCyclePredecessor();
  if (!Predecessor)
    return nullptr;

  assert(isReducible() && "Cycle Predecessor must be in a reducible cycle!");

  if (succ_size(Predecessor) != 1)
    return nullptr;

  // Make sure we are allowed to hoist instructions into the predecessor.
  if (!Predecessor->isLegalToHoistInto())
    return nullptr;

  return Predecessor;
}

template <typename ContextT>
auto GenericCycle<ContextT>::getCyclePredecessor() const -> BlockT * {
  if (!isReducible())
    return nullptr;

  BlockT *Out = nullptr;

  // Loop over the predecessors of the header node...
  BlockT *Header = getHeader();
  for (const auto Pred : predecessors(Header)) {
    if (!contains(Pred)) {
      if (Out && Out != Pred)
        return nullptr;
      Out = Pred;
    }
  }

  return Out;
}

/// \brief Helper class for computing cycle information.
template <typename ContextT> class GenericCycleInfoCompute {
  using BlockT = typename ContextT::BlockT;
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

  DenseMap<BlockT *, DFSInfo> BlockDFSInfo;
  SmallVector<BlockT *, 8> BlockPreorder;

  GenericCycleInfoCompute(const GenericCycleInfoCompute &) = delete;
  GenericCycleInfoCompute &operator=(const GenericCycleInfoCompute &) = delete;

public:
  GenericCycleInfoCompute(CycleInfoT &Info) : Info(Info) {}

  void run(BlockT *EntryBlock);

  static void updateDepth(CycleT *SubTree);

private:
  void dfs(BlockT *EntryBlock);
};

template <typename ContextT>
auto GenericCycleInfo<ContextT>::getTopLevelParentCycle(BlockT *Block)
    -> CycleT * {
  auto Cycle = BlockMapTopLevel.find(Block);
  if (Cycle != BlockMapTopLevel.end())
    return Cycle->second;

  auto MapIt = BlockMap.find(Block);
  if (MapIt == BlockMap.end())
    return nullptr;

  auto *C = MapIt->second;
  while (C->ParentCycle)
    C = C->ParentCycle;
  BlockMapTopLevel.try_emplace(Block, C);
  return C;
}

template <typename ContextT>
void GenericCycleInfo<ContextT>::moveToAdjacentCycle(CycleT *NewParent,
                                                     CycleT *Child) {
  auto *OldParent = Child->getParentCycle();
  assert(!OldParent || OldParent->contains(NewParent));

  // Find the child in its current parent (or toplevel) and move it out of its
  // container, into the new parent.
  auto &CurrentContainer = OldParent ? OldParent->Children : TopLevelCycles;
  auto Pos = llvm::find_if(CurrentContainer, [=](const auto &Ptr) -> bool {
    return Child == Ptr.get();
  });
  assert(Pos != CurrentContainer.end());
  NewParent->Children.push_back(std::move(*Pos));
  // Pos is empty after moving the child out. So we move the last child into its
  // place rather than refilling the whole container.
  *Pos = std::move(CurrentContainer.back());
  CurrentContainer.pop_back();

  Child->ParentCycle = NewParent;

  // Add child blocks to the hierarchy up to the old parent.
  auto *ParentIter = NewParent;
  while (ParentIter != OldParent) {
    ParentIter->Blocks.insert(Child->block_begin(), Child->block_end());
    ParentIter = ParentIter->getParentCycle();
  }

  // If Child was a top-level cycle, update the map.
  if (!OldParent) {
    auto *H = NewParent->getHeader();
    auto *NewTLC = getTopLevelParentCycle(H);
    for (auto &It : BlockMapTopLevel)
      if (It.second == Child)
        It.second = NewTLC;
  }
}

template <typename ContextT>
void GenericCycleInfo<ContextT>::addBlockToCycle(BlockT *Block, CycleT *Cycle) {
  // FixMe: Appending NewBlock is fine as a set of blocks in a cycle. When
  // printing, cycle NewBlock is at the end of list but it should be in the
  // middle to represent actual traversal of a cycle.
  Cycle->appendBlock(Block);
  BlockMap.try_emplace(Block, Cycle);

  CycleT *ParentCycle = Cycle->getParentCycle();
  while (ParentCycle) {
    Cycle = ParentCycle;
    Cycle->appendBlock(Block);
    ParentCycle = Cycle->getParentCycle();
  }

  BlockMapTopLevel.try_emplace(Block, Cycle);
}

/// \brief Main function of the cycle info computations.
template <typename ContextT>
void GenericCycleInfoCompute<ContextT>::run(BlockT *EntryBlock) {
  LLVM_DEBUG(errs() << "Entry block: " << Info.Context.print(EntryBlock)
                    << "\n");
  dfs(EntryBlock);

  SmallVector<BlockT *, 8> Worklist;

  for (BlockT *HeaderCandidate : llvm::reverse(BlockPreorder)) {
    const DFSInfo CandidateInfo = BlockDFSInfo.lookup(HeaderCandidate);

    for (BlockT *Pred : predecessors(HeaderCandidate)) {
      const DFSInfo PredDFSInfo = BlockDFSInfo.lookup(Pred);
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
    std::unique_ptr<CycleT> NewCycle = std::make_unique<CycleT>();
    NewCycle->appendEntry(HeaderCandidate);
    NewCycle->appendBlock(HeaderCandidate);
    Info.BlockMap.try_emplace(HeaderCandidate, NewCycle.get());

    // Helper function to process (non-back-edge) predecessors of a discovered
    // block and either add them to the worklist or recognize that the given
    // block is an additional cycle entry.
    auto ProcessPredecessors = [&](BlockT *Block) {
      LLVM_DEBUG(errs() << "  block " << Info.Context.print(Block) << ": ");

      bool IsEntry = false;
      for (BlockT *Pred : predecessors(Block)) {
        const DFSInfo PredDFSInfo = BlockDFSInfo.lookup(Pred);
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
        assert(!NewCycle->isEntry(Block));
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
      // should become our child.
      if (auto *BlockParent = Info.getTopLevelParentCycle(Block)) {
        LLVM_DEBUG(errs() << "  block " << Info.Context.print(Block) << ": ");

        if (BlockParent != NewCycle.get()) {
          LLVM_DEBUG(errs()
                     << "discovered child cycle "
                     << Info.Context.print(BlockParent->getHeader()) << "\n");
          // Make BlockParent the child of NewCycle.
          Info.moveToAdjacentCycle(NewCycle.get(), BlockParent);

          for (auto *ChildEntry : BlockParent->entries())
            ProcessPredecessors(ChildEntry);
        } else {
          LLVM_DEBUG(errs()
                     << "known child cycle "
                     << Info.Context.print(BlockParent->getHeader()) << "\n");
        }
      } else {
        Info.BlockMap.try_emplace(Block, NewCycle.get());
        assert(!is_contained(NewCycle->Blocks, Block));
        NewCycle->Blocks.insert(Block);
        ProcessPredecessors(Block);
        Info.BlockMapTopLevel.try_emplace(Block, NewCycle.get());
      }
    } while (!Worklist.empty());

    Info.TopLevelCycles.push_back(std::move(NewCycle));
  }

  // Fix top-level cycle links and compute cycle depths.
  for (auto *TLC : Info.toplevel_cycles()) {
    LLVM_DEBUG(errs() << "top-level cycle: "
                      << Info.Context.print(TLC->getHeader()) << "\n");

    TLC->ParentCycle = nullptr;
    updateDepth(TLC);
  }
}

/// \brief Recompute depth values of \p SubTree and all descendants.
template <typename ContextT>
void GenericCycleInfoCompute<ContextT>::updateDepth(CycleT *SubTree) {
  for (CycleT *Cycle : depth_first(SubTree))
    Cycle->Depth = Cycle->ParentCycle ? Cycle->ParentCycle->Depth + 1 : 1;
}

/// \brief Compute a DFS of basic blocks starting at the function entry.
///
/// Fills BlockDFSInfo with start/end counters and BlockPreorder.
template <typename ContextT>
void GenericCycleInfoCompute<ContextT>::dfs(BlockT *EntryBlock) {
  SmallVector<unsigned, 8> DFSTreeStack;
  SmallVector<BlockT *, 8> TraverseStack;
  unsigned Counter = 0;
  TraverseStack.emplace_back(EntryBlock);

  do {
    BlockT *Block = TraverseStack.back();
    LLVM_DEBUG(errs() << "DFS visiting block: " << Info.Context.print(Block)
                      << "\n");
    if (!BlockDFSInfo.count(Block)) {
      // We're visiting the block for the first time. Open its DFSInfo, add
      // successors to the traversal stack, and remember the traversal stack
      // depth at which the block was opened, so that we can correctly record
      // its end time.
      LLVM_DEBUG(errs() << "  first encountered at depth "
                        << TraverseStack.size() << "\n");

      DFSTreeStack.emplace_back(TraverseStack.size());
      llvm::append_range(TraverseStack, successors(Block));

      bool Added = BlockDFSInfo.try_emplace(Block, ++Counter).second;
      (void)Added;
      assert(Added);
      BlockPreorder.push_back(Block);
      LLVM_DEBUG(errs() << "  preorder number: " << Counter << "\n");
    } else {
      assert(!DFSTreeStack.empty());
      if (DFSTreeStack.back() == TraverseStack.size()) {
        LLVM_DEBUG(errs() << "  ended at " << Counter << "\n");
        BlockDFSInfo.find(Block)->second.End = Counter;
        DFSTreeStack.pop_back();
      } else {
        LLVM_DEBUG(errs() << "  already done\n");
      }
      TraverseStack.pop_back();
    }
  } while (!TraverseStack.empty());
  assert(DFSTreeStack.empty());

  LLVM_DEBUG(
    errs() << "Preorder:\n";
    for (int i = 0, e = BlockPreorder.size(); i != e; ++i) {
      errs() << "  " << Info.Context.print(BlockPreorder[i]) << ": " << i << "\n";
    }
  );
}

/// \brief Reset the object to its initial state.
template <typename ContextT> void GenericCycleInfo<ContextT>::clear() {
  TopLevelCycles.clear();
  BlockMap.clear();
  BlockMapTopLevel.clear();
}

/// \brief Compute the cycle info for a function.
template <typename ContextT>
void GenericCycleInfo<ContextT>::compute(FunctionT &F) {
  GenericCycleInfoCompute<ContextT> Compute(*this);
  Context = ContextT(&F);

  LLVM_DEBUG(errs() << "Computing cycles for function: " << F.getName()
                    << "\n");
  Compute.run(&F.front());

  assert(validateTree());
}

template <typename ContextT>
void GenericCycleInfo<ContextT>::splitCriticalEdge(BlockT *Pred, BlockT *Succ,
                                                   BlockT *NewBlock) {
  // Edge Pred-Succ is replaced by edges Pred-NewBlock and NewBlock-Succ, all
  // cycles that had blocks Pred and Succ also get NewBlock.
  CycleT *Cycle = getSmallestCommonCycle(getCycle(Pred), getCycle(Succ));
  if (!Cycle)
    return;

  addBlockToCycle(NewBlock, Cycle);
  assert(validateTree());
}

/// \brief Extend a cycle minimally such that it contains every path from that
///        cycle reaching a a given block.
///
/// The cycle structure is updated such that all predecessors of \p toBlock will
/// be contained (possibly indirectly) in \p cycleToExtend, without removing any
/// cycles.
///
/// If \p transferredBlocks is non-null, all blocks whose direct containing
/// cycle was changed are appended to the vector.
template <typename ContextT>
void GenericCycleInfo<ContextT>::extendCycle(
    CycleT *cycleToExtend, BlockT *toBlock,
    SmallVectorImpl<BlockT *> *transferredBlocks) {
  SmallVector<BlockT *> workList;
  workList.push_back(toBlock);

  assert(cycleToExtend);
  while (!workList.empty()) {
    BlockT *block = workList.pop_back_val();
    CycleT *cycle = getCycle(block);
    if (cycleToExtend->contains(cycle))
      continue;

    auto cycleToInclude = findLargestDisjointAncestor(cycle, cycleToExtend);
    if (cycleToInclude) {
      // Move cycle into cycleToExtend.
      moveToAdjacentCycle(cycleToExtend, cycleToInclude);
      assert(cycleToInclude->Depth <= cycleToExtend->Depth);
      GenericCycleInfoCompute<ContextT>::updateDepth(cycleToInclude);

      // Continue from the entries of the newly included cycle.
      for (BlockT *entry : cycleToInclude->Entries)
        llvm::append_range(workList, predecessors(entry));
    } else {
      // Block is contained in an ancestor of cycleToExtend, just add it
      // to the cycle and proceed.
      BlockMap[block] = cycleToExtend;
      if (transferredBlocks)
        transferredBlocks->push_back(block);

      CycleT *ancestor = cycleToExtend;
      do {
        ancestor->Blocks.insert(block);
        ancestor = ancestor->getParentCycle();
      } while (ancestor != cycle);

      llvm::append_range(workList, predecessors(block));
    }
  }

  assert(validateTree());
}

/// \brief Finds the largest ancestor of \p A that is disjoint from \B.
///
/// The caller must ensure that \p B does not contain \p A. If \p A
/// contains \p B, null is returned.
template <typename ContextT>
auto GenericCycleInfo<ContextT>::findLargestDisjointAncestor(
    const CycleT *A, const CycleT *B) const -> CycleT * {
  if (!A || !B)
    return nullptr;

  while (B && A->Depth < B->Depth)
    B = B->ParentCycle;
  while (A && A->Depth > B->Depth)
    A = A->ParentCycle;

  if (A == B)
    return nullptr;

  assert(A && B);
  assert(A->Depth == B->Depth);

  for (;;) {
    // Since both are at the same depth, the only way for both A and B to be
    // null is when their parents are null, which will terminate the loop.
    assert(A && B);

    if (A->ParentCycle == B->ParentCycle) {
      // const_cast is justified since cycles are owned by this
      // object, which is non-const.
      return const_cast<CycleT *>(A);
    }
    A = A->ParentCycle;
    B = B->ParentCycle;
  }
}

/// \brief Find the innermost cycle containing a given block.
///
/// \returns the innermost cycle containing \p Block or nullptr if
///          it is not contained in any cycle.
template <typename ContextT>
auto GenericCycleInfo<ContextT>::getCycle(const BlockT *Block) const
    -> CycleT * {
  return BlockMap.lookup(Block);
}

/// \brief Find the innermost cycle containing both given cycles.
///
/// \returns the innermost cycle containing both \p A and \p B
///          or nullptr if there is no such cycle.
template <typename ContextT>
auto GenericCycleInfo<ContextT>::getSmallestCommonCycle(CycleT *A,
                                                        CycleT *B) const
    -> CycleT * {
  if (!A || !B)
    return nullptr;

  // If cycles A and B have different depth replace them with parent cycle
  // until they have the same depth.
  while (A->getDepth() > B->getDepth())
    A = A->getParentCycle();
  while (B->getDepth() > A->getDepth())
    B = B->getParentCycle();

  // Cycles A and B are at same depth but may be disjoint, replace them with
  // parent cycles until we find cycle that contains both or we run out of
  // parent cycles.
  while (A != B) {
    A = A->getParentCycle();
    B = B->getParentCycle();
  }

  return A;
}

/// \brief get the depth for the cycle which containing a given block.
///
/// \returns the depth for the innermost cycle containing \p Block or 0 if it is
///          not contained in any cycle.
template <typename ContextT>
unsigned GenericCycleInfo<ContextT>::getCycleDepth(const BlockT *Block) const {
  CycleT *Cycle = getCycle(Block);
  if (!Cycle)
    return 0;
  return Cycle->getDepth();
}

#ifndef NDEBUG
/// \brief Validate the internal consistency of the cycle tree.
///
/// Note that this does \em not check that cycles are really cycles in the CFG,
/// or that the right set of cycles in the CFG were found.
template <typename ContextT>
bool GenericCycleInfo<ContextT>::validateTree() const {
  DenseSet<BlockT *> Blocks;
  DenseSet<BlockT *> Entries;

  auto reportError = [](const char *File, int Line, const char *Cond) {
    errs() << File << ':' << Line
           << ": GenericCycleInfo::validateTree: " << Cond << '\n';
  };
#define check(cond)                                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      reportError(__FILE__, __LINE__, #cond);                                  \
      return false;                                                            \
    }                                                                          \
  } while (false)

  for (const auto *TLC : toplevel_cycles()) {
    for (const CycleT *Cycle : depth_first(TLC)) {
      if (Cycle->ParentCycle)
        check(is_contained(Cycle->ParentCycle->children(), Cycle));

      for (BlockT *Block : Cycle->Blocks) {
        auto MapIt = BlockMap.find(Block);
        check(MapIt != BlockMap.end());
        check(Cycle->contains(MapIt->second));
        check(Blocks.insert(Block).second); // duplicates in block list?
      }
      Blocks.clear();

      check(!Cycle->Entries.empty());
      for (BlockT *Entry : Cycle->Entries) {
        check(Entries.insert(Entry).second); // duplicate entry?
        check(is_contained(Cycle->Blocks, Entry));
      }
      Entries.clear();

      unsigned ChildDepth = 0;
      for (const CycleT *Child : Cycle->children()) {
        check(Child->Depth > Cycle->Depth);
        if (!ChildDepth) {
          ChildDepth = Child->Depth;
        } else {
          check(ChildDepth == Child->Depth);
        }
      }
    }
  }

  for (const auto &Entry : BlockMap) {
    BlockT *Block = Entry.first;
    for (const CycleT *Cycle = Entry.second; Cycle;
         Cycle = Cycle->ParentCycle) {
      check(is_contained(Cycle->Blocks, Block));
    }
  }

#undef check

  return true;
}
#endif

/// \brief Print the cycle info.
template <typename ContextT>
void GenericCycleInfo<ContextT>::print(raw_ostream &Out) const {
  for (const auto *TLC : toplevel_cycles()) {
    for (const CycleT *Cycle : depth_first(TLC)) {
      for (unsigned I = 0; I < Cycle->Depth; ++I)
        Out << "    ";

      Out << Cycle->print(Context) << '\n';
    }
  }
}

} // namespace llvm

#undef DEBUG_TYPE

#endif // LLVM_ADT_GENERICCYCLEIMPL_H
