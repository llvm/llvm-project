//===- GenericDomTreeUpdaterImpl.h ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the GenericDomTreeUpdater class. This file should only
// be included by files that implement a specialization of the relevant
// templates. Currently these are:
// - llvm/lib/Analysis/DomTreeUpdater.cpp
// - llvm/lib/CodeGen/MachineDomTreeUpdater.cpp
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_ANALYSIS_GENERICDOMTREEUPDATERIMPL_H
#define LLVM_ANALYSIS_GENERICDOMTREEUPDATERIMPL_H

#include "llvm/ADT/SmallBitVector.h"
#include "llvm/Analysis/GenericDomTreeUpdater.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {

template <typename DerivedT, typename DomTreeT, typename PostDomTreeT>
template <typename FuncT>
void GenericDomTreeUpdater<DerivedT, DomTreeT, PostDomTreeT>::recalculate(
    FuncT &F) {
  if (Strategy == UpdateStrategy::Eager) {
    if (DT)
      DT->recalculate(F);
    if (PDT)
      PDT->recalculate(F);
    return;
  }

  // There is little performance gain if we pend the recalculation under
  // Lazy UpdateStrategy so we recalculate available trees immediately.

  // Prevent forceFlushDeletedBB() from erasing DomTree or PostDomTree nodes.
  IsRecalculatingDomTree = IsRecalculatingPostDomTree = true;

  // Because all trees are going to be up-to-date after recalculation,
  // flush awaiting deleted BasicBlocks.
  derived().forceFlushDeletedBB();
  if (DT)
    DT->recalculate(F);
  if (PDT)
    PDT->recalculate(F);

  // Resume forceFlushDeletedBB() to erase DomTree or PostDomTree nodes.
  IsRecalculatingDomTree = IsRecalculatingPostDomTree = false;
  PendDTUpdateIndex = PendPDTUpdateIndex = PendUpdates.size();
  dropOutOfDateUpdates();
}

template <typename DerivedT, typename DomTreeT, typename PostDomTreeT>
void GenericDomTreeUpdater<DerivedT, DomTreeT, PostDomTreeT>::applyUpdates(
    ArrayRef<UpdateT> Updates) {
  if (!DT && !PDT)
    return;

  if (Strategy == UpdateStrategy::Lazy) {
    PendUpdates.reserve(PendUpdates.size() + Updates.size());
    for (const auto &U : Updates)
      if (!isSelfDominance(U))
        PendUpdates.push_back(U);

    return;
  }

  if (DT)
    DT->applyUpdates(Updates);
  if (PDT)
    PDT->applyUpdates(Updates);
}

template <typename DerivedT, typename DomTreeT, typename PostDomTreeT>
void GenericDomTreeUpdater<DerivedT, DomTreeT, PostDomTreeT>::
    applyUpdatesPermissive(ArrayRef<UpdateT> Updates) {
  if (!DT && !PDT)
    return;

  SmallSet<std::pair<BasicBlockT *, BasicBlockT *>, 8> Seen;
  SmallVector<UpdateT, 8> DeduplicatedUpdates;
  for (const auto &U : Updates) {
    auto Edge = std::make_pair(U.getFrom(), U.getTo());
    // Because it is illegal to submit updates that have already been applied
    // and updates to an edge need to be strictly ordered,
    // it is safe to infer the existence of an edge from the first update
    // to this edge.
    // If the first update to an edge is "Delete", it means that the edge
    // existed before. If the first update to an edge is "Insert", it means
    // that the edge didn't exist before.
    //
    // For example, if the user submits {{Delete, A, B}, {Insert, A, B}},
    // because
    // 1. it is illegal to submit updates that have already been applied,
    // i.e., user cannot delete an nonexistent edge,
    // 2. updates to an edge need to be strictly ordered,
    // So, initially edge A -> B existed.
    // We can then safely ignore future updates to this edge and directly
    // inspect the current CFG:
    // a. If the edge still exists, because the user cannot insert an existent
    // edge, so both {Delete, A, B}, {Insert, A, B} actually happened and
    // resulted in a no-op. DTU won't submit any update in this case.
    // b. If the edge doesn't exist, we can then infer that {Delete, A, B}
    // actually happened but {Insert, A, B} was an invalid update which never
    // happened. DTU will submit {Delete, A, B} in this case.
    if (!isSelfDominance(U) && Seen.insert(Edge).second) {
      // If the update doesn't appear in the CFG, it means that
      // either the change isn't made or relevant operations
      // result in a no-op.
      if (isUpdateValid(U)) {
        if (isLazy())
          PendUpdates.push_back(U);
        else
          DeduplicatedUpdates.push_back(U);
      }
    }
  }

  if (Strategy == UpdateStrategy::Lazy)
    return;

  if (DT)
    DT->applyUpdates(DeduplicatedUpdates);
  if (PDT)
    PDT->applyUpdates(DeduplicatedUpdates);
}

template <typename DerivedT, typename DomTreeT, typename PostDomTreeT>
void GenericDomTreeUpdater<DerivedT, DomTreeT, PostDomTreeT>::splitCriticalEdge(
    BasicBlockT *FromBB, BasicBlockT *ToBB, BasicBlockT *NewBB) {
  if (!DT && !PDT)
    return;

  CriticalEdge E = {FromBB, ToBB, NewBB};
  if (Strategy == UpdateStrategy::Lazy) {
    PendUpdates.push_back(E);
    return;
  }

  if (DT)
    splitDTCriticalEdges(E);
  if (PDT)
    splitPDTCriticalEdges(E);
}

template <typename DerivedT, typename DomTreeT, typename PostDomTreeT>
DomTreeT &
GenericDomTreeUpdater<DerivedT, DomTreeT, PostDomTreeT>::getDomTree() {
  assert(DT && "Invalid acquisition of a null DomTree");
  applyDomTreeUpdates();
  dropOutOfDateUpdates();
  return *DT;
}

template <typename DerivedT, typename DomTreeT, typename PostDomTreeT>
PostDomTreeT &
GenericDomTreeUpdater<DerivedT, DomTreeT, PostDomTreeT>::getPostDomTree() {
  assert(PDT && "Invalid acquisition of a null PostDomTree");
  applyPostDomTreeUpdates();
  dropOutOfDateUpdates();
  return *PDT;
}

template <typename DerivedT, typename DomTreeT, typename PostDomTreeT>
LLVM_DUMP_METHOD void
GenericDomTreeUpdater<DerivedT, DomTreeT, PostDomTreeT>::dump() const {
#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  raw_ostream &OS = llvm::dbgs();

  OS << "Available Trees: ";
  if (DT || PDT) {
    if (DT)
      OS << "DomTree ";
    if (PDT)
      OS << "PostDomTree ";
    OS << "\n";
  } else
    OS << "None\n";

  OS << "UpdateStrategy: ";
  if (Strategy == UpdateStrategy::Eager) {
    OS << "Eager\n";
    return;
  } else
    OS << "Lazy\n";
  int Index = 0;

  auto printBlockInfo = [&](BasicBlockT *BB, StringRef Ending) {
    if (BB) {
      auto S = BB->getName();
      if (!BB->hasName())
        S = "(no name)";
      OS << S << "(" << BB << ")" << Ending;
    } else {
      OS << "(badref)" << Ending;
    }
  };

  auto printUpdates =
      [&](typename ArrayRef<DomTreeUpdate>::const_iterator begin,
          typename ArrayRef<DomTreeUpdate>::const_iterator end) {
        if (begin == end)
          OS << "  None\n";
        Index = 0;
        for (auto It = begin, ItEnd = end; It != ItEnd; ++It) {
          if (!It->IsCriticalEdgeSplit) {
            auto U = It->Update;
            OS << "  " << Index << " : ";
            ++Index;
            if (U.getKind() == DomTreeT::Insert)
              OS << "Insert, ";
            else
              OS << "Delete, ";
            printBlockInfo(U.getFrom(), ", ");
            printBlockInfo(U.getTo(), "\n");
          } else {
            const auto &Edge = It->EdgeSplit;
            OS << "  " << Index++ << " : Split critical edge, ";
            printBlockInfo(Edge.FromBB, ", ");
            printBlockInfo(Edge.ToBB, ", ");
            printBlockInfo(Edge.NewBB, "\n");
          }
        }
      };

  if (DT) {
    const auto I = PendUpdates.begin() + PendDTUpdateIndex;
    assert(PendUpdates.begin() <= I && I <= PendUpdates.end() &&
           "Iterator out of range.");
    OS << "Applied but not cleared DomTreeUpdates:\n";
    printUpdates(PendUpdates.begin(), I);
    OS << "Pending DomTreeUpdates:\n";
    printUpdates(I, PendUpdates.end());
  }

  if (PDT) {
    const auto I = PendUpdates.begin() + PendPDTUpdateIndex;
    assert(PendUpdates.begin() <= I && I <= PendUpdates.end() &&
           "Iterator out of range.");
    OS << "Applied but not cleared PostDomTreeUpdates:\n";
    printUpdates(PendUpdates.begin(), I);
    OS << "Pending PostDomTreeUpdates:\n";
    printUpdates(I, PendUpdates.end());
  }

  OS << "Pending DeletedBBs:\n";
  Index = 0;
  for (const auto *BB : DeletedBBs) {
    OS << "  " << Index << " : ";
    ++Index;
    if (BB->hasName())
      OS << BB->getName() << "(";
    else
      OS << "(no name)(";
    OS << BB << ")\n";
  }
#endif
}

template <typename DerivedT, typename DomTreeT, typename PostDomTreeT>
template <bool IsForward>
void GenericDomTreeUpdater<DerivedT, DomTreeT,
                           PostDomTreeT>::applyUpdatesImpl() {
  auto *DomTree = [&]() {
    if constexpr (IsForward)
      return DT;
    else
      return PDT;
  }();
  // No pending DomTreeUpdates.
  if (Strategy != UpdateStrategy::Lazy || !DomTree)
    return;
  size_t &PendUpdateIndex = IsForward ? PendDTUpdateIndex : PendPDTUpdateIndex;

  // Only apply updates not are applied by (Post)DomTree.
  while (IsForward ? hasPendingDomTreeUpdates()
                   : hasPendingPostDomTreeUpdates()) {
    auto I = PendUpdates.begin() + PendUpdateIndex;
    const auto E = PendUpdates.end();
    assert(I < E && "Iterator range invalid; there should be DomTree updates.");
    if (!I->IsCriticalEdgeSplit) {
      SmallVector<UpdateT, 32> NormalUpdates;
      for (; I != E && !I->IsCriticalEdgeSplit; ++I)
        NormalUpdates.push_back(I->Update);
      DomTree->applyUpdates(NormalUpdates);
      PendUpdateIndex += NormalUpdates.size();
    } else {
      SmallVector<CriticalEdge> CriticalEdges;
      for (; I != E && I->IsCriticalEdgeSplit; ++I)
        CriticalEdges.push_back(I->EdgeSplit);
      IsForward ? splitDTCriticalEdges(CriticalEdges)
                : splitPDTCriticalEdges(CriticalEdges);
      PendUpdateIndex += CriticalEdges.size();
    }
  }
}

template <typename DerivedT, typename DomTreeT, typename PostDomTreeT>
bool GenericDomTreeUpdater<DerivedT, DomTreeT, PostDomTreeT>::isUpdateValid(
    UpdateT Update) const {
  const auto *From = Update.getFrom();
  const auto *To = Update.getTo();
  const auto Kind = Update.getKind();

  // Discard updates by inspecting the current state of successors of From.
  // Since isUpdateValid() must be called *after* the Terminator of From is
  // altered we can determine if the update is unnecessary for batch updates
  // or invalid for a single update.
  const bool HasEdge = llvm::is_contained(successors(From), To);

  // If the IR does not match the update,
  // 1. In batch updates, this update is unnecessary.
  // 2. When called by insertEdge*()/deleteEdge*(), this update is invalid.
  // Edge does not exist in IR.
  if (Kind == DomTreeT::Insert && !HasEdge)
    return false;

  // Edge exists in IR.
  if (Kind == DomTreeT::Delete && HasEdge)
    return false;

  return true;
}

template <typename DerivedT, typename DomTreeT, typename PostDomTreeT>
void GenericDomTreeUpdater<DerivedT, DomTreeT, PostDomTreeT>::eraseDelBBNode(
    BasicBlockT *DelBB) {
  if (DT && !IsRecalculatingDomTree)
    if (DT->getNode(DelBB))
      DT->eraseNode(DelBB);

  if (PDT && !IsRecalculatingPostDomTree)
    if (PDT->getNode(DelBB))
      PDT->eraseNode(DelBB);
}

template <typename DerivedT, typename DomTreeT, typename PostDomTreeT>
void GenericDomTreeUpdater<DerivedT, DomTreeT,
                           PostDomTreeT>::tryFlushDeletedBB() {
  if (!hasPendingUpdates())
    derived().forceFlushDeletedBB();
}

template <typename DerivedT, typename DomTreeT, typename PostDomTreeT>
void GenericDomTreeUpdater<DerivedT, DomTreeT,
                           PostDomTreeT>::dropOutOfDateUpdates() {
  if (Strategy == UpdateStrategy::Eager)
    return;

  tryFlushDeletedBB();

  // Drop all updates applied by both trees.
  if (!DT)
    PendDTUpdateIndex = PendUpdates.size();
  if (!PDT)
    PendPDTUpdateIndex = PendUpdates.size();

  const size_t dropIndex = std::min(PendDTUpdateIndex, PendPDTUpdateIndex);
  const auto B = PendUpdates.begin();
  const auto E = PendUpdates.begin() + dropIndex;
  assert(B <= E && "Iterator out of range.");
  PendUpdates.erase(B, E);
  // Calculate current index.
  PendDTUpdateIndex -= dropIndex;
  PendPDTUpdateIndex -= dropIndex;
}

template <typename DerivedT, typename DomTreeT, typename PostDomTreeT>
void GenericDomTreeUpdater<DerivedT, DomTreeT, PostDomTreeT>::
    splitDTCriticalEdges(ArrayRef<CriticalEdge> Edges) {
  // Bail out early if there is nothing to do.
  if (!DT || Edges.empty())
    return;

  // Remember all the basic blocks that are inserted during
  // edge splitting.
  // Invariant: NewBBs == all the basic blocks contained in the NewBB
  // field of all the elements of Edges.
  // I.e., forall elt in Edges, it exists BB in NewBBs
  // such as BB == elt.NewBB.
  SmallPtrSet<BasicBlockT *, 32> NewBBs;
  for (auto &Edge : Edges)
    NewBBs.insert(Edge.NewBB);
  // For each element in Edges, remember whether or not element
  // is the new immediate domminator of its successor. The mapping is done by
  // index, i.e., the information for the ith element of Edges is
  // the ith element of IsNewIDom.
  SmallBitVector IsNewIDom(Edges.size(), true);

  // Collect all the dominance properties info, before invalidating
  // the underlying DT.
  for (const auto &[Idx, Edge] : enumerate(Edges)) {
    // Update dominator information.
    BasicBlockT *Succ = Edge.ToBB;
    auto *SuccDTNode = DT->getNode(Succ);

    for (BasicBlockT *PredBB : predecessors(Succ)) {
      if (PredBB == Edge.NewBB)
        continue;
      // If we are in this situation:
      // FromBB1        FromBB2
      //    +              +
      //   + +            + +
      //  +   +          +   +
      // ...  Split1  Split2 ...
      //           +   +
      //            + +
      //             +
      //            Succ
      // Instead of checking the domiance property with Split2, we check it
      // with FromBB2 since Split2 is still unknown of the underlying DT
      // structure.
      if (NewBBs.contains(PredBB)) {
        assert(pred_size(PredBB) == 1 && "A basic block resulting from a "
                                         "critical edge split has more "
                                         "than one predecessor!");
        PredBB = *pred_begin(PredBB);
      }
      if (!DT->dominates(SuccDTNode, DT->getNode(PredBB))) {
        IsNewIDom[Idx] = false;
        break;
      }
    }
  }

  // Now, update DT with the collected dominance properties info.
  for (const auto &[Idx, Edge] : enumerate(Edges)) {
    // We know FromBB dominates NewBB.
    auto *NewDTNode = DT->addNewBlock(Edge.NewBB, Edge.FromBB);

    // If all the other predecessors of "Succ" are dominated by "Succ" itself
    // then the new block is the new immediate dominator of "Succ". Otherwise,
    // the new block doesn't dominate anything.
    if (IsNewIDom[Idx])
      DT->changeImmediateDominator(DT->getNode(Edge.ToBB), NewDTNode);
  }
}

// Post dominator tree is different, the new basic block in critical edge
// may become the new root.
template <typename DerivedT, typename DomTreeT, typename PostDomTreeT>
void GenericDomTreeUpdater<DerivedT, DomTreeT, PostDomTreeT>::
    splitPDTCriticalEdges(ArrayRef<CriticalEdge> Edges) {
  // Bail out early if there is nothing to do.
  if (!PDT || Edges.empty())
    return;

  std::vector<UpdateT> Updates;
  for (const auto &Edge : Edges) {
    Updates.push_back({PostDomTreeT::Insert, Edge.FromBB, Edge.NewBB});
    Updates.push_back({PostDomTreeT::Insert, Edge.NewBB, Edge.ToBB});
    if (!llvm::is_contained(successors(Edge.FromBB), Edge.ToBB))
      Updates.push_back({PostDomTreeT::Delete, Edge.FromBB, Edge.ToBB});
  }
  PDT->applyUpdates(Updates);
}

} // namespace llvm

#endif // LLVM_ANALYSIS_GENERICDOMTREEUPDATERIMPL_H
