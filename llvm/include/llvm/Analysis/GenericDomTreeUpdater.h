//===- GenericDomTreeUpdater.h ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the GenericDomTreeUpdater class, which provides a uniform
// way to update dominator tree related data structures.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_GENERICDOMTREEUPDATER_H
#define LLVM_ANALYSIS_GENERICDOMTREEUPDATER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {

template <typename DerivedT, typename DomTreeT, typename PostDomTreeT>
class GenericDomTreeUpdater {
  DerivedT &derived() { return *static_cast<DerivedT *>(this); }
  const DerivedT &derived() const {
    return *static_cast<const DerivedT *>(this);
  }

public:
  enum class UpdateStrategy : unsigned char { Eager = 0, Lazy = 1 };
  using BasicBlockT = typename DomTreeT::NodeType;

  explicit GenericDomTreeUpdater(UpdateStrategy Strategy_)
      : Strategy(Strategy_) {}
  GenericDomTreeUpdater(DomTreeT &DT_, UpdateStrategy Strategy_)
      : DT(&DT_), Strategy(Strategy_) {}
  GenericDomTreeUpdater(DomTreeT *DT_, UpdateStrategy Strategy_)
      : DT(DT_), Strategy(Strategy_) {}
  GenericDomTreeUpdater(PostDomTreeT &PDT_, UpdateStrategy Strategy_)
      : PDT(&PDT_), Strategy(Strategy_) {}
  GenericDomTreeUpdater(PostDomTreeT *PDT_, UpdateStrategy Strategy_)
      : PDT(PDT_), Strategy(Strategy_) {}
  GenericDomTreeUpdater(DomTreeT &DT_, PostDomTreeT &PDT_,
                        UpdateStrategy Strategy_)
      : DT(&DT_), PDT(&PDT_), Strategy(Strategy_) {}
  GenericDomTreeUpdater(DomTreeT *DT_, PostDomTreeT *PDT_,
                        UpdateStrategy Strategy_)
      : DT(DT_), PDT(PDT_), Strategy(Strategy_) {}

  ~GenericDomTreeUpdater() { flush(); }

  /// Returns true if the current strategy is Lazy.
  bool isLazy() const { return Strategy == UpdateStrategy::Lazy; };

  /// Returns true if the current strategy is Eager.
  bool isEager() const { return Strategy == UpdateStrategy::Eager; };

  /// Returns true if it holds a DomTreeT.
  bool hasDomTree() const { return DT != nullptr; }

  /// Returns true if it holds a PostDomTreeT.
  bool hasPostDomTree() const { return PDT != nullptr; }

  /// Returns true if there is BasicBlockT awaiting deletion.
  /// The deletion will only happen until a flush event and
  /// all available trees are up-to-date.
  /// Returns false under Eager UpdateStrategy.
  bool hasPendingDeletedBB() const { return !DeletedBBs.empty(); }

  /// Returns true if DelBB is awaiting deletion.
  /// Returns false under Eager UpdateStrategy.
  bool isBBPendingDeletion(BasicBlockT *DelBB) const {
    if (Strategy == UpdateStrategy::Eager || DeletedBBs.empty())
      return false;
    return DeletedBBs.contains(DelBB);
  }

  /// Returns true if either of DT or PDT is valid and the tree has at
  /// least one update pending. If DT or PDT is nullptr it is treated
  /// as having no pending updates. This function does not check
  /// whether there is MachineBasicBlock awaiting deletion.
  /// Returns false under Eager UpdateStrategy.
  bool hasPendingUpdates() const {
    return hasPendingDomTreeUpdates() || hasPendingPostDomTreeUpdates();
  }

  /// Returns true if there are DomTreeT updates queued.
  /// Returns false under Eager UpdateStrategy or DT is nullptr.
  bool hasPendingDomTreeUpdates() const {
    if (!DT)
      return false;
    return PendUpdates.size() != PendDTUpdateIndex;
  }

  /// Returns true if there are PostDomTreeT updates queued.
  /// Returns false under Eager UpdateStrategy or PDT is nullptr.
  bool hasPendingPostDomTreeUpdates() const {
    if (!PDT)
      return false;
    return PendUpdates.size() != PendPDTUpdateIndex;
  }

  ///@{
  /// \name Mutation APIs
  ///
  /// These methods provide APIs for submitting updates to the DomTreeT and
  /// the PostDominatorTree.
  ///
  /// Note: There are two strategies to update the DomTreeT and the
  /// PostDominatorTree:
  /// 1. Eager UpdateStrategy: Updates are submitted and then flushed
  /// immediately.
  /// 2. Lazy UpdateStrategy: Updates are submitted but only flushed when you
  /// explicitly call Flush APIs. It is recommended to use this update strategy
  /// when you submit a bunch of updates multiple times which can then
  /// add up to a large number of updates between two queries on the
  /// DomTreeT. The incremental updater can reschedule the updates or
  /// decide to recalculate the dominator tree in order to speedup the updating
  /// process depending on the number of updates.
  ///
  /// Although GenericDomTree provides several update primitives,
  /// it is not encouraged to use these APIs directly.

  /// Notify DTU that the entry block was replaced.
  /// Recalculate all available trees and flush all BasicBlocks
  /// awaiting deletion immediately.
  template <typename FuncT> void recalculate(FuncT &F) {
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

  /// Submit updates to all available trees.
  /// The Eager Strategy flushes updates immediately while the Lazy Strategy
  /// queues the updates.
  ///
  /// Note: The "existence" of an edge in a CFG refers to the CFG which DTU is
  /// in sync with + all updates before that single update.
  ///
  /// CAUTION!
  /// 1. It is required for the state of the LLVM IR to be updated
  /// *before* submitting the updates because the internal update routine will
  /// analyze the current state of the CFG to determine whether an update
  /// is valid.
  /// 2. It is illegal to submit any update that has already been submitted,
  /// i.e., you are supposed not to insert an existent edge or delete a
  /// nonexistent edge.
  void applyUpdates(ArrayRef<typename DomTreeT::UpdateType> Updates) {
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

  /// Submit updates to all available trees. It will also
  /// 1. discard duplicated updates,
  /// 2. remove invalid updates. (Invalid updates means deletion of an edge that
  /// still exists or insertion of an edge that does not exist.)
  /// The Eager Strategy flushes updates immediately while the Lazy Strategy
  /// queues the updates.
  ///
  /// Note: The "existence" of an edge in a CFG refers to the CFG which DTU is
  /// in sync with + all updates before that single update.
  ///
  /// CAUTION!
  /// 1. It is required for the state of the LLVM IR to be updated
  /// *before* submitting the updates because the internal update routine will
  /// analyze the current state of the CFG to determine whether an update
  /// is valid.
  /// 2. It is illegal to submit any update that has already been submitted,
  /// i.e., you are supposed not to insert an existent edge or delete a
  /// nonexistent edge.
  /// 3. It is only legal to submit updates to an edge in the order CFG changes
  /// are made. The order you submit updates on different edges is not
  /// restricted.
  void applyUpdatesPermissive(ArrayRef<typename DomTreeT::UpdateType> Updates) {
    if (!DT && !PDT)
      return;

    SmallSet<std::pair<BasicBlockT *, BasicBlockT *>, 8> Seen;
    SmallVector<typename DomTreeT::UpdateType, 8> DeduplicatedUpdates;
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
      if (!isSelfDominance(U) && Seen.count(Edge) == 0) {
        Seen.insert(Edge);
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

  ///@}

  ///@{
  /// \name Flush APIs
  ///
  /// CAUTION! By the moment these flush APIs are called, the current CFG needs
  /// to be the same as the CFG which DTU is in sync with + all updates
  /// submitted.

  /// Flush DomTree updates and return DomTree.
  /// It flushes Deleted BBs if both trees are up-to-date.
  /// It must only be called when it has a DomTree.
  DomTreeT &getDomTree() {
    assert(DT && "Invalid acquisition of a null DomTree");
    applyDomTreeUpdates();
    dropOutOfDateUpdates();
    return *DT;
  }

  /// Flush PostDomTree updates and return PostDomTree.
  /// It flushes Deleted BBs if both trees are up-to-date.
  /// It must only be called when it has a PostDomTree.
  PostDomTreeT &getPostDomTree() {
    assert(PDT && "Invalid acquisition of a null PostDomTree");
    applyPostDomTreeUpdates();
    dropOutOfDateUpdates();
    return *PDT;
  }

  /// Apply all pending updates to available trees and flush all BasicBlocks
  /// awaiting deletion.

  void flush() {
    applyDomTreeUpdates();
    applyPostDomTreeUpdates();
    dropOutOfDateUpdates();
  }

  ///@}

  /// Debug method to help view the internal state of this class.
  LLVM_DUMP_METHOD void dump() const {
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

    auto printUpdates =
        [&](typename ArrayRef<typename DomTreeT::UpdateType>::const_iterator
                begin,
            typename ArrayRef<typename DomTreeT::UpdateType>::const_iterator
                end) {
          if (begin == end)
            OS << "  None\n";
          Index = 0;
          for (auto It = begin, ItEnd = end; It != ItEnd; ++It) {
            auto U = *It;
            OS << "  " << Index << " : ";
            ++Index;
            if (U.getKind() == DomTreeT::Insert)
              OS << "Insert, ";
            else
              OS << "Delete, ";
            BasicBlockT *From = U.getFrom();
            if (From) {
              auto S = From->getName();
              if (!From->hasName())
                S = "(no name)";
              OS << S << "(" << From << "), ";
            } else {
              OS << "(badref), ";
            }
            BasicBlockT *To = U.getTo();
            if (To) {
              auto S = To->getName();
              if (!To->hasName())
                S = "(no_name)";
              OS << S << "(" << To << ")\n";
            } else {
              OS << "(badref)\n";
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
        OS << "(no_name)(";
      OS << BB << ")\n";
    }
#endif
  }

protected:
  SmallVector<typename DomTreeT::UpdateType, 16> PendUpdates;
  size_t PendDTUpdateIndex = 0;
  size_t PendPDTUpdateIndex = 0;
  DomTreeT *DT = nullptr;
  PostDomTreeT *PDT = nullptr;
  const UpdateStrategy Strategy;
  SmallPtrSet<BasicBlockT *, 8> DeletedBBs;
  bool IsRecalculatingDomTree = false;
  bool IsRecalculatingPostDomTree = false;

  /// Returns true if the update is self dominance.
  bool isSelfDominance(typename DomTreeT::UpdateType Update) const {
    // Won't affect DomTree and PostDomTree.
    return Update.getFrom() == Update.getTo();
  }

  /// Helper function to apply all pending DomTree updates.
  void applyDomTreeUpdates() {
    // No pending DomTreeUpdates.
    if (Strategy != UpdateStrategy::Lazy || !DT)
      return;

    // Only apply updates not are applied by DomTree.
    if (hasPendingDomTreeUpdates()) {
      const auto I = PendUpdates.begin() + PendDTUpdateIndex;
      const auto E = PendUpdates.end();
      assert(I < E &&
             "Iterator range invalid; there should be DomTree updates.");
      DT->applyUpdates(ArrayRef<typename DomTreeT::UpdateType>(I, E));
      PendDTUpdateIndex = PendUpdates.size();
    }
  }

  /// Helper function to apply all pending PostDomTree updates.
  void applyPostDomTreeUpdates() {
    // No pending PostDomTreeUpdates.
    if (Strategy != UpdateStrategy::Lazy || !PDT)
      return;

    // Only apply updates not are applied by PostDomTree.
    if (hasPendingPostDomTreeUpdates()) {
      const auto I = PendUpdates.begin() + PendPDTUpdateIndex;
      const auto E = PendUpdates.end();
      assert(I < E &&
             "Iterator range invalid; there should be PostDomTree updates.");
      PDT->applyUpdates(ArrayRef<typename DomTreeT::UpdateType>(I, E));
      PendPDTUpdateIndex = PendUpdates.size();
    }
  }

  /// Returns true if the update appears in the LLVM IR.
  /// It is used to check whether an update is valid in
  /// insertEdge/deleteEdge or is unnecessary in the batch update.
  bool isUpdateValid(typename DomTreeT::UpdateType Update) const {
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

  /// Erase Basic Block node that has been unlinked from Function
  /// in the DomTree and PostDomTree.
  void eraseDelBBNode(BasicBlockT *DelBB) {
    if (DT && !IsRecalculatingDomTree)
      if (DT->getNode(DelBB))
        DT->eraseNode(DelBB);

    if (PDT && !IsRecalculatingPostDomTree)
      if (PDT->getNode(DelBB))
        PDT->eraseNode(DelBB);
  }

  /// Helper function to flush deleted BasicBlocks if all available
  /// trees are up-to-date.
  void tryFlushDeletedBB() {
    if (!hasPendingUpdates())
      derived().forceFlushDeletedBB();
  }

  /// Drop all updates applied by all available trees and delete BasicBlocks if
  /// all available trees are up-to-date.
  void dropOutOfDateUpdates() {
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
};

} // namespace llvm

#endif // LLVM_ANALYSIS_GENERICDOMTREEUPDATER_H
