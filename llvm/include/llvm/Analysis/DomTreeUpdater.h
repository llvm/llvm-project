//===- DomTreeUpdater.h - DomTree/Post DomTree Updater ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the DomTreeUpdater class, which provides a uniform way to
// update dominator tree related data structures.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_DOMTREEUPDATER_H
#define LLVM_ANALYSIS_DOMTREEUPDATER_H

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/Support/Compiler.h"
#include <cstddef>
#include <functional>
#include <vector>

namespace llvm {
class DomTreeUpdater {
public:
  enum class UpdateStrategy : unsigned char { Eager = 0, Lazy = 1 };

  explicit DomTreeUpdater(UpdateStrategy Strategy_) : Strategy(Strategy_) {}
  DomTreeUpdater(DominatorTree &DT_, UpdateStrategy Strategy_)
      : DT(&DT_), Strategy(Strategy_) {}
  DomTreeUpdater(DominatorTree *DT_, UpdateStrategy Strategy_)
      : DT(DT_), Strategy(Strategy_) {}
  DomTreeUpdater(PostDominatorTree &PDT_, UpdateStrategy Strategy_)
      : PDT(&PDT_), Strategy(Strategy_) {}
  DomTreeUpdater(PostDominatorTree *PDT_, UpdateStrategy Strategy_)
      : PDT(PDT_), Strategy(Strategy_) {}
  DomTreeUpdater(DominatorTree &DT_, PostDominatorTree &PDT_,
                 UpdateStrategy Strategy_)
      : DT(&DT_), PDT(&PDT_), Strategy(Strategy_) {}
  DomTreeUpdater(DominatorTree *DT_, PostDominatorTree *PDT_,
                 UpdateStrategy Strategy_)
      : DT(DT_), PDT(PDT_), Strategy(Strategy_) {}

  ~DomTreeUpdater() { flush(); }

  /// Returns true if the current strategy is Lazy.
  bool isLazy() const { return Strategy == UpdateStrategy::Lazy; };

  /// Returns true if the current strategy is Eager.
  bool isEager() const { return Strategy == UpdateStrategy::Eager; };

  /// Returns true if it holds a DominatorTree.
  bool hasDomTree() const { return DT != nullptr; }

  /// Returns true if it holds a PostDominatorTree.
  bool hasPostDomTree() const { return PDT != nullptr; }

  /// Returns true if there is BasicBlock awaiting deletion.
  /// The deletion will only happen until a flush event and
  /// all available trees are up-to-date.
  /// Returns false under Eager UpdateStrategy.
  bool hasPendingDeletedBB() const { return !DeletedBBs.empty(); }

  /// Returns true if DelBB is awaiting deletion.
  /// Returns false under Eager UpdateStrategy.
  bool isBBPendingDeletion(BasicBlock *DelBB) const;

  /// Returns true if either of DT or PDT is valid and the tree has at
  /// least one update pending. If DT or PDT is nullptr it is treated
  /// as having no pending updates. This function does not check
  /// whether there is BasicBlock awaiting deletion.
  /// Returns false under Eager UpdateStrategy.
  bool hasPendingUpdates() const;

  /// Returns true if there are DominatorTree updates queued.
  /// Returns false under Eager UpdateStrategy or DT is nullptr.
  bool hasPendingDomTreeUpdates() const;

  /// Returns true if there are PostDominatorTree updates queued.
  /// Returns false under Eager UpdateStrategy or PDT is nullptr.
  bool hasPendingPostDomTreeUpdates() const;

  ///@{
  /// \name Mutation APIs
  ///
  /// These methods provide APIs for submitting updates to the DominatorTree and
  /// the PostDominatorTree.
  ///
  /// Note: There are two strategies to update the DominatorTree and the
  /// PostDominatorTree:
  /// 1. Eager UpdateStrategy: Updates are submitted and then flushed
  /// immediately.
  /// 2. Lazy UpdateStrategy: Updates are submitted but only flushed when you
  /// explicitly call Flush APIs. It is recommended to use this update strategy
  /// when you submit a bunch of updates multiple times which can then
  /// add up to a large number of updates between two queries on the
  /// DominatorTree. The incremental updater can reschedule the updates or
  /// decide to recalculate the dominator tree in order to speedup the updating
  /// process depending on the number of updates.
  ///
  /// Although GenericDomTree provides several update primitives,
  /// it is not encouraged to use these APIs directly.

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
  void applyUpdates(ArrayRef<DominatorTree::UpdateType> Updates);

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
  void applyUpdatesPermissive(ArrayRef<DominatorTree::UpdateType> Updates);

  /// Notify DTU that the entry block was replaced.
  /// Recalculate all available trees and flush all BasicBlocks
  /// awaiting deletion immediately.
  void recalculate(Function &F);

  /// \deprecated { Submit an edge insertion to all available trees. The Eager
  /// Strategy flushes this update immediately while the Lazy Strategy queues
  /// the update. An internal function checks if the edge exists in the CFG in
  /// DEBUG mode. CAUTION! This function has to be called *after* making the
  /// update on the actual CFG. It is illegal to submit any update that has
  /// already been applied. }
  LLVM_ATTRIBUTE_DEPRECATED(void insertEdge(BasicBlock *From, BasicBlock *To),
                            "Use applyUpdates() instead.");

  /// \deprecated {Submit an edge insertion to all available trees.
  /// Under either Strategy, an invalid update will be discard silently.
  /// Invalid update means inserting an edge that does not exist in the CFG.
  /// The Eager Strategy flushes this update immediately while the Lazy Strategy
  /// queues the update. It is only recommended to use this method when you
  /// want to discard an invalid update.
  /// CAUTION! It is illegal to submit any update that has already been
  /// submitted. }
  LLVM_ATTRIBUTE_DEPRECATED(void insertEdgeRelaxed(BasicBlock *From,
                                                   BasicBlock *To),
                            "Use applyUpdatesPermissive() instead.");

  /// \deprecated { Submit an edge deletion to all available trees. The Eager
  /// Strategy flushes this update immediately while the Lazy Strategy queues
  /// the update. An internal function checks if the edge doesn't exist in the
  /// CFG in DEBUG mode.
  /// CAUTION! This function has to be called *after* making the update on the
  /// actual CFG. It is illegal to submit any update that has already been
  /// submitted. }
  LLVM_ATTRIBUTE_DEPRECATED(void deleteEdge(BasicBlock *From, BasicBlock *To),
                            "Use applyUpdates() instead.");

  /// \deprecated { Submit an edge deletion to all available trees.
  /// Under either Strategy, an invalid update will be discard silently.
  /// Invalid update means deleting an edge that exists in the CFG.
  /// The Eager Strategy flushes this update immediately while the Lazy Strategy
  /// queues the update. It is only recommended to use this method when you
  /// want to discard an invalid update.
  /// CAUTION! It is illegal to submit any update that has already been
  /// submitted. }
  LLVM_ATTRIBUTE_DEPRECATED(void deleteEdgeRelaxed(BasicBlock *From,
                                                   BasicBlock *To),
                            "Use applyUpdatesPermissive() instead.");

  /// Delete DelBB. DelBB will be removed from its Parent and
  /// erased from available trees if it exists and finally get deleted.
  /// Under Eager UpdateStrategy, DelBB will be processed immediately.
  /// Under Lazy UpdateStrategy, DelBB will be queued until a flush event and
  /// all available trees are up-to-date. Assert if any instruction of DelBB is
  /// modified while awaiting deletion. When both DT and PDT are nullptrs, DelBB
  /// will be queued until flush() is called.
  void deleteBB(BasicBlock *DelBB);

  /// Delete DelBB. DelBB will be removed from its Parent and
  /// erased from available trees if it exists. Then the callback will
  /// be called. Finally, DelBB will be deleted.
  /// Under Eager UpdateStrategy, DelBB will be processed immediately.
  /// Under Lazy UpdateStrategy, DelBB will be queued until a flush event and
  /// all available trees are up-to-date. Assert if any instruction of DelBB is
  /// modified while awaiting deletion. Multiple callbacks can be queued for one
  /// DelBB under Lazy UpdateStrategy.
  void callbackDeleteBB(BasicBlock *DelBB,
                        std::function<void(BasicBlock *)> Callback);

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
  DominatorTree &getDomTree();

  /// Flush PostDomTree updates and return PostDomTree.
  /// It flushes Deleted BBs if both trees are up-to-date.
  /// It must only be called when it has a PostDomTree.
  PostDominatorTree &getPostDomTree();

  /// Apply all pending updates to available trees and flush all BasicBlocks
  /// awaiting deletion.

  void flush();

  ///@}

  /// Debug method to help view the internal state of this class.
  LLVM_DUMP_METHOD void dump() const;

private:
  class CallBackOnDeletion final : public CallbackVH {
  public:
    CallBackOnDeletion(BasicBlock *V,
                       std::function<void(BasicBlock *)> Callback)
        : CallbackVH(V), DelBB(V), Callback_(Callback) {}

  private:
    BasicBlock *DelBB = nullptr;
    std::function<void(BasicBlock *)> Callback_;

    void deleted() override {
      Callback_(DelBB);
      CallbackVH::deleted();
    }
  };

  SmallVector<DominatorTree::UpdateType, 16> PendUpdates;
  size_t PendDTUpdateIndex = 0;
  size_t PendPDTUpdateIndex = 0;
  DominatorTree *DT = nullptr;
  PostDominatorTree *PDT = nullptr;
  const UpdateStrategy Strategy;
  SmallPtrSet<BasicBlock *, 8> DeletedBBs;
  std::vector<CallBackOnDeletion> Callbacks;
  bool IsRecalculatingDomTree = false;
  bool IsRecalculatingPostDomTree = false;

  /// First remove all the instructions of DelBB and then make sure DelBB has a
  /// valid terminator instruction which is necessary to have when DelBB still
  /// has to be inside of its parent Function while awaiting deletion under Lazy
  /// UpdateStrategy to prevent other routines from asserting the state of the
  /// IR is inconsistent. Assert if DelBB is nullptr or has predecessors.
  void validateDeleteBB(BasicBlock *DelBB);

  /// Returns true if at least one BasicBlock is deleted.
  bool forceFlushDeletedBB();

  /// Helper function to apply all pending DomTree updates.
  void applyDomTreeUpdates();

  /// Helper function to apply all pending PostDomTree updates.
  void applyPostDomTreeUpdates();

  /// Helper function to flush deleted BasicBlocks if all available
  /// trees are up-to-date.
  void tryFlushDeletedBB();

  /// Drop all updates applied by all available trees and delete BasicBlocks if
  /// all available trees are up-to-date.
  void dropOutOfDateUpdates();

  /// Erase Basic Block node that has been unlinked from Function
  /// in the DomTree and PostDomTree.
  void eraseDelBBNode(BasicBlock *DelBB);

  /// Returns true if the update appears in the LLVM IR.
  /// It is used to check whether an update is valid in
  /// insertEdge/deleteEdge or is unnecessary in the batch update.
  bool isUpdateValid(DominatorTree::UpdateType Update) const;

  /// Returns true if the update is self dominance.
  bool isSelfDominance(DominatorTree::UpdateType Update) const;
};
} // namespace llvm

#endif // LLVM_ANALYSIS_DOMTREEUPDATER_H
