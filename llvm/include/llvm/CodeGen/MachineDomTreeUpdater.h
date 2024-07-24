//===- llvm/CodeGen/MachineDomTreeUpdater.h -----------------------*- C++-*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file exposes interfaces to post dominance information for
// target-specific code.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MACHINEDOMTREEUPDATER_H
#define LLVM_CODEGEN_MACHINEDOMTREEUPDATER_H

#include "llvm/Analysis/GenericDomTreeUpdater.h"
#include "llvm/CodeGen/MachineDominators.h"

namespace llvm {

class MachinePostDominatorTree;

class MachineDomTreeUpdater
    : public GenericDomTreeUpdater<MachineDomTreeUpdater, MachineDominatorTree,
                                   MachinePostDominatorTree> {
  friend GenericDomTreeUpdater<MachineDomTreeUpdater, MachineDominatorTree,
                               MachinePostDominatorTree>;

public:
  using Base =
      GenericDomTreeUpdater<MachineDomTreeUpdater, MachineDominatorTree,
                            MachinePostDominatorTree>;
  using Base::Base;

  ~MachineDomTreeUpdater() { flush(); }

  ///@{
  /// \name Mutation APIs
  ///

  /// Delete DelBB. DelBB will be removed from its Parent and
  /// erased from available trees if it exists and finally get deleted.
  /// Under Eager UpdateStrategy, DelBB will be processed immediately.
  /// Under Lazy UpdateStrategy, DelBB will be queued until a flush event and
  /// all available trees are up-to-date. Assert if any instruction of DelBB is
  /// modified while awaiting deletion. When both DT and PDT are nullptrs, DelBB
  /// will be queued until flush() is called.
  void deleteBB(MachineBasicBlock *DelBB);

  ///@}

  /// Apply updates that the critical edge (FromBB, ToBB) has been
  /// split with NewBB.
  ///
  /// \note Do not use this method with regular edges.
  ///
  /// \note This method only updates forward dominator tree, and is incompatible
  /// with regular updates.
  void applyUpdatesForCriticalEdgeSplitting(MachineBasicBlock *FromBB,
                                            MachineBasicBlock *ToBB,
                                            MachineBasicBlock *NewBB);

  void flush() {
    if (CriticalEdgesToSplit.empty())
      Base::flush();
    else
      applySplitCriticalEdges();
  }

private:
  /// Helper structure used to hold all the basic blocks
  /// involved in the split of a critical edge.
  struct CriticalEdge {
    MachineBasicBlock *FromBB;
    MachineBasicBlock *ToBB;
    MachineBasicBlock *NewBB;
  };

  /// Pile up all the critical edges to be split.
  /// The splitting of a critical edge is local and thus, it is possible
  /// to apply several of those changes at the same time.
  SmallVector<CriticalEdge, 32> CriticalEdgesToSplit;

  /// Remember all the basic blocks that are inserted during
  /// edge splitting.
  /// Invariant: NewBBs == all the basic blocks contained in the NewBB
  /// field of all the elements of CriticalEdgesToSplit.
  /// I.e., forall elt in CriticalEdgesToSplit, it exists BB in NewBBs
  /// such as BB == elt.NewBB.
  SmallSet<MachineBasicBlock *, 32> NewBBs;

  /// Apply all the recorded critical edges to the DT.
  /// This updates the underlying DT information in a way that uses
  /// the fast query path of DT as much as possible.
  ///
  /// \post CriticalEdgesToSplit.empty().
  void applySplitCriticalEdges();

  /// First remove all the instructions of DelBB and then make sure DelBB has a
  /// valid terminator instruction which is necessary to have when DelBB still
  /// has to be inside of its parent Function while awaiting deletion under Lazy
  /// UpdateStrategy to prevent other routines from asserting the state of the
  /// IR is inconsistent. Assert if DelBB is nullptr or has predecessors.
  void validateDeleteBB(MachineBasicBlock *DelBB);

  /// Returns true if at least one MachineBasicBlock is deleted.
  bool forceFlushDeletedBB();
};

extern template class GenericDomTreeUpdater<
    MachineDomTreeUpdater, MachineDominatorTree, MachinePostDominatorTree>;

extern template void
GenericDomTreeUpdater<MachineDomTreeUpdater, MachineDominatorTree,
                      MachinePostDominatorTree>::recalculate(MachineFunction
                                                                 &MF);
} // namespace llvm
#endif // LLVM_CODEGEN_MACHINEDOMTREEUPDATER_H
