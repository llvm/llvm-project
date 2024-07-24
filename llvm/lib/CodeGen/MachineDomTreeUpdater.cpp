//===- MachineDomTreeUpdater.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the MachineDomTreeUpdater class, which provides a
// uniform way to update dominator tree related data structures.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineDomTreeUpdater.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Analysis/GenericDomTreeUpdaterImpl.h"
#include "llvm/CodeGen/MachinePostDominators.h"
#include "llvm/Support/GenericDomTree.h"
#include <algorithm>
#include <functional>
#include <utility>

namespace llvm {

template class GenericDomTreeUpdater<
    MachineDomTreeUpdater, MachineDominatorTree, MachinePostDominatorTree>;

template void
GenericDomTreeUpdater<MachineDomTreeUpdater, MachineDominatorTree,
                      MachinePostDominatorTree>::recalculate(MachineFunction
                                                                 &MF);

bool MachineDomTreeUpdater::forceFlushDeletedBB() {
  if (DeletedBBs.empty())
    return false;

  for (auto *BB : DeletedBBs) {
    eraseDelBBNode(BB);
    BB->eraseFromParent();
  }
  DeletedBBs.clear();
  return true;
}

// The DT and PDT require the nodes related to updates
// are not deleted when update functions are called.
// So MachineBasicBlock deletions must be pended when the
// UpdateStrategy is Lazy. When the UpdateStrategy is
// Eager, the MachineBasicBlock will be deleted immediately.
void MachineDomTreeUpdater::deleteBB(MachineBasicBlock *DelBB) {
  validateDeleteBB(DelBB);
  if (Strategy == UpdateStrategy::Lazy) {
    DeletedBBs.insert(DelBB);
    return;
  }

  eraseDelBBNode(DelBB);
  DelBB->eraseFromParent();
}

void MachineDomTreeUpdater::validateDeleteBB(MachineBasicBlock *DelBB) {
  assert(DelBB && "Invalid push_back of nullptr DelBB.");
  assert(DelBB->pred_empty() && "DelBB has one or more predecessors.");
}

void MachineDomTreeUpdater::applyUpdatesForCriticalEdgeSplitting(
    MachineBasicBlock *FromBB, MachineBasicBlock *ToBB,
    MachineBasicBlock *NewBB) {

  bool Inserted = NewBBs.insert(NewBB).second;
  (void)Inserted;
  assert(Inserted &&
         "A basic block inserted via edge splitting cannot appear twice");
  CriticalEdgesToSplit.push_back({FromBB, ToBB, NewBB});

  if (isEager())
    applySplitCriticalEdges();
}

void MachineDomTreeUpdater::applySplitCriticalEdges() {
  // Bail out early if there is nothing to do.
  if (!DT || CriticalEdgesToSplit.empty())
    return;
  assert(!hasPendingUpdates() && "applyUpdatesForCriticalEdgeSplitting is "
                                 "incompatible with regular updates!");

  // For each element in CriticalEdgesToSplit, remember whether or not element
  // is the new immediate domminator of its successor. The mapping is done by
  // index, i.e., the information for the ith element of CriticalEdgesToSplit is
  // the ith element of IsNewIDom.
  SmallBitVector IsNewIDom(CriticalEdgesToSplit.size(), true);
  size_t Idx = 0;

  // Collect all the dominance properties info, before invalidating
  // the underlying DT.
  for (CriticalEdge &Edge : CriticalEdgesToSplit) {
    // Update dominator information.
    MachineBasicBlock *Succ = Edge.ToBB;
    MachineDomTreeNode *SuccDTNode = DT->getNode(Succ);

    for (MachineBasicBlock *PredBB : Succ->predecessors()) {
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
      // Instead of checking the domiance property with Split2, we check it with
      // FromBB2 since Split2 is still unknown of the underlying DT structure.
      if (NewBBs.count(PredBB)) {
        assert(PredBB->pred_size() == 1 && "A basic block resulting from a "
                                           "critical edge split has more "
                                           "than one predecessor!");
        PredBB = *PredBB->pred_begin();
      }
      if (!DT->dominates(SuccDTNode, DT->getNode(PredBB))) {
        IsNewIDom[Idx] = false;
        break;
      }
    }
    ++Idx;
  }

  // Now, update DT with the collected dominance properties info.
  Idx = 0;
  for (CriticalEdge &Edge : CriticalEdgesToSplit) {
    // We know FromBB dominates NewBB.
    MachineDomTreeNode *NewDTNode = DT->addNewBlock(Edge.NewBB, Edge.FromBB);

    // If all the other predecessors of "Succ" are dominated by "Succ" itself
    // then the new block is the new immediate dominator of "Succ". Otherwise,
    // the new block doesn't dominate anything.
    if (IsNewIDom[Idx])
      DT->changeImmediateDominator(DT->getNode(Edge.ToBB), NewDTNode);
    ++Idx;
  }
  NewBBs.clear();
  CriticalEdgesToSplit.clear();
}

} // namespace llvm
