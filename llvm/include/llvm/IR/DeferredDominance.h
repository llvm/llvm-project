//===- DeferredDominance.h - Deferred Dominators ----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the DeferredDominance class, which provides deferred
// updates to Dominators.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_DEFERREDDOMINANCE_H
#define LLVM_IR_DEFERREDDOMINANCE_H

#include "llvm/ADT/SmallSet.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instructions.h"

namespace llvm {

/// \brief Class to defer updates to a DominatorTree.
///
/// Definition: Applying updates to every edge insertion and deletion is
/// expensive and not necessary. When one needs the DominatorTree for analysis
/// they can request a flush() to perform a larger batch update. This has the
/// advantage of the DominatorTree inspecting the set of updates to find
/// duplicates or unnecessary subtree updates.
///
/// The scope of DeferredDominance operates at a Function level.
///
/// It is not necessary for the user to scrub the updates for duplicates or
/// updates that point to the same block (Delete, BB_A, BB_A). Performance
/// can be gained if the caller attempts to batch updates before submitting
/// to applyUpdates(ArrayRef) in cases where duplicate edge requests will
/// occur.
///
/// It is required for the state of the LLVM IR to be applied *before*
/// submitting updates. The update routines must analyze the current state
/// between a pair of (From, To) basic blocks to determine if the update
/// needs to be queued.
/// Example (good):
///     TerminatorInstructionBB->removeFromParent();
///     DDT->deleteEdge(BB, Successor);
/// Example (bad):
///     DDT->deleteEdge(BB, Successor);
///     TerminatorInstructionBB->removeFromParent();
class DeferredDominance {
public:
  DeferredDominance(DominatorTree &DT_) : DT(DT_) {}

  /// \brief Queues multiple updates and discards duplicates.
  void applyUpdates(ArrayRef<DominatorTree::UpdateType> Updates) {
    SmallVector<DominatorTree::UpdateType, 8> Seen;
    for (auto U : Updates)
      // Avoid duplicates to applyUpdate() to save on analysis.
      if (std::none_of(Seen.begin(), Seen.end(),
                       [U](DominatorTree::UpdateType S) { return S == U; })) {
        Seen.push_back(U);
        applyUpdate(U.getKind(), U.getFrom(), U.getTo());
      }
  }

  void insertEdge(BasicBlock *From, BasicBlock *To) {
    applyUpdate(DominatorTree::Insert, From, To);
  }

  void deleteEdge(BasicBlock *From, BasicBlock *To) {
    applyUpdate(DominatorTree::Delete, From, To);
  }

  /// \brief Delays the deletion of a basic block until a flush() event.
  void deleteBB(BasicBlock *DelBB) {
    assert(DelBB && "Invalid push_back of nullptr DelBB.");
    assert(pred_empty(DelBB) && "DelBB has one or more predecessors.");
    // DelBB is unreachable and all its instructions are dead.
    while (!DelBB->empty()) {
      Instruction &I = DelBB->back();
      // Replace used instructions with an arbitrary value (undef).
      if (!I.use_empty())
        I.replaceAllUsesWith(llvm::UndefValue::get(I.getType()));
      DelBB->getInstList().pop_back();
    }
    // Make sure DelBB has a valid terminator instruction. As long as DelBB is
    // a Child of Function F it must contain valid IR.
    new UnreachableInst(DelBB->getContext(), DelBB);
    DeletedBBs.insert(DelBB);
  }

  /// \brief Returns true if DelBB is awaiting deletion at a flush() event.
  bool pendingDeletedBB(BasicBlock *DelBB) {
    if (DeletedBBs.empty())
      return false;
    return DeletedBBs.count(DelBB) != 0;
  }

  /// \brief Flushes all pending updates and block deletions. Returns a
  /// correct DominatorTree reference to be used by the caller for analysis.
  DominatorTree &flush() {
    // Updates to DT must happen before blocks are deleted below. Otherwise the
    // DT traversal will encounter badref blocks and assert.
    if (!PendUpdates.empty()) {
      DT.applyUpdates(PendUpdates);
      PendUpdates.clear();
    }
    flushDelBB();
    return DT;
  }

  /// \brief Drops all internal state and forces a (slow) recalculation of the
  /// DominatorTree based on the current state of the LLVM IR in F. This should
  /// only be used in corner cases such as the Entry block of F being deleted.
  void recalculate(Function &F) {
    // flushDelBB must be flushed before the recalculation. The state of the IR
    // must be consistent before the DT traversal algorithm determines the
    // actual DT.
    if (flushDelBB() || !PendUpdates.empty()) {
      DT.recalculate(F);
      PendUpdates.clear();
    }
  }

  /// \brief Debug method to help view the state of pending updates.
  LLVM_DUMP_METHOD void dump() const;

private:
  DominatorTree &DT;
  SmallVector<DominatorTree::UpdateType, 16> PendUpdates;
  SmallPtrSet<BasicBlock *, 8> DeletedBBs;

  /// Apply an update (Kind, From, To) to the internal queued updates. The
  /// update is only added when determined to be necessary. Checks for
  /// self-domination, unnecessary updates, duplicate requests, and balanced
  /// pairs of requests are all performed. Returns true if the update is
  /// queued and false if it is discarded.
  bool applyUpdate(DominatorTree::UpdateKind Kind, BasicBlock *From,
                   BasicBlock *To) {
    if (From == To)
      return false; // Cannot dominate self; discard update.

    // Discard updates by inspecting the current state of successors of From.
    // Since applyUpdate() must be called *after* the Terminator of From is
    // altered we can determine if the update is unnecessary.
    bool HasEdge = std::any_of(succ_begin(From), succ_end(From),
                               [To](BasicBlock *B) { return B == To; });
    if (Kind == DominatorTree::Insert && !HasEdge)
      return false; // Unnecessary Insert: edge does not exist in IR.
    if (Kind == DominatorTree::Delete && HasEdge)
      return false; // Unnecessary Delete: edge still exists in IR.

    // Analyze pending updates to determine if the update is unnecessary.
    DominatorTree::UpdateType Update = {Kind, From, To};
    DominatorTree::UpdateType Invert = {Kind != DominatorTree::Insert
                                            ? DominatorTree::Insert
                                            : DominatorTree::Delete,
                                        From, To};
    for (auto I = PendUpdates.begin(), E = PendUpdates.end(); I != E; ++I) {
      if (Update == *I)
        return false; // Discard duplicate updates.
      if (Invert == *I) {
        // Update and Invert are both valid (equivalent to a no-op). Remove
        // Invert from PendUpdates and discard the Update.
        PendUpdates.erase(I);
        return false;
      }
    }
    PendUpdates.push_back(Update); // Save the valid update.
    return true;
  }

  /// Performs all pending basic block deletions. We have to defer the deletion
  /// of these blocks until after the DominatorTree updates are applied. The
  /// internal workings of the DominatorTree code expect every update's From
  /// and To blocks to exist and to be a member of the same Function.
  bool flushDelBB() {
    if (DeletedBBs.empty())
      return false;
    for (auto *BB : DeletedBBs)
      BB->eraseFromParent();
    DeletedBBs.clear();
    return true;
  }
};

} // end namespace llvm

#endif // LLVM_IR_DEFERREDDOMINANCE_H
