//===- DomTreeUpdater.cpp - DomTree/Post DomTree Updater --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the DomTreeUpdater class, which provides a uniform way
// to update dominator tree related data structures.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/DomTreeUpdater.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/GenericDomTree.h"
#include <algorithm>
#include <functional>
#include <utility>

namespace llvm {

bool DomTreeUpdater::isUpdateValid(
    const DominatorTree::UpdateType Update) const {
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
  if (Kind == DominatorTree::Insert && !HasEdge)
    return false;

  // Edge exists in IR.
  if (Kind == DominatorTree::Delete && HasEdge)
    return false;

  return true;
}

bool DomTreeUpdater::isSelfDominance(
    const DominatorTree::UpdateType Update) const {
  // Won't affect DomTree and PostDomTree.
  return Update.getFrom() == Update.getTo();
}

void DomTreeUpdater::applyDomTreeUpdates() {
  // No pending DomTreeUpdates.
  if (Strategy != UpdateStrategy::Lazy || !DT)
    return;

  // Only apply updates not are applied by DomTree.
  if (hasPendingDomTreeUpdates()) {
    const auto I = PendUpdates.begin() + PendDTUpdateIndex;
    const auto E = PendUpdates.end();
    assert(I < E && "Iterator range invalid; there should be DomTree updates.");
    DT->applyUpdates(ArrayRef<DominatorTree::UpdateType>(I, E));
    PendDTUpdateIndex = PendUpdates.size();
  }
}

void DomTreeUpdater::flush() {
  applyDomTreeUpdates();
  applyPostDomTreeUpdates();
  dropOutOfDateUpdates();
}

void DomTreeUpdater::applyPostDomTreeUpdates() {
  // No pending PostDomTreeUpdates.
  if (Strategy != UpdateStrategy::Lazy || !PDT)
    return;

  // Only apply updates not are applied by PostDomTree.
  if (hasPendingPostDomTreeUpdates()) {
    const auto I = PendUpdates.begin() + PendPDTUpdateIndex;
    const auto E = PendUpdates.end();
    assert(I < E &&
           "Iterator range invalid; there should be PostDomTree updates.");
    PDT->applyUpdates(ArrayRef<DominatorTree::UpdateType>(I, E));
    PendPDTUpdateIndex = PendUpdates.size();
  }
}

void DomTreeUpdater::tryFlushDeletedBB() {
  if (!hasPendingUpdates())
    forceFlushDeletedBB();
}

bool DomTreeUpdater::forceFlushDeletedBB() {
  if (DeletedBBs.empty())
    return false;

  for (auto *BB : DeletedBBs) {
    // After calling deleteBB or callbackDeleteBB under Lazy UpdateStrategy,
    // validateDeleteBB() removes all instructions of DelBB and adds an
    // UnreachableInst as its terminator. So we check whether the BasicBlock to
    // delete only has an UnreachableInst inside.
    assert(BB->getInstList().size() == 1 &&
           isa<UnreachableInst>(BB->getTerminator()) &&
           "DelBB has been modified while awaiting deletion.");
    BB->removeFromParent();
    eraseDelBBNode(BB);
    delete BB;
  }
  DeletedBBs.clear();
  Callbacks.clear();
  return true;
}

void DomTreeUpdater::recalculate(Function &F) {

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
  forceFlushDeletedBB();
  if (DT)
    DT->recalculate(F);
  if (PDT)
    PDT->recalculate(F);

  // Resume forceFlushDeletedBB() to erase DomTree or PostDomTree nodes.
  IsRecalculatingDomTree = IsRecalculatingPostDomTree = false;
  PendDTUpdateIndex = PendPDTUpdateIndex = PendUpdates.size();
  dropOutOfDateUpdates();
}

bool DomTreeUpdater::hasPendingUpdates() const {
  return hasPendingDomTreeUpdates() || hasPendingPostDomTreeUpdates();
}

bool DomTreeUpdater::hasPendingDomTreeUpdates() const {
  if (!DT)
    return false;
  return PendUpdates.size() != PendDTUpdateIndex;
}

bool DomTreeUpdater::hasPendingPostDomTreeUpdates() const {
  if (!PDT)
    return false;
  return PendUpdates.size() != PendPDTUpdateIndex;
}

bool DomTreeUpdater::isBBPendingDeletion(llvm::BasicBlock *DelBB) const {
  if (Strategy == UpdateStrategy::Eager || DeletedBBs.empty())
    return false;
  return DeletedBBs.contains(DelBB);
}

// The DT and PDT require the nodes related to updates
// are not deleted when update functions are called.
// So BasicBlock deletions must be pended when the
// UpdateStrategy is Lazy. When the UpdateStrategy is
// Eager, the BasicBlock will be deleted immediately.
void DomTreeUpdater::deleteBB(BasicBlock *DelBB) {
  validateDeleteBB(DelBB);
  if (Strategy == UpdateStrategy::Lazy) {
    DeletedBBs.insert(DelBB);
    return;
  }

  DelBB->removeFromParent();
  eraseDelBBNode(DelBB);
  delete DelBB;
}

void DomTreeUpdater::callbackDeleteBB(
    BasicBlock *DelBB, std::function<void(BasicBlock *)> Callback) {
  validateDeleteBB(DelBB);
  if (Strategy == UpdateStrategy::Lazy) {
    Callbacks.push_back(CallBackOnDeletion(DelBB, Callback));
    DeletedBBs.insert(DelBB);
    return;
  }

  DelBB->removeFromParent();
  eraseDelBBNode(DelBB);
  Callback(DelBB);
  delete DelBB;
}

void DomTreeUpdater::eraseDelBBNode(BasicBlock *DelBB) {
  if (DT && !IsRecalculatingDomTree)
    if (DT->getNode(DelBB))
      DT->eraseNode(DelBB);

  if (PDT && !IsRecalculatingPostDomTree)
    if (PDT->getNode(DelBB))
      PDT->eraseNode(DelBB);
}

void DomTreeUpdater::validateDeleteBB(BasicBlock *DelBB) {
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
  // Make sure DelBB has a valid terminator instruction. As long as DelBB is a
  // Child of Function F it must contain valid IR.
  new UnreachableInst(DelBB->getContext(), DelBB);
}

void DomTreeUpdater::applyUpdates(ArrayRef<DominatorTree::UpdateType> Updates) {
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

void DomTreeUpdater::applyUpdatesPermissive(
    ArrayRef<DominatorTree::UpdateType> Updates) {
  if (!DT && !PDT)
    return;

  SmallSet<std::pair<BasicBlock *, BasicBlock *>, 8> Seen;
  SmallVector<DominatorTree::UpdateType, 8> DeduplicatedUpdates;
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

DominatorTree &DomTreeUpdater::getDomTree() {
  assert(DT && "Invalid acquisition of a null DomTree");
  applyDomTreeUpdates();
  dropOutOfDateUpdates();
  return *DT;
}

PostDominatorTree &DomTreeUpdater::getPostDomTree() {
  assert(PDT && "Invalid acquisition of a null PostDomTree");
  applyPostDomTreeUpdates();
  dropOutOfDateUpdates();
  return *PDT;
}

void DomTreeUpdater::insertEdge(BasicBlock *From, BasicBlock *To) {

#ifndef NDEBUG
  assert(isUpdateValid({DominatorTree::Insert, From, To}) &&
         "Inserted edge does not appear in the CFG");
#endif

  if (!DT && !PDT)
    return;

  // Won't affect DomTree and PostDomTree; discard update.
  if (From == To)
    return;

  if (Strategy == UpdateStrategy::Eager) {
    if (DT)
      DT->insertEdge(From, To);
    if (PDT)
      PDT->insertEdge(From, To);
    return;
  }

  PendUpdates.push_back({DominatorTree::Insert, From, To});
}

void DomTreeUpdater::insertEdgeRelaxed(BasicBlock *From, BasicBlock *To) {
  if (From == To)
    return;

  if (!DT && !PDT)
    return;

  if (!isUpdateValid({DominatorTree::Insert, From, To}))
    return;

  if (Strategy == UpdateStrategy::Eager) {
    if (DT)
      DT->insertEdge(From, To);
    if (PDT)
      PDT->insertEdge(From, To);
    return;
  }

  PendUpdates.push_back({DominatorTree::Insert, From, To});
}

void DomTreeUpdater::deleteEdge(BasicBlock *From, BasicBlock *To) {

#ifndef NDEBUG
  assert(isUpdateValid({DominatorTree::Delete, From, To}) &&
         "Deleted edge still exists in the CFG!");
#endif

  if (!DT && !PDT)
    return;

  // Won't affect DomTree and PostDomTree; discard update.
  if (From == To)
    return;

  if (Strategy == UpdateStrategy::Eager) {
    if (DT)
      DT->deleteEdge(From, To);
    if (PDT)
      PDT->deleteEdge(From, To);
    return;
  }

  PendUpdates.push_back({DominatorTree::Delete, From, To});
}

void DomTreeUpdater::deleteEdgeRelaxed(BasicBlock *From, BasicBlock *To) {
  if (From == To)
    return;

  if (!DT && !PDT)
    return;

  if (!isUpdateValid({DominatorTree::Delete, From, To}))
    return;

  if (Strategy == UpdateStrategy::Eager) {
    if (DT)
      DT->deleteEdge(From, To);
    if (PDT)
      PDT->deleteEdge(From, To);
    return;
  }

  PendUpdates.push_back({DominatorTree::Delete, From, To});
}

void DomTreeUpdater::dropOutOfDateUpdates() {
  if (Strategy == DomTreeUpdater::UpdateStrategy::Eager)
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

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD void DomTreeUpdater::dump() const {
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
      [&](ArrayRef<DominatorTree::UpdateType>::const_iterator begin,
          ArrayRef<DominatorTree::UpdateType>::const_iterator end) {
        if (begin == end)
          OS << "  None\n";
        Index = 0;
        for (auto It = begin, ItEnd = end; It != ItEnd; ++It) {
          auto U = *It;
          OS << "  " << Index << " : ";
          ++Index;
          if (U.getKind() == DominatorTree::Insert)
            OS << "Insert, ";
          else
            OS << "Delete, ";
          BasicBlock *From = U.getFrom();
          if (From) {
            auto S = From->getName();
            if (!From->hasName())
              S = "(no name)";
            OS << S << "(" << From << "), ";
          } else {
            OS << "(badref), ";
          }
          BasicBlock *To = U.getTo();
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

  OS << "Pending Callbacks:\n";
  Index = 0;
  for (const auto &BB : Callbacks) {
    OS << "  " << Index << " : ";
    ++Index;
    if (BB->hasName())
      OS << BB->getName() << "(";
    else
      OS << "(no_name)(";
    OS << BB << ")\n";
  }
}
#endif
} // namespace llvm
