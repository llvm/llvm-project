//===- Scheduler.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVectorizer/Scheduler.h"
#include "llvm/Transforms/Vectorize/SandboxVectorizer/VecUtils.h"

namespace llvm::sandboxir {

// TODO: Check if we can cache top/bottom to reduce compile-time.
DGNode *SchedBundle::getTop() const {
  DGNode *TopN = Nodes.front();
  for (auto *N : drop_begin(Nodes)) {
    if (N->getInstruction()->comesBefore(TopN->getInstruction()))
      TopN = N;
  }
  return TopN;
}

DGNode *SchedBundle::getBot() const {
  DGNode *BotN = Nodes.front();
  for (auto *N : drop_begin(Nodes)) {
    if (BotN->getInstruction()->comesBefore(N->getInstruction()))
      BotN = N;
  }
  return BotN;
}

void SchedBundle::cluster(BasicBlock::iterator Where) {
  for (auto *N : Nodes) {
    auto *I = N->getInstruction();
    if (I->getIterator() == Where)
      ++Where; // Try to maintain bundle order.
    I->moveBefore(*Where.getNodeParent(), Where);
  }
}

#ifndef NDEBUG
void SchedBundle::dump(raw_ostream &OS) const {
  for (auto *N : Nodes)
    OS << *N;
}

void SchedBundle::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG

#ifndef NDEBUG
void ReadyListContainer::dump(raw_ostream &OS) const {
  auto ListCopy = List;
  while (!ListCopy.empty()) {
    OS << *ListCopy.top() << "\n";
    ListCopy.pop();
  }
}

void ReadyListContainer::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG

void Scheduler::scheduleAndUpdateReadyList(SchedBundle &Bndl) {
  // Find where we should schedule the instructions.
  assert(ScheduleTopItOpt && "Should have been set by now!");
  auto Where = *ScheduleTopItOpt;
  // Move all instructions in `Bndl` to `Where`.
  Bndl.cluster(Where);
  // Update the last scheduled bundle.
  ScheduleTopItOpt = Bndl.getTop()->getInstruction()->getIterator();
  // Set nodes as "scheduled" and decrement the UnsceduledSuccs counter of all
  // dependency predecessors.
  for (DGNode *N : Bndl) {
    N->setScheduled(true);
    for (auto *DepN : N->preds(DAG)) {
      // TODO: preds() should not return nullptr.
      if (DepN == nullptr)
        continue;
      DepN->decrUnscheduledSuccs();
      if (DepN->ready())
        ReadyList.insert(DepN);
    }
  }
}

SchedBundle *Scheduler::createBundle(ArrayRef<Instruction *> Instrs) {
  SchedBundle::ContainerTy Nodes;
  Nodes.reserve(Instrs.size());
  for (auto *I : Instrs)
    Nodes.push_back(DAG.getNode(I));
  auto BndlPtr = std::make_unique<SchedBundle>(std::move(Nodes));
  auto *Bndl = BndlPtr.get();
  Bndls[Bndl] = std::move(BndlPtr);
  return Bndl;
}

void Scheduler::eraseBundle(SchedBundle *SB) { Bndls.erase(SB); }

bool Scheduler::tryScheduleUntil(ArrayRef<Instruction *> Instrs) {
  // Use a set of instructions, instead of `Instrs` for fast lookups.
  DenseSet<Instruction *> InstrsToDefer(Instrs.begin(), Instrs.end());
  // This collects the nodes that correspond to instructions found in `Instrs`
  // that have just become ready. These nodes won't be scheduled right away.
  SmallVector<DGNode *, 8> DeferredNodes;

  // Keep scheduling ready nodes until we either run out of ready nodes (i.e.,
  // ReadyList is empty), or all nodes that correspond to `Instrs` (the nodes of
  // which are collected in DeferredNodes) are all ready to schedule.
  while (!ReadyList.empty()) {
    auto *ReadyN = ReadyList.pop();
    if (InstrsToDefer.contains(ReadyN->getInstruction())) {
      // If the ready instruction is one of those in `Instrs`, then we don't
      // schedule it right away. Instead we defer it until we can schedule it
      // along with the rest of the instructions in `Instrs`, at the same
      // time in a single scheduling bundle.
      DeferredNodes.push_back(ReadyN);
      bool ReadyToScheduleDeferred = DeferredNodes.size() == Instrs.size();
      if (ReadyToScheduleDeferred) {
        scheduleAndUpdateReadyList(*createBundle(Instrs));
        return true;
      }
    } else {
      // If the ready instruction is not found in `Instrs`, then we wrap it in a
      // scheduling bundle and schedule it right away.
      scheduleAndUpdateReadyList(*createBundle({ReadyN->getInstruction()}));
    }
  }
  assert(DeferredNodes.size() != Instrs.size() &&
         "We should have succesfully scheduled and early-returned!");
  return false;
}

Scheduler::BndlSchedState
Scheduler::getBndlSchedState(ArrayRef<Instruction *> Instrs) const {
  assert(!Instrs.empty() && "Expected non-empty bundle");
  bool PartiallyScheduled = false;
  bool FullyScheduled = true;
  for (auto *I : Instrs) {
    auto *N = DAG.getNode(I);
    if (N != nullptr && N->scheduled())
      PartiallyScheduled = true;
    else
      FullyScheduled = false;
  }
  if (FullyScheduled) {
    // If not all instrs in the bundle are in the same SchedBundle then this
    // should be considered as partially-scheduled, because we will need to
    // re-schedule.
    SchedBundle *SB = DAG.getNode(Instrs[0])->getSchedBundle();
    assert(SB != nullptr && "FullyScheduled assumes that there is an SB!");
    if (any_of(drop_begin(Instrs), [this, SB](sandboxir::Value *SBV) {
          return DAG.getNode(cast<sandboxir::Instruction>(SBV))
                     ->getSchedBundle() != SB;
        }))
      FullyScheduled = false;
  }
  return FullyScheduled       ? BndlSchedState::FullyScheduled
         : PartiallyScheduled ? BndlSchedState::PartiallyOrDifferentlyScheduled
                              : BndlSchedState::NoneScheduled;
}

void Scheduler::trimSchedule(ArrayRef<Instruction *> Instrs) {
  Instruction *TopI = &*ScheduleTopItOpt.value();
  Instruction *LowestI = VecUtils::getLowest(Instrs);
  // Destroy the schedule bundles from LowestI all the way to the top.
  for (auto *I = LowestI, *E = TopI->getPrevNode(); I != E;
       I = I->getPrevNode()) {
    auto *N = DAG.getNode(I);
    if (auto *SB = N->getSchedBundle())
      eraseBundle(SB);
  }
  // TODO: For now we clear the DAG. Trim view once it gets implemented.
  Bndls.clear();
  DAG.clear();

  // Since we are scheduling NewRegion from scratch, we clear the ready lists.
  // The nodes currently in the list may not be ready after clearing the View.
  ReadyList.clear();
}

bool Scheduler::trySchedule(ArrayRef<Instruction *> Instrs) {
  assert(all_of(drop_begin(Instrs),
                [Instrs](Instruction *I) {
                  return I->getParent() == (*Instrs.begin())->getParent();
                }) &&
         "Instrs not in the same BB!");
  auto SchedState = getBndlSchedState(Instrs);
  switch (SchedState) {
  case BndlSchedState::FullyScheduled:
    // Nothing to do.
    return true;
  case BndlSchedState::PartiallyOrDifferentlyScheduled:
    // If one or more instrs are already scheduled we need to destroy the
    // top-most part of the schedule that includes the instrs in the bundle and
    // re-schedule.
    trimSchedule(Instrs);
    [[fallthrough]];
  case BndlSchedState::NoneScheduled: {
    // TODO: Set the window of the DAG that we are interested in.
    // We start scheduling at the bottom instr of Instrs.
    ScheduleTopItOpt = std::next(VecUtils::getLowest(Instrs)->getIterator());

    // Extend the DAG to include Instrs.
    Interval<Instruction> Extension = DAG.extend(Instrs);
    // Add nodes to ready list.
    for (auto &I : Extension) {
      auto *N = DAG.getNode(&I);
      if (N->ready())
        ReadyList.insert(N);
    }
    // Try schedule all nodes until we can schedule Instrs back-to-back.
    return tryScheduleUntil(Instrs);
  }
  }
  llvm_unreachable("Unhandled BndlSchedState enum");
}

#ifndef NDEBUG
void Scheduler::dump(raw_ostream &OS) const {
  OS << "ReadyList:\n";
  ReadyList.dump(OS);
}
void Scheduler::dump() const { dump(dbgs()); }
#endif // NDEBUG

} // namespace llvm::sandboxir
