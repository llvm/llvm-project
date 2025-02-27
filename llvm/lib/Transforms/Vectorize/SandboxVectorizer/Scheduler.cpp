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
    for (auto *DepN : N->preds(DAG)) {
      DepN->decrUnscheduledSuccs();
      if (DepN->ready() && !DepN->scheduled())
        ReadyList.insert(DepN);
    }
    N->setScheduled(true);
  }
}

void Scheduler::notifyCreateInstr(Instruction *I) {
  // The DAG notifier should have run by now.
  auto *N = DAG.getNode(I);
  // If there is no DAG node for `I` it means that this is out of scope for the
  // DAG and as such out of scope for the scheduler too, so nothing to do.
  if (N == nullptr)
    return;
  // If the instruction is inserted below the top-of-schedule then we mark it as
  // "scheduled".
  bool IsScheduled = ScheduleTopItOpt &&
                     *ScheduleTopItOpt != I->getParent()->end() &&
                     (*ScheduleTopItOpt.value()).comesBefore(I);
  if (IsScheduled)
    N->setScheduled(true);
  // If the new instruction is above the top of schedule we need to remove its
  // dependency predecessors from the ready list and increment their
  // `UnscheduledSuccs` counters.
  if (!IsScheduled) {
    for (auto *PredN : N->preds(DAG)) {
      ReadyList.remove(PredN);
      PredN->incrUnscheduledSuccs();
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
  auto *N0 = DAG.getNode(Instrs[0]);
  auto *SB0 = N0 != nullptr ? N0->getSchedBundle() : nullptr;
  bool AllUnscheduled = SB0 == nullptr;
  bool FullyScheduled = SB0 != nullptr && !SB0->isSingleton();
  for (auto *I : drop_begin(Instrs)) {
    auto *N = DAG.getNode(I);
    auto *SB = N != nullptr ? N->getSchedBundle() : nullptr;
    if (SB != nullptr) {
      // We found a scheduled instr, so there is now way all are unscheduled.
      AllUnscheduled = false;
      if (SB->isSingleton()) {
        // We found an instruction in a temporarily scheduled singleton. There
        // is no way that all instructions are scheduled in the same bundle.
        FullyScheduled = false;
      }
    }

    if (SB != SB0) {
      // Either one of SB, SB0 is null, or they are in different bundles, so
      // Instrs are definitely not in the same vector bundle.
      FullyScheduled = false;
      // One of SB, SB0 are in a vector bundle and they differ.
      if ((SB != nullptr && !SB->isSingleton()) ||
          (SB0 != nullptr && !SB0->isSingleton()))
        return BndlSchedState::AlreadyScheduled;
    }
  }
  return AllUnscheduled   ? BndlSchedState::NoneScheduled
         : FullyScheduled ? BndlSchedState::FullyScheduled
                          : BndlSchedState::TemporarilyScheduled;
}

void Scheduler::trimSchedule(ArrayRef<Instruction *> Instrs) {
  //                                |  Legend: N: DGNode
  //  N <- DAGInterval.top()        |          B: SchedBundle
  //  N                             |          *: Contains instruction in Instrs
  //  B <- TopI (Top of schedule)   +-------------------------------------------
  //  B
  //  B *
  //  B
  //  B * <- LowestI (Lowest in Instrs)
  //  B
  //  N
  //  N
  //  N <- DAGInterval.bottom()
  //
  Instruction *TopI = &*ScheduleTopItOpt.value();
  Instruction *LowestI = VecUtils::getLowest(Instrs);
  // Destroy the singleton schedule bundles from LowestI all the way to the top.
  for (auto *I = LowestI, *E = TopI->getPrevNode(); I != E;
       I = I->getPrevNode()) {
    auto *N = DAG.getNode(I);
    if (N == nullptr)
      continue;
    auto *SB = N->getSchedBundle();
    if (SB->isSingleton())
      eraseBundle(SB);
  }
  // The DAG Nodes contain state like the number of UnscheduledSuccs and the
  // Scheduled flag. We need to reset their state. We need to do this for all
  // nodes from LowestI to the top of the schedule. DAG Nodes that are above the
  // top of schedule that depend on nodes that got reset need to have their
  // UnscheduledSuccs adjusted.
  Interval<Instruction> ResetIntvl(TopI, LowestI);
  for (Instruction &I : ResetIntvl) {
    auto *N = DAG.getNode(&I);
    N->resetScheduleState();
    // Recompute UnscheduledSuccs for nodes not only in ResetIntvl but even for
    // nodes above the top of schedule.
    for (auto *PredN : N->preds(DAG))
      PredN->incrUnscheduledSuccs();
  }
  // Refill the ready list by visiting all nodes from the top of DAG to LowestI.
  ReadyList.clear();
  Interval<Instruction> RefillIntvl(DAG.getInterval().top(), LowestI);
  for (Instruction &I : RefillIntvl) {
    auto *N = DAG.getNode(&I);
    if (N->ready())
      ReadyList.insert(N);
  }
}

bool Scheduler::trySchedule(ArrayRef<Instruction *> Instrs) {
  assert(all_of(drop_begin(Instrs),
                [Instrs](Instruction *I) {
                  return I->getParent() == (*Instrs.begin())->getParent();
                }) &&
         "Instrs not in the same BB, should have been rejected by Legality!");
  // TODO: For now don't cross BBs.
  if (!DAG.getInterval().empty()) {
    auto *BB = DAG.getInterval().top()->getParent();
    if (any_of(Instrs, [BB](auto *I) { return I->getParent() != BB; }))
      return false;
  }
  if (ScheduledBB == nullptr)
    ScheduledBB = Instrs[0]->getParent();
  // We don't support crossing BBs for now.
  if (any_of(Instrs,
             [this](Instruction *I) { return I->getParent() != ScheduledBB; }))
    return false;
  auto SchedState = getBndlSchedState(Instrs);
  switch (SchedState) {
  case BndlSchedState::FullyScheduled:
    // Nothing to do.
    return true;
  case BndlSchedState::AlreadyScheduled:
    // Instructions are part of a different vector schedule, so we can't
    // schedule \p Instrs in the same bundle (without destroying the existing
    // schedule).
    return false;
  case BndlSchedState::TemporarilyScheduled:
    // If one or more instrs are already scheduled we need to destroy the
    // top-most part of the schedule that includes the instrs in the bundle and
    // re-schedule.
    trimSchedule(Instrs);
    ScheduleTopItOpt = std::next(VecUtils::getLowest(Instrs)->getIterator());
    return tryScheduleUntil(Instrs);
  case BndlSchedState::NoneScheduled: {
    // TODO: Set the window of the DAG that we are interested in.
    if (!ScheduleTopItOpt)
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
  OS << "Top of schedule: ";
  if (ScheduleTopItOpt)
    OS << **ScheduleTopItOpt;
  else
    OS << "Empty";
  OS << "\n";
}
void Scheduler::dump() const { dump(dbgs()); }
#endif // NDEBUG

} // namespace llvm::sandboxir
