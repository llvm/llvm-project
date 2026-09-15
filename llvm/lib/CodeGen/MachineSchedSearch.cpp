//===- MachineSchedSearch.cpp - Complete schedule search ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineSchedSearch.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include <algorithm>
#include <functional>
#include <queue>
#include <vector>

using namespace llvm;

MachineSchedCompleteScheduleOptimizer::
    ~MachineSchedCompleteScheduleOptimizer() = default;

MachineSchedSearchRegion::MachineSchedSearchRegion(ArrayRef<SUnit> Nodes)
    : Nodes(Nodes), Predecessors(Nodes.size()), Successors(Nodes.size()) {
  DenseMap<const SUnit *, unsigned> Ordinals;
  for (auto [Ordinal, SU] : enumerate(Nodes))
    Ordinals[&SU] = Ordinal;

  for (auto [Ordinal, SU] : enumerate(Nodes)) {
    for (const SDep &Pred : SU.Preds) {
      const SUnit *PredSU = Pred.getSUnit();
      if (Pred.isWeak() || PredSU->isBoundaryNode())
        continue;
      auto PredOrdinal = Ordinals.find(PredSU);
      assert(PredOrdinal != Ordinals.end() &&
             "predecessor must belong to the scheduling region");
      if (PredOrdinal == Ordinals.end())
        continue;
      Predecessors[Ordinal].push_back(PredOrdinal->second);
      Successors[PredOrdinal->second].push_back(Ordinal);
    }
  }
}

MachineSchedSearchRegion::MachineSchedSearchRegion(ScheduleDAGMI &DAG)
    : MachineSchedSearchRegion(DAG.SUnits) {
  this->DAG = &DAG;
}

const SUnit &MachineSchedSearchRegion::getSUnit(unsigned Node) const {
  assert(Node < size() && "invalid scheduling node ordinal");
  return Nodes[Node];
}

SmallVector<unsigned, 0> MachineSchedSearchRegion::getInitialOrder() const {
  SmallVector<unsigned, 0> Order = to_vector<0>(seq<unsigned>(size()));
  return Order;
}

SmallVector<unsigned, 0> MachineSchedSearchRegion::getTopologicalOrder() const {
  SmallVector<unsigned, 0> RemainingPredecessors;
  RemainingPredecessors.reserve(size());
  for (unsigned Node = 0; Node != size(); ++Node)
    RemainingPredecessors.push_back(predecessors(Node).size());

  std::priority_queue<unsigned, std::vector<unsigned>, std::greater<unsigned>>
      Ready;
  for (unsigned Node = 0; Node != size(); ++Node)
    if (RemainingPredecessors[Node] == 0)
      Ready.push(Node);

  SmallVector<unsigned, 0> Order;
  Order.reserve(size());
  while (!Ready.empty()) {
    unsigned Node = Ready.top();
    Ready.pop();
    Order.push_back(Node);
    for (unsigned Succ : successors(Node)) {
      assert(RemainingPredecessors[Succ] != 0 &&
             "inconsistent machine scheduling DAG");
      --RemainingPredecessors[Succ];
      if (RemainingPredecessors[Succ] == 0)
        Ready.push(Succ);
    }
  }
  assert(Order.size() == size() &&
         "machine scheduling DAG contains a strong dependency cycle");
  return Order;
}

bool MachineSchedSearchRegion::isLegalOrder(ArrayRef<unsigned> Order) const {
  if (Order.size() != size())
    return false;

  SmallVector<unsigned, 0> Position(size(), size());
  for (auto [Pos, Node] : enumerate(Order)) {
    if (Node >= size() || Position[Node] != size())
      return false;
    Position[Node] = Pos;
  }

  for (unsigned Node = 0; Node != size(); ++Node)
    for (unsigned Pred : predecessors(Node))
      if (Position[Pred] >= Position[Node])
        return false;
  return true;
}

bool MachineSchedSearchRegion::getLegalMoveRange(ArrayRef<unsigned> Order,
                                                 unsigned Node,
                                                 MoveRange &Range) const {
  if (Node >= size() || !isLegalOrder(Order))
    return false;

  SmallVector<unsigned, 0> Position(size());
  for (auto [Pos, Ordinal] : enumerate(Order))
    Position[Ordinal] = Pos;

  Range.Begin = 0;
  Range.End = size() - 1;
  for (unsigned Pred : predecessors(Node))
    Range.Begin = std::max(Range.Begin, Position[Pred] + 1);
  for (unsigned Succ : successors(Node))
    Range.End = std::min(Range.End, Position[Succ] - 1);
  return true;
}

MachineSchedCompleteScheduleReplayer::MachineSchedCompleteScheduleReplayer(
    std::unique_ptr<MachineSchedCompleteScheduleOptimizer> Optimizer)
    : Optimizer(std::move(Optimizer)) {
  assert(this->Optimizer && "complete-schedule optimizer must be provided");
}

MachineSchedCompleteScheduleReplayer::~MachineSchedCompleteScheduleReplayer() =
    default;

void MachineSchedCompleteScheduleReplayer::initialize(ScheduleDAGMI *DAG) {
  MachineSchedSearchRegion Region(*DAG);
  SmallVector<unsigned, 0> Founder = Region.getInitialOrder();
  if (!Region.isLegalOrder(Founder))
    Founder = Region.getTopologicalOrder();

  SmallVector<unsigned, 0> Order;
  UsedOptimizedSchedule =
      Optimizer->optimizeCompleteSchedule(Region, Founder, Order) &&
      Region.isLegalOrder(Order) &&
      Optimizer->validateCompleteSchedule(Region, Order);
  if (!UsedOptimizedSchedule) {
    Order = std::move(Founder);
  }

  CompleteSchedule.clear();
  CompleteSchedule.reserve(Order.size());
  for (unsigned Node : Order)
    CompleteSchedule.push_back(&DAG->SUnits[Node]);
  NextNodeToReplay = 0;
}

SUnit *MachineSchedCompleteScheduleReplayer::pickNode(bool &IsTopNode) {
  if (NextNodeToReplay == CompleteSchedule.size())
    return nullptr;
  SUnit *SU = CompleteSchedule[NextNodeToReplay++];
  assert(SU->isTopReady() &&
         "validated complete schedule contains an unavailable node");
  IsTopNode = true;
  return SU;
}

void ScheduleDAGMI::setPostScheduleOptimizer(
    std::unique_ptr<MachineSchedCompleteScheduleOptimizer> Optimizer) {
  PostSchedOptimizer = std::move(Optimizer);
}

void ScheduleDAGMI::runPostScheduleOptimizer() {
  if (!PostSchedOptimizer || SUnits.empty())
    return;

  // A debug scheduling cutoff makes CurrentTop meet CurrentBottom without
  // necessarily scheduling the whole region. There is no complete founder to
  // optimize in that case.
  if (!llvm::all_of(SUnits, [](const SUnit &SU) { return SU.isScheduled; }))
    return;

  MachineSchedSearchRegion Region(*this);
  DenseMap<const MachineInstr *, unsigned> Ordinals;
  for (auto [Ordinal, SU] : enumerate(SUnits))
    Ordinals[SU.getInstr()] = Ordinal;

  SmallVector<unsigned, 0> Founder;
  Founder.reserve(SUnits.size());
  for (auto I = RegionBegin; I != RegionEnd; ++I) {
    auto Ordinal = Ordinals.find(&*I);
    if (Ordinal != Ordinals.end())
      Founder.push_back(Ordinal->second);
  }
  if (!Region.isLegalOrder(Founder))
    return;

  SmallVector<unsigned, 0> Result;
  if (!PostSchedOptimizer->optimizeCompleteSchedule(Region, Founder, Result) ||
      !Region.isLegalOrder(Result) ||
      !PostSchedOptimizer->validateCompleteSchedule(Region, Result) ||
      Result == Founder)
    return;

  applyCompleteSchedule(Result);
}

void ScheduleDAGMI::applyCompleteSchedule(ArrayRef<unsigned> Order) {
  MachineBasicBlock::iterator InsertPos = RegionBegin;
  for (unsigned Node : Order) {
    MachineInstr *MI = SUnits[Node].getInstr();
    while (InsertPos != RegionEnd && InsertPos->isDebugInstr())
      ++InsertPos;
    if (InsertPos != RegionEnd && &*InsertPos == MI) {
      ++InsertPos;
      continue;
    }
    moveInstruction(MI, InsertPos);
    InsertPos = std::next(MI->getIterator());
  }
}

void ScheduleDAGMILive::applyCompleteSchedule(ArrayRef<unsigned> Order) {
  ScheduleDAGMI::applyCompleteSchedule(Order);
  if (!ShouldTrackPressure)
    return;

  for (unsigned Node : Order) {
    MachineInstr *MI = SUnits[Node].getInstr();
    // A prior ordering may have introduced read-undef flags that are no longer
    // valid. Recompute them together with the other liveness flags below.
    for (MachineOperand &Op : MI->all_defs())
      Op.setIsUndef(false);

    RegisterOperands RegOpers;
    RegOpers.collect(*MI, *TRI, MRI, ShouldTrackLaneMasks,
                     /*IgnoreDead=*/false);
    if (ShouldTrackLaneMasks)
      RegOpers.adjustLaneLiveness(*LIS, MRI, *MI);
    else
      RegOpers.detectDeadDefs(*MI, *LIS);
  }
}
