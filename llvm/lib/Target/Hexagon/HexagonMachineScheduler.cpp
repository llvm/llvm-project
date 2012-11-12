//===- HexagonMachineScheduler.cpp - MI Scheduler for Hexagon -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// MachineScheduler schedules machine instructions after phi elimination. It
// preserves LiveIntervals so it can be invoked before register allocation.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "misched"

#include "HexagonMachineScheduler.h"

#include <queue>

using namespace llvm;

/// Platform specific modifications to DAG.
void VLIWMachineScheduler::postprocessDAG() {
  SUnit* LastSequentialCall = NULL;
  // Currently we only catch the situation when compare gets scheduled
  // before preceding call.
  for (unsigned su = 0, e = SUnits.size(); su != e; ++su) {
    // Remember the call.
    if (SUnits[su].getInstr()->isCall())
      LastSequentialCall = &(SUnits[su]);
    // Look for a compare that defines a predicate.
    else if (SUnits[su].getInstr()->isCompare() && LastSequentialCall)
      SUnits[su].addPred(SDep(LastSequentialCall, SDep::Barrier));
  }
}

/// Check if scheduling of this SU is possible
/// in the current packet.
/// It is _not_ precise (statefull), it is more like
/// another heuristic. Many corner cases are figured
/// empirically.
bool VLIWResourceModel::isResourceAvailable(SUnit *SU) {
  if (!SU || !SU->getInstr())
    return false;

  // First see if the pipeline could receive this instruction
  // in the current cycle.
  switch (SU->getInstr()->getOpcode()) {
  default:
    if (!ResourcesModel->canReserveResources(SU->getInstr()))
      return false;
  case TargetOpcode::EXTRACT_SUBREG:
  case TargetOpcode::INSERT_SUBREG:
  case TargetOpcode::SUBREG_TO_REG:
  case TargetOpcode::REG_SEQUENCE:
  case TargetOpcode::IMPLICIT_DEF:
  case TargetOpcode::COPY:
  case TargetOpcode::INLINEASM:
    break;
  }

  // Now see if there are no other dependencies to instructions already
  // in the packet.
  for (unsigned i = 0, e = Packet.size(); i != e; ++i) {
    if (Packet[i]->Succs.size() == 0)
      continue;
    for (SUnit::const_succ_iterator I = Packet[i]->Succs.begin(),
         E = Packet[i]->Succs.end(); I != E; ++I) {
      // Since we do not add pseudos to packets, might as well
      // ignore order dependencies.
      if (I->isCtrl())
        continue;

      if (I->getSUnit() == SU)
        return false;
    }
  }
  return true;
}

/// Keep track of available resources.
bool VLIWResourceModel::reserveResources(SUnit *SU) {
  bool startNewCycle = false;
  // Artificially reset state.
  if (!SU) {
    ResourcesModel->clearResources();
    Packet.clear();
    TotalPackets++;
    return false;
  }
  // If this SU does not fit in the packet
  // start a new one.
  if (!isResourceAvailable(SU)) {
    ResourcesModel->clearResources();
    Packet.clear();
    TotalPackets++;
    startNewCycle = true;
  }

  switch (SU->getInstr()->getOpcode()) {
  default:
    ResourcesModel->reserveResources(SU->getInstr());
    break;
  case TargetOpcode::EXTRACT_SUBREG:
  case TargetOpcode::INSERT_SUBREG:
  case TargetOpcode::SUBREG_TO_REG:
  case TargetOpcode::REG_SEQUENCE:
  case TargetOpcode::IMPLICIT_DEF:
  case TargetOpcode::KILL:
  case TargetOpcode::PROLOG_LABEL:
  case TargetOpcode::EH_LABEL:
  case TargetOpcode::COPY:
  case TargetOpcode::INLINEASM:
    break;
  }
  Packet.push_back(SU);

#ifndef NDEBUG
  DEBUG(dbgs() << "Packet[" << TotalPackets << "]:\n");
  for (unsigned i = 0, e = Packet.size(); i != e; ++i) {
    DEBUG(dbgs() << "\t[" << i << "] SU(");
    DEBUG(dbgs() << Packet[i]->NodeNum << ")\t");
    DEBUG(Packet[i]->getInstr()->dump());
  }
#endif

  // If packet is now full, reset the state so in the next cycle
  // we start fresh.
  if (Packet.size() >= SchedModel->getIssueWidth()) {
    ResourcesModel->clearResources();
    Packet.clear();
    TotalPackets++;
    startNewCycle = true;
  }

  return startNewCycle;
}

/// schedule - Called back from MachineScheduler::runOnMachineFunction
/// after setting up the current scheduling region. [RegionBegin, RegionEnd)
/// only includes instructions that have DAG nodes, not scheduling boundaries.
void VLIWMachineScheduler::schedule() {
  DEBUG(dbgs()
        << "********** MI Converging Scheduling VLIW BB#" << BB->getNumber()
        << " " << BB->getName()
        << " in_func " << BB->getParent()->getFunction()->getName()
        << " at loop depth "  << MLI.getLoopDepth(BB)
        << " \n");

  buildDAGWithRegPressure();

  // Postprocess the DAG to add platform specific artificial dependencies.
  postprocessDAG();

  // To view Height/Depth correctly, they should be accessed at least once.
  DEBUG(unsigned maxH = 0;
        for (unsigned su = 0, e = SUnits.size(); su != e; ++su)
          if (SUnits[su].getHeight() > maxH)
            maxH = SUnits[su].getHeight();
        dbgs() << "Max Height " << maxH << "\n";);
  DEBUG(unsigned maxD = 0;
        for (unsigned su = 0, e = SUnits.size(); su != e; ++su)
          if (SUnits[su].getDepth() > maxD)
            maxD = SUnits[su].getDepth();
        dbgs() << "Max Depth " << maxD << "\n";);
  DEBUG(for (unsigned su = 0, e = SUnits.size(); su != e; ++su)
          SUnits[su].dumpAll(this));

  initQueues();

  bool IsTopNode = false;
  while (SUnit *SU = SchedImpl->pickNode(IsTopNode)) {
    if (!checkSchedLimit())
      break;

    scheduleMI(SU, IsTopNode);

    updateQueues(SU, IsTopNode);
  }
  assert(CurrentTop == CurrentBottom && "Nonempty unscheduled zone.");

  placeDebugValues();
}

void ConvergingVLIWScheduler::initialize(ScheduleDAGMI *dag) {
  DAG = static_cast<VLIWMachineScheduler*>(dag);
  SchedModel = DAG->getSchedModel();
  TRI = DAG->TRI;
  Top.init(DAG, SchedModel);
  Bot.init(DAG, SchedModel);

  // Initialize the HazardRecognizers. If itineraries don't exist, are empty, or
  // are disabled, then these HazardRecs will be disabled.
  const InstrItineraryData *Itin = DAG->getSchedModel()->getInstrItineraries();
  const TargetMachine &TM = DAG->MF.getTarget();
  Top.HazardRec = TM.getInstrInfo()->CreateTargetMIHazardRecognizer(Itin, DAG);
  Bot.HazardRec = TM.getInstrInfo()->CreateTargetMIHazardRecognizer(Itin, DAG);

  Top.ResourceModel = new VLIWResourceModel(TM, DAG->getSchedModel());
  Bot.ResourceModel = new VLIWResourceModel(TM, DAG->getSchedModel());

  assert((!llvm::ForceTopDown || !llvm::ForceBottomUp) &&
         "-misched-topdown incompatible with -misched-bottomup");
}

void ConvergingVLIWScheduler::releaseTopNode(SUnit *SU) {
  if (SU->isScheduled)
    return;

  for (SUnit::succ_iterator I = SU->Preds.begin(), E = SU->Preds.end();
       I != E; ++I) {
    unsigned PredReadyCycle = I->getSUnit()->TopReadyCycle;
    unsigned MinLatency = I->getMinLatency();
#ifndef NDEBUG
    Top.MaxMinLatency = std::max(MinLatency, Top.MaxMinLatency);
#endif
    if (SU->TopReadyCycle < PredReadyCycle + MinLatency)
      SU->TopReadyCycle = PredReadyCycle + MinLatency;
  }
  Top.releaseNode(SU, SU->TopReadyCycle);
}

void ConvergingVLIWScheduler::releaseBottomNode(SUnit *SU) {
  if (SU->isScheduled)
    return;

  assert(SU->getInstr() && "Scheduled SUnit must have instr");

  for (SUnit::succ_iterator I = SU->Succs.begin(), E = SU->Succs.end();
       I != E; ++I) {
    unsigned SuccReadyCycle = I->getSUnit()->BotReadyCycle;
    unsigned MinLatency = I->getMinLatency();
#ifndef NDEBUG
    Bot.MaxMinLatency = std::max(MinLatency, Bot.MaxMinLatency);
#endif
    if (SU->BotReadyCycle < SuccReadyCycle + MinLatency)
      SU->BotReadyCycle = SuccReadyCycle + MinLatency;
  }
  Bot.releaseNode(SU, SU->BotReadyCycle);
}

/// Does this SU have a hazard within the current instruction group.
///
/// The scheduler supports two modes of hazard recognition. The first is the
/// ScheduleHazardRecognizer API. It is a fully general hazard recognizer that
/// supports highly complicated in-order reservation tables
/// (ScoreboardHazardRecognizer) and arbitrary target-specific logic.
///
/// The second is a streamlined mechanism that checks for hazards based on
/// simple counters that the scheduler itself maintains. It explicitly checks
/// for instruction dispatch limitations, including the number of micro-ops that
/// can dispatch per cycle.
///
/// TODO: Also check whether the SU must start a new group.
bool ConvergingVLIWScheduler::SchedBoundary::checkHazard(SUnit *SU) {
  if (HazardRec->isEnabled())
    return HazardRec->getHazardType(SU) != ScheduleHazardRecognizer::NoHazard;

  unsigned uops = SchedModel->getNumMicroOps(SU->getInstr());
  if (IssueCount + uops > SchedModel->getIssueWidth())
    return true;

  return false;
}

void ConvergingVLIWScheduler::SchedBoundary::releaseNode(SUnit *SU,
                                                     unsigned ReadyCycle) {
  if (ReadyCycle < MinReadyCycle)
    MinReadyCycle = ReadyCycle;

  // Check for interlocks first. For the purpose of other heuristics, an
  // instruction that cannot issue appears as if it's not in the ReadyQueue.
  if (ReadyCycle > CurrCycle || checkHazard(SU))

    Pending.push(SU);
  else
    Available.push(SU);
}

/// Move the boundary of scheduled code by one cycle.
void ConvergingVLIWScheduler::SchedBoundary::bumpCycle() {
  unsigned Width = SchedModel->getIssueWidth();
  IssueCount = (IssueCount <= Width) ? 0 : IssueCount - Width;

  assert(MinReadyCycle < UINT_MAX && "MinReadyCycle uninitialized");
  unsigned NextCycle = std::max(CurrCycle + 1, MinReadyCycle);

  if (!HazardRec->isEnabled()) {
    // Bypass HazardRec virtual calls.
    CurrCycle = NextCycle;
  } else {
    // Bypass getHazardType calls in case of long latency.
    for (; CurrCycle != NextCycle; ++CurrCycle) {
      if (isTop())
        HazardRec->AdvanceCycle();
      else
        HazardRec->RecedeCycle();
    }
  }
  CheckPending = true;

  DEBUG(dbgs() << "*** " << Available.getName() << " cycle "
        << CurrCycle << '\n');
}

/// Move the boundary of scheduled code by one SUnit.
void ConvergingVLIWScheduler::SchedBoundary::bumpNode(SUnit *SU) {
  bool startNewCycle = false;

  // Update the reservation table.
  if (HazardRec->isEnabled()) {
    if (!isTop() && SU->isCall) {
      // Calls are scheduled with their preceding instructions. For bottom-up
      // scheduling, clear the pipeline state before emitting.
      HazardRec->Reset();
    }
    HazardRec->EmitInstruction(SU);
  }

  // Update DFA model.
  startNewCycle = ResourceModel->reserveResources(SU);

  // Check the instruction group dispatch limit.
  // TODO: Check if this SU must end a dispatch group.
  IssueCount += SchedModel->getNumMicroOps(SU->getInstr());
  if (startNewCycle) {
    DEBUG(dbgs() << "*** Max instrs at cycle " << CurrCycle << '\n');
    bumpCycle();
  }
  else
    DEBUG(dbgs() << "*** IssueCount " << IssueCount
          << " at cycle " << CurrCycle << '\n');
}

/// Release pending ready nodes in to the available queue. This makes them
/// visible to heuristics.
void ConvergingVLIWScheduler::SchedBoundary::releasePending() {
  // If the available queue is empty, it is safe to reset MinReadyCycle.
  if (Available.empty())
    MinReadyCycle = UINT_MAX;

  // Check to see if any of the pending instructions are ready to issue.  If
  // so, add them to the available queue.
  for (unsigned i = 0, e = Pending.size(); i != e; ++i) {
    SUnit *SU = *(Pending.begin()+i);
    unsigned ReadyCycle = isTop() ? SU->TopReadyCycle : SU->BotReadyCycle;

    if (ReadyCycle < MinReadyCycle)
      MinReadyCycle = ReadyCycle;

    if (ReadyCycle > CurrCycle)
      continue;

    if (checkHazard(SU))
      continue;

    Available.push(SU);
    Pending.remove(Pending.begin()+i);
    --i; --e;
  }
  CheckPending = false;
}

/// Remove SU from the ready set for this boundary.
void ConvergingVLIWScheduler::SchedBoundary::removeReady(SUnit *SU) {
  if (Available.isInQueue(SU))
    Available.remove(Available.find(SU));
  else {
    assert(Pending.isInQueue(SU) && "bad ready count");
    Pending.remove(Pending.find(SU));
  }
}

/// If this queue only has one ready candidate, return it. As a side effect,
/// advance the cycle until at least one node is ready. If multiple instructions
/// are ready, return NULL.
SUnit *ConvergingVLIWScheduler::SchedBoundary::pickOnlyChoice() {
  if (CheckPending)
    releasePending();

  for (unsigned i = 0; Available.empty(); ++i) {
    assert(i <= (HazardRec->getMaxLookAhead() + MaxMinLatency) &&
           "permanent hazard"); (void)i;
    ResourceModel->reserveResources(0);
    bumpCycle();
    releasePending();
  }
  if (Available.size() == 1)
    return *Available.begin();
  return NULL;
}

#ifndef NDEBUG
void ConvergingVLIWScheduler::traceCandidate(const char *Label,
                                             const ReadyQueue &Q,
                                             SUnit *SU, PressureElement P) {
  dbgs() << Label << " " << Q.getName() << " ";
  if (P.isValid())
    dbgs() << TRI->getRegPressureSetName(P.PSetID) << ":" << P.UnitIncrease
           << " ";
  else
    dbgs() << "     ";
  SU->dump(DAG);
}
#endif

/// getSingleUnscheduledPred - If there is exactly one unscheduled predecessor
/// of SU, return it, otherwise return null.
static SUnit *getSingleUnscheduledPred(SUnit *SU) {
  SUnit *OnlyAvailablePred = 0;
  for (SUnit::const_pred_iterator I = SU->Preds.begin(), E = SU->Preds.end();
       I != E; ++I) {
    SUnit &Pred = *I->getSUnit();
    if (!Pred.isScheduled) {
      // We found an available, but not scheduled, predecessor.  If it's the
      // only one we have found, keep track of it... otherwise give up.
      if (OnlyAvailablePred && OnlyAvailablePred != &Pred)
        return 0;
      OnlyAvailablePred = &Pred;
    }
  }
  return OnlyAvailablePred;
}

/// getSingleUnscheduledSucc - If there is exactly one unscheduled successor
/// of SU, return it, otherwise return null.
static SUnit *getSingleUnscheduledSucc(SUnit *SU) {
  SUnit *OnlyAvailableSucc = 0;
  for (SUnit::const_succ_iterator I = SU->Succs.begin(), E = SU->Succs.end();
       I != E; ++I) {
    SUnit &Succ = *I->getSUnit();
    if (!Succ.isScheduled) {
      // We found an available, but not scheduled, successor.  If it's the
      // only one we have found, keep track of it... otherwise give up.
      if (OnlyAvailableSucc && OnlyAvailableSucc != &Succ)
        return 0;
      OnlyAvailableSucc = &Succ;
    }
  }
  return OnlyAvailableSucc;
}

// Constants used to denote relative importance of
// heuristic components for cost computation.
static const unsigned PriorityOne = 200;
static const unsigned PriorityTwo = 100;
static const unsigned PriorityThree = 50;
static const unsigned PriorityFour = 20;
static const unsigned ScaleTwo = 10;
static const unsigned FactorOne = 2;

/// Single point to compute overall scheduling cost.
/// TODO: More heuristics will be used soon.
int ConvergingVLIWScheduler::SchedulingCost(ReadyQueue &Q, SUnit *SU,
                                            SchedCandidate &Candidate,
                                            RegPressureDelta &Delta,
                                            bool verbose) {
  // Initial trivial priority.
  int ResCount = 1;

  // Do not waste time on a node that is already scheduled.
  if (!SU || SU->isScheduled)
    return ResCount;

  // Forced priority is high.
  if (SU->isScheduleHigh)
    ResCount += PriorityOne;

  // Critical path first.
  if (Q.getID() == TopQID) {
    ResCount += (SU->getHeight() * ScaleTwo);

    // If resources are available for it, multiply the
    // chance of scheduling.
    if (Top.ResourceModel->isResourceAvailable(SU))
      ResCount <<= FactorOne;
  } else {
    ResCount += (SU->getDepth() * ScaleTwo);

    // If resources are available for it, multiply the
    // chance of scheduling.
    if (Bot.ResourceModel->isResourceAvailable(SU))
      ResCount <<= FactorOne;
  }

  unsigned NumNodesBlocking = 0;
  if (Q.getID() == TopQID) {
    // How many SUs does it block from scheduling?
    // Look at all of the successors of this node.
    // Count the number of nodes that
    // this node is the sole unscheduled node for.
    for (SUnit::const_succ_iterator I = SU->Succs.begin(), E = SU->Succs.end();
         I != E; ++I)
      if (getSingleUnscheduledPred(I->getSUnit()) == SU)
        ++NumNodesBlocking;
  } else {
    // How many unscheduled predecessors block this node?
    for (SUnit::const_pred_iterator I = SU->Preds.begin(), E = SU->Preds.end();
         I != E; ++I)
      if (getSingleUnscheduledSucc(I->getSUnit()) == SU)
        ++NumNodesBlocking;
  }
  ResCount += (NumNodesBlocking * ScaleTwo);

  // Factor in reg pressure as a heuristic.
  ResCount -= (Delta.Excess.UnitIncrease*PriorityThree);
  ResCount -= (Delta.CriticalMax.UnitIncrease*PriorityThree);

  DEBUG(if (verbose) dbgs() << " Total(" << ResCount << ")");

  return ResCount;
}

/// Pick the best candidate from the top queue.
///
/// TODO: getMaxPressureDelta results can be mostly cached for each SUnit during
/// DAG building. To adjust for the current scheduling location we need to
/// maintain the number of vreg uses remaining to be top-scheduled.
ConvergingVLIWScheduler::CandResult ConvergingVLIWScheduler::
pickNodeFromQueue(ReadyQueue &Q, const RegPressureTracker &RPTracker,
                  SchedCandidate &Candidate) {
  DEBUG(Q.dump());

  // getMaxPressureDelta temporarily modifies the tracker.
  RegPressureTracker &TempTracker = const_cast<RegPressureTracker&>(RPTracker);

  // BestSU remains NULL if no top candidates beat the best existing candidate.
  CandResult FoundCandidate = NoCand;
  for (ReadyQueue::iterator I = Q.begin(), E = Q.end(); I != E; ++I) {
    RegPressureDelta RPDelta;
    TempTracker.getMaxPressureDelta((*I)->getInstr(), RPDelta,
                                    DAG->getRegionCriticalPSets(),
                                    DAG->getRegPressure().MaxSetPressure);

    int CurrentCost = SchedulingCost(Q, *I, Candidate, RPDelta, false);

    // Initialize the candidate if needed.
    if (!Candidate.SU) {
      Candidate.SU = *I;
      Candidate.RPDelta = RPDelta;
      Candidate.SCost = CurrentCost;
      FoundCandidate = NodeOrder;
      continue;
    }

    // Best cost.
    if (CurrentCost > Candidate.SCost) {
      DEBUG(traceCandidate("CCAND", Q, *I));
      Candidate.SU = *I;
      Candidate.RPDelta = RPDelta;
      Candidate.SCost = CurrentCost;
      FoundCandidate = BestCost;
      continue;
    }

    // Fall through to original instruction order.
    // Only consider node order if Candidate was chosen from this Q.
    if (FoundCandidate == NoCand)
      continue;
  }
  return FoundCandidate;
}

/// Pick the best candidate node from either the top or bottom queue.
SUnit *ConvergingVLIWScheduler::pickNodeBidrectional(bool &IsTopNode) {
  // Schedule as far as possible in the direction of no choice. This is most
  // efficient, but also provides the best heuristics for CriticalPSets.
  if (SUnit *SU = Bot.pickOnlyChoice()) {
    IsTopNode = false;
    return SU;
  }
  if (SUnit *SU = Top.pickOnlyChoice()) {
    IsTopNode = true;
    return SU;
  }
  SchedCandidate BotCand;
  // Prefer bottom scheduling when heuristics are silent.
  CandResult BotResult = pickNodeFromQueue(Bot.Available,
                                           DAG->getBotRPTracker(), BotCand);
  assert(BotResult != NoCand && "failed to find the first candidate");

  // If either Q has a single candidate that provides the least increase in
  // Excess pressure, we can immediately schedule from that Q.
  //
  // RegionCriticalPSets summarizes the pressure within the scheduled region and
  // affects picking from either Q. If scheduling in one direction must
  // increase pressure for one of the excess PSets, then schedule in that
  // direction first to provide more freedom in the other direction.
  if (BotResult == SingleExcess || BotResult == SingleCritical) {
    IsTopNode = false;
    return BotCand.SU;
  }
  // Check if the top Q has a better candidate.
  SchedCandidate TopCand;
  CandResult TopResult = pickNodeFromQueue(Top.Available,
                                           DAG->getTopRPTracker(), TopCand);
  assert(TopResult != NoCand && "failed to find the first candidate");

  if (TopResult == SingleExcess || TopResult == SingleCritical) {
    IsTopNode = true;
    return TopCand.SU;
  }
  // If either Q has a single candidate that minimizes pressure above the
  // original region's pressure pick it.
  if (BotResult == SingleMax) {
    IsTopNode = false;
    return BotCand.SU;
  }
  if (TopResult == SingleMax) {
    IsTopNode = true;
    return TopCand.SU;
  }
  if (TopCand.SCost > BotCand.SCost) {
    IsTopNode = true;
    return TopCand.SU;
  }
  // Otherwise prefer the bottom candidate in node order.
  IsTopNode = false;
  return BotCand.SU;
}

/// Pick the best node to balance the schedule. Implements MachineSchedStrategy.
SUnit *ConvergingVLIWScheduler::pickNode(bool &IsTopNode) {
  if (DAG->top() == DAG->bottom()) {
    assert(Top.Available.empty() && Top.Pending.empty() &&
           Bot.Available.empty() && Bot.Pending.empty() && "ReadyQ garbage");
    return NULL;
  }
  SUnit *SU;
  if (llvm::ForceTopDown) {
    SU = Top.pickOnlyChoice();
    if (!SU) {
      SchedCandidate TopCand;
      CandResult TopResult =
        pickNodeFromQueue(Top.Available, DAG->getTopRPTracker(), TopCand);
      assert(TopResult != NoCand && "failed to find the first candidate");
      (void)TopResult;
      SU = TopCand.SU;
    }
    IsTopNode = true;
  } else if (llvm::ForceBottomUp) {
    SU = Bot.pickOnlyChoice();
    if (!SU) {
      SchedCandidate BotCand;
      CandResult BotResult =
        pickNodeFromQueue(Bot.Available, DAG->getBotRPTracker(), BotCand);
      assert(BotResult != NoCand && "failed to find the first candidate");
      (void)BotResult;
      SU = BotCand.SU;
    }
    IsTopNode = false;
  } else {
    SU = pickNodeBidrectional(IsTopNode);
  }
  if (SU->isTopReady())
    Top.removeReady(SU);
  if (SU->isBottomReady())
    Bot.removeReady(SU);

  DEBUG(dbgs() << "*** " << (IsTopNode ? "Top" : "Bottom")
        << " Scheduling Instruction in cycle "
        << (IsTopNode ? Top.CurrCycle : Bot.CurrCycle) << '\n';
        SU->dump(DAG));
  return SU;
}

/// Update the scheduler's state after scheduling a node. This is the same node
/// that was just returned by pickNode(). However, VLIWMachineScheduler needs
/// to update it's state based on the current cycle before MachineSchedStrategy
/// does.
void ConvergingVLIWScheduler::schedNode(SUnit *SU, bool IsTopNode) {
  if (IsTopNode) {
    SU->TopReadyCycle = Top.CurrCycle;
    Top.bumpNode(SU);
  } else {
    SU->BotReadyCycle = Bot.CurrCycle;
    Bot.bumpNode(SU);
  }
}

