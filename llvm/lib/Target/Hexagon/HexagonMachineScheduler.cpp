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

#include "HexagonMachineScheduler.h"
#include "HexagonInstrInfo.h"
#include "HexagonSubtarget.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/DFAPacketizer.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/RegisterPressure.h"
#include "llvm/CodeGen/ScheduleDAG.h"
#include "llvm/CodeGen/ScheduleHazardRecognizer.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetOpcodes.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/TargetSchedule.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>

using namespace llvm;

#define DEBUG_TYPE "machine-scheduler"

static cl::opt<bool> IgnoreBBRegPressure("ignore-bb-reg-pressure",
    cl::Hidden, cl::ZeroOrMore, cl::init(false));

static cl::opt<unsigned> SchedDebugVerboseLevel("misched-verbose-level",
    cl::Hidden, cl::ZeroOrMore, cl::init(1));

static cl::opt<bool> TopUseShorterTie("top-use-shorter-tie",
    cl::Hidden, cl::ZeroOrMore, cl::init(false));

static cl::opt<bool> BotUseShorterTie("bot-use-shorter-tie",
    cl::Hidden, cl::ZeroOrMore, cl::init(false));

static cl::opt<bool> DisableTCTie("disable-tc-tie",
    cl::Hidden, cl::ZeroOrMore, cl::init(false));

// Check if the scheduler should penalize instructions that are available to
// early due to a zero-latency dependence.
static cl::opt<bool> CheckEarlyAvail("check-early-avail", cl::Hidden,
    cl::ZeroOrMore, cl::init(true));

/// Save the last formed packet
void VLIWResourceModel::savePacket() {
  OldPacket = Packet;
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
    if (!ResourcesModel->canReserveResources(*SU->getInstr()))
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

  MachineFunction &MF = *SU->getInstr()->getParent()->getParent();
  auto &QII = *MF.getSubtarget<HexagonSubtarget>().getInstrInfo();

  // Now see if there are no other dependencies to instructions already
  // in the packet.
  for (unsigned i = 0, e = Packet.size(); i != e; ++i) {
    if (Packet[i]->Succs.size() == 0)
      continue;

    // Enable .cur formation.
    if (QII.mayBeCurLoad(*Packet[i]->getInstr()))
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
    savePacket();
    Packet.clear();
    TotalPackets++;
    return false;
  }
  // If this SU does not fit in the packet
  // start a new one.
  if (!isResourceAvailable(SU)) {
    ResourcesModel->clearResources();
    savePacket();
    Packet.clear();
    TotalPackets++;
    startNewCycle = true;
  }

  switch (SU->getInstr()->getOpcode()) {
  default:
    ResourcesModel->reserveResources(*SU->getInstr());
    break;
  case TargetOpcode::EXTRACT_SUBREG:
  case TargetOpcode::INSERT_SUBREG:
  case TargetOpcode::SUBREG_TO_REG:
  case TargetOpcode::REG_SEQUENCE:
  case TargetOpcode::IMPLICIT_DEF:
  case TargetOpcode::KILL:
  case TargetOpcode::CFI_INSTRUCTION:
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
    savePacket();
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
  DEBUG(dbgs() << "********** MI Converging Scheduling VLIW "
               << printMBBReference(*BB) << " " << BB->getName() << " in_func "
               << BB->getParent()->getName() << " at loop depth "
               << MLI->getLoopDepth(BB) << " \n");

  buildDAGWithRegPressure();

  SmallVector<SUnit*, 8> TopRoots, BotRoots;
  findRootsAndBiasEdges(TopRoots, BotRoots);

  // Initialize the strategy before modifying the DAG.
  SchedImpl->initialize(this);

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

  initQueues(TopRoots, BotRoots);

  bool IsTopNode = false;
  while (true) {
    DEBUG(dbgs() << "** VLIWMachineScheduler::schedule picking next node\n");
    SUnit *SU = SchedImpl->pickNode(IsTopNode);
    if (!SU) break;

    if (!checkSchedLimit())
      break;

    scheduleMI(SU, IsTopNode);

    updateQueues(SU, IsTopNode);

    // Notify the scheduling strategy after updating the DAG.
    SchedImpl->schedNode(SU, IsTopNode);
  }
  assert(CurrentTop == CurrentBottom && "Nonempty unscheduled zone.");

  placeDebugValues();

  DEBUG({
    dbgs() << "*** Final schedule for "
           << printMBBReference(*begin()->getParent()) << " ***\n";
    dumpSchedule();
    dbgs() << '\n';
  });
}

void ConvergingVLIWScheduler::initialize(ScheduleDAGMI *dag) {
  DAG = static_cast<VLIWMachineScheduler*>(dag);
  SchedModel = DAG->getSchedModel();

  Top.init(DAG, SchedModel);
  Bot.init(DAG, SchedModel);

  // Initialize the HazardRecognizers. If itineraries don't exist, are empty, or
  // are disabled, then these HazardRecs will be disabled.
  const InstrItineraryData *Itin = DAG->getSchedModel()->getInstrItineraries();
  const TargetSubtargetInfo &STI = DAG->MF.getSubtarget();
  const TargetInstrInfo *TII = STI.getInstrInfo();
  delete Top.HazardRec;
  delete Bot.HazardRec;
  Top.HazardRec = TII->CreateTargetMIHazardRecognizer(Itin, DAG);
  Bot.HazardRec = TII->CreateTargetMIHazardRecognizer(Itin, DAG);

  delete Top.ResourceModel;
  delete Bot.ResourceModel;
  Top.ResourceModel = new VLIWResourceModel(STI, DAG->getSchedModel());
  Bot.ResourceModel = new VLIWResourceModel(STI, DAG->getSchedModel());

  assert((!ForceTopDown || !ForceBottomUp) &&
         "-misched-topdown incompatible with -misched-bottomup");
}

void ConvergingVLIWScheduler::releaseTopNode(SUnit *SU) {
  if (SU->isScheduled)
    return;

  for (const SDep &PI : SU->Preds) {
    unsigned PredReadyCycle = PI.getSUnit()->TopReadyCycle;
    unsigned MinLatency = PI.getLatency();
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
    unsigned MinLatency = I->getLatency();
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
bool ConvergingVLIWScheduler::VLIWSchedBoundary::checkHazard(SUnit *SU) {
  if (HazardRec->isEnabled())
    return HazardRec->getHazardType(SU) != ScheduleHazardRecognizer::NoHazard;

  unsigned uops = SchedModel->getNumMicroOps(SU->getInstr());
  if (IssueCount + uops > SchedModel->getIssueWidth())
    return true;

  return false;
}

void ConvergingVLIWScheduler::VLIWSchedBoundary::releaseNode(SUnit *SU,
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
void ConvergingVLIWScheduler::VLIWSchedBoundary::bumpCycle() {
  unsigned Width = SchedModel->getIssueWidth();
  IssueCount = (IssueCount <= Width) ? 0 : IssueCount - Width;

  assert(MinReadyCycle < std::numeric_limits<unsigned>::max() &&
         "MinReadyCycle uninitialized");
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

  DEBUG(dbgs() << "*** Next cycle " << Available.getName() << " cycle "
               << CurrCycle << '\n');
}

/// Move the boundary of scheduled code by one SUnit.
void ConvergingVLIWScheduler::VLIWSchedBoundary::bumpNode(SUnit *SU) {
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
void ConvergingVLIWScheduler::VLIWSchedBoundary::releasePending() {
  // If the available queue is empty, it is safe to reset MinReadyCycle.
  if (Available.empty())
    MinReadyCycle = std::numeric_limits<unsigned>::max();

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
void ConvergingVLIWScheduler::VLIWSchedBoundary::removeReady(SUnit *SU) {
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
SUnit *ConvergingVLIWScheduler::VLIWSchedBoundary::pickOnlyChoice() {
  if (CheckPending)
    releasePending();

  for (unsigned i = 0; Available.empty(); ++i) {
    assert(i <= (HazardRec->getMaxLookAhead() + MaxMinLatency) &&
           "permanent hazard"); (void)i;
    ResourceModel->reserveResources(nullptr);
    bumpCycle();
    releasePending();
  }
  if (Available.size() == 1)
    return *Available.begin();
  return nullptr;
}

#ifndef NDEBUG
void ConvergingVLIWScheduler::traceCandidate(const char *Label,
      const ReadyQueue &Q, SUnit *SU, int Cost, PressureChange P) {
  dbgs() << Label << " " << Q.getName() << " ";
  if (P.isValid())
    dbgs() << DAG->TRI->getRegPressureSetName(P.getPSet()) << ":"
           << P.getUnitInc() << " ";
  else
    dbgs() << "     ";
  dbgs() << "cost(" << Cost << ")\t";
  SU->dump(DAG);
}

// Very detailed queue dump, to be used with higher verbosity levels.
void ConvergingVLIWScheduler::readyQueueVerboseDump(
      const RegPressureTracker &RPTracker, SchedCandidate &Candidate,
      ReadyQueue &Q) {
  RegPressureTracker &TempTracker = const_cast<RegPressureTracker &>(RPTracker);

  dbgs() << ">>> " << Q.getName() << "\n";
  for (ReadyQueue::iterator I = Q.begin(), E = Q.end(); I != E; ++I) {
    RegPressureDelta RPDelta;
    TempTracker.getMaxPressureDelta((*I)->getInstr(), RPDelta,
                                    DAG->getRegionCriticalPSets(),
                                    DAG->getRegPressure().MaxSetPressure);
    std::stringstream dbgstr;
    dbgstr << "SU(" << std::setw(3) << (*I)->NodeNum << ")";
    dbgs() << dbgstr.str();
    SchedulingCost(Q, *I, Candidate, RPDelta, true);
    dbgs() << "\t";
    (*I)->getInstr()->dump();
  }
  dbgs() << "\n";
}
#endif

/// isSingleUnscheduledPred - If SU2 is the only unscheduled predecessor
/// of SU, return true (we may have duplicates)
static inline bool isSingleUnscheduledPred(SUnit *SU, SUnit *SU2) {
  if (SU->NumPredsLeft == 0)
    return false;

  for (auto &Pred : SU->Preds) {
    // We found an available, but not scheduled, predecessor.
    if (!Pred.getSUnit()->isScheduled && (Pred.getSUnit() != SU2))
      return false;
  }

  return true;
}

/// isSingleUnscheduledSucc - If SU2 is the only unscheduled successor
/// of SU, return true (we may have duplicates)
static inline bool isSingleUnscheduledSucc(SUnit *SU, SUnit *SU2) {
  if (SU->NumSuccsLeft == 0)
    return false;

  for (auto &Succ : SU->Succs) {
    // We found an available, but not scheduled, successor.
    if (!Succ.getSUnit()->isScheduled && (Succ.getSUnit() != SU2))
      return false;
  }
  return true;
}

// Constants used to denote relative importance of
// heuristic components for cost computation.
static const unsigned PriorityOne = 200;
static const unsigned PriorityTwo = 50;
static const unsigned PriorityThree = 75;
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

  MachineInstr &Instr = *SU->getInstr();

  DEBUG(if (verbose) dbgs() << ((Q.getID() == TopQID) ? "(top|" : "(bot|"));
  // Forced priority is high.
  if (SU->isScheduleHigh) {
    ResCount += PriorityOne;
    DEBUG(dbgs() << "H|");
  }

  // Critical path first.
  if (Q.getID() == TopQID) {
    ResCount += (SU->getHeight() * ScaleTwo);

    DEBUG(if (verbose) {
      std::stringstream dbgstr;
      dbgstr << "h" << std::setw(3) << SU->getHeight() << "|";
      dbgs() << dbgstr.str();
    });

    // If resources are available for it, multiply the
    // chance of scheduling.
    if (Top.ResourceModel->isResourceAvailable(SU)) {
      ResCount <<= FactorOne;
      ResCount += PriorityThree;
      DEBUG(if (verbose) dbgs() << "A|");
    } else
      DEBUG(if (verbose) dbgs() << " |");
  } else {
    ResCount += (SU->getDepth() * ScaleTwo);

    DEBUG(if (verbose) {
      std::stringstream dbgstr;
      dbgstr << "d" << std::setw(3) << SU->getDepth() << "|";
      dbgs() << dbgstr.str();
    });

    // If resources are available for it, multiply the
    // chance of scheduling.
    if (Bot.ResourceModel->isResourceAvailable(SU)) {
      ResCount <<= FactorOne;
      ResCount += PriorityThree;
      DEBUG(if (verbose) dbgs() << "A|");
    } else
      DEBUG(if (verbose) dbgs() << " |");
  }

  unsigned NumNodesBlocking = 0;
  if (Q.getID() == TopQID) {
    // How many SUs does it block from scheduling?
    // Look at all of the successors of this node.
    // Count the number of nodes that
    // this node is the sole unscheduled node for.
    for (const SDep &SI : SU->Succs)
      if (isSingleUnscheduledPred(SI.getSUnit(), SU))
        ++NumNodesBlocking;
  } else {
    // How many unscheduled predecessors block this node?
    for (const SDep &PI : SU->Preds)
      if (isSingleUnscheduledSucc(PI.getSUnit(), SU))
        ++NumNodesBlocking;
  }
  ResCount += (NumNodesBlocking * ScaleTwo);

  DEBUG(if (verbose) {
    std::stringstream dbgstr;
    dbgstr << "blk " << std::setw(2) << NumNodesBlocking << ")|";
    dbgs() << dbgstr.str();
  });

  // Factor in reg pressure as a heuristic.
  if (!IgnoreBBRegPressure) {
    // Decrease priority by the amount that register pressure exceeds the limit.
    ResCount -= (Delta.Excess.getUnitInc()*PriorityOne);
    // Decrease priority if register pressure exceeds the limit.
    ResCount -= (Delta.CriticalMax.getUnitInc()*PriorityOne);
    // Decrease priority slightly if register pressure would increase over the
    // current maximum.
    ResCount -= (Delta.CurrentMax.getUnitInc()*PriorityTwo);
    DEBUG(if (verbose) {
        dbgs() << "RP " << Delta.Excess.getUnitInc() << "/"
               << Delta.CriticalMax.getUnitInc() <<"/"
               << Delta.CurrentMax.getUnitInc() << ")|";
    });
  }

  // Give a little extra priority to a .cur instruction if there is a resource
  // available for it.
  auto &QST = DAG->MF.getSubtarget<HexagonSubtarget>();
  auto &QII = *QST.getInstrInfo();
  if (SU->isInstr() && QII.mayBeCurLoad(*SU->getInstr())) {
    if (Q.getID() == TopQID && Top.ResourceModel->isResourceAvailable(SU)) {
      ResCount += PriorityTwo;
      DEBUG(if (verbose) dbgs() << "C|");
    } else if (Q.getID() == BotQID &&
               Bot.ResourceModel->isResourceAvailable(SU)) {
      ResCount += PriorityTwo;
      DEBUG(if (verbose) dbgs() << "C|");
    }
  }

  // Give preference to a zero latency instruction if the dependent
  // instruction is in the current packet.
  if (Q.getID() == TopQID) {
    for (const SDep &PI : SU->Preds) {
      if (!PI.getSUnit()->getInstr()->isPseudo() && PI.isAssignedRegDep() &&
          PI.getLatency() == 0 &&
          Top.ResourceModel->isInPacket(PI.getSUnit())) {
        ResCount += PriorityThree;
        DEBUG(if (verbose) dbgs() << "Z|");
      }
    }
  } else {
    for (const SDep &SI : SU->Succs) {
      if (!SI.getSUnit()->getInstr()->isPseudo() && SI.isAssignedRegDep() &&
          SI.getLatency() == 0 &&
          Bot.ResourceModel->isInPacket(SI.getSUnit())) {
        ResCount += PriorityThree;
        DEBUG(if (verbose) dbgs() << "Z|");
      }
    }
  }

  // Give less preference to an instruction that will cause a stall with
  // an instruction in the previous packet.
  if (QII.isHVXVec(Instr)) {
    // Check for stalls in the previous packet.
    if (Q.getID() == TopQID) {
      for (auto J : Top.ResourceModel->OldPacket)
        if (QII.producesStall(*J->getInstr(), Instr))
          ResCount -= PriorityOne;
    } else {
      for (auto J : Bot.ResourceModel->OldPacket)
        if (QII.producesStall(Instr, *J->getInstr()))
          ResCount -= PriorityOne;
    }
  }

  // If the instruction has a non-zero latency dependence with an instruction in
  // the current packet, then it should not be scheduled yet. The case occurs
  // when the dependent instruction is scheduled in a new packet, so the
  // scheduler updates the current cycle and pending instructions become
  // available.
  if (CheckEarlyAvail) {
    if (Q.getID() == TopQID) {
      for (const auto &PI : SU->Preds) {
        if (PI.getLatency() > 0 &&
            Top.ResourceModel->isInPacket(PI.getSUnit())) {
          ResCount -= PriorityOne;
          DEBUG(if (verbose) dbgs() << "D|");
        }
      }
    } else {
      for (const auto &SI : SU->Succs) {
        if (SI.getLatency() > 0 &&
            Bot.ResourceModel->isInPacket(SI.getSUnit())) {
          ResCount -= PriorityOne;
          DEBUG(if (verbose) dbgs() << "D|");
        }
      }
    }
  }

  DEBUG(if (verbose) {
    std::stringstream dbgstr;
    dbgstr << "Total " << std::setw(4) << ResCount << ")";
    dbgs() << dbgstr.str();
  });

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
  DEBUG(if (SchedDebugVerboseLevel > 1)
        readyQueueVerboseDump(RPTracker, Candidate, Q);
        else Q.dump(););

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
      DEBUG(traceCandidate("DCAND", Q, *I, CurrentCost));
      Candidate.SU = *I;
      Candidate.RPDelta = RPDelta;
      Candidate.SCost = CurrentCost;
      FoundCandidate = NodeOrder;
      continue;
    }

    // Best cost.
    if (CurrentCost > Candidate.SCost) {
      DEBUG(traceCandidate("CCAND", Q, *I, CurrentCost));
      Candidate.SU = *I;
      Candidate.RPDelta = RPDelta;
      Candidate.SCost = CurrentCost;
      FoundCandidate = BestCost;
      continue;
    }

    // Tie breaker using Timing Class.
    if (!DisableTCTie) {
      auto &QST = DAG->MF.getSubtarget<HexagonSubtarget>();
      auto &QII = *QST.getInstrInfo();

      const MachineInstr *MI = (*I)->getInstr();
      const MachineInstr *CandI = Candidate.SU->getInstr();
      const InstrItineraryData *InstrItins = QST.getInstrItineraryData();

      unsigned InstrLatency = QII.getInstrTimingClassLatency(InstrItins, *MI);
      unsigned CandLatency = QII.getInstrTimingClassLatency(InstrItins, *CandI);
      DEBUG(dbgs() << "TC Tie Breaker Cand: "
                   << CandLatency << " Instr:" << InstrLatency << "\n"
                   << *MI << *CandI << "\n");
      if (Q.getID() == TopQID && CurrentCost == Candidate.SCost) {
        if (InstrLatency < CandLatency && TopUseShorterTie) {
          Candidate.SU = *I;
          Candidate.RPDelta = RPDelta;
          Candidate.SCost = CurrentCost;
          FoundCandidate = BestCost;
          DEBUG(dbgs() << "Used top shorter tie breaker\n");
          continue;
        } else if (InstrLatency > CandLatency && !TopUseShorterTie) {
          Candidate.SU = *I;
          Candidate.RPDelta = RPDelta;
          Candidate.SCost = CurrentCost;
          FoundCandidate = BestCost;
          DEBUG(dbgs() << "Used top longer tie breaker\n");
          continue;
        }
      } else if (Q.getID() == BotQID && CurrentCost == Candidate.SCost) {
        if (InstrLatency < CandLatency && BotUseShorterTie) {
          Candidate.SU = *I;
          Candidate.RPDelta = RPDelta;
          Candidate.SCost = CurrentCost;
          FoundCandidate = BestCost;
          DEBUG(dbgs() << "Used Bot shorter tie breaker\n");
          continue;
        } else if (InstrLatency > CandLatency && !BotUseShorterTie) {
          Candidate.SU = *I;
          Candidate.RPDelta = RPDelta;
          Candidate.SCost = CurrentCost;
          FoundCandidate = BestCost;
          DEBUG(dbgs() << "Used Bot longer tie breaker\n");
          continue;
        }
      }
    }

    if (CurrentCost == Candidate.SCost) {
      if ((Q.getID() == TopQID &&
           (*I)->Succs.size() > Candidate.SU->Succs.size()) ||
          (Q.getID() == BotQID &&
           (*I)->Preds.size() < Candidate.SU->Preds.size())) {
        DEBUG(traceCandidate("SPCAND", Q, *I, CurrentCost));
        Candidate.SU = *I;
        Candidate.RPDelta = RPDelta;
        Candidate.SCost = CurrentCost;
        FoundCandidate = BestCost;
        continue;
      }
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
    DEBUG(dbgs() << "Picked only Bottom\n");
    IsTopNode = false;
    return SU;
  }
  if (SUnit *SU = Top.pickOnlyChoice()) {
    DEBUG(dbgs() << "Picked only Top\n");
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
    DEBUG(dbgs() << "Prefered Bottom Node\n");
    IsTopNode = false;
    return BotCand.SU;
  }
  // Check if the top Q has a better candidate.
  SchedCandidate TopCand;
  CandResult TopResult = pickNodeFromQueue(Top.Available,
                                           DAG->getTopRPTracker(), TopCand);
  assert(TopResult != NoCand && "failed to find the first candidate");

  if (TopResult == SingleExcess || TopResult == SingleCritical) {
    DEBUG(dbgs() << "Prefered Top Node\n");
    IsTopNode = true;
    return TopCand.SU;
  }
  // If either Q has a single candidate that minimizes pressure above the
  // original region's pressure pick it.
  if (BotResult == SingleMax) {
    DEBUG(dbgs() << "Prefered Bottom Node SingleMax\n");
    IsTopNode = false;
    return BotCand.SU;
  }
  if (TopResult == SingleMax) {
    DEBUG(dbgs() << "Prefered Top Node SingleMax\n");
    IsTopNode = true;
    return TopCand.SU;
  }
  if (TopCand.SCost > BotCand.SCost) {
    DEBUG(dbgs() << "Prefered Top Node Cost\n");
    IsTopNode = true;
    return TopCand.SU;
  }
  // Otherwise prefer the bottom candidate in node order.
  DEBUG(dbgs() << "Prefered Bottom in Node order\n");
  IsTopNode = false;
  return BotCand.SU;
}

/// Pick the best node to balance the schedule. Implements MachineSchedStrategy.
SUnit *ConvergingVLIWScheduler::pickNode(bool &IsTopNode) {
  if (DAG->top() == DAG->bottom()) {
    assert(Top.Available.empty() && Top.Pending.empty() &&
           Bot.Available.empty() && Bot.Pending.empty() && "ReadyQ garbage");
    return nullptr;
  }
  SUnit *SU;
  if (ForceTopDown) {
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
  } else if (ForceBottomUp) {
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
