//-- SystemZMachineScheduler.cpp - SystemZ Scheduler Interface -*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SystemZMachineScheduler.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include <cmath>

using namespace llvm;

#define DEBUG_TYPE "machine-scheduler"

/// Pre-RA scheduling ///

// Options needed for testing.
// TopRegionSUs is the number of SUs that are considered to be part of the
// "top" of a region. Liveness reduction is not done in regions smaller than
// this. The idea is to prioritize latency more after branches and help
// liveness only when the decoder is ahead of execution anyway.
static cl::opt<unsigned> TopRegionSUs("top-region", cl::Hidden, cl::init(36));
static cl::opt<bool> DisableLatency("disable-latency", cl::Hidden,
                                    cl::init(false));

static bool isRegDef(const MachineOperand &MO) {
  return MO.isReg() && MO.isDef();
}

bool SystemZPreRASchedStrategy::definesCmp0Src(const MachineInstr *MI,
                                               bool CCDef) const {
  if (Cmp0SrcReg != SystemZ::NoRegister && MI->getNumOperands() &&
      (MI->getDesc().hasImplicitDefOfPhysReg(SystemZ::CC) || !CCDef)) {
    const MachineOperand &MO0 = MI->getOperand(0);
    if (isRegDef(MO0) && MO0.getReg() == Cmp0SrcReg)
      return true;
  }
  return false;
}

bool SystemZPreRASchedStrategy::closesLiveRange(const SUnit *SU,
                                                ScheduleDAGMILive *DAG) const {
  if (SU->getInstr()->isCopy())
    return false;

  // Extract the PressureChanges that all fp/vector or GR64/GR32/GRH32 regs
  // affect respectively. misched-prera-pdiffs.mir tests against any future
  // change in the PressureSets modelling, so simply hard-code them here.
  int VR16PChange = 0, GRX32PChange = 0;
  const PressureDiff &PDiff = DAG->getPressureDiff(SU);
  for (const PressureChange &PC : PDiff) {
    if (!PC.isValid())
      break;
    if (PC.getPSet() == SystemZ::VR16Bit)
      VR16PChange = PC.getUnitInc();
    else if (PC.getPSet() == SystemZ::GRX32Bit)
      GRX32PChange = PC.getUnitInc();
  }

  // Return true for a (vreg) def when no uses become live. Prioritize
  // FP/vector regs over GPRs.
  const MachineOperand &MO0 = SU->getInstr()->getOperand(0);
  if (isRegDef(MO0)) {
    const TargetRegisterClass *RC = DAG->MRI.getRegClass(MO0.getReg());
    int RegWeight = TRI->getRegClassWeight(RC).RegWeight;
    bool VR16DefNoKill = VR16PChange == -RegWeight;
    bool GRX32DefNoKill = GRX32PChange == -RegWeight;
    return VR16DefNoKill || (!VR16PChange && GRX32DefNoKill);
  }
  return false;
}

bool SystemZPreRASchedStrategy::tryCandidate(SchedCandidate &Cand,
                                             SchedCandidate &TryCand,
                                             SchedBoundary *Zone) const {
  assert(Zone && !Zone->isTop() && "Bottom-Up scheduling only.");

  // Initialize the candidate if needed.
  if (!Cand.isValid()) {
    TryCand.Reason = FirstValid;
    return true;
  }

  // Bias physreg defs and copies to their uses and definitions respectively.
  if (tryBiasPhysRegs(TryCand, Cand, Zone, /*BiasPRegsExtra=*/true))
    return TryCand.Reason != NoCand;

  if (RegionPolicy.ShouldTrackPressure) {
    auto schedLow = [&](const SUnit *SU) {
      return SU->getHeight() <= Zone->getScheduledLatency() &&
             SU->getHeight() < LivenessHeightCutOff && closesLiveRange(SU, DAG);
    };
    // One SU closes a live range while preserving the scheduled latency.
    if (tryGreater(schedLow(TryCand.SU), schedLow(Cand.SU), TryCand, Cand,
                   RegExcess))
      return TryCand.Reason != NoCand;
  }

  if (!RegionPolicy.DisableLatencyHeuristic)
    if (const SUnit *HigherSU =
            TryCand.SU->getHeight() > Cand.SU->getHeight()   ? TryCand.SU
            : TryCand.SU->getHeight() < Cand.SU->getHeight() ? Cand.SU
                                                             : nullptr)
      if (HigherSU->getHeight() > Zone->getScheduledLatency() &&
          HigherSU->getDepth() < computeRemLatency(*Zone)) {
        // The higher SU increases the scheduled latency but is not on the
        // Critical Path by Depth, so put it above the other one.
        tryLess(TryCand.SU->getHeight(), Cand.SU->getHeight(), TryCand, Cand,
                GenericSchedulerBase::BotHeightReduce);
        return TryCand.Reason != NoCand;
      }

  // Weak edges help copy coalescing.
  if (tryLess(TryCand.SU->WeakSuccsLeft, Cand.SU->WeakSuccsLeft, TryCand, Cand,
              Weak))
    return TryCand.Reason != NoCand;

  // Help compare with zero elimination.
  if (tryGreater(definesCmp0Src(TryCand.SU->getInstr()),
                 definesCmp0Src(Cand.SU->getInstr()), TryCand, Cand, Weak))
    return TryCand.Reason != NoCand;

  // Fall through to original instruction order.
  if (TryCand.SU->NodeNum > Cand.SU->NodeNum) {
    TryCand.Reason = NodeOrder;
    return true;
  }

  return false;
}

void SystemZPreRASchedStrategy::initPolicy(MachineBasicBlock::iterator Begin,
                                           MachineBasicBlock::iterator End,
                                           unsigned NumRegionInstrs) {
  // Avoid setting up the register pressure tracker unless needed to save
  // compile time.
  RegionPolicy.ShouldTrackPressure = NumRegionInstrs > TopRegionSUs;

  // These heuristics has so far seemed to work better without adding a
  // top-down boundary.
  RegionPolicy.OnlyBottomUp = true;

  BotIdx = NumRegionInstrs - 1;
  this->NumRegionInstrs = NumRegionInstrs;
}

void SystemZPreRASchedStrategy::initialize(ScheduleDAGMI *dag) {
  GenericScheduler::initialize(dag);

  Cmp0SrcReg = SystemZ::NoRegister;

  unsigned DAGHeight = 0;
  for (unsigned Idx = 0, End = DAG->SUnits.size(); Idx != End; ++Idx)
    DAGHeight = std::max(DAGHeight, DAG->SUnits[Idx].getHeight());

  if (RegionPolicy.ShouldTrackPressure)
    LivenessHeightCutOff = DAGHeight / (DAG->SUnits.size() < 50 ? 4 : 2);

  if (DisableLatency)
    RegionPolicy.DisableLatencyHeuristic = true;
  else
    // Disable latency reduction if region has many SUs relative to the
    // overall height.
    RegionPolicy.DisableLatencyHeuristic =
        DAG->SUnits.size() >= 3 * std::max(DAGHeight, 1u);
}

void SystemZPreRASchedStrategy::schedNode(SUnit *SU, bool IsTopNode) {
  GenericScheduler::schedNode(SU, IsTopNode);

  const SystemZInstrInfo *TII = static_cast<const SystemZInstrInfo *>(DAG->TII);
  MachineInstr *MI = SU->getInstr();
  if (TII->isCompareZero(*MI))
    Cmp0SrcReg = TII->getCompareSourceReg(*MI);
  else if (MI->getDesc().hasImplicitDefOfPhysReg(SystemZ::CC) ||
           definesCmp0Src(MI, /*CCDef=*/false))
    Cmp0SrcReg = SystemZ::NoRegister;
}

/// Post-RA scheduling ///

#ifndef NDEBUG
// Print the set of SUs
void SystemZPostRASchedStrategy::SUSet::
dump(SystemZHazardRecognizer &HazardRec) const {
  dbgs() << "{";
  for (auto &SU : *this) {
    HazardRec.dumpSU(SU, dbgs());
    if (SU != *rbegin())
      dbgs() << ",  ";
  }
  dbgs() << "}\n";
}
#endif

// Try to find a single predecessor that would be interesting for the
// scheduler in the top-most region of MBB.
static MachineBasicBlock *getSingleSchedPred(MachineBasicBlock *MBB,
                                             const MachineLoop *Loop) {
  MachineBasicBlock *PredMBB = nullptr;
  if (MBB->pred_size() == 1)
    PredMBB = *MBB->pred_begin();

  // The loop header has two predecessors, return the latch, but not for a
  // single block loop.
  if (MBB->pred_size() == 2 && Loop != nullptr && Loop->getHeader() == MBB) {
    for (MachineBasicBlock *Pred : MBB->predecessors())
      if (Loop->contains(Pred))
        PredMBB = (Pred == MBB ? nullptr : Pred);
  }

  assert ((PredMBB == nullptr || !Loop || Loop->contains(PredMBB))
          && "Loop MBB should not consider predecessor outside of loop.");

  return PredMBB;
}

void SystemZPostRASchedStrategy::
advanceTo(MachineBasicBlock::iterator NextBegin) {
  MachineBasicBlock::iterator LastEmittedMI = HazardRec->getLastEmittedMI();
  MachineBasicBlock::iterator I =
    ((LastEmittedMI != nullptr && LastEmittedMI->getParent() == MBB) ?
     std::next(LastEmittedMI) : MBB->begin());

  for (; I != NextBegin; ++I) {
    if (I->isPosition() || I->isDebugInstr())
      continue;
    HazardRec->emitInstruction(&*I);
  }
}

void SystemZPostRASchedStrategy::initialize(ScheduleDAGMI *dag) {
  Available.clear();  // -misched-cutoff.
  LLVM_DEBUG(HazardRec->dumpState(););
}

void SystemZPostRASchedStrategy::enterMBB(MachineBasicBlock *NextMBB) {
  assert ((SchedStates.find(NextMBB) == SchedStates.end()) &&
          "Entering MBB twice?");
  LLVM_DEBUG(dbgs() << "** Entering " << printMBBReference(*NextMBB));

  MBB = NextMBB;

  /// Create a HazardRec for MBB, save it in SchedStates and set HazardRec to
  /// point to it.
  HazardRec = SchedStates[MBB] = new SystemZHazardRecognizer(TII, &SchedModel);
  LLVM_DEBUG(const MachineLoop *Loop = MLI->getLoopFor(MBB);
             if (Loop && Loop->getHeader() == MBB) dbgs() << " (Loop header)";
             dbgs() << ":\n";);

  // Try to take over the state from a single predecessor, if it has been
  // scheduled. If this is not possible, we are done.
  MachineBasicBlock *SinglePredMBB =
    getSingleSchedPred(MBB, MLI->getLoopFor(MBB));
  if (SinglePredMBB == nullptr)
    return;
  auto It = SchedStates.find(SinglePredMBB);
  if (It == SchedStates.end())
    return;

  LLVM_DEBUG(dbgs() << "** Continued scheduling from "
                    << printMBBReference(*SinglePredMBB) << "\n";);

  HazardRec->copyState(It->second);
  LLVM_DEBUG(HazardRec->dumpState(););

  // Emit incoming terminator(s). Be optimistic and assume that branch
  // prediction will generally do "the right thing".
  for (MachineInstr &MI : SinglePredMBB->terminators()) {
    LLVM_DEBUG(dbgs() << "** Emitting incoming branch: "; MI.dump(););
    bool TakenBranch = (MI.isBranch() &&
                        (TII->getBranchInfo(MI).isIndirect() ||
                         TII->getBranchInfo(MI).getMBBTarget() == MBB));
    HazardRec->emitInstruction(&MI, TakenBranch);
    if (TakenBranch)
      break;
  }
}

void SystemZPostRASchedStrategy::leaveMBB() {
  LLVM_DEBUG(dbgs() << "** Leaving " << printMBBReference(*MBB) << "\n";);

  // Advance to first terminator. The successor block will handle terminators
  // dependent on CFG layout (T/NT branch etc).
  advanceTo(MBB->getFirstTerminator());
}

SystemZPostRASchedStrategy::
SystemZPostRASchedStrategy(const MachineSchedContext *C)
  : MLI(C->MLI),
    TII(static_cast<const SystemZInstrInfo *>
        (C->MF->getSubtarget().getInstrInfo())),
    MBB(nullptr), HazardRec(nullptr) {
  const TargetSubtargetInfo *ST = &C->MF->getSubtarget();
  SchedModel.init(ST);
}

SystemZPostRASchedStrategy::~SystemZPostRASchedStrategy() {
  // Delete hazard recognizers kept around for each MBB.
  for (auto I : SchedStates) {
    SystemZHazardRecognizer *hazrec = I.second;
    delete hazrec;
  }
}

void SystemZPostRASchedStrategy::initPolicy(MachineBasicBlock::iterator Begin,
                                            MachineBasicBlock::iterator End,
                                            unsigned NumRegionInstrs) {
  // Don't emit the terminators.
  if (Begin->isTerminator())
    return;

  // Emit any instructions before start of region.
  advanceTo(Begin);
}

// Pick the next node to schedule.
SUnit *SystemZPostRASchedStrategy::pickNode(bool &IsTopNode) {
  // Only scheduling top-down.
  IsTopNode = true;

  if (Available.empty())
    return nullptr;

  // If only one choice, return it.
  if (Available.size() == 1) {
    LLVM_DEBUG(dbgs() << "** Only one: ";
               HazardRec->dumpSU(*Available.begin(), dbgs()); dbgs() << "\n";);
    return *Available.begin();
  }

  // All nodes that are possible to schedule are stored in the Available set.
  LLVM_DEBUG(dbgs() << "** Available: "; Available.dump(*HazardRec););

  Candidate Best;
  for (auto *SU : Available) {

    // SU is the next candidate to be compared against current Best.
    Candidate c(SU, *HazardRec);

    // Remeber which SU is the best candidate.
    if (Best.SU == nullptr || c < Best) {
      Best = c;
      LLVM_DEBUG(dbgs() << "** Best so far: ";);
    } else
      LLVM_DEBUG(dbgs() << "** Tried      : ";);
    LLVM_DEBUG(HazardRec->dumpSU(c.SU, dbgs()); c.dumpCosts();
               dbgs() << " Height:" << c.SU->getHeight(); dbgs() << "\n";);

    // Once we know we have seen all SUs that affect grouping or use unbuffered
    // resources, we can stop iterating if Best looks good.
    if (!SU->isScheduleHigh && Best.noCost())
      break;
  }

  assert (Best.SU != nullptr);
  return Best.SU;
}

SystemZPostRASchedStrategy::Candidate::
Candidate(SUnit *SU_, SystemZHazardRecognizer &HazardRec) : Candidate() {
  SU = SU_;

  // Check the grouping cost. For a node that must begin / end a
  // group, it is positive if it would do so prematurely, or negative
  // if it would fit naturally into the schedule.
  GroupingCost = HazardRec.groupingCost(SU);

  // Check the resources cost for this SU.
  ResourcesCost = HazardRec.resourcesCost(SU);
}

bool SystemZPostRASchedStrategy::Candidate::
operator<(const Candidate &other) {

  // Check decoder grouping.
  if (GroupingCost < other.GroupingCost)
    return true;
  if (GroupingCost > other.GroupingCost)
    return false;

  // Compare the use of resources.
  if (ResourcesCost < other.ResourcesCost)
    return true;
  if (ResourcesCost > other.ResourcesCost)
    return false;

  // Higher SU is otherwise generally better.
  if (SU->getHeight() > other.SU->getHeight())
    return true;
  if (SU->getHeight() < other.SU->getHeight())
    return false;

  // If all same, fall back to original order.
  if (SU->NodeNum < other.SU->NodeNum)
    return true;

  return false;
}

void SystemZPostRASchedStrategy::schedNode(SUnit *SU, bool IsTopNode) {
  LLVM_DEBUG(dbgs() << "** Scheduling SU(" << SU->NodeNum << ") ";
             if (Available.size() == 1) dbgs() << "(only one) ";
             Candidate c(SU, *HazardRec); c.dumpCosts(); dbgs() << "\n";);

  // Remove SU from Available set and update HazardRec.
  Available.erase(SU);
  HazardRec->EmitInstruction(SU);
}

void SystemZPostRASchedStrategy::releaseTopNode(SUnit *SU) {
  // Set isScheduleHigh flag on all SUs that we want to consider first in
  // pickNode().
  const MCSchedClassDesc *SC = HazardRec->getSchedClass(SU);
  bool AffectsGrouping = (SC->isValid() && (SC->BeginGroup || SC->EndGroup));
  SU->isScheduleHigh = (AffectsGrouping || SU->isUnbuffered);

  // Put all released SUs in the Available set.
  Available.insert(SU);
}
