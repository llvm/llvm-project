//-- SystemZMachineScheduler.cpp - SystemZ Scheduler Interface -*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SystemZMachineScheduler.h"
#include "llvm/CodeGen/LiveInterval.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineLoopInfo.h"

using namespace llvm;

#define DEBUG_TYPE "machine-scheduler"

/// Pre-RA scheduling ///

static cl::opt<unsigned> TinyRegionLim(
    "tiny-region-lim", cl::Hidden, cl::init(10),
    cl::desc("Run limited pre-ra scheduling on regions of this size or "
             "smaller. Mainly for testing."));

static bool isRegDef(const MachineOperand &MO) {
  return MO.isReg() && MO.isDef();
}

static bool isVirtRegDef(const MachineOperand &MO) {
  return isRegDef(MO) && MO.getReg().isVirtual();
}

static bool isPhysRegDef(const MachineOperand &MO) {
  return isRegDef(MO) && MO.getReg().isPhysical();
}

static bool isVirtRegUse(const MachineOperand &MO) {
  return MO.isReg() && MO.isUse() && MO.readsReg() && MO.getReg().isVirtual();
}

void SystemZPreRASchedStrategy::initializePrioRegClasses(
    const TargetRegisterInfo *TRI) {
  for (const TargetRegisterClass *RC : TRI->regclasses()) {
    for (MVT VT : MVT::fp_valuetypes())
      if (TRI->isTypeLegalForClass(*RC, VT)) {
        PrioRegClasses.insert(RC->getID());
        break;
      }

    // On SystemZ vector and FP registers overlap: add any vector RC.
    if (!PrioRegClasses.count(RC->getID()))
      for (MVT VT : MVT::fp_fixedlen_vector_valuetypes())
        if (TRI->isTypeLegalForClass(*RC, VT)) {
          PrioRegClasses.insert(RC->getID());
          break;
        }
  }
}

void SystemZPreRASchedStrategy::VRegSet::dump(std::string Msg) {
  dbgs() << Msg.c_str();
  bool First = true;
  for (auto R : *this) {
    if (!First)
      dbgs() << ", ";
    else
      First = false;
    dbgs() << "%" << R.virtRegIndex();
  }
  dbgs() << "\n";
}

unsigned SystemZPreRASchedStrategy::getRemLat(SchedBoundary *Zone) const {
  if (RemLat == ~0U)
    RemLat = computeRemLatency(*Zone);
  return RemLat;
}

void SystemZPreRASchedStrategy::initializeStoresGroup() {
  StoresGroup.clear();
  FirstStoreInGroupScheduled = false;

  unsigned CurrMaxDepth = 0;
  for (unsigned Idx = DAG->SUnits.size() - 1; Idx + 1 != 0; --Idx) {
    const SUnit *SU = &DAG->SUnits[Idx];
    const MachineInstr *MI = SU->getInstr();
    if (!MI->getNumOperands() || MI->isCopy())
      continue;

    bool HasVirtDef = false;
    bool HasVirtUse = false;
    for (unsigned I = 0; I < MI->getDesc().getNumOperands(); ++I) {
      const MachineOperand &MO = MI->getOperand(I);
      if (isVirtRegDef(MO) && !MO.isDead())
        HasVirtDef = true;
      else if (isVirtRegUse(MO) &&
               MI->getDesc().operands()[I].OperandType != MCOI::OPERAND_MEMORY)
        HasVirtUse = true;
    }
    bool IsStore = !HasVirtDef && HasVirtUse;

    // Find a group of stores that all are at the bottom while avoiding
    // regions with any additional group of lesser depth.
    if (SU->getDepth() > CurrMaxDepth) {
      CurrMaxDepth = SU->getDepth();
      bool PrevGroup = StoresGroup.size() > 1;
      StoresGroup.clear();
      if (PrevGroup)
        return;
      if (IsStore)
        StoresGroup.insert(SU);
    }
    else if (IsStore && !StoresGroup.empty() && SU->getDepth() == CurrMaxDepth) {
      // The group members should all have the same opcode.
      if ((*StoresGroup.begin())->getInstr()->getOpcode() != MI->getOpcode()) {
        StoresGroup.clear();
        return;
      }
      StoresGroup.insert(SU);
    }
  }

  // Value of 8 handles a known regression (with group of 20).
  // TODO: Would some other value be better?
  if (StoresGroup.size() < 8)
    StoresGroup.clear();
}

static int biasPhysRegExtra(const SUnit *SU) {
  if (int Res = biasPhysReg(SU, /*isTop=*/false))
    return Res;

  // Also recognize Load Address of stack slot. There are (at least
  // currently) no instructions here defining a physreg that uses a vreg.
  const MachineInstr *MI = SU->getInstr();
  if (MI->getNumOperands() && !MI->isCopy()) {
    const MachineOperand &DefMO = MI->getOperand(0);
    if (isPhysRegDef(DefMO))
      return 1;
  }

  return 0;
}

int SystemZPreRASchedStrategy::
computeSULivenessScore(SchedCandidate &C, ScheduleDAGMILive *DAG,
                       SchedBoundary *Zone) const {
  // Not all data deps are modelled around the SUnit - some data edges near
  // boundaries are missing: Look directly at the MI operands instead.
  const SUnit *SU = C.SU;
  const MachineInstr *MI = SU->getInstr();
  if (!MI->getNumOperands() || MI->isCopy())
    return 0;

  // Find uses of registers that are not already live (kills).
  bool PrioKill = false;
  bool GPRKill = false;
  bool AddrKill = false;
  bool HasPrioUse = false;
  for (unsigned I = 0; I < MI->getDesc().getNumOperands(); ++I) {
    const MachineOperand &MO = MI->getOperand(I);
    if (!isVirtRegUse(MO))
      continue;
    HasPrioUse |= isPrioVirtReg(MO.getReg(), &DAG->MRI);
    if (LiveRegs.count(MO.getReg()))
      continue;
    if (isPrioVirtReg(MO.getReg(), &DAG->MRI))
      PrioKill = true;
    else if (MI->getDesc().operands()[I].OperandType != MCOI::OPERAND_MEMORY)
      GPRKill = true;
    else
      AddrKill = true;
  }

  // Find the interesting properties.
  const MachineOperand &DefMO = MI->getOperand(0);
  assert(!isPhysRegDef(DefMO) && "Did not expect physreg def!");
  bool IsLoad =
      isRegDef(DefMO) && !DefMO.isDead() && !IsRedefining[SU->NodeNum];
  bool IsStore = (!isRegDef(DefMO) || DefMO.isDead());
  // Prioritize FP: Ignore GPR/Addr kills with an FP def.
  bool UsesLivePrio =
      IsLoad && !PrioKill &&
      (isPrioVirtReg(DefMO.getReg(), &DAG->MRI) || (!GPRKill && !AddrKill));
  bool UsesLiveAll = !PrioKill && !GPRKill && !AddrKill;
  bool PreservesSchedLat = SU->getHeight() <= Zone->getScheduledLatency();
  const unsigned Cycles = 2;
  unsigned Margin = SchedModel->getIssueWidth() * (Cycles + SU->Latency - 1);
  bool HasDistToTop = NumLeft > Margin;

  // Pull down a defining SU if it preserves the scheduled latency while not
  // causing any (prioritized) register uses to become live. If however there
  // will be relatively many SUs scheduled above this one and all uses are
  // already live it should not be a problem to increase the scheduled
  // latency given the OOO execution.
  // TODO: Try schedulling small (DFSResult) subtrees as a unit.
  bool SchedLow = IsLoad && ((PreservesSchedLat && UsesLivePrio) ||
                             (HasDistToTop && UsesLiveAll));

  // This handles regions with many chained stores of the same depth at the
  // bottom in the input order (cactus). Push them upwards during scheduling.
  bool SchedHigh = IsStore && FirstStoreInGroupScheduled &&
                   StoresGroup.count(SU) &&
                   (PrioKill || (!HasPrioUse && GPRKill));

  if (SchedLow)
    return -1;
  if (SchedHigh)
    return 1;
  return 0;
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
  int TryCandPRegBias = biasPhysRegExtra(TryCand.SU);
  int CandPRegBias = biasPhysRegExtra(Cand.SU);
  if (tryGreater(TryCandPRegBias, CandPRegBias, TryCand, Cand, PhysReg))
    return TryCand.Reason != NoCand;
  if (TryCandPRegBias && CandPRegBias) {
    // Both biased same way.
    tryGreater(TryCand.SU->NodeNum, Cand.SU->NodeNum, TryCand, Cand, NodeOrder);
    return TryCand.Reason != NoCand;
  }

  if (TinyRegion) {
    // Prioritize instructions that read unbuffered resources by stall cycles.
    // TODO: Try this in bigger regions as well.
    if (tryLess(Zone->getLatencyStallCycles(TryCand.SU),
                Zone->getLatencyStallCycles(Cand.SU), TryCand, Cand, Stall))
      return TryCand.Reason != NoCand;
  } else {
    // Look for an opportunity to reduce register liveness.
    int TryCandScore = computeSULivenessScore(TryCand, DAG, Zone);
    int CandScore = computeSULivenessScore(Cand, DAG, Zone);
    if (tryLess(TryCandScore, CandScore, TryCand, Cand, LivenessReduce))
      return TryCand.Reason != NoCand;

    // Don't extend the scheduled latency.
    if (ShouldReduceLatency && TryCand.SU->getHeight() != Cand.SU->getHeight() &&
        (std::max(TryCand.SU->getHeight(), Cand.SU->getHeight()) >
         Zone->getScheduledLatency())) {
      unsigned HigherSUDepth = TryCand.SU->getHeight() < Cand.SU->getHeight() ?
        Cand.SU->getDepth() : TryCand.SU->getDepth();
      if (HigherSUDepth != getRemLat(Zone) &&
          tryLess(TryCand.SU->getHeight(), Cand.SU->getHeight(),
                  TryCand, Cand, GenericSchedulerBase::BotHeightReduce)) {
        return TryCand.Reason != NoCand;
      }
    }
  }

  // Weak edges are for clustering and other constraints.
  if (tryLess(TryCand.SU->WeakSuccsLeft, Cand.SU->WeakSuccsLeft,
              TryCand, Cand, Weak))
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
  // Keep track of live regs instead of using the generic reg pressure tracking.
  RegionPolicy.ShouldTrackPressure = false;
  // These heuristics has so far seemed to work better without adding a
  // top-down boundary.
  RegionPolicy.OnlyBottomUp = true;
}

void SystemZPreRASchedStrategy::initialize(ScheduleDAGMI *dag) {
  GenericScheduler::initialize(dag);

  TinyRegion = DAG->SUnits.size() <= TinyRegionLim;
  const SystemZInstrInfo *TII = static_cast<const SystemZInstrInfo *>(DAG->TII);
  if (TinyRegion) {
    // A tiny region with long latency instructions is better handled using
    // normal heuristics, except in regions that have COPYs of a physreg both
    // ways and/or have a compare-0 likely to be eliminated.
    const SUnit *CmpZeroSU = nullptr;
    const SUnit *CmpSrcSU = nullptr;
    Register CmpSrcReg = 0;
    bool OtherCCClob = false;
    unsigned MaxLat = 0;
    std::set<Register> PRegs;
    bool CopysPRegDep = false;
    for (unsigned Idx = DAG->SUnits.size() - 1; Idx + 1 != 0; --Idx) {
      const SUnit *SU = &DAG->SUnits[Idx];
      const MachineInstr *MI = SU->getInstr();

      // Check for a (likely) eliminable compare-0.
      if (TII->isCompareZero(*MI)) {
        CmpZeroSU = SU;
        CmpSrcReg = TII->getCompareSourceReg(*MI);
        continue;
      }
      if (MI->getNumOperands()) {
        const MachineOperand &DefMO = MI->getOperand(0);
        // Doing this instead of SU data preds happens to also handle the
        // case where CmpSrcReg is redefined.
        if (isVirtRegDef(DefMO) && DefMO.getReg() == CmpSrcReg &&
            MI->getDesc().hasImplicitDefOfPhysReg(SystemZ::CC))
          CmpSrcSU = SU;
      }
      if (SU != CmpZeroSU && SU != CmpSrcSU &&
          MI->getDesc().hasImplicitDefOfPhysReg(SystemZ::CC))
        OtherCCClob = true;

      // Check for long latency instructions.
      MaxLat = std::max(MaxLat, unsigned(SU->Latency));

      // Check for COPYs of pregs both in and out of the region.
      if (MI->isCopy()) {
        Register DstReg = MI->getOperand(0).getReg();
        Register SrcReg = MI->getOperand(1).getReg();
        if (DstReg.isPhysical() && DAG->MRI.isAllocatable(DstReg) &&
            SrcReg.isVirtual())
          PRegs.insert(DstReg);
        else if (SrcReg.isPhysical() && DAG->MRI.isAllocatable(SrcReg) &&
                 DstReg.isVirtual()) {
          if (!PRegs.insert(SrcReg).second)
            CopysPRegDep = true;
        }
      }
    }
    bool CmpElimRegion = CmpZeroSU && CmpSrcSU && OtherCCClob;

    if (DAG->SUnits.size() > 6 && MaxLat >= 6 && !CopysPRegDep &&
        !CmpElimRegion)
      TinyRegion = false;
  }
  LLVM_DEBUG(dbgs() << "Region is" << (TinyRegion ? "" : " not") << " tiny.\n");
  if (TinyRegion)
    return;

  NumLeft = DAG->SUnits.size();
  RemLat = ~0U;

  // It seems to work best to include the latencies in this heuristic (as
  // opposed to something like a "unit SU height" with all latencies counted
  // as 1).
  unsigned DAGHeight = 0;
  for (unsigned Idx = 0, End = DAG->SUnits.size(); Idx != End; ++Idx)
    DAGHeight = std::max(DAGHeight, DAG->SUnits[Idx].getHeight());
  ShouldReduceLatency = DAG->SUnits.size() < 3 * std::max(DAGHeight, 1u);
  LLVM_DEBUG(if (ShouldReduceLatency) dbgs() << "Latency scheduling enabled.\n";
             else dbgs() << "Latency scheduling disabled.\n";);

  // Find the registers that are live at the bottom, before scheduling.
  LiveRegs.clear();
  for (unsigned I = 0, E = DAG->MRI.getNumVirtRegs(); I != E; ++I) {
    Register VirtReg = Register::index2VirtReg(I);
    const LiveInterval &LI = DAG->getLIS()->getInterval(VirtReg);
    LiveQueryResult LRQ = LI.Query(
        DAG->getLIS()->getInstructionIndex(*DAG->SUnits.back().getInstr()));
    if (LRQ.valueOut())
      LiveRegs.insert(VirtReg);
  }
  LLVM_DEBUG(LiveRegs.dump("Live out at bottom: "););

  // If MI uses the register it defines, record it one time here.
  IsRedefining = std::vector<bool>(DAG->SUnits.size(), false);
  for (unsigned Idx = 0, End = DAG->SUnits.size(); Idx != End; ++Idx) {
    const MachineInstr *MI = DAG->SUnits[Idx].getInstr();
    if (MI->getNumOperands()) {
      const MachineOperand &DefMO = MI->getOperand(0);
      if (isVirtRegDef(DefMO))
        IsRedefining[Idx] = MI->readsVirtualRegister(DefMO.getReg());
    }
  }

  initializeStoresGroup();
  LLVM_DEBUG(if (!StoresGroup.empty()) dbgs()
                 << "Has StoresGroup of " << StoresGroup.size() << " stores.\n";
             else dbgs() << "No StoresGroup.\n";);
}

void SystemZPreRASchedStrategy::schedNode(SUnit *SU, bool IsTopNode) {
  GenericScheduler::schedNode(SU, IsTopNode);
  if (TinyRegion)
    return;

  LLVM_DEBUG(LiveRegs.dump("Live regs was: "););

  if (!FirstStoreInGroupScheduled && StoresGroup.count(SU))
    FirstStoreInGroupScheduled = true;

  // Update LiveRegs.
  MachineInstr *MI = SU->getInstr();
  for (auto &MO : MI->explicit_operands())
    if (MO.isReg() && MO.getReg().isVirtual()) {
      if (MO.isDef()) {
        // A subreg def may not be in LiveRegs if the use of it was implicit.
        assert(LiveRegs.count(MO.getReg()) || MO.isDead() || MO.getSubReg());
        if (!IsRedefining[SU->NodeNum])
          LiveRegs.erase(MO.getReg());
      } else if (MO.readsReg())
        LiveRegs.insert(MO.getReg());
    }

  assert(NumLeft > 0);
  --NumLeft;
  RemLat = ~0U;
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
