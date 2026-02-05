//-- SystemZMachineScheduler.cpp - SystemZ Scheduler Interface -*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SystemZMachineScheduler.h"
#include "llvm/CodeGen/MachineLoopInfo.h"

using namespace llvm;

#define DEBUG_TYPE "machine-scheduler"

/// Pre-RA scheduling ///

namespace SystemZSched {
enum LatencyReduction { Always, Never, More, Heuristics, CycleBased };
} // namespace SystemZSched

static cl::opt<SystemZSched::LatencyReduction> PreRALatRed(
    "prera-lat-red", cl::Hidden,
    cl::desc("Tuning of latency reduction during pre-ra mi-scheduling."),
    cl::init(SystemZSched::LatencyReduction::Heuristics),
    cl::values(
        clEnumValN(SystemZSched::LatencyReduction::Always, "always",
                   "Reduce scheduled latency always."),
        clEnumValN(SystemZSched::LatencyReduction::Never, "never",
                   "Don't reduce scheduled latency."),
        clEnumValN(SystemZSched::LatencyReduction::More, "more",
                   "Reduce scheduled latency on most DAGs."),
        clEnumValN(SystemZSched::LatencyReduction::Heuristics, "heuristics",
                   "Use heuristics for reduction of scheduled latency."),
        clEnumValN(SystemZSched::LatencyReduction::CycleBased, "cycle-based",
                   "Use GenericSched cycle based decisions for reduction of "
                   "scheduled latency.")));

static bool isRegDef(const MachineOperand &MO) {
  return MO.isReg() && MO.isDef();
}

static bool isPhysRegDef(const MachineOperand &MO) {
  return isRegDef(MO) && MO.getReg().isPhysical();
}

void SystemZPreRASchedStrategy::initializeLatencyReduction() {
  // Enable latency reduction for a region that has a considerable amount of
  // data sequences that should be interlaved. These are SUs that only have
  // one data predecessor / successor edge(s) to their adjacent instruction(s)
  // in the input order. Disable if region has many SUs relative to the
  // overall height.
  unsigned DAGHeight = 0;
  for (unsigned Idx = 0, End = DAG->SUnits.size(); Idx != End; ++Idx)
    DAGHeight = std::max(DAGHeight, DAG->SUnits[Idx].getHeight());
  IsWideDAG = DAG->SUnits.size() >= 3 * std::max(DAGHeight, 1u);
  if ((HasDataSequences = !IsWideDAG)) {
    unsigned CurrSequence = 0, NumSeqNodes = 0;
    auto countSequence = [&CurrSequence, &NumSeqNodes]() {
      if (CurrSequence >= 2)
        NumSeqNodes += CurrSequence;
      CurrSequence = 0;
    };
    for (unsigned Idx = 0, End = DAG->SUnits.size(); Idx != End; ++Idx) {
      const SUnit *SU = &DAG->SUnits[Idx];
      bool InDataSequence = true;
      // One Data pred to MI just above, or no preds.
      unsigned NumPreds = 0;
      for (const SDep &Pred : SU->Preds)
        if (++NumPreds != 1 || Pred.getKind() != SDep::Data ||
            Pred.getSUnit()->NodeNum != Idx - 1)
          InDataSequence = false;
      // One Data succ or no succs (ignoring ExitSU).
      unsigned NumSuccs = 0;
      for (const SDep &Succ : SU->Succs)
        if (Succ.getSUnit() != &DAG->ExitSU &&
            (++NumSuccs != 1 || Succ.getKind() != SDep::Data))
          InDataSequence = false;
      // Another type of node or one that does not have a single data pred
      // ends any previous sequence.
      if (!InDataSequence || !NumPreds)
        countSequence();
      if (InDataSequence)
        CurrSequence++;
    }
    countSequence();
    if (NumSeqNodes >= std::max(size_t(4), DAG->SUnits.size() / 4)) {
      LLVM_DEBUG(dbgs() << "Number of nodes in def-use sequences: "
                        << NumSeqNodes << ". ";);
    } else
      HasDataSequences = false;
  }
}

bool SystemZPreRASchedStrategy::definesCmp0Src(const MachineInstr *MI,
                                               bool CCDef) const {
  if (Cmp0SrcReg != SystemZ::NoRegister && MI->getNumOperands() &&
      (MI->getDesc().hasImplicitDefOfPhysReg(SystemZ::CC) || !CCDef)) {
    const MachineOperand &MO0 = MI->getOperand(0);
    assert(!isPhysRegDef(MO0) && "Did not expect physreg def!");
    if (isRegDef(MO0) && MO0.getReg() == Cmp0SrcReg)
      return true;
  }
  return false;
}

bool SystemZPreRASchedStrategy::shouldReduceLatency(SchedBoundary *Zone) const {
  if (PreRALatRed == SystemZSched::Always)
    return true;
  if (PreRALatRed == SystemZSched::Never)
    return false;

  if (IsWideDAG)
    return false;

  if (PreRALatRed == SystemZSched::More)
    return true;

  if (PreRALatRed == SystemZSched::Heuristics)
    // Don't extend the scheduled latency in regions with many nodes in data
    // sequences, or for (single block loop) regions that are acyclically
    // (within a single loop iteration) latency limited.
    return HasDataSequences || Rem.IsAcyclicLatencyLimited;

  if (PreRALatRed == SystemZSched::CycleBased) {
    CandPolicy P;
    getRemLat(Zone);
    return GenericScheduler::shouldReduceLatency(P, *Zone, false, RemLat);
  }

  llvm_unreachable("Unhandled option value.");
}

unsigned SystemZPreRASchedStrategy::getRemLat(SchedBoundary *Zone) const {
  if (RemLat == ~0U)
    RemLat = computeRemLatency(*Zone);
  return RemLat;
}

static int biasPhysRegExtra(const SUnit *SU) {
  if (int Res = biasPhysReg(SU, /*isTop=*/false))
    return Res;

  // Also recognize Load Address. Most of these are with an FI operand.
  const MachineInstr *MI = SU->getInstr();
  return MI->getNumOperands() && !MI->isCopy() &&
         isPhysRegDef(MI->getOperand(0));
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

  auto shouldReallyReduceLatency = [&]() {
    if (shouldReduceLatency(Zone))
      if (const SUnit *HigherSU =
              TryCand.SU->getHeight() > Cand.SU->getHeight()   ? TryCand.SU
              : TryCand.SU->getHeight() < Cand.SU->getHeight() ? Cand.SU
                                                               : nullptr) {
        if (HigherSU->getHeight() > Zone->getScheduledLatency() &&
            HigherSU->getDepth() < getRemLat(Zone))
          return true;
      }
    return false;
  };

  // One or both SUs increase the scheduled latency.
  if (shouldReallyReduceLatency()) {
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
  // Avoid setting up the register pressure tracker for small regions to save
  // compile time. Currently only used for computeCyclicCriticalPath() which
  // is used for single block loops.
  MachineBasicBlock *MBB = Begin->getParent();
  RegionPolicy.ShouldTrackPressure =
    MBB->isSuccessor(MBB) && NumRegionInstrs >= 8;

  // These heuristics has so far seemed to work better without adding a
  // top-down boundary.
  RegionPolicy.OnlyBottomUp = true;
  BotIdx = NumRegionInstrs - 1;
  this->NumRegionInstrs = NumRegionInstrs;
}

void SystemZPreRASchedStrategy::initialize(ScheduleDAGMI *dag) {
  GenericScheduler::initialize(dag);

  RemLat = ~0U;
  Cmp0SrcReg = SystemZ::NoRegister;

  initializeLatencyReduction();
  LLVM_DEBUG(dbgs() << "Latency scheduling " << (HasDataSequences ? "" : "not ")
                    << "enabled for data sequences.\n";);
}

void SystemZPreRASchedStrategy::schedNode(SUnit *SU, bool IsTopNode) {
  GenericScheduler::schedNode(SU, IsTopNode);

  RemLat = ~0U;

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
