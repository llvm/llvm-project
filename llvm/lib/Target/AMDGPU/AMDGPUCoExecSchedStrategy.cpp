//===- AMDGPUCoExecSchedStrategy.cpp - CoExec Scheduling Strategy ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Coexecution-focused scheduling strategy for AMDGPU.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUCoExecSchedStrategy.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

#define DEBUG_TYPE "machine-scheduler"

namespace {

// Used to disable post-RA scheduling with function level granularity.
class GCNNoopPostScheduleDAG final : public ScheduleDAGInstrs {
public:
  explicit GCNNoopPostScheduleDAG(MachineSchedContext *C)
      : ScheduleDAGInstrs(*C->MF, C->MLI, /*RemoveKillFlags=*/true) {}

  // Do nothing.
  void schedule() override {}
};

} // namespace

static SUnit *pickOnlyChoice(SchedBoundary &Zone) {
  // pickOnlyChoice() releases pending instructions and checks for new hazards.
  SUnit *OnlyChoice = Zone.pickOnlyChoice();
  if (!Zone.Pending.empty())
    return nullptr;

  return OnlyChoice;
}

InstructionFlavor llvm::classifyFlavor(const MachineInstr *MI,
                                       const SIInstrInfo *SII) {
  if (!MI || MI->isDebugInstr())
    return InstructionFlavor::Other;

  unsigned Opc = MI->getOpcode();

  // Check for specific opcodes first.
  if (Opc == AMDGPU::ATOMIC_FENCE || Opc == AMDGPU::S_WAIT_ASYNCCNT ||
      Opc == AMDGPU::S_WAIT_TENSORCNT || Opc == AMDGPU::S_BARRIER_WAIT ||
      Opc == AMDGPU::S_BARRIER_SIGNAL_IMM)
    return InstructionFlavor::Fence;

  if ((SII->isFLAT(*MI) || SII->isFLATGlobal(*MI)) && SII->isDS(*MI))
    return InstructionFlavor::DMA;

  if (SII->isMFMAorWMMA(*MI))
    return InstructionFlavor::WMMA;

  if (SII->isTRANS(*MI))
    return InstructionFlavor::TRANS;

  if (SII->isVALU(*MI))
    return InstructionFlavor::SingleCycleVALU;

  if (SII->isDS(*MI))
    return InstructionFlavor::DS;

  if (SII->isFLAT(*MI) || SII->isFLATGlobal(*MI) || SII->isFLATScratch(*MI))
    return InstructionFlavor::VMEM;

  if (SII->isSALU(*MI))
    return InstructionFlavor::SALU;

  return InstructionFlavor::Other;
}

SUnit *HardwareUnitInfo::getNextTargetSU(bool LookDeep) {
  for (auto *PrioritySU : PrioritySUs) {
    if (!PrioritySU->isTopReady())
      return PrioritySU;
  }

  if (!LookDeep)
    return nullptr;

  // TODO -- we may want to think about more advance strategies here.
  unsigned MinDepth = std::numeric_limits<unsigned int>::max();
  SUnit *TargetSU = nullptr;
  for (auto *SU : AllSUs) {
    if (SU->isScheduled)
      continue;

    if (SU->isTopReady())
      continue;

    if (SU->getDepth() < MinDepth) {
      MinDepth = SU->getDepth();
      TargetSU = SU;
    }
  }
  return TargetSU;
}

void HardwareUnitInfo::insert(SUnit *SU, unsigned BlockingCycles) {
  bool Inserted = AllSUs.insert(SU);
  TotalCycles += BlockingCycles;

  assert(Inserted);
  if (PrioritySUs.empty()) {
    PrioritySUs.insert(SU);
    return;
  }
  unsigned SUDepth = SU->getDepth();
  unsigned CurrDepth = (*PrioritySUs.begin())->getDepth();
  if (SUDepth > CurrDepth)
    return;

  if (SUDepth == CurrDepth) {
    PrioritySUs.insert(SU);
    return;
  }

  // SU is lower depth and should be prioritized.
  PrioritySUs.clear();
  PrioritySUs.insert(SU);
}

void HardwareUnitInfo::schedule(SUnit *SU, unsigned BlockingCycles) {
  // We may want to ignore some HWUIs (e.g. InstructionFlavor::Other). To do so,
  // we just clear the HWUI. However, we still have instructions which map to
  // this HWUI. Don't bother managing the state for these HWUI.
  if (TotalCycles == 0)
    return;

  AllSUs.remove(SU);
  PrioritySUs.remove(SU);

  TotalCycles -= BlockingCycles;

  if (AllSUs.empty())
    return;
  if (PrioritySUs.empty()) {
    for (auto SU : AllSUs) {
      if (PrioritySUs.empty()) {
        PrioritySUs.insert(SU);
        continue;
      }
      unsigned SUDepth = SU->getDepth();
      unsigned CurrDepth = (*PrioritySUs.begin())->getDepth();
      if (SUDepth > CurrDepth)
        continue;

      if (SUDepth == CurrDepth) {
        PrioritySUs.insert(SU);
        continue;
      }

      // SU is lower depth and should be prioritized.
      PrioritySUs.clear();
      PrioritySUs.insert(SU);
    }
  }
}

HardwareUnitInfo *
CandidateHeuristics::getHWUIFromFlavor(InstructionFlavor Flavor) {
  for (auto &HWUICand : HWUInfo) {
    if (HWUICand.getType() == Flavor) {
      return &HWUICand;
    }
  }
  return nullptr;
}

unsigned CandidateHeuristics::getHWUICyclesForInst(SUnit *SU) {
  if (SchedModel && SchedModel->hasInstrSchedModel()) {
    unsigned ReleaseAtCycle = 0;
    const MCSchedClassDesc *SC = DAG->getSchedClass(SU);
    for (TargetSchedModel::ProcResIter
             PI = SchedModel->getWriteProcResBegin(SC),
             PE = SchedModel->getWriteProcResEnd(SC);
         PI != PE; ++PI) {
      ReleaseAtCycle = std::max(ReleaseAtCycle, (unsigned)PI->ReleaseAtCycle);
    }
    return ReleaseAtCycle;
  }
  return -1;
}

void CandidateHeuristics::schedNode(SUnit *SU) {
  HardwareUnitInfo *HWUI =
      getHWUIFromFlavor(classifyFlavor(SU->getInstr(), SII));
  HWUI->schedule(SU, getHWUICyclesForInst(SU));
}

void CandidateHeuristics::initialize(ScheduleDAGMI *SchedDAG,
                                     const TargetSchedModel *TargetSchedModel,
                                     const TargetRegisterInfo *TRI) {
  DAG = SchedDAG;
  SchedModel = TargetSchedModel;

  SRI = static_cast<const SIRegisterInfo *>(TRI);
  SII = static_cast<const SIInstrInfo *>(DAG->TII);

  HWUInfo.resize((int)InstructionFlavor::NUM_FLAVORS);

  for (unsigned I = 0; I < HWUInfo.size(); I++) {
    HWUInfo[I].setType(I);
    HWUInfo[I].reset();
  }

  HWUInfo[(int)InstructionFlavor::WMMA].setProducesCoexecWindow(true);
  HWUInfo[(int)InstructionFlavor::MultiCycleVALU].setProducesCoexecWindow(true);
  HWUInfo[(int)InstructionFlavor::TRANS].setProducesCoexecWindow(true);

  collectHWUIPressure();
}

void CandidateHeuristics::collectHWUIPressure() {
  if (!SchedModel || !SchedModel->hasInstrSchedModel())
    return;

  for (auto &SU : DAG->SUnits) {
    InstructionFlavor Flavor = classifyFlavor(SU.getInstr(), SII);
    HWUInfo[(int)(Flavor)].insert(&SU, getHWUICyclesForInst(&SU));
  }

  LLVM_DEBUG(dumpRegionSummary());
}

void CandidateHeuristics::dumpRegionSummary() {
  MachineBasicBlock *BB = DAG->begin()->getParent();
  dbgs() << "\n=== Region: " << DAG->MF.getName() << " BB" << BB->getNumber()
         << " (" << DAG->SUnits.size() << " SUs) ===\n";

  dbgs() << "\nHWUI Resource Pressure:\n";
  for (auto &HWUI : HWUInfo) {
    if (HWUI.getTotalCycles() == 0)
      continue;

    StringRef Name = getFlavorName(HWUI.getType());
    dbgs() << "  [" << HWUI.getIdx() << "] " << Name << ": "
           << HWUI.getTotalCycles() << " cycles, " << HWUI.size()
           << " instrs\n";
  }
  dbgs() << "\n";
}

void CandidateHeuristics::sortHWUIResources() {
  // Highest priority should be first.
  llvm::sort(HWUInfo, [](HardwareUnitInfo &A, HardwareUnitInfo &B) {
    // Prefer CoexecWindow producers
    if (A.producesCoexecWindow() != B.producesCoexecWindow())
      return A.producesCoexecWindow();

    // Prefer more demanded resources
    if (A.getTotalCycles() != B.getTotalCycles())
      return A.getTotalCycles() > B.getTotalCycles();

    // In ties -- prefer the resource with longer latency instructions
    if (A.size() != B.size())
      return A.size() < B.size();

    // Default to HardwareUnitInfo order
    return A.getIdx() < B.getIdx();
  });
}

bool CandidateHeuristics::tryCriticalResourceDependency(
    GenericSchedulerBase::SchedCandidate &TryCand,
    GenericSchedulerBase::SchedCandidate &Cand, SchedBoundary *Zone) const {

  auto IsCandidateResource = [this, &Cand, &TryCand](unsigned ResourceIdx) {
    HardwareUnitInfo HWUI = HWUInfo[ResourceIdx];

    auto CandFlavor = classifyFlavor(Cand.SU->getInstr(), SII);
    auto TryCandFlavor = classifyFlavor(TryCand.SU->getInstr(), SII);
    bool LookDeep = (CandFlavor == InstructionFlavor::DS ||
                     TryCandFlavor == InstructionFlavor::DS) &&
                    HWUI.getType() == InstructionFlavor::WMMA;
    auto *TargetSU = HWUI.getNextTargetSU(LookDeep);

    // If we do not have a TargetSU for this resource, then it is not critical.
    if (!TargetSU)
      return false;

    return true;
  };

  auto TryEnablesResource = [&Cand, &TryCand, this](unsigned ResourceIdx) {
    HardwareUnitInfo HWUI = HWUInfo[ResourceIdx];
    auto CandFlavor = classifyFlavor(Cand.SU->getInstr(), SII);

    // We want to ensure our DS order matches WMMA order.
    bool LookDeep = CandFlavor == InstructionFlavor::DS &&
                    HWUI.getType() == InstructionFlavor::WMMA;
    auto *TargetSU = HWUI.getNextTargetSU(LookDeep);

    bool CandEnables =
        TargetSU != Cand.SU && DAG->IsReachable(TargetSU, Cand.SU);
    bool TryCandEnables =
        TargetSU != TryCand.SU && DAG->IsReachable(TargetSU, TryCand.SU);

    if (!CandEnables && !TryCandEnables)
      return false;

    if (CandEnables && !TryCandEnables) {
      if (Cand.Reason > GenericSchedulerBase::RegCritical)
        Cand.Reason = GenericSchedulerBase::RegCritical;

      return true;
    }

    if (!CandEnables && TryCandEnables) {
      TryCand.Reason = GenericSchedulerBase::RegCritical;
      return true;
    }

    // Both enable, prefer the critical path.
    bool CandHeight = Cand.SU->getHeight();
    bool TryCandHeight = TryCand.SU->getHeight();

    if (CandHeight > TryCandHeight) {
      if (Cand.Reason > GenericSchedulerBase::RegCritical)
        Cand.Reason = GenericSchedulerBase::RegCritical;

      return true;
    }

    if (CandHeight < TryCandHeight) {
      TryCand.Reason = GenericSchedulerBase::RegCritical;
      return true;
    }

    // Same critical path, just prefer original candidate.
    if (Cand.Reason > GenericSchedulerBase::RegCritical)
      Cand.Reason = GenericSchedulerBase::RegCritical;

    return true;
  };

  for (unsigned I = 0; I < HWUInfo.size(); I++) {
    // If we have encountered a resource that is not critical, then neither
    // candidate enables a critical resource
    if (!IsCandidateResource(I))
      return false;

    bool Enabled = TryEnablesResource(I);
    // If neither has enabled the resource, continue to the next resource
    if (Enabled)
      return true;
  }
  return false;
}

bool CandidateHeuristics::tryCriticalResource(
    GenericSchedulerBase::SchedCandidate &TryCand,
    GenericSchedulerBase::SchedCandidate &Cand, SchedBoundary *Zone) const {
  for (unsigned I = 0; I < HWUInfo.size(); I++) {
    HardwareUnitInfo HWUI = HWUInfo[I];

    bool CandUsesCrit = HWUI.contains(Cand.SU);
    bool TryCandUsesCrit = HWUI.contains(TryCand.SU);

    if (!CandUsesCrit && !TryCandUsesCrit)
      continue;

    if (CandUsesCrit != TryCandUsesCrit) {
      if (CandUsesCrit) {
        if (Cand.Reason > GenericSchedulerBase::RegCritical)
          Cand.Reason = GenericSchedulerBase::RegCritical;
        return true;
      }
      TryCand.Reason = GenericSchedulerBase::RegCritical;
      return true;
    }

    // Otherwise, both use the critical resource
    // For longer latency InstructionFlavors, we should prioritize first by
    // their enablement of critical resources
    if (HWUI.getType() == InstructionFlavor::DS) {
      if (tryCriticalResourceDependency(TryCand, Cand, Zone))
        return true;
    }

    // Prioritize based on HWUI priorities.
    SUnit *Match = HWUI.getHigherPriority(Cand.SU, TryCand.SU);
    if (Match) {
      if (Match == Cand.SU) {
        if (Cand.Reason > GenericSchedulerBase::RegCritical)
          Cand.Reason = GenericSchedulerBase::RegCritical;
        return true;
      }
      TryCand.Reason = GenericSchedulerBase::RegCritical;
      return true;
    }
  }

  return false;
}

AMDGPUCoExecSchedStrategy::AMDGPUCoExecSchedStrategy(
    const MachineSchedContext *C)
    : GCNSchedStrategy(C) {
  SchedStages.push_back(GCNSchedStageID::ILPInitialSchedule);
  SchedStages.push_back(GCNSchedStageID::PreRARematerialize);
  // Use more accurate GCN pressure trackers.
  UseGCNTrackers = true;
}

void AMDGPUCoExecSchedStrategy::initialize(ScheduleDAGMI *DAG) {
  assert((PreRADirection == MISched::Unspecified ||
          PreRADirection == MISched::TopDown) &&
         "coexec scheduler only supports top-down scheduling");
  // Coexecution scheduling strategy is only done top-down to support new
  // resource balancing heuristics.
  RegionPolicy.OnlyTopDown = true;
  RegionPolicy.OnlyBottomUp = false;

  GCNSchedStrategy::initialize(DAG);
  Heurs.initialize(DAG, SchedModel, TRI);
}

void AMDGPUCoExecSchedStrategy::schedNode(SUnit *SU, bool IsTopNode) {
  Heurs.schedNode(SU);
  GCNSchedStrategy::schedNode(SU, IsTopNode);
}

SUnit *AMDGPUCoExecSchedStrategy::pickNode(bool &IsTopNode) {
  assert(RegionPolicy.OnlyTopDown && !RegionPolicy.OnlyBottomUp &&
         "coexec scheduler only supports top-down scheduling");

  if (DAG->top() == DAG->bottom()) {
    assert(Top.Available.empty() && Top.Pending.empty() &&
           Bot.Available.empty() && Bot.Pending.empty() && "ReadyQ garbage");
    return nullptr;
  }

  bool PickedPending = false;
  SUnit *SU = nullptr;
  SchedCandidate *PickedCand = nullptr;
  do {
    PickedPending = false;
    SU = pickOnlyChoice(Top);
    if (!SU) {
      CandPolicy NoPolicy;
      TopCand.reset(NoPolicy);
      pickNodeFromQueue(Top, NoPolicy, DAG->getTopRPTracker(), TopCand,
                        PickedPending, /*IsBottomUp=*/false);
      assert(TopCand.Reason != NoCand && "failed to find a candidate");
      SU = TopCand.SU;
      PickedCand = &TopCand;
    }
    IsTopNode = true;
  } while (SU->isScheduled);

  LLVM_DEBUG(if (PickedCand) dumpPickSummary(SU, IsTopNode, *PickedCand));

  if (PickedPending) {
    unsigned ReadyCycle = SU->TopReadyCycle;
    unsigned CurrentCycle = Top.getCurrCycle();
    if (ReadyCycle > CurrentCycle)
      Top.bumpCycle(ReadyCycle);

    // checkHazard() does not expose the exact cycle where the hazard clears.
    while (Top.checkHazard(SU))
      Top.bumpCycle(Top.getCurrCycle() + 1);

    Top.releasePending();
  }

  if (SU->isTopReady())
    Top.removeReady(SU);
  if (SU->isBottomReady())
    Bot.removeReady(SU);

  LLVM_DEBUG(dbgs() << "Scheduling SU(" << SU->NodeNum << ") "
                    << *SU->getInstr());

  assert(IsTopNode && "coexec scheduler must only schedule from top boundary");
  return SU;
}

void AMDGPUCoExecSchedStrategy::pickNodeFromQueue(
    SchedBoundary &Zone, const CandPolicy &ZonePolicy,
    const RegPressureTracker &RPTracker, SchedCandidate &Cand,
    bool &PickedPending, bool IsBottomUp) {
  assert(Zone.isTop() && "coexec scheduler only supports top boundary");
  assert(!IsBottomUp && "coexec scheduler only supports top-down scheduling");

  const SIRegisterInfo *SRI = static_cast<const SIRegisterInfo *>(TRI);
  ArrayRef<unsigned> Pressure = RPTracker.getRegSetPressureAtPos();
  unsigned SGPRPressure = 0;
  unsigned VGPRPressure = 0;
  PickedPending = false;
  if (DAG->isTrackingPressure()) {
    if (!useGCNTrackers()) {
      SGPRPressure = Pressure[AMDGPU::RegisterPressureSets::SReg_32];
      VGPRPressure = Pressure[AMDGPU::RegisterPressureSets::VGPR_32];
    } else {
      SGPRPressure = DownwardTracker.getPressure().getSGPRNum();
      VGPRPressure = DownwardTracker.getPressure().getArchVGPRNum();
    }
  }

  auto EvaluateQueue = [&](ReadyQueue &Q, bool FromPending) {
    for (SUnit *SU : Q) {
      SchedCandidate TryCand(ZonePolicy);
      initCandidate(TryCand, SU, Zone.isTop(), RPTracker, SRI, SGPRPressure,
                    VGPRPressure, IsBottomUp);
      SchedBoundary *ZoneArg = Cand.AtTop == TryCand.AtTop ? &Zone : nullptr;
      tryCandidateCoexec(Cand, TryCand, ZoneArg);
      if (TryCand.Reason != NoCand) {
        if (TryCand.ResDelta == SchedResourceDelta())
          TryCand.initResourceDelta(Zone.DAG, SchedModel);
        LLVM_DEBUG(printCandidateDecision(Cand, TryCand));
        PickedPending = FromPending;
        Cand.setBest(TryCand);
      } else {
        printCandidateDecision(TryCand, Cand);
      }
    }
  };

  LLVM_DEBUG(dbgs() << "Available Q:\n");
  EvaluateQueue(Zone.Available, /*FromPending=*/false);

  LLVM_DEBUG(dbgs() << "Pending Q:\n");
  EvaluateQueue(Zone.Pending, /*FromPending=*/true);
}

void AMDGPUCoExecSchedStrategy::dumpPickSummary(SUnit *SU, bool IsTopNode,
                                                SchedCandidate &Cand) {
  const SIInstrInfo *SII = static_cast<const SIInstrInfo *>(DAG->TII);
  unsigned Cycle = IsTopNode ? Top.getCurrCycle() : Bot.getCurrCycle();

  dbgs() << "=== Pick @ Cycle " << Cycle << " ===\n";

  InstructionFlavor Flavor = classifyFlavor(SU->getInstr(), SII);
  dbgs() << "Picked: SU(" << SU->NodeNum << ") ";
  SU->getInstr()->print(dbgs(), /*IsStandalone=*/true, /*SkipOpers=*/false,
                        /*SkipDebugLoc=*/true);
  dbgs() << " [" << getFlavorName(Flavor) << "]\n";

  dbgs() << "  Reason: ";
  if (LastAMDGPUReason != AMDGPUSchedReason::None)
    dbgs() << getReasonName(LastAMDGPUReason);
  else if (Cand.Reason != NoCand)
    dbgs() << GenericSchedulerBase::getReasonStr(Cand.Reason);
  else
    dbgs() << "Unknown";
  dbgs() << "\n\n";

  LastAMDGPUReason = AMDGPUSchedReason::None;
}

bool AMDGPUCoExecSchedStrategy::tryCandidateCoexec(SchedCandidate &Cand,
                                                   SchedCandidate &TryCand,
                                                   SchedBoundary *Zone) {
  // Initialize the candidate if needed.
  if (!Cand.isValid()) {
    TryCand.Reason = FirstValid;
    return true;
  }

  // Bias PhysReg Defs and copies to their uses and defined respectively.
  if (tryGreater(biasPhysReg(TryCand.SU, TryCand.AtTop),
                 biasPhysReg(Cand.SU, Cand.AtTop), TryCand, Cand, PhysReg))
    return TryCand.Reason != NoCand;

  // Avoid exceeding the target's limit.
  if (DAG->isTrackingPressure() &&
      tryPressure(TryCand.RPDelta.Excess, Cand.RPDelta.Excess, TryCand, Cand,
                  RegExcess, TRI, DAG->MF))
    return TryCand.Reason != NoCand;

  // We only compare a subset of features when comparing nodes between
  // Top and Bottom boundary. Some properties are simply incomparable, in many
  // other instances we should only override the other boundary if something
  // is a clear good pick on one boundary. Skip heuristics that are more
  // "tie-breaking" in nature.
  bool SameBoundary = Zone != nullptr;
  if (SameBoundary) {
    // Compare candidates by the stall they would introduce if
    // scheduled in the current cycle.
    if (tryEffectiveStall(Cand, TryCand, *Zone))
      return TryCand.Reason != NoCand;

    Heurs.sortHWUIResources();
    if (Heurs.tryCriticalResource(TryCand, Cand, Zone)) {
      LastAMDGPUReason = AMDGPUSchedReason::CritResourceBalance;
      return TryCand.Reason != NoCand;
    }

    if (Heurs.tryCriticalResourceDependency(TryCand, Cand, Zone)) {
      LastAMDGPUReason = AMDGPUSchedReason::CritResourceDep;
      return TryCand.Reason != NoCand;
    }
  }

  // Keep clustered nodes together to encourage downstream peephole
  // optimizations which may reduce resource requirements.
  //
  // This is a best effort to set things up for a post-RA pass. Optimizations
  // like generating loads of multiple registers should ideally be done within
  // the scheduler pass by combining the loads during DAG postprocessing.
  unsigned CandZoneCluster = Cand.AtTop ? TopClusterID : BotClusterID;
  unsigned TryCandZoneCluster = TryCand.AtTop ? TopClusterID : BotClusterID;
  bool CandIsClusterSucc =
      isTheSameCluster(CandZoneCluster, Cand.SU->ParentClusterIdx);
  bool TryCandIsClusterSucc =
      isTheSameCluster(TryCandZoneCluster, TryCand.SU->ParentClusterIdx);

  if (tryGreater(TryCandIsClusterSucc, CandIsClusterSucc, TryCand, Cand,
                 Cluster))
    return TryCand.Reason != NoCand;

  if (SameBoundary) {
    // Weak edges are for clustering and other constraints.
    if (tryLess(getWeakLeft(TryCand.SU, TryCand.AtTop),
                getWeakLeft(Cand.SU, Cand.AtTop), TryCand, Cand, Weak))
      return TryCand.Reason != NoCand;
  }

  // Avoid increasing the max pressure of the entire region.
  if (DAG->isTrackingPressure() &&
      tryPressure(TryCand.RPDelta.CurrentMax, Cand.RPDelta.CurrentMax, TryCand,
                  Cand, RegMax, TRI, DAG->MF))
    return TryCand.Reason != NoCand;

  if (SameBoundary) {
    // Avoid serializing long latency dependence chains.
    // For acyclic path limited loops, latency was already checked above.
    if (!RegionPolicy.DisableLatencyHeuristic && TryCand.Policy.ReduceLatency &&
        !Rem.IsAcyclicLatencyLimited && tryLatency(TryCand, Cand, *Zone))
      return TryCand.Reason != NoCand;

    // Fall through to original instruction order.
    if ((Zone->isTop() && TryCand.SU->NodeNum < Cand.SU->NodeNum) ||
        (!Zone->isTop() && TryCand.SU->NodeNum > Cand.SU->NodeNum)) {
      TryCand.Reason = NodeOrder;
      return true;
    }
  }

  return false;
}

bool AMDGPUCoExecSchedStrategy::tryEffectiveStall(SchedCandidate &Cand,
                                                  SchedCandidate &TryCand,
                                                  SchedBoundary &Zone) const {
  // Treat structural and latency stalls as a single scheduling cost for the
  // current cycle.
  unsigned TryStructStall = getStructuralStallCycles(Zone, TryCand.SU);
  unsigned TryLatencyStall = Zone.getLatencyStallCycles(TryCand.SU);
  unsigned TryEffectiveStall = std::max(TryStructStall, TryLatencyStall);

  unsigned CandStructStall = getStructuralStallCycles(Zone, Cand.SU);
  unsigned CandLatencyStall = Zone.getLatencyStallCycles(Cand.SU);
  unsigned CandEffectiveStall = std::max(CandStructStall, CandLatencyStall);

  LLVM_DEBUG(if (TryEffectiveStall || CandEffectiveStall) {
    dbgs() << "Effective stalls: try=" << TryEffectiveStall
           << " (struct=" << TryStructStall << ", lat=" << TryLatencyStall
           << ") cand=" << CandEffectiveStall << " (struct=" << CandStructStall
           << ", lat=" << CandLatencyStall << ")\n";
  });

  return tryLess(TryEffectiveStall, CandEffectiveStall, TryCand, Cand, Stall);
}

ScheduleDAGInstrs *
llvm::createGCNCoExecMachineScheduler(MachineSchedContext *C) {
  LLVM_DEBUG(dbgs() << "AMDGPU coexec preRA scheduler selected for "
                    << C->MF->getName() << '\n');
  return new GCNScheduleDAGMILive(
      C, std::make_unique<AMDGPUCoExecSchedStrategy>(C));
}

ScheduleDAGInstrs *
llvm::createGCNCoExecPostMachineScheduler(MachineSchedContext *C) {
  LLVM_DEBUG(dbgs() << "AMDGPU coexec postRA scheduler selected (nop) for "
                    << C->MF->getName() << '\n');
  return new GCNNoopPostScheduleDAG(C);
}
