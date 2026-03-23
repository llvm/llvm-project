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

AMDGPUCoExecSchedStrategy::AMDGPUCoExecSchedStrategy(
    const MachineSchedContext *C)
    : GCNSchedStrategy(C) {
  SchedStages.push_back(GCNSchedStageID::ILPInitialSchedule);
  SchedStages.push_back(GCNSchedStageID::PreRARematerialize);
  // Use more accurate GCN pressure trackers.
  UseGCNTrackers = true;
}

void AMDGPUCoExecSchedStrategy::initPolicy(MachineBasicBlock::iterator Begin,
                                           MachineBasicBlock::iterator End,
                                           unsigned NumRegionInstrs) {
  GCNSchedStrategy::initPolicy(Begin, End, NumRegionInstrs);
  assert((PreRADirection == MISched::Unspecified ||
          PreRADirection == MISched::TopDown) &&
         "coexec scheduler only supports top-down scheduling");
  RegionPolicy.OnlyTopDown = true;
  RegionPolicy.OnlyBottomUp = false;
}

void AMDGPUCoExecSchedStrategy::initialize(ScheduleDAGMI *DAG) {
  // Coexecution scheduling strategy is only done top-down to support new
  // resource balancing heuristics.
  RegionPolicy.OnlyTopDown = true;
  RegionPolicy.OnlyBottomUp = false;

  GCNSchedStrategy::initialize(DAG);
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
    }
    IsTopNode = true;
  } while (SU->isScheduled);

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
      tryCandidate(Cand, TryCand, ZoneArg);
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

bool AMDGPUCoExecSchedStrategy::tryCandidate(SchedCandidate &Cand,
                                             SchedCandidate &TryCand,
                                             SchedBoundary *Zone) const {
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
    // For loops that are acyclic path limited, aggressively schedule for
    // latency. Within an single cycle, whenever CurrMOps > 0, allow normal
    // heuristics to take precedence.
    if (Rem.IsAcyclicLatencyLimited && !Zone->getCurrMOps() &&
        tryLatency(TryCand, Cand, *Zone))
      return TryCand.Reason != NoCand;

    // Prioritize instructions that read unbuffered resources by stall cycles.
    if (tryLess(Zone->getLatencyStallCycles(TryCand.SU),
                Zone->getLatencyStallCycles(Cand.SU), TryCand, Cand, Stall))
      return TryCand.Reason != NoCand;
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
    // Avoid critical resource consumption and balance the schedule.
    TryCand.initResourceDelta(DAG, SchedModel);
    if (tryLess(TryCand.ResDelta.CritResources, Cand.ResDelta.CritResources,
                TryCand, Cand, ResourceReduce))
      return TryCand.Reason != NoCand;
    if (tryGreater(TryCand.ResDelta.DemandedResources,
                   Cand.ResDelta.DemandedResources, TryCand, Cand,
                   ResourceDemand))
      return TryCand.Reason != NoCand;

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

ScheduleDAGInstrs *
llvm::createGCNCoExecMachineScheduler(MachineSchedContext *C) {
  LLVM_DEBUG(dbgs() << "AMDGPU coexec preRA scheduler selected for "
                    << C->MF->getName() << '\n');
  return new GCNScheduleDAGMILive(
      C, std::make_unique<AMDGPUCoExecSchedStrategy>(C));
}

ScheduleDAGInstrs *
llvm::createGCNNoopPostMachineScheduler(MachineSchedContext *C) {
  LLVM_DEBUG(dbgs() << "AMDGPU nop postRA scheduler selected for "
                    << C->MF->getName() << '\n');
  return new GCNNoopPostScheduleDAG(C);
}
