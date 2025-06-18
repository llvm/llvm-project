//===-- GCNSchedStrategy.cpp - GCN Scheduler Strategy ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This contains a MachineSchedStrategy implementation for maximizing wave
/// occupancy on GCN hardware.
///
/// This pass will apply multiple scheduling stages to the same function.
/// Regions are first recorded in GCNScheduleDAGMILive::schedule. The actual
/// entry point for the scheduling of those regions is
/// GCNScheduleDAGMILive::runSchedStages.

/// Generally, the reason for having multiple scheduling stages is to account
/// for the kernel-wide effect of register usage on occupancy.  Usually, only a
/// few scheduling regions will have register pressure high enough to limit
/// occupancy for the kernel, so constraints can be relaxed to improve ILP in
/// other regions.
///
//===----------------------------------------------------------------------===//

#include "GCNSchedStrategy.h"
#include "AMDGPUIGroupLP.h"
#include "GCNRegPressure.h"
#include "SIMachineFunctionInfo.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/RegisterClassInfo.h"
#include "llvm/MC/LaneBitmask.h"
#include "llvm/Support/ErrorHandling.h"

#define DEBUG_TYPE "machine-scheduler"

using namespace llvm;

static cl::opt<bool> DisableUnclusterHighRP(
    "amdgpu-disable-unclustered-high-rp-reschedule", cl::Hidden,
    cl::desc("Disable unclustered high register pressure "
             "reduction scheduling stage."),
    cl::init(false));

static cl::opt<bool> DisableClusteredLowOccupancy(
    "amdgpu-disable-clustered-low-occupancy-reschedule", cl::Hidden,
    cl::desc("Disable clustered low occupancy "
             "rescheduling for ILP scheduling stage."),
    cl::init(false));

static cl::opt<unsigned> ScheduleMetricBias(
    "amdgpu-schedule-metric-bias", cl::Hidden,
    cl::desc(
        "Sets the bias which adds weight to occupancy vs latency. Set it to "
        "100 to chase the occupancy only."),
    cl::init(10));

static cl::opt<bool>
    RelaxedOcc("amdgpu-schedule-relaxed-occupancy", cl::Hidden,
               cl::desc("Relax occupancy targets for kernels which are memory "
                        "bound (amdgpu-membound-threshold), or "
                        "Wave Limited (amdgpu-limit-wave-threshold)."),
               cl::init(false));

static cl::opt<bool> GCNTrackers(
    "amdgpu-use-amdgpu-trackers", cl::Hidden,
    cl::desc("Use the AMDGPU specific RPTrackers during scheduling"),
    cl::init(false));

const unsigned ScheduleMetrics::ScaleFactor = 100;

GCNSchedStrategy::GCNSchedStrategy(const MachineSchedContext *C)
    : GenericScheduler(C), TargetOccupancy(0), MF(nullptr),
      DownwardTracker(*C->LIS), UpwardTracker(*C->LIS), HasHighPressure(false) {
}

void GCNSchedStrategy::initialize(ScheduleDAGMI *DAG) {
  GenericScheduler::initialize(DAG);

  MF = &DAG->MF;

  const GCNSubtarget &ST = MF->getSubtarget<GCNSubtarget>();

  SGPRExcessLimit =
      Context->RegClassInfo->getNumAllocatableRegs(&AMDGPU::SGPR_32RegClass);
  VGPRExcessLimit =
      Context->RegClassInfo->getNumAllocatableRegs(&AMDGPU::VGPR_32RegClass);

  SIMachineFunctionInfo &MFI = *MF->getInfo<SIMachineFunctionInfo>();
  // Set the initial TargetOccupnacy to the maximum occupancy that we can
  // achieve for this function. This effectively sets a lower bound on the
  // 'Critical' register limits in the scheduler.
  // Allow for lower occupancy targets if kernel is wave limited or memory
  // bound, and using the relaxed occupancy feature.
  TargetOccupancy =
      RelaxedOcc ? MFI.getMinAllowedOccupancy() : MFI.getOccupancy();
  SGPRCriticalLimit =
      std::min(ST.getMaxNumSGPRs(TargetOccupancy, true), SGPRExcessLimit);

  if (!KnownExcessRP) {
    VGPRCriticalLimit =
        std::min(ST.getMaxNumVGPRs(TargetOccupancy), VGPRExcessLimit);
  } else {
    // This is similar to ST.getMaxNumVGPRs(TargetOccupancy) result except
    // returns a reasonably small number for targets with lots of VGPRs, such
    // as GFX10 and GFX11.
    LLVM_DEBUG(dbgs() << "Region is known to spill, use alternative "
                         "VGPRCriticalLimit calculation method.\n");

    unsigned Granule = AMDGPU::IsaInfo::getVGPRAllocGranule(&ST);
    unsigned Addressable = AMDGPU::IsaInfo::getAddressableNumVGPRs(&ST);
    unsigned VGPRBudget = alignDown(Addressable / TargetOccupancy, Granule);
    VGPRBudget = std::max(VGPRBudget, Granule);
    VGPRCriticalLimit = std::min(VGPRBudget, VGPRExcessLimit);
  }

  // Subtract error margin and bias from register limits and avoid overflow.
  SGPRCriticalLimit -= std::min(SGPRLimitBias + ErrorMargin, SGPRCriticalLimit);
  VGPRCriticalLimit -= std::min(VGPRLimitBias + ErrorMargin, VGPRCriticalLimit);
  SGPRExcessLimit -= std::min(SGPRLimitBias + ErrorMargin, SGPRExcessLimit);
  VGPRExcessLimit -= std::min(VGPRLimitBias + ErrorMargin, VGPRExcessLimit);

  LLVM_DEBUG(dbgs() << "VGPRCriticalLimit = " << VGPRCriticalLimit
                    << ", VGPRExcessLimit = " << VGPRExcessLimit
                    << ", SGPRCriticalLimit = " << SGPRCriticalLimit
                    << ", SGPRExcessLimit = " << SGPRExcessLimit << "\n\n");
}

/// Checks whether \p SU can use the cached DAG pressure diffs to compute the
/// current register pressure.
///
/// This works for the common case, but it has a few exceptions that have been
/// observed through trial and error:
///   - Explicit physical register operands
///   - Subregister definitions
///
/// In both of those cases, PressureDiff doesn't represent the actual pressure,
/// and querying LiveIntervals through the RegPressureTracker is needed to get
/// an accurate value.
///
/// We should eventually only use PressureDiff for maximum performance, but this
/// already allows 80% of SUs to take the fast path without changing scheduling
/// at all. Further changes would either change scheduling, or require a lot
/// more logic to recover an accurate pressure estimate from the PressureDiffs.
static bool canUsePressureDiffs(const SUnit &SU) {
  if (!SU.isInstr())
    return false;

  // Cannot use pressure diffs for subregister defs or with physregs, it's
  // imprecise in both cases.
  for (const auto &Op : SU.getInstr()->operands()) {
    if (!Op.isReg() || Op.isImplicit())
      continue;
    if (Op.getReg().isPhysical() ||
        (Op.isDef() && Op.getSubReg() != AMDGPU::NoSubRegister))
      return false;
  }
  return true;
}

static void getRegisterPressures(
    bool AtTop, const RegPressureTracker &RPTracker, SUnit *SU,
    std::vector<unsigned> &Pressure, std::vector<unsigned> &MaxPressure,
    GCNDownwardRPTracker &DownwardTracker, GCNUpwardRPTracker &UpwardTracker,
    ScheduleDAGMI *DAG, const SIRegisterInfo *SRI) {
  // getDownwardPressure() and getUpwardPressure() make temporary changes to
  // the tracker, so we need to pass those function a non-const copy.
  RegPressureTracker &TempTracker = const_cast<RegPressureTracker &>(RPTracker);
  if (!GCNTrackers) {
    AtTop
        ? TempTracker.getDownwardPressure(SU->getInstr(), Pressure, MaxPressure)
        : TempTracker.getUpwardPressure(SU->getInstr(), Pressure, MaxPressure);

    return;
  }

  // GCNTrackers
  Pressure.resize(4, 0);
  MachineInstr *MI = SU->getInstr();
  GCNRegPressure NewPressure;
  if (AtTop) {
    GCNDownwardRPTracker TempDownwardTracker(DownwardTracker);
    NewPressure = TempDownwardTracker.bumpDownwardPressure(MI, SRI);
  } else {
    GCNUpwardRPTracker TempUpwardTracker(UpwardTracker);
    TempUpwardTracker.recede(*MI);
    NewPressure = TempUpwardTracker.getPressure();
  }
  Pressure[AMDGPU::RegisterPressureSets::SReg_32] = NewPressure.getSGPRNum();
  Pressure[AMDGPU::RegisterPressureSets::VGPR_32] =
      NewPressure.getArchVGPRNum();
  Pressure[AMDGPU::RegisterPressureSets::AGPR_32] = NewPressure.getAGPRNum();
}

void GCNSchedStrategy::initCandidate(SchedCandidate &Cand, SUnit *SU,
                                     bool AtTop,
                                     const RegPressureTracker &RPTracker,
                                     const SIRegisterInfo *SRI,
                                     unsigned SGPRPressure,
                                     unsigned VGPRPressure, bool IsBottomUp) {
  Cand.SU = SU;
  Cand.AtTop = AtTop;

  if (!DAG->isTrackingPressure())
    return;

  Pressure.clear();
  MaxPressure.clear();

  // We try to use the cached PressureDiffs in the ScheduleDAG whenever
  // possible over querying the RegPressureTracker.
  //
  // RegPressureTracker will make a lot of LIS queries which are very
  // expensive, it is considered a slow function in this context.
  //
  // PressureDiffs are precomputed and cached, and getPressureDiff is just a
  // trivial lookup into an array. It is pretty much free.
  //
  // In EXPENSIVE_CHECKS, we always query RPTracker to verify the results of
  // PressureDiffs.
  if (AtTop || !canUsePressureDiffs(*SU) || GCNTrackers) {
    getRegisterPressures(AtTop, RPTracker, SU, Pressure, MaxPressure,
                         DownwardTracker, UpwardTracker, DAG, SRI);
  } else {
    // Reserve 4 slots.
    Pressure.resize(4, 0);
    Pressure[AMDGPU::RegisterPressureSets::SReg_32] = SGPRPressure;
    Pressure[AMDGPU::RegisterPressureSets::VGPR_32] = VGPRPressure;

    for (const auto &Diff : DAG->getPressureDiff(SU)) {
      if (!Diff.isValid())
        continue;
      // PressureDiffs is always bottom-up so if we're working top-down we need
      // to invert its sign.
      Pressure[Diff.getPSet()] +=
          (IsBottomUp ? Diff.getUnitInc() : -Diff.getUnitInc());
    }

#ifdef EXPENSIVE_CHECKS
    std::vector<unsigned> CheckPressure, CheckMaxPressure;
    getRegisterPressures(AtTop, RPTracker, SU, CheckPressure, CheckMaxPressure,
                         DownwardTracker, UpwardTracker, DAG, SRI);
    if (Pressure[AMDGPU::RegisterPressureSets::SReg_32] !=
            CheckPressure[AMDGPU::RegisterPressureSets::SReg_32] ||
        Pressure[AMDGPU::RegisterPressureSets::VGPR_32] !=
            CheckPressure[AMDGPU::RegisterPressureSets::VGPR_32]) {
      errs() << "Register Pressure is inaccurate when calculated through "
                "PressureDiff\n"
             << "SGPR got " << Pressure[AMDGPU::RegisterPressureSets::SReg_32]
             << ", expected "
             << CheckPressure[AMDGPU::RegisterPressureSets::SReg_32] << "\n"
             << "VGPR got " << Pressure[AMDGPU::RegisterPressureSets::VGPR_32]
             << ", expected "
             << CheckPressure[AMDGPU::RegisterPressureSets::VGPR_32] << "\n";
      report_fatal_error("inaccurate register pressure calculation");
    }
#endif
  }

  unsigned NewSGPRPressure = Pressure[AMDGPU::RegisterPressureSets::SReg_32];
  unsigned NewVGPRPressure = Pressure[AMDGPU::RegisterPressureSets::VGPR_32];

  // If two instructions increase the pressure of different register sets
  // by the same amount, the generic scheduler will prefer to schedule the
  // instruction that increases the set with the least amount of registers,
  // which in our case would be SGPRs.  This is rarely what we want, so
  // when we report excess/critical register pressure, we do it either
  // only for VGPRs or only for SGPRs.

  // FIXME: Better heuristics to determine whether to prefer SGPRs or VGPRs.
  const unsigned MaxVGPRPressureInc = 16;
  bool ShouldTrackVGPRs = VGPRPressure + MaxVGPRPressureInc >= VGPRExcessLimit;
  bool ShouldTrackSGPRs = !ShouldTrackVGPRs && SGPRPressure >= SGPRExcessLimit;

  // FIXME: We have to enter REG-EXCESS before we reach the actual threshold
  // to increase the likelihood we don't go over the limits.  We should improve
  // the analysis to look through dependencies to find the path with the least
  // register pressure.

  // We only need to update the RPDelta for instructions that increase register
  // pressure. Instructions that decrease or keep reg pressure the same will be
  // marked as RegExcess in tryCandidate() when they are compared with
  // instructions that increase the register pressure.
  if (ShouldTrackVGPRs && NewVGPRPressure >= VGPRExcessLimit) {
    HasHighPressure = true;
    Cand.RPDelta.Excess = PressureChange(AMDGPU::RegisterPressureSets::VGPR_32);
    Cand.RPDelta.Excess.setUnitInc(NewVGPRPressure - VGPRExcessLimit);
  }

  if (ShouldTrackSGPRs && NewSGPRPressure >= SGPRExcessLimit) {
    HasHighPressure = true;
    Cand.RPDelta.Excess = PressureChange(AMDGPU::RegisterPressureSets::SReg_32);
    Cand.RPDelta.Excess.setUnitInc(NewSGPRPressure - SGPRExcessLimit);
  }

  // Register pressure is considered 'CRITICAL' if it is approaching a value
  // that would reduce the wave occupancy for the execution unit.  When
  // register pressure is 'CRITICAL', increasing SGPR and VGPR pressure both
  // has the same cost, so we don't need to prefer one over the other.

  int SGPRDelta = NewSGPRPressure - SGPRCriticalLimit;
  int VGPRDelta = NewVGPRPressure - VGPRCriticalLimit;

  if (SGPRDelta >= 0 || VGPRDelta >= 0) {
    HasHighPressure = true;
    if (SGPRDelta > VGPRDelta) {
      Cand.RPDelta.CriticalMax =
          PressureChange(AMDGPU::RegisterPressureSets::SReg_32);
      Cand.RPDelta.CriticalMax.setUnitInc(SGPRDelta);
    } else {
      Cand.RPDelta.CriticalMax =
          PressureChange(AMDGPU::RegisterPressureSets::VGPR_32);
      Cand.RPDelta.CriticalMax.setUnitInc(VGPRDelta);
    }
  }
}

// This function is mostly cut and pasted from
// GenericScheduler::pickNodeFromQueue()
void GCNSchedStrategy::pickNodeFromQueue(SchedBoundary &Zone,
                                         const CandPolicy &ZonePolicy,
                                         const RegPressureTracker &RPTracker,
                                         SchedCandidate &Cand,
                                         bool IsBottomUp) {
  const SIRegisterInfo *SRI = static_cast<const SIRegisterInfo *>(TRI);
  ArrayRef<unsigned> Pressure = RPTracker.getRegSetPressureAtPos();
  unsigned SGPRPressure = 0;
  unsigned VGPRPressure = 0;
  if (DAG->isTrackingPressure()) {
    if (!GCNTrackers) {
      SGPRPressure = Pressure[AMDGPU::RegisterPressureSets::SReg_32];
      VGPRPressure = Pressure[AMDGPU::RegisterPressureSets::VGPR_32];
    } else {
      GCNRPTracker *T = IsBottomUp
                            ? static_cast<GCNRPTracker *>(&UpwardTracker)
                            : static_cast<GCNRPTracker *>(&DownwardTracker);
      SGPRPressure = T->getPressure().getSGPRNum();
      VGPRPressure = T->getPressure().getArchVGPRNum();
    }
  }
  ReadyQueue &Q = Zone.Available;
  for (SUnit *SU : Q) {

    SchedCandidate TryCand(ZonePolicy);
    initCandidate(TryCand, SU, Zone.isTop(), RPTracker, SRI, SGPRPressure,
                  VGPRPressure, IsBottomUp);
    // Pass SchedBoundary only when comparing nodes from the same boundary.
    SchedBoundary *ZoneArg = Cand.AtTop == TryCand.AtTop ? &Zone : nullptr;
    tryCandidate(Cand, TryCand, ZoneArg);
    if (TryCand.Reason != NoCand) {
      // Initialize resource delta if needed in case future heuristics query it.
      if (TryCand.ResDelta == SchedResourceDelta())
        TryCand.initResourceDelta(Zone.DAG, SchedModel);
      Cand.setBest(TryCand);
      LLVM_DEBUG(traceCandidate(Cand));
    }
  }
}

// This function is mostly cut and pasted from
// GenericScheduler::pickNodeBidirectional()
SUnit *GCNSchedStrategy::pickNodeBidirectional(bool &IsTopNode) {
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
  // Set the bottom-up policy based on the state of the current bottom zone and
  // the instructions outside the zone, including the top zone.
  CandPolicy BotPolicy;
  setPolicy(BotPolicy, /*IsPostRA=*/false, Bot, &Top);
  // Set the top-down policy based on the state of the current top zone and
  // the instructions outside the zone, including the bottom zone.
  CandPolicy TopPolicy;
  setPolicy(TopPolicy, /*IsPostRA=*/false, Top, &Bot);

  // See if BotCand is still valid (because we previously scheduled from Top).
  LLVM_DEBUG(dbgs() << "Picking from Bot:\n");
  if (!BotCand.isValid() || BotCand.SU->isScheduled ||
      BotCand.Policy != BotPolicy) {
    BotCand.reset(CandPolicy());
    pickNodeFromQueue(Bot, BotPolicy, DAG->getBotRPTracker(), BotCand,
                      /*IsBottomUp=*/true);
    assert(BotCand.Reason != NoCand && "failed to find the first candidate");
  } else {
    LLVM_DEBUG(traceCandidate(BotCand));
#ifndef NDEBUG
    if (VerifyScheduling) {
      SchedCandidate TCand;
      TCand.reset(CandPolicy());
      pickNodeFromQueue(Bot, BotPolicy, DAG->getBotRPTracker(), TCand,
                        /*IsBottomUp=*/true);
      assert(TCand.SU == BotCand.SU &&
             "Last pick result should correspond to re-picking right now");
    }
#endif
  }

  // Check if the top Q has a better candidate.
  LLVM_DEBUG(dbgs() << "Picking from Top:\n");
  if (!TopCand.isValid() || TopCand.SU->isScheduled ||
      TopCand.Policy != TopPolicy) {
    TopCand.reset(CandPolicy());
    pickNodeFromQueue(Top, TopPolicy, DAG->getTopRPTracker(), TopCand,
                      /*IsBottomUp=*/false);
    assert(TopCand.Reason != NoCand && "failed to find the first candidate");
  } else {
    LLVM_DEBUG(traceCandidate(TopCand));
#ifndef NDEBUG
    if (VerifyScheduling) {
      SchedCandidate TCand;
      TCand.reset(CandPolicy());
      pickNodeFromQueue(Top, TopPolicy, DAG->getTopRPTracker(), TCand,
                        /*IsBottomUp=*/false);
      assert(TCand.SU == TopCand.SU &&
             "Last pick result should correspond to re-picking right now");
    }
#endif
  }

  // Pick best from BotCand and TopCand.
  LLVM_DEBUG(dbgs() << "Top Cand: "; traceCandidate(TopCand);
             dbgs() << "Bot Cand: "; traceCandidate(BotCand););
  SchedCandidate Cand = BotCand;
  TopCand.Reason = NoCand;
  tryCandidate(Cand, TopCand, nullptr);
  if (TopCand.Reason != NoCand) {
    Cand.setBest(TopCand);
  }
  LLVM_DEBUG(dbgs() << "Picking: "; traceCandidate(Cand););

  IsTopNode = Cand.AtTop;
  return Cand.SU;
}

// This function is mostly cut and pasted from
// GenericScheduler::pickNode()
SUnit *GCNSchedStrategy::pickNode(bool &IsTopNode) {
  if (DAG->top() == DAG->bottom()) {
    assert(Top.Available.empty() && Top.Pending.empty() &&
           Bot.Available.empty() && Bot.Pending.empty() && "ReadyQ garbage");
    return nullptr;
  }
  SUnit *SU;
  do {
    if (RegionPolicy.OnlyTopDown) {
      SU = Top.pickOnlyChoice();
      if (!SU) {
        CandPolicy NoPolicy;
        TopCand.reset(NoPolicy);
        pickNodeFromQueue(Top, NoPolicy, DAG->getTopRPTracker(), TopCand,
                          /*IsBottomUp=*/false);
        assert(TopCand.Reason != NoCand && "failed to find a candidate");
        SU = TopCand.SU;
      }
      IsTopNode = true;
    } else if (RegionPolicy.OnlyBottomUp) {
      SU = Bot.pickOnlyChoice();
      if (!SU) {
        CandPolicy NoPolicy;
        BotCand.reset(NoPolicy);
        pickNodeFromQueue(Bot, NoPolicy, DAG->getBotRPTracker(), BotCand,
                          /*IsBottomUp=*/true);
        assert(BotCand.Reason != NoCand && "failed to find a candidate");
        SU = BotCand.SU;
      }
      IsTopNode = false;
    } else {
      SU = pickNodeBidirectional(IsTopNode);
    }
  } while (SU->isScheduled);

  if (SU->isTopReady())
    Top.removeReady(SU);
  if (SU->isBottomReady())
    Bot.removeReady(SU);

  LLVM_DEBUG(dbgs() << "Scheduling SU(" << SU->NodeNum << ") "
                    << *SU->getInstr());
  return SU;
}

void GCNSchedStrategy::schedNode(SUnit *SU, bool IsTopNode) {
  if (GCNTrackers) {
    MachineInstr *MI = SU->getInstr();
    IsTopNode ? (void)DownwardTracker.advance(MI, false)
              : UpwardTracker.recede(*MI);
  }

  return GenericScheduler::schedNode(SU, IsTopNode);
}

GCNSchedStageID GCNSchedStrategy::getCurrentStage() {
  assert(CurrentStage && CurrentStage != SchedStages.end());
  return *CurrentStage;
}

bool GCNSchedStrategy::advanceStage() {
  assert(CurrentStage != SchedStages.end());
  if (!CurrentStage)
    CurrentStage = SchedStages.begin();
  else
    CurrentStage++;

  return CurrentStage != SchedStages.end();
}

bool GCNSchedStrategy::hasNextStage() const {
  assert(CurrentStage);
  return std::next(CurrentStage) != SchedStages.end();
}

GCNSchedStageID GCNSchedStrategy::getNextStage() const {
  assert(CurrentStage && std::next(CurrentStage) != SchedStages.end());
  return *std::next(CurrentStage);
}

GCNMaxOccupancySchedStrategy::GCNMaxOccupancySchedStrategy(
    const MachineSchedContext *C, bool IsLegacyScheduler)
    : GCNSchedStrategy(C) {
  SchedStages.push_back(GCNSchedStageID::OccInitialSchedule);
  SchedStages.push_back(GCNSchedStageID::UnclusteredHighRPReschedule);
  SchedStages.push_back(GCNSchedStageID::ClusteredLowOccupancyReschedule);
  SchedStages.push_back(GCNSchedStageID::PreRARematerialize);
  GCNTrackers = GCNTrackers & !IsLegacyScheduler;
}

GCNMaxILPSchedStrategy::GCNMaxILPSchedStrategy(const MachineSchedContext *C)
    : GCNSchedStrategy(C) {
  SchedStages.push_back(GCNSchedStageID::ILPInitialSchedule);
}

bool GCNMaxILPSchedStrategy::tryCandidate(SchedCandidate &Cand,
                                          SchedCandidate &TryCand,
                                          SchedBoundary *Zone) const {
  // Initialize the candidate if needed.
  if (!Cand.isValid()) {
    TryCand.Reason = NodeOrder;
    return true;
  }

  // Avoid spilling by exceeding the register limit.
  if (DAG->isTrackingPressure() &&
      tryPressure(TryCand.RPDelta.Excess, Cand.RPDelta.Excess, TryCand, Cand,
                  RegExcess, TRI, DAG->MF))
    return TryCand.Reason != NoCand;

  // Bias PhysReg Defs and copies to their uses and defined respectively.
  if (tryGreater(biasPhysReg(TryCand.SU, TryCand.AtTop),
                 biasPhysReg(Cand.SU, Cand.AtTop), TryCand, Cand, PhysReg))
    return TryCand.Reason != NoCand;

  bool SameBoundary = Zone != nullptr;
  if (SameBoundary) {
    // Prioritize instructions that read unbuffered resources by stall cycles.
    if (tryLess(Zone->getLatencyStallCycles(TryCand.SU),
                Zone->getLatencyStallCycles(Cand.SU), TryCand, Cand, Stall))
      return TryCand.Reason != NoCand;

    // Avoid critical resource consumption and balance the schedule.
    TryCand.initResourceDelta(DAG, SchedModel);
    if (tryLess(TryCand.ResDelta.CritResources, Cand.ResDelta.CritResources,
                TryCand, Cand, ResourceReduce))
      return TryCand.Reason != NoCand;
    if (tryGreater(TryCand.ResDelta.DemandedResources,
                   Cand.ResDelta.DemandedResources, TryCand, Cand,
                   ResourceDemand))
      return TryCand.Reason != NoCand;

    // Unconditionally try to reduce latency.
    if (tryLatency(TryCand, Cand, *Zone))
      return TryCand.Reason != NoCand;

    // Weak edges are for clustering and other constraints.
    if (tryLess(getWeakLeft(TryCand.SU, TryCand.AtTop),
                getWeakLeft(Cand.SU, Cand.AtTop), TryCand, Cand, Weak))
      return TryCand.Reason != NoCand;
  }

  // Keep clustered nodes together to encourage downstream peephole
  // optimizations which may reduce resource requirements.
  //
  // This is a best effort to set things up for a post-RA pass. Optimizations
  // like generating loads of multiple registers should ideally be done within
  // the scheduler pass by combining the loads during DAG postprocessing.
  const ClusterInfo *CandCluster = Cand.AtTop ? TopCluster : BotCluster;
  const ClusterInfo *TryCandCluster = TryCand.AtTop ? TopCluster : BotCluster;
  if (tryGreater(TryCandCluster && TryCandCluster->contains(TryCand.SU),
                 CandCluster && CandCluster->contains(Cand.SU), TryCand, Cand,
                 Cluster))
    return TryCand.Reason != NoCand;

  // Avoid increasing the max critical pressure in the scheduled region.
  if (DAG->isTrackingPressure() &&
      tryPressure(TryCand.RPDelta.CriticalMax, Cand.RPDelta.CriticalMax,
                  TryCand, Cand, RegCritical, TRI, DAG->MF))
    return TryCand.Reason != NoCand;

  // Avoid increasing the max pressure of the entire region.
  if (DAG->isTrackingPressure() &&
      tryPressure(TryCand.RPDelta.CurrentMax, Cand.RPDelta.CurrentMax, TryCand,
                  Cand, RegMax, TRI, DAG->MF))
    return TryCand.Reason != NoCand;

  if (SameBoundary) {
    // Fall through to original instruction order.
    if ((Zone->isTop() && TryCand.SU->NodeNum < Cand.SU->NodeNum) ||
        (!Zone->isTop() && TryCand.SU->NodeNum > Cand.SU->NodeNum)) {
      TryCand.Reason = NodeOrder;
      return true;
    }
  }
  return false;
}

GCNMaxMemoryClauseSchedStrategy::GCNMaxMemoryClauseSchedStrategy(
    const MachineSchedContext *C)
    : GCNSchedStrategy(C) {
  SchedStages.push_back(GCNSchedStageID::MemoryClauseInitialSchedule);
}

/// GCNMaxMemoryClauseSchedStrategy tries best to clause memory instructions as
/// much as possible. This is achieved by:
//  1. Prioritize clustered operations before stall latency heuristic.
//  2. Prioritize long-latency-load before stall latency heuristic.
///
/// \param Cand provides the policy and current best candidate.
/// \param TryCand refers to the next SUnit candidate, otherwise uninitialized.
/// \param Zone describes the scheduled zone that we are extending, or nullptr
///             if Cand is from a different zone than TryCand.
/// \return \c true if TryCand is better than Cand (Reason is NOT NoCand)
bool GCNMaxMemoryClauseSchedStrategy::tryCandidate(SchedCandidate &Cand,
                                                   SchedCandidate &TryCand,
                                                   SchedBoundary *Zone) const {
  // Initialize the candidate if needed.
  if (!Cand.isValid()) {
    TryCand.Reason = NodeOrder;
    return true;
  }

  // Bias PhysReg Defs and copies to their uses and defined respectively.
  if (tryGreater(biasPhysReg(TryCand.SU, TryCand.AtTop),
                 biasPhysReg(Cand.SU, Cand.AtTop), TryCand, Cand, PhysReg))
    return TryCand.Reason != NoCand;

  if (DAG->isTrackingPressure()) {
    // Avoid exceeding the target's limit.
    if (tryPressure(TryCand.RPDelta.Excess, Cand.RPDelta.Excess, TryCand, Cand,
                    RegExcess, TRI, DAG->MF))
      return TryCand.Reason != NoCand;

    // Avoid increasing the max critical pressure in the scheduled region.
    if (tryPressure(TryCand.RPDelta.CriticalMax, Cand.RPDelta.CriticalMax,
                    TryCand, Cand, RegCritical, TRI, DAG->MF))
      return TryCand.Reason != NoCand;
  }

  // MaxMemoryClause-specific: We prioritize clustered instructions as we would
  // get more benefit from clausing these memory instructions.
  const ClusterInfo *CandCluster = Cand.AtTop ? TopCluster : BotCluster;
  const ClusterInfo *TryCandCluster = TryCand.AtTop ? TopCluster : BotCluster;
  if (tryGreater(TryCandCluster && TryCandCluster->contains(TryCand.SU),
                 CandCluster && CandCluster->contains(Cand.SU), TryCand, Cand,
                 Cluster))
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

    // MaxMemoryClause-specific: Prioritize long latency memory load
    // instructions in top-bottom order to hide more latency. The mayLoad check
    // is used to exclude store-like instructions, which we do not want to
    // scheduler them too early.
    bool TryMayLoad =
        TryCand.SU->isInstr() && TryCand.SU->getInstr()->mayLoad();
    bool CandMayLoad = Cand.SU->isInstr() && Cand.SU->getInstr()->mayLoad();

    if (TryMayLoad || CandMayLoad) {
      bool TryLongLatency =
          TryCand.SU->Latency > 10 * Cand.SU->Latency && TryMayLoad;
      bool CandLongLatency =
          10 * TryCand.SU->Latency < Cand.SU->Latency && CandMayLoad;

      if (tryGreater(Zone->isTop() ? TryLongLatency : CandLongLatency,
                     Zone->isTop() ? CandLongLatency : TryLongLatency, TryCand,
                     Cand, Stall))
        return TryCand.Reason != NoCand;
    }
    // Prioritize instructions that read unbuffered resources by stall cycles.
    if (tryLess(Zone->getLatencyStallCycles(TryCand.SU),
                Zone->getLatencyStallCycles(Cand.SU), TryCand, Cand, Stall))
      return TryCand.Reason != NoCand;
  }

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
    if (Zone->isTop() == (TryCand.SU->NodeNum < Cand.SU->NodeNum)) {
      assert(TryCand.SU->NodeNum != Cand.SU->NodeNum);
      TryCand.Reason = NodeOrder;
      return true;
    }
  }

  return false;
}

GCNScheduleDAGMILive::GCNScheduleDAGMILive(
    MachineSchedContext *C, std::unique_ptr<MachineSchedStrategy> S)
    : ScheduleDAGMILive(C, std::move(S)), ST(MF.getSubtarget<GCNSubtarget>()),
      MFI(*MF.getInfo<SIMachineFunctionInfo>()),
      StartingOccupancy(MFI.getOccupancy()), MinOccupancy(StartingOccupancy),
      RegionLiveOuts(this, /*IsLiveOut=*/true) {

  // We want regions with a single MI to be scheduled so that we can reason
  // about them correctly during scheduling stages that move MIs between regions
  // (e.g., rematerialization).
  ScheduleSingleMIRegions = true;
  LLVM_DEBUG(dbgs() << "Starting occupancy is " << StartingOccupancy << ".\n");
  if (RelaxedOcc) {
    MinOccupancy = std::min(MFI.getMinAllowedOccupancy(), StartingOccupancy);
    if (MinOccupancy != StartingOccupancy)
      LLVM_DEBUG(dbgs() << "Allowing Occupancy drops to " << MinOccupancy
                        << ".\n");
  }
}

std::unique_ptr<GCNSchedStage>
GCNScheduleDAGMILive::createSchedStage(GCNSchedStageID SchedStageID) {
  switch (SchedStageID) {
  case GCNSchedStageID::OccInitialSchedule:
    return std::make_unique<OccInitialScheduleStage>(SchedStageID, *this);
  case GCNSchedStageID::UnclusteredHighRPReschedule:
    return std::make_unique<UnclusteredHighRPStage>(SchedStageID, *this);
  case GCNSchedStageID::ClusteredLowOccupancyReschedule:
    return std::make_unique<ClusteredLowOccStage>(SchedStageID, *this);
  case GCNSchedStageID::PreRARematerialize:
    return std::make_unique<PreRARematStage>(SchedStageID, *this);
  case GCNSchedStageID::ILPInitialSchedule:
    return std::make_unique<ILPInitialScheduleStage>(SchedStageID, *this);
  case GCNSchedStageID::MemoryClauseInitialSchedule:
    return std::make_unique<MemoryClauseInitialScheduleStage>(SchedStageID,
                                                              *this);
  }

  llvm_unreachable("Unknown SchedStageID.");
}

void GCNScheduleDAGMILive::schedule() {
  // Collect all scheduling regions. The actual scheduling is performed in
  // GCNScheduleDAGMILive::finalizeSchedule.
  Regions.push_back(std::pair(RegionBegin, RegionEnd));
}

GCNRegPressure
GCNScheduleDAGMILive::getRealRegPressure(unsigned RegionIdx) const {
  GCNDownwardRPTracker RPTracker(*LIS);
  RPTracker.advance(begin(), end(), &LiveIns[RegionIdx]);
  return RPTracker.moveMaxPressure();
}

static MachineInstr *getLastMIForRegion(MachineBasicBlock::iterator RegionBegin,
                                        MachineBasicBlock::iterator RegionEnd) {
  auto REnd = RegionEnd == RegionBegin->getParent()->end()
                  ? std::prev(RegionEnd)
                  : RegionEnd;
  return &*skipDebugInstructionsBackward(REnd, RegionBegin);
}

void GCNScheduleDAGMILive::computeBlockPressure(unsigned RegionIdx,
                                                const MachineBasicBlock *MBB) {
  GCNDownwardRPTracker RPTracker(*LIS);

  // If the block has the only successor then live-ins of that successor are
  // live-outs of the current block. We can reuse calculated live set if the
  // successor will be sent to scheduling past current block.

  // However, due to the bug in LiveInterval analysis it may happen that two
  // predecessors of the same successor block have different lane bitmasks for
  // a live-out register. Workaround that by sticking to one-to-one relationship
  // i.e. one predecessor with one successor block.
  const MachineBasicBlock *OnlySucc = nullptr;
  if (MBB->succ_size() == 1) {
    auto *Candidate = *MBB->succ_begin();
    if (!Candidate->empty() && Candidate->pred_size() == 1) {
      SlotIndexes *Ind = LIS->getSlotIndexes();
      if (Ind->getMBBStartIdx(MBB) < Ind->getMBBStartIdx(Candidate))
        OnlySucc = Candidate;
    }
  }

  // Scheduler sends regions from the end of the block upwards.
  size_t CurRegion = RegionIdx;
  for (size_t E = Regions.size(); CurRegion != E; ++CurRegion)
    if (Regions[CurRegion].first->getParent() != MBB)
      break;
  --CurRegion;

  auto I = MBB->begin();
  auto LiveInIt = MBBLiveIns.find(MBB);
  auto &Rgn = Regions[CurRegion];
  auto *NonDbgMI = &*skipDebugInstructionsForward(Rgn.first, Rgn.second);
  if (LiveInIt != MBBLiveIns.end()) {
    auto LiveIn = std::move(LiveInIt->second);
    RPTracker.reset(*MBB->begin(), &LiveIn);
    MBBLiveIns.erase(LiveInIt);
  } else {
    I = Rgn.first;
    auto LRS = BBLiveInMap.lookup(NonDbgMI);
#ifdef EXPENSIVE_CHECKS
    assert(isEqual(getLiveRegsBefore(*NonDbgMI, *LIS), LRS));
#endif
    RPTracker.reset(*I, &LRS);
  }

  for (;;) {
    I = RPTracker.getNext();

    if (Regions[CurRegion].first == I || NonDbgMI == I) {
      LiveIns[CurRegion] = RPTracker.getLiveRegs();
      RPTracker.clearMaxPressure();
    }

    if (Regions[CurRegion].second == I) {
      Pressure[CurRegion] = RPTracker.moveMaxPressure();
      if (CurRegion-- == RegionIdx)
        break;
      auto &Rgn = Regions[CurRegion];
      NonDbgMI = &*skipDebugInstructionsForward(Rgn.first, Rgn.second);
    }
    RPTracker.advanceToNext();
    RPTracker.advanceBeforeNext();
  }

  if (OnlySucc) {
    if (I != MBB->end()) {
      RPTracker.advanceToNext();
      RPTracker.advance(MBB->end());
    }
    RPTracker.advanceBeforeNext();
    MBBLiveIns[OnlySucc] = RPTracker.moveLiveRegs();
  }
}

DenseMap<MachineInstr *, GCNRPTracker::LiveRegSet>
GCNScheduleDAGMILive::getRegionLiveInMap() const {
  assert(!Regions.empty());
  std::vector<MachineInstr *> RegionFirstMIs;
  RegionFirstMIs.reserve(Regions.size());
  auto I = Regions.rbegin(), E = Regions.rend();
  do {
    const MachineBasicBlock *MBB = I->first->getParent();
    auto *MI = &*skipDebugInstructionsForward(I->first, I->second);
    RegionFirstMIs.push_back(MI);
    do {
      ++I;
    } while (I != E && I->first->getParent() == MBB);
  } while (I != E);
  return getLiveRegMap(RegionFirstMIs, /*After=*/false, *LIS);
}

DenseMap<MachineInstr *, GCNRPTracker::LiveRegSet>
GCNScheduleDAGMILive::getRegionLiveOutMap() const {
  assert(!Regions.empty());
  std::vector<MachineInstr *> RegionLastMIs;
  RegionLastMIs.reserve(Regions.size());
  for (auto &[RegionBegin, RegionEnd] : reverse(Regions))
    RegionLastMIs.push_back(getLastMIForRegion(RegionBegin, RegionEnd));

  return getLiveRegMap(RegionLastMIs, /*After=*/true, *LIS);
}

void RegionPressureMap::buildLiveRegMap() {
  IdxToInstruction.clear();

  RegionLiveRegMap =
      IsLiveOut ? DAG->getRegionLiveOutMap() : DAG->getRegionLiveInMap();
  for (unsigned I = 0; I < DAG->Regions.size(); I++) {
    MachineInstr *RegionKey =
        IsLiveOut
            ? getLastMIForRegion(DAG->Regions[I].first, DAG->Regions[I].second)
            : &*DAG->Regions[I].first;
    IdxToInstruction[I] = RegionKey;
  }
}

void GCNScheduleDAGMILive::finalizeSchedule() {
  // Start actual scheduling here. This function is called by the base
  // MachineScheduler after all regions have been recorded by
  // GCNScheduleDAGMILive::schedule().
  LiveIns.resize(Regions.size());
  Pressure.resize(Regions.size());
  RegionsWithHighRP.resize(Regions.size());
  RegionsWithExcessRP.resize(Regions.size());
  RegionsWithMinOcc.resize(Regions.size());
  RegionsWithIGLPInstrs.resize(Regions.size());
  RegionsWithHighRP.reset();
  RegionsWithExcessRP.reset();
  RegionsWithMinOcc.reset();
  RegionsWithIGLPInstrs.reset();

  runSchedStages();
}

void GCNScheduleDAGMILive::runSchedStages() {
  LLVM_DEBUG(dbgs() << "All regions recorded, starting actual scheduling.\n");

  if (!Regions.empty()) {
    BBLiveInMap = getRegionLiveInMap();
    if (GCNTrackers)
      RegionLiveOuts.buildLiveRegMap();
  }

  GCNSchedStrategy &S = static_cast<GCNSchedStrategy &>(*SchedImpl);
  while (S.advanceStage()) {
    auto Stage = createSchedStage(S.getCurrentStage());
    if (!Stage->initGCNSchedStage())
      continue;

    for (auto Region : Regions) {
      RegionBegin = Region.first;
      RegionEnd = Region.second;
      // Setup for scheduling the region and check whether it should be skipped.
      if (!Stage->initGCNRegion()) {
        Stage->advanceRegion();
        exitRegion();
        continue;
      }

      if (GCNTrackers) {
        GCNDownwardRPTracker *DownwardTracker = S.getDownwardTracker();
        GCNUpwardRPTracker *UpwardTracker = S.getUpwardTracker();
        GCNRPTracker::LiveRegSet *RegionLiveIns =
            &LiveIns[Stage->getRegionIdx()];

        reinterpret_cast<GCNRPTracker *>(DownwardTracker)
            ->reset(MRI, *RegionLiveIns);
        reinterpret_cast<GCNRPTracker *>(UpwardTracker)
            ->reset(MRI, RegionLiveOuts.getLiveRegsForRegionIdx(
                             Stage->getRegionIdx()));
      }

      ScheduleDAGMILive::schedule();
      Stage->finalizeGCNRegion();
    }

    Stage->finalizeGCNSchedStage();
  }
}

#ifndef NDEBUG
raw_ostream &llvm::operator<<(raw_ostream &OS, const GCNSchedStageID &StageID) {
  switch (StageID) {
  case GCNSchedStageID::OccInitialSchedule:
    OS << "Max Occupancy Initial Schedule";
    break;
  case GCNSchedStageID::UnclusteredHighRPReschedule:
    OS << "Unclustered High Register Pressure Reschedule";
    break;
  case GCNSchedStageID::ClusteredLowOccupancyReschedule:
    OS << "Clustered Low Occupancy Reschedule";
    break;
  case GCNSchedStageID::PreRARematerialize:
    OS << "Pre-RA Rematerialize";
    break;
  case GCNSchedStageID::ILPInitialSchedule:
    OS << "Max ILP Initial Schedule";
    break;
  case GCNSchedStageID::MemoryClauseInitialSchedule:
    OS << "Max memory clause Initial Schedule";
    break;
  }

  return OS;
}
#endif

GCNSchedStage::GCNSchedStage(GCNSchedStageID StageID, GCNScheduleDAGMILive &DAG)
    : DAG(DAG), S(static_cast<GCNSchedStrategy &>(*DAG.SchedImpl)), MF(DAG.MF),
      MFI(DAG.MFI), ST(DAG.ST), StageID(StageID) {}

bool GCNSchedStage::initGCNSchedStage() {
  if (!DAG.LIS)
    return false;

  LLVM_DEBUG(dbgs() << "Starting scheduling stage: " << StageID << "\n");
  return true;
}

bool UnclusteredHighRPStage::initGCNSchedStage() {
  if (DisableUnclusterHighRP)
    return false;

  if (!GCNSchedStage::initGCNSchedStage())
    return false;

  if (DAG.RegionsWithHighRP.none() && DAG.RegionsWithExcessRP.none())
    return false;

  SavedMutations.swap(DAG.Mutations);
  DAG.addMutation(
      createIGroupLPDAGMutation(AMDGPU::SchedulingPhase::PreRAReentry));

  InitialOccupancy = DAG.MinOccupancy;
  // Aggressivly try to reduce register pressure in the unclustered high RP
  // stage. Temporarily increase occupancy target in the region.
  S.SGPRLimitBias = S.HighRPSGPRBias;
  S.VGPRLimitBias = S.HighRPVGPRBias;
  if (MFI.getMaxWavesPerEU() > DAG.MinOccupancy)
    MFI.increaseOccupancy(MF, ++DAG.MinOccupancy);

  LLVM_DEBUG(
      dbgs()
      << "Retrying function scheduling without clustering. "
         "Aggressivly try to reduce register pressure to achieve occupancy "
      << DAG.MinOccupancy << ".\n");

  return true;
}

bool ClusteredLowOccStage::initGCNSchedStage() {
  if (DisableClusteredLowOccupancy)
    return false;

  if (!GCNSchedStage::initGCNSchedStage())
    return false;

  // Don't bother trying to improve ILP in lower RP regions if occupancy has not
  // been dropped. All regions will have already been scheduled with the ideal
  // occupancy targets.
  if (DAG.StartingOccupancy <= DAG.MinOccupancy)
    return false;

  LLVM_DEBUG(
      dbgs() << "Retrying function scheduling with lowest recorded occupancy "
             << DAG.MinOccupancy << ".\n");
  return true;
}

/// Allows to easily filter for this stage's debug output.
#define REMAT_DEBUG(X) LLVM_DEBUG(dbgs() << "[PreRARemat] "; X;)

bool PreRARematStage::initGCNSchedStage() {
  // FIXME: This pass will invalidate cached BBLiveInMap and MBBLiveIns for
  // regions inbetween the defs and region we sinked the def to. Will need to be
  // fixed if there is another pass after this pass.
  assert(!S.hasNextStage());

  if (!GCNSchedStage::initGCNSchedStage() || DAG.RegionsWithMinOcc.none() ||
      DAG.Regions.size() == 1)
    return false;

  // Before performing any IR modification record the parent region of each MI
  // and the parent MBB of each region.
  const unsigned NumRegions = DAG.Regions.size();
  RegionBB.reserve(NumRegions);
  for (unsigned I = 0; I < NumRegions; ++I) {
    RegionBoundaries Region = DAG.Regions[I];
    for (auto MI = Region.first; MI != Region.second; ++MI)
      MIRegion.insert({&*MI, I});
    RegionBB.push_back(Region.first->getParent());
  }

  if (!canIncreaseOccupancyOrReduceSpill())
    return false;

  // Rematerialize identified instructions and update scheduler's state.
  rematerialize();
  if (GCNTrackers)
    DAG.RegionLiveOuts.buildLiveRegMap();
  REMAT_DEBUG(
      dbgs() << "Retrying function scheduling with new min. occupancy of "
             << AchievedOcc << " from rematerializing (original was "
             << DAG.MinOccupancy << ", target was " << TargetOcc << ")\n");
  if (AchievedOcc > DAG.MinOccupancy) {
    DAG.MinOccupancy = AchievedOcc;
    SIMachineFunctionInfo &MFI = *MF.getInfo<SIMachineFunctionInfo>();
    MFI.increaseOccupancy(MF, DAG.MinOccupancy);
  }
  return true;
}

void GCNSchedStage::finalizeGCNSchedStage() {
  DAG.finishBlock();
  LLVM_DEBUG(dbgs() << "Ending scheduling stage: " << StageID << "\n");
}

void UnclusteredHighRPStage::finalizeGCNSchedStage() {
  SavedMutations.swap(DAG.Mutations);
  S.SGPRLimitBias = S.VGPRLimitBias = 0;
  if (DAG.MinOccupancy > InitialOccupancy) {
    for (unsigned IDX = 0; IDX < DAG.Pressure.size(); ++IDX)
      DAG.RegionsWithMinOcc[IDX] =
          DAG.Pressure[IDX].getOccupancy(DAG.ST) == DAG.MinOccupancy;

    LLVM_DEBUG(dbgs() << StageID
                      << " stage successfully increased occupancy to "
                      << DAG.MinOccupancy << '\n');
  }

  GCNSchedStage::finalizeGCNSchedStage();
}

bool GCNSchedStage::initGCNRegion() {
  // Check whether this new region is also a new block.
  if (DAG.RegionBegin->getParent() != CurrentMBB)
    setupNewBlock();

  unsigned NumRegionInstrs = std::distance(DAG.begin(), DAG.end());
  DAG.enterRegion(CurrentMBB, DAG.begin(), DAG.end(), NumRegionInstrs);

  // Skip empty scheduling regions (0 or 1 schedulable instructions).
  if (DAG.begin() == DAG.end() || DAG.begin() == std::prev(DAG.end()))
    return false;

  LLVM_DEBUG(dbgs() << "********** MI Scheduling **********\n");
  LLVM_DEBUG(dbgs() << MF.getName() << ":" << printMBBReference(*CurrentMBB)
                    << " " << CurrentMBB->getName()
                    << "\n  From: " << *DAG.begin() << "    To: ";
             if (DAG.RegionEnd != CurrentMBB->end()) dbgs() << *DAG.RegionEnd;
             else dbgs() << "End";
             dbgs() << " RegionInstrs: " << NumRegionInstrs << '\n');

  // Save original instruction order before scheduling for possible revert.
  Unsched.clear();
  Unsched.reserve(DAG.NumRegionInstrs);
  if (StageID == GCNSchedStageID::OccInitialSchedule ||
      StageID == GCNSchedStageID::ILPInitialSchedule) {
    const SIInstrInfo *SII = static_cast<const SIInstrInfo *>(DAG.TII);
    for (auto &I : DAG) {
      Unsched.push_back(&I);
      if (SII->isIGLPMutationOnly(I.getOpcode()))
        DAG.RegionsWithIGLPInstrs[RegionIdx] = true;
    }
  } else {
    for (auto &I : DAG)
      Unsched.push_back(&I);
  }

  PressureBefore = DAG.Pressure[RegionIdx];

  LLVM_DEBUG(
      dbgs() << "Pressure before scheduling:\nRegion live-ins:"
             << print(DAG.LiveIns[RegionIdx], DAG.MRI)
             << "Region live-in pressure:  "
             << print(llvm::getRegPressure(DAG.MRI, DAG.LiveIns[RegionIdx]))
             << "Region register pressure: " << print(PressureBefore));

  S.HasHighPressure = false;
  S.KnownExcessRP = isRegionWithExcessRP();

  if (DAG.RegionsWithIGLPInstrs[RegionIdx] &&
      StageID != GCNSchedStageID::UnclusteredHighRPReschedule) {
    SavedMutations.clear();
    SavedMutations.swap(DAG.Mutations);
    bool IsInitialStage = StageID == GCNSchedStageID::OccInitialSchedule ||
                          StageID == GCNSchedStageID::ILPInitialSchedule;
    DAG.addMutation(createIGroupLPDAGMutation(
        IsInitialStage ? AMDGPU::SchedulingPhase::Initial
                       : AMDGPU::SchedulingPhase::PreRAReentry));
  }

  return true;
}

bool UnclusteredHighRPStage::initGCNRegion() {
  // Only reschedule regions with the minimum occupancy or regions that may have
  // spilling (excess register pressure).
  if ((!DAG.RegionsWithMinOcc[RegionIdx] ||
       DAG.MinOccupancy <= InitialOccupancy) &&
      !DAG.RegionsWithExcessRP[RegionIdx])
    return false;

  return GCNSchedStage::initGCNRegion();
}

bool ClusteredLowOccStage::initGCNRegion() {
  // We may need to reschedule this region if it wasn't rescheduled in the last
  // stage, or if we found it was testing critical register pressure limits in
  // the unclustered reschedule stage. The later is because we may not have been
  // able to raise the min occupancy in the previous stage so the region may be
  // overly constrained even if it was already rescheduled.
  if (!DAG.RegionsWithHighRP[RegionIdx])
    return false;

  return GCNSchedStage::initGCNRegion();
}

bool PreRARematStage::initGCNRegion() {
  return RescheduleRegions[RegionIdx] && GCNSchedStage::initGCNRegion();
}

void GCNSchedStage::setupNewBlock() {
  if (CurrentMBB)
    DAG.finishBlock();

  CurrentMBB = DAG.RegionBegin->getParent();
  DAG.startBlock(CurrentMBB);
  // Get real RP for the region if it hasn't be calculated before. After the
  // initial schedule stage real RP will be collected after scheduling.
  if (StageID == GCNSchedStageID::OccInitialSchedule ||
      StageID == GCNSchedStageID::ILPInitialSchedule ||
      StageID == GCNSchedStageID::MemoryClauseInitialSchedule)
    DAG.computeBlockPressure(RegionIdx, CurrentMBB);
}

void GCNSchedStage::finalizeGCNRegion() {
  DAG.Regions[RegionIdx] = std::pair(DAG.RegionBegin, DAG.RegionEnd);
  if (S.HasHighPressure)
    DAG.RegionsWithHighRP[RegionIdx] = true;

  // Revert scheduling if we have dropped occupancy or there is some other
  // reason that the original schedule is better.
  checkScheduling();

  if (DAG.RegionsWithIGLPInstrs[RegionIdx] &&
      StageID != GCNSchedStageID::UnclusteredHighRPReschedule)
    SavedMutations.swap(DAG.Mutations);

  DAG.exitRegion();
  advanceRegion();
}

void GCNSchedStage::checkScheduling() {
  // Check the results of scheduling.
  PressureAfter = DAG.getRealRegPressure(RegionIdx);

  LLVM_DEBUG(dbgs() << "Pressure after scheduling: " << print(PressureAfter));
  LLVM_DEBUG(dbgs() << "Region: " << RegionIdx << ".\n");

  if (PressureAfter.getSGPRNum() <= S.SGPRCriticalLimit &&
      PressureAfter.getVGPRNum(ST.hasGFX90AInsts()) <= S.VGPRCriticalLimit) {
    DAG.Pressure[RegionIdx] = PressureAfter;
    DAG.RegionsWithMinOcc[RegionIdx] =
        PressureAfter.getOccupancy(ST) == DAG.MinOccupancy;

    // Early out if we have achieved the occupancy target.
    LLVM_DEBUG(dbgs() << "Pressure in desired limits, done.\n");
    return;
  }

  unsigned TargetOccupancy = std::min(
      S.getTargetOccupancy(), ST.getOccupancyWithWorkGroupSizes(MF).second);
  unsigned WavesAfter =
      std::min(TargetOccupancy, PressureAfter.getOccupancy(ST));
  unsigned WavesBefore =
      std::min(TargetOccupancy, PressureBefore.getOccupancy(ST));
  LLVM_DEBUG(dbgs() << "Occupancy before scheduling: " << WavesBefore
                    << ", after " << WavesAfter << ".\n");

  // We may not be able to keep the current target occupancy because of the just
  // scheduled region. We might still be able to revert scheduling if the
  // occupancy before was higher, or if the current schedule has register
  // pressure higher than the excess limits which could lead to more spilling.
  unsigned NewOccupancy = std::max(WavesAfter, WavesBefore);

  // Allow memory bound functions to drop to 4 waves if not limited by an
  // attribute.
  if (WavesAfter < WavesBefore && WavesAfter < DAG.MinOccupancy &&
      WavesAfter >= MFI.getMinAllowedOccupancy()) {
    LLVM_DEBUG(dbgs() << "Function is memory bound, allow occupancy drop up to "
                      << MFI.getMinAllowedOccupancy() << " waves\n");
    NewOccupancy = WavesAfter;
  }

  if (NewOccupancy < DAG.MinOccupancy) {
    DAG.MinOccupancy = NewOccupancy;
    MFI.limitOccupancy(DAG.MinOccupancy);
    DAG.RegionsWithMinOcc.reset();
    LLVM_DEBUG(dbgs() << "Occupancy lowered for the function to "
                      << DAG.MinOccupancy << ".\n");
  }
  // The maximum number of arch VGPR on non-unified register file, or the
  // maximum VGPR + AGPR in the unified register file case.
  unsigned MaxVGPRs = ST.getMaxNumVGPRs(MF);
  // The maximum number of arch VGPR for both unified and non-unified register
  // file.
  unsigned MaxArchVGPRs = std::min(MaxVGPRs, ST.getAddressableNumArchVGPRs());
  unsigned MaxSGPRs = ST.getMaxNumSGPRs(MF);

  if (PressureAfter.getVGPRNum(ST.hasGFX90AInsts()) > MaxVGPRs ||
      PressureAfter.getArchVGPRNum() > MaxArchVGPRs ||
      PressureAfter.getAGPRNum() > MaxArchVGPRs ||
      PressureAfter.getSGPRNum() > MaxSGPRs) {
    DAG.RegionsWithHighRP[RegionIdx] = true;
    DAG.RegionsWithExcessRP[RegionIdx] = true;
  }

  // Revert if this region's schedule would cause a drop in occupancy or
  // spilling.
  if (shouldRevertScheduling(WavesAfter)) {
    revertScheduling();
  } else {
    DAG.Pressure[RegionIdx] = PressureAfter;
    DAG.RegionsWithMinOcc[RegionIdx] =
        PressureAfter.getOccupancy(ST) == DAG.MinOccupancy;
  }
}

unsigned
GCNSchedStage::computeSUnitReadyCycle(const SUnit &SU, unsigned CurrCycle,
                                      DenseMap<unsigned, unsigned> &ReadyCycles,
                                      const TargetSchedModel &SM) {
  unsigned ReadyCycle = CurrCycle;
  for (auto &D : SU.Preds) {
    if (D.isAssignedRegDep()) {
      MachineInstr *DefMI = D.getSUnit()->getInstr();
      unsigned Latency = SM.computeInstrLatency(DefMI);
      unsigned DefReady = ReadyCycles[DAG.getSUnit(DefMI)->NodeNum];
      ReadyCycle = std::max(ReadyCycle, DefReady + Latency);
    }
  }
  ReadyCycles[SU.NodeNum] = ReadyCycle;
  return ReadyCycle;
}

#ifndef NDEBUG
struct EarlierIssuingCycle {
  bool operator()(std::pair<MachineInstr *, unsigned> A,
                  std::pair<MachineInstr *, unsigned> B) const {
    return A.second < B.second;
  }
};

static void printScheduleModel(std::set<std::pair<MachineInstr *, unsigned>,
                                        EarlierIssuingCycle> &ReadyCycles) {
  if (ReadyCycles.empty())
    return;
  unsigned BBNum = ReadyCycles.begin()->first->getParent()->getNumber();
  dbgs() << "\n################## Schedule time ReadyCycles for MBB : " << BBNum
         << " ##################\n# Cycle #\t\t\tInstruction          "
            "             "
            "                            \n";
  unsigned IPrev = 1;
  for (auto &I : ReadyCycles) {
    if (I.second > IPrev + 1)
      dbgs() << "****************************** BUBBLE OF " << I.second - IPrev
             << " CYCLES DETECTED ******************************\n\n";
    dbgs() << "[ " << I.second << " ]  :  " << *I.first << "\n";
    IPrev = I.second;
  }
}
#endif

ScheduleMetrics
GCNSchedStage::getScheduleMetrics(const std::vector<SUnit> &InputSchedule) {
#ifndef NDEBUG
  std::set<std::pair<MachineInstr *, unsigned>, EarlierIssuingCycle>
      ReadyCyclesSorted;
#endif
  const TargetSchedModel &SM = ST.getInstrInfo()->getSchedModel();
  unsigned SumBubbles = 0;
  DenseMap<unsigned, unsigned> ReadyCycles;
  unsigned CurrCycle = 0;
  for (auto &SU : InputSchedule) {
    unsigned ReadyCycle =
        computeSUnitReadyCycle(SU, CurrCycle, ReadyCycles, SM);
    SumBubbles += ReadyCycle - CurrCycle;
#ifndef NDEBUG
    ReadyCyclesSorted.insert(std::make_pair(SU.getInstr(), ReadyCycle));
#endif
    CurrCycle = ++ReadyCycle;
  }
#ifndef NDEBUG
  LLVM_DEBUG(
      printScheduleModel(ReadyCyclesSorted);
      dbgs() << "\n\t"
             << "Metric: "
             << (SumBubbles
                     ? (SumBubbles * ScheduleMetrics::ScaleFactor) / CurrCycle
                     : 1)
             << "\n\n");
#endif

  return ScheduleMetrics(CurrCycle, SumBubbles);
}

ScheduleMetrics
GCNSchedStage::getScheduleMetrics(const GCNScheduleDAGMILive &DAG) {
#ifndef NDEBUG
  std::set<std::pair<MachineInstr *, unsigned>, EarlierIssuingCycle>
      ReadyCyclesSorted;
#endif
  const TargetSchedModel &SM = ST.getInstrInfo()->getSchedModel();
  unsigned SumBubbles = 0;
  DenseMap<unsigned, unsigned> ReadyCycles;
  unsigned CurrCycle = 0;
  for (auto &MI : DAG) {
    SUnit *SU = DAG.getSUnit(&MI);
    if (!SU)
      continue;
    unsigned ReadyCycle =
        computeSUnitReadyCycle(*SU, CurrCycle, ReadyCycles, SM);
    SumBubbles += ReadyCycle - CurrCycle;
#ifndef NDEBUG
    ReadyCyclesSorted.insert(std::make_pair(SU->getInstr(), ReadyCycle));
#endif
    CurrCycle = ++ReadyCycle;
  }
#ifndef NDEBUG
  LLVM_DEBUG(
      printScheduleModel(ReadyCyclesSorted);
      dbgs() << "\n\t"
             << "Metric: "
             << (SumBubbles
                     ? (SumBubbles * ScheduleMetrics::ScaleFactor) / CurrCycle
                     : 1)
             << "\n\n");
#endif

  return ScheduleMetrics(CurrCycle, SumBubbles);
}

bool GCNSchedStage::shouldRevertScheduling(unsigned WavesAfter) {
  if (WavesAfter < DAG.MinOccupancy)
    return true;

  // For dynamic VGPR mode, we don't want to waste any VGPR blocks.
  if (ST.isDynamicVGPREnabled()) {
    unsigned BlocksBefore = AMDGPU::IsaInfo::getAllocatedNumVGPRBlocks(
        &ST, PressureBefore.getVGPRNum(false));
    unsigned BlocksAfter = AMDGPU::IsaInfo::getAllocatedNumVGPRBlocks(
        &ST, PressureAfter.getVGPRNum(false));
    if (BlocksAfter > BlocksBefore)
      return true;
  }

  return false;
}

bool OccInitialScheduleStage::shouldRevertScheduling(unsigned WavesAfter) {
  if (PressureAfter == PressureBefore)
    return false;

  if (GCNSchedStage::shouldRevertScheduling(WavesAfter))
    return true;

  if (mayCauseSpilling(WavesAfter))
    return true;

  return false;
}

bool UnclusteredHighRPStage::shouldRevertScheduling(unsigned WavesAfter) {
  // If RP is not reduced in the unclustered reschedule stage, revert to the
  // old schedule.
  if ((WavesAfter <= PressureBefore.getOccupancy(ST) &&
       mayCauseSpilling(WavesAfter)) ||
      GCNSchedStage::shouldRevertScheduling(WavesAfter)) {
    LLVM_DEBUG(dbgs() << "Unclustered reschedule did not help.\n");
    return true;
  }

  // Do not attempt to relax schedule even more if we are already spilling.
  if (isRegionWithExcessRP())
    return false;

  LLVM_DEBUG(
      dbgs()
      << "\n\t      *** In shouldRevertScheduling ***\n"
      << "      *********** BEFORE UnclusteredHighRPStage ***********\n");
  ScheduleMetrics MBefore = getScheduleMetrics(DAG.SUnits);
  LLVM_DEBUG(
      dbgs()
      << "\n      *********** AFTER UnclusteredHighRPStage ***********\n");
  ScheduleMetrics MAfter = getScheduleMetrics(DAG);
  unsigned OldMetric = MBefore.getMetric();
  unsigned NewMetric = MAfter.getMetric();
  unsigned WavesBefore =
      std::min(S.getTargetOccupancy(), PressureBefore.getOccupancy(ST));
  unsigned Profit =
      ((WavesAfter * ScheduleMetrics::ScaleFactor) / WavesBefore *
       ((OldMetric + ScheduleMetricBias) * ScheduleMetrics::ScaleFactor) /
       NewMetric) /
      ScheduleMetrics::ScaleFactor;
  LLVM_DEBUG(dbgs() << "\tMetric before " << MBefore << "\tMetric after "
                    << MAfter << "Profit: " << Profit << "\n");
  return Profit < ScheduleMetrics::ScaleFactor;
}

bool ClusteredLowOccStage::shouldRevertScheduling(unsigned WavesAfter) {
  if (PressureAfter == PressureBefore)
    return false;

  if (GCNSchedStage::shouldRevertScheduling(WavesAfter))
    return true;

  if (mayCauseSpilling(WavesAfter))
    return true;

  return false;
}

bool PreRARematStage::shouldRevertScheduling(unsigned WavesAfter) {
  return GCNSchedStage::shouldRevertScheduling(WavesAfter) ||
         mayCauseSpilling(WavesAfter) ||
         (IncreaseOccupancy && WavesAfter < TargetOcc);
}

bool ILPInitialScheduleStage::shouldRevertScheduling(unsigned WavesAfter) {
  if (mayCauseSpilling(WavesAfter))
    return true;

  return false;
}

bool MemoryClauseInitialScheduleStage::shouldRevertScheduling(
    unsigned WavesAfter) {
  return mayCauseSpilling(WavesAfter);
}

bool GCNSchedStage::mayCauseSpilling(unsigned WavesAfter) {
  if (WavesAfter <= MFI.getMinWavesPerEU() && isRegionWithExcessRP() &&
      !PressureAfter.less(MF, PressureBefore)) {
    LLVM_DEBUG(dbgs() << "New pressure will result in more spilling.\n");
    return true;
  }

  return false;
}

void GCNSchedStage::revertScheduling() {
  DAG.RegionsWithMinOcc[RegionIdx] =
      PressureBefore.getOccupancy(ST) == DAG.MinOccupancy;
  LLVM_DEBUG(dbgs() << "Attempting to revert scheduling.\n");
  DAG.RegionEnd = DAG.RegionBegin;
  int SkippedDebugInstr = 0;
  for (MachineInstr *MI : Unsched) {
    if (MI->isDebugInstr()) {
      ++SkippedDebugInstr;
      continue;
    }

    if (MI->getIterator() != DAG.RegionEnd) {
      DAG.BB->splice(DAG.RegionEnd, DAG.BB, MI);
      if (!MI->isDebugInstr())
        DAG.LIS->handleMove(*MI, true);
    }

    // Reset read-undef flags and update them later.
    for (auto &Op : MI->all_defs())
      Op.setIsUndef(false);
    RegisterOperands RegOpers;
    RegOpers.collect(*MI, *DAG.TRI, DAG.MRI, DAG.ShouldTrackLaneMasks, false);
    if (!MI->isDebugInstr()) {
      if (DAG.ShouldTrackLaneMasks) {
        // Adjust liveness and add missing dead+read-undef flags.
        SlotIndex SlotIdx = DAG.LIS->getInstructionIndex(*MI).getRegSlot();
        RegOpers.adjustLaneLiveness(*DAG.LIS, DAG.MRI, SlotIdx, MI);
      } else {
        // Adjust for missing dead-def flags.
        RegOpers.detectDeadDefs(*MI, *DAG.LIS);
      }
    }
    DAG.RegionEnd = MI->getIterator();
    ++DAG.RegionEnd;
    LLVM_DEBUG(dbgs() << "Scheduling " << *MI);
  }

  // After reverting schedule, debug instrs will now be at the end of the block
  // and RegionEnd will point to the first debug instr. Increment RegionEnd
  // pass debug instrs to the actual end of the scheduling region.
  while (SkippedDebugInstr-- > 0)
    ++DAG.RegionEnd;

  // If Unsched.front() instruction is a debug instruction, this will actually
  // shrink the region since we moved all debug instructions to the end of the
  // block. Find the first instruction that is not a debug instruction.
  DAG.RegionBegin = Unsched.front()->getIterator();
  if (DAG.RegionBegin->isDebugInstr()) {
    for (MachineInstr *MI : Unsched) {
      if (MI->isDebugInstr())
        continue;
      DAG.RegionBegin = MI->getIterator();
      break;
    }
  }

  // Then move the debug instructions back into their correct place and set
  // RegionBegin and RegionEnd if needed.
  DAG.placeDebugValues();

  DAG.Regions[RegionIdx] = std::pair(DAG.RegionBegin, DAG.RegionEnd);
}

bool PreRARematStage::allUsesAvailableAt(const MachineInstr *InstToRemat,
                                         SlotIndex OriginalIdx,
                                         SlotIndex RematIdx) const {

  LiveIntervals *LIS = DAG.LIS;
  MachineRegisterInfo &MRI = DAG.MRI;
  OriginalIdx = OriginalIdx.getRegSlot(true);
  RematIdx = std::max(RematIdx, RematIdx.getRegSlot(true));
  for (const MachineOperand &MO : InstToRemat->operands()) {
    if (!MO.isReg() || !MO.getReg() || !MO.readsReg())
      continue;

    if (!MO.getReg().isVirtual()) {
      // Do not attempt to reason about PhysRegs
      // TODO: better analysis of PhysReg livness
      if (!DAG.MRI.isConstantPhysReg(MO.getReg()) &&
          !DAG.TII->isIgnorableUse(MO))
        return false;

      // Constant PhysRegs and IgnorableUses are okay
      continue;
    }

    LiveInterval &LI = LIS->getInterval(MO.getReg());
    const VNInfo *OVNI = LI.getVNInfoAt(OriginalIdx);
    assert(OVNI);

    // Don't allow rematerialization immediately after the original def.
    // It would be incorrect if InstToRemat redefines the register.
    // See PR14098.
    if (SlotIndex::isSameInstr(OriginalIdx, RematIdx))
      return false;

    if (OVNI != LI.getVNInfoAt(RematIdx))
      return false;

    // Check that subrange is live at RematIdx.
    if (LI.hasSubRanges()) {
      const TargetRegisterInfo *TRI = MRI.getTargetRegisterInfo();
      unsigned SubReg = MO.getSubReg();
      LaneBitmask LM = SubReg ? TRI->getSubRegIndexLaneMask(SubReg)
                              : MRI.getMaxLaneMaskForVReg(MO.getReg());
      for (LiveInterval::SubRange &SR : LI.subranges()) {
        if ((SR.LaneMask & LM).none())
          continue;
        if (!SR.liveAt(RematIdx))
          return false;

        // Early exit if all used lanes are checked. No need to continue.
        LM &= ~SR.LaneMask;
        if (LM.none())
          break;
      }
    }
  }
  return true;
}

namespace {
/// Models excess register pressure in a region and tracks our progress as we
/// identify rematerialization opportunities.
struct ExcessRP {
  /// Number of excess ArchVGPRs.
  unsigned ArchVGPRs = 0;
  /// Number of excess AGPRs.
  unsigned AGPRs = 0;
  /// For unified register files, number of excess VGPRs.
  unsigned VGPRs = 0;
  /// For unified register files with AGPR usage, number of excess ArchVGPRs to
  /// save before we are able to save a whole allocation granule.
  unsigned ArchVGPRsToAlignment = 0;
  /// Whether the region uses AGPRs.
  bool HasAGPRs = false;
  /// Whether the subtarget has a unified RF.
  bool UnifiedRF;

  /// Constructs the excess RP model; determines the excess pressure w.r.t. a
  /// maximum number of allowed VGPRs.
  ExcessRP(const GCNSubtarget &ST, const GCNRegPressure &RP, unsigned MaxVGPRs);

  /// Accounts for \p NumRegs saved ArchVGPRs in the model. If \p
  /// UseArchVGPRForAGPRSpill is true, saved ArchVGPRs are used to save excess
  /// AGPRs once excess ArchVGPR pressure has been eliminated. Returns whether
  /// saving these ArchVGPRs helped reduce excess pressure.
  bool saveArchVGPRs(unsigned NumRegs, bool UseArchVGPRForAGPRSpill);

  /// Accounts for \p NumRegs saved AGPRS in the model. Returns whether saving
  /// these ArchVGPRs helped reduce excess pressure.
  bool saveAGPRs(unsigned NumRegs);

  /// Returns whether there is any excess register pressure.
  operator bool() const { return ArchVGPRs != 0 || AGPRs != 0 || VGPRs != 0; }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  friend raw_ostream &operator<<(raw_ostream &OS, const ExcessRP &Excess) {
    OS << Excess.ArchVGPRs << " ArchVGPRs, " << Excess.AGPRs << " AGPRs, and "
       << Excess.VGPRs << " VGPRs (next ArchVGPR aligment in "
       << Excess.ArchVGPRsToAlignment << " registers)\n";
    return OS;
  }
#endif

private:
  static inline bool saveRegs(unsigned &LeftToSave, unsigned &NumRegs) {
    unsigned NumSaved = std::min(LeftToSave, NumRegs);
    NumRegs -= NumSaved;
    LeftToSave -= NumSaved;
    return NumSaved;
  }
};
} // namespace

ExcessRP::ExcessRP(const GCNSubtarget &ST, const GCNRegPressure &RP,
                   unsigned MaxVGPRs)
    : UnifiedRF(ST.hasGFX90AInsts()) {
  unsigned NumArchVGPRs = RP.getArchVGPRNum();
  unsigned NumAGPRs = RP.getAGPRNum();
  HasAGPRs = NumAGPRs;

  if (!UnifiedRF) {
    // Non-unified RF. Account for excess pressure for ArchVGPRs and AGPRs
    // independently.
    if (NumArchVGPRs > MaxVGPRs)
      ArchVGPRs = NumArchVGPRs - MaxVGPRs;
    if (NumAGPRs > MaxVGPRs)
      AGPRs = NumAGPRs - MaxVGPRs;
    return;
  }

  // Independently of whether overall VGPR pressure is under the limit, we still
  // have to check whether ArchVGPR pressure or AGPR pressure alone exceeds the
  // number of addressable registers in each category.
  const unsigned MaxArchVGPRs = ST.getAddressableNumArchVGPRs();
  if (NumArchVGPRs > MaxArchVGPRs) {
    ArchVGPRs = NumArchVGPRs - MaxArchVGPRs;
    NumArchVGPRs = MaxArchVGPRs;
  }
  if (NumAGPRs > MaxArchVGPRs) {
    AGPRs = NumAGPRs - MaxArchVGPRs;
    NumAGPRs = MaxArchVGPRs;
  }

  // Check overall VGPR usage against the limit; any excess above addressable
  // register limits has already been accounted for.
  const unsigned Granule = AMDGPU::IsaInfo::getArchVGPRAllocGranule();
  unsigned NumVGPRs = GCNRegPressure::getUnifiedVGPRNum(NumArchVGPRs, NumAGPRs);
  if (NumVGPRs > MaxVGPRs) {
    VGPRs = NumVGPRs - MaxVGPRs;
    ArchVGPRsToAlignment = NumArchVGPRs - alignDown(NumArchVGPRs, Granule);
    if (!ArchVGPRsToAlignment)
      ArchVGPRsToAlignment = Granule;
  }
}

bool ExcessRP::saveArchVGPRs(unsigned NumRegs, bool UseArchVGPRForAGPRSpill) {
  bool Progress = saveRegs(ArchVGPRs, NumRegs);
  if (!NumRegs)
    return Progress;

  if (!UnifiedRF) {
    if (UseArchVGPRForAGPRSpill)
      Progress |= saveRegs(AGPRs, NumRegs);
  } else if (HasAGPRs && (VGPRs || (UseArchVGPRForAGPRSpill && AGPRs))) {
    // There is progress as long as there are VGPRs left to save, even if the
    // save induced by this particular call does not cross an ArchVGPR alignment
    // barrier.
    Progress = true;

    // ArchVGPRs can only be allocated as a multiple of a granule in unified RF.
    unsigned NumSavedRegs = 0;

    // Count the number of whole ArchVGPR allocation granules we can save.
    const unsigned Granule = AMDGPU::IsaInfo::getArchVGPRAllocGranule();
    if (unsigned NumGranules = NumRegs / Granule; NumGranules) {
      NumSavedRegs = NumGranules * Granule;
      NumRegs -= NumSavedRegs;
    }

    // We may be able to save one more whole ArchVGPR allocation granule.
    if (NumRegs >= ArchVGPRsToAlignment) {
      NumSavedRegs += Granule;
      ArchVGPRsToAlignment = Granule - (NumRegs - ArchVGPRsToAlignment);
    } else {
      ArchVGPRsToAlignment -= NumRegs;
    }

    // Prioritize saving generic VGPRs, then AGPRs if we allow AGPR-to-ArchVGPR
    // spilling and have some free ArchVGPR slots.
    saveRegs(VGPRs, NumSavedRegs);
    if (UseArchVGPRForAGPRSpill)
      saveRegs(AGPRs, NumSavedRegs);
  } else {
    // No AGPR usage in the region i.e., no allocation granule to worry about.
    Progress |= saveRegs(VGPRs, NumRegs);
  }

  return Progress;
}

bool ExcessRP::saveAGPRs(unsigned NumRegs) {
  return saveRegs(AGPRs, NumRegs) || saveRegs(VGPRs, NumRegs);
}

bool PreRARematStage::canIncreaseOccupancyOrReduceSpill() {
  const SIRegisterInfo *SRI = static_cast<const SIRegisterInfo *>(DAG.TRI);

  REMAT_DEBUG({
    dbgs() << "Collecting rematerializable instructions in ";
    MF.getFunction().printAsOperand(dbgs(), false);
    dbgs() << '\n';
  });

  // Maps optimizable regions (i.e., regions at minimum and VGPR-limited
  // occupancy, or regions with VGPR spilling) to a model of their excess RP.
  DenseMap<unsigned, ExcessRP> OptRegions;
  const Function &F = MF.getFunction();

  std::pair<unsigned, unsigned> WavesPerEU = ST.getWavesPerEU(F);
  const unsigned MaxSGPRsNoSpill = ST.getMaxNumSGPRs(F);
  const unsigned MaxVGPRsNoSpill = ST.getMaxNumVGPRs(F);
  const unsigned MaxSGPRsIncOcc =
      ST.getMaxNumSGPRs(DAG.MinOccupancy + 1, false);
  const unsigned MaxVGPRsIncOcc = ST.getMaxNumVGPRs(DAG.MinOccupancy + 1);
  IncreaseOccupancy = WavesPerEU.second > DAG.MinOccupancy;

  auto ClearOptRegionsIf = [&](bool Cond) -> bool {
    if (Cond) {
      // We won't try to increase occupancy.
      IncreaseOccupancy = false;
      OptRegions.clear();
    }
    return Cond;
  };

  // Collect optimizable regions. If there is spilling in any region we will
  // just try to reduce ArchVGPR spilling. Otherwise we will try to increase
  // occupancy by one in the whole function.
  for (unsigned I = 0, E = DAG.Regions.size(); I != E; ++I) {
    GCNRegPressure &RP = DAG.Pressure[I];

    // Check whether SGPR pressures prevents us from eliminating spilling.
    unsigned NumSGPRs = RP.getSGPRNum();
    if (NumSGPRs > MaxSGPRsNoSpill)
      ClearOptRegionsIf(IncreaseOccupancy);

    ExcessRP Excess(ST, RP, MaxVGPRsNoSpill);
    if (Excess) {
      ClearOptRegionsIf(IncreaseOccupancy);
    } else if (IncreaseOccupancy) {
      // Check whether SGPR pressure prevents us from increasing occupancy.
      if (ClearOptRegionsIf(NumSGPRs > MaxSGPRsIncOcc)) {
        if (DAG.MinOccupancy >= WavesPerEU.first)
          return false;
        continue;
      }
      if ((Excess = ExcessRP(ST, RP, MaxVGPRsIncOcc))) {
        // We can only rematerialize ArchVGPRs at this point.
        unsigned NumArchVGPRsToRemat = Excess.ArchVGPRs + Excess.VGPRs;
        bool NotEnoughArchVGPRs = NumArchVGPRsToRemat > RP.getArchVGPRNum();
        if (ClearOptRegionsIf(Excess.AGPRs || NotEnoughArchVGPRs)) {
          if (DAG.MinOccupancy >= WavesPerEU.first)
            return false;
          continue;
        }
      }
    }
    if (Excess)
      OptRegions.insert({I, Excess});
  }
  if (OptRegions.empty())
    return false;

#ifndef NDEBUG
  if (IncreaseOccupancy)
    REMAT_DEBUG(dbgs() << "Occupancy minimal in regions:\n");
  else
    REMAT_DEBUG(dbgs() << "Spilling in regions:\n");
  for (unsigned I = 0, E = DAG.Regions.size(); I != E; ++I) {
    if (auto OptIt = OptRegions.find(I); OptIt != OptRegions.end())
      REMAT_DEBUG(dbgs() << "  " << I << ": " << OptIt->getSecond() << '\n');
  }
#endif

  // When we are reducing spilling, the target is the minimum target number of
  // waves/EU determined by the subtarget.
  TargetOcc = IncreaseOccupancy ? DAG.MinOccupancy + 1 : WavesPerEU.first;

  // Accounts for a reduction in RP in an optimizable region. Returns whether we
  // estimate that we have identified enough rematerialization opportunities to
  // achieve our goal, and sets Progress to true when this particular reduction
  // in pressure was helpful toward that goal.
  auto ReduceRPInRegion = [&](auto OptIt, LaneBitmask Mask,
                              bool &Progress) -> bool {
    ExcessRP &Excess = OptIt->getSecond();
    // We allow saved ArchVGPRs to be considered as free spill slots for AGPRs
    // only when we are just trying to eliminate spilling to memory. At this
    // point we err on the conservative side and do not increase
    // register-to-register spilling for the sake of increasing occupancy.
    Progress |=
        Excess.saveArchVGPRs(SIRegisterInfo::getNumCoveredRegs(Mask),
                             /*UseArchVGPRForAGPRSpill=*/!IncreaseOccupancy);
    if (!Excess)
      OptRegions.erase(OptIt->getFirst());
    return OptRegions.empty();
  };

  // We need up-to-date live-out info. to query live-out register masks in
  // regions containing rematerializable instructions.
  DAG.RegionLiveOuts.buildLiveRegMap();

  // Cache set of registers that are going to be rematerialized.
  DenseSet<unsigned> RematRegs;

  // Identify rematerializable instructions in the function.
  for (unsigned I = 0, E = DAG.Regions.size(); I != E; ++I) {
    auto Region = DAG.Regions[I];
    for (auto MI = Region.first; MI != Region.second; ++MI) {
      // The instruction must be trivially rematerializable.
      MachineInstr &DefMI = *MI;
      if (!isTriviallyReMaterializable(DefMI))
        continue;

      // We only support rematerializing virtual VGPRs with one definition.
      Register Reg = DefMI.getOperand(0).getReg();
      if (!Reg.isVirtual() || !SRI->isVGPRClass(DAG.MRI.getRegClass(Reg)) ||
          !DAG.MRI.hasOneDef(Reg))
        continue;

      // We only care to rematerialize the instruction if it has a single
      // non-debug user in a different region. The using MI may not belong to a
      // region if it is a lone region terminator.
      MachineInstr *UseMI = DAG.MRI.getOneNonDBGUser(Reg);
      if (!UseMI)
        continue;
      auto UseRegion = MIRegion.find(UseMI);
      if (UseRegion != MIRegion.end() && UseRegion->second == I)
        continue;

      // Do not rematerialize an instruction if it uses or is used by an
      // instruction that we have designated for rematerialization.
      // FIXME: Allow for rematerialization chains: this requires 1. updating
      // remat points to account for uses that are rematerialized, and 2. either
      // rematerializing the candidates in careful ordering, or deferring the
      // MBB RP walk until the entire chain has been rematerialized.
      if (Rematerializations.contains(UseMI) ||
          llvm::any_of(DefMI.operands(), [&RematRegs](MachineOperand &MO) {
            return MO.isReg() && RematRegs.contains(MO.getReg());
          }))
        continue;

      // Do not rematerialize an instruction it it uses registers that aren't
      // available at its use. This ensures that we are not extending any live
      // range while rematerializing.
      SlotIndex DefIdx = DAG.LIS->getInstructionIndex(DefMI);
      SlotIndex UseIdx = DAG.LIS->getInstructionIndex(*UseMI).getRegSlot(true);
      if (!allUsesAvailableAt(&DefMI, DefIdx, UseIdx))
        continue;

      REMAT_DEBUG(dbgs() << "Region " << I << ": remat instruction " << DefMI);
      RematInstruction &Remat =
          Rematerializations.try_emplace(&DefMI, UseMI).first->second;

      bool RematUseful = false;
      if (auto It = OptRegions.find(I); It != OptRegions.end()) {
        // Optimistically consider that moving the instruction out of its
        // defining region will reduce RP in the latter; this assumes that
        // maximum RP in the region is reached somewhere between the defining
        // instruction and the end of the region.
        REMAT_DEBUG(dbgs() << "  Defining region is optimizable\n");
        LaneBitmask Mask = DAG.RegionLiveOuts.getLiveRegsForRegionIdx(I)[Reg];
        if (ReduceRPInRegion(It, Mask, RematUseful))
          return true;
      }

      for (unsigned LIRegion = 0; LIRegion != E; ++LIRegion) {
        // We are only collecting regions in which the register is a live-in
        // (and may be live-through).
        auto It = DAG.LiveIns[LIRegion].find(Reg);
        if (It == DAG.LiveIns[LIRegion].end() || It->second.none())
          continue;
        Remat.LiveInRegions.insert(LIRegion);

        // Account for the reduction in RP due to the rematerialization in an
        // optimizable region in which the defined register is a live-in. This
        // is exact for live-through region but optimistic in the using region,
        // where RP is actually reduced only if maximum RP is reached somewhere
        // between the beginning of the region and the rematerializable
        // instruction's use.
        if (auto It = OptRegions.find(LIRegion); It != OptRegions.end()) {
          REMAT_DEBUG(dbgs() << "  Live-in in region " << LIRegion << '\n');
          if (ReduceRPInRegion(It, DAG.LiveIns[LIRegion][Reg], RematUseful))
            return true;
        }
      }

      // If the instruction is not a live-in or live-out in any optimizable
      // region then there is no point in rematerializing it.
      if (!RematUseful) {
        Rematerializations.pop_back();
        REMAT_DEBUG(dbgs() << "  No impact, not rematerializing instruction\n");
      } else {
        RematRegs.insert(Reg);
      }
    }
  }

  if (IncreaseOccupancy) {
    // We were trying to increase occupancy but failed, abort the stage.
    REMAT_DEBUG(dbgs() << "Cannot increase occupancy\n");
    Rematerializations.clear();
    return false;
  }
  REMAT_DEBUG(dbgs() << "Can reduce but not eliminate spilling\n");
  return !Rematerializations.empty();
}

void PreRARematStage::rematerialize() {
  const SIInstrInfo *TII = MF.getSubtarget<GCNSubtarget>().getInstrInfo();

  // Collect regions whose RP changes in unpredictable way; we will have to
  // fully recompute their RP after all rematerailizations.
  DenseSet<unsigned> RecomputeRP;

  // Rematerialize all instructions.
  for (auto &[DefMI, Remat] : Rematerializations) {
    MachineBasicBlock::iterator InsertPos(Remat.UseMI);
    Register Reg = DefMI->getOperand(0).getReg();
    unsigned SubReg = DefMI->getOperand(0).getSubReg();
    unsigned DefRegion = MIRegion.at(DefMI);

    // Rematerialize DefMI to its use block.
    TII->reMaterialize(*InsertPos->getParent(), InsertPos, Reg, SubReg, *DefMI,
                       *DAG.TRI);
    Remat.RematMI = &*std::prev(InsertPos);
    Remat.RematMI->getOperand(0).setSubReg(SubReg);
    DAG.LIS->InsertMachineInstrInMaps(*Remat.RematMI);

    // Update region boundaries in regions we sinked from (remove defining MI)
    // and to (insert MI rematerialized in use block). Only then we can erase
    // the original MI.
    DAG.updateRegionBoundaries(DAG.Regions[DefRegion], DefMI, nullptr);
    auto UseRegion = MIRegion.find(Remat.UseMI);
    if (UseRegion != MIRegion.end()) {
      DAG.updateRegionBoundaries(DAG.Regions[UseRegion->second], InsertPos,
                                 Remat.RematMI);
    }
    DAG.LIS->RemoveMachineInstrFromMaps(*DefMI);
    DefMI->eraseFromParent();

    // Collect all regions impacted by the rematerialization and update their
    // live-in/RP information.
    for (unsigned I : Remat.LiveInRegions) {
      ImpactedRegions.insert({I, DAG.Pressure[I]});
      GCNRPTracker::LiveRegSet &RegionLiveIns = DAG.LiveIns[I];

#ifdef EXPENSIVE_CHECKS
      // All uses are known to be available / live at the remat point. Thus, the
      // uses should already be live in to the region.
      for (MachineOperand &MO : DefMI->operands()) {
        if (!MO.isReg() || !MO.getReg() || !MO.readsReg())
          continue;

        Register UseReg = MO.getReg();
        if (!UseReg.isVirtual())
          continue;

        LiveInterval &LI = DAG.LIS->getInterval(UseReg);
        LaneBitmask LM = DAG.MRI.getMaxLaneMaskForVReg(MO.getReg());
        if (LI.hasSubRanges() && MO.getSubReg())
          LM = DAG.TRI->getSubRegIndexLaneMask(MO.getSubReg());

        LaneBitmask LiveInMask = RegionLiveIns.at(UseReg);
        LaneBitmask UncoveredLanes = LM & ~(LiveInMask & LM);
        // If this register has lanes not covered by the LiveIns, be sure they
        // do not map to any subrange. ref:
        // machine-scheduler-sink-trivial-remats.mir::omitted_subrange
        if (UncoveredLanes.any()) {
          assert(LI.hasSubRanges());
          for (LiveInterval::SubRange &SR : LI.subranges())
            assert((SR.LaneMask & UncoveredLanes).none());
        }
      }
#endif

      // The register is no longer a live-in in all regions but the one that
      // contains the single use. In live-through regions, maximum register
      // pressure decreases predictably so we can directly update it. In the
      // using region, maximum RP may or may not decrease, so we will mark it
      // for re-computation after all materializations have taken place.
      LaneBitmask PrevMask = RegionLiveIns[Reg];
      RegionLiveIns.erase(Reg);
      RegMasks.insert({{I, Remat.RematMI->getOperand(0).getReg()}, PrevMask});
      if (Remat.UseMI->getParent() != DAG.Regions[I].first->getParent())
        DAG.Pressure[I].inc(Reg, PrevMask, LaneBitmask::getNone(), DAG.MRI);
      else
        RecomputeRP.insert(I);
    }
    // RP in the region from which the instruction was rematerialized may or may
    // not decrease.
    ImpactedRegions.insert({DefRegion, DAG.Pressure[DefRegion]});
    RecomputeRP.insert(DefRegion);

    // Recompute live interval to reflect the register's rematerialization.
    Register RematReg = Remat.RematMI->getOperand(0).getReg();
    DAG.LIS->removeInterval(RematReg);
    DAG.LIS->createAndComputeVirtRegInterval(RematReg);
  }

  // All regions impacted by at least one rematerialization must be rescheduled.
  // Maximum pressure must also be recomputed for all regions where it changed
  // non-predictably and checked against the target occupancy.
  AchievedOcc = TargetOcc;
  for (auto &[I, OriginalRP] : ImpactedRegions) {
    bool IsEmptyRegion = DAG.Regions[I].first == DAG.Regions[I].second;
    RescheduleRegions[I] = !IsEmptyRegion;
    if (!RecomputeRP.contains(I))
      continue;

    GCNRegPressure RP;
    if (IsEmptyRegion) {
      RP = getRegPressure(DAG.MRI, DAG.LiveIns[I]);
    } else {
      GCNDownwardRPTracker RPT(*DAG.LIS);
      auto *NonDbgMI = &*skipDebugInstructionsForward(DAG.Regions[I].first,
                                                      DAG.Regions[I].second);
      if (NonDbgMI == DAG.Regions[I].second) {
        // Region is non-empty but contains only debug instructions.
        RP = getRegPressure(DAG.MRI, DAG.LiveIns[I]);
      } else {
        RPT.reset(*NonDbgMI, &DAG.LiveIns[I]);
        RPT.advance(DAG.Regions[I].second);
        RP = RPT.moveMaxPressure();
      }
    }
    DAG.Pressure[I] = RP;
    AchievedOcc = std::min(AchievedOcc, RP.getOccupancy(ST));
  }
  REMAT_DEBUG(dbgs() << "Achieved occupancy " << AchievedOcc << "\n");
}

// Copied from MachineLICM
bool PreRARematStage::isTriviallyReMaterializable(const MachineInstr &MI) {
  if (!DAG.TII->isTriviallyReMaterializable(MI))
    return false;

  for (const MachineOperand &MO : MI.all_uses()) {
    // We can't remat physreg uses, unless it is a constant or an ignorable
    // use (e.g. implicit exec use on VALU instructions)
    if (MO.getReg().isPhysical()) {
      if (DAG.MRI.isConstantPhysReg(MO.getReg()) || DAG.TII->isIgnorableUse(MO))
        continue;
      return false;
    }
  }

  return true;
}

void PreRARematStage::finalizeGCNSchedStage() {
  // We consider that reducing spilling is always beneficial so we never
  // rollback rematerializations in such cases. It's also possible that
  // rescheduling lowers occupancy over the one achieved just through remats, in
  // which case we do not want to rollback either (the rescheduling was already
  // reverted in PreRARematStage::shouldRevertScheduling in such cases).
  unsigned MaxOcc = std::max(AchievedOcc, DAG.MinOccupancy);
  if (!IncreaseOccupancy || MaxOcc >= TargetOcc)
    return;

  REMAT_DEBUG(dbgs() << "Rolling back all rematerializations\n");
  const SIInstrInfo *TII = MF.getSubtarget<GCNSubtarget>().getInstrInfo();

  // Rollback the rematerializations.
  for (const auto &[DefMI, Remat] : Rematerializations) {
    MachineInstr &RematMI = *Remat.RematMI;
    unsigned DefRegion = MIRegion.at(DefMI);
    MachineBasicBlock::iterator InsertPos(DAG.Regions[DefRegion].second);
    MachineBasicBlock *MBB = RegionBB[DefRegion];
    Register Reg = RematMI.getOperand(0).getReg();
    unsigned SubReg = RematMI.getOperand(0).getSubReg();

    // Re-rematerialize MI at the end of its original region. Note that it may
    // not be rematerialized exactly in the same position as originally within
    // the region, but it should not matter much.
    TII->reMaterialize(*MBB, InsertPos, Reg, SubReg, RematMI, *DAG.TRI);
    MachineInstr *NewMI = &*std::prev(InsertPos);
    NewMI->getOperand(0).setSubReg(SubReg);
    DAG.LIS->InsertMachineInstrInMaps(*NewMI);

    auto UseRegion = MIRegion.find(Remat.UseMI);
    if (UseRegion != MIRegion.end()) {
      DAG.updateRegionBoundaries(DAG.Regions[UseRegion->second], RematMI,
                                 nullptr);
    }
    DAG.updateRegionBoundaries(DAG.Regions[DefRegion], InsertPos, NewMI);

    // Erase rematerialized MI.
    DAG.LIS->RemoveMachineInstrFromMaps(RematMI);
    RematMI.eraseFromParent();

    // Recompute live interval for the re-rematerialized register
    DAG.LIS->removeInterval(Reg);
    DAG.LIS->createAndComputeVirtRegInterval(Reg);

    // Re-add the register as a live-in in all regions it used to be one in.
    for (unsigned LIRegion : Remat.LiveInRegions)
      DAG.LiveIns[LIRegion].insert({Reg, RegMasks.at({LIRegion, Reg})});
  }

  // Reset RP in all impacted regions.
  for (auto &[I, OriginalRP] : ImpactedRegions)
    DAG.Pressure[I] = OriginalRP;

  GCNSchedStage::finalizeGCNSchedStage();
}

void GCNScheduleDAGMILive::updateRegionBoundaries(
    RegionBoundaries &RegionBounds, MachineBasicBlock::iterator MI,
    MachineInstr *NewMI) {
  assert((!NewMI || NewMI != RegionBounds.second) &&
         "cannot remove at region end");

  if (RegionBounds.first == RegionBounds.second) {
    assert(NewMI && "cannot remove from an empty region");
    RegionBounds.first = NewMI;
    return;
  }

  // We only care for modifications at the beginning of a non-empty region since
  // the upper region boundary is exclusive.
  if (MI != RegionBounds.first)
    return;
  if (!NewMI)
    RegionBounds.first = std::next(MI); // Removal
  else
    RegionBounds.first = NewMI; // Insertion
}

static bool hasIGLPInstrs(ScheduleDAGInstrs *DAG) {
  const SIInstrInfo *SII = static_cast<const SIInstrInfo *>(DAG->TII);
  return any_of(*DAG, [SII](MachineBasicBlock::iterator MI) {
    return SII->isIGLPMutationOnly(MI->getOpcode());
  });
}

GCNPostScheduleDAGMILive::GCNPostScheduleDAGMILive(
    MachineSchedContext *C, std::unique_ptr<MachineSchedStrategy> S,
    bool RemoveKillFlags)
    : ScheduleDAGMI(C, std::move(S), RemoveKillFlags) {}

void GCNPostScheduleDAGMILive::schedule() {
  HasIGLPInstrs = hasIGLPInstrs(this);
  if (HasIGLPInstrs) {
    SavedMutations.clear();
    SavedMutations.swap(Mutations);
    addMutation(createIGroupLPDAGMutation(AMDGPU::SchedulingPhase::PostRA));
  }

  ScheduleDAGMI::schedule();
}

void GCNPostScheduleDAGMILive::finalizeSchedule() {
  if (HasIGLPInstrs)
    SavedMutations.swap(Mutations);

  ScheduleDAGMI::finalizeSchedule();
}
