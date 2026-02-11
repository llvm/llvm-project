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
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/CalcSpillWeights.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineBlockFrequencyInfo.h"
#include "llvm/CodeGen/MachineBranchProbabilityInfo.h"
#include "llvm/CodeGen/MachineCycleAnalysis.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/RegisterClassInfo.h"
#include "llvm/MC/LaneBitmask.h"
#include "llvm/MC/MCInstrItineraries.h"
#include "llvm/MC/MCSchedule.h"
#include "llvm/MC/TargetRegistry.h"
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

static cl::opt<unsigned> PendingQueueLimit(
    "amdgpu-scheduler-pending-queue-limit", cl::Hidden,
    cl::desc(
        "Max (Available+Pending) size to inspect pending queue (0 disables)"),
    cl::init(256));

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
#define DUMP_MAX_REG_PRESSURE
static cl::opt<bool> PrintMaxRPRegUsageBeforeScheduler(
    "amdgpu-print-max-reg-pressure-regusage-before-scheduler", cl::Hidden,
    cl::desc("Print a list of live registers along with their def/uses at the "
             "point of maximum register pressure before scheduling."),
    cl::init(false));

static cl::opt<bool> PrintMaxRPRegUsageAfterScheduler(
    "amdgpu-print-max-reg-pressure-regusage-after-scheduler", cl::Hidden,
    cl::desc("Print a list of live registers along with their def/uses at the "
             "point of maximum register pressure after scheduling."),
    cl::init(false));
#endif

static cl::opt<bool> DisableRewriteMFMAFormSchedStage(
    "amdgpu-disable-rewrite-mfma-form-sched-stage", cl::Hidden,
    cl::desc("Disable rewrie mfma rewrite scheduling stage"), cl::init(true));

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
    VGPRCriticalLimit = std::min(
        ST.getMaxNumVGPRs(TargetOccupancy, MFI.getDynamicVGPRBlockSize()),
        VGPRExcessLimit);
  } else {
    // This is similar to ST.getMaxNumVGPRs(TargetOccupancy) result except
    // returns a reasonably small number for targets with lots of VGPRs, such
    // as GFX10 and GFX11.
    LLVM_DEBUG(dbgs() << "Region is known to spill, use alternative "
                         "VGPRCriticalLimit calculation method.\n");
    unsigned DynamicVGPRBlockSize = MFI.getDynamicVGPRBlockSize();
    unsigned Granule =
        AMDGPU::IsaInfo::getVGPRAllocGranule(&ST, DynamicVGPRBlockSize);
    unsigned Addressable =
        AMDGPU::IsaInfo::getAddressableNumVGPRs(&ST, DynamicVGPRBlockSize);
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

static bool shouldCheckPending(SchedBoundary &Zone,
                               const TargetSchedModel *SchedModel) {
  bool HasBufferedModel =
      SchedModel->hasInstrSchedModel() && SchedModel->getMicroOpBufferSize();
  unsigned Combined = Zone.Available.size() + Zone.Pending.size();
  return Combined <= PendingQueueLimit && HasBufferedModel;
}

static SUnit *pickOnlyChoice(SchedBoundary &Zone,
                             const TargetSchedModel *SchedModel) {
  // pickOnlyChoice() releases pending instructions and checks for new hazards.
  SUnit *OnlyChoice = Zone.pickOnlyChoice();
  if (!shouldCheckPending(Zone, SchedModel) || Zone.Pending.empty())
    return OnlyChoice;

  return nullptr;
}

void GCNSchedStrategy::printCandidateDecision(const SchedCandidate &Current,
                                              const SchedCandidate &Preferred) {
  LLVM_DEBUG({
    dbgs() << "Prefer:\t\t";
    DAG->dumpNode(*Preferred.SU);

    if (Current.SU) {
      dbgs() << "Not:\t";
      DAG->dumpNode(*Current.SU);
    }

    dbgs() << "Reason:\t\t";
    traceCandidate(Preferred);
  });
}

// This function is mostly cut and pasted from
// GenericScheduler::pickNodeFromQueue()
void GCNSchedStrategy::pickNodeFromQueue(SchedBoundary &Zone,
                                         const CandPolicy &ZonePolicy,
                                         const RegPressureTracker &RPTracker,
                                         SchedCandidate &Cand, bool &IsPending,
                                         bool IsBottomUp) {
  const SIRegisterInfo *SRI = static_cast<const SIRegisterInfo *>(TRI);
  ArrayRef<unsigned> Pressure = RPTracker.getRegSetPressureAtPos();
  unsigned SGPRPressure = 0;
  unsigned VGPRPressure = 0;
  IsPending = false;
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
  LLVM_DEBUG(dbgs() << "Available Q:\n");
  ReadyQueue &AQ = Zone.Available;
  for (SUnit *SU : AQ) {

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
      LLVM_DEBUG(printCandidateDecision(Cand, TryCand));
      Cand.setBest(TryCand);
    } else {
      printCandidateDecision(TryCand, Cand);
    }
  }

  if (!shouldCheckPending(Zone, SchedModel))
    return;

  LLVM_DEBUG(dbgs() << "Pending Q:\n");
  ReadyQueue &PQ = Zone.Pending;
  for (SUnit *SU : PQ) {

    SchedCandidate TryCand(ZonePolicy);
    initCandidate(TryCand, SU, Zone.isTop(), RPTracker, SRI, SGPRPressure,
                  VGPRPressure, IsBottomUp);
    // Pass SchedBoundary only when comparing nodes from the same boundary.
    SchedBoundary *ZoneArg = Cand.AtTop == TryCand.AtTop ? &Zone : nullptr;
    tryPendingCandidate(Cand, TryCand, ZoneArg);
    if (TryCand.Reason != NoCand) {
      // Initialize resource delta if needed in case future heuristics query it.
      if (TryCand.ResDelta == SchedResourceDelta())
        TryCand.initResourceDelta(Zone.DAG, SchedModel);
      LLVM_DEBUG(printCandidateDecision(Cand, TryCand));
      IsPending = true;
      Cand.setBest(TryCand);
    } else {
      printCandidateDecision(TryCand, Cand);
    }
  }
}

// This function is mostly cut and pasted from
// GenericScheduler::pickNodeBidirectional()
SUnit *GCNSchedStrategy::pickNodeBidirectional(bool &IsTopNode,
                                               bool &PickedPending) {
  // Schedule as far as possible in the direction of no choice. This is most
  // efficient, but also provides the best heuristics for CriticalPSets.
  if (SUnit *SU = pickOnlyChoice(Bot, SchedModel)) {
    IsTopNode = false;
    return SU;
  }
  if (SUnit *SU = pickOnlyChoice(Top, SchedModel)) {
    IsTopNode = true;
    return SU;
  }
  // Set the bottom-up policy based on the state of the current bottom zone
  // and the instructions outside the zone, including the top zone.
  CandPolicy BotPolicy;
  setPolicy(BotPolicy, /*IsPostRA=*/false, Bot, &Top);
  // Set the top-down policy based on the state of the current top zone and
  // the instructions outside the zone, including the bottom zone.
  CandPolicy TopPolicy;
  setPolicy(TopPolicy, /*IsPostRA=*/false, Top, &Bot);

  bool BotPending = false;
  // See if BotCand is still valid (because we previously scheduled from Top).
  LLVM_DEBUG(dbgs() << "Picking from Bot:\n");
  if (!BotCand.isValid() || BotCand.SU->isScheduled ||
      BotCand.Policy != BotPolicy) {
    BotCand.reset(CandPolicy());
    pickNodeFromQueue(Bot, BotPolicy, DAG->getBotRPTracker(), BotCand,
                      BotPending,
                      /*IsBottomUp=*/true);
    assert(BotCand.Reason != NoCand && "failed to find the first candidate");
  } else {
    LLVM_DEBUG(traceCandidate(BotCand));
#ifndef NDEBUG
    if (VerifyScheduling) {
      SchedCandidate TCand;
      TCand.reset(CandPolicy());
      pickNodeFromQueue(Bot, BotPolicy, DAG->getBotRPTracker(), TCand,
                        BotPending,
                        /*IsBottomUp=*/true);
      assert(TCand.SU == BotCand.SU &&
             "Last pick result should correspond to re-picking right now");
    }
#endif
  }

  bool TopPending = false;
  // Check if the top Q has a better candidate.
  LLVM_DEBUG(dbgs() << "Picking from Top:\n");
  if (!TopCand.isValid() || TopCand.SU->isScheduled ||
      TopCand.Policy != TopPolicy) {
    TopCand.reset(CandPolicy());
    pickNodeFromQueue(Top, TopPolicy, DAG->getTopRPTracker(), TopCand,
                      TopPending,
                      /*IsBottomUp=*/false);
    assert(TopCand.Reason != NoCand && "failed to find the first candidate");
  } else {
    LLVM_DEBUG(traceCandidate(TopCand));
#ifndef NDEBUG
    if (VerifyScheduling) {
      SchedCandidate TCand;
      TCand.reset(CandPolicy());
      pickNodeFromQueue(Top, TopPolicy, DAG->getTopRPTracker(), TCand,
                        TopPending,
                        /*IsBottomUp=*/false);
      assert(TCand.SU == TopCand.SU &&
             "Last pick result should correspond to re-picking right now");
    }
#endif
  }

  // Pick best from BotCand and TopCand.
  LLVM_DEBUG(dbgs() << "Top Cand: "; traceCandidate(TopCand);
             dbgs() << "Bot Cand: "; traceCandidate(BotCand););
  SchedCandidate Cand = BotPending ? TopCand : BotCand;
  SchedCandidate TryCand = BotPending ? BotCand : TopCand;
  PickedPending = BotPending && TopPending;

  TryCand.Reason = NoCand;
  if (BotPending || TopPending) {
    PickedPending |= tryPendingCandidate(Cand, TopCand, nullptr);
  } else {
    tryCandidate(Cand, TryCand, nullptr);
  }

  if (TryCand.Reason != NoCand) {
    Cand.setBest(TryCand);
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
  bool PickedPending;
  SUnit *SU;
  do {
    PickedPending = false;
    if (RegionPolicy.OnlyTopDown) {
      SU = pickOnlyChoice(Top, SchedModel);
      if (!SU) {
        CandPolicy NoPolicy;
        TopCand.reset(NoPolicy);
        pickNodeFromQueue(Top, NoPolicy, DAG->getTopRPTracker(), TopCand,
                          PickedPending,
                          /*IsBottomUp=*/false);
        assert(TopCand.Reason != NoCand && "failed to find a candidate");
        SU = TopCand.SU;
      }
      IsTopNode = true;
    } else if (RegionPolicy.OnlyBottomUp) {
      SU = pickOnlyChoice(Bot, SchedModel);
      if (!SU) {
        CandPolicy NoPolicy;
        BotCand.reset(NoPolicy);
        pickNodeFromQueue(Bot, NoPolicy, DAG->getBotRPTracker(), BotCand,
                          PickedPending,
                          /*IsBottomUp=*/true);
        assert(BotCand.Reason != NoCand && "failed to find a candidate");
        SU = BotCand.SU;
      }
      IsTopNode = false;
    } else {
      SU = pickNodeBidirectional(IsTopNode, PickedPending);
    }
  } while (SU->isScheduled);

  if (PickedPending) {
    unsigned ReadyCycle = IsTopNode ? SU->TopReadyCycle : SU->BotReadyCycle;
    SchedBoundary &Zone = IsTopNode ? Top : Bot;
    unsigned CurrentCycle = Zone.getCurrCycle();
    if (ReadyCycle > CurrentCycle)
      Zone.bumpCycle(ReadyCycle);

    // FIXME: checkHazard() doesn't give information about which cycle the
    // hazard will resolve so just keep bumping the cycle by 1. This could be
    // made more efficient if checkHazard() returned more details.
    while (Zone.checkHazard(SU))
      Zone.bumpCycle(Zone.getCurrCycle() + 1);

    Zone.releasePending();
  }

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

bool GCNSchedStrategy::tryPendingCandidate(SchedCandidate &Cand,
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

  // Avoid exceeding the target's limit.
  if (DAG->isTrackingPressure() &&
      tryPressure(TryCand.RPDelta.Excess, Cand.RPDelta.Excess, TryCand, Cand,
                  RegExcess, TRI, DAG->MF))
    return TryCand.Reason != NoCand;

  // Avoid increasing the max critical pressure in the scheduled region.
  if (DAG->isTrackingPressure() &&
      tryPressure(TryCand.RPDelta.CriticalMax, Cand.RPDelta.CriticalMax,
                  TryCand, Cand, RegCritical, TRI, DAG->MF))
    return TryCand.Reason != NoCand;

  bool SameBoundary = Zone != nullptr;
  if (SameBoundary) {
    TryCand.initResourceDelta(DAG, SchedModel);
    if (tryLess(TryCand.ResDelta.CritResources, Cand.ResDelta.CritResources,
                TryCand, Cand, ResourceReduce))
      return TryCand.Reason != NoCand;
    if (tryGreater(TryCand.ResDelta.DemandedResources,
                   Cand.ResDelta.DemandedResources, TryCand, Cand,
                   ResourceDemand))
      return TryCand.Reason != NoCand;
  }

  return false;
}

GCNMaxOccupancySchedStrategy::GCNMaxOccupancySchedStrategy(
    const MachineSchedContext *C, bool IsLegacyScheduler)
    : GCNSchedStrategy(C) {
  SchedStages.push_back(GCNSchedStageID::OccInitialSchedule);
  if (!DisableRewriteMFMAFormSchedStage)
    SchedStages.push_back(GCNSchedStageID::RewriteMFMAForm);
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
  unsigned CandZoneCluster = Cand.AtTop ? TopClusterID : BotClusterID;
  unsigned TryCandZoneCluster = TryCand.AtTop ? TopClusterID : BotClusterID;
  bool CandIsClusterSucc =
      isTheSameCluster(CandZoneCluster, Cand.SU->ParentClusterIdx);
  bool TryCandIsClusterSucc =
      isTheSameCluster(TryCandZoneCluster, TryCand.SU->ParentClusterIdx);
  if (tryGreater(TryCandIsClusterSucc, CandIsClusterSucc, TryCand, Cand,
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
  unsigned CandZoneCluster = Cand.AtTop ? TopClusterID : BotClusterID;
  unsigned TryCandZoneCluster = TryCand.AtTop ? TopClusterID : BotClusterID;
  bool CandIsClusterSucc =
      isTheSameCluster(CandZoneCluster, Cand.SU->ParentClusterIdx);
  bool TryCandIsClusterSucc =
      isTheSameCluster(TryCandZoneCluster, TryCand.SU->ParentClusterIdx);
  if (tryGreater(TryCandIsClusterSucc, CandIsClusterSucc, TryCand, Cand,
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
  case GCNSchedStageID::RewriteMFMAForm:
    return std::make_unique<RewriteMFMAFormStage>(SchedStageID, *this);
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
  if (Regions[RegionIdx].first == Regions[RegionIdx].second)
    return llvm::getRegPressure(MRI, LiveIns[RegionIdx]);
  GCNDownwardRPTracker RPTracker(*LIS);
  RPTracker.advance(Regions[RegionIdx].first, Regions[RegionIdx].second,
                    &LiveIns[RegionIdx]);
  return RPTracker.moveMaxPressure();
}

static MachineInstr *getLastMIForRegion(MachineBasicBlock::iterator RegionBegin,
                                        MachineBasicBlock::iterator RegionEnd) {
  assert(RegionBegin != RegionEnd && "Region must not be empty");
  return &*skipDebugInstructionsBackward(std::prev(RegionEnd), RegionBegin);
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
  for (auto &[RegionBegin, RegionEnd] : reverse(Regions))
    RegionFirstMIs.push_back(
        &*skipDebugInstructionsForward(RegionBegin, RegionEnd));

  return getLiveRegMap(RegionFirstMIs, /*After=*/false, *LIS);
}

DenseMap<MachineInstr *, GCNRPTracker::LiveRegSet>
GCNScheduleDAGMILive::getRegionLiveOutMap() const {
  assert(!Regions.empty());
  std::vector<MachineInstr *> RegionLastMIs;
  RegionLastMIs.reserve(Regions.size());
  for (auto &[RegionBegin, RegionEnd] : reverse(Regions)) {
    // Skip empty regions.
    if (RegionBegin == RegionEnd)
      continue;
    RegionLastMIs.push_back(getLastMIForRegion(RegionBegin, RegionEnd));
  }
  return getLiveRegMap(RegionLastMIs, /*After=*/true, *LIS);
}

void RegionPressureMap::buildLiveRegMap() {
  IdxToInstruction.clear();

  RegionLiveRegMap =
      IsLiveOut ? DAG->getRegionLiveOutMap() : DAG->getRegionLiveInMap();
  for (unsigned I = 0; I < DAG->Regions.size(); I++) {
    auto &[RegionBegin, RegionEnd] = DAG->Regions[I];
    // Skip empty regions.
    if (RegionBegin == RegionEnd)
      continue;
    MachineInstr *RegionKey =
        IsLiveOut ? getLastMIForRegion(RegionBegin, RegionEnd) : &*RegionBegin;
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
  RegionsWithIGLPInstrs.resize(Regions.size());
  RegionsWithHighRP.reset();
  RegionsWithExcessRP.reset();
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

#ifdef DUMP_MAX_REG_PRESSURE
  if (PrintMaxRPRegUsageBeforeScheduler) {
    dumpMaxRegPressure(MF, GCNRegPressure::VGPR, *LIS, MLI);
    dumpMaxRegPressure(MF, GCNRegPressure::SGPR, *LIS, MLI);
    LIS->dump();
  }
#endif

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
      Stage->advanceRegion();
      exitRegion();
    }

    Stage->finalizeGCNSchedStage();
  }

#ifdef DUMP_MAX_REG_PRESSURE
  if (PrintMaxRPRegUsageAfterScheduler) {
    dumpMaxRegPressure(MF, GCNRegPressure::VGPR, *LIS, MLI);
    dumpMaxRegPressure(MF, GCNRegPressure::SGPR, *LIS, MLI);
    LIS->dump();
  }
#endif
}

#ifndef NDEBUG
raw_ostream &llvm::operator<<(raw_ostream &OS, const GCNSchedStageID &StageID) {
  switch (StageID) {
  case GCNSchedStageID::OccInitialSchedule:
    OS << "Max Occupancy Initial Schedule";
    break;
  case GCNSchedStageID::RewriteMFMAForm:
    OS << "Instruction Rewriting Reschedule";
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

void RewriteMFMAFormStage::findReachingDefs(
    MachineOperand &UseMO, LiveIntervals *LIS,
    SmallVectorImpl<SlotIndex> &DefIdxs) {
  MachineInstr *UseMI = UseMO.getParent();
  LiveInterval &UseLI = LIS->getInterval(UseMO.getReg());
  VNInfo *VNI = UseLI.getVNInfoAt(LIS->getInstructionIndex(*UseMI));

  // If the def is not a PHI, then it must be the only reaching def.
  if (!VNI->isPHIDef()) {
    DefIdxs.push_back(VNI->def);
    return;
  }

  SmallPtrSet<MachineBasicBlock *, 8> Visited = {UseMI->getParent()};
  SmallVector<MachineBasicBlock *, 8> Worklist;

  // Mark the predecessor blocks for traversal
  for (MachineBasicBlock *PredMBB : UseMI->getParent()->predecessors()) {
    Worklist.push_back(PredMBB);
    Visited.insert(PredMBB);
  }

  while (!Worklist.empty()) {
    MachineBasicBlock *CurrMBB = Worklist.pop_back_val();

    SlotIndex CurrMBBEnd = LIS->getMBBEndIdx(CurrMBB);
    VNInfo *VNI = UseLI.getVNInfoAt(CurrMBBEnd.getPrevSlot());

    MachineBasicBlock *DefMBB = LIS->getMBBFromIndex(VNI->def);

    // If there is a def in this block, then add it to the list. This is the
    // reaching def of this path.
    if (!VNI->isPHIDef()) {
      DefIdxs.push_back(VNI->def);
      continue;
    }

    for (MachineBasicBlock *PredMBB : DefMBB->predecessors()) {
      if (Visited.insert(PredMBB).second)
        Worklist.push_back(PredMBB);
    }
  }
}

void RewriteMFMAFormStage::findReachingUses(
    MachineInstr *DefMI, LiveIntervals *LIS,
    SmallVectorImpl<MachineOperand *> &ReachingUses) {
  SlotIndex DefIdx = LIS->getInstructionIndex(*DefMI);
  for (MachineOperand &UseMO :
       DAG.MRI.use_nodbg_operands(DefMI->getOperand(0).getReg())) {
    SmallVector<SlotIndex, 8> ReachingDefIndexes;
    findReachingDefs(UseMO, LIS, ReachingDefIndexes);

    // If we find a use that contains this DefMI in its reachingDefs, then it is
    // a reaching use.
    if (any_of(ReachingDefIndexes, [DefIdx](SlotIndex RDIdx) {
          return SlotIndex::isSameInstr(RDIdx, DefIdx);
        }))
      ReachingUses.push_back(&UseMO);
  }
}

bool RewriteMFMAFormStage::initGCNSchedStage() {
  // We only need to run this pass if the architecture supports AGPRs.
  // Additionally, we don't use AGPRs at occupancy levels above 1 so there
  // is no need for this pass in that case, either.
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  if (!ST.hasGFX90AInsts() || MFI.getMinWavesPerEU() > 1)
    return false;

  RegionsWithExcessArchVGPR.resize(DAG.Regions.size());
  RegionsWithExcessArchVGPR.reset();
  for (unsigned Region = 0; Region < DAG.Regions.size(); Region++) {
    GCNRegPressure PressureBefore = DAG.Pressure[Region];
    if (PressureBefore.getArchVGPRNum() > ST.getAddressableNumArchVGPRs())
      RegionsWithExcessArchVGPR[Region] = true;
  }

  if (RegionsWithExcessArchVGPR.none())
    return false;

  TII = ST.getInstrInfo();
  SRI = ST.getRegisterInfo();

  std::vector<std::pair<MachineInstr *, unsigned>> RewriteCands;
  DenseMap<MachineBasicBlock *, std::set<Register>> CopyForUse;
  SmallPtrSet<MachineInstr *, 8> CopyForDef;

  if (!initHeuristics(RewriteCands, CopyForUse, CopyForDef))
    return false;

  double Cost = getRewriteCost(RewriteCands, CopyForUse, CopyForDef);

  // If we haven't found the beneficial conditions, prefer the VGPR form which
  // may result in less cross RC copies.
  if (Cost > 0.0)
    return false;

  return rewrite(RewriteCands);
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
  // Aggressively try to reduce register pressure in the unclustered high RP
  // stage. Temporarily increase occupancy target in the region.
  TempTargetOccupancy = MFI.getMaxWavesPerEU() > DAG.MinOccupancy
                            ? InitialOccupancy + 1
                            : InitialOccupancy;
  IsAnyRegionScheduled = false;
  S.SGPRLimitBias = S.HighRPSGPRBias;
  S.VGPRLimitBias = S.HighRPVGPRBias;

  LLVM_DEBUG(
      dbgs()
      << "Retrying function scheduling without clustering. "
         "Aggressively try to reduce register pressure to achieve occupancy "
      << TempTargetOccupancy << ".\n");

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
#define REMAT_PREFIX "[PreRARemat] "
#define REMAT_DEBUG(X) LLVM_DEBUG(dbgs() << REMAT_PREFIX; X;)

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
Printable PreRARematStage::ScoredRemat::print() const {
  return Printable([&](raw_ostream &OS) {
    OS << '(' << MaxFreq << ", " << FreqDiff << ", " << RegionImpact << ')';
  });
}
#endif

bool PreRARematStage::initGCNSchedStage() {
  // FIXME: This pass will invalidate cached BBLiveInMap and MBBLiveIns for
  // regions inbetween the defs and region we sinked the def to. Will need to be
  // fixed if there is another pass after this pass.
  assert(!S.hasNextStage());

  if (!GCNSchedStage::initGCNSchedStage() || DAG.Regions.size() <= 1)
    return false;

  // Maps all MIs (except lone terminators, which are not part of any region) to
  // their parent region. Non-lone terminators are considered part of the region
  // they delimitate.
  DenseMap<MachineInstr *, unsigned> MIRegion(MF.getInstructionCount());

  // Before performing any IR modification record the parent region of each MI
  // and the parent MBB of each region.
  const unsigned NumRegions = DAG.Regions.size();
  for (unsigned I = 0; I < NumRegions; ++I) {
    RegionBoundaries Region = DAG.Regions[I];
    for (auto MI = Region.first; MI != Region.second; ++MI)
      MIRegion.insert({&*MI, I});
    MachineBasicBlock *ParentMBB = Region.first->getParent();
    if (Region.second != ParentMBB->end())
      MIRegion.insert({&*Region.second, I});
    RegionBB.push_back(ParentMBB);
  }

#ifndef NDEBUG
  auto PrintTargetRegions = [&]() -> void {
    if (TargetRegions.none()) {
      dbgs() << REMAT_PREFIX << "No target regions\n";
      return;
    }
    dbgs() << REMAT_PREFIX << "Target regions:\n";
    for (unsigned I : TargetRegions.set_bits())
      dbgs() << REMAT_PREFIX << "  [" << I << "] " << RPTargets[I] << '\n';
  };
  auto PrintRematReg = [&](const RematReg &Remat) -> Printable {
    return Printable([&, Remat](raw_ostream &OS) {
      // Concatenate all region numbers in which the register is unused and
      // live-through.
      bool HasLiveThroughRegion = false;
      OS << '[' << Remat.DefRegion << " -";
      for (unsigned I = 0; I < NumRegions; ++I) {
        if (Remat.isUnusedLiveThrough(I)) {
          if (HasLiveThroughRegion) {
            OS << ',';
          } else {
            OS << "- ";
            HasLiveThroughRegion = true;
          }
          OS << I;
        }
      }
      if (HasLiveThroughRegion)
        OS << " -";
      OS << "-> " << Remat.UseRegion << "] ";
      Remat.DefMI->print(OS, /*IsStandalone=*/true, /*SkipOpers=*/false,
                         /*SkipDebugLoc=*/false, /*AddNewLine=*/false);
    });
  };
#endif

  // Set an objective for the stage based on current RP in each region.
  REMAT_DEBUG({
    dbgs() << "Analyzing ";
    MF.getFunction().printAsOperand(dbgs(), false);
    dbgs() << ": ";
  });
  if (!setObjective()) {
    LLVM_DEBUG(dbgs() << "no objective to achieve, occupancy is maximal at "
                      << MFI.getMaxWavesPerEU() << '\n');
    return false;
  }
  LLVM_DEBUG({
    if (TargetOcc) {
      dbgs() << "increase occupancy from " << *TargetOcc - 1 << '\n';
    } else {
      dbgs() << "reduce spilling (minimum target occupancy is "
             << MFI.getMinWavesPerEU() << ")\n";
    }
    PrintTargetRegions();
  });

  if (!collectRematRegs(MIRegion)) {
    REMAT_DEBUG(dbgs() << "No rematerializable registers\n");
    return false;
  }
  const ScoredRemat::FreqInfo FreqInfo(MF, DAG);
  REMAT_DEBUG({
    dbgs() << "Rematerializable registers:\n";
    for (const RematReg &Remat : RematRegs)
      dbgs() << REMAT_PREFIX << "  " << PrintRematReg(Remat) << '\n';
    dbgs() << REMAT_PREFIX << "Region frequencies\n";
    for (auto [I, Freq] : enumerate(FreqInfo.Regions)) {
      dbgs() << REMAT_PREFIX << "  [" << I << "] ";
      if (Freq)
        dbgs() << Freq;
      else
        dbgs() << "unknown ";
      dbgs() << " | " << *DAG.Regions[I].first;
    }
  });

  SmallVector<ScoredRemat> ScoredRemats;
  for (RematReg &Remat : RematRegs)
    ScoredRemats.emplace_back(&Remat, FreqInfo, DAG);

// Rematerialize registers in successive rounds until all RP targets are
// satisifed or until we run out of rematerialization candidates.
#ifndef NDEBUG
  unsigned RoundNum = 0;
#endif
  BitVector RecomputeRP(NumRegions);
  do {
    assert(!ScoredRemats.empty() && "no more remat candidates");

    // (Re-)Score and (re-)sort all remats in increasing score order.
    for (ScoredRemat &Remat : ScoredRemats)
      Remat.update(TargetRegions, RPTargets, FreqInfo, !TargetOcc);
    sort(ScoredRemats);

    REMAT_DEBUG({
      dbgs() << "==== ROUND " << RoundNum++ << " ====\n"
             << REMAT_PREFIX
             << "Candidates with non-null score, in rematerialization order:\n";
      for (const ScoredRemat &RematDecision : reverse(ScoredRemats)) {
        if (RematDecision.hasNullScore())
          break;
        dbgs() << REMAT_PREFIX << "  " << RematDecision.print() << " | "
               << *RematDecision.Remat->DefMI;
      }
      PrintTargetRegions();
    });

    RecomputeRP.reset();
    unsigned RematIdx = ScoredRemats.size();

    // Rematerialize registers in decreasing score order until we estimate
    // that all RP targets are satisfied or until rematerialization candidates
    // are no longer useful to decrease RP.
    for (; RematIdx && TargetRegions.any(); --RematIdx) {
      const ScoredRemat &Candidate = ScoredRemats[RematIdx - 1];
      // Stop rematerializing on encountering a null score. Since scores
      // monotonically decrease as we rematerialize, we know there is nothing
      // useful left to do in such cases, even if we were to re-score.
      if (Candidate.hasNullScore()) {
        RematIdx = 0;
        break;
      }

      RematReg &Remat = *Candidate.Remat;
      // When previous rematerializations in this round have already satisfied
      // RP targets in all regions this rematerialization can impact, we have a
      // good indication that our scores have diverged significantly from
      // reality, in which case we interrupt this round and re-score. This also
      // ensures that every rematerialization we perform is possibly impactful
      // in at least one target region.
      if (!Remat.maybeBeneficial(TargetRegions, RPTargets))
        break;

      REMAT_DEBUG(dbgs() << "** REMAT " << PrintRematReg(Remat) << '\n';);
      // Every rematerialization we do here is likely to move the instruction
      // into a higher frequency region, increasing the total sum latency of the
      // instruction itself. This is acceptable if we are eliminating a spill in
      // the process, but when the goal is increasing occupancy we get nothing
      // out of rematerialization if occupancy is not increased in the end; in
      // such cases we want to roll back the rematerialization.
      RollbackInfo *Rollback =
          TargetOcc ? &Rollbacks.emplace_back(&Remat) : nullptr;
      rematerialize(Remat, RecomputeRP, Rollback);
      unsetSatisifedRPTargets(Remat.Live);
    }

    REMAT_DEBUG({
      if (!TargetRegions.any()) {
        dbgs() << "** Interrupt round on all targets achieved\n";
      } else if (RematIdx) {
        dbgs() << "** Interrupt round on stale score for "
               << *ScoredRemats[RematIdx - 1].Remat->DefMI;
      } else {
        dbgs() << "** Stop on exhausted rematerialization candidates\n";
      }
    });

    // Peel off registers we already rematerialized from the vector's tail.
    ScoredRemats.truncate(RematIdx);
  } while ((updateAndVerifyRPTargets(RecomputeRP) || TargetRegions.any()) &&
           !ScoredRemats.empty());
  if (RescheduleRegions.none())
    return false;

  // Commit all pressure changes to the DAG and compute minimum achieved
  // occupancy in impacted regions.
  REMAT_DEBUG(dbgs() << "==== REMAT RESULTS ====\n");
  unsigned DynamicVGPRBlockSize = MFI.getDynamicVGPRBlockSize();
  for (unsigned I : RescheduleRegions.set_bits()) {
    DAG.Pressure[I] = RPTargets[I].getCurrentRP();
    REMAT_DEBUG(dbgs() << '[' << I << "] Achieved occupancy "
                       << DAG.Pressure[I].getOccupancy(ST, DynamicVGPRBlockSize)
                       << " (" << RPTargets[I] << ")\n");
  }
  AchievedOcc = MFI.getMaxWavesPerEU();
  for (const GCNRegPressure &RP : DAG.Pressure) {
    AchievedOcc =
        std::min(AchievedOcc, RP.getOccupancy(ST, DynamicVGPRBlockSize));
  }

  REMAT_DEBUG({
    dbgs() << "Retrying function scheduling with new min. occupancy of "
           << AchievedOcc << " from rematerializing (original was "
           << DAG.MinOccupancy;
    if (TargetOcc)
      dbgs() << ", target was " << *TargetOcc;
    dbgs() << ")\n";
  });

  DAG.setTargetOccupancy(getStageTargetOccupancy());
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
    assert(IsAnyRegionScheduled);
    LLVM_DEBUG(dbgs() << StageID
                      << " stage successfully increased occupancy to "
                      << DAG.MinOccupancy << '\n');
  } else if (!IsAnyRegionScheduled) {
    assert(DAG.MinOccupancy == InitialOccupancy);
    LLVM_DEBUG(dbgs() << StageID
                      << ": No regions scheduled, min occupancy stays at "
                      << DAG.MinOccupancy << ", MFI occupancy stays at "
                      << MFI.getOccupancy() << ".\n");
  }

  GCNSchedStage::finalizeGCNSchedStage();
}

bool GCNSchedStage::initGCNRegion() {
  // Skip empty scheduling region.
  if (DAG.begin() == DAG.end())
    return false;

  // Check whether this new region is also a new block.
  if (DAG.RegionBegin->getParent() != CurrentMBB)
    setupNewBlock();

  unsigned NumRegionInstrs = std::distance(DAG.begin(), DAG.end());
  DAG.enterRegion(CurrentMBB, DAG.begin(), DAG.end(), NumRegionInstrs);

  // Skip regions with 1 schedulable instruction.
  if (DAG.begin() == std::prev(DAG.end()))
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
  // Only reschedule regions that have excess register pressure (i.e. spilling)
  // or had minimum occupancy at the beginning of the stage (as long as
  // rescheduling of previous regions did not make occupancy drop back down to
  // the initial minimum).
  unsigned DynamicVGPRBlockSize = DAG.MFI.getDynamicVGPRBlockSize();
  // If no region has been scheduled yet, the DAG has not yet been updated with
  // the occupancy target. So retrieve it from the temporary.
  unsigned CurrentTargetOccupancy =
      IsAnyRegionScheduled ? DAG.MinOccupancy : TempTargetOccupancy;
  if (!DAG.RegionsWithExcessRP[RegionIdx] &&
      (CurrentTargetOccupancy <= InitialOccupancy ||
       DAG.Pressure[RegionIdx].getOccupancy(ST, DynamicVGPRBlockSize) !=
           InitialOccupancy))
    return false;

  bool IsSchedulingThisRegion = GCNSchedStage::initGCNRegion();
  // If this is the first region scheduled during this stage, make the target
  // occupancy changes in the DAG and MFI.
  if (!IsAnyRegionScheduled && IsSchedulingThisRegion) {
    IsAnyRegionScheduled = true;
    if (MFI.getMaxWavesPerEU() > DAG.MinOccupancy)
      DAG.setTargetOccupancy(TempTargetOccupancy);
  }
  return IsSchedulingThisRegion;
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
}

void PreRARematStage::finalizeGCNRegion() {
  GCNSchedStage::finalizeGCNRegion();
  // When the goal is to increase occupancy, all regions must reach the target
  // occupancy for rematerializations to be possibly useful, otherwise we will
  // just hurt latency for no benefit. If minimum occupancy drops below the
  // target there is no point in trying to re-schedule further regions.
  if (!TargetOcc)
    return;
  RegionReverts.emplace_back(RegionIdx, Unsched, PressureBefore);
  if (DAG.MinOccupancy < *TargetOcc) {
    REMAT_DEBUG(dbgs() << "Region " << RegionIdx
                       << " cannot meet occupancy target, interrupting "
                          "re-scheduling in all regions\n");
    RescheduleRegions.reset();
  }
}

void GCNSchedStage::checkScheduling() {
  // Check the results of scheduling.
  PressureAfter = DAG.getRealRegPressure(RegionIdx);

  LLVM_DEBUG(dbgs() << "Pressure after scheduling: " << print(PressureAfter));
  LLVM_DEBUG(dbgs() << "Region: " << RegionIdx << ".\n");

  unsigned DynamicVGPRBlockSize = DAG.MFI.getDynamicVGPRBlockSize();

  if (PressureAfter.getSGPRNum() <= S.SGPRCriticalLimit &&
      PressureAfter.getVGPRNum(ST.hasGFX90AInsts()) <= S.VGPRCriticalLimit) {
    DAG.Pressure[RegionIdx] = PressureAfter;

    // Early out if we have achieved the occupancy target.
    LLVM_DEBUG(dbgs() << "Pressure in desired limits, done.\n");
    return;
  }

  unsigned TargetOccupancy = std::min(
      S.getTargetOccupancy(), ST.getOccupancyWithWorkGroupSizes(MF).second);
  unsigned WavesAfter = std::min(
      TargetOccupancy, PressureAfter.getOccupancy(ST, DynamicVGPRBlockSize));
  unsigned WavesBefore = std::min(
      TargetOccupancy, PressureBefore.getOccupancy(ST, DynamicVGPRBlockSize));
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
    modifyRegionSchedule(RegionIdx, DAG.BB, Unsched);
    std::tie(DAG.RegionBegin, DAG.RegionEnd) = DAG.Regions[RegionIdx];
  } else {
    DAG.Pressure[RegionIdx] = PressureAfter;
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
  if (DAG.MFI.isDynamicVGPREnabled()) {
    unsigned BlocksBefore = AMDGPU::IsaInfo::getAllocatedNumVGPRBlocks(
        &ST, DAG.MFI.getDynamicVGPRBlockSize(),
        PressureBefore.getVGPRNum(false));
    unsigned BlocksAfter = AMDGPU::IsaInfo::getAllocatedNumVGPRBlocks(
        &ST, DAG.MFI.getDynamicVGPRBlockSize(),
        PressureAfter.getVGPRNum(false));
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
  if ((WavesAfter <=
           PressureBefore.getOccupancy(ST, DAG.MFI.getDynamicVGPRBlockSize()) &&
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
  unsigned WavesBefore = std::min(
      S.getTargetOccupancy(),
      PressureBefore.getOccupancy(ST, DAG.MFI.getDynamicVGPRBlockSize()));
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
  // When trying to increase occupancy (TargetOcc == true) the stage manages
  // region reverts globally (all or none), so we always return false here.
  return !TargetOcc && mayCauseSpilling(WavesAfter);
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

void GCNSchedStage::modifyRegionSchedule(unsigned RegionIdx,
                                         MachineBasicBlock *MBB,
                                         ArrayRef<MachineInstr *> MIOrder) {
  assert(static_cast<size_t>(std::distance(DAG.Regions[RegionIdx].first,
                                           DAG.Regions[RegionIdx].second)) ==
             MIOrder.size() &&
         "instruction number mismatch");
  if (MIOrder.empty())
    return;

  LLVM_DEBUG(dbgs() << "Reverting scheduling for region " << RegionIdx << '\n');

  // Reconstruct MI sequence by moving instructions in desired order before
  // the current region's start.
  MachineBasicBlock::iterator RegionEnd = DAG.Regions[RegionIdx].first;
  for (MachineInstr *MI : MIOrder) {
    // Either move the next MI in order before the end of the region or move the
    // region end past the MI if it is at the correct position.
    MachineBasicBlock::iterator MII = MI->getIterator();
    if (MII != RegionEnd) {
      // Will subsequent splice move MI up past a non-debug instruction?
      bool NonDebugReordered =
          !MI->isDebugInstr() &&
          skipDebugInstructionsForward(RegionEnd, MII) != MII;
      MBB->splice(RegionEnd, MBB, MI);
      // Only update LiveIntervals information if non-debug instructions are
      // reordered. Otherwise debug instructions could cause code generation to
      // change.
      if (NonDebugReordered)
        DAG.LIS->handleMove(*MI, true);
    } else {
      ++RegionEnd;
    }
    if (MI->isDebugInstr()) {
      LLVM_DEBUG(dbgs() << "Scheduling " << *MI);
      continue;
    }

    // Reset read-undef flags and update them later.
    for (MachineOperand &Op : MI->all_defs())
      Op.setIsUndef(false);
    RegisterOperands RegOpers;
    RegOpers.collect(*MI, *DAG.TRI, DAG.MRI, DAG.ShouldTrackLaneMasks, false);
    if (DAG.ShouldTrackLaneMasks) {
      // Adjust liveness and add missing dead+read-undef flags.
      SlotIndex SlotIdx = DAG.LIS->getInstructionIndex(*MI).getRegSlot();
      RegOpers.adjustLaneLiveness(*DAG.LIS, DAG.MRI, SlotIdx, MI);
    } else {
      // Adjust for missing dead-def flags.
      RegOpers.detectDeadDefs(*MI, *DAG.LIS);
    }
    LLVM_DEBUG(dbgs() << "Scheduling " << *MI);
  }

  // The region end doesn't change throughout scheduling since it itself is
  // outside the region (whether that is a MBB end or a terminator MI).
  assert(RegionEnd == DAG.Regions[RegionIdx].second && "region end mismatch");
  DAG.Regions[RegionIdx].first = MIOrder.front();
}

bool RewriteMFMAFormStage::isRewriteCandidate(MachineInstr *MI) const {

  if (!static_cast<const SIInstrInfo *>(DAG.TII)->isMAI(*MI))
    return false;
  return AMDGPU::getMFMASrcCVDstAGPROp(MI->getOpcode()) != -1;
}

bool RewriteMFMAFormStage::initHeuristics(
    std::vector<std::pair<MachineInstr *, unsigned>> &RewriteCands,
    DenseMap<MachineBasicBlock *, std::set<Register>> &CopyForUse,
    SmallPtrSetImpl<MachineInstr *> &CopyForDef) {
  bool Changed = false;

  // Prepare for the heuristics
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      if (!isRewriteCandidate(&MI))
        continue;

      int ReplacementOp = AMDGPU::getMFMASrcCVDstAGPROp(MI.getOpcode());
      assert(ReplacementOp != -1);

      RewriteCands.push_back({&MI, MI.getOpcode()});
      MI.setDesc(TII->get(ReplacementOp));

      MachineOperand *Src2 = TII->getNamedOperand(MI, AMDGPU::OpName::src2);
      if (Src2->isReg()) {
        SmallVector<SlotIndex, 8> Src2ReachingDefs;
        findReachingDefs(*Src2, DAG.LIS, Src2ReachingDefs);

        // For any definition of the src2 register which is non-MFMA, we
        // insert a copy.
        for (SlotIndex RDIdx : Src2ReachingDefs) {
          MachineInstr *RD = DAG.LIS->getInstructionFromIndex(RDIdx);
          if (!TII->isMAI(*RD))
            CopyForDef.insert(RD);
        }
      }

      MachineOperand &Dst = MI.getOperand(0);
      SmallVector<MachineOperand *, 8> DstReachingUses;

      findReachingUses(&MI, DAG.LIS, DstReachingUses);

      for (MachineOperand *RUOp : DstReachingUses) {
        if (TII->isMAI(*RUOp->getParent()))
          continue;

        // For any user of the result of the MFMA which is not an MFMA, we
        // insert a copy. For a given register, we will only insert one copy
        // per user block.
        CopyForUse[RUOp->getParent()->getParent()].insert(RUOp->getReg());

        SmallVector<SlotIndex, 8> DstUsesReachingDefs;
        findReachingDefs(*RUOp, DAG.LIS, DstUsesReachingDefs);

        for (SlotIndex RDIndex : DstUsesReachingDefs) {
          MachineInstr *RD = DAG.LIS->getInstructionFromIndex(RDIndex);
          if (TII->isMAI(*RD))
            continue;

          // For any definition of the user of the MFMA which is not an MFMA,
          // we insert a copy. We do this to transform all the reaching defs
          // of this use to AGPR. By doing this, we can insert a copy from
          // AGPR to VGPR at the user rather than after the MFMA.
          CopyForDef.insert(RD);
        }
      }

      // Do the rewrite to allow for updated RP calculation.
      const TargetRegisterClass *VDefRC = DAG.MRI.getRegClass(Dst.getReg());
      const TargetRegisterClass *ADefRC = SRI->getEquivalentAGPRClass(VDefRC);
      DAG.MRI.setRegClass(Dst.getReg(), ADefRC);
      if (Src2->isReg()) {
        // Have to get src types separately since subregs may cause C and D
        // registers to be different types even though the actual operand is
        // the same size.
        const TargetRegisterClass *VUseRC = DAG.MRI.getRegClass(Src2->getReg());
        const TargetRegisterClass *AUseRC = SRI->getEquivalentAGPRClass(VUseRC);
        DAG.MRI.setRegClass(Src2->getReg(), AUseRC);
      }
      Changed = true;
    }
  }

  return Changed;
}

double RewriteMFMAFormStage::getRewriteCost(
    const std::vector<std::pair<MachineInstr *, unsigned>> &RewriteCands,
    const DenseMap<MachineBasicBlock *, std::set<Register>> &CopyForUse,
    const SmallPtrSetImpl<MachineInstr *> &CopyForDef) {
  MachineBlockFrequencyInfo *MBFI = DAG.MBFI;

  double BestSpillCost = 0.0;
  double Cost = 0.0;

  std::pair<unsigned, unsigned> MaxVectorRegs =
      ST.getMaxNumVectorRegs(MF.getFunction());
  unsigned ArchVGPRThreshold = MaxVectorRegs.first;
  unsigned AGPRThreshold = MaxVectorRegs.second;
  unsigned CombinedThreshold = ST.getMaxNumVGPRs(MF);

  for (unsigned Region = 0; Region < DAG.Regions.size(); Region++) {
    if (!RegionsWithExcessArchVGPR[Region])
      continue;

    GCNRegPressure &PressureBefore = DAG.Pressure[Region];
    unsigned SpillCostBefore = PressureBefore.getVGPRSpills(
        MF, ArchVGPRThreshold, AGPRThreshold, CombinedThreshold);
    LLVM_DEBUG(dbgs() << "RewriteMFMA: Region " << Region
                      << " spill cost before: " << SpillCostBefore << "\n");

    // For the cases we care about (i.e. ArchVGPR usage is greater than the
    // addressable limit), rewriting alone should bring pressure to manageable
    // level. If we find any such region, then the rewrite is potentially
    // beneficial.
    GCNRegPressure PressureAfter = DAG.getRealRegPressure(Region);
    unsigned SpillCostAfter = PressureAfter.getVGPRSpills(
        MF, ArchVGPRThreshold, AGPRThreshold, CombinedThreshold);
    LLVM_DEBUG(dbgs() << "RewriteMFMA: Region " << Region
                      << " spill cost after: " << SpillCostAfter << "\n");

    MachineBasicBlock *MBB = DAG.Regions[Region].first->getParent();
    double BlockFreq = MBFI->getBlockFreqRelativeToEntryBlock(MBB);

    // This assumes perfect spilling / splitting -- using one spill / copy
    // instruction and one restoreFrom / copy for each excess register,
    double SpillCost = ((double)SpillCostAfter - (double)SpillCostBefore) * 2;

    // Also account for the block frequency.
    SpillCost *= BlockFreq;

    // If we have increased spilling in any block, just bail.
    if (SpillCost > 0.0)
      return SpillCost;

    if (SpillCost < BestSpillCost)
      BestSpillCost = SpillCost;
  }

  // Set the cost to the largest decrease in spill cost in order to not double
  // count spill reductions.
  LLVM_DEBUG(dbgs() << "RewriteMFMA: BestSpillCost: " << BestSpillCost << "\n");
  Cost = BestSpillCost;
  assert(Cost <= 0.0);

  // For each CopyForDef, increase the cost by the register size while
  // accounting for block frequency.
  double DefCopyCost = 0.0;
  for (MachineInstr *DefMI : CopyForDef) {
    Register DefReg = DefMI->getOperand(0).getReg();
    MachineBasicBlock *DefMBB = DefMI->getParent();
    double DefFreq = MBFI->getBlockFreqRelativeToEntryBlock(DefMBB);

    const TargetRegisterClass *RC = DAG.MRI.getRegClass(DefReg);
    DefCopyCost += (double)RC->getCopyCost() * DefFreq;
  }
  LLVM_DEBUG(dbgs() << "RewriteMFMA: Def copy Costs: " << DefCopyCost << "\n");

  // Account for CopyForUse copies in each block that the register is used.
  double UseCopyCost = 0.0;
  for (auto &[UseBlock, UseRegs] : CopyForUse) {
    uint64_t UseFreq = MBFI->getBlockFreqRelativeToEntryBlock(UseBlock);

    for (Register UseReg : UseRegs) {
      const TargetRegisterClass *RC = DAG.MRI.getRegClass(UseReg);
      UseCopyCost += (double)RC->getCopyCost() * UseFreq;
    }
  }

  LLVM_DEBUG(dbgs() << "RewriteMFMA: Use copy Costs: " << UseCopyCost << "\n");
  double CopyCost = UseCopyCost + DefCopyCost;

  // Reset the classes that were changed to AGPR for better RB analysis.
  // We must do rewriting after copy-insertion, as some defs of the register
  // may require VGPR.  Additionally, if we bail out and don't perform the
  // rewrite then these need to be restored anyway.
  for (auto &[MI, OriginalOpcode] : RewriteCands) {
    assert(TII->isMAI(*MI));
    const TargetRegisterClass *ADefRC =
        DAG.MRI.getRegClass(MI->getOperand(0).getReg());
    const TargetRegisterClass *VDefRC = SRI->getEquivalentVGPRClass(ADefRC);
    DAG.MRI.setRegClass(MI->getOperand(0).getReg(), VDefRC);
    MI->setDesc(TII->get(OriginalOpcode));

    MachineOperand *Src2 = TII->getNamedOperand(*MI, AMDGPU::OpName::src2);
    assert(Src2);

    // Have to get src types separately since subregs may cause C and D
    // registers to be different types even though the actual operand is
    // the same size.
    const TargetRegisterClass *AUseRC = DAG.MRI.getRegClass(Src2->getReg());
    const TargetRegisterClass *VUseRC = SRI->getEquivalentVGPRClass(AUseRC);
    DAG.MRI.setRegClass(Src2->getReg(), VUseRC);
  }

  return Cost + CopyCost;
}

bool RewriteMFMAFormStage::rewrite(
    const std::vector<std::pair<MachineInstr *, unsigned>> &RewriteCands) {
  DenseMap<MachineInstr *, unsigned> FirstMIToRegion;
  DenseMap<MachineInstr *, unsigned> LastMIToRegion;

  for (unsigned Region = 0; Region < DAG.Regions.size(); Region++) {
    RegionBoundaries Entry = DAG.Regions[Region];
    if (Entry.first == Entry.second)
      continue;

    FirstMIToRegion[&*Entry.first] = Region;
    if (Entry.second != Entry.first->getParent()->end())
      LastMIToRegion[&*Entry.second] = Region;
  }

  // Rewrite the MFMAs to AGPR, and insert any copies as needed.
  // The general assumption of the algorithm (and the previous cost calculation)
  // is that it is better to insert the copies in the MBB of the def of the src2
  // operands, and in the MBB of the user of the dest operands. This is based on
  // the assumption that the MFMAs are likely to appear in loop bodies, while
  // the src2 and dest operands are live-in / live-out of the loop. Due to this
  // design, the algorithm for finding copy insertion points is more
  // complicated.
  //
  // There are three main cases to handle: 1. the reaching defs of the src2
  // operands, 2. the reaching uses of the dst operands, and 3. the reaching
  // defs of the reaching uses of the dst operand.
  //
  // In the first case, we simply insert copies after each of the reaching
  // definitions. In the second case, we collect all the uses of a given dest
  // and organize them by MBB. Then, we insert 1 copy for each MBB before the
  // earliest use. Since the use may have multiple reaching defs, and since we
  // want to replace the register it is using with the result of the copy, we
  // must handle case 3. In the third case, we simply insert a copy after each
  // of the reaching defs to connect to the copy of the reaching uses of the dst
  // reg. This allows us to avoid inserting copies next to the MFMAs.
  //
  // While inserting the copies, we maintain a map of operands which will use
  // different regs (i.e. the result of the copies). For example, a case 1 src2
  // operand will use the register result of the copies after the reaching defs,
  // as opposed to the original register. Now that we have completed our copy
  // analysis and placement, we can bulk update the registers. We do this
  // separately as to avoid complicating the reachingDef and reachingUse
  // queries.
  //
  // While inserting the copies, we also maintain a list or registers which we
  // will want to reclassify as AGPR. After doing the copy insertion and the
  // register replacement, we can finally do the reclassification. This uses the
  // redef map, as the registers we are interested in reclassifying may be
  // replaced by the result of a copy. We must do this after the copy analysis
  // and placement as we must have an accurate redef map -- otherwise we may end
  // up creating illegal instructions.

  // The original registers of the MFMA that need to be reclassified as AGPR.
  DenseSet<Register> RewriteRegs;
  // The map of an original register in the MFMA to a new register (result of a
  // copy) that it should be replaced with.
  DenseMap<Register, Register> RedefMap;
  // The map of the original MFMA registers to the relevant MFMA operands.
  DenseMap<Register, DenseSet<MachineOperand *>> ReplaceMap;
  // The map of reaching defs for a given register -- to avoid duplicate copies.
  DenseMap<Register, SmallPtrSet<MachineInstr *, 8>> ReachingDefCopyMap;
  // The map of reaching uses for a given register by basic block -- to avoid
  // duplicate copies and to calculate per MBB insert pts.
  DenseMap<unsigned, DenseMap<Register, SmallPtrSet<MachineOperand *, 8>>>
      ReachingUseTracker;

  for (auto &[MI, OriginalOpcode] : RewriteCands) {
    int ReplacementOp = AMDGPU::getMFMASrcCVDstAGPROp(MI->getOpcode());
    if (ReplacementOp == -1)
      continue;
    MI->setDesc(TII->get(ReplacementOp));

    // Case 1: insert copies for the reaching defs of the Src2Reg.
    MachineOperand *Src2 = TII->getNamedOperand(*MI, AMDGPU::OpName::src2);
    if (Src2->isReg()) {
      Register Src2Reg = Src2->getReg();
      if (!Src2Reg.isVirtual())
        return false;

      Register MappedReg = Src2->getReg();
      SmallVector<SlotIndex, 8> Src2ReachingDefs;
      findReachingDefs(*Src2, DAG.LIS, Src2ReachingDefs);
      SmallSetVector<MachineInstr *, 8> Src2DefsReplace;

      for (SlotIndex RDIndex : Src2ReachingDefs) {
        MachineInstr *RD = DAG.LIS->getInstructionFromIndex(RDIndex);
        if (TII->isMAI(*RD))
          continue;

        // If there is a non mai reaching def, then we need a copy.
        Src2DefsReplace.insert(RD);
      }

      if (!Src2DefsReplace.empty()) {
        DenseMap<Register, Register>::iterator RI = RedefMap.find(Src2Reg);
        if (RI != RedefMap.end()) {
          MappedReg = RI->second;
        } else {
          assert(!ReachingDefCopyMap.contains(Src2Reg));
          const TargetRegisterClass *Src2RC = DAG.MRI.getRegClass(Src2Reg);
          const TargetRegisterClass *VGPRRC =
              SRI->getEquivalentVGPRClass(Src2RC);

          // Track the mapping of the original register to the new register.
          MappedReg = DAG.MRI.createVirtualRegister(VGPRRC);
          RedefMap[Src2Reg] = MappedReg;
        }

        // If none exists, create a copy from this reaching def.
        // We may have inserted a copy already in an earlier iteration.
        for (MachineInstr *RD : Src2DefsReplace) {
          // Do not create redundant copies.
          if (ReachingDefCopyMap[Src2Reg].insert(RD).second) {
            MachineInstrBuilder VGPRCopy =
                BuildMI(*RD->getParent(), std::next(RD->getIterator()),
                        RD->getDebugLoc(), TII->get(TargetOpcode::COPY))
                    .addDef(MappedReg, {}, 0)
                    .addUse(Src2Reg, {}, 0);
            DAG.LIS->InsertMachineInstrInMaps(*VGPRCopy);

            // If this reaching def was the last MI in the region, update the
            // region boundaries.
            if (LastMIToRegion.contains(RD)) {
              unsigned UpdateRegion = LastMIToRegion[RD];
              DAG.Regions[UpdateRegion].second = VGPRCopy;
              LastMIToRegion.erase(RD);
            }
          }
        }
      }

      // Track the register for reclassification
      RewriteRegs.insert(Src2Reg);

      // Always insert the operand for replacement. If this corresponds with a
      // chain of tied-def we may not see the VGPR requirement until later.
      ReplaceMap[Src2Reg].insert(Src2);
    }

    // Case 2 and Case 3: insert copies before the reaching uses of the dsts,
    // and after the reaching defs of the reaching uses of the dsts.

    MachineOperand *Dst = &MI->getOperand(0);
    Register DstReg = Dst->getReg();
    if (!DstReg.isVirtual())
      return false;

    Register MappedReg = DstReg;
    SmallVector<MachineOperand *, 8> DstReachingUses;

    SmallVector<MachineOperand *, 8> DstReachingUseCopies;
    SmallVector<MachineInstr *, 8> DstUseDefsReplace;

    findReachingUses(MI, DAG.LIS, DstReachingUses);

    for (MachineOperand *RUOp : DstReachingUses) {
      if (TII->isMAI(*RUOp->getParent()))
        continue;

      // If there is a non mai reaching use, then we need a copy.
      if (find(DstReachingUseCopies, RUOp) == DstReachingUseCopies.end())
        DstReachingUseCopies.push_back(RUOp);
      SmallVector<SlotIndex, 8> DstUsesReachingDefs;
      findReachingDefs(*RUOp, DAG.LIS, DstUsesReachingDefs);

      for (SlotIndex RDIndex : DstUsesReachingDefs) {
        MachineInstr *RD = DAG.LIS->getInstructionFromIndex(RDIndex);
        if (TII->isMAI(*RD))
          continue;

        // If there is a non mai reaching def of this reaching use, then we will
        // need a copy.
        if (find(DstUseDefsReplace, RD) == DstUseDefsReplace.end())
          DstUseDefsReplace.push_back(RD);
      }
    }

    if (!DstUseDefsReplace.empty()) {
      DenseMap<Register, Register>::iterator RI = RedefMap.find(DstReg);
      if (RI != RedefMap.end()) {
        MappedReg = RI->second;
      } else {
        assert(!ReachingDefCopyMap.contains(DstReg));
        const TargetRegisterClass *DstRC = DAG.MRI.getRegClass(DstReg);
        const TargetRegisterClass *VGPRRC = SRI->getEquivalentVGPRClass(DstRC);

        // Track the mapping of the original register to the new register.
        MappedReg = DAG.MRI.createVirtualRegister(VGPRRC);
        RedefMap[DstReg] = MappedReg;
      }

      // If none exists, create a copy from this reaching def.
      // We may have inserted a copy already in an earlier iteration.
      for (MachineInstr *RD : DstUseDefsReplace) {
        // Do not create reundant copies.
        if (ReachingDefCopyMap[DstReg].insert(RD).second) {
          MachineInstrBuilder VGPRCopy =
              BuildMI(*RD->getParent(), std::next(RD->getIterator()),
                      RD->getDebugLoc(), TII->get(TargetOpcode::COPY))
                  .addDef(MappedReg, {}, 0)
                  .addUse(DstReg, {}, 0);
          DAG.LIS->InsertMachineInstrInMaps(*VGPRCopy);

          // If this reaching def was the last MI in the region, update the
          // region boundaries.
          DenseMap<MachineInstr *, unsigned>::iterator LMI =
              LastMIToRegion.find(RD);
          if (LMI != LastMIToRegion.end()) {
            unsigned UpdateRegion = LMI->second;
            DAG.Regions[UpdateRegion].second = VGPRCopy;
            LastMIToRegion.erase(RD);
          }
        }
      }
    }

    DenseSet<MachineOperand *> &DstRegSet = ReplaceMap[DstReg];
    for (MachineOperand *RU : DstReachingUseCopies) {
      MachineBasicBlock *RUBlock = RU->getParent()->getParent();
      // Just keep track of the reaching use of this register by block. After we
      // have scanned all the MFMAs we can find optimal insert pts.
      if (RUBlock != MI->getParent()) {
        ReachingUseTracker[RUBlock->getNumber()][DstReg].insert(RU);
        continue;
      }

      // Special case, the use is in the same block as the MFMA. Insert the copy
      // just before the use.
      const TargetRegisterClass *DstRC = DAG.MRI.getRegClass(DstReg);
      const TargetRegisterClass *VGPRRC = SRI->getEquivalentVGPRClass(DstRC);
      Register NewUseReg = DAG.MRI.createVirtualRegister(VGPRRC);
      MachineInstr *UseInst = RU->getParent();
      MachineInstrBuilder VGPRCopy =
          BuildMI(*UseInst->getParent(), UseInst->getIterator(),
                  UseInst->getDebugLoc(), TII->get(TargetOpcode::COPY))
              .addDef(NewUseReg, {}, 0)
              .addUse(DstReg, {}, 0);
      DAG.LIS->InsertMachineInstrInMaps(*VGPRCopy);
      // Since we know this use has only one reaching def, we can replace the
      // use reg.
      RU->setReg(NewUseReg);
      // Track the copy source operand for r eplacement.
      DstRegSet.insert(&VGPRCopy->getOperand(1));
    }

    // Track the register for reclassification
    RewriteRegs.insert(DstReg);

    // Insert the dst operand for replacement. If this dst is in a chain of
    // tied-def MFMAs, and the first src2 needs to be replaced with a new reg,
    // all the correspond operands need to be replaced.
    DstRegSet.insert(Dst);
  }

  // Handle the copies for dst uses.
  using RUBType =
      std::pair<unsigned, DenseMap<Register, SmallPtrSet<MachineOperand *, 8>>>;
  for (RUBType RUBlockEntry : ReachingUseTracker) {
    using RUDType = std::pair<Register, SmallPtrSet<MachineOperand *, 8>>;
    for (RUDType RUDst : RUBlockEntry.second) {
      MachineOperand *OpBegin = *RUDst.second.begin();
      SlotIndex InstPt = DAG.LIS->getInstructionIndex(*OpBegin->getParent());

      // Find the earliest use in this block.
      for (MachineOperand *User : RUDst.second) {
        SlotIndex NewInstPt = DAG.LIS->getInstructionIndex(*User->getParent());
        if (SlotIndex::isEarlierInstr(NewInstPt, InstPt))
          InstPt = NewInstPt;
      }

      const TargetRegisterClass *DstRC = DAG.MRI.getRegClass(RUDst.first);
      const TargetRegisterClass *VGPRRC = SRI->getEquivalentVGPRClass(DstRC);
      Register NewUseReg = DAG.MRI.createVirtualRegister(VGPRRC);
      MachineInstr *UseInst = DAG.LIS->getInstructionFromIndex(InstPt);

      MachineInstrBuilder VGPRCopy =
          BuildMI(*UseInst->getParent(), UseInst->getIterator(),
                  UseInst->getDebugLoc(), TII->get(TargetOpcode::COPY))
              .addDef(NewUseReg, {}, 0)
              .addUse(RUDst.first, {}, 0);
      DAG.LIS->InsertMachineInstrInMaps(*VGPRCopy);

      // If this UseInst was the first MI in the region, update the region
      // boundaries.
      DenseMap<MachineInstr *, unsigned>::iterator FI =
          FirstMIToRegion.find(UseInst);
      if (FI != FirstMIToRegion.end()) {
        unsigned UpdateRegion = FI->second;
        DAG.Regions[UpdateRegion].first = VGPRCopy;
        FirstMIToRegion.erase(UseInst);
      }

      // Replace the operand for all users.
      for (MachineOperand *User : RUDst.second) {
        User->setReg(NewUseReg);
      }

      // Track the copy source operand for replacement.
      ReplaceMap[RUDst.first].insert(&VGPRCopy->getOperand(1));
    }
  }

  // We may have needed to insert copies after the reaching defs of the MFMAs.
  // Replace the original register with the result of the copy for all relevant
  // operands.
  for (std::pair<Register, Register> NewDef : RedefMap) {
    Register OldReg = NewDef.first;
    Register NewReg = NewDef.second;

    // Replace the register for any associated operand in the MFMA chain.
    for (MachineOperand *ReplaceOp : ReplaceMap[OldReg])
      ReplaceOp->setReg(NewReg);
  }

  // Finally, do the reclassification of the MFMA registers.
  for (Register RewriteReg : RewriteRegs) {
    Register RegToRewrite = RewriteReg;

    // Be sure to update the replacement register and not the original.
    DenseMap<Register, Register>::iterator RI = RedefMap.find(RewriteReg);
    if (RI != RedefMap.end())
      RegToRewrite = RI->second;

    const TargetRegisterClass *CurrRC = DAG.MRI.getRegClass(RegToRewrite);
    const TargetRegisterClass *AGPRRC = SRI->getEquivalentAGPRClass(CurrRC);

    DAG.MRI.setRegClass(RegToRewrite, AGPRRC);
  }

  // Bulk update the LIS.
  DAG.LIS->reanalyze(DAG.MF);
  // Liveins may have been modified for cross RC copies
  RegionPressureMap LiveInUpdater(&DAG, false);
  LiveInUpdater.buildLiveRegMap();

  for (unsigned Region = 0; Region < DAG.Regions.size(); Region++)
    DAG.LiveIns[Region] = LiveInUpdater.getLiveRegsForRegionIdx(Region);

  DAG.Pressure[RegionIdx] = DAG.getRealRegPressure(RegionIdx);

  return true;
}

unsigned PreRARematStage::getStageTargetOccupancy() const {
  return TargetOcc ? *TargetOcc : MFI.getMinWavesPerEU();
}

bool PreRARematStage::setObjective() {
  const Function &F = MF.getFunction();

  // Set up "spilling targets" for all regions.
  unsigned MaxSGPRs = ST.getMaxNumSGPRs(F);
  unsigned MaxVGPRs = ST.getMaxNumVGPRs(F);
  bool HasVectorRegisterExcess = false;
  for (unsigned I = 0, E = DAG.Regions.size(); I != E; ++I) {
    const GCNRegPressure &RP = DAG.Pressure[I];
    GCNRPTarget &Target = RPTargets.emplace_back(MaxSGPRs, MaxVGPRs, MF, RP);
    if (!Target.satisfied())
      TargetRegions.set(I);
    HasVectorRegisterExcess |= Target.hasVectorRegisterExcess();
  }

  if (HasVectorRegisterExcess || DAG.MinOccupancy >= MFI.getMaxWavesPerEU()) {
    // In addition to register usage being above addressable limits, occupancy
    // below the minimum is considered like "spilling" as well.
    TargetOcc = std::nullopt;
  } else {
    // There is no spilling and room to improve occupancy; set up "increased
    // occupancy targets" for all regions.
    TargetOcc = DAG.MinOccupancy + 1;
    const unsigned VGPRBlockSize = MFI.getDynamicVGPRBlockSize();
    MaxSGPRs = ST.getMaxNumSGPRs(*TargetOcc, false);
    MaxVGPRs = ST.getMaxNumVGPRs(*TargetOcc, VGPRBlockSize);
    for (auto [I, Target] : enumerate(RPTargets)) {
      Target.setTarget(MaxSGPRs, MaxVGPRs);
      if (!Target.satisfied())
        TargetRegions.set(I);
    }
  }

  return TargetRegions.any();
}

bool PreRARematStage::collectRematRegs(
    const DenseMap<MachineInstr *, unsigned> &MIRegion) {
  // We need up-to-date live-out info. to query live-out register masks in
  // regions containing rematerializable instructions.
  DAG.RegionLiveOuts.buildLiveRegMap();

  // Set of registers already marked for potential remterialization; used to
  // avoid rematerialization chains.
  SmallSet<Register, 4> MarkedRegs;
  auto IsMarkedForRemat = [&MarkedRegs](const MachineOperand &MO) -> bool {
    return MO.isReg() && MarkedRegs.contains(MO.getReg());
  };

  // Identify rematerializable instructions in the function.
  for (unsigned I = 0, E = DAG.Regions.size(); I != E; ++I) {
    RegionBoundaries Bounds = DAG.Regions[I];
    for (auto MI = Bounds.first; MI != Bounds.second; ++MI) {
      // The instruction must be rematerializable.
      MachineInstr &DefMI = *MI;
      if (!isReMaterializable(DefMI))
        continue;

      // We only support rematerializing virtual registers with one
      // definition.
      Register Reg = DefMI.getOperand(0).getReg();
      if (!Reg.isVirtual() || !DAG.MRI.hasOneDef(Reg))
        continue;

      // We only care to rematerialize the instruction if it has a single
      // non-debug user in a different region.
      // FIXME: Allow rematerializations with multiple uses. This should be
      // relatively easy to support using the current cost model.
      MachineInstr *UseMI = DAG.MRI.getOneNonDBGUser(Reg);
      if (!UseMI)
        continue;
      auto UseRegion = MIRegion.find(UseMI);
      if (UseRegion == MIRegion.end() || UseRegion->second == I)
        continue;

      // Do not rematerialize an instruction if it uses or is used by an
      // instruction that we have designated for rematerialization.
      // FIXME: Allow for rematerialization chains: this requires 1. updating
      // remat points to account for uses that are rematerialized, and 2.
      // either rematerializing the candidates in careful ordering, or
      // deferring the MBB RP walk until the entire chain has been
      // rematerialized.
      const MachineOperand &UseMO = UseMI->getOperand(0);
      if (IsMarkedForRemat(UseMO) ||
          llvm::any_of(DefMI.operands(), IsMarkedForRemat))
        continue;

      // Do not rematerialize an instruction it it uses registers that aren't
      // available at its use. This ensures that we are not extending any live
      // range while rematerializing.
      SlotIndex UseIdx = DAG.LIS->getInstructionIndex(*UseMI).getRegSlot(true);
      if (!VirtRegAuxInfo::allUsesAvailableAt(&DefMI, UseIdx, *DAG.LIS, DAG.MRI,
                                              *DAG.TII))
        continue;

      // Add the instruction to the rematerializable list.
      MarkedRegs.insert(Reg);
      RematRegs.emplace_back(&DefMI, UseMI, DAG, MIRegion);
    }
  }

  return !RematRegs.empty();
}

PreRARematStage::RematReg::RematReg(
    MachineInstr *DefMI, MachineInstr *UseMI, GCNScheduleDAGMILive &DAG,
    const DenseMap<MachineInstr *, unsigned> &MIRegion)
    : DefMI(DefMI), UseMI(UseMI), LiveIn(DAG.Regions.size()),
      LiveOut(DAG.Regions.size()), Live(DAG.Regions.size()),
      DefRegion(MIRegion.at(DefMI)), UseRegion(MIRegion.at(UseMI)) {

  // Mark regions in which the rematerializable register is live.
  Register Reg = getReg();
  for (unsigned I = 0, E = DAG.Regions.size(); I != E; ++I) {
    auto LiveInIt = DAG.LiveIns[I].find(Reg);
    if (LiveInIt != DAG.LiveIns[I].end())
      LiveIn.set(I);
    const auto &LiveOuts = DAG.RegionLiveOuts.getLiveRegsForRegionIdx(I);
    if (auto LiveOutIt = LiveOuts.find(Reg); LiveOutIt != LiveOuts.end())
      LiveOut.set(I);
  }
  Live |= LiveIn;
  Live |= LiveOut;
  Mask = DAG.RegionLiveOuts.getLiveRegsForRegionIdx(DefRegion).at(Reg);
}

bool PreRARematStage::RematReg::maybeBeneficial(
    const BitVector &TargetRegions, ArrayRef<GCNRPTarget> RPTargets) const {
  Register Reg = getReg();
  for (unsigned I : TargetRegions.set_bits()) {
    if (Live[I] && RPTargets[I].isSaveBeneficial(Reg))
      return true;
  }
  return false;
}

void PreRARematStage::RematReg::insertMI(unsigned RegionIdx,
                                         MachineInstr *RematMI,
                                         GCNScheduleDAGMILive &DAG) const {
  RegionBoundaries &Bounds = DAG.Regions[RegionIdx];
  if (Bounds.first == std::next(MachineBasicBlock::iterator(RematMI)))
    Bounds.first = RematMI;
  DAG.LIS->InsertMachineInstrInMaps(*RematMI);
  DAG.LIS->createAndComputeVirtRegInterval(RematMI->getOperand(0).getReg());
}

PreRARematStage::ScoredRemat::FreqInfo::FreqInfo(
    MachineFunction &MF, const GCNScheduleDAGMILive &DAG) {
  assert(DAG.MLI && "MLI not defined in DAG");
  MachineBranchProbabilityInfo MBPI;
  MachineBlockFrequencyInfo MBFI(MF, MBPI, *DAG.MLI);

  const unsigned NumRegions = DAG.Regions.size();
  MinFreq = MBFI.getEntryFreq().getFrequency();
  MaxFreq = 0;
  Regions.reserve(NumRegions);
  for (unsigned I = 0; I < NumRegions; ++I) {
    MachineBasicBlock *MBB = DAG.Regions[I].first->getParent();
    uint64_t BlockFreq = MBFI.getBlockFreq(MBB).getFrequency();
    Regions.push_back(BlockFreq);
    if (BlockFreq && BlockFreq < MinFreq)
      MinFreq = BlockFreq;
    else if (BlockFreq > MaxFreq)
      MaxFreq = BlockFreq;
  }
  if (!MinFreq)
    return;

  // Scale everything down if frequencies are high.
  if (MinFreq >= ScaleFactor * ScaleFactor) {
    for (uint64_t &Freq : Regions)
      Freq /= ScaleFactor;
    MinFreq /= ScaleFactor;
    MaxFreq /= ScaleFactor;
  }
}

PreRARematStage::ScoredRemat::ScoredRemat(RematReg *Remat, const FreqInfo &Freq,
                                          const GCNScheduleDAGMILive &DAG)
    : Remat(Remat), NumRegs(getNumRegs(DAG)), FreqDiff(getFreqDiff(Freq)) {}

unsigned PreRARematStage::ScoredRemat::getNumRegs(
    const GCNScheduleDAGMILive &DAG) const {
  const TargetRegisterClass &RC = *DAG.MRI.getRegClass(Remat->getReg());
  unsigned RegSize = DAG.TRI->getRegSizeInBits(RC);
  if (unsigned SubIdx = Remat->DefMI->getOperand(0).getSubReg()) {
    // The following may return -1 (i.e., a large unsigned number) on indices
    // that may be used to access subregisters of multiple sizes; in such cases
    // fallback on the size derived from the register class.
    unsigned SubRegSize = DAG.TRI->getSubRegIdxSize(SubIdx);
    if (SubRegSize < RegSize)
      RegSize = SubRegSize;
  }
  return divideCeil(RegSize, 32);
}

int64_t PreRARematStage::ScoredRemat::getFreqDiff(const FreqInfo &Freq) const {
  // Get frequencies of defining and using regions. A rematerialization from the
  // least frequent region to the most frequent region will yield the greatest
  // latency penalty and therefore should get minimum score. Reciprocally, a
  // rematerialization in the other direction should get maximum score. Default
  // to values that will yield the worst possible score given known frequencies
  // in order to penalize rematerializations from or into regions whose
  // frequency is unknown.
  int64_t DefOrMin = std::max(Freq.Regions[Remat->DefRegion], Freq.MinFreq);
  int64_t UseOrMax = Freq.Regions[Remat->UseRegion];
  if (!UseOrMax)
    UseOrMax = Freq.MaxFreq;
  return DefOrMin - UseOrMax;
}

void PreRARematStage::ScoredRemat::update(const BitVector &TargetRegions,
                                          ArrayRef<GCNRPTarget> RPTargets,
                                          const FreqInfo &FreqInfo,
                                          bool ReduceSpill) {
  MaxFreq = 0;
  RegionImpact = 0;
  for (unsigned I : TargetRegions.set_bits()) {
    if (!Remat->Live[I] || !RPTargets[I].isSaveBeneficial(Remat->getReg()))
      continue;
    bool UnusedLT = Remat->isUnusedLiveThrough(I);

    // Regions in which RP is guaranteed to decrease have more weight.
    RegionImpact += UnusedLT ? 2 : 1;

    if (ReduceSpill) {
      uint64_t Freq = FreqInfo.Regions[I];
      if (!UnusedLT) {
        // Apply a frequency penalty in regions in which we are not sure that RP
        // will decrease.
        Freq /= 2;
      }
      MaxFreq = std::max(MaxFreq, Freq);
    }
  }
  RegionImpact *= NumRegs;
}

void PreRARematStage::rematerialize(const RematReg &Remat,
                                    BitVector &RecomputeRP,
                                    RollbackInfo *Rollback) {
  const SIInstrInfo *TII = MF.getSubtarget<GCNSubtarget>().getInstrInfo();
  MachineInstr &DefMI = *Remat.DefMI;
  Register Reg = DefMI.getOperand(0).getReg();
  Register NewReg = DAG.MRI.cloneVirtualRegister(Reg);

  // Rematerialize the register in the region where it is used.
  MachineBasicBlock::iterator InsertPos = Remat.UseMI;
  TII->reMaterialize(*InsertPos->getParent(), InsertPos, NewReg, 0, DefMI);
  MachineInstr *RematMI = &*std::prev(InsertPos);
  Remat.UseMI->substituteRegister(Reg, NewReg, 0, *DAG.TRI);
  Remat.insertMI(Remat.UseRegion, RematMI, DAG);
  if (Rollback) {
    Rollback->RematMI = RematMI;
    // Make the original MI a debug value so that it does not influence
    // scheduling and replace all read registers with a sentinel register to
    // prevent operands to appear in use-lists of other MIs during LIS
    // updates. Store mappings between operand indices and original registers
    // for potential rollback.
    DefMI.setDesc(TII->get(TargetOpcode::DBG_VALUE));
    for (auto [Idx, MO] : enumerate(Remat.DefMI->operands())) {
      if (MO.isReg() && MO.readsReg()) {
        Rollback->RegMap.insert({Idx, MO.getReg()});
        MO.setReg(Register());
      }
    }
  } else {
    // Just delete the original instruction if it cannot be rolled back.
    DAG.deleteMI(Remat.DefRegion, &DefMI);
  }

#ifdef EXPENSIVE_CHECKS
  // All uses are known to be available / live at the remat point. Thus,
  // the uses should already be live in to the using region.
  for (MachineOperand &MO : DefMI.operands()) {
    if (!MO.isReg() || !MO.getReg() || !MO.readsReg())
      continue;

    Register UseReg = MO.getReg();
    if (!UseReg.isVirtual())
      continue;

    LiveInterval &LI = DAG.LIS->getInterval(UseReg);
    LaneBitmask LM = DAG.MRI.getMaxLaneMaskForVReg(MO.getReg());
    if (LI.hasSubRanges() && MO.getSubReg())
      LM = DAG.TRI->getSubRegIndexLaneMask(MO.getSubReg());

    LaneBitmask LiveInMask = DAG.LiveIns[Remat.UseRegion].at(UseReg);
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

  // Remove the register from all regions where it is a live-in or live-out
  // and adjust RP targets. The save is guaranteed in regions in which the
  // register is live-through and unused but optimistic in all other regions
  // where the register is live.
  for (unsigned I : Remat.Live.set_bits()) {
    RPTargets[I].saveReg(Reg, Remat.Mask, DAG.MRI);
    DAG.LiveIns[I].erase(Reg);
    DAG.RegionLiveOuts.getLiveRegsForRegionIdx(I).erase(Reg);
    if (!Remat.isUnusedLiveThrough(I))
      RecomputeRP.set(I);
  }

  RescheduleRegions |= Remat.Live;
}

void PreRARematStage::commitRematerializations() const {
  REMAT_DEBUG(dbgs() << "Commiting all rematerializations\n");
  for (const RollbackInfo &Rollback : Rollbacks)
    DAG.deleteMI(Rollback.Remat->DefRegion, Rollback.Remat->DefMI);
}

void PreRARematStage::unsetSatisifedRPTargets(const BitVector &Regions) {
  for (unsigned I : Regions.set_bits()) {
    if (TargetRegions[I] && RPTargets[I].satisfied()) {
      REMAT_DEBUG(dbgs() << "  [" << I << "] Target reached!\n");
      TargetRegions.reset(I);
    }
  }
}

bool PreRARematStage::updateAndVerifyRPTargets(const BitVector &Regions) {
  bool TooOptimistic = false;
  for (unsigned I : Regions.set_bits()) {
    GCNRPTarget &Target = RPTargets[I];
    Target.setRP(DAG.getRealRegPressure(I));

    // Since we were optimistic in assessing RP decreases in these regions, we
    // may need to remark the target as a target region if RP didn't decrease
    // as expected.
    if (!TargetRegions[I] && !Target.satisfied()) {
      REMAT_DEBUG(dbgs() << "  [" << I << "] Incorrect RP estimation\n");
      TooOptimistic = true;
      TargetRegions.set(I);
    }
  }
  return TooOptimistic;
}

// Copied from MachineLICM
bool PreRARematStage::isReMaterializable(const MachineInstr &MI) {
  if (!DAG.TII->isReMaterializable(MI))
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
  // rollback rematerializations or revert scheduling in such cases.
  if (!TargetOcc)
    return;

  // When increasing occupancy, it is possible that re-scheduling is not able to
  // achieve the target occupancy in all regions, in which case re-scheduling in
  // all regions should be reverted.
  if (DAG.MinOccupancy >= *TargetOcc) {
    commitRematerializations();
    return;
  }

  // It is possible that re-scheduling lowers occupancy over the one achieved
  // just through rematerializations, in which case we revert re-scheduling in
  // all regions but do not roll back rematerializations.
  const bool ShouldRollbackRemats = AchievedOcc < *TargetOcc;

  // When we both need to revert re-scheduling and rollback rematerializations,
  // restore rematerialized MIs' original state before reverting so that they
  // are treated as non-debug instructions by the revert logic.
  if (ShouldRollbackRemats) {
    for (const RollbackInfo &Rollback : Rollbacks) {
      const auto &[Remat, RematMI, RegMap] = Rollback;
      Remat->DefMI->setDesc(DAG.TII->get(RematMI->getOpcode()));
      for (const auto &[MOIdx, Reg] : RegMap)
        Remat->DefMI->getOperand(MOIdx).setReg(Reg);
    }
  }

  // Revert re-scheduling in all affected regions.
  for (const auto &[RegionIdx, OrigMIOrder, MaxPressure] : RegionReverts) {
    REMAT_DEBUG(dbgs() << "Reverting re-scheduling in region " << RegionIdx
                       << '\n');
    DAG.Pressure[RegionIdx] = MaxPressure;
    modifyRegionSchedule(RegionIdx, RegionBB[RegionIdx], OrigMIOrder);
  }

  if (!ShouldRollbackRemats) {
    commitRematerializations();
    DAG.setTargetOccupancy(AchievedOcc);
    return;
  }

  // Reset the target occupancy to what it was pre-rematerialization.
  DAG.setTargetOccupancy(*TargetOcc - 1);

  // Finish rolling back rematerializations, then recompute pressure in all
  // affected regions.
  REMAT_DEBUG(dbgs() << "==== ROLLBACK ====\n");
  BitVector RecomputeRP(DAG.Regions.size());
  DenseSet<Register> RecomputeLI;
  for (const RollbackInfo &Rollback : Rollbacks) {
    const auto &[Remat, RematMI, RegMap] = Rollback;

    // Switch back to using the original register and delete the
    // rematerialization.
    Register Reg = RematMI->getOperand(0).getReg();
    Register OriginalReg = Remat->DefMI->getOperand(0).getReg();
    Remat->UseMI->substituteRegister(Reg, OriginalReg, 0, *DAG.TRI);
    REMAT_DEBUG(dbgs() << '[' << Remat->UseRegion
                       << "] Deleting rematerialization " << *RematMI);
    DAG.deleteMI(Remat->UseRegion, RematMI);

    // Re-add the defined register as a live-in/live-out in all regions it used
    // to be one in.
    std::pair<Register, LaneBitmask> LiveReg(OriginalReg, Remat->Mask);
    for (unsigned I : Remat->LiveIn.set_bits())
      DAG.LiveIns[I].insert(LiveReg);
    for (unsigned I : Remat->LiveOut.set_bits())
      DAG.RegionLiveOuts.getLiveRegsForRegionIdx(I).insert(LiveReg);

    RecomputeRP |= Rollback.Remat->Live;
    // Regenerate intervals for all register operands of rematerialized MIs as
    // slot indices may have changed slightly from before re-scheduling.
    for (MachineOperand &MO : Rollback.Remat->DefMI->operands()) {
      if (MO.isReg() && MO.getReg().isVirtual())
        RecomputeLI.insert(MO.getReg());
    }
  }
  for (Register Reg : RecomputeLI) {
    DAG.LIS->removeInterval(Reg);
    DAG.LIS->createAndComputeVirtRegInterval(Reg);
  }
#ifdef EXPENSIVE_CHECKS
  // In particular, we want to check for coherent MI/slot order in regions in
  // which reverts and/or rollbacks may have happened.
  MF.verify();
#endif
  for (unsigned I : RecomputeRP.set_bits())
    DAG.Pressure[I] = DAG.getRealRegPressure(I);

  GCNSchedStage::finalizeGCNSchedStage();
}

void GCNScheduleDAGMILive::deleteMI(unsigned RegionIdx, MachineInstr *MI) {
  // It's not possible for the deleted instruction to be upper region boundary
  // since we don't delete region terminators.
  if (Regions[RegionIdx].first == MI)
    Regions[RegionIdx].first = std::next(MachineBasicBlock::iterator(MI));
  LIS->removeInterval(MI->getOperand(0).getReg());
  LIS->RemoveMachineInstrFromMaps(*MI);
  MI->eraseFromParent();
}

void GCNScheduleDAGMILive::setTargetOccupancy(unsigned TargetOccupancy) {
  MinOccupancy = TargetOccupancy;
  if (MFI.getOccupancy() < TargetOccupancy)
    MFI.increaseOccupancy(MF, MinOccupancy);
  else
    MFI.limitOccupancy(MinOccupancy);
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
