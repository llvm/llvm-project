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
#include "GCNHazardRecognizer.h"
#include "GCNRegPressure.h"
#include "GCNRematStrategy.h"
#include "GCNRewriteMFMAForm.h"
#include "SIMachineFunctionInfo.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/CalcSpillWeights.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineBlockFrequencyInfo.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/RegisterPressure.h"
#include "llvm/MC/MCSchedule.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/ErrorHandling.h"
#include <limits>

#define DEBUG_TYPE "machine-scheduler"

using namespace llvm;

bool VGPRThresholdParser::parse(cl::Option &O, StringRef ArgName, StringRef Arg,
                                unsigned &Value) {
  if (Arg.getAsInteger(0, Value))
    return O.error("'" + Arg + "' value invalid for uint argument!");

  if (Value > 100)
    return O.error("'" + Arg + "' value must be in the range [0, 100]!");

  return false;
}

cl::opt<unsigned, false, VGPRThresholdParser> llvm::VGPRThresholdPercentOpt(
    "amdgpu-vgpr-threshold-percent", cl::Hidden,
    cl::desc("Percent of VGPR limits that we should use as RP threshold "
             "during scheduling. We have two limits relevant to scheduling: "
             "Critical (avoid decreasing occupancy), Excess (avoid spilling). "
             "This flag scales both limits back by an equal percent: (0 = use "
             " default calculation, 1-100 = use percentage), default: 0"),
    cl::init(0));

const unsigned ScheduleMetrics::ScaleFactor = 100;

GCNSchedStrategy::GCNSchedStrategy(const MachineSchedContext *C)
    : GenericScheduler(C),
      CLOpts(C->MF->getSubtarget<GCNSubtarget>().getCLOpts()),
      TargetOccupancy(0), MF(nullptr), DownwardTracker(*C->LIS),
      UpwardTracker(*C->LIS), HasHighPressure(false) {
  if (CLOpts.use_amdgpu_trackers != BoolOrDefault::Default)
    GCNTrackersOverride = CLOpts.use_amdgpu_trackers == BoolOrDefault::True;
  VGPRThresholdPercent = VGPRThresholdPercentOpt;
}

void GCNSchedStrategy::initialize(ScheduleDAGMI *DAG) {
  GenericScheduler::initialize(DAG);

  MF = &DAG->MF;

  const GCNSubtarget &ST = MF->getSubtarget<GCNSubtarget>();

  SGPRExcessLimit =
      Context->RegClassInfo->getNumAllocatableRegs(&AMDGPU::SGPR_32RegClass);
  VGPRExcessLimit =
      Context->RegClassInfo->getNumAllocatableRegs(&AMDGPU::VGPR_32RegClass);
  AGPRExcessLimit =
      Context->RegClassInfo->getNumAllocatableRegs(&AMDGPU::AGPR_32RegClass);

  ArchVGPRBankLimit = VGPRExcessLimit;
  AGPRBankLimit = AGPRExcessLimit;
  UnifiedVGPRExcessLimit = ST.hasGFX90AInsts() ? ST.getMaxNumVGPRs(*MF) : 0;

  SIMachineFunctionInfo &MFI = *MF->getInfo<SIMachineFunctionInfo>();
  // Set the initial TargetOccupnacy to the maximum occupancy that we can
  // achieve for this function. This effectively sets a lower bound on the
  // 'Critical' register limits in the scheduler.
  // Allow for lower occupancy targets if kernel is wave limited or memory
  // bound, and using the relaxed occupancy feature.
  TargetOccupancy = CLOpts.schedule_relaxed_occupancy
                        ? MFI.getMinAllowedOccupancy()
                        : MFI.getOccupancy();
  SGPRCriticalLimit =
      std::min(ST.getMaxNumSGPRs(TargetOccupancy, true), SGPRExcessLimit);

  if (!KnownExcessRP) {
    // Preserve the combined occupancy budget before clamping to one bank.
    UnifiedVGPRCriticalLimit =
        ST.getMaxNumVGPRs(TargetOccupancy, MFI.getDynamicVGPRBlockSize());
    VGPRCriticalLimit = std::min(UnifiedVGPRCriticalLimit, VGPRExcessLimit);
  } else {
    // This is similar to ST.getMaxNumVGPRs(TargetOccupancy) result except
    // returns a reasonably small number for targets with lots of VGPRs, such
    // as GFX10 and GFX11.
    LLVM_DEBUG(dbgs() << "Region is known to spill, use alternative "
                         "VGPRCriticalLimit calculation method.\n");
    unsigned DynamicVGPRBlockSize = MFI.getDynamicVGPRBlockSize();
    unsigned Granule =
        AMDGPU::IsaInfo::getVGPRAllocGranule(ST, DynamicVGPRBlockSize);
    unsigned Addressable =
        AMDGPU::IsaInfo::getAddressableNumVGPRs(ST, DynamicVGPRBlockSize);
    unsigned VGPRBudget = alignDown(Addressable / TargetOccupancy, Granule);
    VGPRBudget = std::max(VGPRBudget, Granule);
    UnifiedVGPRCriticalLimit = VGPRBudget;
    VGPRCriticalLimit = std::min(VGPRBudget, VGPRExcessLimit);
  }

  // Reuse VGPR critical limit
  AGPRCriticalLimit = std::min(VGPRCriticalLimit, AGPRExcessLimit);

  // Apply VGPR excess threshold percentage if specified.
  if (VGPRThresholdPercent > 0) {
    [[maybe_unused]] unsigned OriginalVGPRExcessLimit = VGPRExcessLimit;
    [[maybe_unused]] unsigned OriginalVGPRCriticalLimit = VGPRCriticalLimit;
    VGPRExcessLimit = (VGPRThresholdPercent * VGPRExcessLimit + 99) / 100;
    VGPRCriticalLimit = (VGPRThresholdPercent * VGPRCriticalLimit + 99) / 100;
    UnifiedVGPRCriticalLimit =
        (VGPRThresholdPercent * UnifiedVGPRCriticalLimit + 99) / 100;
    LLVM_DEBUG(dbgs() << "Applied VGPR excess threshold "
                      << VGPRThresholdPercent << "%, VGPRExcessLimit: "
                      << OriginalVGPRExcessLimit << " -> " << VGPRExcessLimit
                      << ". VGPRCriticalLimit: " << OriginalVGPRCriticalLimit
                      << " -> " << VGPRCriticalLimit << '\n');
  } else {
    VGPRExcessLimit -= std::min(VGPRLimitBias + ErrorMargin, VGPRExcessLimit);
    VGPRCriticalLimit -=
        std::min(VGPRLimitBias + ErrorMargin, VGPRCriticalLimit);
    UnifiedVGPRCriticalLimit -=
        std::min(VGPRLimitBias + ErrorMargin, UnifiedVGPRCriticalLimit);
  }

  // Subtract error margin and bias from register limits and avoid overflow.
  SGPRCriticalLimit -= std::min(SGPRLimitBias + ErrorMargin, SGPRCriticalLimit);
  SGPRExcessLimit -= std::min(SGPRLimitBias + ErrorMargin, SGPRExcessLimit);

  AGPRExcessLimit -= std::min(VGPRLimitBias + ErrorMargin, AGPRExcessLimit);
  AGPRCriticalLimit -= std::min(VGPRLimitBias + ErrorMargin, AGPRCriticalLimit);

  LLVM_DEBUG(dbgs() << "VGPRCriticalLimit = " << VGPRCriticalLimit
                    << ", UnifiedVGPRCriticalLimit = "
                    << UnifiedVGPRCriticalLimit
                    << ", VGPRExcessLimit = " << VGPRExcessLimit
                    << ", AGPRCriticalLimit = " << AGPRCriticalLimit
                    << ", AGPRExcessLimit = " << AGPRExcessLimit
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
  // imprecise in both cases. For a bundle, check the instructions inside it:
  // the BUNDLE header only has implicit operands.
  for (const auto &Op : const_mi_bundle_ops(*SU.getInstr())) {
    if (!Op.isReg() || Op.isImplicit())
      continue;
    if (Op.getReg().isPhysical() ||
        (Op.isDef() && Op.getSubReg() != AMDGPU::NoSubRegister))
      return false;
  }
  return true;
}

std::pair<unsigned, unsigned>
GCNSchedStrategy::getBankedPressure(const GCNRegPressure &RP) const {
  if (!CLOpts.avgpr_banking)
    return {RP.getArchVGPRNum(), RP.getAGPRNum()};
  return RP.getBankedVGPRNum(ArchVGPRBankLimit, AGPRBankLimit);
}

void GCNSchedStrategy::getRegisterPressures(
    bool AtTop, const RegPressureTracker &RPTracker, SUnit *SU,
    std::vector<unsigned> &Pressure, std::vector<unsigned> &MaxPressure,
    const GCNDownwardRPTrackerBase &DownwardTracker,
    const GCNUpwardRPTracker &UpwardTracker, ScheduleDAGMI *DAG,
    const SIRegisterInfo *SRI) {
  // getDownwardPressure() and getUpwardPressure() make temporary changes to
  // the tracker, so we need to pass those function a non-const copy.
  RegPressureTracker &TempTracker = const_cast<RegPressureTracker &>(RPTracker);
  if (!useGCNTrackers()) {
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
    NewPressure = DownwardTracker.bumpDownwardPressure(MI, SRI);
  } else {
    GCNUpwardRPTracker TempUpwardTracker(UpwardTracker);
    TempUpwardTracker.recede(*MI);
    NewPressure = TempUpwardTracker.getPressure();
  }
  auto [ArchNum, AccNum] = getBankedPressure(NewPressure);
  Pressure[AMDGPU::RegisterPressureSets::SReg_32] = NewPressure.getSGPRNum();
  Pressure[AMDGPU::RegisterPressureSets::VGPR_32] = ArchNum;
  Pressure[AMDGPU::RegisterPressureSets::AGPR_32] = AccNum;
  CandUnifiedVGPRPressure =
      GCNRegPressure::getBankedUnifiedVGPRNum(ArchNum, AccNum);
}

void GCNSchedStrategy::initCandidate(SchedCandidate &Cand, SUnit *SU,
                                     bool AtTop,
                                     const RegPressureTracker &RPTracker,
                                     const SIRegisterInfo *SRI,
                                     unsigned SGPRPressure,
                                     unsigned VGPRPressure,
                                     unsigned AGPRPressure, bool IsBottomUp) {
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
  if (AtTop || !canUsePressureDiffs(*SU) || useGCNTrackers()) {
    getRegisterPressures(AtTop, RPTracker, SU, Pressure, MaxPressure,
                         DownwardTracker, UpwardTracker, DAG, SRI);
  } else {
    // Reserve 4 slots.
    Pressure.resize(4, 0);
    Pressure[AMDGPU::RegisterPressureSets::SReg_32] = SGPRPressure;
    Pressure[AMDGPU::RegisterPressureSets::VGPR_32] = VGPRPressure;
    Pressure[AMDGPU::RegisterPressureSets::AGPR_32] = AGPRPressure;

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
            CheckPressure[AMDGPU::RegisterPressureSets::VGPR_32] ||
        Pressure[AMDGPU::RegisterPressureSets::AGPR_32] !=
            CheckPressure[AMDGPU::RegisterPressureSets::AGPR_32]) {
      errs() << "Register Pressure is inaccurate when calculated through "
                "PressureDiff\n"
             << "SGPR got " << Pressure[AMDGPU::RegisterPressureSets::SReg_32]
             << ", expected "
             << CheckPressure[AMDGPU::RegisterPressureSets::SReg_32] << "\n"
             << "VGPR got " << Pressure[AMDGPU::RegisterPressureSets::VGPR_32]
             << ", expected "
             << CheckPressure[AMDGPU::RegisterPressureSets::VGPR_32] << "\n"
             << "AGPR got " << Pressure[AMDGPU::RegisterPressureSets::AGPR_32]
             << ", expected "
             << CheckPressure[AMDGPU::RegisterPressureSets::AGPR_32] << "\n";
      report_fatal_error("inaccurate register pressure calculation");
    }
#endif
  }

  unsigned NewAGPRPressure = Pressure[AMDGPU::RegisterPressureSets::AGPR_32];
  unsigned NewSGPRPressure = Pressure[AMDGPU::RegisterPressureSets::SReg_32];
  unsigned NewVGPRPressure = Pressure[AMDGPU::RegisterPressureSets::VGPR_32];

  // If two instructions increase the pressure of different register sets
  // by the same amount, the generic scheduler will prefer to schedule the
  // instruction that increases the set with the least amount of registers,
  // which in our case would be SGPRs.  This is rarely what we want, so
  // when we report excess/critical register pressure, we do it either
  // only for VGPRs, AGPRs or SGPRs. Priority: VGPR > AGPR > SGPR.

  // FIXME: Better heuristics to determine whether to prefer SGPRs or VGPRs.
  const unsigned MaxVGPRPressureInc = 16;
  bool ShouldTrackVGPRs = VGPRPressure + MaxVGPRPressureInc >= VGPRExcessLimit;
  bool ShouldTrackAGPRs = AGPRExcessLimit > 0 && !ShouldTrackVGPRs &&
                          AGPRPressure + MaxVGPRPressureInc >= AGPRExcessLimit;
  bool ShouldTrackSGPRs =
      !ShouldTrackVGPRs && !ShouldTrackAGPRs && SGPRPressure >= SGPRExcessLimit;
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

  if (ShouldTrackAGPRs && NewAGPRPressure >= AGPRExcessLimit) {
    HasHighPressure = true;
    Cand.RPDelta.Excess = PressureChange(AMDGPU::RegisterPressureSets::AGPR_32);
    Cand.RPDelta.Excess.setUnitInc(NewAGPRPressure - AGPRExcessLimit);
  }

  if (ShouldTrackSGPRs && NewSGPRPressure >= SGPRExcessLimit) {
    HasHighPressure = true;
    Cand.RPDelta.Excess = PressureChange(AMDGPU::RegisterPressureSets::SReg_32);
    Cand.RPDelta.Excess.setUnitInc(NewSGPRPressure - SGPRExcessLimit);
  }

  // Because of alignment padding, the combined file can overflow when both
  // banks are within limits. This check catches overflow that the others above
  // miss.
  if (CLOpts.avgpr_banking && UnifiedVGPRExcessLimit &&
      CandUnifiedVGPRPressure >= UnifiedVGPRExcessLimit) {
    HasHighPressure = true;
    int UnifiedExcess = CandUnifiedVGPRPressure - UnifiedVGPRExcessLimit;
    PressureChange &Excess = Cand.RPDelta.Excess;
    bool HasBankExcess =
        Excess.isValid() &&
        Excess.getPSet() != AMDGPU::RegisterPressureSets::SReg_32;
    if (!HasBankExcess || UnifiedExcess > Excess.getUnitInc()) {
      Excess = PressureChange(AMDGPU::RegisterPressureSets::VGPR_32);
      Excess.setUnitInc(UnifiedExcess);
    }
  }

  // Register pressure is considered 'CRITICAL' if it is approaching a value
  // that would reduce the wave occupancy for the execution unit.  When
  // register pressure is 'CRITICAL', increasing SGPR, VGPR, and AGPR
  // pressure all has the same cost, so we pick the most critical type.

  // Unified targets allocate both vector banks from the same occupancy budget.
  // Keep the bank pressures above for excess checks, and do not align VGPRs
  // when there are no AGPRs, matching GCNRegPressure::getVGPRNum().
  bool UnifiedVGPRFile = MF->getSubtarget<GCNSubtarget>().hasGFX90AInsts();
  unsigned VectorCriticalLimit = VGPRCriticalLimit;
  if (UnifiedVGPRFile) {
    if (NewAGPRPressure)
      NewVGPRPressure = GCNRegPressure::getUnifiedVGPRNum(NewVGPRPressure,
                                                          NewAGPRPressure, 0);
    VectorCriticalLimit = UnifiedVGPRCriticalLimit;
  }

  int SGPRDelta = NewSGPRPressure - SGPRCriticalLimit;
  int VGPRDelta = NewVGPRPressure - VectorCriticalLimit;
  int AGPRDelta = !UnifiedVGPRFile && AGPRExcessLimit > 0
                      ? NewAGPRPressure - AGPRCriticalLimit
                      : std::numeric_limits<int>::min();

  if (SGPRDelta >= 0 || VGPRDelta >= 0 || AGPRDelta >= 0) {
    HasHighPressure = true;
    // Pick the most critical type.
    if (VGPRDelta >= SGPRDelta && VGPRDelta >= AGPRDelta) {
      Cand.RPDelta.CriticalMax =
          PressureChange(AMDGPU::RegisterPressureSets::VGPR_32);
      Cand.RPDelta.CriticalMax.setUnitInc(VGPRDelta);
    } else if (AGPRDelta >= SGPRDelta) {
      Cand.RPDelta.CriticalMax =
          PressureChange(AMDGPU::RegisterPressureSets::AGPR_32);
      Cand.RPDelta.CriticalMax.setUnitInc(AGPRDelta);
    } else {
      Cand.RPDelta.CriticalMax =
          PressureChange(AMDGPU::RegisterPressureSets::SReg_32);
      Cand.RPDelta.CriticalMax.setUnitInc(SGPRDelta);
    }
  }
}

static bool shouldCheckPending(unsigned PendingQueueLimit, SchedBoundary &Zone,
                               const TargetSchedModel *SchedModel) {
  bool HasBufferedModel =
      SchedModel->hasInstrSchedModel() && SchedModel->getMicroOpBufferSize();
  unsigned Combined = Zone.Available.size() + Zone.Pending.size();
  return Combined <= PendingQueueLimit && HasBufferedModel;
}

static SUnit *pickOnlyChoice(unsigned PendingQueueLimit, SchedBoundary &Zone,
                             const TargetSchedModel *SchedModel) {
  // pickOnlyChoice() releases pending instructions and checks for new hazards.
  SUnit *OnlyChoice = Zone.pickOnlyChoice();
  if (!shouldCheckPending(PendingQueueLimit, Zone, SchedModel) ||
      Zone.Pending.empty())
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
  unsigned AGPRPressure = 0;
  IsPending = false;
  if (DAG->isTrackingPressure()) {
    if (!useGCNTrackers()) {
      SGPRPressure = Pressure[AMDGPU::RegisterPressureSets::SReg_32];
      VGPRPressure = Pressure[AMDGPU::RegisterPressureSets::VGPR_32];
      AGPRPressure = Pressure[AMDGPU::RegisterPressureSets::AGPR_32];
    } else {
      GCNRPTracker *T = IsBottomUp
                            ? static_cast<GCNRPTracker *>(&UpwardTracker)
                            : static_cast<GCNRPTracker *>(&DownwardTracker);
      SGPRPressure = T->getPressure().getSGPRNum();
      std::tie(VGPRPressure, AGPRPressure) =
          getBankedPressure(T->getPressure());
    }
  }
  LLVM_DEBUG(dbgs() << "Available Q:\n");
  ReadyQueue &AQ = Zone.Available;
  for (SUnit *SU : AQ) {

    SchedCandidate TryCand(ZonePolicy);
    initCandidate(TryCand, SU, Zone.isTop(), RPTracker, SRI, SGPRPressure,
                  VGPRPressure, AGPRPressure, IsBottomUp);
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

  if (!shouldCheckPending(CLOpts.scheduler_pending_queue_limit, Zone,
                          SchedModel))
    return;

  LLVM_DEBUG(dbgs() << "Pending Q:\n");
  ReadyQueue &PQ = Zone.Pending;
  for (SUnit *SU : PQ) {

    SchedCandidate TryCand(ZonePolicy);
    initCandidate(TryCand, SU, Zone.isTop(), RPTracker, SRI, SGPRPressure,
                  VGPRPressure, AGPRPressure, IsBottomUp);
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
  if (SUnit *SU = pickOnlyChoice(CLOpts.scheduler_pending_queue_limit, Bot,
                                 SchedModel)) {
    IsTopNode = false;
    return SU;
  }
  if (SUnit *SU = pickOnlyChoice(CLOpts.scheduler_pending_queue_limit, Top,
                                 SchedModel)) {
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
    if (shouldVerifyScheduling(DAG->MF)) {
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
    if (shouldVerifyScheduling(DAG->MF)) {
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
      SU =
          pickOnlyChoice(CLOpts.scheduler_pending_queue_limit, Top, SchedModel);
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
      SU =
          pickOnlyChoice(CLOpts.scheduler_pending_queue_limit, Bot, SchedModel);
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

  LLVM_DEBUG(dbgs() << "Scheduling " << *SU << " " << *SU->getInstr());
  return SU;
}

void GCNSchedStrategy::schedNode(SUnit *SU, bool IsTopNode) {
  if (useGCNTrackers()) {
    MachineInstr *MI = SU->getInstr();
    IsTopNode ? DownwardTracker.advance(*MI) : UpwardTracker.recede(*MI);
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
  if (CLOpts.rewrite_mfma_form_sched_stage)
    SchedStages.push_back(GCNSchedStageID::RewriteMFMAForm);
  SchedStages.push_back(GCNSchedStageID::UnclusteredHighRPReschedule);
  SchedStages.push_back(GCNSchedStageID::ClusteredLowOccupancyReschedule);
  SchedStages.push_back(GCNSchedStageID::PreRARematerialize);
  if (IsLegacyScheduler)
    GCNTrackersOverride = std::nullopt;
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
  if (ST.getCLOpts().schedule_relaxed_occupancy) {
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
  case GCNSchedStageID::LiveIntervalRPReschedule:
    return std::make_unique<LiveIntervalRPStage>(SchedStageID, *this);
  }

  llvm_unreachable("Unknown SchedStageID.");
}

void GCNScheduleDAGMILive::schedule() {
  // Collect all scheduling regions. The actual scheduling is performed in
  // GCNScheduleDAGMILive::finalizeSchedule.
  Regions.push_back(std::pair(RegionBegin, RegionEnd));
}

GCNRegPressure GCNScheduleDAGMILive::getRealRegPressure(
    unsigned RegionIdx,
    std::function<void(const GCNRegPressure &)> PressureCallback) const {
  if (Regions[RegionIdx].first == Regions[RegionIdx].second) {
    GCNRegPressure RP = llvm::getRegPressure(MRI, LiveIns[RegionIdx]);
    if (PressureCallback)
      PressureCallback(RP);
    return RP;
  }
  GCNDownwardRPTracker RPTracker(*LIS);
  bool HasInstrs =
      RPTracker.reset(*Regions[RegionIdx].first, Regions[RegionIdx].second,
                      &LiveIns[RegionIdx]);
  // reset() does not pass the live-in pressure to the callback.
  if (PressureCallback) {
    PressureCallback(RPTracker.getPressure());
    RPTracker.setPressureCallback(std::move(PressureCallback));
  }
  if (HasInstrs)
    RPTracker.advance(Regions[RegionIdx].second);
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
    RPTracker.reset(*MBB->begin(), MBB->end(), &LiveIn);
    MBBLiveIns.erase(LiveInIt);
  } else {
    I = Rgn.first;
    auto LRS = BBLiveInMap.lookup(NonDbgMI);
#ifdef EXPENSIVE_CHECKS
    assert(isEqual(getLiveRegsBefore(*NonDbgMI, *LIS), LRS));
#endif
    RPTracker.reset(*I, I->getParent()->end(), &LRS);
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
    RPTracker.advanceBeforeNext();
    RPTracker.advanceToNext();
  }

  if (OnlySucc) {
    if (I != MBB->end()) {
      RPTracker.advanceBeforeNext();
      RPTracker.advanceToNext();
      RPTracker.advance(MBB->end());
    }
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
  RegionsWithHighRP.resize(static_cast<unsigned>(Regions.size()));
  RegionsWithExcessRP.resize(static_cast<unsigned>(Regions.size()));
  RegionsWithIGLPInstrs.resize(static_cast<unsigned>(Regions.size()));
  RegionsWithHighRP.reset();
  RegionsWithExcessRP.reset();
  RegionsWithIGLPInstrs.reset();

  runSchedStages();
}

void GCNScheduleDAGMILive::runSchedStages() {
  LLVM_DEBUG(dbgs() << "All regions recorded, starting actual scheduling.\n");

  GCNSchedStrategy &S = static_cast<GCNSchedStrategy &>(*SchedImpl);
  if (!Regions.empty()) {
    BBLiveInMap = getRegionLiveInMap();
    if (S.useGCNTrackers())
      RegionLiveOuts.buildLiveRegMap();
  }

#ifndef NDEBUG
  if (ST.getCLOpts().print_max_reg_pressure_regusage_before_scheduler) {
    dumpMaxRegPressure(MF, GCNRegPressure::VGPR, *LIS, MLI);
    dumpMaxRegPressure(MF, GCNRegPressure::SGPR, *LIS, MLI);
    LIS->dump();
  }
#endif

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

      if (S.useGCNTrackers()) {
        const unsigned RegionIdx = Stage->getRegionIdx();
        S.getDownwardTracker().reset(MRI, LiveIns[RegionIdx]);
        S.getUpwardTracker().reset(
            MRI, RegionLiveOuts.getLiveRegsForRegionIdx(RegionIdx));
      }

      ScheduleDAGMILive::schedule();
      Stage->finalizeGCNRegion();
      Stage->advanceRegion();
      exitRegion();
    }

    Stage->finalizeGCNSchedStage();
  }

#ifndef NDEBUG
  if (ST.getCLOpts().print_max_reg_pressure_regusage_after_scheduler) {
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
  case GCNSchedStageID::LiveIntervalRPReschedule:
    OS << "Live Interval RP Reschedule";
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
  if (!ST.getCLOpts().unclustered_high_rp_reschedule)
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
  if (!ST.getCLOpts().clustered_low_occupancy_reschedule)
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

bool PreRARematStage::initGCNSchedStage() {
  // FIXME: This pass will invalidate cached BBLiveInMap and MBBLiveIns for
  // regions inbetween the defs and region we sinked the def to. Will need to be
  // fixed if there is another pass after this pass.
  assert(!S.hasNextStage());

  if (!GCNSchedStage::initGCNSchedStage() || DAG.Regions.size() <= 1)
    return false;

  // Set up "spilling targets" for all regions.
  const Function &F = MF.getFunction();
  unsigned MaxSGPRs = ST.getMaxNumSGPRs(F);
  unsigned MaxVGPRs = ST.getMaxNumVGPRs(F);

  const unsigned VGPRBlockSize = MFI.getDynamicVGPRBlockSize();
  bool HasVectorRegisterExcess = false;
  for (const GCNRegPressure &RP : DAG.Pressure) {
    if (GCNRPTarget(MaxSGPRs, MaxVGPRs, MF, RP).hasVectorRegisterExcess()) {
      HasVectorRegisterExcess = true;
      break;
    }
  }
  if (HasVectorRegisterExcess || DAG.MinOccupancy >= MFI.getMaxWavesPerEU()) {
    // In addition to register usage being above addressable limits, occupancy
    // below the minimum is considered like "spilling" as well.
    TargetOcc = std::nullopt;
  } else {
    // There is no spilling and room to improve occupancy; set up "increased
    // occupancy targets" for all regions.
    TargetOcc = DAG.MinOccupancy + 1;
    MaxSGPRs = ST.getMaxNumSGPRs(*TargetOcc, false);
    MaxVGPRs = ST.getMaxNumVGPRs(*TargetOcc, VGPRBlockSize);
  }

  LLVM_DEBUG({
    dbgs() << "In function ";
    MF.getFunction().printAsOperand(dbgs(), false);
    dbgs() << ": rematerializing to ";
    if (TargetOcc) {
      dbgs() << "increase occupancy from " << *TargetOcc - 1 << '\n';
    } else {
      dbgs() << "reduce spilling (minimum target occupancy is "
             << MFI.getMinWavesPerEU() << ")\n";
    }
  });

  LiveOuts.reserve(DAG.Regions.size());
  DAG.RegionLiveOuts.buildLiveRegMap();
  for (unsigned RegionIdx : seq(DAG.Regions.size()))
    LiveOuts.push_back(DAG.RegionLiveOuts.getLiveRegsForRegionIdx(RegionIdx));

  Strategy = std::make_unique<GCNRematStrategy>(MF, DAG.Regions, DAG.LiveIns,
                                                LiveOuts, *DAG.LIS, *DAG.MBFI);
  if (TargetOcc.has_value()) {
    // Rematerialization is likely to move instructions into higher frequency
    // regions, increasing their total sum latency. This is acceptable if we are
    // eliminating spills in the process, but when the goal is increasing
    // occupancy we get nothing out of rematerialization if occupancy is not
    // increased in the end; in such cases we want to be able to roll back
    // rematerializations.
    Strategy->startRecordingRemats();
  }

  RescheduleRegions.resize(DAG.Regions.size());
  MutableArrayRef<GCNRegPressure> MaxPressure = DAG.Pressure;
  Strategy->rematerializeFor(MaxSGPRs, MaxVGPRs,
                             /*ReduceSpill=*/!TargetOcc.has_value(),
                             RescheduleRegions, &MaxPressure);
  if (RescheduleRegions.none())
    return false;

  // Reflect changes in the DAG and compute achieved occupancy.
  LLVM_DEBUG(dbgs() << "==== REMAT RESULTS ====\n");
  for (unsigned RegionIdx : RescheduleRegions.set_bits()) {
    DAG.RegionLiveOuts.getLiveRegsForRegionIdx(RegionIdx) = LiveOuts[RegionIdx];
    LLVM_DEBUG(dbgs() << '[' << RegionIdx << "] Achieved occupancy "
                      << DAG.Pressure[RegionIdx].getOccupancy(ST, VGPRBlockSize)
                      << '\n');
  }
  AchievedOcc = MFI.getMaxWavesPerEU();
  for (const GCNRegPressure &RP : DAG.Pressure)
    AchievedOcc = std::min(AchievedOcc, RP.getOccupancy(ST, VGPRBlockSize));

  LLVM_DEBUG({
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

  unsigned NumRegionInstrs =
      static_cast<unsigned>(std::distance(DAG.begin(), DAG.end()));
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
  return !RevertAllRegions && RescheduleRegions[RegionIdx] &&
         GCNSchedStage::initGCNRegion();
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
    LLVM_DEBUG(dbgs() << "Region " << RegionIdx
                      << " cannot meet occupancy target, interrupting "
                         "re-scheduling in all regions\n");
    RevertAllRegions = true;
  }
}

void GCNSchedStage::checkScheduling() {
  // The maximum number of arch VGPR on non-unified register file, or the
  // maximum VGPR + AGPR in the unified register file case.
  unsigned MaxVGPRs = ST.getMaxNumVGPRs(MF);
  // The maximum number of arch VGPR for both unified and non-unified register
  // file.
  unsigned MaxArchVGPRs = std::min(MaxVGPRs, ST.getAddressableNumArchVGPRs());

  // The region maximum keeps the VGPR and AVGPR pressure of a single point, so
  // distributing it across the banks can miss a rigid peak at another point.
  // Distribute the pressure of each point instead.
  bool AVGPRBanking = ST.getCLOpts().avgpr_banking;
  unsigned ArchVGPRBankLimit, AGPRBankLimit;
  std::tie(ArchVGPRBankLimit, AGPRBankLimit) = S.getBankLimits();
  unsigned BankedArchVGPRPeak = 0, BankedAGPRPeak = 0, BankedUnifiedPeak = 0;
  std::function<void(const GCNRegPressure &)> TrackBankedPeaks;
  if (AVGPRBanking) {
    TrackBankedPeaks = [&](const GCNRegPressure &RP) {
      auto [ArchNum, AccNum] =
          RP.getBankedVGPRNum(ArchVGPRBankLimit, AGPRBankLimit);
      BankedArchVGPRPeak = std::max(BankedArchVGPRPeak, ArchNum);
      BankedAGPRPeak = std::max(BankedAGPRPeak, AccNum);
      BankedUnifiedPeak =
          std::max(BankedUnifiedPeak,
                   GCNRegPressure::getBankedUnifiedVGPRNum(ArchNum, AccNum));
    };
  }

  // Check the results of scheduling.
  PressureAfter = DAG.getRealRegPressure(RegionIdx, TrackBankedPeaks);

  LLVM_DEBUG(dbgs() << "Pressure after scheduling: " << print(PressureAfter));
  LLVM_DEBUG(dbgs() << "Region: " << RegionIdx << ".\n");

  unsigned DynamicVGPRBlockSize = DAG.MFI.getDynamicVGPRBlockSize();

  bool UnifiedVGPRFile = ST.hasGFX90AInsts();
  unsigned VectorCriticalLimit =
      UnifiedVGPRFile ? S.UnifiedVGPRCriticalLimit : S.VGPRCriticalLimit;
  // Fitting the combined occupancy budget does not rule out spilling in one
  // bank. Keep the per-bank excess checks before taking this early exit.
  if (PressureAfter.getSGPRNum() <= S.SGPRCriticalLimit &&
      PressureAfter.getVGPRNum(UnifiedVGPRFile) <= VectorCriticalLimit &&
      (!UnifiedVGPRFile ||
       (PressureAfter.getArchVGPRNum() <= S.VGPRExcessLimit &&
        PressureAfter.getAGPRNum() <= S.AGPRExcessLimit))) {
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
  unsigned MaxSGPRs = ST.getMaxNumSGPRs(MF);

  bool BankExcess = AVGPRBanking
                        ? BankedArchVGPRPeak > ArchVGPRBankLimit ||
                              BankedAGPRPeak > AGPRBankLimit
                        : PressureAfter.getArchVGPRNum() > MaxArchVGPRs ||
                              PressureAfter.getAGPRNum() > MaxArchVGPRs;

  unsigned VGPRNum = AVGPRBanking && UnifiedVGPRFile
                         ? BankedUnifiedPeak
                         : PressureAfter.getVGPRNum(UnifiedVGPRFile);
  if (VGPRNum > MaxVGPRs || BankExcess ||
      PressureAfter.getSGPRNum() > MaxSGPRs) {
    DAG.RegionsWithHighRP[RegionIdx] = true;
    DAG.RegionsWithExcessRP[RegionIdx] = true;
  }

  // Revert if this region's schedule would cause a drop in occupancy or
  // spilling.
  if (shouldRevertScheduling(WavesAfter)) {
    modifyRegionSchedule(RegionIdx, Unsched);
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
        ST, PressureBefore.getVGPRNum(false),
        DAG.MFI.getDynamicVGPRBlockSize());
    unsigned BlocksAfter = AMDGPU::IsaInfo::getAllocatedNumVGPRBlocks(
        ST, PressureAfter.getVGPRNum(false), DAG.MFI.getDynamicVGPRBlockSize());
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

  if (shouldRevertInitialSchedule(WavesAfter))
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

  // Do not attempt to relax schedule even more if we are already spilling
  // VGPRs, or if reverting would spill VGPRs. SGPRs spill to VGPR lanes, which
  // is cheap, so if only SGPRs are in excess, still check that the new
  // schedule does not cost too much latency.
  if (isRegionWithExcessRP() &&
      (GCNRPTarget(MF, PressureBefore).hasVectorRegisterExcess() ||
       GCNRPTarget(MF, PressureAfter).hasVectorRegisterExcess()))
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
  unsigned ScheduleMetricBias = ST.getCLOpts().schedule_metric_bias;
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
  return GCNSchedStage::shouldRevertScheduling(WavesAfter) ||
         mayCauseSpilling(WavesAfter);
}

bool ILPInitialScheduleStage::shouldRevertScheduling(unsigned WavesAfter) {
  if (shouldRevertInitialSchedule(WavesAfter))
    return true;

  return false;
}

bool MemoryClauseInitialScheduleStage::shouldRevertScheduling(
    unsigned WavesAfter) {
  return shouldRevertInitialSchedule(WavesAfter);
}

bool LiveIntervalRPStage::initGCNSchedStage() {
  if (!ST.getCLOpts().lirp_reschedule)
    return false;

  if (!GCNSchedStage::initGCNSchedStage())
    return false;

  if (!S.VGPRThresholdPercent) {
    LLVM_DEBUG(dbgs() << "LIRP: expected VGPRThresholdPercent to be enabled, "
                         "not using live interval RP reschedule stage\n");
    return false;
  }

  return true;
}

bool LiveIntervalRPStage::initGCNRegion() {
  const AMDGPUOptions &CLOpts = ST.getCLOpts();
  unsigned InstantRP = DAG.Pressure[RegionIdx].getArchVGPRNum();
  auto [RegionBegin, RegionEnd] = DAG.Regions[RegionIdx];
  if (RegionBegin == RegionEnd)
    return false;

  unsigned LIRP = estimateGreedyVGPRPressure(
      RegionBegin, RegionEnd, DAG.LiveIns[RegionIdx], *DAG.getLIS(),
      DAG.MF.getRegInfo(), static_cast<const SIRegisterInfo &>(*DAG.TRI));

  unsigned NewVGPRThresholdPercent =
      (S.VGPRThresholdPercent * CLOpts.lirp_vgpr_reduction + 99) / 100;

  LLVM_DEBUG(dbgs() << "LIRP: Region " << RegionIdx
                    << ", VGPRThresholdPercent: " << S.VGPRThresholdPercent
                    << " -> " << NewVGPRThresholdPercent
                    << ", VGPRExcessLimit=" << S.VGPRExcessLimit
                    << ", VGPRCriticalLimit=" << S.VGPRCriticalLimit
                    << ", InstantRP=" << InstantRP << ", LIRP=" << LIRP);

  bool DoRescheduling = false;
  // Lower bound on InstantRP to skip over tiny regions.
  unsigned InstantRPLowerBound =
      S.VGPRExcessLimit * CLOpts.lirp_instant_lower_bound / 100;
  if (LIRP > S.VGPRExcessLimit) {
    LLVM_DEBUG(dbgs() << " [LIRP exceeds the limit (" << S.VGPRExcessLimit
                      << "), rescheduling]");
    DoRescheduling = true;
  } else if (LIRP > InstantRP && InstantRP > InstantRPLowerBound) {
    unsigned IncreasePercent = ((LIRP - InstantRP) * 100) / InstantRP;
    if (IncreasePercent > CLOpts.lirp_threshold) {
      LLVM_DEBUG(dbgs() << " [" << IncreasePercent << "% > "
                        << CLOpts.lirp_threshold << "%, rescheduling]");
      DoRescheduling = true;
    }
  }
  LLVM_DEBUG(dbgs() << '\n');

  if (DoRescheduling && GCNSchedStage::initGCNRegion()) {
    SavedVGPRExcessLimit = S.VGPRExcessLimit;
    SavedVGPRCriticalLimit = S.VGPRCriticalLimit;
    SavedUnifiedVGPRCriticalLimit = S.UnifiedVGPRCriticalLimit;
    SavedVGPRThresholdPercent = S.VGPRThresholdPercent;
    S.VGPRThresholdPercent = NewVGPRThresholdPercent;
    return true;
  }

  return false;
}

void LiveIntervalRPStage::finalizeGCNRegion() {
  S.VGPRExcessLimit = SavedVGPRExcessLimit;
  S.VGPRCriticalLimit = SavedVGPRCriticalLimit;
  S.UnifiedVGPRCriticalLimit = SavedUnifiedVGPRCriticalLimit;
  S.VGPRThresholdPercent = SavedVGPRThresholdPercent;
  GCNSchedStage::finalizeGCNRegion();
}

bool GCNSchedStage::mayCauseSpilling(unsigned WavesAfter) {
  if (WavesAfter <= MFI.getMinWavesPerEU() && isRegionWithExcessRP() &&
      !PressureAfter.less(MF, PressureBefore)) {
    LLVM_DEBUG(dbgs() << "New pressure will result in more spilling.\n");
    return true;
  }

  return false;
}

bool GCNSchedStage::reducesLatency() {
  if (DAG.SUnits.empty())
    return false;

  ScheduleMetrics MBefore = getScheduleMetrics(DAG.SUnits);
  ScheduleMetrics MAfter = getScheduleMetrics(DAG);
  LLVM_DEBUG(dbgs() << "Metric before " << MBefore << "\tMetric after "
                    << MAfter);
  return MAfter.getMetric() < MBefore.getMetric();
}

bool GCNSchedStage::shouldRevertInitialSchedule(unsigned WavesAfter) {
  if (!mayCauseSpilling(WavesAfter))
    return false;
  if (PressureBefore.less(MF, PressureAfter))
    return true;

  if (!reducesLatency()) {
    LLVM_DEBUG(dbgs() << "Equivalent RP, new schedule does not reduce "
                         "latency.\n");
    return true;
  }
  LLVM_DEBUG(dbgs() << "Equivalent RP, keeping schedule with better "
                       "latency.\n");
  return false;
}

void GCNSchedStage::modifyRegionSchedule(unsigned RegionIdx,
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
  MachineBasicBlock *MBB = MIOrder.front()->getParent();
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
      // MI is already at the expected position. However, earlier splices in
      // this loop may have changed neighboring slot indices, so this MI's
      // slot index can become non-monotonic w.r.t. the physical MBB order.
      // Only re-seat when monotonicity is actually violated to avoid
      // unnecessary LiveInterval changes that could perturb scheduling.
      if (!MI->isDebugInstr()) {
        SlotIndex MIIdx = DAG.LIS->getInstructionIndex(*MI);
        SlotIndex PrevIdx = DAG.LIS->getSlotIndexes()->getIndexBefore(*MI);
        if (PrevIdx >= MIIdx)
          DAG.LIS->handleMove(*MI, true);
      }
      ++RegionEnd;
    }
    if (MI->isDebugInstr()) {
      LLVM_DEBUG(dbgs() << "Scheduling " << *MI);
      continue;
    }

    // Reset read-undef flags and update them later.
    RegisterOperands::restoreLivenessFlags(*MI, *DAG.TRI, DAG.MRI, *DAG.LIS,
                                           DAG.ShouldTrackLaneMasks);
    LLVM_DEBUG(dbgs() << "Scheduling " << *MI);
  }

  // The region end doesn't change throughout scheduling since it itself is
  // outside the region (whether that is a MBB end or a terminator MI).
  assert(RegionEnd == DAG.Regions[RegionIdx].second && "region end mismatch");
  DAG.Regions[RegionIdx].first = MIOrder.front();
}

unsigned PreRARematStage::getStageTargetOccupancy() const {
  return TargetOcc ? *TargetOcc : MFI.getMinWavesPerEU();
}

void PreRARematStage::finalizeGCNSchedStage() {
  // We consider that reducing spilling is always beneficial so we never
  // rollback rematerializations or revert scheduling in such cases.
  if (!TargetOcc)
    return;

  // When increasing occupancy, it is possible that re-scheduling is not able to
  // achieve the target occupancy in all regions, in which case re-scheduling in
  // all regions should be reverted.
  if (DAG.MinOccupancy >= *TargetOcc)
    return;

  // Revert re-scheduling in all affected regions.
  for (const auto &[RegionIdx, OrigMIOrder, MaxPressure] : RegionReverts) {
    LLVM_DEBUG(dbgs() << "Reverting re-scheduling in region " << RegionIdx
                      << '\n');
    DAG.Pressure[RegionIdx] = MaxPressure;
    modifyRegionSchedule(RegionIdx, OrigMIOrder);
  }

  // It is possible that re-scheduling lowers occupancy over the one achieved
  // just through rematerializations, in which case we revert re-scheduling in
  // all regions but do not roll back rematerializations.
  if (AchievedOcc >= *TargetOcc) {
    DAG.setTargetOccupancy(AchievedOcc);
    return;
  }

  // Reset the target occupancy to what it was pre-rematerialization.
  DAG.setTargetOccupancy(*TargetOcc - 1);
  Strategy->rollbackRecordedRemats();
  for (unsigned RegionIdx : RescheduleRegions.set_bits()) {
    DAG.RegionLiveOuts.getLiveRegsForRegionIdx(RegionIdx) = LiveOuts[RegionIdx];
    DAG.Pressure[RegionIdx] = DAG.getRealRegPressure(RegionIdx);
  }

#ifdef EXPENSIVE_CHECKS
  // In particular, we want to check for coherent MI/slot order in regions in
  // which reverts and/or rollbacks may have happened.
  MF.verify();
#endif

  GCNSchedStage::finalizeGCNSchedStage();
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
