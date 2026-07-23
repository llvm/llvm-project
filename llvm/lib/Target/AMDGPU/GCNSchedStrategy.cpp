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
#include "AMDGPU.h"
#include "AMDGPUIGroupLP.h"
#include "GCNHazardRecognizer.h"
#include "GCNRegPressure.h"
#include "SIMachineFunctionInfo.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/CodeGen/CalcSpillWeights.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineBlockFrequencyInfo.h"
#include "llvm/CodeGen/MachineBranchProbabilityInfo.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/RegisterClassInfo.h"
#include "llvm/CodeGen/Rematerializer.h"
#include "llvm/MC/LaneBitmask.h"
#include "llvm/MC/MCSchedule.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/ErrorHandling.h"

#include <limits>

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

static cl::opt<bool> EnableMFMAFragmentSchedulerOpt(
    "amdgpu-enable-mfma-fragment-scheduler", cl::Hidden,
    cl::desc("Enable gfx950 DS_READ/MFMA fragment scheduling"), cl::init(true));

bool llvm::isMFMAFragmentSchedulerEnabled(const GCNSubtarget &ST) {
  return EnableMFMAFragmentSchedulerOpt && ST.hasGFX950Insts();
}

struct MFMAFragmentSchedSettings {
  // These values encode the gfx950 sequence measured for DS_READ_B128 feeding
  // V_MFMA_F32_32X32X16_F16. They must not be applied to other subtargets.

  // Maximum DS_READ/MFMA fragment-window size before the first MFMA is picked
  // in a scheduling region.
  unsigned PrologueWindow = 4;

  // Number of DS_READs and MFMAs in one steady-state pipeline group. This also
  // defines the target DS_READ maturity spacing and the minimum useful MFMA
  // fanout of a pull-forward producer.
  unsigned PipelineGroupSize = 2;

  // Maximum number of DS_READs used to fill an MFMA issue-resource stall. It
  // currently follows the pipeline group size, but is a separate policy.
  unsigned MaxDSReadsInMFMAStall = PipelineGroupSize;

  // Maximum effective stall allowed when forcing an MFMA drain consumer over
  // another DS_READ to maintain the fragment drain burst pattern.
  unsigned MaxDrainMFMAStall = 8;

  // Maximum number of trailing MFMAs that the bottom boundary may reserve as
  // an epilogue before yielding fragment-pipeline work to the top boundary.
  unsigned BottomEpilogueMFMAs = 4;
};

static constexpr MFMAFragmentSchedSettings MFMAFragmentSched;

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
    cl::desc("Disable rewrite mfma rewrite scheduling stage"), cl::init(true));

namespace {

struct VGPRThresholdParser : public cl::parser<unsigned> {
  VGPRThresholdParser(cl::Option &O) : cl::parser<unsigned>(O) {}

  bool parse(cl::Option &O, StringRef ArgName, StringRef Arg, unsigned &Value) {
    if (Arg.getAsInteger(0, Value))
      return O.error("'" + Arg + "' value invalid for uint argument!");

    if (Value > 100)
      return O.error("'" + Arg + "' value must be in the range [0, 100]!");

    return false;
  }
};

} // end anonymous namespace

static cl::opt<unsigned, false, VGPRThresholdParser> VGPRThresholdPercentOpt(
    "amdgpu-vgpr-threshold-percent", cl::Hidden,
    cl::desc("Percent of VGPR limits that we should use as RP threshold "
             "during scheduling. We have two limits relevant to scheduling: "
             "Critical (avoid decreasing occupancy), Excess (avoid spilling). "
             "This flag scales both limits back by an equal percent: (0 = use "
             " default calculation, 1-100 = use percentage), default: 0"),
    cl::init(0));

const unsigned ScheduleMetrics::ScaleFactor = 100;

static bool hasMFMAFragmentPipeline(const ScheduleDAGMI &DAG);

GCNSchedStrategy::GCNSchedStrategy(const MachineSchedContext *C)
    : GenericScheduler(C), TargetOccupancy(0), MF(nullptr),
      DownwardTracker(*C->LIS), UpwardTracker(*C->LIS), HasHighPressure(false) {
  if (GCNTrackers.getNumOccurrences() > 0)
    GCNTrackersOverride = GCNTrackers;
}

void GCNSchedStrategy::initialize(ScheduleDAGMI *DAG) {
  GenericScheduler::initialize(DAG);

  MF = &DAG->MF;
  resetFragmentWindows();

  const GCNSubtarget &ST = MF->getSubtarget<GCNSubtarget>();
  // The subtarget option only makes the tune available. Keep the generic
  // scheduler completely unchanged for regions without the DS_READ/MFMA data
  // flow this policy models.
  EnableMFMAFragmentScheduler =
      isMFMAFragmentSchedulerEnabled(ST) && hasMFMAFragmentPipeline(*DAG);

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
        AMDGPU::IsaInfo::getVGPRAllocGranule(ST, DynamicVGPRBlockSize);
    unsigned Addressable =
        AMDGPU::IsaInfo::getAddressableNumVGPRs(ST, DynamicVGPRBlockSize);
    unsigned VGPRBudget = alignDown(Addressable / TargetOccupancy, Granule);
    VGPRBudget = std::max(VGPRBudget, Granule);
    VGPRCriticalLimit = std::min(VGPRBudget, VGPRExcessLimit);
  }
  // Apply VGPR excess threshold percentage if specified.
  if (VGPRThresholdPercentOpt > 0) {
    [[maybe_unused]] unsigned OriginalVGPRExcessLimit = VGPRExcessLimit;
    [[maybe_unused]] unsigned OriginalVGPRCriticalLimit = VGPRCriticalLimit;
    VGPRExcessLimit = (VGPRThresholdPercentOpt * VGPRExcessLimit + 99) / 100;
    VGPRCriticalLimit =
        (VGPRThresholdPercentOpt * VGPRCriticalLimit + 99) / 100;
    LLVM_DEBUG(dbgs() << "Applied VGPR excess threshold "
                      << VGPRThresholdPercentOpt << "%, VGPRExcessLimit: "
                      << OriginalVGPRExcessLimit << " -> " << VGPRExcessLimit
                      << ". VGPRCriticalLimit: " << OriginalVGPRCriticalLimit
                      << " -> " << VGPRCriticalLimit << '\n');
  } else {
    VGPRExcessLimit -= std::min(VGPRLimitBias + ErrorMargin, VGPRExcessLimit);
    VGPRCriticalLimit -=
        std::min(VGPRLimitBias + ErrorMargin, VGPRCriticalLimit);
  }

  // Subtract error margin and bias from register limits and avoid overflow.
  SGPRCriticalLimit -= std::min(SGPRLimitBias + ErrorMargin, SGPRCriticalLimit);
  SGPRExcessLimit -= std::min(SGPRLimitBias + ErrorMargin, SGPRExcessLimit);
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

static bool isTargetDSReadOpcode(unsigned Opcode) {
  switch (Opcode) {
  case AMDGPU::DS_READ_B128:
  case AMDGPU::DS_READ_B128_gfx9:
    return true;
  default:
    return false;
  }
}

static bool isTargetMFMAOpcode(unsigned Opcode) {
  switch (Opcode) {
  case AMDGPU::V_MFMA_F32_32X32X16_F16_mac_vgprcd_e64:
    return true;
  default:
    return false;
  }
}

template <typename PredicateT>
static bool hasBundledMI(const MachineInstr *MI, PredicateT Predicate) {
  if (!MI)
    return false;

  if (Predicate(*MI))
    return true;

  if (!MI->isBundle())
    return false;

  MachineBasicBlock::const_instr_iterator BundleI = MI->getIterator();
  for (++BundleI;
       BundleI != MI->getParent()->instr_end() && BundleI->isBundledWithPred();
       ++BundleI) {
    if (Predicate(*BundleI))
      return true;
  }

  return false;
}

static bool isDSReadLike(const SUnit *SU) {
  if (!SU)
    return false;

  return hasBundledMI(SU->getInstr(), [](const MachineInstr &MI) {
    return isTargetDSReadOpcode(MI.getOpcode());
  });
}

static bool isMFMALike(const SUnit *SU) {
  if (!SU)
    return false;

  return hasBundledMI(SU->getInstr(), [](const MachineInstr &MI) {
    return isTargetMFMAOpcode(MI.getOpcode());
  });
}

static bool hasMFMAFragmentPipeline(const ScheduleDAGMI &DAG) {
  for (const SUnit &SU : DAG.SUnits) {
    if (!isDSReadLike(&SU))
      continue;
    for (const SDep &Succ : SU.Succs)
      if (Succ.getKind() == SDep::Data && isMFMALike(Succ.getSUnit()))
        return true;
  }
  return false;
}

static bool isSafeMFMAFragmentRampUpFiller(const SUnit *SU) {
  if (!SU || isDSReadLike(SU) || isMFMALike(SU))
    return false;

  // This tune does not model buffered-load maturity windows or available
  // in-flight memory capacity. Keep memory operations, calls, barriers, and
  // other side effects under the normal scheduler policy.
  return !hasBundledMI(SU->getInstr(), [](const MachineInstr &MI) {
    return MI.mayLoadOrStore() || MI.isCall() || MI.isTerminator() ||
           MI.isBarrier() || MI.hasUnmodeledSideEffects();
  });
}

static unsigned getBoundaryReadyCycle(const SchedBoundary &Zone,
                                      const SUnit *SU) {
  return Zone.isTop() ? SU->TopReadyCycle : SU->BotReadyCycle;
}

static unsigned getReadyStall(const SchedBoundary &Zone, const SUnit *SU) {
  unsigned ReadyCycle = getBoundaryReadyCycle(Zone, SU);
  unsigned CurrCycle = Zone.getCurrCycle();
  return ReadyCycle > CurrCycle ? ReadyCycle - CurrCycle : 0;
}

static unsigned getIssueStall(SchedBoundary &Zone, SUnit *SU) {
  if (!SU || !SU->hasReservedResource || !Zone.SchedModel ||
      !Zone.SchedModel->hasInstrSchedModel())
    return 0;

  const MCSchedClassDesc *SC = Zone.DAG->getSchedClass(SU);
  unsigned Stall = 0;
  for (const MCWriteProcResEntry &PE :
       make_range(Zone.SchedModel->getWriteProcResBegin(SC),
                  Zone.SchedModel->getWriteProcResEnd(SC))) {
    unsigned NextCycle;
    unsigned InstanceIdx;
    std::tie(NextCycle, InstanceIdx) = Zone.getNextResourceCycle(
        SC, PE.ProcResourceIdx, PE.ReleaseAtCycle, PE.AcquireAtCycle);
    (void)InstanceIdx;
    if (NextCycle > Zone.getCurrCycle())
      Stall = std::max(Stall, NextCycle - Zone.getCurrCycle());
  }

  return Stall;
}

static unsigned getEffectiveStall(SchedBoundary &Zone, SUnit *SU) {
  return std::max(getReadyStall(Zone, SU), getIssueStall(Zone, SU));
}

static bool consumesThisDSRead(const SUnit *SU, const SUnit *Pred) {
  if (!isMFMALike(SU) || !isDSReadLike(Pred))
    return false;

  for (const SDep &PredDep : SU->Preds) {
    if (PredDep.getSUnit() == Pred)
      return true;
  }

  return false;
}

static bool hasScheduledMFMASuccessor(const SUnit *SU) {
  if (!isDSReadLike(SU))
    return false;

  for (const SDep &SuccDep : SU->Succs) {
    const SUnit *Succ = SuccDep.getSUnit();
    if (isMFMALike(Succ) && Succ->isScheduled)
      return true;
  }

  return false;
}

enum class PipeKind {
  // Not part of the DS_READ/MFMA fragment pipeline.
  None,

  // A DS_READ that may produce an operand fragment for an MFMA.
  FragmentProducer,

  // An MFMA with no remaining scheduler-model dependency stall.
  ReadyMFMA,

  // An MFMA whose scheduler-model dependency-ready cycle is still ahead of
  // the current scheduling boundary.
  PendingMFMA,
};

static PipeKind classifyPipeKind(SchedBoundary &Zone, SUnit *SU) {
  if (isDSReadLike(SU))
    return PipeKind::FragmentProducer;

  if (isMFMALike(SU))
    return getReadyStall(Zone, SU) ? PipeKind::PendingMFMA
                                   : PipeKind::ReadyMFMA;

  return PipeKind::None;
}

static unsigned getNumDSReadsToUnlockClosestMFMA(const SUnit *SU) {
  // For every unscheduled MFMA successor, count its other unscheduled DS_READ
  // predecessors. Return the minimum count, excluding SU itself, to prefer the
  // DS_READ that can unlock any MFMA with the fewest additional reads.
  if (!isDSReadLike(SU))
    return std::numeric_limits<unsigned>::max();

  unsigned MinRemaining = std::numeric_limits<unsigned>::max();
  for (const SDep &SuccDep : SU->Succs) {
    const SUnit *Succ = SuccDep.getSUnit();
    if (!isMFMALike(Succ) || Succ->isScheduled)
      continue;

    unsigned Remaining = 0;
    for (const SDep &PredDep : Succ->Preds) {
      const SUnit *Pred = PredDep.getSUnit();
      if (!isDSReadLike(Pred) || Pred == SU || Pred->isScheduled)
        continue;
      ++Remaining;
    }

    MinRemaining = std::min(MinRemaining, Remaining);
  }

  return MinRemaining;
}

static unsigned
countScheduledDSReadPredsToMFMASuccessorAfter(const SUnit *SU,
                                              const SUnit *Succ) {
  unsigned Count = 0;
  for (const SDep &PredDep : Succ->Preds) {
    const SUnit *Pred = PredDep.getSUnit();
    if (!isDSReadLike(Pred))
      continue;
    if (Pred == SU || Pred->isScheduled)
      ++Count;
  }

  return Count;
}

static unsigned
countUnscheduledDSReadPredsToMFMASuccessorAfter(const SUnit *SU,
                                                const SUnit *Succ) {
  unsigned Count = 0;
  for (const SDep &PredDep : Succ->Preds) {
    const SUnit *Pred = PredDep.getSUnit();
    if (!isDSReadLike(Pred) || Pred == SU || Pred->isScheduled)
      continue;
    ++Count;
  }

  return Count;
}

static bool
hasUnscheduledNonDSReadPredsToMFMASuccessorAfter(const SUnit *SU,
                                                 const SUnit *Succ) {
  for (const SDep &PredDep : Succ->Preds) {
    const SUnit *Pred = PredDep.getSUnit();
    if (isDSReadLike(Pred) || Pred == SU || Pred->isScheduled)
      continue;
    return true;
  }

  return false;
}

static unsigned countMFMASuccessorsUnlockedBy(const SUnit *SU) {
  if (!isDSReadLike(SU))
    return 0;

  unsigned Count = 0;
  for (const SDep &SuccDep : SU->Succs) {
    const SUnit *Succ = SuccDep.getSUnit();
    if (!isMFMALike(Succ) || Succ->isScheduled)
      continue;

    if (hasUnscheduledNonDSReadPredsToMFMASuccessorAfter(SU, Succ))
      continue;

    if (countUnscheduledDSReadPredsToMFMASuccessorAfter(SU, Succ) == 0)
      ++Count;
  }

  return Count;
}

static unsigned countMFMASuccessorsUnlockedByAfterFiller(const SUnit *SU,
                                                         const SUnit *Filler) {
  if (!isDSReadLike(SU) || !isMFMALike(Filler))
    return 0;

  unsigned Count = 0;
  for (const SDep &SuccDep : SU->Succs) {
    const SUnit *Succ = SuccDep.getSUnit();
    if (!isMFMALike(Succ) || Succ->isScheduled || Succ == Filler)
      continue;

    bool HasFillerPred = false;
    bool Blocked = false;
    for (const SDep &PredDep : Succ->Preds) {
      const SUnit *Pred = PredDep.getSUnit();
      if (Pred == SU || Pred->isScheduled)
        continue;

      if (Pred == Filler) {
        HasFillerPred = true;
        continue;
      }

      Blocked = true;
      break;
    }

    if (!Blocked && HasFillerPred)
      ++Count;
  }

  return Count;
}

static unsigned getBestMFMASuccessorMissingDSReadPredsAfter(const SUnit *SU) {
  if (!isDSReadLike(SU))
    return std::numeric_limits<unsigned>::max();

  unsigned BestMissing = std::numeric_limits<unsigned>::max();
  for (const SDep &SuccDep : SU->Succs) {
    const SUnit *Succ = SuccDep.getSUnit();
    if (!isMFMALike(Succ) || Succ->isScheduled)
      continue;

    if (hasUnscheduledNonDSReadPredsToMFMASuccessorAfter(SU, Succ))
      continue;

    unsigned Missing =
        countUnscheduledDSReadPredsToMFMASuccessorAfter(SU, Succ);
    BestMissing = std::min(BestMissing, Missing);
  }

  return BestMissing;
}

static unsigned getBestMFMASuccessorScheduledDSReadPredsAfter(const SUnit *SU) {
  if (!isDSReadLike(SU))
    return 0;

  unsigned BestMissing = getBestMFMASuccessorMissingDSReadPredsAfter(SU);
  if (BestMissing == std::numeric_limits<unsigned>::max())
    return 0;

  unsigned BestScheduled = 0;
  for (const SDep &SuccDep : SU->Succs) {
    const SUnit *Succ = SuccDep.getSUnit();
    if (!isMFMALike(Succ) || Succ->isScheduled)
      continue;

    if (hasUnscheduledNonDSReadPredsToMFMASuccessorAfter(SU, Succ))
      continue;

    if (countUnscheduledDSReadPredsToMFMASuccessorAfter(SU, Succ) !=
        BestMissing)
      continue;

    BestScheduled = std::max(
        BestScheduled, countScheduledDSReadPredsToMFMASuccessorAfter(SU, Succ));
  }

  return BestScheduled;
}

static unsigned getBestMFMASuccessorNodeNumAfter(const SUnit *SU) {
  if (!isDSReadLike(SU))
    return std::numeric_limits<unsigned>::max();

  unsigned BestMissing = getBestMFMASuccessorMissingDSReadPredsAfter(SU);
  if (BestMissing == std::numeric_limits<unsigned>::max())
    return std::numeric_limits<unsigned>::max();

  unsigned BestNodeNum = std::numeric_limits<unsigned>::max();
  for (const SDep &SuccDep : SU->Succs) {
    const SUnit *Succ = SuccDep.getSUnit();
    if (!isMFMALike(Succ) || Succ->isScheduled)
      continue;

    if (hasUnscheduledNonDSReadPredsToMFMASuccessorAfter(SU, Succ))
      continue;

    if (countUnscheduledDSReadPredsToMFMASuccessorAfter(SU, Succ) !=
        BestMissing)
      continue;

    BestNodeNum = std::min(BestNodeNum, Succ->NodeNum);
  }

  return BestNodeNum;
}

static unsigned
countMFMASuccessorsWithFewMissingDSReadPredsAfter(const SUnit *SU,
                                                  unsigned MaxMissing) {
  if (!isDSReadLike(SU))
    return 0;

  unsigned Count = 0;
  for (const SDep &SuccDep : SU->Succs) {
    const SUnit *Succ = SuccDep.getSUnit();
    if (!isMFMALike(Succ) || Succ->isScheduled)
      continue;

    if (hasUnscheduledNonDSReadPredsToMFMASuccessorAfter(SU, Succ))
      continue;

    unsigned Missing =
        countUnscheduledDSReadPredsToMFMASuccessorAfter(SU, Succ);
    if (Missing <= MaxMissing)
      ++Count;
  }

  return Count;
}

struct MFMAFragmentProducerScore {
  bool Valid = false;
  unsigned BestSuccNode = std::numeric_limits<unsigned>::max();
  unsigned BestMissingDS = std::numeric_limits<unsigned>::max();
  unsigned BestScheduledDS = 0;
  unsigned Unlocks = 0;
  unsigned NearUnlocks = 0;
};

static MFMAFragmentProducerScore
getDirectFragmentProducerScore(const SUnit *SU) {
  MFMAFragmentProducerScore Score;
  if (!isDSReadLike(SU))
    return Score;

  for (const SDep &SuccDep : SU->Succs) {
    const SUnit *Succ = SuccDep.getSUnit();
    if (!isMFMALike(Succ) || Succ->isScheduled)
      continue;

    unsigned Missing =
        countUnscheduledDSReadPredsToMFMASuccessorAfter(SU, Succ);
    unsigned Scheduled =
        countScheduledDSReadPredsToMFMASuccessorAfter(SU, Succ);

    Score.Valid = true;
    if (Missing < Score.BestMissingDS) {
      Score.BestMissingDS = Missing;
      Score.BestSuccNode = Succ->NodeNum;
      Score.BestScheduledDS = Scheduled;
      continue;
    }

    if (Missing == Score.BestMissingDS) {
      Score.BestSuccNode = std::min(Score.BestSuccNode, Succ->NodeNum);
      Score.BestScheduledDS = std::max(Score.BestScheduledDS, Scheduled);
    }
  }

  if (!Score.Valid)
    return Score;

  Score.Unlocks = countMFMASuccessorsUnlockedBy(SU);
  Score.NearUnlocks = countMFMASuccessorsWithFewMissingDSReadPredsAfter(SU, 1);
  return Score;
}

static unsigned countDSReadSpacingFillers(SchedBoundary &Zone,
                                          const SUnit *DSRead) {
  if (!isDSReadLike(DSRead))
    return 0;

  SmallPtrSet<SUnit *, 16> Seen;
  unsigned Fillers = 0;
  auto TryFiller = [&](SUnit *SU) {
    if (!SU || SU->isScheduled || !Seen.insert(SU).second || !isMFMALike(SU))
      return;
    if (consumesThisDSRead(SU, DSRead))
      return;
    if (getIssueStall(Zone, SU) != 0)
      return;
    if (getEffectiveStall(Zone, SU) > MFMAFragmentSched.MaxDrainMFMAStall)
      return;

    ++Fillers;
  };

  for (SUnit *SU : Zone.Available)
    TryFiller(SU);
  for (SUnit *SU : Zone.Pending)
    TryFiller(SU);

  return Fillers;
}

static unsigned countIssueReadyFragmentProducers(SchedBoundary &Zone) {
  SmallPtrSet<SUnit *, 16> Seen;
  unsigned Count = 0;
  auto Consider = [&](SUnit *SU) {
    if (!SU || SU->isScheduled || !Seen.insert(SU).second ||
        !isDSReadLike(SU) || getEffectiveStall(Zone, SU) != 0)
      return;
    ++Count;
  };
  for (SUnit *SU : Zone.Available)
    Consider(SU);
  for (SUnit *SU : Zone.Pending)
    Consider(SU);
  return Count;
}

static bool
isBetterPrologueFragmentScore(const MFMAFragmentProducerScore &TryScore,
                              const MFMAFragmentProducerScore &CandScore) {
  if (TryScore.Valid != CandScore.Valid)
    return TryScore.Valid;
  if (!TryScore.Valid)
    return false;

  if (TryScore.BestMissingDS != CandScore.BestMissingDS)
    return TryScore.BestMissingDS < CandScore.BestMissingDS;
  if (TryScore.Unlocks != CandScore.Unlocks)
    return TryScore.Unlocks > CandScore.Unlocks;
  if (TryScore.BestScheduledDS != CandScore.BestScheduledDS)
    return TryScore.BestScheduledDS > CandScore.BestScheduledDS;
  if (TryScore.BestSuccNode != CandScore.BestSuccNode)
    return TryScore.BestSuccNode < CandScore.BestSuccNode;
  return false;
}

static bool mergePrologueFragmentScore(MFMAFragmentProducerScore &Best,
                                       MFMAFragmentProducerScore TryScore) {
  if (!TryScore.Valid)
    return false;

  if (!Best.Valid || isBetterPrologueFragmentScore(TryScore, Best)) {
    Best = TryScore;
    return true;
  }

  return false;
}

static MFMAFragmentProducerScore
getReachableFragmentProducerScoreImpl(const SUnit *SU,
                                      SmallPtrSetImpl<const SUnit *> &Visited,
                                      unsigned DepthLeft) {
  MFMAFragmentProducerScore Best;
  if (!SU || SU->isScheduled || !Visited.insert(SU).second)
    return Best;

  mergePrologueFragmentScore(Best, getDirectFragmentProducerScore(SU));
  if (!DepthLeft)
    return Best;

  for (const SDep &SuccDep : SU->Succs) {
    const SUnit *Succ = SuccDep.getSUnit();
    if (!Succ || Succ->isScheduled || isMFMALike(Succ))
      continue;

    mergePrologueFragmentScore(Best, getReachableFragmentProducerScoreImpl(
                                         Succ, Visited, DepthLeft - 1));
  }

  return Best;
}

static MFMAFragmentProducerScore
getReachableFragmentProducerScore(const SUnit *SU) {
  SmallPtrSet<const SUnit *, 8> Visited;
  return getReachableFragmentProducerScoreImpl(SU, Visited, 6);
}

void GCNSchedStrategy::getRegisterPressures(
    bool AtTop, const RegPressureTracker &RPTracker, SUnit *SU,
    std::vector<unsigned> &Pressure, std::vector<unsigned> &MaxPressure,
    GCNDownwardRPTracker &DownwardTracker, GCNUpwardRPTracker &UpwardTracker,
    ScheduleDAGMI *DAG, const SIRegisterInfo *SRI) {
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

unsigned GCNSchedStrategy::getStructuralStallCycles(SchedBoundary &Zone,
                                                    SUnit *SU) const {
  // Only implemented for top-down scheduling currently.
  if (!Zone.isTop() || !SU)
    return 0;

  MachineInstr *MI = SU->getInstr();
  unsigned CurrCycle = Zone.getCurrCycle();
  unsigned Stall = 0;

  // Query SchedModel for resource stalls (unbuffered resources).
  if (SchedModel->hasInstrSchedModel() && SU->hasReservedResource) {
    const MCSchedClassDesc *SC = DAG->getSchedClass(SU);
    for (const MCWriteProcResEntry &PE :
         make_range(SchedModel->getWriteProcResBegin(SC),
                    SchedModel->getWriteProcResEnd(SC))) {
      unsigned NextAvail =
          Zone.getNextResourceCycle(SC, PE.ProcResourceIdx, PE.ReleaseAtCycle,
                                    PE.AcquireAtCycle)
              .first;
      if (NextAvail > CurrCycle)
        Stall = std::max(Stall, NextAvail - CurrCycle);
    }
  }

  // Query HazardRecognizer for sequence-dependent hazard penalties.
  // AMDGPU currently installs GCNHazardRecognizer for MI scheduling only in
  // the post-RA configuration without vreg liveness.
  if (!DAG->hasVRegLiveness() && Zone.HazardRec &&
      Zone.HazardRec->isEnabled()) {
    auto *HR = static_cast<GCNHazardRecognizer *>(Zone.HazardRec);
    Stall = std::max(Stall, HR->getHazardWaitStates(MI));
  }

  return Stall;
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
  if (AtTop || !canUsePressureDiffs(*SU) || useGCNTrackers()) {
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

static bool hasReadyMFMAPending(SchedBoundary &Zone) {
  for (SUnit *SU : Zone.Pending)
    if (classifyPipeKind(Zone, SU) == PipeKind::ReadyMFMA)
      return true;
  return false;
}

static bool shouldKeepAvailableUnlockingProducer(
    SchedBoundary &Zone, const SUnit *SU, unsigned LiveFragments,
    unsigned MaxWindow, bool LastPickWasDSRead, bool HasPickedMFMA,
    unsigned MFMAsSinceLastDSRead,
    const DenseMap<const SUnit *, unsigned> &RecentDSReadUnrelatedMFMAs) {
  return SU && isDSReadLike(SU) && HasPickedMFMA && !LastPickWasDSRead &&
         MFMAsSinceLastDSRead >= MFMAFragmentSched.PipelineGroupSize &&
         LiveFragments < MaxWindow && countMFMASuccessorsUnlockedBy(SU) != 0 &&
         hasReadyMFMAPending(Zone);
}

static bool tryReadyStall(GenericSchedulerBase::SchedCandidate &TryCand,
                          GenericSchedulerBase::SchedCandidate &Cand,
                          SchedBoundary &Zone) {
  if (!TryCand.SU || !Cand.SU)
    return false;

  unsigned TryStall = getReadyStall(Zone, TryCand.SU);
  unsigned CandStall = getReadyStall(Zone, Cand.SU);
  if (TryStall == CandStall)
    return false;

  if (tryLess(TryStall, CandStall, TryCand, Cand, GenericSchedulerBase::Stall))
    return true;

  return false;
}

static unsigned getRecentDSReadSpacing(
    PipeKind Kind, const SUnit *SU,
    const DenseMap<const SUnit *, unsigned> &RecentDSReadUnrelatedMFMAs) {
  unsigned MinSpacing = MFMAFragmentSched.PipelineGroupSize;
  unsigned BestSpacing = std::numeric_limits<unsigned>::max();
  if (Kind != PipeKind::ReadyMFMA && Kind != PipeKind::PendingMFMA)
    return MinSpacing;
  if (!MinSpacing)
    return MinSpacing;

  for (const auto &DSRead : RecentDSReadUnrelatedMFMAs)
    if (consumesThisDSRead(SU, DSRead.first))
      BestSpacing = std::min(BestSpacing, DSRead.second);
  if (BestSpacing == std::numeric_limits<unsigned>::max())
    return MinSpacing;
  return BestSpacing;
}

static bool hasRecentDSReadSpacingDebt(
    const DenseMap<const SUnit *, unsigned> &RecentDSReadUnrelatedMFMAs) {
  if (!MFMAFragmentSched.PipelineGroupSize)
    return false;

  for (const auto &DSRead : RecentDSReadUnrelatedMFMAs)
    if (DSRead.second < MFMAFragmentSched.PipelineGroupSize)
      return true;

  return false;
}

static bool consumesImmatureRecentDSRead(
    const SUnit *SU,
    const DenseMap<const SUnit *, unsigned> &RecentDSReadUnrelatedMFMAs) {
  if (!isMFMALike(SU) || !MFMAFragmentSched.PipelineGroupSize)
    return false;

  for (const auto &DSRead : RecentDSReadUnrelatedMFMAs)
    if (DSRead.second < MFMAFragmentSched.PipelineGroupSize &&
        consumesThisDSRead(SU, DSRead.first))
      return true;

  return false;
}

static bool isRecentDSReadSpacingAlternative(
    SchedBoundary &Zone, SUnit *SU,
    const DenseMap<const SUnit *, unsigned> &RecentDSReadUnrelatedMFMAs) {
  if (!SU || !MFMAFragmentSched.PipelineGroupSize ||
      RecentDSReadUnrelatedMFMAs.empty())
    return false;

  PipeKind Kind = classifyPipeKind(Zone, SU);
  if (Kind == PipeKind::FragmentProducer)
    return getEffectiveStall(Zone, SU) == 0 &&
           getDirectFragmentProducerScore(SU).Valid;

  if (Kind != PipeKind::ReadyMFMA && Kind != PipeKind::PendingMFMA)
    return false;

  if (consumesImmatureRecentDSRead(SU, RecentDSReadUnrelatedMFMAs))
    return false;

  if (getIssueStall(Zone, SU) != 0)
    return false;

  if (Kind == PipeKind::ReadyMFMA)
    return true;

  return getEffectiveStall(Zone, SU) <= MFMAFragmentSched.MaxDrainMFMAStall;
}

static bool shouldPreferBoundaryForRecentDSReadSpacing(
    SchedBoundary &PreferredZone,
    GenericSchedulerBase::SchedCandidate &PreferredCand,
    GenericSchedulerBase::SchedCandidate &OtherCand,
    const DenseMap<const SUnit *, unsigned> &RecentDSReadUnrelatedMFMAs) {
  if (!PreferredCand.isValid() || !OtherCand.isValid())
    return false;

  if (!consumesImmatureRecentDSRead(OtherCand.SU, RecentDSReadUnrelatedMFMAs))
    return false;

  return isRecentDSReadSpacingAlternative(PreferredZone, PreferredCand.SU,
                                          RecentDSReadUnrelatedMFMAs);
}

static bool isPullForwardUnlockingProducer(
    SchedBoundary &Zone, SUnit *SU, unsigned LiveFragments,
    unsigned UsefulWindow, bool LastPickWasDSRead, bool HasPickedMFMA,
    bool AllowFullWindow,
    const DenseMap<const SUnit *, unsigned> &RecentDSReadUnrelatedMFMAs) {
  if (!MFMAFragmentSched.PipelineGroupSize)
    return false;

  if (!HasPickedMFMA || LastPickWasDSRead || !isDSReadLike(SU))
    return false;

  if (AllowFullWindow ? LiveFragments > UsefulWindow
                      : LiveFragments >= UsefulWindow)
    return false;

  if (getEffectiveStall(Zone, SU) != 0)
    return false;

  if (!getDirectFragmentProducerScore(SU).Valid)
    return false;

  if (countMFMASuccessorsUnlockedBy(SU) < MFMAFragmentSched.PipelineGroupSize)
    return false;

  return countDSReadSpacingFillers(Zone, SU) >=
         MFMAFragmentSched.PipelineGroupSize;
}

static bool isBetterPullForwardProducer(SchedBoundary &Zone, SUnit *Try,
                                        SUnit *Cand) {
  unsigned TryUnlocks = countMFMASuccessorsUnlockedBy(Try);
  unsigned CandUnlocks = Cand ? countMFMASuccessorsUnlockedBy(Cand) : 0;
  if (TryUnlocks != CandUnlocks)
    return TryUnlocks > CandUnlocks;

  unsigned TryFillers = countDSReadSpacingFillers(Zone, Try);
  unsigned CandFillers = Cand ? countDSReadSpacingFillers(Zone, Cand) : 0;
  if (TryFillers != CandFillers)
    return TryFillers > CandFillers;

  MFMAFragmentProducerScore TryScore = getReachableFragmentProducerScore(Try);
  MFMAFragmentProducerScore CandScore = getReachableFragmentProducerScore(Cand);
  if (isBetterPrologueFragmentScore(TryScore, CandScore))
    return true;
  if (isBetterPrologueFragmentScore(CandScore, TryScore))
    return false;

  return !Cand || Try->NodeNum < Cand->NodeNum;
}

static SUnit *findPullForwardUnlockingProducer(
    SchedBoundary &Zone, GenericSchedulerBase::SchedCandidate &Cand,
    unsigned LiveFragments, unsigned UsefulWindow, bool LastPickWasDSRead,
    bool HasPickedMFMA,
    const DenseMap<const SUnit *, unsigned> &RecentDSReadUnrelatedMFMAs) {
  if (!Cand.isValid())
    return nullptr;

  PipeKind CandK = classifyPipeKind(Zone, Cand.SU);
  if (CandK != PipeKind::None && CandK != PipeKind::FragmentProducer &&
      CandK != PipeKind::ReadyMFMA)
    return nullptr;

  bool AllowFullWindow = CandK == PipeKind::ReadyMFMA;
  if (isPullForwardUnlockingProducer(
          Zone, Cand.SU, LiveFragments, UsefulWindow, LastPickWasDSRead,
          HasPickedMFMA, AllowFullWindow, RecentDSReadUnrelatedMFMAs))
    return nullptr;

  SUnit *Best = nullptr;
  for (SUnit *SU : Zone.Available) {
    bool IsPullForwardProducer = isPullForwardUnlockingProducer(
        Zone, SU, LiveFragments, UsefulWindow, LastPickWasDSRead, HasPickedMFMA,
        AllowFullWindow, RecentDSReadUnrelatedMFMAs);
    if (!IsPullForwardProducer)
      continue;

    if (!Best || isBetterPullForwardProducer(Zone, SU, Best))
      Best = SU;
  }

  return Best;
}

static bool tryRecentDSReadSpacing(
    GenericSchedulerBase::SchedCandidate &TryCand,
    GenericSchedulerBase::SchedCandidate &Cand, SchedBoundary &Zone,
    bool HasPickedMFMA,
    const DenseMap<const SUnit *, unsigned> &RecentDSReadUnrelatedMFMAs) {
  if (!TryCand.SU || !Cand.SU || !HasPickedMFMA ||
      !MFMAFragmentSched.PipelineGroupSize ||
      RecentDSReadUnrelatedMFMAs.empty())
    return false;

  PipeKind TryK = classifyPipeKind(Zone, TryCand.SU);
  PipeKind CandK = classifyPipeKind(Zone, Cand.SU);
  if ((TryK != PipeKind::ReadyMFMA && TryK != PipeKind::PendingMFMA) ||
      (CandK != PipeKind::ReadyMFMA && CandK != PipeKind::PendingMFMA))
    return false;

  unsigned TrySpacing =
      getRecentDSReadSpacing(TryK, TryCand.SU, RecentDSReadUnrelatedMFMAs);
  unsigned CandSpacing =
      getRecentDSReadSpacing(CandK, Cand.SU, RecentDSReadUnrelatedMFMAs);
  if (TrySpacing == CandSpacing)
    return false;

  if (tryGreater(TrySpacing, CandSpacing, TryCand, Cand,
                 GenericSchedulerBase::Stall))
    return true;

  return false;
}

bool GCNSchedStrategy::tryMFMAFragmentOpener(
    GenericSchedulerBase::SchedCandidate &TryCand,
    GenericSchedulerBase::SchedCandidate &Cand, SchedBoundary &Zone,
    const FragmentWindowState &FWS, bool EnablePrologueSetupBias) const {
  PipeKind TryK = classifyPipeKind(Zone, TryCand.SU);
  PipeKind CandK = classifyPipeKind(Zone, Cand.SU);
  bool TryProducer = TryK == PipeKind::FragmentProducer;
  bool CandProducer = CandK == PipeKind::FragmentProducer;
  unsigned EffectiveUsefulWindow =
      FWS.HasPickedMFMA
          ? FWS.UsefulWindow
          : std::min(FWS.UsefulWindow, MFMAFragmentSched.PrologueWindow);

  // Opener: build the four-fragment window and use independent address setup
  // to cover otherwise exposed DS_READ latency.
  // Once the four-fragment opener is full, finish useful address setup for
  // the next fragment microcluster before selecting the first MFMA.  The
  // setup does not grow the live-fragment window and fills cycles that would
  // otherwise be exposed LDS latency.
  if (!FWS.HasPickedMFMA && FWS.LiveFragments >= EffectiveUsefulWindow) {
    bool TryMFMA = TryK == PipeKind::ReadyMFMA || TryK == PipeKind::PendingMFMA;
    bool CandMFMA =
        CandK == PipeKind::ReadyMFMA || CandK == PipeKind::PendingMFMA;
    bool TrySetup = TryK == PipeKind::None &&
                    getReachableFragmentProducerScore(TryCand.SU).Valid;
    bool CandSetup = CandK == PipeKind::None &&
                     getReachableFragmentProducerScore(Cand.SU).Valid;
    if ((TrySetup && CandMFMA) || (CandSetup && TryMFMA)) {
      if (TrySetup) {
        TryCand.Reason = GenericSchedulerBase::TopPathReduce;
        return true;
      }

      Cand.Reason = GenericSchedulerBase::TopPathReduce;
      return true;
    }
  }

  if (!FWS.HasPickedMFMA && FWS.LiveFragments >= EffectiveUsefulWindow &&
      (TryProducer || CandProducer) && TryK != CandK) {
    // PrologueWindow is a hard cap. Even a producer that immediately unlocks
    // another MFMA belongs to the post-MFMA1 microcluster once the two direct
    // and two prefetched opener fragments have been issued.
    if (!TryProducer) {
      TryCand.Reason = GenericSchedulerBase::TopPathReduce;
      return true;
    }

    Cand.Reason = GenericSchedulerBase::TopPathReduce;
    return true;
  }

  if (EnablePrologueSetupBias && !FWS.HasPickedMFMA && FWS.LiveFragments == 0 &&
      TryProducer != CandProducer) {
    SUnit *ProducerSU = TryProducer ? TryCand.SU : Cand.SU;
    SUnit *SetupSU = TryProducer ? Cand.SU : TryCand.SU;
    MFMAFragmentProducerScore ProducerScore =
        getReachableFragmentProducerScore(ProducerSU);
    MFMAFragmentProducerScore SetupScore =
        getReachableFragmentProducerScore(SetupSU);

    if (SetupScore.Valid &&
        !isBetterPrologueFragmentScore(ProducerScore, SetupScore)) {
      if (!TryProducer) {
        TryCand.Reason = GenericSchedulerBase::TopPathReduce;
        return true;
      }

      Cand.Reason = GenericSchedulerBase::TopPathReduce;
      return true;
    }
  }

  if (!FWS.HasPickedMFMA) {
    MFMAFragmentProducerScore TryScore =
        getReachableFragmentProducerScore(TryCand.SU);
    MFMAFragmentProducerScore CandScore =
        getReachableFragmentProducerScore(Cand.SU);

    if (isBetterPrologueFragmentScore(TryScore, CandScore)) {
      TryCand.Reason = GenericSchedulerBase::TopPathReduce;
      return true;
    }

    if (isBetterPrologueFragmentScore(CandScore, TryScore)) {
      Cand.Reason = GenericSchedulerBase::TopPathReduce;
      return true;
    }

    if (EnablePrologueSetupBias && TryK == PipeKind::None &&
        CandK == PipeKind::None && TryScore.Valid && CandScore.Valid &&
        TryCand.SU->getHeight() != Cand.SU->getHeight()) {
      if (tryLess(TryCand.SU->getHeight(), Cand.SU->getHeight(), TryCand, Cand,
                  GenericSchedulerBase::TopPathReduce))
        return true;

      return false;
    }
  }

  if (TryK == PipeKind::FragmentProducer &&
      CandK == PipeKind::FragmentProducer) {
    MFMAFragmentProducerScore TryScore =
        getReachableFragmentProducerScore(TryCand.SU);
    MFMAFragmentProducerScore CandScore =
        getReachableFragmentProducerScore(Cand.SU);

    if (isBetterPrologueFragmentScore(TryScore, CandScore)) {
      TryCand.Reason = GenericSchedulerBase::TopPathReduce;
      return true;
    }

    if (isBetterPrologueFragmentScore(CandScore, TryScore)) {
      Cand.Reason = GenericSchedulerBase::TopPathReduce;
      return true;
    }
  }

  return false;
}

bool GCNSchedStrategy::tryMFMASteadyState(
    GenericSchedulerBase::SchedCandidate &TryCand,
    GenericSchedulerBase::SchedCandidate &Cand, SchedBoundary &Zone,
    const FragmentWindowState &FWS, unsigned EffectiveMaxWindow) const {
  PipeKind TryK = classifyPipeKind(Zone, TryCand.SU);
  PipeKind CandK = classifyPipeKind(Zone, Cand.SU);
  auto ShouldStaggerPendingMFMA = [&](PipeKind Kind, const SUnit *SU) {
    return Kind == PipeKind::PendingMFMA && FWS.LastPickWasDSRead &&
           FWS.HasPickedMFMA && consumesThisDSRead(SU, FWS.LastDSRead);
  };

  bool TryStagger = ShouldStaggerPendingMFMA(TryK, TryCand.SU);
  bool CandStagger = ShouldStaggerPendingMFMA(CandK, Cand.SU);
  bool TryProducer = TryK == PipeKind::FragmentProducer;
  bool CandProducer = CandK == PipeKind::FragmentProducer;
  auto IsIssueReadyMFMA = [&](PipeKind Kind, SUnit *SU) {
    return Kind == PipeKind::ReadyMFMA && getIssueStall(Zone, SU) == 0;
  };
  bool TryFiller = IsIssueReadyMFMA(TryK, TryCand.SU);
  bool CandFiller = IsIssueReadyMFMA(CandK, Cand.SU);
  auto IsDrainBurstMFMA = [&](PipeKind Kind, SUnit *SU) {
    if (Kind == PipeKind::ReadyMFMA)
      return getIssueStall(Zone, SU) == 0;
    if (Kind != PipeKind::PendingMFMA || getIssueStall(Zone, SU) != 0)
      return false;
    return getEffectiveStall(Zone, SU) <= MFMAFragmentSched.MaxDrainMFMAStall;
  };
  bool TryDrainBurstMFMA = IsDrainBurstMFMA(TryK, TryCand.SU);
  bool CandDrainBurstMFMA = IsDrainBurstMFMA(CandK, Cand.SU);
  auto ConsumesImmatureRecentDSRead = [&](PipeKind Kind, const SUnit *SU) {
    return (Kind == PipeKind::ReadyMFMA || Kind == PipeKind::PendingMFMA) &&
           consumesImmatureRecentDSRead(SU, FWS.RecentDSReadUnrelatedMFMAs);
  };
  bool TryConsumesImmatureDSRead =
      ConsumesImmatureRecentDSRead(TryK, TryCand.SU);
  bool CandConsumesImmatureDSRead =
      ConsumesImmatureRecentDSRead(CandK, Cand.SU);
  unsigned TryRecentDSReadSpacing =
      getRecentDSReadSpacing(TryK, TryCand.SU, FWS.RecentDSReadUnrelatedMFMAs);
  unsigned CandRecentDSReadSpacing =
      getRecentDSReadSpacing(CandK, Cand.SU, FWS.RecentDSReadUnrelatedMFMAs);
  bool TryUnrelatedDrainMFMA = TryDrainBurstMFMA && !TryConsumesImmatureDSRead;
  bool CandUnrelatedDrainMFMA =
      CandDrainBurstMFMA && !CandConsumesImmatureDSRead;
  auto IsUsefulSpacingProducer = [&](PipeKind Kind, SUnit *SU) {
    return Kind == PipeKind::FragmentProducer && FWS.HasPickedMFMA &&
           MFMAFragmentSched.PipelineGroupSize &&
           getEffectiveStall(Zone, SU) == 0 &&
           FWS.LiveFragments < FWS.MaxWindow &&
           getDirectFragmentProducerScore(SU).Valid;
  };
  bool TryUsefulSpacingProducer = IsUsefulSpacingProducer(TryK, TryCand.SU);
  bool CandUsefulSpacingProducer = IsUsefulSpacingProducer(CandK, Cand.SU);
  auto IsUnderfilledSpacingProducer = [&](PipeKind Kind, SUnit *SU) {
    return Kind == PipeKind::FragmentProducer && FWS.HasPickedMFMA &&
           MFMAFragmentSched.PipelineGroupSize && !FWS.LastPickWasDSRead &&
           getEffectiveStall(Zone, SU) == 0 &&
           getDirectFragmentProducerScore(SU).Valid &&
           countDSReadSpacingFillers(Zone, SU) <
               MFMAFragmentSched.PipelineGroupSize;
  };
  bool TryUnderfilledSpacingProducer =
      IsUnderfilledSpacingProducer(TryK, TryCand.SU);
  bool CandUnderfilledSpacingProducer =
      IsUnderfilledSpacingProducer(CandK, Cand.SU);
  auto IsHideableUnlockingProducer = [&](PipeKind Kind, const SUnit *SU,
                                         const SUnit *Filler) {
    return Kind == PipeKind::FragmentProducer && FWS.HasPickedMFMA &&
           !FWS.LastPickWasDSRead &&
           FWS.MFMAsSinceLastDSRead >= MFMAFragmentSched.PipelineGroupSize &&
           FWS.LiveFragments < EffectiveMaxWindow &&
           (countMFMASuccessorsUnlockedBy(SU) != 0 ||
            countMFMASuccessorsUnlockedByAfterFiller(SU, Filler) != 0);
  };
  bool TryHideableProducer =
      IsHideableUnlockingProducer(TryK, TryCand.SU, Cand.SU);
  bool CandHideableProducer =
      IsHideableUnlockingProducer(CandK, Cand.SU, TryCand.SU);

  // Steady state: mature each new DS_READ with unrelated MFMAs, and prefer
  // reads that can start the next useful two-fragment microcluster.
  if (FWS.HasPickedMFMA && MFMAFragmentSched.PipelineGroupSize &&
      !FWS.RecentDSReadUnrelatedMFMAs.empty()) {
    if (TryDrainBurstMFMA && CandDrainBurstMFMA &&
        TryRecentDSReadSpacing != CandRecentDSReadSpacing) {
      if (tryGreater(TryRecentDSReadSpacing, CandRecentDSReadSpacing, TryCand,
                     Cand, GenericSchedulerBase::Stall))
        return true;

      return false;
    }

    if (TryConsumesImmatureDSRead && CandUnrelatedDrainMFMA) {
      Cand.Reason = GenericSchedulerBase::Stall;
      return true;
    }

    if (CandConsumesImmatureDSRead && TryUnrelatedDrainMFMA) {
      TryCand.Reason = GenericSchedulerBase::Stall;
      return true;
    }

    if (TryConsumesImmatureDSRead && CandUsefulSpacingProducer) {
      Cand.Reason = GenericSchedulerBase::Stall;
      return true;
    }

    if (CandConsumesImmatureDSRead && TryUsefulSpacingProducer) {
      TryCand.Reason = GenericSchedulerBase::Stall;
      return true;
    }
  }

  if (TryStagger && CandFiller) {
    Cand.Reason = GenericSchedulerBase::Stall;
    return true;
  }

  if (CandStagger && TryFiller) {
    TryCand.Reason = GenericSchedulerBase::Stall;
    return true;
  }

  if (TryStagger && CandProducer && getEffectiveStall(Zone, Cand.SU) == 0) {
    Cand.Reason = GenericSchedulerBase::Stall;
    return true;
  }

  if (CandStagger && TryProducer && getEffectiveStall(Zone, TryCand.SU) == 0) {
    TryCand.Reason = GenericSchedulerBase::Stall;
    return true;
  }

  if (TryUnderfilledSpacingProducer && CandUnrelatedDrainMFMA) {
    Cand.Reason = GenericSchedulerBase::Stall;
    return true;
  }

  if (CandUnderfilledSpacingProducer && TryUnrelatedDrainMFMA) {
    TryCand.Reason = GenericSchedulerBase::Stall;
    return true;
  }

  if (TryHideableProducer && CandFiller) {
    TryCand.Reason = GenericSchedulerBase::TopPathReduce;
    return true;
  }

  if (CandHideableProducer && TryFiller) {
    Cand.Reason = GenericSchedulerBase::TopPathReduce;
    return true;
  }

  // Finish a two-read DS microcluster before draining unrelated MFMAs to avoid
  // the gfx950 lone-DS issue-cycle penalty.
  bool TryDrainMFMA = TryDrainBurstMFMA;
  bool CandDrainMFMA = CandDrainBurstMFMA;
  if (FWS.HasPickedMFMA && FWS.LastPickWasDSRead &&
      FWS.LiveFragments < EffectiveMaxWindow &&
      ((TryProducer && CandDrainMFMA) || (CandProducer && TryDrainMFMA))) {
    bool NeedsSecondDS =
        FWS.DSReadsSinceLastMFMA < MFMAFragmentSched.PipelineGroupSize;
    if (NeedsSecondDS) {
      if (TryProducer) {
        TryCand.Reason = GenericSchedulerBase::TopPathReduce;
        return true;
      }

      Cand.Reason = GenericSchedulerBase::TopPathReduce;
      return true;
    }
  }

  if (FWS.HasPickedMFMA &&
      FWS.MFMAsSinceLastDSRead < MFMAFragmentSched.PipelineGroupSize &&
      ((TryDrainBurstMFMA && CandProducer) ||
       (CandDrainBurstMFMA && TryProducer))) {
    if (TryDrainBurstMFMA) {
      TryCand.Reason = GenericSchedulerBase::TopPathReduce;
      return true;
    }

    Cand.Reason = GenericSchedulerBase::TopPathReduce;
    return true;
  }

  return false;
}

bool GCNSchedStrategy::tryMFMAFragmentCandidate(
    GenericSchedulerBase::SchedCandidate &TryCand,
    GenericSchedulerBase::SchedCandidate &Cand, SchedBoundary &Zone,
    const FragmentWindowState &FWS, bool EnablePrologueSetupBias) const {
  // Follow the generic try* protocol: return true when the policy chooses
  // either candidate and set the reason on the preferred candidate. Return
  // false only when the policy has no preference.
  if (!TryCand.SU || !Cand.SU)
    return false;

  unsigned EffectiveUsefulWindow =
      FWS.HasPickedMFMA
          ? FWS.UsefulWindow
          : std::min(FWS.UsefulWindow, MFMAFragmentSched.PrologueWindow);
  bool IsSteadyState =
      FWS.HasPickedMFMA || FWS.LiveFragments >= EffectiveUsefulWindow;
  unsigned EffectiveMaxWindow =
      IsSteadyState ? EffectiveUsefulWindow : FWS.MaxWindow;

  if (tryMFMAFragmentOpener(TryCand, Cand, Zone, FWS, EnablePrologueSetupBias))
    return true;
  if (tryMFMASteadyState(TryCand, Cand, Zone, FWS, EffectiveMaxWindow))
    return true;
  if (tryMFMAResourceStall(TryCand, Cand, Zone, FWS, EffectiveMaxWindow))
    return true;
  if (tryMFMAProducerOrder(TryCand, Cand, Zone))
    return true;
  return tryMFMAWindowFallback(TryCand, Cand, Zone, FWS, EffectiveUsefulWindow,
                               EffectiveMaxWindow);
}

bool GCNSchedStrategy::tryMFMAProducerOrder(
    GenericSchedulerBase::SchedCandidate &TryCand,
    GenericSchedulerBase::SchedCandidate &Cand, SchedBoundary &Zone) const {
  if (classifyPipeKind(Zone, TryCand.SU) != PipeKind::FragmentProducer ||
      classifyPipeKind(Zone, Cand.SU) != PipeKind::FragmentProducer)
    return false;

  // Prefer the producer closest to making an MFMA ready. The successive
  // tie-breakers make this ordering deterministic without changing the DAG.
  unsigned TryBestMissing =
      getBestMFMASuccessorMissingDSReadPredsAfter(TryCand.SU);
  unsigned CandBestMissing =
      getBestMFMASuccessorMissingDSReadPredsAfter(Cand.SU);
  if (TryBestMissing != CandBestMissing)
    return tryLess(TryBestMissing, CandBestMissing, TryCand, Cand,
                   GenericSchedulerBase::TopPathReduce);

  unsigned TrySuccNode = getBestMFMASuccessorNodeNumAfter(TryCand.SU);
  unsigned CandSuccNode = getBestMFMASuccessorNodeNumAfter(Cand.SU);
  if (TrySuccNode != CandSuccNode)
    return tryLess(TrySuccNode, CandSuccNode, TryCand, Cand,
                   GenericSchedulerBase::TopPathReduce);

  unsigned TryScheduled =
      getBestMFMASuccessorScheduledDSReadPredsAfter(TryCand.SU);
  unsigned CandScheduled =
      getBestMFMASuccessorScheduledDSReadPredsAfter(Cand.SU);
  if (TryScheduled != CandScheduled)
    return tryGreater(TryScheduled, CandScheduled, TryCand, Cand,
                      GenericSchedulerBase::TopPathReduce);

  unsigned TryUnlocks = countMFMASuccessorsUnlockedBy(TryCand.SU);
  unsigned CandUnlocks = countMFMASuccessorsUnlockedBy(Cand.SU);
  if (TryUnlocks != CandUnlocks)
    return tryGreater(TryUnlocks, CandUnlocks, TryCand, Cand,
                      GenericSchedulerBase::TopPathReduce);

  unsigned TryRemaining = getNumDSReadsToUnlockClosestMFMA(TryCand.SU);
  unsigned CandRemaining = getNumDSReadsToUnlockClosestMFMA(Cand.SU);
  if (TryRemaining != CandRemaining)
    return tryLess(TryRemaining, CandRemaining, TryCand, Cand,
                   GenericSchedulerBase::TopPathReduce);

  return false;
}

bool GCNSchedStrategy::tryMFMAResourceStall(
    GenericSchedulerBase::SchedCandidate &TryCand,
    GenericSchedulerBase::SchedCandidate &Cand, SchedBoundary &Zone,
    const FragmentWindowState &FWS, unsigned EffectiveMaxWindow) const {
  PipeKind TryK = classifyPipeKind(Zone, TryCand.SU);
  PipeKind CandK = classifyPipeKind(Zone, Cand.SU);
  bool TryProducer = TryK == PipeKind::FragmentProducer;
  bool CandProducer = CandK == PipeKind::FragmentProducer;
  bool TryIssueStalledMFMA =
      TryK == PipeKind::ReadyMFMA && getIssueStall(Zone, TryCand.SU) != 0;
  bool CandIssueStalledMFMA =
      CandK == PipeKind::ReadyMFMA && getIssueStall(Zone, Cand.SU) != 0;

  // Fill a short MFMA issue gap with a useful read when possible, but drain
  // the window rather than admitting unbounded fragments.
  if ((TryIssueStalledMFMA && CandProducer) ||
      (CandIssueStalledMFMA && TryProducer)) {
    bool TryIsProducer = TryProducer;
    SUnit *ProducerSU = TryIsProducer ? TryCand.SU : Cand.SU;
    SUnit *MFMA = TryIssueStalledMFMA ? TryCand.SU : Cand.SU;
    bool HasWindowHeadroom =
        !FWS.HasPickedMFMA || FWS.LiveFragments < EffectiveMaxWindow;
    bool CanFillIssueGap =
        getEffectiveStall(Zone, ProducerSU) <= getIssueStall(Zone, MFMA) &&
        HasWindowHeadroom &&
        (!FWS.HasPickedMFMA ||
         FWS.DSReadsSinceLastMFMA < MFMAFragmentSched.MaxDSReadsInMFMAStall);

    if (CanFillIssueGap) {
      (TryIsProducer ? TryCand : Cand).Reason = GenericSchedulerBase::Stall;
      return true;
    }

    (TryIssueStalledMFMA ? TryCand : Cand).Reason =
        GenericSchedulerBase::RegExcess;
    return true;
  }

  if (TryIssueStalledMFMA && CandIssueStalledMFMA) {
    unsigned TryIssueStall = getIssueStall(Zone, TryCand.SU);
    unsigned CandIssueStall = getIssueStall(Zone, Cand.SU);
    if (TryIssueStall != CandIssueStall)
      return tryLess(TryIssueStall, CandIssueStall, TryCand, Cand,
                     GenericSchedulerBase::Stall);
  }

  if (FWS.LiveFragments < EffectiveMaxWindow)
    return false;

  bool TryMFMA = TryK == PipeKind::ReadyMFMA;
  bool CandMFMA = CandK == PipeKind::ReadyMFMA;
  if ((TryMFMA && CandProducer) || (CandMFMA && TryProducer)) {
    (TryMFMA ? TryCand : Cand).Reason = GenericSchedulerBase::RegExcess;
    return true;
  }

  if (TryMFMA && CandMFMA)
    return tryReadyStall(TryCand, Cand, Zone);

  return false;
}

bool GCNSchedStrategy::tryMFMAWindowFallback(
    GenericSchedulerBase::SchedCandidate &TryCand,
    GenericSchedulerBase::SchedCandidate &Cand, SchedBoundary &Zone,
    const FragmentWindowState &FWS, unsigned EffectiveUsefulWindow,
    unsigned EffectiveMaxWindow) const {
  PipeKind TryK = classifyPipeKind(Zone, TryCand.SU);
  PipeKind CandK = classifyPipeKind(Zone, Cand.SU);

  if (TryK == PipeKind::PendingMFMA || CandK == PipeKind::PendingMFMA) {
    if (TryK != CandK) {
      (TryK != PipeKind::PendingMFMA ? TryCand : Cand).Reason =
          GenericSchedulerBase::Stall;
      return true;
    }
    return tryReadyStall(TryCand, Cand, Zone);
  }

  bool TryProducer = TryK == PipeKind::FragmentProducer;
  bool CandProducer = CandK == PipeKind::FragmentProducer;
  if (FWS.LiveFragments < EffectiveUsefulWindow &&
      TryProducer != CandProducer) {
    (TryProducer ? TryCand : Cand).Reason = GenericSchedulerBase::TopPathReduce;
    return true;
  }

  bool TryReadyMFMA =
      TryK == PipeKind::ReadyMFMA && getIssueStall(Zone, TryCand.SU) == 0;
  bool CandReadyMFMA =
      CandK == PipeKind::ReadyMFMA && getIssueStall(Zone, Cand.SU) == 0;
  if (FWS.LiveFragments >= EffectiveUsefulWindow &&
      TryReadyMFMA != CandReadyMFMA) {
    (TryReadyMFMA ? TryCand : Cand).Reason =
        GenericSchedulerBase::TopPathReduce;
    return true;
  }

  if (FWS.LiveFragments >= EffectiveMaxWindow && TryProducer != CandProducer) {
    (TryProducer ? Cand : TryCand).Reason = GenericSchedulerBase::RegExcess;
    return true;
  }

  return false;
}

bool GCNSchedStrategy::shouldKeepMFMAFragmentCandidate(
    GenericSchedulerBase::SchedCandidate &TryCand,
    GenericSchedulerBase::SchedCandidate &Cand, SchedBoundary &Zone,
    const FragmentWindowState &FWS) const {
  // tryCandidate() compares TryCand against an incumbent Cand. Some generic
  // heuristics run before the fragment policy, so protect an already selected
  // pipeline candidate from being replaced by a locally attractive choice
  // that would break DS maturity spacing or producer ordering.
  if (!TryCand.SU || !Cand.SU)
    return false;

  PipeKind TryK = classifyPipeKind(Zone, TryCand.SU);
  PipeKind CandK = classifyPipeKind(Zone, Cand.SU);

  bool TryIsStaggeredPendingMFMA =
      TryK == PipeKind::PendingMFMA && FWS.LastPickWasDSRead &&
      FWS.HasPickedMFMA && consumesThisDSRead(TryCand.SU, FWS.LastDSRead);
  bool CandIsReadyMFMA =
      CandK == PipeKind::ReadyMFMA && getIssueStall(Zone, Cand.SU) == 0;
  if (TryIsStaggeredPendingMFMA && CandIsReadyMFMA) {
    Cand.Reason = GenericSchedulerBase::Stall;
    return true;
  }

  auto IsDrainMFMA = [&](PipeKind Kind, SUnit *SU) {
    if (Kind == PipeKind::ReadyMFMA)
      return getIssueStall(Zone, SU) == 0;
    if (Kind != PipeKind::PendingMFMA)
      return false;
    if (getIssueStall(Zone, SU) != 0)
      return false;
    return getEffectiveStall(Zone, SU) <= MFMAFragmentSched.MaxDrainMFMAStall;
  };

  auto ConsumesImmatureRecentDSRead = [&](PipeKind Kind, const SUnit *SU) {
    if (Kind != PipeKind::ReadyMFMA && Kind != PipeKind::PendingMFMA)
      return false;
    return consumesImmatureRecentDSRead(SU, FWS.RecentDSReadUnrelatedMFMAs);
  };
  bool TryConsumesImmatureDSRead =
      ConsumesImmatureRecentDSRead(TryK, TryCand.SU);
  bool CandUnrelatedDrainMFMA = IsDrainMFMA(CandK, Cand.SU) &&
                                !ConsumesImmatureRecentDSRead(CandK, Cand.SU);
  bool CandUsefulSpacingProducer =
      CandK == PipeKind::FragmentProducer && FWS.HasPickedMFMA &&
      MFMAFragmentSched.PipelineGroupSize &&
      getEffectiveStall(Zone, Cand.SU) == 0 &&
      FWS.LiveFragments < FWS.MaxWindow &&
      getDirectFragmentProducerScore(Cand.SU).Valid;
  if (FWS.HasPickedMFMA && MFMAFragmentSched.PipelineGroupSize &&
      !FWS.RecentDSReadUnrelatedMFMAs.empty() &&
      IsDrainMFMA(TryK, TryCand.SU) && IsDrainMFMA(CandK, Cand.SU) &&
      getRecentDSReadSpacing(CandK, Cand.SU, FWS.RecentDSReadUnrelatedMFMAs) >
          getRecentDSReadSpacing(TryK, TryCand.SU,
                                 FWS.RecentDSReadUnrelatedMFMAs)) {
    Cand.Reason = GenericSchedulerBase::Stall;
    return true;
  }

  if (FWS.HasPickedMFMA && MFMAFragmentSched.PipelineGroupSize &&
      !FWS.RecentDSReadUnrelatedMFMAs.empty() && TryConsumesImmatureDSRead &&
      CandUnrelatedDrainMFMA) {
    Cand.Reason = GenericSchedulerBase::Stall;
    return true;
  }

  if (FWS.HasPickedMFMA && MFMAFragmentSched.PipelineGroupSize &&
      !FWS.RecentDSReadUnrelatedMFMAs.empty() && TryConsumesImmatureDSRead &&
      CandUsefulSpacingProducer) {
    Cand.Reason = GenericSchedulerBase::Stall;
    return true;
  }

  bool TryIsReadyMFMA =
      TryK == PipeKind::ReadyMFMA && getIssueStall(Zone, TryCand.SU) == 0;
  bool CandIsHideableProducer =
      CandK == PipeKind::FragmentProducer && FWS.HasPickedMFMA &&
      !FWS.LastPickWasDSRead &&
      FWS.MFMAsSinceLastDSRead >= MFMAFragmentSched.PipelineGroupSize &&
      FWS.LiveFragments < FWS.MaxWindow &&
      (countMFMASuccessorsUnlockedBy(Cand.SU) != 0 ||
       countMFMASuccessorsUnlockedByAfterFiller(Cand.SU, TryCand.SU) != 0);
  if (TryIsReadyMFMA && CandIsHideableProducer) {
    Cand.Reason = GenericSchedulerBase::TopPathReduce;
    return true;
  }

  if (TryK == PipeKind::FragmentProducer &&
      CandK == PipeKind::FragmentProducer) {
    MFMAFragmentProducerScore TryScore =
        getReachableFragmentProducerScore(TryCand.SU);
    MFMAFragmentProducerScore CandScore =
        getReachableFragmentProducerScore(Cand.SU);
    if (isBetterPrologueFragmentScore(CandScore, TryScore)) {
      Cand.Reason = GenericSchedulerBase::TopPathReduce;
      return true;
    }
  }

  return false;
}

static bool
tryBidirectionalReadyStall(GenericSchedulerBase::SchedCandidate &TryCand,
                           GenericSchedulerBase::SchedCandidate &Cand,
                           SchedBoundary &TryZone, SchedBoundary &CandZone) {
  if (!TryCand.SU || !Cand.SU)
    return false;

  unsigned TryStall = getReadyStall(TryZone, TryCand.SU);
  unsigned CandStall = getReadyStall(CandZone, Cand.SU);
  if (TryStall == CandStall)
    return false;

  if (tryLess(TryStall, CandStall, TryCand, Cand, GenericSchedulerBase::Stall))
    return TryCand.Reason != GenericSchedulerBase::NoCand;

  return false;
}

static bool isTopDownBiasCandidate(SchedBoundary &Zone, SUnit *SU) {
  PipeKind Kind = classifyPipeKind(Zone, SU);
  return Kind != PipeKind::None && Kind != PipeKind::PendingMFMA;
}

static bool shouldPreferTopOverBottomStalledDS(
    SchedBoundary &Top, SchedBoundary &Bot,
    GenericSchedulerBase::SchedCandidate &TopCand,
    GenericSchedulerBase::SchedCandidate &BotCand) {
  if (!TopCand.isValid() || !BotCand.isValid())
    return false;

  if (classifyPipeKind(Bot, BotCand.SU) != PipeKind::FragmentProducer ||
      getReadyStall(Bot, BotCand.SU) == 0)
    return false;

  PipeKind TopKind = classifyPipeKind(Top, TopCand.SU);
  if (TopKind == PipeKind::FragmentProducer)
    return getReadyStall(Top, TopCand.SU) == 0;

  if (TopKind != PipeKind::ReadyMFMA)
    return false;

  return getReadyStall(Top, TopCand.SU) == 0;
}

static SUnit *findTopMFMAFillerForRecentDS(
    SchedBoundary &Top, SUnit *TopSU, bool HasPickedMFMA,
    const DenseMap<const SUnit *, unsigned> &RecentDSReadUnrelatedMFMAs,
    bool &FillerPending) {
  if (!TopSU || !HasPickedMFMA ||
      classifyPipeKind(Top, TopSU) != PipeKind::FragmentProducer ||
      !hasRecentDSReadSpacingDebt(RecentDSReadUnrelatedMFMAs))
    return nullptr;

  SUnit *Best = nullptr;
  unsigned BestBenefit = 0;
  unsigned BestConsumedSpacing = 0;
  unsigned BestStall = std::numeric_limits<unsigned>::max();
  unsigned BestReadyStall = std::numeric_limits<unsigned>::max();
  bool BestPending = false;
  auto TryFiller = [&](SUnit *SU, bool IsPending) {
    if (!SU || SU->isScheduled || !isMFMALike(SU))
      return;
    if (getIssueStall(Top, SU) != 0)
      return;

    unsigned Benefit = 0;
    unsigned ConsumedSpacing = MFMAFragmentSched.PipelineGroupSize;
    for (const auto &DSRead : RecentDSReadUnrelatedMFMAs) {
      if (DSRead.second >= MFMAFragmentSched.PipelineGroupSize)
        continue;

      if (consumesThisDSRead(SU, DSRead.first)) {
        ConsumedSpacing = std::min(ConsumedSpacing, DSRead.second);
        continue;
      }

      Benefit += MFMAFragmentSched.PipelineGroupSize - DSRead.second;
    }
    if (!Benefit || ConsumedSpacing == 0)
      return;

    unsigned Stall = getEffectiveStall(Top, SU);
    if (Stall > MFMAFragmentSched.MaxDrainMFMAStall)
      return;

    unsigned ReadyStall = getReadyStall(Top, SU);
    bool Better = !Best;
    if (!Better && Benefit != BestBenefit)
      Better = Benefit > BestBenefit;
    if (!Better && Benefit == BestBenefit &&
        ConsumedSpacing != BestConsumedSpacing)
      Better = ConsumedSpacing > BestConsumedSpacing;
    if (!Better && Benefit == BestBenefit &&
        ConsumedSpacing == BestConsumedSpacing && Stall != BestStall)
      Better = Stall < BestStall;
    if (!Better && Benefit == BestBenefit &&
        ConsumedSpacing == BestConsumedSpacing && Stall == BestStall &&
        ReadyStall != BestReadyStall)
      Better = ReadyStall < BestReadyStall;
    if (!Better && Benefit == BestBenefit &&
        ConsumedSpacing == BestConsumedSpacing && Stall == BestStall &&
        ReadyStall == BestReadyStall && IsPending != BestPending)
      Better = IsPending < BestPending;
    if (!Better && Benefit == BestBenefit &&
        ConsumedSpacing == BestConsumedSpacing && Stall == BestStall &&
        ReadyStall == BestReadyStall && IsPending == BestPending)
      Better = SU->NodeNum < Best->NodeNum;

    if (Better) {
      Best = SU;
      BestBenefit = Benefit;
      BestConsumedSpacing = ConsumedSpacing;
      BestStall = Stall;
      BestReadyStall = ReadyStall;
      BestPending = IsPending;
    }
  };

  for (SUnit *SU : Top.Available)
    TryFiller(SU, false);
  for (SUnit *SU : Top.Pending)
    TryFiller(SU, true);

  if (!Best)
    return nullptr;

  FillerPending = BestPending;
  return Best;
}

static SUnit *findTopMFMAFillerForRecentDS(
    SchedBoundary &Top, GenericSchedulerBase::SchedCandidate &TopCand,
    bool HasPickedMFMA,
    const DenseMap<const SUnit *, unsigned> &RecentDSReadUnrelatedMFMAs,
    bool &FillerPending) {
  if (!TopCand.isValid())
    return nullptr;

  return findTopMFMAFillerForRecentDS(Top, TopCand.SU, HasPickedMFMA,
                                      RecentDSReadUnrelatedMFMAs,
                                      FillerPending);
}

static SUnit *findBottomMFMAFillerForStalledDS(
    SchedBoundary &Bot, GenericSchedulerBase::SchedCandidate &BotCand,
    bool HasPickedMFMA,
    DenseMap<const SUnit *, unsigned> &DeferredDSReadFillers,
    bool &FillerPending) {
  if (!BotCand.isValid())
    return nullptr;

  if (classifyPipeKind(Bot, BotCand.SU) != PipeKind::FragmentProducer ||
      !HasPickedMFMA || !hasScheduledMFMASuccessor(BotCand.SU))
    return nullptr;

  unsigned &DeferredFillers = DeferredDSReadFillers[BotCand.SU];
  // Bottom-up sees a fragment's consumer before its producer. Reserve one
  // extra MFMA slot because the closest scheduled MFMA can be that consumer;
  // only the remaining slots are unrelated latency-hiding work in final order.
  if (DeferredFillers > MFMAFragmentSched.PipelineGroupSize)
    return nullptr;

  SUnit *Best = nullptr;
  unsigned BestStall = std::numeric_limits<unsigned>::max();
  unsigned BestReadyStall = std::numeric_limits<unsigned>::max();
  bool BestPending = false;
  auto TryFiller = [&](SUnit *SU, bool IsPending) {
    if (!SU || SU->isScheduled || !isMFMALike(SU))
      return;
    if (consumesThisDSRead(SU, BotCand.SU))
      return;

    unsigned Stall = getEffectiveStall(Bot, SU);
    if (Stall > MFMAFragmentSched.MaxDrainMFMAStall)
      return;

    unsigned ReadyStall = getReadyStall(Bot, SU);
    if (!Best || Stall < BestStall ||
        (Stall == BestStall && ReadyStall < BestReadyStall)) {
      Best = SU;
      BestStall = Stall;
      BestReadyStall = ReadyStall;
      BestPending = IsPending;
    }
  };

  for (SUnit *SU : Bot.Available)
    TryFiller(SU, false);
  for (SUnit *SU : Bot.Pending)
    TryFiller(SU, true);

  if (!Best)
    return nullptr;

  ++DeferredFillers;
  FillerPending = BestPending;
  return Best;
}

void GCNSchedStrategy::updateFragmentWindow(SUnit *SU, bool IsTopNode) {
  FragmentWindowState &FWS = IsTopNode ? TopFragmentWindow : BotFragmentWindow;
  SchedBoundary &Zone = IsTopNode ? Top : Bot;
  if (isDSReadLike(SU)) {
    FWS.DeferredDSReadFillers.erase(SU);
    ++FWS.LiveFragments;
    if (FWS.HasPickedMFMA)
      ++FWS.DSReadsSinceLastMFMA;
    FWS.LastDSRead = SU;
    FWS.MFMAsSinceLastDSRead = 0;
    FWS.UnrelatedMFMAsSinceLastDSRead = 0;
    FWS.RecentDSReadUnrelatedMFMAs[SU] = 0;
    FWS.LastPickWasDSRead = true;
    return;
  }

  FWS.LastPickWasDSRead = false;

  if (isMFMALike(SU)) {
    bool CompletesDSMicrocluster =
        FWS.HasPickedMFMA &&
        FWS.DSReadsSinceLastMFMA >= MFMAFragmentSched.PipelineGroupSize;
    bool IsUsefulFiller =
        getReadyStall(Zone, SU) == 0 && getIssueStall(Zone, SU) == 0;
    FWS.HasPickedMFMA = true;
    ++FWS.PickedMFMAs;
    FWS.DSReadsSinceLastMFMA = 0;
    if (CompletesDSMicrocluster)
      ++FWS.CompletedDSMicroclusters;
    if (IsUsefulFiller)
      ++FWS.MFMAsSinceLastDSRead;
    if (IsUsefulFiller && !consumesThisDSRead(SU, FWS.LastDSRead))
      ++FWS.UnrelatedMFMAsSinceLastDSRead;
    if (MFMAFragmentSched.PipelineGroupSize) {
      SmallVector<const SUnit *, 8> MatureDSReads;
      for (auto &DSRead : FWS.RecentDSReadUnrelatedMFMAs) {
        if (consumesThisDSRead(SU, DSRead.first)) {
          MatureDSReads.push_back(DSRead.first);
          continue;
        }
        if (IsUsefulFiller) {
          ++DSRead.second;
          if (MFMAFragmentSched.PipelineGroupSize &&
              DSRead.second > MFMAFragmentSched.PipelineGroupSize)
            MatureDSReads.push_back(DSRead.first);
        }
      }
      for (const SUnit *DSRead : MatureDSReads)
        FWS.RecentDSReadUnrelatedMFMAs.erase(DSRead);
    }
    if (FWS.LiveFragments)
      --FWS.LiveFragments;
  }
}

void GCNSchedStrategy::resetFragmentWindows() {
  TopFragmentWindow = FragmentWindowState();
  BotFragmentWindow = FragmentWindowState();
}

bool GCNSchedStrategy::tryCandidate(SchedCandidate &Cand,
                                    SchedCandidate &TryCand,
                                    SchedBoundary *Zone) const {
  if (!EnableMFMAFragmentScheduler)
    return GenericScheduler::tryCandidate(Cand, TryCand, Zone);

  if (!Cand.isValid()) {
    TryCand.Reason = FirstValid;
    return true;
  }

  if (tryGreater(biasPhysReg(TryCand.SU, TryCand.AtTop),
                 biasPhysReg(Cand.SU, Cand.AtTop), TryCand, Cand, PhysReg))
    return TryCand.Reason != NoCand;

  if (DAG->isTrackingPressure() &&
      tryPressure(TryCand.RPDelta.Excess, Cand.RPDelta.Excess, TryCand, Cand,
                  RegExcess, TRI, DAG->MF))
    return TryCand.Reason != NoCand;

  if (DAG->isTrackingPressure() &&
      tryPressure(TryCand.RPDelta.CriticalMax, Cand.RPDelta.CriticalMax,
                  TryCand, Cand, RegCritical, TRI, DAG->MF))
    return TryCand.Reason != NoCand;

  bool SameBoundary = Zone != nullptr;
  if (SameBoundary) {
    const FragmentWindowState &FWS =
        Zone->isTop() ? TopFragmentWindow : BotFragmentWindow;
    if (tryRecentDSReadSpacing(TryCand, Cand, *Zone, FWS.HasPickedMFMA,
                               FWS.RecentDSReadUnrelatedMFMAs))
      return TryCand.Reason != NoCand;

    if (shouldKeepMFMAFragmentCandidate(TryCand, Cand, *Zone, FWS))
      return false;

    if (tryMFMAFragmentCandidate(TryCand, Cand, *Zone, FWS, true))
      return TryCand.Reason != NoCand;

    if (Rem.IsAcyclicLatencyLimited && !Zone->getCurrMOps() &&
        tryLatency(TryCand, Cand, *Zone))
      return TryCand.Reason != NoCand;

    if (tryLess(Zone->getLatencyStallCycles(TryCand.SU),
                Zone->getLatencyStallCycles(Cand.SU), TryCand, Cand, Stall))
      return TryCand.Reason != NoCand;
  }

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
    if (tryLess(getWeakLeft(TryCand.SU, TryCand.AtTop),
                getWeakLeft(Cand.SU, Cand.AtTop), TryCand, Cand, Weak))
      return TryCand.Reason != NoCand;
  }

  if (DAG->isTrackingPressure() &&
      tryPressure(TryCand.RPDelta.CurrentMax, Cand.RPDelta.CurrentMax, TryCand,
                  Cand, RegMax, TRI, DAG->MF))
    return TryCand.Reason != NoCand;

  if (SameBoundary) {
    TryCand.initResourceDelta(DAG, SchedModel);
    if (tryLess(TryCand.ResDelta.CritResources, Cand.ResDelta.CritResources,
                TryCand, Cand, ResourceReduce))
      return TryCand.Reason != NoCand;
    if (tryGreater(TryCand.ResDelta.DemandedResources,
                   Cand.ResDelta.DemandedResources, TryCand, Cand,
                   ResourceDemand))
      return TryCand.Reason != NoCand;

    if (!RegionPolicy.DisableLatencyHeuristic && TryCand.Policy.ReduceLatency &&
        !Rem.IsAcyclicLatencyLimited && tryLatency(TryCand, Cand, *Zone))
      return TryCand.Reason != NoCand;

    if ((Zone->isTop() && TryCand.SU->NodeNum < Cand.SU->NodeNum) ||
        (!Zone->isTop() && TryCand.SU->NodeNum > Cand.SU->NodeNum)) {
      TryCand.Reason = NodeOrder;
      return true;
    }
  }

  return false;
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
    if (!useGCNTrackers()) {
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

  if (EnableMFMAFragmentScheduler) {
    const FragmentWindowState &FWS =
        Zone.isTop() ? TopFragmentWindow : BotFragmentWindow;
    if (SUnit *PullForwardProducer = findPullForwardUnlockingProducer(
            Zone, Cand, FWS.LiveFragments, FWS.UsefulWindow,
            FWS.LastPickWasDSRead, FWS.HasPickedMFMA,
            FWS.RecentDSReadUnrelatedMFMAs)) {
      Cand.reset(ZonePolicy);
      Cand.SU = PullForwardProducer;
      Cand.AtTop = Zone.isTop();
      Cand.Reason = TopPathReduce;
      return;
    }

    if (shouldKeepAvailableUnlockingProducer(
            Zone, Cand.SU, FWS.LiveFragments, FWS.MaxWindow,
            FWS.LastPickWasDSRead, FWS.HasPickedMFMA, FWS.MFMAsSinceLastDSRead,
            FWS.RecentDSReadUnrelatedMFMAs)) {
      Cand.Reason = TopPathReduce;
      return;
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
  unsigned TopEffectiveUsefulWindow =
      TopFragmentWindow.HasPickedMFMA
          ? TopFragmentWindow.UsefulWindow
          : std::min(TopFragmentWindow.UsefulWindow,
                     MFMAFragmentSched.PrologueWindow);
  bool HasTopFragmentPipelineCandidate =
      TopCand.isValid() && classifyPipeKind(Top, TopCand.SU) != PipeKind::None;
  bool BiasActiveFragmentWindowTopDown =
      EnableMFMAFragmentScheduler &&
      (!TopFragmentWindow.HasPickedMFMA
           ? TopFragmentWindow.LiveFragments >= TopEffectiveUsefulWindow
           : TopFragmentWindow.LiveFragments >= TopEffectiveUsefulWindow ||
                 HasTopFragmentPipelineCandidate ||
                 !TopFragmentWindow.RecentDSReadUnrelatedMFMAs.empty() ||
                 TopFragmentWindow.MFMAsSinceLastDSRead <
                     MFMAFragmentSched.PipelineGroupSize);
  bool IsFirstFragmentMFMA =
      !TopFragmentWindow.HasPickedMFMA && isMFMALike(TopCand.SU);
  bool IsOverfullTopFragmentProducer =
      TopCand.isValid() &&
      classifyPipeKind(Top, TopCand.SU) == PipeKind::FragmentProducer &&
      TopFragmentWindow.LiveFragments >= TopEffectiveUsefulWindow;
  bool TopCandHasFragmentWork =
      TopCand.isValid() &&
      (classifyPipeKind(Top, TopCand.SU) != PipeKind::None ||
       getReachableFragmentProducerScore(TopCand.SU).Valid);
  bool IsFragmentPrologue =
      EnableMFMAFragmentScheduler && !TopFragmentWindow.HasPickedMFMA &&
      (TopFragmentWindow.LiveFragments != 0 || TopCandHasFragmentWork);
  SUnit *PrologueMFMA = nullptr;
  bool PrologueMFMAPending = false;
  if (IsFragmentPrologue && IsOverfullTopFragmentProducer) {
    auto ConsiderMFMA = [&](SUnit *SU, bool FromPending) {
      if (!isMFMALike(SU))
        return;
      if (!PrologueMFMA ||
          getEffectiveStall(Top, SU) < getEffectiveStall(Top, PrologueMFMA) ||
          (getEffectiveStall(Top, SU) == getEffectiveStall(Top, PrologueMFMA) &&
           SU->NodeNum < PrologueMFMA->NodeNum)) {
        PrologueMFMA = SU;
        PrologueMFMAPending = FromPending;
      }
    };
    for (SUnit *SU : Top.Available)
      ConsiderMFMA(SU, false);
    for (SUnit *SU : Top.Pending)
      ConsiderMFMA(SU, true);
  }
  SUnit *SteadyStateMFMA = nullptr;
  bool SteadyStateMFMAPending = false;
  bool NeedsSteadyStateFillers =
      TopFragmentWindow.CompletedDSMicroclusters >= 2 &&
      TopFragmentWindow.MFMAsSinceLastDSRead <
          MFMAFragmentSched.PipelineGroupSize;
  if (EnableMFMAFragmentScheduler && TopFragmentWindow.HasPickedMFMA &&
      TopCand.isValid() &&
      classifyPipeKind(Top, TopCand.SU) == PipeKind::FragmentProducer &&
      (TopFragmentWindow.DSReadsSinceLastMFMA >=
           MFMAFragmentSched.PipelineGroupSize ||
       NeedsSteadyStateFillers)) {
    auto ConsiderMFMA = [&](SUnit *SU, bool FromPending) {
      if (!isMFMALike(SU) ||
          getEffectiveStall(Top, SU) > MFMAFragmentSched.MaxDrainMFMAStall)
        return;
      bool IsUnrelated = !consumesImmatureRecentDSRead(
          SU, TopFragmentWindow.RecentDSReadUnrelatedMFMAs);
      bool BestIsUnrelated =
          SteadyStateMFMA &&
          !consumesImmatureRecentDSRead(
              SteadyStateMFMA, TopFragmentWindow.RecentDSReadUnrelatedMFMAs);
      bool Better =
          !SteadyStateMFMA || (IsUnrelated != BestIsUnrelated && IsUnrelated);
      if (SteadyStateMFMA && IsUnrelated == BestIsUnrelated) {
        unsigned Stall = getEffectiveStall(Top, SU);
        unsigned BestStall = getEffectiveStall(Top, SteadyStateMFMA);
        Better = Stall < BestStall ||
                 (Stall == BestStall && SU->NodeNum < SteadyStateMFMA->NodeNum);
      }
      if (Better) {
        SteadyStateMFMA = SU;
        SteadyStateMFMAPending = FromPending;
      }
    };
    for (SUnit *SU : Top.Available)
      ConsiderMFMA(SU, false);
    for (SUnit *SU : Top.Pending)
      ConsiderMFMA(SU, true);
  }
  SUnit *RampUpFiller = nullptr;
  bool RampUpFillerPending = false;
  SUnit *RampUpMFMA = PrologueMFMA ? PrologueMFMA : SteadyStateMFMA;
  if (RampUpMFMA && TopFragmentWindow.PickedMFMAs < 2 &&
      getEffectiveStall(Top, RampUpMFMA) != 0) {
    auto ConsiderFiller = [&](SUnit *SU, bool FromPending) {
      if (!isSafeMFMAFragmentRampUpFiller(SU) ||
          getEffectiveStall(Top, SU) != 0)
        return;
      if (!RampUpFiller || SU->NodeNum < RampUpFiller->NodeNum) {
        RampUpFiller = SU;
        RampUpFillerPending = FromPending;
      }
    };
    for (SUnit *SU : Top.Available)
      ConsiderFiller(SU, false);
    for (SUnit *SU : Top.Pending)
      ConsiderFiller(SU, true);
  }
  SUnit *MicroclusterDS = nullptr;
  bool MicroclusterDSPending = false;
  unsigned ReadyFragmentProducers = countIssueReadyFragmentProducers(Top);
  bool NeedsSecondClusterDS = TopFragmentWindow.HasPickedMFMA &&
                              TopFragmentWindow.LastPickWasDSRead &&
                              TopFragmentWindow.DSReadsSinceLastMFMA <
                                  MFMAFragmentSched.PipelineGroupSize;
  bool StartReadyMicrocluster =
      TopFragmentWindow.HasPickedMFMA && !TopFragmentWindow.LastPickWasDSRead &&
      ReadyFragmentProducers >= 2 &&
      TopFragmentWindow.MFMAsSinceLastDSRead >=
          (TopFragmentWindow.CompletedDSMicroclusters < 2
               ? 1
               : MFMAFragmentSched.PipelineGroupSize);
  if (EnableMFMAFragmentScheduler &&
      TopFragmentWindow.LiveFragments < TopFragmentWindow.MaxWindow &&
      (StartReadyMicrocluster || NeedsSecondClusterDS)) {
    auto ConsiderDS = [&](SUnit *SU, bool FromPending) {
      if (!isDSReadLike(SU) || getEffectiveStall(Top, SU) != 0)
        return;
      if (!MicroclusterDS ||
          isBetterPullForwardProducer(Top, SU, MicroclusterDS)) {
        MicroclusterDS = SU;
        MicroclusterDSPending = FromPending;
      }
    };
    for (SUnit *SU : Top.Available)
      ConsiderDS(SU, false);
    for (SUnit *SU : Top.Pending)
      ConsiderDS(SU, true);
  }

  // Bottom-up picks are committed in reverse program order. Complete their
  // microcluster here as well; otherwise intervening bottom MFMA picks fix
  // each producer at a separate point before top scheduling can group it.
  SUnit *BottomMicroclusterDS = nullptr;
  bool BottomMicroclusterDSPending = false;
  bool BottomNeedsSecondClusterDS = BotFragmentWindow.HasPickedMFMA &&
                                    BotFragmentWindow.LastPickWasDSRead &&
                                    BotFragmentWindow.DSReadsSinceLastMFMA <
                                        MFMAFragmentSched.PipelineGroupSize;
  if (EnableMFMAFragmentScheduler &&
      BotFragmentWindow.LiveFragments < BotFragmentWindow.MaxWindow &&
      BottomNeedsSecondClusterDS) {
    auto ConsiderDS = [&](SUnit *SU, bool FromPending) {
      if (!isDSReadLike(SU) || getEffectiveStall(Bot, SU) != 0)
        return;
      if (!BottomMicroclusterDS ||
          isBetterPullForwardProducer(Bot, SU, BottomMicroclusterDS)) {
        BottomMicroclusterDS = SU;
        BottomMicroclusterDSPending = FromPending;
      }
    };
    for (SUnit *SU : Bot.Available)
      ConsiderDS(SU, false);
    for (SUnit *SU : Bot.Pending)
      ConsiderDS(SU, true);
  }
  bool BottomFillerPending = false;
  bool TopFillerPending = false;
  bool TopCandIsFragmentProducer =
      TopCand.isValid() &&
      classifyPipeKind(Top, TopCand.SU) == PipeKind::FragmentProducer;
  bool TopCandIsPullForwardProducer =
      TopCandIsFragmentProducer &&
      isPullForwardUnlockingProducer(
          Top, TopCand.SU, TopFragmentWindow.LiveFragments,
          TopFragmentWindow.UsefulWindow, TopFragmentWindow.LastPickWasDSRead,
          TopFragmentWindow.HasPickedMFMA, /*AllowFullWindow=*/true,
          TopFragmentWindow.RecentDSReadUnrelatedMFMAs);
  SUnit *TopFiller =
      !EnableMFMAFragmentScheduler || TopCandIsPullForwardProducer
          ? nullptr
          : findTopMFMAFillerForRecentDS(
                Top, TopCand, TopFragmentWindow.HasPickedMFMA,
                TopFragmentWindow.RecentDSReadUnrelatedMFMAs, TopFillerPending);
  SUnit *BottomFiller =
      !EnableMFMAFragmentScheduler || TopCandIsFragmentProducer
          ? nullptr
          : findBottomMFMAFillerForStalledDS(
                Bot, BotCand, BotFragmentWindow.HasPickedMFMA,
                BotFragmentWindow.DeferredDSReadFillers, BottomFillerPending);
  bool BottomEpilogueIsFull =
      EnableMFMAFragmentScheduler &&
      (TopFragmentWindow.HasPickedMFMA || IsFragmentPrologue) &&
      TopCand.isValid() && BotCand.isValid() && isMFMALike(BotCand.SU) &&
      BotFragmentWindow.MFMAsSinceLastDSRead >=
          MFMAFragmentSched.BottomEpilogueMFMAs;
  if (RampUpFiller) {
    Cand.reset(TopPolicy);
    Cand.SU = RampUpFiller;
    Cand.AtTop = true;
    Cand.Reason = Stall;
    PickedPending = RampUpFillerPending;
  } else if (PrologueMFMA) {
    Cand.reset(TopPolicy);
    Cand.SU = PrologueMFMA;
    Cand.AtTop = true;
    Cand.Reason = Stall;
    PickedPending = PrologueMFMAPending;
  } else if (MicroclusterDS) {
    Cand.reset(TopPolicy);
    Cand.SU = MicroclusterDS;
    Cand.AtTop = true;
    Cand.Reason = TopPathReduce;
    PickedPending = MicroclusterDSPending;
  } else if (SteadyStateMFMA) {
    Cand.reset(TopPolicy);
    Cand.SU = SteadyStateMFMA;
    Cand.AtTop = true;
    Cand.Reason = Stall;
    PickedPending = SteadyStateMFMAPending;
  } else if (BottomEpilogueIsFull) {
    // Keep the bottom boundary from consuming the accumulator chain that the
    // top boundary needs for repeated DS/MFMA steady-state groups. Bottom may
    // still form a compact, deliberately loose MFMA epilogue.
    Cand.setBest(TopCand);
    PickedPending = TopPending;
  } else if (BottomMicroclusterDS) {
    Cand.reset(BotPolicy);
    Cand.SU = BottomMicroclusterDS;
    Cand.AtTop = false;
    Cand.Reason = BotPathReduce;
    PickedPending = BottomMicroclusterDSPending;
  } else if (IsFragmentPrologue && TopCand.isValid() &&
             !IsOverfullTopFragmentProducer) {
    // Keep the opener on one boundary. Bottom-up picks have an independent
    // fragment count and can otherwise inflate the final linear prologue past
    // PrologueWindow even when the top window is already full.
    Cand.setBest(TopCand);
    PickedPending = TopPending;
  } else if (TopFiller) {
    Cand.reset(TopPolicy);
    Cand.SU = TopFiller;
    Cand.AtTop = true;
    Cand.Reason = Stall;
    PickedPending = TopFillerPending;
  } else if (EnableMFMAFragmentScheduler &&
             shouldPreferBoundaryForRecentDSReadSpacing(
                 Top, TopCand, BotCand,
                 TopFragmentWindow.RecentDSReadUnrelatedMFMAs)) {
    Cand.setBest(TopCand);
    PickedPending = TopPending;
  } else if (EnableMFMAFragmentScheduler &&
             shouldPreferBoundaryForRecentDSReadSpacing(
                 Bot, BotCand, TopCand,
                 BotFragmentWindow.RecentDSReadUnrelatedMFMAs)) {
    Cand.setBest(BotCand);
    PickedPending = BotPending;
  } else if (BottomFiller) {
    Cand.reset(BotPolicy);
    Cand.SU = BottomFiller;
    Cand.AtTop = false;
    Cand.Reason = Stall;
    PickedPending = BottomFillerPending;
  } else if (BiasActiveFragmentWindowTopDown && TopCand.isValid() &&
             !IsOverfullTopFragmentProducer &&
             (IsFirstFragmentMFMA || isTopDownBiasCandidate(Top, TopCand.SU))) {
    Cand.setBest(TopCand);
    PickedPending = TopPending;
  } else if (EnableMFMAFragmentScheduler &&
             shouldPreferTopOverBottomStalledDS(Top, Bot, TopCand, BotCand)) {
    Cand.setBest(TopCand);
    PickedPending = TopPending;
  } else if (EnableMFMAFragmentScheduler &&
             tryBidirectionalReadyStall(TryCand, Cand,
                                        TryCand.AtTop ? Top : Bot,
                                        Cand.AtTop ? Top : Bot)) {
    Cand.setBest(TryCand);
    PickedPending = Cand.AtTop ? TopPending : BotPending;
  } else if (BotPending || TopPending) {
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
      if (EnableMFMAFragmentScheduler) {
        bool TopFillerPending = false;
        if (SUnit *TopFiller = findTopMFMAFillerForRecentDS(
                Top, SU, TopFragmentWindow.HasPickedMFMA,
                TopFragmentWindow.RecentDSReadUnrelatedMFMAs,
                TopFillerPending)) {
          SU = TopFiller;
          PickedPending = TopFillerPending;
        }
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
  if (EnableMFMAFragmentScheduler) {
    updateFragmentWindow(SU, IsTopNode);
    // Fragment policy participates in candidate comparison. A pick changes
    // the pipeline phase seen by both boundaries, so neither cached candidate
    // remains valid for the next comparison.
    TopCand.SU = nullptr;
    BotCand.SU = nullptr;
  }

  if (useGCNTrackers()) {
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
    const FragmentWindowState &FWS =
        Zone->isTop() ? TopFragmentWindow : BotFragmentWindow;
    if (EnableMFMAFragmentScheduler &&
        tryRecentDSReadSpacing(TryCand, Cand, *Zone, FWS.HasPickedMFMA,
                               FWS.RecentDSReadUnrelatedMFMAs))
      return TryCand.Reason != NoCand;

    if (EnableMFMAFragmentScheduler &&
        shouldKeepMFMAFragmentCandidate(TryCand, Cand, *Zone, FWS))
      return false;

    if (EnableMFMAFragmentScheduler &&
        tryMFMAFragmentCandidate(TryCand, Cand, *Zone, FWS, true))
      return TryCand.Reason != NoCand;

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
    const FragmentWindowState &FWS =
        Zone->isTop() ? TopFragmentWindow : BotFragmentWindow;
    if (EnableMFMAFragmentScheduler &&
        shouldKeepMFMAFragmentCandidate(TryCand, Cand, *Zone, FWS))
      return false;

    if (EnableMFMAFragmentScheduler &&
        tryMFMAFragmentCandidate(TryCand, Cand, *Zone, FWS, true))
      return TryCand.Reason != NoCand;

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

  GCNSchedStrategy &S = static_cast<GCNSchedStrategy &>(*SchedImpl);
  if (!Regions.empty()) {
    BBLiveInMap = getRegionLiveInMap();
    if (S.useGCNTrackers())
      RegionLiveOuts.buildLiveRegMap();
  }

#ifdef DUMP_MAX_REG_PRESSURE
  if (PrintMaxRPRegUsageBeforeScheduler) {
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
        S.getDownwardTracker()->reset(MRI, LiveIns[RegionIdx]);
        S.getUpwardTracker()->reset(
            MRI, RegionLiveOuts.getLiveRegsForRegionIdx(RegionIdx));
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
    const MachineInstr *DefMI, LiveIntervals *LIS,
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

  int64_t Cost = getRewriteCost(RewriteCands, CopyForUse, CopyForDef);

  // If we haven't found the beneficial conditions, prefer the VGPR form which
  // may result in less cross RC copies.
  if (Cost > 0)
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

  // We need up-to-date live-out info. to query live-out register masks in
  // regions containing rematerializable instructions.
  DAG.RegionLiveOuts.buildLiveRegMap();

  if (!Remater.analyze()) {
    REMAT_DEBUG(dbgs() << "No rematerializable registers\n");
    return false;
  }
  const ScoredRemat::FreqInfo FreqInfo(MF, DAG);

  // Set of registers already marked for potential remterialization; used to
  // avoid rematerialization chains.
  SmallSet<Register, 4> MarkedRegs;

  // Collect candidates. We have more restrictions on what we can track here
  // compared to the rematerializer.
  SmallVector<ScoredRemat, 8> Candidates;
  SmallVector<unsigned> CandidateOrder;
  for (unsigned RegIdx = 0, E = Remater.getNumRegs(); RegIdx < E; ++RegIdx) {
    const Rematerializer::Reg &CandReg = Remater.getReg(RegIdx);

    // Single user only.
    unsigned NumUsers = 0;
    for (const auto &[_, RegionUses] : CandReg.Uses)
      NumUsers += RegionUses.size();
    if (NumUsers != 1)
      continue;

    // We further filter the registers that we can rematerialize based on our
    // current tracking capabilities in the stage. The user cannot itself be
    // marked rematerializable, and no register operand of the defining MI can
    // be marked rematerializable. We also do not rematerialize an instruction
    // if it uses registers that aren't available at its use. This ensures that
    // we are not extending any live range while rematerializing.
    MachineInstr *UseMI = *CandReg.Uses.begin()->getSecond().begin();
    const MachineOperand &UseMO = UseMI->getOperand(0);
    if (UseMO.isReg() && MarkedRegs.contains(UseMO.getReg()))
      continue;
    SlotIndex UseIdx = DAG.LIS->getInstructionIndex(*UseMI).getRegSlot(true);
    SlotIndex RefIdx =
        DAG.LIS->getInstructionIndex(*CandReg.DefMI).getRegSlot(true);
    if (llvm::any_of(CandReg.Dependencies, [&](RegisterIdx DepRegIdx) {
          const Rematerializer::Reg &DepReg = Remater.getReg(DepRegIdx);
          Register DepDefReg = DepReg.getDefReg();
          return MarkedRegs.contains(DepDefReg) ||
                 !Remater.isRegIdenticalAtUses(DepDefReg, DepReg.Mask, RefIdx,
                                               {UseIdx});
        }))
      continue;
    if (llvm::any_of(Remater.getUnrematableDeps(RegIdx),
                     [&](const std::pair<Register, LaneBitmask> &RegAndMask) {
                       const auto &[Reg, Mask] = RegAndMask;
                       return !Remater.isRegIdenticalAtUses(Reg, Mask, RefIdx,
                                                            {UseIdx});
                     }))
      continue;

    MarkedRegs.insert(CandReg.getDefReg());
    ScoredRemat &Cand = Candidates.emplace_back();
    Cand.init(RegIdx, FreqInfo, Remater, DAG);
    Cand.update(TargetRegions, RPTargets, FreqInfo, !TargetOcc);
    if (!Cand.hasNullScore())
      CandidateOrder.push_back(Candidates.size() - 1);
  }

  if (TargetOcc) {
    // Every rematerialization we do here is likely to move the instruction
    // into a higher frequency region, increasing the total sum latency of the
    // instruction itself. This is acceptable if we are eliminating a spill in
    // the process, but when the goal is increasing occupancy we get nothing
    // out of rematerialization if occupancy is not increased in the end; in
    // such cases we want to roll back the rematerialization.
    Rollback = std::make_unique<RollbackSupport>(Remater);
  }

  // Rematerialize registers in successive rounds until all RP targets are
  // satisifed or until we run out of rematerialization candidates.
  BitVector RecomputeRP(DAG.Regions.size());
  for (;;) {
    RecomputeRP.reset();

    // Sort candidates in increasing score order.
    sort(CandidateOrder, [&](unsigned LHSIndex, unsigned RHSIndex) {
      return Candidates[LHSIndex] < Candidates[RHSIndex];
    });

    REMAT_DEBUG({
      dbgs() << "==== NEW REMAT ROUND ====\n"
             << REMAT_PREFIX
             << "Candidates with non-null score, in rematerialization order:\n";
      for (const ScoredRemat &Cand : reverse(Candidates)) {
        dbgs() << REMAT_PREFIX << "  " << Cand.print() << " | "
               << Remater.printRematReg(Cand.RegIdx) << '\n';
      }
      PrintTargetRegions();
    });

    // Rematerialize registers in decreasing score order until we estimate
    // that all RP targets are satisfied or until rematerialization candidates
    // are no longer useful to decrease RP.
    while (!CandidateOrder.empty()) {
      const ScoredRemat &Cand = Candidates[CandidateOrder.back()];
      const Rematerializer::Reg &Reg = Remater.getReg(Cand.RegIdx);

      // When previous rematerializations in this round have already satisfied
      // RP targets in all regions this rematerialization can impact, we have a
      // good indication that our scores have diverged significantly from
      // reality, in which case we interrupt this round and re-score. This also
      // ensures that every rematerialization we perform is possibly impactful
      // in at least one target region.
      if (!Cand.maybeBeneficial(TargetRegions, RPTargets)) {
        REMAT_DEBUG(dbgs() << "Interrupt round on stale score for "
                           << Cand.print() << " | "
                           << Remater.printRematReg(Cand.RegIdx));
        break;
      }
      CandidateOrder.pop_back();

#ifdef EXPENSIVE_CHECKS
      // All uses are known to be available / live at the remat point. Thus,
      // the uses should already be live in to the using region.
      for (MachineOperand &MO : Reg.DefMI->operands()) {
        if (!MO.isReg() || !MO.getReg() || !MO.readsReg())
          continue;

        Register UseReg = MO.getReg();
        if (!UseReg.isVirtual())
          continue;

        LiveInterval &LI = DAG.LIS->getInterval(UseReg);
        LaneBitmask LM = DAG.MRI.getMaxLaneMaskForVReg(MO.getReg());
        if (LI.hasSubRanges() && MO.getSubReg())
          LM = DAG.TRI->getSubRegIndexLaneMask(MO.getSubReg());

        const unsigned UseRegion = Reg.Uses.begin()->first;
        LaneBitmask LiveInMask = DAG.LiveIns[UseRegion].at(UseReg);
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

      // Remove the register from all regions where it is a live-in or live-out,
      // then rematerialize the register.
      REMAT_DEBUG(dbgs() << "** REMAT " << Remater.printRematReg(Cand.RegIdx)
                         << '\n');
      removeFromLiveMaps(Reg.getDefReg(), Cand.LiveIn, Cand.LiveOut);
      if (Rollback) {
        Rollback->LiveMapUpdates.emplace_back(Cand.RegIdx, Cand.LiveIn,
                                              Cand.LiveOut);
      }
      Cand.rematerialize(Remater);

      // Adjust RP targets. The save is guaranteed in regions in which the
      // register is live-through and unused but optimistic in all other regions
      // where the register is live.
      updateRPTargets(Cand.Live, Cand.RPSave);
      RecomputeRP |= Cand.UnpredictableRPSave;
      RescheduleRegions |= Cand.Live;
      if (!TargetRegions.any()) {
        REMAT_DEBUG(dbgs() << "All targets cleared, verifying...\n");
        break;
      }
    }

    if (!updateAndVerifyRPTargets(RecomputeRP) && !TargetRegions.any()) {
      REMAT_DEBUG(dbgs() << "Objectives achieved!\n");
      break;
    }

    // Update the score of remaining candidates and filter out those that have
    // become useless from the vector. Candidates never become useful after
    // having been useless for a round, so we can freely drop them without
    // losing any future rematerialization opportunity.
    unsigned NumUsefulCandidates = 0;
    for (unsigned CandIdx : CandidateOrder) {
      ScoredRemat &Candidate = Candidates[CandIdx];
      Candidate.update(TargetRegions, RPTargets, FreqInfo, !TargetOcc);
      if (!Candidate.hasNullScore())
        CandidateOrder[NumUsefulCandidates++] = CandIdx;
    }
    if (NumUsefulCandidates == 0) {
      REMAT_DEBUG(dbgs() << "Stop on exhausted rematerialization candidates\n");
      break;
    }
    CandidateOrder.truncate(NumUsefulCandidates);
  }

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
    REMAT_DEBUG(dbgs() << "Region " << RegionIdx
                       << " cannot meet occupancy target, interrupting "
                          "re-scheduling in all regions\n");
    RevertAllRegions = true;
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
        ST, DAG.MFI.getDynamicVGPRBlockSize(),
        PressureBefore.getVGPRNum(false));
    unsigned BlocksAfter = AMDGPU::IsaInfo::getAllocatedNumVGPRBlocks(
        ST, DAG.MFI.getDynamicVGPRBlockSize(), PressureAfter.getVGPRNum(false));
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

/// Returns true if reaching def \p RD will be in AGPR form after the rewrite
/// and so needs no bridge copy: a candidate MFMA in \p RewriteSet, an
/// AV_MOV_*_IMM_PSEUDO, or a copy from a candidate src2 reg in \p CandSrc2Regs.
/// A non-candidate MFMA stays in VGPR form and still needs a bridge.
static bool isReachingDefAGPRForm(
    MachineInstr *RD, const SmallPtrSetImpl<MachineInstr *> &RewriteSet,
    const DenseSet<Register> &CandSrc2Regs, const SIInstrInfo &TII) {
  if (TII.isMAI(*RD))
    return RewriteSet.contains(RD);
  if (RD->getOpcode() == AMDGPU::AV_MOV_B32_IMM_PSEUDO ||
      RD->getOpcode() == AMDGPU::AV_MOV_B64_IMM_PSEUDO)
    return true;
  if (RD->isCopy() && CandSrc2Regs.contains(RD->getOperand(1).getReg()))
    return true;
  return false;
}

bool RewriteMFMAFormStage::hasUseRequiringVGPR(
    ArrayRef<SlotIndex> Src2ReachingDefs,
    const SmallPtrSetImpl<MachineInstr *> &RewriteSet) {
  for (SlotIndex RDIdx : Src2ReachingDefs) {
    const MachineInstr *RD = DAG.LIS->getInstructionFromIndex(RDIdx);
    SmallVector<MachineOperand *, 8> ReachingUses;
    findReachingUses(RD, DAG.LIS, ReachingUses);
    for (const MachineOperand *UseMO : ReachingUses) {
      const MachineInstr *UseMI = UseMO->getParent();
      if (UseMI->isCopy())
        continue;
      if (TII->isMAI(*UseMI) && RewriteSet.contains(UseMI))
        continue;
      return true;
    }
  }
  return false;
}

void RewriteMFMAFormStage::resetRewriteCandsToVGPR(
    ArrayRef<std::pair<MachineInstr *, unsigned>> RewriteCands) {
  for (auto [MI, OriginalOpcode] : RewriteCands) {
    assert(TII->isMAI(*MI));
    const TargetRegisterClass *ADefRC =
        DAG.MRI.getRegClass(MI->getOperand(0).getReg());
    const TargetRegisterClass *VDefRC = SRI->getEquivalentVGPRClass(ADefRC);
    DAG.MRI.setRegClass(MI->getOperand(0).getReg(), VDefRC);
    MI->setDesc(TII->get(OriginalOpcode));

    MachineOperand *Src2 = TII->getNamedOperand(*MI, AMDGPU::OpName::src2);
    if (!Src2->isReg())
      continue;

    // Have to get src types separately since subregs may cause C and D
    // registers to be different types even though the actual operand is
    // the same size.
    const TargetRegisterClass *AUseRC = DAG.MRI.getRegClass(Src2->getReg());
    const TargetRegisterClass *VUseRC = SRI->getEquivalentVGPRClass(AUseRC);
    DAG.MRI.setRegClass(Src2->getReg(), VUseRC);
  }
}

bool RewriteMFMAFormStage::isRewriteCandidate(MachineInstr *MI) const {
  if (!static_cast<const SIInstrInfo *>(DAG.TII)->isMAI(*MI))
    return false;
  if (AMDGPU::getMFMASrcCVDstAGPROp(MI->getOpcode()) == -1)
    return false;
  // Reject candidates whose users force an unavoidable bridge copy.
  Register DstReg = MI->getOperand(0).getReg();
  for (const MachineOperand &Use : DAG.MRI.use_nodbg_operands(DstReg)) {
    if (!TII->isMAI(*Use.getParent()) && !Use.getParent()->isCopy())
      return false;
  }
  return true;
}

bool RewriteMFMAFormStage::initHeuristics(
    std::vector<std::pair<MachineInstr *, unsigned>> &RewriteCands,
    DenseMap<MachineBasicBlock *, std::set<Register>> &CopyForUse,
    SmallPtrSetImpl<MachineInstr *> &CopyForDef) {
  bool Changed = false;

  // Collect the candidate group, its members share AGPR-form operands
  // post-rewrite, so reaching defs feeding any member don't need bridge copy.
  SmallPtrSet<MachineInstr *, 16> RewriteSet;
  DenseSet<Register> CandSrc2Regs;
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      if (!isRewriteCandidate(&MI))
        continue;
      RewriteSet.insert(&MI);
      MachineOperand *Src2 = TII->getNamedOperand(MI, AMDGPU::OpName::src2);
      if (Src2 && Src2->isReg())
        CandSrc2Regs.insert(Src2->getReg());
    }
  }

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

        // If src2 has a use that must remain VGPR, it cannot be reclassified to
        // AGPR.
        bool Src2NeedsVGPR = hasUseRequiringVGPR(Src2ReachingDefs, RewriteSet);
        Src2NeedsVGPRCache[&MI] = Src2NeedsVGPR;

        for (SlotIndex RDIdx : Src2ReachingDefs) {
          MachineInstr *RD = DAG.LIS->getInstructionFromIndex(RDIdx);
          if (!Src2NeedsVGPR &&
              isReachingDefAGPRForm(RD, RewriteSet, CandSrc2Regs, *TII))
            continue;
          CopyForDef.insert(RD);
        }
      }

      MachineOperand &Dst = MI.getOperand(0);
      SmallVector<MachineOperand *, 8> DstReachingUses;

      findReachingUses(&MI, DAG.LIS, DstReachingUses);

      for (MachineOperand *RUOp : DstReachingUses) {
        MachineInstr *UserMI = RUOp->getParent();
        // Group members read the AGPR result directly.
        if (TII->isMAI(*UserMI) && RewriteSet.contains(UserMI))
          continue;

        // For any user of the result of the MFMA which is not an MFMA, we
        // insert a copy. For a given register, we will only insert one copy
        // per user block.
        CopyForUse[UserMI->getParent()].insert(RUOp->getReg());

        if (TII->isMAI(*UserMI))
          continue;

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

int64_t RewriteMFMAFormStage::getRewriteCost(
    ArrayRef<std::pair<MachineInstr *, unsigned>> RewriteCands,
    const DenseMap<MachineBasicBlock *, std::set<Register>> &CopyForUse,
    const SmallPtrSetImpl<MachineInstr *> &CopyForDef) {
  MachineBlockFrequencyInfo *MBFI = DAG.MBFI;

  int64_t BestSpillCost = 0;
  int64_t Cost = 0;
  uint64_t EntryFreq = MBFI->getEntryFreq().getFrequency();

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

    // For the cases we care about (i.e. ArchVGPR usage is greater than the
    // addressable limit), rewriting alone should bring pressure to manageable
    // level. If we find any such region, then the rewrite is potentially
    // beneficial.
    GCNRegPressure PressureAfter = DAG.getRealRegPressure(Region);
    unsigned SpillCostAfter = PressureAfter.getVGPRSpills(
        MF, ArchVGPRThreshold, AGPRThreshold, CombinedThreshold);

    uint64_t BlockFreq =
        MBFI->getBlockFreq(DAG.Regions[Region].first->getParent())
            .getFrequency();

    bool RelativeFreqIsDenom = EntryFreq > BlockFreq;
    uint64_t RelativeFreq = EntryFreq && BlockFreq
                                ? (RelativeFreqIsDenom ? EntryFreq / BlockFreq
                                                       : BlockFreq / EntryFreq)
                                : 1;

    // This assumes perfect spilling / splitting -- using one spill / copy
    // instruction and one restoreFrom / copy for each excess register,
    int64_t SpillCost = ((int)SpillCostAfter - (int)SpillCostBefore) * 2;

    // Also account for the block frequency.
    if (RelativeFreqIsDenom)
      SpillCost /= (int64_t)RelativeFreq;
    else
      SpillCost *= (int64_t)RelativeFreq;

    // If we have increased spilling in any block, just bail.
    if (SpillCost > 0) {
      resetRewriteCandsToVGPR(RewriteCands);
      return SpillCost;
    }

    if (SpillCost < BestSpillCost)
      BestSpillCost = SpillCost;
  }

  // Set the cost to the largest decrease in spill cost in order to not double
  // count spill reductions.
  Cost = BestSpillCost;
  assert(Cost <= 0);

  unsigned CopyCost = 0;

  // For each CopyForDef, increase the cost by the register size while
  // accounting for block frequency.
  for (MachineInstr *DefMI : CopyForDef) {
    Register DefReg = DefMI->getOperand(0).getReg();
    uint64_t DefFreq =
        EntryFreq
            ? MBFI->getBlockFreq(DefMI->getParent()).getFrequency() / EntryFreq
            : 1;

    const TargetRegisterClass *RC = DAG.MRI.getRegClass(DefReg);
    CopyCost += RC->getCopyCost() * DefFreq;
  }

  // Account for CopyForUse copies in each block that the register is used.
  for (auto &[UseBlock, UseRegs] : CopyForUse) {
    uint64_t UseFreq =
        EntryFreq ? MBFI->getBlockFreq(UseBlock).getFrequency() / EntryFreq : 1;

    for (Register UseReg : UseRegs) {
      const TargetRegisterClass *RC = DAG.MRI.getRegClass(UseReg);
      CopyCost += RC->getCopyCost() * UseFreq;
    }
  }

  // Reset the classes that were changed to AGPR for better register bank
  // analysis. We must do rewriting after copy-insertion, as some defs of the
  // register may require VGPR.  Additionally, if we bail out and don't perform
  // the rewrite then these need to be restored anyway.
  resetRewriteCandsToVGPR(RewriteCands);

  return Cost + CopyCost;
}

bool RewriteMFMAFormStage::rewrite(
    ArrayRef<std::pair<MachineInstr *, unsigned>> RewriteCands) {
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

  // Collect the candidate group; its members share AGPR-form operands
  // post-rewrite, so reaching defs feeding any member need no bridge copy.
  SmallPtrSet<MachineInstr *, 16> RewriteCandsSet;
  DenseSet<Register> RewriteSrc2Regs;
  for (auto &[MI, OriginalOpcode] : RewriteCands) {
    RewriteCandsSet.insert(MI);
    MachineOperand *Src2 = TII->getNamedOperand(*MI, AMDGPU::OpName::src2);
    if (Src2 && Src2->isReg())
      RewriteSrc2Regs.insert(Src2->getReg());
  }

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

      // If src2 has a use that must remain VGPR, it cannot be reclassified to
      // AGPR.
      bool Src2NeedsVGPR = Src2NeedsVGPRCache.lookup(MI);

      for (SlotIndex RDIndex : Src2ReachingDefs) {
        MachineInstr *RD = DAG.LIS->getInstructionFromIndex(RDIndex);
        if (!Src2NeedsVGPR &&
            isReachingDefAGPRForm(RD, RewriteCandsSet, RewriteSrc2Regs, *TII))
          continue;

        Src2DefsReplace.insert(RD);
      }

      if (!Src2DefsReplace.empty()) {
        auto RI = RedefMap.find(Src2Reg);
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
      MachineInstr *UserMI = RUOp->getParent();
      // Group members read the AGPR result directly.
      if (TII->isMAI(*UserMI) && RewriteCandsSet.contains(UserMI))
        continue;

      // If there is a non mai reaching use, then we need a copy.
      if (find(DstReachingUseCopies, RUOp) == DstReachingUseCopies.end())
        DstReachingUseCopies.push_back(RUOp);

      // Non-rewritten MAI: its defs aren't being reclassified.
      if (TII->isMAI(*UserMI))
        continue;

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
      auto RI = RedefMap.find(DstReg);
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
          auto LMI = LastMIToRegion.find(RD);
          if (LMI != LastMIToRegion.end()) {
            unsigned UpdateRegion = LMI->second;
            DAG.Regions[UpdateRegion].second = VGPRCopy;
            LastMIToRegion.erase(RD);
          }
        }
      }
    }

    DenseSet<MachineOperand *> &DstRegSet = ReplaceMap[DstReg];
    // One AGPR→VGPR copy per dst register, shared by all same-block uses.
    Register SameBlockCopyReg;
    MachineInstr *EarliestSameBlockUse = nullptr;
    for (MachineOperand *RU : DstReachingUseCopies) {
      MachineBasicBlock *RUBlock = RU->getParent()->getParent();
      // Just keep track of the reaching use of this register by block. After we
      // have scanned all the MFMAs we can find optimal insert pts.
      if (RUBlock != MI->getParent()) {
        ReachingUseTracker[RUBlock->getNumber()][DstReg].insert(RU);
        continue;
      }

      // Lazily create the copy register on first same-block use.
      if (!SameBlockCopyReg.isValid()) {
        const TargetRegisterClass *DstRC = DAG.MRI.getRegClass(DstReg);
        const TargetRegisterClass *VGPRRC = SRI->getEquivalentVGPRClass(DstRC);
        SameBlockCopyReg = DAG.MRI.createVirtualRegister(VGPRRC);
      }

      // Track the earliest use for copy insertion point.
      MachineInstr *UseInst = RU->getParent();
      if (!EarliestSameBlockUse ||
          SlotIndex::isEarlierInstr(
              DAG.LIS->getInstructionIndex(*UseInst),
              DAG.LIS->getInstructionIndex(*EarliestSameBlockUse)))
        EarliestSameBlockUse = UseInst;
      RU->setReg(SameBlockCopyReg);
    }

    // Insert the copy before the earliest same-block use.
    if (SameBlockCopyReg.isValid()) {
      MachineInstrBuilder VGPRCopy =
          BuildMI(*EarliestSameBlockUse->getParent(),
                  EarliestSameBlockUse->getIterator(), DebugLoc(),
                  TII->get(TargetOpcode::COPY), SameBlockCopyReg)
              .addUse(DstReg, {}, 0);
      DAG.LIS->InsertMachineInstrInMaps(*VGPRCopy);
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
      auto FI = FirstMIToRegion.find(UseInst);
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
    auto RI = RedefMap.find(RewriteReg);
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

bool PreRARematStage::ScoredRemat::maybeBeneficial(
    const BitVector &TargetRegions, ArrayRef<GCNRPTarget> RPTargets) const {
  for (unsigned I : TargetRegions.set_bits()) {
    if (Live[I] && RPTargets[I].isSaveBeneficial(RPSave))
      return true;
  }
  return false;
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

void PreRARematStage::ScoredRemat::init(RegisterIdx RegIdx,
                                        const FreqInfo &Freq,
                                        const Rematerializer &Remater,
                                        GCNScheduleDAGMILive &DAG) {
  this->RegIdx = RegIdx;
  const unsigned NumRegions = DAG.Regions.size();
  LiveIn.resize(NumRegions);
  LiveOut.resize(NumRegions);
  Live.resize(NumRegions);
  UnpredictableRPSave.resize(NumRegions);

  const Rematerializer::Reg &Reg = Remater.getReg(RegIdx);
  Register DefReg = Reg.getDefReg();
  assert(Reg.Uses.size() == 1 && "expected users in single region");
  const unsigned UseRegion = Reg.Uses.begin()->first;

  // Mark regions in which the rematerializable register is live.
  for (unsigned I = 0, E = NumRegions; I != E; ++I) {
    if (DAG.LiveIns[I].contains(DefReg))
      LiveIn.set(I);
    if (DAG.RegionLiveOuts.getLiveRegsForRegionIdx(I).contains(DefReg))
      LiveOut.set(I);

    // If the register is both unused and live-through in the region, the
    // latter's RP is guaranteed to decrease.
    if (!LiveIn[I] || !LiveOut[I] || I == UseRegion)
      UnpredictableRPSave.set(I);
  }
  Live |= LiveIn;
  Live |= LiveOut;
  RPSave.inc(DefReg, LaneBitmask::getNone(), Reg.Mask, DAG.MRI);

  // Get frequencies of defining and using regions. A rematerialization from the
  // least frequent region to the most frequent region will yield the greatest
  // in order to penalize rematerializations from or into regions whose
  int64_t DefOrMin = std::max(Freq.Regions[Reg.DefRegion], Freq.MinFreq);
  int64_t UseOrMax = Freq.Regions[UseRegion];
  if (!UseOrMax)
    UseOrMax = Freq.MaxFreq;
  FreqDiff = DefOrMin - UseOrMax;
}

void PreRARematStage::ScoredRemat::update(const BitVector &TargetRegions,
                                          ArrayRef<GCNRPTarget> RPTargets,
                                          const FreqInfo &FreqInfo,
                                          bool ReduceSpill) {
  MaxFreq = 0;
  RegionImpact = 0;
  for (unsigned I : TargetRegions.set_bits()) {
    if (!Live[I])
      continue;

    // The rematerialization must contribute positively in at least one
    // register class with usage above the RP target for this region to
    // contribute to the score.
    const GCNRPTarget &RegionTarget = RPTargets[I];
    const unsigned NumRegsBenefit = RegionTarget.getNumRegsBenefit(RPSave);
    if (!NumRegsBenefit)
      continue;

    // Regions in which RP is guaranteed to decrease have more weight.
    RegionImpact += (UnpredictableRPSave[I] ? 1 : 2) * NumRegsBenefit;

    if (ReduceSpill) {
      uint64_t Freq = FreqInfo.Regions[I];
      if (UnpredictableRPSave[I]) {
        // Apply a frequency penalty in regions in which we are not sure that RP
        // will decrease.
        Freq /= 2;
      }
      MaxFreq = std::max(MaxFreq, Freq);
    }
  }
}

void PreRARematStage::ScoredRemat::rematerialize(
    Rematerializer &Remater) const {
  const Rematerializer::Reg &Reg = Remater.getReg(RegIdx);
  Rematerializer::DependencyReuseInfo DRI;
  for (RegisterIdx DepRegIdx : Reg.Dependencies)
    DRI.reuse(DepRegIdx);
  unsigned UseRegion = Reg.Uses.begin()->first;
  Remater.rematerializeToRegion(RegIdx, UseRegion, DRI);
}

void PreRARematStage::updateRPTargets(const BitVector &Regions,
                                      const GCNRegPressure &RPSave) {
  for (unsigned I : Regions.set_bits()) {
    RPTargets[I].saveRP(RPSave);
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

void PreRARematStage::removeFromLiveMaps(Register Reg, const BitVector &LiveIn,
                                         const BitVector &LiveOut) {
  assert(LiveIn.size() == DAG.Regions.size() &&
         LiveOut.size() == DAG.Regions.size() && "region num mismatch");
  for (unsigned I : LiveIn.set_bits())
    DAG.LiveIns[I].erase(Reg);
  for (unsigned I : LiveOut.set_bits())
    DAG.RegionLiveOuts.getLiveRegsForRegionIdx(I).erase(Reg);
}

void PreRARematStage::addToLiveMaps(Register Reg, LaneBitmask Mask,
                                    const BitVector &LiveIn,
                                    const BitVector &LiveOut) {
  assert(LiveIn.size() == DAG.Regions.size() &&
         LiveOut.size() == DAG.Regions.size() && "region num mismatch");
  std::pair<Register, LaneBitmask> LiveReg(Reg, Mask);
  for (unsigned I : LiveIn.set_bits())
    DAG.LiveIns[I].insert(LiveReg);
  for (unsigned I : LiveOut.set_bits())
    DAG.RegionLiveOuts.getLiveRegsForRegionIdx(I).insert(LiveReg);
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
    REMAT_DEBUG(dbgs() << "Reverting re-scheduling in region " << RegionIdx
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

  // Roll back changes made by the stage, then recompute pressure in all
  // affected regions.
  REMAT_DEBUG(dbgs() << "==== ROLLBACK ====\n");
  assert(Rollback && "rollbacker should be defined");
  Rollback->Listener.rollback(Remater);
  for (const auto &[RegIdx, LiveIn, LiveOut] : Rollback->LiveMapUpdates) {
    const Rematerializer::Reg &Reg = Remater.getReg(RegIdx);
    addToLiveMaps(Reg.getDefReg(), Reg.Mask, LiveIn, LiveOut);
  }

#ifdef EXPENSIVE_CHECKS
  // In particular, we want to check for coherent MI/slot order in regions in
  // which reverts and/or rollbacks may have happened.
  MF.verify();
#endif
  for (unsigned I : RescheduleRegions.set_bits())
    DAG.Pressure[I] = DAG.getRealRegPressure(I);

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
    if (isMFMAFragmentSchedulerEnabled(MF.getSubtarget<GCNSubtarget>()))
      addMutation(createMFMAFragmentPostRASchedOrderDAGMutation());
  }

  ScheduleDAGMI::schedule();
}

void GCNPostScheduleDAGMILive::finalizeSchedule() {
  if (HasIGLPInstrs)
    SavedMutations.swap(Mutations);

  ScheduleDAGMI::finalizeSchedule();
}
