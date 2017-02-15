//===-- GCNSchedStrategy.cpp - GCN Scheduler Strategy ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This contains a MachineSchedStrategy implementation for maximizing wave
/// occupancy on GCN hardware.
//===----------------------------------------------------------------------===//

#include "GCNSchedStrategy.h"
#include "AMDGPUSubtarget.h"
#include "SIInstrInfo.h"
#include "SIMachineFunctionInfo.h"
#include "SIRegisterInfo.h"
#include "llvm/CodeGen/RegisterClassInfo.h"

#define DEBUG_TYPE "misched"

using namespace llvm;

GCNMaxOccupancySchedStrategy::GCNMaxOccupancySchedStrategy(
    const MachineSchedContext *C) :
    GenericScheduler(C) { }

static unsigned getMaxWaves(unsigned SGPRs, unsigned VGPRs,
                            const MachineFunction &MF) {

  const SISubtarget &ST = MF.getSubtarget<SISubtarget>();
  const SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();
  unsigned MinRegOccupancy = std::min(ST.getOccupancyWithNumSGPRs(SGPRs),
                                      ST.getOccupancyWithNumVGPRs(VGPRs));
  return std::min(MinRegOccupancy,
                  ST.getOccupancyWithLocalMemSize(MFI->getLDSSize(),
                                                  *MF.getFunction()));
}

void GCNMaxOccupancySchedStrategy::initialize(ScheduleDAGMI *DAG) {
  GenericScheduler::initialize(DAG);

  const SIRegisterInfo *SRI = static_cast<const SIRegisterInfo*>(TRI);

  // FIXME: This is also necessary, because some passes that run after
  // scheduling and before regalloc increase register pressure.
  const int ErrorMargin = 3;

  SGPRExcessLimit = Context->RegClassInfo
    ->getNumAllocatableRegs(&AMDGPU::SGPR_32RegClass) - ErrorMargin;
  VGPRExcessLimit = Context->RegClassInfo
    ->getNumAllocatableRegs(&AMDGPU::VGPR_32RegClass) - ErrorMargin;
  SGPRCriticalLimit = SRI->getRegPressureSetLimit(DAG->MF,
                        SRI->getSGPRPressureSet()) - ErrorMargin;
  VGPRCriticalLimit = SRI->getRegPressureSetLimit(DAG->MF,
                        SRI->getVGPRPressureSet()) - ErrorMargin;
}

void GCNMaxOccupancySchedStrategy::initCandidate(SchedCandidate &Cand, SUnit *SU,
                                     bool AtTop, const RegPressureTracker &RPTracker,
                                     const SIRegisterInfo *SRI,
                                     unsigned SGPRPressure,
                                     unsigned VGPRPressure) {

  Cand.SU = SU;
  Cand.AtTop = AtTop;

  // getDownwardPressure() and getUpwardPressure() make temporary changes to
  // the the tracker, so we need to pass those function a non-const copy.
  RegPressureTracker &TempTracker = const_cast<RegPressureTracker&>(RPTracker);

  std::vector<unsigned> Pressure;
  std::vector<unsigned> MaxPressure;

  if (AtTop)
    TempTracker.getDownwardPressure(SU->getInstr(), Pressure, MaxPressure);
  else {
    // FIXME: I think for bottom up scheduling, the register pressure is cached
    // and can be retrieved by DAG->getPressureDif(SU).
    TempTracker.getUpwardPressure(SU->getInstr(), Pressure, MaxPressure);
  }

  unsigned NewSGPRPressure = Pressure[SRI->getSGPRPressureSet()];
  unsigned NewVGPRPressure = Pressure[SRI->getVGPRPressureSet()];

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

  // We only need to update the RPDelata for instructions that increase
  // register pressure.  Instructions that decrease or keep reg pressure
  // the same will be marked as RegExcess in tryCandidate() when they
  // are compared with instructions that increase the register pressure.
  if (ShouldTrackVGPRs && NewVGPRPressure >= VGPRExcessLimit) {
    Cand.RPDelta.Excess = PressureChange(SRI->getVGPRPressureSet());
    Cand.RPDelta.Excess.setUnitInc(NewVGPRPressure - VGPRExcessLimit);
  }

  if (ShouldTrackSGPRs && NewSGPRPressure >= SGPRExcessLimit) {
    Cand.RPDelta.Excess = PressureChange(SRI->getSGPRPressureSet());
    Cand.RPDelta.Excess.setUnitInc(NewSGPRPressure - SGPRExcessLimit);
  }

  // Register pressure is considered 'CRITICAL' if it is approaching a value
  // that would reduce the wave occupancy for the execution unit.  When
  // register pressure is 'CRITICAL', increading SGPR and VGPR pressure both
  // has the same cost, so we don't need to prefer one over the other.

  int SGPRDelta = NewSGPRPressure - SGPRCriticalLimit;
  int VGPRDelta = NewVGPRPressure - VGPRCriticalLimit;

  if (SGPRDelta >= 0 || VGPRDelta >= 0) {
    if (SGPRDelta > VGPRDelta) {
      Cand.RPDelta.CriticalMax = PressureChange(SRI->getSGPRPressureSet());
      Cand.RPDelta.CriticalMax.setUnitInc(SGPRDelta);
    } else {
      Cand.RPDelta.CriticalMax = PressureChange(SRI->getVGPRPressureSet());
      Cand.RPDelta.CriticalMax.setUnitInc(VGPRDelta);
    }
  }
}

// This function is mostly cut and pasted from
// GenericScheduler::pickNodeFromQueue()
void GCNMaxOccupancySchedStrategy::pickNodeFromQueue(SchedBoundary &Zone,
                                         const CandPolicy &ZonePolicy,
                                         const RegPressureTracker &RPTracker,
                                         SchedCandidate &Cand) {
  const SIRegisterInfo *SRI = static_cast<const SIRegisterInfo*>(TRI);
  ArrayRef<unsigned> Pressure = RPTracker.getRegSetPressureAtPos();
  unsigned SGPRPressure = Pressure[SRI->getSGPRPressureSet()];
  unsigned VGPRPressure = Pressure[SRI->getVGPRPressureSet()];
  ReadyQueue &Q = Zone.Available;
  for (SUnit *SU : Q) {

    SchedCandidate TryCand(ZonePolicy);
    initCandidate(TryCand, SU, Zone.isTop(), RPTracker, SRI,
                  SGPRPressure, VGPRPressure);
    // Pass SchedBoundary only when comparing nodes from the same boundary.
    SchedBoundary *ZoneArg = Cand.AtTop == TryCand.AtTop ? &Zone : nullptr;
    GenericScheduler::tryCandidate(Cand, TryCand, ZoneArg);
    if (TryCand.Reason != NoCand) {
      // Initialize resource delta if needed in case future heuristics query it.
      if (TryCand.ResDelta == SchedResourceDelta())
        TryCand.initResourceDelta(Zone.DAG, SchedModel);
      Cand.setBest(TryCand);
    }
  }
}

static int getBidirectionalReasonRank(GenericSchedulerBase::CandReason Reason) {
  switch (Reason) {
  default:
    return Reason;
  case GenericSchedulerBase::RegCritical:
  case GenericSchedulerBase::RegExcess:
    return -Reason;
 }
}

// This function is mostly cut and pasted from
// GenericScheduler::pickNodeBidirectional()
SUnit *GCNMaxOccupancySchedStrategy::pickNodeBidirectional(bool &IsTopNode) {
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
  DEBUG(dbgs() << "Picking from Bot:\n");
  if (!BotCand.isValid() || BotCand.SU->isScheduled ||
      BotCand.Policy != BotPolicy) {
    BotCand.reset(CandPolicy());
    pickNodeFromQueue(Bot, BotPolicy, DAG->getBotRPTracker(), BotCand);
    assert(BotCand.Reason != NoCand && "failed to find the first candidate");
  } else {
    DEBUG(traceCandidate(BotCand));
  }

  // Check if the top Q has a better candidate.
  DEBUG(dbgs() << "Picking from Top:\n");
  if (!TopCand.isValid() || TopCand.SU->isScheduled ||
      TopCand.Policy != TopPolicy) {
    TopCand.reset(CandPolicy());
    pickNodeFromQueue(Top, TopPolicy, DAG->getTopRPTracker(), TopCand);
    assert(TopCand.Reason != NoCand && "failed to find the first candidate");
  } else {
    DEBUG(traceCandidate(TopCand));
  }

  // Pick best from BotCand and TopCand.
  DEBUG(
    dbgs() << "Top Cand: ";
    traceCandidate(TopCand);
    dbgs() << "Bot Cand: ";
    traceCandidate(BotCand);
  );
  SchedCandidate Cand;
  if (TopCand.Reason == BotCand.Reason) {
    Cand = BotCand;
    GenericSchedulerBase::CandReason TopReason = TopCand.Reason;
    TopCand.Reason = NoCand;
    GenericScheduler::tryCandidate(Cand, TopCand, nullptr);
    if (TopCand.Reason != NoCand) {
      Cand.setBest(TopCand);
    } else {
      TopCand.Reason = TopReason;
    }
  } else {
    if (TopCand.Reason == RegExcess && TopCand.RPDelta.Excess.getUnitInc() <= 0) {
      Cand = TopCand;
    } else if (BotCand.Reason == RegExcess && BotCand.RPDelta.Excess.getUnitInc() <= 0) {
      Cand = BotCand;
    } else if (TopCand.Reason == RegCritical && TopCand.RPDelta.CriticalMax.getUnitInc() <= 0) {
      Cand = TopCand;
    } else if (BotCand.Reason == RegCritical && BotCand.RPDelta.CriticalMax.getUnitInc() <= 0) {
      Cand = BotCand;
    } else {
      int TopRank = getBidirectionalReasonRank(TopCand.Reason);
      int BotRank = getBidirectionalReasonRank(BotCand.Reason);
      if (TopRank > BotRank) {
        Cand = TopCand;
      } else {
        Cand = BotCand;
      }
    }
  }
  DEBUG(
    dbgs() << "Picking: ";
    traceCandidate(Cand);
  );

  IsTopNode = Cand.AtTop;
  return Cand.SU;
}

// This function is mostly cut and pasted from
// GenericScheduler::pickNode()
SUnit *GCNMaxOccupancySchedStrategy::pickNode(bool &IsTopNode) {
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
        pickNodeFromQueue(Top, NoPolicy, DAG->getTopRPTracker(), TopCand);
        assert(TopCand.Reason != NoCand && "failed to find a candidate");
        SU = TopCand.SU;
      }
      IsTopNode = true;
    } else if (RegionPolicy.OnlyBottomUp) {
      SU = Bot.pickOnlyChoice();
      if (!SU) {
        CandPolicy NoPolicy;
        BotCand.reset(NoPolicy);
        pickNodeFromQueue(Bot, NoPolicy, DAG->getBotRPTracker(), BotCand);
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

  DEBUG(dbgs() << "Scheduling SU(" << SU->NodeNum << ") " << *SU->getInstr());
  return SU;
}

void GCNScheduleDAGMILive::schedule() {
  const SIRegisterInfo *SRI = static_cast<const SIRegisterInfo*>(TRI);

  std::vector<MachineInstr*> Unsched;
  Unsched.reserve(NumRegionInstrs);
  for (auto &I : *this)
    Unsched.push_back(&I);

  ScheduleDAGMILive::schedule();

  // Check the results of scheduling.
  GCNMaxOccupancySchedStrategy &S = (GCNMaxOccupancySchedStrategy&)*SchedImpl;
  std::vector<unsigned> UnschedPressure = getRegPressure().MaxSetPressure;
  unsigned MaxSGPRs = std::max(
    getTopRPTracker().getPressure().MaxSetPressure[SRI->getSGPRPressureSet()],
    getBotRPTracker().getPressure().MaxSetPressure[SRI->getSGPRPressureSet()]);
  unsigned MaxVGPRs = std::max(
    getTopRPTracker().getPressure().MaxSetPressure[SRI->getVGPRPressureSet()],
    getBotRPTracker().getPressure().MaxSetPressure[SRI->getVGPRPressureSet()]);
  DEBUG(dbgs() << "Pressure after scheduling:\nSGPR = " << MaxSGPRs
               << "\nVGPR = " << MaxVGPRs << '\n');
  if (MaxSGPRs <= S.SGPRCriticalLimit &&
      MaxVGPRs <= S.VGPRCriticalLimit) {
    DEBUG(dbgs() << "Pressure in desired limits, done.\n");
    return;
  }
  unsigned WavesAfter = getMaxWaves(MaxSGPRs, MaxVGPRs, MF);
  unsigned WavesUnsched = getMaxWaves(UnschedPressure[SRI->getSGPRPressureSet()],
                            UnschedPressure[SRI->getVGPRPressureSet()], MF);
  DEBUG(dbgs() << "Occupancy before scheduling: " << WavesUnsched <<
        ", after " << WavesAfter << ".\n");
  if (WavesAfter >= WavesUnsched)
    return;

  DEBUG(dbgs() << "Attempting to revert scheduling.\n");
  RegionEnd = RegionBegin;
  for (MachineInstr *MI : Unsched) {
    if (MI->getIterator() != RegionEnd) {
      BB->remove(MI);
      BB->insert(RegionEnd, MI);
      if (LIS) {
        LIS->handleMove(*MI, true);
        RegisterOperands RegOpers;
        RegOpers.collect(*MI, *TRI, MRI, ShouldTrackLaneMasks, false);
        if (ShouldTrackLaneMasks) {
          // Adjust liveness and add missing dead+read-undef flags.
          SlotIndex SlotIdx = LIS->getInstructionIndex(*MI).getRegSlot();
          RegOpers.adjustLaneLiveness(*LIS, MRI, SlotIdx, MI);
        } else {
          // Adjust for missing dead-def flags.
          RegOpers.detectDeadDefs(*MI, *LIS);
        }
      }
    }
    RegionEnd = MI->getIterator();
    ++RegionEnd;
    DEBUG(dbgs() << "Scheduling " << *MI);
  }
  RegionBegin = Unsched.front()->getIterator();

  placeDebugValues();
}
