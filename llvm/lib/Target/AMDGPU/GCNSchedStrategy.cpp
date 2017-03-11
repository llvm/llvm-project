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
#include "llvm/Support/MathExtras.h"

#define DEBUG_TYPE "misched"

using namespace llvm;

GCNMaxOccupancySchedStrategy::GCNMaxOccupancySchedStrategy(
    const MachineSchedContext *C) :
    GenericScheduler(C), TargetOccupancy(0), MF(nullptr) { }

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

  if (MF != &DAG->MF)
    TargetOccupancy = 0;
  MF = &DAG->MF;

  const SISubtarget &ST = MF->getSubtarget<SISubtarget>();

  // FIXME: This is also necessary, because some passes that run after
  // scheduling and before regalloc increase register pressure.
  const int ErrorMargin = 3;

  SGPRExcessLimit = Context->RegClassInfo
    ->getNumAllocatableRegs(&AMDGPU::SGPR_32RegClass) - ErrorMargin;
  VGPRExcessLimit = Context->RegClassInfo
    ->getNumAllocatableRegs(&AMDGPU::VGPR_32RegClass) - ErrorMargin;
  if (TargetOccupancy) {
    SGPRCriticalLimit = ST.getMaxNumSGPRs(TargetOccupancy, true);
    VGPRCriticalLimit = ST.getMaxNumVGPRs(TargetOccupancy);
  } else {
    SGPRCriticalLimit = SRI->getRegPressureSetLimit(DAG->MF,
                                                    SRI->getSGPRPressureSet());
    VGPRCriticalLimit = SRI->getRegPressureSetLimit(DAG->MF,
                                                    SRI->getVGPRPressureSet());
  }

  SGPRCriticalLimit -= ErrorMargin;
  VGPRCriticalLimit -= ErrorMargin;
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
      if (BotCand.Reason > TopCand.Reason) {
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

GCNScheduleDAGMILive::GCNScheduleDAGMILive(MachineSchedContext *C,
                        std::unique_ptr<MachineSchedStrategy> S) :
  ScheduleDAGMILive(C, std::move(S)),
  ST(MF.getSubtarget<SISubtarget>()),
  MFI(*MF.getInfo<SIMachineFunctionInfo>()),
  StartingOccupancy(ST.getOccupancyWithLocalMemSize(MFI.getLDSSize(),
                                                    *MF.getFunction())),
  MinOccupancy(StartingOccupancy), Stage(0) {

  DEBUG(dbgs() << "Starting occupancy is " << StartingOccupancy << ".\n");
}

void GCNScheduleDAGMILive::enterRegion(MachineBasicBlock *bb,
                                       MachineBasicBlock::iterator begin,
                                       MachineBasicBlock::iterator end,
                                       unsigned regioninstrs) {
  ScheduleDAGMILive::enterRegion(bb, begin, end, regioninstrs);

  if (Stage == 0)
    Regions.push_back(std::make_pair(begin, end));
}

void GCNScheduleDAGMILive::schedule() {
  std::vector<MachineInstr*> Unsched;
  Unsched.reserve(NumRegionInstrs);
  for (auto &I : *this)
    Unsched.push_back(&I);

  std::pair<unsigned, unsigned> PressureBefore;
  if (LIS) {
    DEBUG(dbgs() << "Pressure before scheduling:\n");
    discoverLiveIns();
    PressureBefore = getRealRegPressure();
  }

  ScheduleDAGMILive::schedule();
  if (!LIS)
    return;

  // Check the results of scheduling.
  GCNMaxOccupancySchedStrategy &S = (GCNMaxOccupancySchedStrategy&)*SchedImpl;
  DEBUG(dbgs() << "Pressure after scheduling:\n");
  auto PressureAfter = getRealRegPressure();
  LiveIns.clear();

  if (PressureAfter.first <= S.SGPRCriticalLimit &&
      PressureAfter.second <= S.VGPRCriticalLimit) {
    DEBUG(dbgs() << "Pressure in desired limits, done.\n");
    return;
  }
  unsigned WavesAfter = getMaxWaves(PressureAfter.first,
                                    PressureAfter.second, MF);
  unsigned WavesBefore = getMaxWaves(PressureBefore.first,
                                      PressureBefore.second, MF);
  DEBUG(dbgs() << "Occupancy before scheduling: " << WavesBefore <<
                  ", after " << WavesAfter << ".\n");

  // We could not keep current target occupancy because of the just scheduled
  // region. Record new occupancy for next scheduling cycle.
  unsigned NewOccupancy = std::max(WavesAfter, WavesBefore);
  if (NewOccupancy < MinOccupancy) {
    MinOccupancy = NewOccupancy;
    DEBUG(dbgs() << "Occupancy lowered for the function to "
                 << MinOccupancy << ".\n");
  }

  if (WavesAfter >= WavesBefore)
    return;

  DEBUG(dbgs() << "Attempting to revert scheduling.\n");
  RegionEnd = RegionBegin;
  for (MachineInstr *MI : Unsched) {
    if (MI->getIterator() != RegionEnd) {
      BB->remove(MI);
      BB->insert(RegionEnd, MI);
      LIS->handleMove(*MI, true);
    }
    // Reset read-undef flags and update them later.
    for (auto &Op : MI->operands())
      if (Op.isReg() && Op.isDef())
        Op.setIsUndef(false);
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
    RegionEnd = MI->getIterator();
    ++RegionEnd;
    DEBUG(dbgs() << "Scheduling " << *MI);
  }
  RegionBegin = Unsched.front()->getIterator();

  placeDebugValues();
}

static inline void setMask(const MachineRegisterInfo &MRI,
                           const SIRegisterInfo *SRI, unsigned Reg,
                           LaneBitmask &PrevMask, LaneBitmask NewMask,
                           unsigned &SGPRs, unsigned &VGPRs) {
  int NewRegs = countPopulation(NewMask.getAsInteger()) -
                countPopulation(PrevMask.getAsInteger());
  if (SRI->isSGPRReg(MRI, Reg))
    SGPRs += NewRegs;
  if (SRI->isVGPR(MRI, Reg))
    VGPRs += NewRegs;
  assert ((int)SGPRs >= 0 && (int)VGPRs >= 0);
  PrevMask = NewMask;
}

void GCNScheduleDAGMILive::discoverLiveIns() {
  unsigned SGPRs = 0;
  unsigned VGPRs = 0;

  const SIRegisterInfo *SRI = static_cast<const SIRegisterInfo*>(TRI);
  SlotIndex SI = LIS->getInstructionIndex(*begin()).getBaseIndex();
  assert (SI.isValid());

  DEBUG(dbgs() << "Region live-ins:");
  for (unsigned I = 0, E = MRI.getNumVirtRegs(); I != E; ++I) {
    unsigned Reg = TargetRegisterInfo::index2VirtReg(I);
    if (MRI.reg_nodbg_empty(Reg))
      continue;
    const LiveInterval &LI = LIS->getInterval(Reg);
    LaneBitmask LaneMask = LaneBitmask::getNone();
    if (LI.hasSubRanges()) {
      for (const auto &S : LI.subranges())
        if (S.liveAt(SI))
          LaneMask |= S.LaneMask;
    } else if (LI.liveAt(SI)) {
      LaneMask = MRI.getMaxLaneMaskForVReg(Reg);
    }

    if (LaneMask.any()) {
      setMask(MRI, SRI, Reg, LiveIns[Reg], LaneMask, SGPRs, VGPRs);

      DEBUG(dbgs() << ' ' << PrintVRegOrUnit(Reg, SRI) << ':'
                   << PrintLaneMask(LiveIns[Reg]));
    }
  }

  LiveInPressure = std::make_pair(SGPRs, VGPRs);

  DEBUG(dbgs() << "\nLive-in pressure:\nSGPR = " << SGPRs
               << "\nVGPR = " << VGPRs << '\n');
}

std::pair<unsigned, unsigned>
GCNScheduleDAGMILive::getRealRegPressure() const {
  unsigned SGPRs, MaxSGPRs, VGPRs, MaxVGPRs;
  SGPRs = MaxSGPRs = LiveInPressure.first;
  VGPRs = MaxVGPRs = LiveInPressure.second;

  const SIRegisterInfo *SRI = static_cast<const SIRegisterInfo*>(TRI);
  DenseMap<unsigned, LaneBitmask> LiveRegs(LiveIns);

  for (const MachineInstr &MI : *this) {
    if (MI.isDebugValue())
      continue;
    SlotIndex SI = LIS->getInstructionIndex(MI).getBaseIndex();
    assert (SI.isValid());

    // Remove dead registers or mask bits.
    for (auto &It : LiveRegs) {
      if (It.second.none())
        continue;
      const LiveInterval &LI = LIS->getInterval(It.first);
      if (LI.hasSubRanges()) {
        for (const auto &S : LI.subranges())
          if (!S.liveAt(SI))
            setMask(MRI, SRI, It.first, It.second, It.second & ~S.LaneMask,
                    SGPRs, VGPRs);
      } else if (!LI.liveAt(SI)) {
        setMask(MRI, SRI, It.first, It.second, LaneBitmask::getNone(),
                SGPRs, VGPRs);
      }
    }

    // Add new registers or mask bits.
    for (const auto &MO : MI.defs()) {
      if (!MO.isReg())
        continue;
      unsigned Reg = MO.getReg();
      if (!TargetRegisterInfo::isVirtualRegister(Reg))
        continue;
      unsigned SubRegIdx = MO.getSubReg();
      LaneBitmask LaneMask = SubRegIdx != 0
                             ? TRI->getSubRegIndexLaneMask(SubRegIdx)
                             : MRI.getMaxLaneMaskForVReg(Reg);
      LaneBitmask &LM = LiveRegs[Reg];
      setMask(MRI, SRI, Reg, LM, LM | LaneMask, SGPRs, VGPRs);
    }
    MaxSGPRs = std::max(MaxSGPRs, SGPRs);
    MaxVGPRs = std::max(MaxVGPRs, VGPRs);
  }

  DEBUG(dbgs() << "Real region's register pressure:\nSGPR = " << MaxSGPRs
               << "\nVGPR = " << MaxVGPRs << '\n');

  return std::make_pair(MaxSGPRs, MaxVGPRs);
}

void GCNScheduleDAGMILive::finalizeSchedule() {
  // Retry function scheduling if we found resulting occupancy and it is
  // lower than used for first pass scheduling. This will give more freedom
  // to schedule low register pressure blocks.
  // Code is partially copied from MachineSchedulerBase::scheduleRegions().

  if (!LIS || StartingOccupancy <= MinOccupancy)
    return;

  DEBUG(dbgs() << "Retrying function scheduling with lowest recorded occupancy "
               << MinOccupancy << ".\n");

  Stage++;
  GCNMaxOccupancySchedStrategy &S = (GCNMaxOccupancySchedStrategy&)*SchedImpl;
  S.TargetOccupancy = MinOccupancy;

  MachineBasicBlock *MBB = nullptr;
  for (auto Region : Regions) {
    RegionBegin = Region.first;
    RegionEnd = Region.second;

    if (RegionBegin->getParent() != MBB) {
      if (MBB) finishBlock();
      MBB = RegionBegin->getParent();
      startBlock(MBB);
    }

    unsigned NumRegionInstrs = std::distance(begin(), end());
    enterRegion(MBB, begin(), end(), NumRegionInstrs);

    // Skip empty scheduling regions (0 or 1 schedulable instructions).
    if (begin() == end() || begin() == std::prev(end())) {
      exitRegion();
      continue;
    }
    DEBUG(dbgs() << "********** MI Scheduling **********\n");
    DEBUG(dbgs() << MF.getName()
          << ":BB#" << MBB->getNumber() << " " << MBB->getName()
          << "\n  From: " << *begin() << "    To: ";
          if (RegionEnd != MBB->end()) dbgs() << *RegionEnd;
          else dbgs() << "End";
          dbgs() << " RegionInstrs: " << NumRegionInstrs << '\n');

    schedule();

    exitRegion();
  }
  finishBlock();
  LiveIns.shrink_and_clear();
}
