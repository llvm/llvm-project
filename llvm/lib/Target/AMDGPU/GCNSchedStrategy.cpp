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
//===----------------------------------------------------------------------===//

#include "GCNSchedStrategy.h"
#include "SIMachineFunctionInfo.h"
#include "llvm/CodeGen/RegisterClassInfo.h"

#define DEBUG_TYPE "machine-scheduler"

using namespace llvm;

GCNMaxOccupancySchedStrategy::GCNMaxOccupancySchedStrategy(
    const MachineSchedContext *C) :
    GenericScheduler(C), TargetOccupancy(0), HasClusteredNodes(false),
    HasExcessPressure(false), MF(nullptr) { }

void GCNMaxOccupancySchedStrategy::initialize(ScheduleDAGMI *DAG) {
  GenericScheduler::initialize(DAG);

  MF = &DAG->MF;

  const GCNSubtarget &ST = MF->getSubtarget<GCNSubtarget>();

  // FIXME: This is also necessary, because some passes that run after
  // scheduling and before regalloc increase register pressure.
  const unsigned ErrorMargin = 3;

  SGPRExcessLimit =
      Context->RegClassInfo->getNumAllocatableRegs(&AMDGPU::SGPR_32RegClass);
  VGPRExcessLimit =
      Context->RegClassInfo->getNumAllocatableRegs(&AMDGPU::VGPR_32RegClass);

  SIMachineFunctionInfo &MFI = *MF->getInfo<SIMachineFunctionInfo>();
  // Set the initial TargetOccupnacy to the maximum occupancy that we can
  // achieve for this function. This effectively sets a lower bound on the
  // 'Critical' register limits in the scheduler.
  TargetOccupancy = MFI.getOccupancy();
  SGPRCriticalLimit =
      std::min(ST.getMaxNumSGPRs(TargetOccupancy, true), SGPRExcessLimit);
  VGPRCriticalLimit =
      std::min(ST.getMaxNumVGPRs(TargetOccupancy), VGPRExcessLimit);

  // Subtract error margin from register limits and avoid overflow.
  SGPRCriticalLimit =
      std::min(SGPRCriticalLimit - ErrorMargin, SGPRCriticalLimit);
  VGPRCriticalLimit =
      std::min(VGPRCriticalLimit - ErrorMargin, VGPRCriticalLimit);
  SGPRExcessLimit = std::min(SGPRExcessLimit - ErrorMargin, SGPRExcessLimit);
  VGPRExcessLimit = std::min(VGPRExcessLimit - ErrorMargin, VGPRExcessLimit);
}

void GCNMaxOccupancySchedStrategy::initCandidate(SchedCandidate &Cand, SUnit *SU,
                                     bool AtTop, const RegPressureTracker &RPTracker,
                                     const SIRegisterInfo *SRI,
                                     unsigned SGPRPressure,
                                     unsigned VGPRPressure) {

  Cand.SU = SU;
  Cand.AtTop = AtTop;

  // getDownwardPressure() and getUpwardPressure() make temporary changes to
  // the tracker, so we need to pass those function a non-const copy.
  RegPressureTracker &TempTracker = const_cast<RegPressureTracker&>(RPTracker);

  Pressure.clear();
  MaxPressure.clear();

  if (AtTop)
    TempTracker.getDownwardPressure(SU->getInstr(), Pressure, MaxPressure);
  else {
    // FIXME: I think for bottom up scheduling, the register pressure is cached
    // and can be retrieved by DAG->getPressureDif(SU).
    TempTracker.getUpwardPressure(SU->getInstr(), Pressure, MaxPressure);
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
    HasExcessPressure = true;
    Cand.RPDelta.Excess = PressureChange(AMDGPU::RegisterPressureSets::VGPR_32);
    Cand.RPDelta.Excess.setUnitInc(NewVGPRPressure - VGPRExcessLimit);
  }

  if (ShouldTrackSGPRs && NewSGPRPressure >= SGPRExcessLimit) {
    HasExcessPressure = true;
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
    HasExcessPressure = true;
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
void GCNMaxOccupancySchedStrategy::pickNodeFromQueue(SchedBoundary &Zone,
                                         const CandPolicy &ZonePolicy,
                                         const RegPressureTracker &RPTracker,
                                         SchedCandidate &Cand) {
  const SIRegisterInfo *SRI = static_cast<const SIRegisterInfo*>(TRI);
  ArrayRef<unsigned> Pressure = RPTracker.getRegSetPressureAtPos();
  unsigned SGPRPressure = Pressure[AMDGPU::RegisterPressureSets::SReg_32];
  unsigned VGPRPressure = Pressure[AMDGPU::RegisterPressureSets::VGPR_32];
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
      LLVM_DEBUG(traceCandidate(Cand));
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
  LLVM_DEBUG(dbgs() << "Picking from Bot:\n");
  if (!BotCand.isValid() || BotCand.SU->isScheduled ||
      BotCand.Policy != BotPolicy) {
    BotCand.reset(CandPolicy());
    pickNodeFromQueue(Bot, BotPolicy, DAG->getBotRPTracker(), BotCand);
    assert(BotCand.Reason != NoCand && "failed to find the first candidate");
  } else {
    LLVM_DEBUG(traceCandidate(BotCand));
#ifndef NDEBUG
    if (VerifyScheduling) {
      SchedCandidate TCand;
      TCand.reset(CandPolicy());
      pickNodeFromQueue(Bot, BotPolicy, DAG->getBotRPTracker(), TCand);
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
    pickNodeFromQueue(Top, TopPolicy, DAG->getTopRPTracker(), TopCand);
    assert(TopCand.Reason != NoCand && "failed to find the first candidate");
  } else {
    LLVM_DEBUG(traceCandidate(TopCand));
#ifndef NDEBUG
    if (VerifyScheduling) {
      SchedCandidate TCand;
      TCand.reset(CandPolicy());
      pickNodeFromQueue(Top, TopPolicy, DAG->getTopRPTracker(), TCand);
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
  GenericScheduler::tryCandidate(Cand, TopCand, nullptr);
  if (TopCand.Reason != NoCand) {
    Cand.setBest(TopCand);
  }
  LLVM_DEBUG(dbgs() << "Picking: "; traceCandidate(Cand););

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

  if (!HasClusteredNodes && SU->getInstr()->mayLoadOrStore()) {
    for (SDep &Dep : SU->Preds) {
      if (Dep.isCluster()) {
        HasClusteredNodes = true;
        break;
      }
    }
  }

  LLVM_DEBUG(dbgs() << "Scheduling SU(" << SU->NodeNum << ") "
                    << *SU->getInstr());
  return SU;
}

GCNScheduleDAGMILive::GCNScheduleDAGMILive(MachineSchedContext *C,
                        std::unique_ptr<MachineSchedStrategy> S) :
  ScheduleDAGMILive(C, std::move(S)),
  ST(MF.getSubtarget<GCNSubtarget>()),
  MFI(*MF.getInfo<SIMachineFunctionInfo>()),
  StartingOccupancy(MFI.getOccupancy()),
  MinOccupancy(StartingOccupancy), Stage(Collect), RegionIdx(0) {

  LLVM_DEBUG(dbgs() << "Starting occupancy is " << StartingOccupancy << ".\n");
}

void GCNScheduleDAGMILive::schedule() {
  if (Stage == Collect) {
    // Just record regions at the first pass.
    Regions.push_back(std::make_pair(RegionBegin, RegionEnd));
    return;
  }

  std::vector<MachineInstr*> Unsched;
  Unsched.reserve(NumRegionInstrs);
  for (auto &I : *this) {
    Unsched.push_back(&I);
  }

  GCNRegPressure PressureBefore;
  if (LIS) {
    PressureBefore = Pressure[RegionIdx];

    LLVM_DEBUG(dbgs() << "Pressure before scheduling:\nRegion live-ins:";
               GCNRPTracker::printLiveRegs(dbgs(), LiveIns[RegionIdx], MRI);
               dbgs() << "Region live-in pressure:  ";
               llvm::getRegPressure(MRI, LiveIns[RegionIdx]).print(dbgs());
               dbgs() << "Region register pressure: ";
               PressureBefore.print(dbgs()));
  }

  GCNMaxOccupancySchedStrategy &S = (GCNMaxOccupancySchedStrategy&)*SchedImpl;
  // Set HasClusteredNodes to true for late stages where we have already
  // collected it. That way pickNode() will not scan SDep's when not needed.
  S.HasClusteredNodes = Stage > InitialSchedule;
  S.HasExcessPressure = false;
  ScheduleDAGMILive::schedule();
  Regions[RegionIdx] = std::make_pair(RegionBegin, RegionEnd);
  RescheduleRegions[RegionIdx] = false;
  if (Stage == InitialSchedule && S.HasClusteredNodes)
    RegionsWithClusters[RegionIdx] = true;
  if (S.HasExcessPressure)
    RegionsWithHighRP[RegionIdx] = true;

  if (!LIS)
    return;

  // Check the results of scheduling.
  auto PressureAfter = getRealRegPressure();

  LLVM_DEBUG(dbgs() << "Pressure after scheduling: ";
             PressureAfter.print(dbgs()));

  if (PressureAfter.getSGPRNum() <= S.SGPRCriticalLimit &&
      PressureAfter.getVGPRNum(ST.hasGFX90AInsts()) <= S.VGPRCriticalLimit) {
    Pressure[RegionIdx] = PressureAfter;
    RegionsWithMinOcc[RegionIdx] =
        PressureAfter.getOccupancy(ST) == MinOccupancy;

    LLVM_DEBUG(dbgs() << "Pressure in desired limits, done.\n");
    return;
  }

  unsigned WavesAfter =
      std::min(S.TargetOccupancy, PressureAfter.getOccupancy(ST));
  unsigned WavesBefore =
      std::min(S.TargetOccupancy, PressureBefore.getOccupancy(ST));
  LLVM_DEBUG(dbgs() << "Occupancy before scheduling: " << WavesBefore
                    << ", after " << WavesAfter << ".\n");

  // We may not be able to keep the current target occupancy because of the just
  // scheduled region. We might still be able to revert scheduling if the
  // occupancy before was higher, or if the current schedule has register
  // pressure higher than the excess limits which could lead to more spilling.
  unsigned NewOccupancy = std::max(WavesAfter, WavesBefore);

  // Allow memory bound functions to drop to 4 waves if not limited by an
  // attribute.
  if (WavesAfter < WavesBefore && WavesAfter < MinOccupancy &&
      WavesAfter >= MFI.getMinAllowedOccupancy()) {
    LLVM_DEBUG(dbgs() << "Function is memory bound, allow occupancy drop up to "
                      << MFI.getMinAllowedOccupancy() << " waves\n");
    NewOccupancy = WavesAfter;
  }

  if (NewOccupancy < MinOccupancy) {
    MinOccupancy = NewOccupancy;
    MFI.limitOccupancy(MinOccupancy);
    RegionsWithMinOcc.reset();
    LLVM_DEBUG(dbgs() << "Occupancy lowered for the function to "
                      << MinOccupancy << ".\n");
  }

  unsigned MaxVGPRs = ST.getMaxNumVGPRs(MF);
  unsigned MaxSGPRs = ST.getMaxNumSGPRs(MF);
  if (PressureAfter.getVGPRNum(false) > MaxVGPRs ||
      PressureAfter.getAGPRNum() > MaxVGPRs ||
      PressureAfter.getSGPRNum() > MaxSGPRs) {
    RescheduleRegions[RegionIdx] = true;
    RegionsWithHighRP[RegionIdx] = true;
  }

  // If this condition is true, then either the occupancy before and after
  // scheduling is the same, or we are allowing the occupancy to drop because
  // the function is memory bound. Even if we are OK with the current occupancy,
  // we still need to verify that we will not introduce any extra chance of
  // spilling.
  if (WavesAfter >= MinOccupancy) {
    if (Stage == UnclusteredReschedule &&
        !PressureAfter.less(ST, PressureBefore)) {
      LLVM_DEBUG(dbgs() << "Unclustered reschedule did not help.\n");
    } else if (WavesAfter > MFI.getMinWavesPerEU() ||
        PressureAfter.less(ST, PressureBefore) ||
        !RescheduleRegions[RegionIdx]) {
      Pressure[RegionIdx] = PressureAfter;
      RegionsWithMinOcc[RegionIdx] =
          PressureAfter.getOccupancy(ST) == MinOccupancy;
      if (!RegionsWithClusters[RegionIdx] &&
          (Stage + 1) == UnclusteredReschedule)
        RescheduleRegions[RegionIdx] = false;
      return;
    } else {
      LLVM_DEBUG(dbgs() << "New pressure will result in more spilling.\n");
    }
  }

  RegionsWithMinOcc[RegionIdx] =
      PressureBefore.getOccupancy(ST) == MinOccupancy;
  LLVM_DEBUG(dbgs() << "Attempting to revert scheduling.\n");
  RescheduleRegions[RegionIdx] = RegionsWithClusters[RegionIdx] ||
                                 (Stage + 1) != UnclusteredReschedule;
  RegionEnd = RegionBegin;
  int SkippedDebugInstr = 0;
  for (MachineInstr *MI : Unsched) {
    if (MI->isDebugInstr()) {
      ++SkippedDebugInstr;
      continue;
    }

    if (MI->getIterator() != RegionEnd) {
      BB->remove(MI);
      BB->insert(RegionEnd, MI);
      if (!MI->isDebugInstr())
        LIS->handleMove(*MI, true);
    }
    // Reset read-undef flags and update them later.
    for (auto &Op : MI->operands())
      if (Op.isReg() && Op.isDef())
        Op.setIsUndef(false);
    RegisterOperands RegOpers;
    RegOpers.collect(*MI, *TRI, MRI, ShouldTrackLaneMasks, false);
    if (!MI->isDebugInstr()) {
      if (ShouldTrackLaneMasks) {
        // Adjust liveness and add missing dead+read-undef flags.
        SlotIndex SlotIdx = LIS->getInstructionIndex(*MI).getRegSlot();
        RegOpers.adjustLaneLiveness(*LIS, MRI, SlotIdx, MI);
      } else {
        // Adjust for missing dead-def flags.
        RegOpers.detectDeadDefs(*MI, *LIS);
      }
    }
    RegionEnd = MI->getIterator();
    ++RegionEnd;
    LLVM_DEBUG(dbgs() << "Scheduling " << *MI);
  }

  // After reverting schedule, debug instrs will now be at the end of the block
  // and RegionEnd will point to the first debug instr. Increment RegionEnd
  // pass debug instrs to the actual end of the scheduling region.
  while (SkippedDebugInstr-- > 0)
    ++RegionEnd;

  // If Unsched.front() instruction is a debug instruction, this will actually
  // shrink the region since we moved all debug instructions to the end of the
  // block. Find the first instruction that is not a debug instruction.
  RegionBegin = Unsched.front()->getIterator();
  if (RegionBegin->isDebugInstr()) {
    for (MachineInstr *MI : Unsched) {
      if (MI->isDebugInstr())
        continue;
      RegionBegin = MI->getIterator();
      break;
    }
  }

  // Then move the debug instructions back into their correct place and set
  // RegionBegin and RegionEnd if needed.
  placeDebugValues();

  Regions[RegionIdx] = std::make_pair(RegionBegin, RegionEnd);
}

GCNRegPressure GCNScheduleDAGMILive::getRealRegPressure() const {
  GCNDownwardRPTracker RPTracker(*LIS);
  RPTracker.advance(begin(), end(), &LiveIns[RegionIdx]);
  return RPTracker.moveMaxPressure();
}

void GCNScheduleDAGMILive::computeBlockPressure(const MachineBasicBlock *MBB) {
  GCNDownwardRPTracker RPTracker(*LIS);

  // If the block has the only successor then live-ins of that successor are
  // live-outs of the current block. We can reuse calculated live set if the
  // successor will be sent to scheduling past current block.
  const MachineBasicBlock *OnlySucc = nullptr;
  if (MBB->succ_size() == 1 && !(*MBB->succ_begin())->empty()) {
    SlotIndexes *Ind = LIS->getSlotIndexes();
    if (Ind->getMBBStartIdx(MBB) < Ind->getMBBStartIdx(*MBB->succ_begin()))
      OnlySucc = *MBB->succ_begin();
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

  for ( ; ; ) {
    I = RPTracker.getNext();

    if (Regions[CurRegion].first == I || NonDbgMI == I) {
      LiveIns[CurRegion] = RPTracker.getLiveRegs();
      RPTracker.clearMaxPressure();
    }

    if (Regions[CurRegion].second == I) {
      Pressure[CurRegion] = RPTracker.moveMaxPressure();
      if (CurRegion-- == RegionIdx)
        break;
    }
    RPTracker.advanceToNext();
    RPTracker.advanceBeforeNext();
  }

  if (OnlySucc) {
    if (I != MBB->end()) {
      RPTracker.advanceToNext();
      RPTracker.advance(MBB->end());
    }
    RPTracker.reset(*OnlySucc->begin(), &RPTracker.getLiveRegs());
    RPTracker.advanceBeforeNext();
    MBBLiveIns[OnlySucc] = RPTracker.moveLiveRegs();
  }
}

DenseMap<MachineInstr *, GCNRPTracker::LiveRegSet>
GCNScheduleDAGMILive::getBBLiveInMap() const {
  assert(!Regions.empty());
  std::vector<MachineInstr *> BBStarters;
  BBStarters.reserve(Regions.size());
  auto I = Regions.rbegin(), E = Regions.rend();
  auto *BB = I->first->getParent();
  do {
    auto *MI = &*skipDebugInstructionsForward(I->first, I->second);
    BBStarters.push_back(MI);
    do {
      ++I;
    } while (I != E && I->first->getParent() == BB);
  } while (I != E);
  return getLiveRegMap(BBStarters, false /*After*/, *LIS);
}

void GCNScheduleDAGMILive::finalizeSchedule() {
  LLVM_DEBUG(dbgs() << "All regions recorded, starting actual scheduling.\n");

  LiveIns.resize(Regions.size());
  Pressure.resize(Regions.size());
  RescheduleRegions.resize(Regions.size());
  RegionsWithClusters.resize(Regions.size());
  RegionsWithHighRP.resize(Regions.size());
  RegionsWithMinOcc.resize(Regions.size());
  RescheduleRegions.set();
  RegionsWithClusters.reset();
  RegionsWithHighRP.reset();
  RegionsWithMinOcc.reset();

  if (!Regions.empty())
    BBLiveInMap = getBBLiveInMap();

  std::vector<std::unique_ptr<ScheduleDAGMutation>> SavedMutations;

  do {
    Stage++;
    RegionIdx = 0;
    MachineBasicBlock *MBB = nullptr;

    if (Stage > InitialSchedule) {
      if (!LIS)
        break;

      // Retry function scheduling if we found resulting occupancy and it is
      // lower than used for first pass scheduling. This will give more freedom
      // to schedule low register pressure blocks.
      // Code is partially copied from MachineSchedulerBase::scheduleRegions().

      if (Stage == UnclusteredReschedule) {
        if (RescheduleRegions.none())
          continue;
        LLVM_DEBUG(dbgs() <<
          "Retrying function scheduling without clustering.\n");
      }

      if (Stage == ClusteredLowOccupancyReschedule) {
        if (StartingOccupancy <= MinOccupancy)
          break;

        LLVM_DEBUG(
            dbgs()
            << "Retrying function scheduling with lowest recorded occupancy "
            << MinOccupancy << ".\n");
      }

      if (Stage == PreRARematerialize) {
        if (RegionsWithMinOcc.none() || Regions.size() == 1)
          break;

        const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
        const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
        // Check maximum occupancy
        if (ST.computeOccupancy(MF.getFunction(), MFI.getLDSSize()) ==
            MinOccupancy)
          break;

        // FIXME: This pass will invalidate cached MBBLiveIns for regions
        // inbetween the defs and region we sinked the def to. Cached pressure
        // for regions where a def is sinked from will also be invalidated. Will
        // need to be fixed if there is another pass after this pass.
        static_assert(LastStage == PreRARematerialize,
                      "Passes after PreRARematerialize are not supported");

        collectRematerializableInstructions();
        if (RematerializableInsts.empty() || !sinkTriviallyRematInsts(ST, TII))
          break;

        LLVM_DEBUG(
            dbgs() << "Retrying function scheduling with improved occupancy of "
                   << MinOccupancy << " from rematerializing\n");
      }
    }

    if (Stage == UnclusteredReschedule)
      SavedMutations.swap(Mutations);

    for (auto Region : Regions) {
      if (((Stage == UnclusteredReschedule || Stage == PreRARematerialize) &&
           !RescheduleRegions[RegionIdx]) ||
          (Stage == ClusteredLowOccupancyReschedule &&
           !RegionsWithClusters[RegionIdx] && !RegionsWithHighRP[RegionIdx])) {

        ++RegionIdx;
        continue;
      }

      RegionBegin = Region.first;
      RegionEnd = Region.second;

      if (RegionBegin->getParent() != MBB) {
        if (MBB) finishBlock();
        MBB = RegionBegin->getParent();
        startBlock(MBB);
        if (Stage == InitialSchedule)
          computeBlockPressure(MBB);
      }

      unsigned NumRegionInstrs = std::distance(begin(), end());
      enterRegion(MBB, begin(), end(), NumRegionInstrs);

      // Skip empty scheduling regions (0 or 1 schedulable instructions).
      if (begin() == end() || begin() == std::prev(end())) {
        exitRegion();
        ++RegionIdx;
        continue;
      }

      LLVM_DEBUG(dbgs() << "********** MI Scheduling **********\n");
      LLVM_DEBUG(dbgs() << MF.getName() << ":" << printMBBReference(*MBB) << " "
                        << MBB->getName() << "\n  From: " << *begin()
                        << "    To: ";
                 if (RegionEnd != MBB->end()) dbgs() << *RegionEnd;
                 else dbgs() << "End";
                 dbgs() << " RegionInstrs: " << NumRegionInstrs << '\n');

      schedule();

      exitRegion();
      ++RegionIdx;
    }
    finishBlock();

    if (Stage == UnclusteredReschedule)
      SavedMutations.swap(Mutations);
  } while (Stage != LastStage);
}

void GCNScheduleDAGMILive::collectRematerializableInstructions() {
  const SIRegisterInfo *SRI = static_cast<const SIRegisterInfo *>(TRI);
  for (unsigned I = 0, E = MRI.getNumVirtRegs(); I != E; ++I) {
    Register Reg = Register::index2VirtReg(I);
    if (!LIS->hasInterval(Reg))
      continue;

    // TODO: Handle AGPR and SGPR rematerialization
    if (!SRI->isVGPRClass(MRI.getRegClass(Reg)) || !MRI.hasOneDef(Reg) ||
        !MRI.hasOneNonDBGUse(Reg))
      continue;

    MachineOperand *Op = MRI.getOneDef(Reg);
    MachineInstr *Def = Op->getParent();
    if (Op->getSubReg() != 0 || !isTriviallyReMaterializable(*Def, AA))
      continue;

    MachineInstr *UseI = &*MRI.use_instr_nodbg_begin(Reg);
    if (Def->getParent() == UseI->getParent())
      continue;

    // We are only collecting defs that are defined in another block and are
    // live-through or used inside regions at MinOccupancy. This means that the
    // register must be in the live-in set for the region.
    bool AddedToRematList = false;
    for (unsigned I = 0, E = Regions.size(); I != E; ++I) {
      auto It = LiveIns[I].find(Reg);
      if (It != LiveIns[I].end() && !It->second.none()) {
        if (RegionsWithMinOcc[I]) {
          RematerializableInsts[I][Def] = UseI;
          AddedToRematList = true;
        }

        // Collect regions with rematerializable reg as live-in to avoid
        // searching later when updating RP.
        RematDefToLiveInRegions[Def].push_back(I);
      }
    }
    if (!AddedToRematList)
      RematDefToLiveInRegions.erase(Def);
  }
}

bool GCNScheduleDAGMILive::sinkTriviallyRematInsts(const GCNSubtarget &ST,
                                                   const TargetInstrInfo *TII) {
  // Temporary copies of cached variables we will be modifying and replacing if
  // sinking succeeds.
  SmallVector<
      std::pair<MachineBasicBlock::iterator, MachineBasicBlock::iterator>, 32>
      NewRegions;
  DenseMap<unsigned, GCNRPTracker::LiveRegSet> NewLiveIns;
  DenseMap<unsigned, GCNRegPressure> NewPressure;
  BitVector NewRescheduleRegions;

  NewRegions.resize(Regions.size());
  NewRescheduleRegions.resize(Regions.size());

  // Collect only regions that has a rematerializable def as a live-in.
  SmallSet<unsigned, 16> ImpactedRegions;
  for (const auto &It : RematDefToLiveInRegions)
    ImpactedRegions.insert(It.second.begin(), It.second.end());

  // Make copies of register pressure and live-ins cache that will be updated
  // as we rematerialize.
  for (auto Idx : ImpactedRegions) {
    NewPressure[Idx] = Pressure[Idx];
    NewLiveIns[Idx] = LiveIns[Idx];
  }
  NewRegions = Regions;
  NewRescheduleRegions.reset();

  DenseMap<MachineInstr *, MachineInstr *> InsertedMIToOldDef;
  bool Improved = false;
  for (auto I : ImpactedRegions) {
    if (!RegionsWithMinOcc[I])
      continue;

    Improved = false;
    int VGPRUsage = NewPressure[I].getVGPRNum(ST.hasGFX90AInsts());
    int SGPRUsage = NewPressure[I].getSGPRNum();

    // TODO: Handle occupancy drop due to AGPR and SGPR.
    // Check if cause of occupancy drop is due to VGPR usage and not SGPR.
    if (ST.getOccupancyWithNumSGPRs(SGPRUsage) == MinOccupancy)
      break;

    // The occupancy of this region could have been improved by a previous
    // iteration's sinking of defs.
    if (NewPressure[I].getOccupancy(ST) > MinOccupancy) {
      NewRescheduleRegions[I] = true;
      Improved = true;
      continue;
    }

    // First check if we have enough trivially rematerializable instructions to
    // improve occupancy. Optimistically assume all instructions we are able to
    // sink decreased RP.
    int TotalSinkableRegs = 0;
    for (const auto &It : RematerializableInsts[I]) {
      MachineInstr *Def = It.first;
      Register DefReg = Def->getOperand(0).getReg();
      TotalSinkableRegs +=
          SIRegisterInfo::getNumCoveredRegs(NewLiveIns[I][DefReg]);
    }
    int VGPRsAfterSink = VGPRUsage - TotalSinkableRegs;
    unsigned OptimisticOccupancy = ST.getOccupancyWithNumVGPRs(VGPRsAfterSink);
    // If in the most optimistic scenario, we cannot improve occupancy, then do
    // not attempt to sink any instructions.
    if (OptimisticOccupancy <= MinOccupancy)
      break;

    unsigned ImproveOccupancy = 0;
    SmallVector<MachineInstr *, 4> SinkedDefs;
    for (auto &It : RematerializableInsts[I]) {
      MachineInstr *Def = It.first;
      MachineBasicBlock::iterator InsertPos =
          MachineBasicBlock::iterator(It.second);
      Register Reg = Def->getOperand(0).getReg();
      // Rematerialize MI to its use block. Since we are only rematerializing
      // instructions that do not have any virtual reg uses, we do not need to
      // call LiveRangeEdit::allUsesAvailableAt() and
      // LiveRangeEdit::canRematerializeAt().
      TII->reMaterialize(*InsertPos->getParent(), InsertPos, Reg,
                         Def->getOperand(0).getSubReg(), *Def, *TRI);
      MachineInstr *NewMI = &*(--InsertPos);
      LIS->InsertMachineInstrInMaps(*NewMI);
      LIS->removeInterval(Reg);
      LIS->createAndComputeVirtRegInterval(Reg);
      InsertedMIToOldDef[NewMI] = Def;

      // Update region boundaries in scheduling region we sinked from since we
      // may sink an instruction that was at the beginning or end of its region
      updateRegionBoundaries(NewRegions, Def, /*NewMI =*/nullptr,
                             /*Removing =*/true);

      // Update region boundaries in region we sinked to.
      updateRegionBoundaries(NewRegions, InsertPos, NewMI);

      LaneBitmask PrevMask = NewLiveIns[I][Reg];
      // FIXME: Also update cached pressure for where the def was sinked from.
      // Update RP for all regions that has this reg as a live-in and remove
      // the reg from all regions as a live-in.
      for (auto Idx : RematDefToLiveInRegions[Def]) {
        NewLiveIns[Idx].erase(Reg);
        if (InsertPos->getParent() != Regions[Idx].first->getParent()) {
          // Def is live-through and not used in this block.
          NewPressure[Idx].inc(Reg, PrevMask, LaneBitmask::getNone(), MRI);
        } else {
          // Def is used and rematerialized into this block.
          GCNDownwardRPTracker RPT(*LIS);
          auto *NonDbgMI = &*skipDebugInstructionsForward(
              NewRegions[Idx].first, NewRegions[Idx].second);
          RPT.reset(*NonDbgMI, &NewLiveIns[Idx]);
          RPT.advance(NewRegions[Idx].second);
          NewPressure[Idx] = RPT.moveMaxPressure();
        }
      }

      SinkedDefs.push_back(Def);
      ImproveOccupancy = NewPressure[I].getOccupancy(ST);
      if (ImproveOccupancy > MinOccupancy)
        break;
    }

    // Remove defs we just sinked from all regions' list of sinkable defs
    for (auto &Def : SinkedDefs)
      for (auto TrackedIdx : RematDefToLiveInRegions[Def])
        RematerializableInsts[TrackedIdx].erase(Def);

    if (ImproveOccupancy <= MinOccupancy)
      break;

    NewRescheduleRegions[I] = true;
    Improved = true;
  }

  if (!Improved) {
    // Occupancy was not improved for all regions that were at MinOccupancy.
    // Undo sinking and remove newly rematerialized instructions.
    for (auto &Entry : InsertedMIToOldDef) {
      MachineInstr *MI = Entry.first;
      MachineInstr *OldMI = Entry.second;
      Register Reg = MI->getOperand(0).getReg();
      LIS->RemoveMachineInstrFromMaps(*MI);
      MI->eraseFromParent();
      OldMI->clearRegisterDeads(Reg);
      LIS->removeInterval(Reg);
      LIS->createAndComputeVirtRegInterval(Reg);
    }
    return false;
  }

  // Occupancy was improved for all regions.
  for (auto &Entry : InsertedMIToOldDef) {
    MachineInstr *MI = Entry.first;
    MachineInstr *OldMI = Entry.second;

    // Remove OldMI from BBLiveInMap since we are sinking it from its MBB.
    BBLiveInMap.erase(OldMI);

    // Remove OldMI and update LIS
    Register Reg = MI->getOperand(0).getReg();
    LIS->RemoveMachineInstrFromMaps(*OldMI);
    OldMI->eraseFromParent();
    LIS->removeInterval(Reg);
    LIS->createAndComputeVirtRegInterval(Reg);
  }

  // Update live-ins, register pressure, and regions caches.
  for (auto Idx : ImpactedRegions) {
    LiveIns[Idx] = NewLiveIns[Idx];
    Pressure[Idx] = NewPressure[Idx];
    MBBLiveIns.erase(Regions[Idx].first->getParent());
  }
  Regions = NewRegions;
  RescheduleRegions = NewRescheduleRegions;

  SIMachineFunctionInfo &MFI = *MF.getInfo<SIMachineFunctionInfo>();
  MFI.increaseOccupancy(MF, ++MinOccupancy);

  return true;
}

// Copied from MachineLICM
bool GCNScheduleDAGMILive::isTriviallyReMaterializable(const MachineInstr &MI,
                                                       AAResults *AA) {
  if (!TII->isTriviallyReMaterializable(MI, AA))
    return false;

  for (const MachineOperand &MO : MI.operands())
    if (MO.isReg() && MO.isUse() && MO.getReg().isVirtual())
      return false;

  return true;
}

// When removing, we will have to check both beginning and ending of the region.
// When inserting, we will only have to check if we are inserting NewMI in front
// of a scheduling region and do not need to check the ending since we will only
// ever be inserting before an already existing MI.
void GCNScheduleDAGMILive::updateRegionBoundaries(
    SmallVectorImpl<std::pair<MachineBasicBlock::iterator,
                              MachineBasicBlock::iterator>> &RegionBoundaries,
    MachineBasicBlock::iterator MI, MachineInstr *NewMI, bool Removing) {
  unsigned I = 0, E = RegionBoundaries.size();
  // Search for first region of the block where MI is located
  while (I != E && MI->getParent() != RegionBoundaries[I].first->getParent())
    ++I;

  for (; I != E; ++I) {
    if (MI->getParent() != RegionBoundaries[I].first->getParent())
      return;

    if (Removing && MI == RegionBoundaries[I].first &&
        MI == RegionBoundaries[I].second) {
      // MI is in a region with size 1, after removing, the region will be
      // size 0, set RegionBegin and RegionEnd to pass end of block iterator.
      RegionBoundaries[I] =
          std::make_pair(MI->getParent()->end(), MI->getParent()->end());
      return;
    }
    if (MI == RegionBoundaries[I].first) {
      if (Removing)
        RegionBoundaries[I] =
            std::make_pair(std::next(MI), RegionBoundaries[I].second);
      else
        // Inserted NewMI in front of region, set new RegionBegin to NewMI
        RegionBoundaries[I] = std::make_pair(MachineBasicBlock::iterator(NewMI),
                                             RegionBoundaries[I].second);
      return;
    }
    if (Removing && MI == RegionBoundaries[I].second) {
      RegionBoundaries[I] =
          std::make_pair(RegionBoundaries[I].first, std::prev(MI));
      return;
    }
  }
}
