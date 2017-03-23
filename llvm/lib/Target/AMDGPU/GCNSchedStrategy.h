//===-- GCNSchedStrategy.h - GCN Scheduler Strategy -*- C++ -*-------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_GCNSCHEDSTRATEGY_H
#define LLVM_LIB_TARGET_AMDGPU_GCNSCHEDSTRATEGY_H

#include "llvm/CodeGen/MachineScheduler.h"

namespace llvm {

class SIMachineFunctionInfo;
class SIRegisterInfo;
class SISubtarget;

/// This is a minimal scheduler strategy.  The main difference between this
/// and the GenericScheduler is that GCNSchedStrategy uses different
/// heuristics to determine excess/critical pressure sets.  Its goal is to
/// maximize kernel occupancy (i.e. maximum number of waves per simd).
class GCNMaxOccupancySchedStrategy : public GenericScheduler {
  friend class GCNScheduleDAGMILive;

  SUnit *pickNodeBidirectional(bool &IsTopNode);

  void pickNodeFromQueue(SchedBoundary &Zone, const CandPolicy &ZonePolicy,
                         const RegPressureTracker &RPTracker,
                         SchedCandidate &Cand);

  void initCandidate(SchedCandidate &Cand, SUnit *SU,
                     bool AtTop, const RegPressureTracker &RPTracker,
                     const SIRegisterInfo *SRI,
                     unsigned SGPRPressure, unsigned VGPRPressure);

  unsigned SGPRExcessLimit;
  unsigned VGPRExcessLimit;
  unsigned SGPRCriticalLimit;
  unsigned VGPRCriticalLimit;

  unsigned TargetOccupancy;

  MachineFunction *MF;

public:
  GCNMaxOccupancySchedStrategy(const MachineSchedContext *C);

  SUnit *pickNode(bool &IsTopNode) override;

  void initialize(ScheduleDAGMI *DAG) override;

  void setTargetOccupancy(unsigned Occ) { TargetOccupancy = Occ; }
};

class GCNScheduleDAGMILive : public ScheduleDAGMILive {

  const SISubtarget &ST;

  const SIMachineFunctionInfo &MFI;

  // Occupancy target at the begining of function scheduling cycle.
  unsigned StartingOccupancy;

  // Minimal real occupancy recorder for the function.
  unsigned MinOccupancy;

  // Scheduling stage number.
  unsigned Stage;

  // Vecor of regions recorder for later rescheduling
  SmallVector<std::pair<const MachineBasicBlock::iterator,
                        const MachineBasicBlock::iterator>, 32> Regions;

  // Region live-ins.
  DenseMap<unsigned, LaneBitmask> LiveIns;

  // Number of live-ins to the current region, first SGPR then VGPR.
  std::pair<unsigned, unsigned> LiveInPressure;

  // Collect current region live-ins.
  void discoverLiveIns();

  // Return current region pressure. First value is SGPR number, second is VGPR.
  std::pair<unsigned, unsigned> getRealRegPressure() const;

public:
  GCNScheduleDAGMILive(MachineSchedContext *C,
                       std::unique_ptr<MachineSchedStrategy> S);

  void enterRegion(MachineBasicBlock *bb,
                   MachineBasicBlock::iterator begin,
                   MachineBasicBlock::iterator end,
                   unsigned regioninstrs) override;

  void schedule() override;

  void finalizeSchedule() override;
};

} // End namespace llvm

#endif // GCNSCHEDSTRATEGY_H
