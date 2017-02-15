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

class SIRegisterInfo;

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

public:
  GCNMaxOccupancySchedStrategy(const MachineSchedContext *C);

  SUnit *pickNode(bool &IsTopNode) override;

  void initialize(ScheduleDAGMI *DAG) override;
};

class GCNScheduleDAGMILive : public ScheduleDAGMILive {
public:
  GCNScheduleDAGMILive(MachineSchedContext *C,
                       std::unique_ptr<MachineSchedStrategy> S) :
    ScheduleDAGMILive(C, std::move(S)) {}

  void schedule() override;
};

} // End namespace llvm

#endif // GCNSCHEDSTRATEGY_H
