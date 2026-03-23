//===- AMDGPUCoExecSchedStrategy.h - CoExec Scheduling Strategy -*- C++ -*-===//
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

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUCOEXECSCHEDSTRATEGY_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUCOEXECSCHEDSTRATEGY_H

#include "GCNSchedStrategy.h"
#include "llvm/CodeGen/MachineScheduler.h"

namespace llvm {

class AMDGPUCoExecSchedStrategy final : public GCNSchedStrategy {
protected:
  bool tryCandidate(SchedCandidate &Cand, SchedCandidate &TryCand,
                    SchedBoundary *Zone) const override;
  void pickNodeFromQueue(SchedBoundary &Zone, const CandPolicy &ZonePolicy,
                         const RegPressureTracker &RPTracker,
                         SchedCandidate &Cand, bool &PickedPending,
                         bool IsBottomUp);

public:
  AMDGPUCoExecSchedStrategy(const MachineSchedContext *C);

  void initPolicy(MachineBasicBlock::iterator Begin,
                  MachineBasicBlock::iterator End,
                  unsigned NumRegionInstrs) override;
  void initialize(ScheduleDAGMI *DAG) override;
  SUnit *pickNode(bool &IsTopNode) override;
};

ScheduleDAGInstrs *createGCNCoExecMachineScheduler(MachineSchedContext *C);
ScheduleDAGInstrs *createGCNNoopPostMachineScheduler(MachineSchedContext *C);

} // End namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPUCOEXECSCHEDSTRATEGY_H
