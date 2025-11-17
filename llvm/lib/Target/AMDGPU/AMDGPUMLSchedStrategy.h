//===-- AMDGPUMLSchedStrategy.h - ML-focused Scheduler Strategy -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// ML-focused scheduling strategy for AMDGPU.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUMLSCHEDSTRATEGY_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUMLSCHEDSTRATEGY_H

#include "GCNSchedStrategy.h"
#include "llvm/CodeGen/MachineScheduler.h"

namespace llvm {

class AMDGPUMLSchedStrategy final : public GCNSchedStrategy {
protected:
  bool tryCandidate(SchedCandidate &Cand, SchedCandidate &TryCand,
                    SchedBoundary *Zone) const override;

public:
  AMDGPUMLSchedStrategy(const MachineSchedContext *C);

  void initialize(ScheduleDAGMI *DAG) override;
};

class AMDGPUMLPostSchedStrategy : public PostGenericScheduler {
protected:
  bool tryCandidate(SchedCandidate &Cand, SchedCandidate &TryCand) override;

public:
  AMDGPUMLPostSchedStrategy(const MachineSchedContext *C);
};

} // End namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPUMLSCHEDSTRATEGY_H