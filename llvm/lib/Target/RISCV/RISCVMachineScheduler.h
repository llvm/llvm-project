//===--- RISCVMachineScheduler.h - Custom RISC-V MI scheduler ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Custom RISC-V MI scheduler.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_RISCV_RISCVMACHINESCHEDULER_H
#define LLVM_LIB_TARGET_RISCV_RISCVMACHINESCHEDULER_H

#include "llvm/CodeGen/MachineScheduler.h"
#include "llvm/TargetParser/RISCVTargetParser.h"

namespace llvm {

// TODO: We should use the infrastructure in RISCV/RISCVInsertVSETVLI.cpp.
// TODO: We should take vl into consideration.
using VTypeInfo = std::pair<RISCVVType::VLMUL, unsigned>;

/// A GenericScheduler implementation for RISCV pre RA scheduling.
class RISCVPreRAMachineSchedStrategy : public GenericScheduler {
private:
  VTypeInfo TopVType;
  VTypeInfo BottomVType;

  bool tryVType(VTypeInfo TryVType, VTypeInfo CandVtype,
                GenericSchedulerBase::SchedCandidate &TryCand,
                GenericSchedulerBase::SchedCandidate &Cand,
                GenericSchedulerBase::CandReason Reason) const;

public:
  RISCVPreRAMachineSchedStrategy(const MachineSchedContext *C)
      : GenericScheduler(C) {}

protected:
  bool tryCandidate(SchedCandidate &Cand, SchedCandidate &TryCand,
                    SchedBoundary *Zone) const override;
  void enterMBB(MachineBasicBlock *MBB) override;
  void leaveMBB() override;
  void schedNode(SUnit *SU, bool IsTopNode) override;
};

} // end namespace llvm

#endif
