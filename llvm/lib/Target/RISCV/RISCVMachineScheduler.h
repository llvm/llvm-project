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

/// A GenericScheduler implementation for RISCV pre RA scheduling.
class RISCVPreRAMachineSchedStrategy : public GenericScheduler {
private:
  RISCVII::VLMUL PrevVLMUL;
  unsigned PrevVSEW;

public:
  RISCVPreRAMachineSchedStrategy(const MachineSchedContext *C)
      : GenericScheduler(C) {}

protected:
  SUnit *pickNode(bool &IsTopNode) override;

  bool tryCandidate(SchedCandidate &Cand, SchedCandidate &TryCand,
                    SchedBoundary *Zone) const override;

  void schedNode(SUnit *SU, bool IsTopNode) override;
};

} // end namespace llvm

#endif
