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

#include "RISCVSubtarget.h"
#include "RISCVVSETVLIInfoAnalysis.h"
#include "llvm/CodeGen/MachineScheduler.h"

namespace llvm {

/// A GenericScheduler implementation for RISCV pre RA scheduling.
class RISCVPreRAMachineSchedStrategy : public GenericScheduler {
  const RISCVSubtarget *ST;
  RISCV::RISCVVSETVLIInfoAnalysis VIA;
  RISCV::VSETVLIInfo TopInfo;
  RISCV::VSETVLIInfo BottomInfo;

  RISCV::VSETVLIInfo getVSETVLIInfo(const MachineInstr *MI) const;
  bool tryVSETVLIInfo(const RISCV::VSETVLIInfo &TryInfo,
                      const RISCV::VSETVLIInfo &CandInfo,
                      SchedCandidate &TryCand, SchedCandidate &Cand,
                      CandReason Reason) const;

public:
  RISCVPreRAMachineSchedStrategy(const MachineSchedContext *C)
      : GenericScheduler(C), ST(&C->MF->getSubtarget<RISCVSubtarget>()),
        VIA(ST, C->LIS) {}

protected:
  bool tryCandidate(SchedCandidate &Cand, SchedCandidate &TryCand,
                    SchedBoundary *Zone) const override;
  void enterMBB(MachineBasicBlock *MBB) override;
  void leaveMBB() override;
  void schedNode(SUnit *SU, bool IsTopNode) override;
};

} // end namespace llvm

#endif
