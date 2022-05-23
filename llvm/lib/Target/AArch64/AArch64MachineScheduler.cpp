//===- AArch64MachineScheduler.cpp - MI Scheduler for AArch64 -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AArch64MachineScheduler.h"
#include "MCTargetDesc/AArch64MCTargetDesc.h"

using namespace llvm;

bool AArch64PostRASchedStrategy::tryCandidate(SchedCandidate &Cand,
                                              SchedCandidate &TryCand) {
  bool OriginalResult = PostGenericScheduler::tryCandidate(Cand, TryCand);

  if (Cand.isValid()) {
    MachineInstr *Instr0 = TryCand.SU->getInstr();
    MachineInstr *Instr1 = Cand.SU->getInstr();
    // When dealing with two STPqi's.
    if (Instr0 && Instr1 && Instr0->getOpcode() == Instr1->getOpcode () &&
        Instr0->getOpcode() == AArch64::STPQi)
    {
      MachineOperand &Base0 = Instr0->getOperand(2);
      MachineOperand &Base1 = Instr1->getOperand(2);
      int64_t Off0 = Instr0->getOperand(3).getImm();
      int64_t Off1 = Instr1->getOperand(3).getImm();
      // With the same base address and non-overlapping writes.
      if (Base0.isIdenticalTo(Base1) && llabs (Off0 - Off1) >= 2) {
        TryCand.Reason = NodeOrder;
        // Order them by ascending offsets.
        return Off0 < Off1;
      }
    }
  }

  return OriginalResult;
}
