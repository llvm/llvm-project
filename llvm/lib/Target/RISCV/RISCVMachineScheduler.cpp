//===- RISCVMachineScheduler.cpp - MI Scheduler for RISC-V ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RISCVMachineScheduler.h"
#include "MCTargetDesc/RISCVBaseInfo.h"
#include "MCTargetDesc/RISCVMCTargetDesc.h"
#include "RISCVInstrInfo.h"
#include "RISCVSubtarget.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineScheduler.h"
#include "llvm/CodeGen/ScheduleDAG.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/Support/Debug.h"
#include "llvm/TargetParser/RISCVTargetParser.h"

using namespace llvm;

#define DEBUG_TYPE "riscv-prera-sched-strategy"

static cl::opt<bool> EnableScheduleSameVType(
    "riscv-enable-schedule-same-vtype", cl::init(false), cl::Hidden,
    cl::desc("Enable scheduling RVV instructions with same vtype first"));

SUnit *RISCVPreRAMachineSchedStrategy::pickNode(bool &IsTopNode) {
  if (EnableScheduleSameVType) {
    for (SUnit *SU : Bot.Available) {
      MachineInstr *MI = SU->getInstr();
      const MCInstrDesc &Desc = MI->getDesc();
      if (RISCVII::hasSEWOp(Desc.TSFlags)) {
        unsigned CurVSEW = MI->getOperand(RISCVII::getSEWOpNum(Desc)).getImm();
        RISCVII::VLMUL CurVLMUL = RISCVII::getLMul(Desc.TSFlags);
        if (CurVSEW == PrevVSEW && CurVLMUL == PrevVLMUL) {
          Bot.removeReady(SU);
          IsTopNode = true;
          return SU;
        }
      }
    }
    for (SUnit *SU : Bot.Pending) {
      MachineInstr *MI = SU->getInstr();
      const MCInstrDesc &Desc = MI->getDesc();
      if (RISCVII::hasSEWOp(Desc.TSFlags)) {
        unsigned CurVSEW = MI->getOperand(RISCVII::getSEWOpNum(Desc)).getImm();
        RISCVII::VLMUL CurVLMUL = RISCVII::getLMul(Desc.TSFlags);
        if (CurVSEW == PrevVSEW && CurVLMUL == PrevVLMUL) {
          Bot.removeReady(SU);
          IsTopNode = false;
          return SU;
        }
      }
    }
  }
  return GenericScheduler::pickNode(IsTopNode);
}

bool RISCVPreRAMachineSchedStrategy::tryCandidate(SchedCandidate &Cand,
                                                  SchedCandidate &TryCand,
                                                  SchedBoundary *Zone) const {
  bool OriginalResult = GenericScheduler::tryCandidate(Cand, TryCand, Zone);

  return OriginalResult;
}

void RISCVPreRAMachineSchedStrategy::schedNode(SUnit *SU, bool IsTopNode) {
  GenericScheduler::schedNode(SU, IsTopNode);
  MachineInstr *MI = SU->getInstr();
  const MCInstrDesc &Desc = MI->getDesc();
  if (RISCVII::hasSEWOp(Desc.TSFlags)) {
    PrevVSEW = MI->getOperand(RISCVII::getSEWOpNum(Desc)).getImm();
    PrevVLMUL = RISCVII::getLMul(Desc.TSFlags);
  }
  LLVM_DEBUG(dbgs() << "Previous scheduled Unit: ";
             dbgs() << "SU(" << SU->NodeNum << ") - "; SU->getInstr()->dump(););
  LLVM_DEBUG(dbgs() << "Previous VSEW : " << (1 << PrevVSEW) << "\n";
             auto LMUL = RISCVVType::decodeVLMUL(PrevVLMUL);
             dbgs() << "Previous VLMUL: m" << (LMUL.second ? "f" : "")
                    << LMUL.first << "\n";);
}
