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
  auto FindPotentialRVVInstructionInQueue =
      [&](SchedBoundary &Boundary, ReadyQueue Q, bool ShouldBeTop) -> SUnit * {
    for (SUnit *SU : Q) {
      if (SU->isScheduled)
        continue;

      MachineInstr *MI = SU->getInstr();
      const MCInstrDesc &Desc = MI->getDesc();
      if (RISCVII::hasSEWOp(Desc.TSFlags)) {
        unsigned CurVSEW = MI->getOperand(RISCVII::getSEWOpNum(Desc)).getImm();
        RISCVII::VLMUL CurVLMUL = RISCVII::getLMul(Desc.TSFlags);
        // FIXME: We should consider vl and policy here.
        if (CurVSEW == PrevVSEW && CurVLMUL == PrevVLMUL) {
          IsTopNode = ShouldBeTop;
          // Boundary.removeReady(SU);
          if (SU->isTopReady())
            Top.removeReady(SU);
          if (SU->isBottomReady())
            Bot.removeReady(SU);
          LLVM_DEBUG(dbgs() << "Scheduling SU(" << SU->NodeNum << ") "
                            << *SU->getInstr());
          return SU;
        }
      }
    }
    return nullptr;
  };

  auto FindPotentialRVVInstruction = [&](SchedBoundary &Boundary,
                                         bool ShouldBeTop) -> SUnit * {
    if (SUnit *Available = FindPotentialRVVInstructionInQueue(
            Boundary, Boundary.Available, ShouldBeTop))
      return Available;
    if (SUnit *Pending = FindPotentialRVVInstructionInQueue(
            Boundary, Boundary.Pending, ShouldBeTop))
      return Pending;
    return nullptr;
  };

  if (EnableScheduleSameVType) {
    if (RegionPolicy.OnlyBottomUp) {
      if (SUnit *SU = FindPotentialRVVInstruction(Bot, false))
        return SU;
    } else if (RegionPolicy.OnlyTopDown) {
      if (SUnit *SU = FindPotentialRVVInstruction(Top, true))
        return SU;
    } else {
      if (SUnit *SU =
              FindPotentialRVVInstructionInQueue(Bot, Bot.Available, false))
        return SU;
      if (SUnit *SU =
              FindPotentialRVVInstructionInQueue(Top, Top.Available, true))
        return SU;
      if (SUnit *SU =
              FindPotentialRVVInstructionInQueue(Bot, Bot.Pending, false))
        return SU;
      if (SUnit *SU =
              FindPotentialRVVInstructionInQueue(Top, Top.Pending, true))
        return SU;
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
    LLVM_DEBUG(dbgs() << "Previous scheduled Unit: ";
               dbgs() << "SU(" << SU->NodeNum << ") - ";
               SU->getInstr()->dump(););
    LLVM_DEBUG(dbgs() << "Previous VSEW : " << (1 << PrevVSEW) << "\n";
               auto LMUL = RISCVVType::decodeVLMUL(PrevVLMUL);
               dbgs() << "Previous VLMUL: m" << (LMUL.second ? "f" : "")
                      << LMUL.first << "\n";);
  }
}
