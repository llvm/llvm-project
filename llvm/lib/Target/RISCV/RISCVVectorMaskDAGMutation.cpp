//===- RISCVVectorMaskDAGMutation.cpp - RISC-V Vector Mask DAGMutation ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A schedule mutation that adds an artificial dependency between masks producer
// instructions and masked instructions, so that we can reduce the live range
// overlaps of mask registers.
//
// If there are multiple masks producers followed by multiple masked
// instructions, then at each masked instructions add dependency edges between
// every producer and masked instruction.
//
// The reason why we need to do this:
// 1. When tracking register pressure, we don't track physical registers.
// 2. We have a RegisterClass for mask register (which is `VMV0`), but we don't
//    use it by the time we reach scheduling. Instead, we use physical
//    register V0 directly and insert a `$v0 = COPY ...` before the use.
// 3. For mask producers, we are using VR RegisterClass (we can allocate V0-V31
//    to it). So if V0 is not available, there are still 31 available registers
//    out there.
//
// This means that the RegPressureTracker can't track the pressure of mask
// registers correctly.
//
// This schedule mutation is a workaround to fix this issue.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/RISCVBaseInfo.h"
#include "MCTargetDesc/RISCVMCTargetDesc.h"
#include "RISCVTargetMachine.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/ScheduleDAGInstrs.h"
#include "llvm/CodeGen/ScheduleDAGMutation.h"
#include "llvm/TargetParser/RISCVTargetParser.h"

#define DEBUG_TYPE "machine-scheduler"

namespace llvm {

static bool isCopyFromV0(const MachineInstr &MI) {
  return MI.isFullCopy() && MI.getOperand(1).getReg() == RISCV::V0 &&
         MI.getOperand(0).getReg().isVirtual();
}

static bool isCopyToV0(const MachineInstr &MI) {
  return MI.isFullCopy() && MI.getOperand(0).getReg() == RISCV::V0 &&
         MI.getOperand(1).getReg().isVirtual();
}

static bool isSoleUseCopyToV0(SUnit &SU) {
  if (SU.Succs.size() != 1)
    return false;
  SDep &Dep = SU.Succs[0];
  // Ignore dependencies other than data or strong ordering.
  if (Dep.isWeak())
    return false;

  SUnit &DepSU = *Dep.getSUnit();
  if (DepSU.isBoundaryNode())
    return false;
  return isCopyToV0(*DepSU.getInstr());
}

class RISCVVectorMaskDAGMutation : public ScheduleDAGMutation {
private:
  const TargetRegisterInfo *TRI;

public:
  RISCVVectorMaskDAGMutation(const TargetRegisterInfo *TRI) : TRI(TRI) {}

  void apply(ScheduleDAGInstrs *DAG) override {
    SUnit *NearestUseV0SU = nullptr;
    SmallVector<SUnit *, 2> DefMask;
    SUnit *CopyFromV0 = nullptr;
    SUnit *NearestHasVMV0OpSU = nullptr;

    auto HasVMV0Op = [&](const MachineInstr *MI) {
      for (const MachineOperand &Use : MI->uses()) {
        if (!Use.isReg())
          continue;
        Register Reg = Use.getReg();
        if (!Reg.isVirtual())
          continue;
        if (DAG->MRI.getRegClass(Reg)->getID() == RISCV::VMV0RegClassID)
          return true;
      }
      return false;
    };

    for (SUnit &SU : DAG->SUnits) {
      const MachineInstr *MI = SU.getInstr();
      // The copies of v0 should not go accross these instructions with vmv0
      // operands.
      if (HasVMV0Op(MI)) {
        NearestHasVMV0OpSU = &SU;
        if (CopyFromV0) {
          DAG->addEdge(&SU, SDep(CopyFromV0, SDep::Artificial));
          CopyFromV0 = nullptr;
        }
      }

      if (NearestHasVMV0OpSU && NearestHasVMV0OpSU != &SU && isCopyToV0(*MI))
        DAG->addEdge(&SU, SDep(NearestHasVMV0OpSU, SDep::Artificial));

      if (isCopyFromV0(*MI))
        CopyFromV0 = &SU;

      bool UseV0 = MI->findRegisterUseOperand(RISCV::V0, TRI);
      if (isSoleUseCopyToV0(SU) && !UseV0)
        DefMask.push_back(&SU);

      if (UseV0) {
        NearestUseV0SU = &SU;

        // Copy may not be a real use, so skip it here.
        if (DefMask.size() > 1 && !MI->isCopy()) {
          for (SUnit *Def : DefMask)
            if (DAG->canAddEdge(Def, &SU))
              DAG->addEdge(Def, SDep(&SU, SDep::Artificial));
        }

        if (!DefMask.empty())
          DefMask.erase(DefMask.begin());
      }

      if (NearestUseV0SU && NearestUseV0SU != &SU && isSoleUseCopyToV0(SU) &&
          // For LMUL=8 cases, there will be more possibilities to spill.
          // FIXME: We should use RegPressureTracker to do fine-grained
          // controls.
          RISCVII::getLMul(MI->getDesc().TSFlags) != RISCVVType::LMUL_8)
        DAG->addEdge(&SU, SDep(NearestUseV0SU, SDep::Artificial));
    }
  }
};

std::unique_ptr<ScheduleDAGMutation>
createRISCVVectorMaskDAGMutation(const TargetRegisterInfo *TRI) {
  return std::make_unique<RISCVVectorMaskDAGMutation>(TRI);
}

} // namespace llvm
