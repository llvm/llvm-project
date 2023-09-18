//===-- RISCVInsertReadWriteCSR.cpp - Insert Read/Write of RISC-V CSR -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This file implements the machine function pass to insert read/write of CSR-s
// of the RISC-V instructions.
//
// Currently the pass implements:
// -Naive insertion of a write to vxrm before an RVV fixed-point instruction.
// -Writing and saving frm before an RVV floating-point instruction with a
//  static rounding mode and restores the value after.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/RISCVBaseInfo.h"
#include "RISCV.h"
#include "RISCVSubtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
using namespace llvm;

#define DEBUG_TYPE "riscv-insert-read-write-csr"
#define RISCV_INSERT_READ_WRITE_CSR_NAME "RISC-V Insert Read/Write CSR Pass"

namespace {

class RISCVInsertReadWriteCSR : public MachineFunctionPass {
  const TargetInstrInfo *TII;

public:
  static char ID;

  RISCVInsertReadWriteCSR() : MachineFunctionPass(ID) {
    initializeRISCVInsertReadWriteCSRPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  StringRef getPassName() const override {
    return RISCV_INSERT_READ_WRITE_CSR_NAME;
  }

private:
  bool emitWriteRoundingMode(MachineBasicBlock &MBB);
};

} // end anonymous namespace

char RISCVInsertReadWriteCSR::ID = 0;

INITIALIZE_PASS(RISCVInsertReadWriteCSR, DEBUG_TYPE,
                RISCV_INSERT_READ_WRITE_CSR_NAME, false, false)

// This function inserts a write to vxrm when encountering an RVV fixed-point
// instruction. This function also swaps frm and restores it when encountering
// an RVV floating point instruction with a static rounding mode.
bool RISCVInsertReadWriteCSR::emitWriteRoundingMode(MachineBasicBlock &MBB) {
  bool Changed = false;
  for (MachineInstr &MI : MBB) {
    int VXRMIdx = RISCVII::getVXRMOpNum(MI.getDesc());
    if (VXRMIdx >= 0) {
      unsigned VXRMImm = MI.getOperand(VXRMIdx).getImm();

      Changed = true;

      BuildMI(MBB, MI, MI.getDebugLoc(), TII->get(RISCV::WriteVXRMImm))
          .addImm(VXRMImm);
      MI.addOperand(MachineOperand::CreateReg(RISCV::VXRM, /*IsDef*/ false,
                                              /*IsImp*/ true));
      continue;
    }

    int FRMIdx = RISCVII::getFRMOpNum(MI.getDesc());
    if (FRMIdx < 0)
      continue;

    unsigned FRMImm = MI.getOperand(FRMIdx).getImm();

    // The value is a hint to this pass to not alter the frm value.
    if (FRMImm == RISCVFPRndMode::DYN)
      continue;

    Changed = true;

    // Save
    MachineRegisterInfo *MRI = &MBB.getParent()->getRegInfo();
    Register SavedFRM = MRI->createVirtualRegister(&RISCV::GPRRegClass);
    BuildMI(MBB, MI, MI.getDebugLoc(), TII->get(RISCV::SwapFRMImm),
            SavedFRM)
        .addImm(FRMImm);
    MI.addOperand(MachineOperand::CreateReg(RISCV::FRM, /*IsDef*/ false,
                                            /*IsImp*/ true));
    // Restore
    MachineInstrBuilder MIB =
        BuildMI(*MBB.getParent(), {}, TII->get(RISCV::WriteFRM))
            .addReg(SavedFRM);
    MBB.insertAfter(MI, MIB);
  }
  return Changed;
}

bool RISCVInsertReadWriteCSR::runOnMachineFunction(MachineFunction &MF) {
  // Skip if the vector extension is not enabled.
  const RISCVSubtarget &ST = MF.getSubtarget<RISCVSubtarget>();
  if (!ST.hasVInstructions())
    return false;

  TII = ST.getInstrInfo();

  bool Changed = false;

  for (MachineBasicBlock &MBB : MF)
    Changed |= emitWriteRoundingMode(MBB);

  return Changed;
}

FunctionPass *llvm::createRISCVInsertReadWriteCSRPass() {
  return new RISCVInsertReadWriteCSR();
}
