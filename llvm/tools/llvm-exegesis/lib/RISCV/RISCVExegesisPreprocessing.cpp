//===- RISCVExegesisPreprocessing.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// \file
//
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVExegesisPasses.h"
#include "RISCVRegisterInfo.h"
#include "RISCVSubtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"

using namespace llvm;

#define DEBUG_TYPE "riscv-exegesis-preprocessing"

namespace {
struct RISCVExegesisPreprocessing : public MachineFunctionPass {
  static char ID;

  RISCVExegesisPreprocessing() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};
} // anonymous namespace

char RISCVExegesisPreprocessing::ID = 0;

static bool processAVLOperand(MachineInstr &MI, MachineRegisterInfo &MRI,
                              const TargetInstrInfo &TII) {
  const MCInstrDesc &Desc = TII.get(MI.getOpcode());
  uint64_t TSFlags = Desc.TSFlags;
  if (!RISCVII::hasVLOp(TSFlags))
    return false;

  const MachineOperand &VLOp = MI.getOperand(RISCVII::getVLOpNum(Desc));
  if (VLOp.isReg()) {
    Register VLReg = VLOp.getReg();
    if (VLReg.isVirtual())
      return false;
    assert(RISCV::GPRRegClass.contains(VLReg));
    // Replace all uses of the original physical register with a new virtual
    // register. The only reason we can do such replacement here is because it's
    // almost certain that VLReg only has a single definition.
    Register NewVLReg = MRI.createVirtualRegister(&RISCV::GPRRegClass);
    MRI.replaceRegWith(VLReg, NewVLReg);
    return true;
  }

  return false;
}

bool RISCVExegesisPreprocessing::runOnMachineFunction(MachineFunction &MF) {
  MachineRegisterInfo &MRI = MF.getRegInfo();
  const auto &STI = MF.getSubtarget<RISCVSubtarget>();
  if (!STI.hasVInstructions())
    return false;
  const TargetInstrInfo &TII = *STI.getInstrInfo();

  LLVM_DEBUG(MF.print(dbgs() << "===Before RISCVExegesisPoreprocessing===\n");
             dbgs() << "\n");

  bool Changed = false;
  for (auto &MBB : MF)
    for (auto &MI : MBB) {
      Changed |= processAVLOperand(MI, MRI, TII);
    }

  return Changed;
}

FunctionPass *llvm::exegesis::createRISCVPreprocessingPass() {
  return new RISCVExegesisPreprocessing();
}
