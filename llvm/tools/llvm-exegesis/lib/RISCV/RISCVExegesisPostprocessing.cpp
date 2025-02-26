//===- RISCVExegesisPostprocessing.cpp - Post processing MI for exegesis---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// \file
// Currently there is only one post-processing we need to do for exegesis:
// Assign a physical register to VSETVL's rd if it's not X0 (i.e. VLMAX).
//
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVExegesisPasses.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

#define DEBUG_TYPE "riscv-exegesis-post-processing"

namespace {
struct RISCVExegesisPostprocessing : public MachineFunctionPass {
  static char ID;

  RISCVExegesisPostprocessing() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

private:
  // Extremely simple register allocator that picks a register that hasn't
  // been defined or used in this function.
  Register allocateGPRRegister(const MachineFunction &MF,
                               const MachineRegisterInfo &MRI);

  bool processVSETVL(MachineInstr &MI, MachineRegisterInfo &MRI);
  bool processWriteFRM(MachineInstr &MI, MachineRegisterInfo &MRI);
};
} // anonymous namespace

char RISCVExegesisPostprocessing::ID = 0;

bool RISCVExegesisPostprocessing::runOnMachineFunction(MachineFunction &MF) {
  bool Changed = false;
  for (auto &MBB : MF)
    for (auto &MI : MBB) {
      unsigned Opcode = MI.getOpcode();
      switch (Opcode) {
      case RISCV::VSETVLI:
      case RISCV::VSETVL:
      case RISCV::PseudoVSETVLI:
      case RISCV::PseudoVSETVLIX0:
        Changed |= processVSETVL(MI, MF.getRegInfo());
        break;
      case RISCV::SwapFRMImm:
      case RISCV::WriteFRM:
        Changed |= processWriteFRM(MI, MF.getRegInfo());
        break;
      default:
        break;
      }
    }

  if (Changed)
    MF.getRegInfo().clearVirtRegs();

  LLVM_DEBUG(MF.print(dbgs() << "===After RISCVExegesisPostprocessing===\n");
             dbgs() << "\n");

  return Changed;
}

Register RISCVExegesisPostprocessing::allocateGPRRegister(
    const MachineFunction &MF, const MachineRegisterInfo &MRI) {
  const auto &TRI = *MRI.getTargetRegisterInfo();

  // We hope to avoid allocating callee-saved registers. And GPRTC
  // happens to account for nearly all caller-saved registers.
  const TargetRegisterClass *GPRClass = TRI.getRegClass(RISCV::GPRTCRegClassID);
  BitVector Candidates = TRI.getAllocatableSet(MF, GPRClass);

  for (unsigned SetIdx : Candidates.set_bits()) {
    if (MRI.reg_empty(Register(SetIdx)))
      return Register(SetIdx);
  }

  // All bets are off, assign a fixed one.
  return RISCV::X5;
}

bool RISCVExegesisPostprocessing::processVSETVL(MachineInstr &MI,
                                                MachineRegisterInfo &MRI) {
  bool Changed = false;
  // Replace both AVL and VL (i.e. the result) operands with physical
  // registers.
  for (unsigned Idx = 0U; Idx < 2; ++Idx)
    if (MI.getOperand(Idx).isReg()) {
      Register RegOp = MI.getOperand(Idx).getReg();
      if (RegOp.isVirtual()) {
        MRI.replaceRegWith(RegOp, allocateGPRRegister(*MI.getMF(), MRI));
        Changed = true;
      }
    }

  return Changed;
}

bool RISCVExegesisPostprocessing::processWriteFRM(MachineInstr &MI,
                                                  MachineRegisterInfo &MRI) {
  // The virtual register will be the first operand in both SwapFRMImm and
  // WriteFRM.
  if (MI.getOperand(0).isReg()) {
    Register DestReg = MI.getOperand(0).getReg();
    if (DestReg.isVirtual()) {
      MRI.replaceRegWith(DestReg, allocateGPRRegister(*MI.getMF(), MRI));
      return true;
    }
  }
  return false;
}

FunctionPass *llvm::exegesis::createRISCVPostprocessingPass() {
  return new RISCVExegesisPostprocessing();
}
