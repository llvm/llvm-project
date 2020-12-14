//===- RISCVCleanupVSETVLI.cpp - Cleanup unneeded VSETVLI instructions ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a function pass that removes duplicate vsetvli
// instructions within a basic block.
//
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVSubtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
using namespace llvm;

#define DEBUG_TYPE "riscv-cleanup-vsetvli"
#define RISCV_CLEANUP_VSETVLI_NAME "RISCV Cleanup VSETVLI pass"

namespace {

class RISCVCleanupVSETVLI : public MachineFunctionPass {
public:
  static char ID;

  RISCVCleanupVSETVLI() : MachineFunctionPass(ID) {
    initializeRISCVCleanupVSETVLIPass(*PassRegistry::getPassRegistry());
  }
  bool runOnMachineFunction(MachineFunction &MF) override;
  bool runOnMachineBasicBlock(MachineBasicBlock &MBB);

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().set(
        MachineFunctionProperties::Property::IsSSA);
  }

  // This pass modifies the program, but does not modify the CFG
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  StringRef getPassName() const override { return RISCV_CLEANUP_VSETVLI_NAME; }
};

} // end anonymous namespace

char RISCVCleanupVSETVLI::ID = 0;

INITIALIZE_PASS(RISCVCleanupVSETVLI, DEBUG_TYPE,
                RISCV_CLEANUP_VSETVLI_NAME, false, false)

bool RISCVCleanupVSETVLI::runOnMachineBasicBlock(MachineBasicBlock &MBB) {
  bool Changed = false;
  MachineInstr *PrevVSETVLI = nullptr;

  for (auto MII = MBB.begin(), MIE = MBB.end(); MII != MIE;) {
    MachineInstr &MI = *MII++;

    if (MI.getOpcode() != RISCV::PseudoVSETVLI) {
      if (PrevVSETVLI &&
          (MI.isCall() || MI.modifiesRegister(RISCV::VL) ||
           MI.modifiesRegister(RISCV::VTYPE))) {
        // Old VL/VTYPE is overwritten.
        PrevVSETVLI = nullptr;
      }
      continue;
    }

    // If we don't have a previous VSETVLI or the VL output isn't dead, we
    // can't remove this VSETVLI.
    if (!PrevVSETVLI || !MI.getOperand(0).isDead()) {
      PrevVSETVLI = &MI;
      continue;
    }

    Register PrevAVLReg = PrevVSETVLI->getOperand(1).getReg();
    Register AVLReg = MI.getOperand(1).getReg();
    int64_t PrevVTYPEImm = PrevVSETVLI->getOperand(2).getImm();
    int64_t VTYPEImm = MI.getOperand(2).getImm();

    // Does this VSETVLI use the same AVL register and VTYPE immediate?
    if (PrevAVLReg != AVLReg || PrevVTYPEImm != VTYPEImm) {
      PrevVSETVLI = &MI;
      continue;
    }

    // If the AVLReg is X0 we need to look at the output VL of both VSETVLIs.
    if (AVLReg == RISCV::X0) {
      Register PrevOutVL = PrevVSETVLI->getOperand(0).getReg();
      Register OutVL = MI.getOperand(0).getReg();
      // We can't remove if the previous VSETVLI left VL unchanged and the
      // current instruction is setting it to VLMAX. Without knowing the VL
      // before the previous instruction we don't know if this is a change.
      if (PrevOutVL == RISCV::X0 && OutVL != RISCV::X0) {
        PrevVSETVLI = &MI;
        continue;
      }
    }

    // This VSETVLI is redundant, remove it.
    MI.eraseFromParent();
    Changed = true;
  }

  return Changed;
}

bool RISCVCleanupVSETVLI::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;

  // Skip if the vector extension is not enabled.
  const RISCVSubtarget &ST = MF.getSubtarget<RISCVSubtarget>();
  if (!ST.hasStdExtV())
    return false;

  bool Changed = false;

  for (MachineBasicBlock &MBB : MF)
    Changed |= runOnMachineBasicBlock(MBB);

  return Changed;
}

/// Returns an instance of the Cleanup VSETVLI pass.
FunctionPass *llvm::createRISCVCleanupVSETVLIPass() {
  return new RISCVCleanupVSETVLI();
}
