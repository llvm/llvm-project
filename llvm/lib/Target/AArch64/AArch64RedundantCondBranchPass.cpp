//=- AArch64RedundantCondBranch.cpp - Remove redundant conditional branches -=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Late in the pipeline, especially with zero phi operands propagated after tail
// duplications, we can end up with CBZ/CBNZ/TBZ/TBNZ with a zero register. This
// simple pass looks at the terminators to a block, removing the redundant
// instructions where necessary.
//
//===----------------------------------------------------------------------===//

#include "AArch64.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

#define DEBUG_TYPE "aarch64-redundantcondbranch"

namespace {
class AArch64RedundantCondBranch : public MachineFunctionPass {
public:
  static char ID;
  AArch64RedundantCondBranch() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().setNoVRegs();
  }
  StringRef getPassName() const override {
    return "AArch64 Redundant Conditional Branch Elimination";
  }
};
char AArch64RedundantCondBranch::ID = 0;
} // namespace

INITIALIZE_PASS(AArch64RedundantCondBranch, "aarch64-redundantcondbranch",
                "AArch64 Redundant Conditional Branch Elimination pass", false,
                false)

static bool optimizeTerminators(MachineBasicBlock *MBB,
                                const TargetInstrInfo &TII) {
  for (MachineInstr &MI : make_early_inc_range(MBB->terminators())) {
    unsigned Opc = MI.getOpcode();
    switch (Opc) {
    case AArch64::CBZW:
    case AArch64::CBZX:
    case AArch64::TBZW:
    case AArch64::TBZX:
      // CBZ/TBZ with WZR/XZR -> unconditional B
      if (MI.getOperand(0).getReg() == AArch64::WZR ||
          MI.getOperand(0).getReg() == AArch64::XZR) {
        LLVM_DEBUG(dbgs() << "Removing redundant branch: " << MI);
        MachineBasicBlock *Target = TII.getBranchDestBlock(MI);
        SmallVector<MachineBasicBlock *> Succs(MBB->successors());
        for (auto *S : Succs)
          if (S != Target)
            MBB->removeSuccessor(S);
        DebugLoc DL = MI.getDebugLoc();
        while (MBB->rbegin() != &MI)
          MBB->rbegin()->eraseFromParent();
        MI.eraseFromParent();
        BuildMI(MBB, DL, TII.get(AArch64::B)).addMBB(Target);
        return true;
      }
      break;
    case AArch64::CBNZW:
    case AArch64::CBNZX:
    case AArch64::TBNZW:
    case AArch64::TBNZX:
      // CBNZ/TBNZ with WZR/XZR -> never taken, remove branch and successor
      if (MI.getOperand(0).getReg() == AArch64::WZR ||
          MI.getOperand(0).getReg() == AArch64::XZR) {
        LLVM_DEBUG(dbgs() << "Removing redundant branch: " << MI);
        MachineBasicBlock *Target = TII.getBranchDestBlock(MI);
        MI.getParent()->removeSuccessor(Target);
        MI.eraseFromParent();
        return true;
      }
      break;
    }
  }
  return false;
}

bool AArch64RedundantCondBranch::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;

  const TargetInstrInfo &TII = *MF.getSubtarget().getInstrInfo();

  bool Changed = false;
  for (MachineBasicBlock &MBB : MF)
    Changed |= optimizeTerminators(&MBB, TII);
  return Changed;
}

FunctionPass *llvm::createAArch64RedundantCondBranchPass() {
  return new AArch64RedundantCondBranch();
}
