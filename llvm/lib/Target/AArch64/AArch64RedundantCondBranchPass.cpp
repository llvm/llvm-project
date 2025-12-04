//=- AArch64RedundantCondBranch.cpp - Remove redundant cbz wzr --------------=//
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

static bool optimizeTerminators(MachineBasicBlock *MBB) {
  for (MachineInstr &MI : make_early_inc_range(MBB->terminators())) {
    unsigned Opc = MI.getOpcode();
    switch (Opc) {
    case AArch64::CBZW:
    case AArch64::CBZX:
    case AArch64::TBZW:
    case AArch64::TBZX:
      // CBZ XZR -> B
      if (MI.getOperand(0).getReg() == AArch64::WZR ||
          MI.getOperand(0).getReg() == AArch64::XZR) {
        LLVM_DEBUG(dbgs() << "Removing redundant branch: " << MI);
        MachineBasicBlock *Target =
            MI.getOperand(Opc == AArch64::TBZW || Opc == AArch64::TBZX ? 2 : 1)
                .getMBB();
        MachineBasicBlock *MBB = MI.getParent();
        SmallVector<MachineBasicBlock *> Succs(MBB->successors());
        for (auto *S : Succs)
          if (S != Target)
            MBB->removeSuccessor(S);
        SmallVector<MachineInstr *> DeadInstrs;
        for (auto It = MI.getIterator(); It != MBB->end(); ++It)
          DeadInstrs.push_back(&*It);
        const MachineFunction *MF = MBB->getParent();
        const TargetInstrInfo *TII = MF->getSubtarget().getInstrInfo();
        BuildMI(MBB, MI.getDebugLoc(), TII->get(AArch64::B)).addMBB(Target);
        for (auto It : DeadInstrs)
          It->eraseFromParent();
        return true;
      }
      break;
    case AArch64::CBNZW:
    case AArch64::CBNZX:
    case AArch64::TBNZW:
    case AArch64::TBNZX:
      // CBNZ XZR -> nop
      if (MI.getOperand(0).getReg() == AArch64::WZR ||
          MI.getOperand(0).getReg() == AArch64::XZR) {
        LLVM_DEBUG(dbgs() << "Removing redundant branch: " << MI);
        MachineBasicBlock *Target =
            MI.getOperand((Opc == AArch64::TBNZW || Opc == AArch64::TBNZX) ? 2
                                                                           : 1)
                .getMBB();
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

  bool Changed = false;
  for (MachineBasicBlock &MBB : MF)
    Changed |= optimizeTerminators(&MBB);
  return Changed;
}

FunctionPass *llvm::createAArch64RedundantCondBranchPass() {
  return new AArch64RedundantCondBranch();
}
