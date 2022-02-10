//=- RISCVRedundantCopyElimination.cpp - Remove useless copy for RISCV ------=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass removes unnecessary zero copies in BBs that are targets of
// beqz/bnez instructions. For instance, the copy instruction in the code below
// can be removed because the beqz jumps to BB#2 when a0 is zero.
//  BB#1:
//    beqz %a0, <BB#2>
//  BB#2:
//    %a0 = COPY %x0
// This pass should be run after register allocation.
//
// This pass is based on the earliest versions of
// AArch64RedundantCopyElimination.
//
// FIXME: Support compares with constants other than zero? This is harder to
// do on RISC-V since branches can't have immediates.
//
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

#define DEBUG_TYPE "riscv-copyelim"

STATISTIC(NumCopiesRemoved, "Number of copies removed.");

namespace {
class RISCVRedundantCopyElimination : public MachineFunctionPass {
  const MachineRegisterInfo *MRI;
  const TargetRegisterInfo *TRI;

public:
  static char ID;
  RISCVRedundantCopyElimination() : MachineFunctionPass(ID) {
    initializeRISCVRedundantCopyEliminationPass(
        *PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;
  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().set(
        MachineFunctionProperties::Property::NoVRegs);
  }

  StringRef getPassName() const override {
    return "RISCV Redundant Copy Elimination";
  }

private:
  bool optimizeBlock(MachineBasicBlock &MBB);
};

} // end anonymous namespace

char RISCVRedundantCopyElimination::ID = 0;

INITIALIZE_PASS(RISCVRedundantCopyElimination, "riscv-copyelim",
                "RISCV redundant copy elimination pass", false, false)

static bool guaranteesZeroRegInBlock(const MachineInstr &MI,
                                     const MachineBasicBlock &MBB) {
  unsigned Opc = MI.getOpcode();
  if (Opc == RISCV::BEQ && MI.getOperand(1).getReg() == RISCV::X0 &&
      &MBB == MI.getOperand(2).getMBB())
    return true;
  if (Opc == RISCV::BNE && MI.getOperand(1).getReg() == RISCV::X0 &&
      &MBB != MI.getOperand(2).getMBB())
    return true;

  return false;
}

bool RISCVRedundantCopyElimination::optimizeBlock(MachineBasicBlock &MBB) {
  // Check if the current basic block has a single predecessor.
  if (MBB.pred_size() != 1)
    return false;

  // Check if the predecessor has two successors, implying the block ends in a
  // conditional branch.
  MachineBasicBlock *PredMBB = *MBB.pred_begin();
  if (PredMBB->succ_size() != 2)
    return false;

  MachineBasicBlock::iterator CondBr = PredMBB->getLastNonDebugInstr();
  if (CondBr == PredMBB->end())
    return false;

  while (true) {
    // If we run out of terminators, give up.
    if (!CondBr->isTerminator())
      return false;
    // If we found a branch with X0, stop searching and try to remove copies.
    // TODO: Handle multiple branches with different registers.
    if (guaranteesZeroRegInBlock(*CondBr, MBB))
      break;
    // If we reached the beginning of the basic block, give up.
    if (CondBr == PredMBB->begin())
      return false;
    --CondBr;
  }

  Register TargetReg = CondBr->getOperand(0).getReg();
  if (!TargetReg)
    return false;

  bool Changed = false;
  MachineBasicBlock::iterator LastChange = MBB.begin();
  // Remove redundant Copy instructions unless TargetReg is modified.
  for (MachineBasicBlock::iterator I = MBB.begin(), E = MBB.end(); I != E;) {
    MachineInstr *MI = &*I;
    ++I;
    if (MI->isCopy() && MI->getOperand(0).isReg() &&
        MI->getOperand(1).isReg()) {
      Register DefReg = MI->getOperand(0).getReg();
      Register SrcReg = MI->getOperand(1).getReg();

      if (SrcReg == RISCV::X0 && !MRI->isReserved(DefReg) &&
          TargetReg == DefReg) {
        LLVM_DEBUG(dbgs() << "Remove redundant Copy : ");
        LLVM_DEBUG(MI->print(dbgs()));

        MI->eraseFromParent();
        Changed = true;
        LastChange = I;
        ++NumCopiesRemoved;
        continue;
      }
    }

    if (MI->modifiesRegister(TargetReg, TRI))
      break;
  }

  if (!Changed)
    return false;

  // Otherwise, we have to fixup the use-def chain, starting with the
  // BEQ/BNE. Conservatively mark as much as we can live.
  CondBr->clearRegisterKills(TargetReg, TRI);

  // Add newly used reg to the block's live-in list if it isn't there already.
  if (!MBB.isLiveIn(TargetReg))
    MBB.addLiveIn(TargetReg);

  // Clear any kills of TargetReg between CondBr and the last removed COPY.
  for (MachineInstr &MMI : make_range(MBB.begin(), LastChange))
    MMI.clearRegisterKills(TargetReg, TRI);

  return true;
}

bool RISCVRedundantCopyElimination::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;

  TRI = MF.getSubtarget().getRegisterInfo();
  MRI = &MF.getRegInfo();

  bool Changed = false;
  for (MachineBasicBlock &MBB : MF)
    Changed |= optimizeBlock(MBB);

  return Changed;
}

FunctionPass *llvm::createRISCVRedundantCopyEliminationPass() {
  return new RISCVRedundantCopyElimination();
}
