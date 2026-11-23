//===---- LoongArchLateBranchOpt.cpp - Late Stage Branch Optimization -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file provides LoongArch specific target optimizations, currently it's
/// limited to convert conditional branches into unconditional branches when
/// the condition can be statically evaluated.
///
//===----------------------------------------------------------------------===//

#include "LoongArch.h"
#include "LoongArchInstrInfo.h"
#include "LoongArchSubtarget.h"
#include "llvm/CodeGen/MachinePassManager.h"
#include "llvm/IR/Analysis.h"

using namespace llvm;

#define LOONGARCH_LATE_BRANCH_OPT_NAME "LoongArch Late Branch Optimisation Pass"

namespace {

struct LoongArchLateBranchOptLegacy : public MachineFunctionPass {
  static char ID;

  LoongArchLateBranchOptLegacy() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override {
    return LOONGARCH_LATE_BRANCH_OPT_NAME;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;
};
} // namespace

char LoongArchLateBranchOptLegacy::ID = 0;
INITIALIZE_PASS(LoongArchLateBranchOptLegacy, "loongarch-late-branch-opt",
                LOONGARCH_LATE_BRANCH_OPT_NAME, false, false)

static bool runOnBasicBlock(MachineBasicBlock &MBB,
                            const LoongArchInstrInfo *TII) {
  MachineBasicBlock *TBB, *FBB;
  SmallVector<MachineOperand, 4> Cond;
  if (TII->analyzeBranch(MBB, TBB, FBB, Cond, /*AllowModify=*/false))
    return false;

  if (!TBB || Cond.size() < 1)
    return false;

  // Try and convert a conditional branch that can be evaluated statically
  // into an unconditional branch.
  unsigned Opc = Cond[0].getImm();
  MachineBasicBlock *Folded;
  switch (Opc) {
  case LoongArch::BEQZ:
  case LoongArch::BNEZ:
    if (Cond.size() < 2 || !Cond[1].isReg() ||
        Cond[1].getReg() != LoongArch::R0)
      return false;
    Folded = (Opc == LoongArch::BEQZ) ? TBB : FBB;
    break;
  case LoongArch::BEQ:
  case LoongArch::BNE:
    if (Cond.size() < 3 || !Cond[1].isReg() || !Cond[2].isReg() ||
        Cond[1].getReg() != Cond[2].getReg())
      return false;
    Folded = (Opc == LoongArch::BEQ) ? TBB : FBB;
    break;
  default:
    return false;
  }

  // At this point, its legal to optimize.
  TII->removeBranch(MBB);

  // Only need to insert a branch if we're not falling through.
  if (Folded) {
    DebugLoc DL = MBB.findBranchDebugLoc();
    TII->insertBranch(MBB, Folded, nullptr, {}, DL);
  }

  // Update the successors. Remove them all and add back the correct one.
  while (!MBB.succ_empty())
    MBB.removeSuccessor(MBB.succ_end() - 1);

  // If it's a fallthrough, we need to figure out where MBB is going.
  if (!Folded) {
    MachineFunction::iterator Fallthrough = ++MBB.getIterator();
    if (Fallthrough != MBB.getParent()->end())
      MBB.addSuccessor(&*Fallthrough);
  } else
    MBB.addSuccessor(Folded);

  return true;
}

static bool runLateBranchOpt(MachineFunction &MF) {
  auto &ST = MF.getSubtarget<LoongArchSubtarget>();
  const LoongArchInstrInfo *TII = ST.getInstrInfo();

  bool Changed = false;
  for (MachineBasicBlock &MBB : MF)
    Changed |= runOnBasicBlock(MBB, TII);
  return Changed;
}

bool LoongArchLateBranchOptLegacy::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;
  return runLateBranchOpt(MF);
}

FunctionPass *llvm::createLoongArchLateBranchOptLegacyPass() {
  return new LoongArchLateBranchOptLegacy();
}

PreservedAnalyses
LoongArchLateBranchOptPass::run(MachineFunction &MF,
                                MachineFunctionAnalysisManager &MFAM) {
  return runLateBranchOpt(MF) ? getMachineFunctionPassPreservedAnalyses()
                              : PreservedAnalyses::all();
}
