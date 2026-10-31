//===-- RISCVLateBranchOpt.cpp - Late Stage Branch Optimization -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file provides RISC-V specific target optimizations, currently it's
/// limited to convert conditional branches into unconditional branches when
/// the condition can be statically evaluated.
///
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVInstrInfo.h"
#include "RISCVSubtarget.h"
#include "llvm/CodeGen/MachineFunctionAnalysisManager.h"
#include "llvm/CodeGen/MachinePassManager.h"
#include "llvm/IR/Analysis.h"

using namespace llvm;

#define RISCV_LATE_BRANCH_OPT_NAME "RISC-V Late Branch Optimisation Pass"

namespace {

struct RISCVLateBranchOptLegacy : public MachineFunctionPass {
  static char ID;

  RISCVLateBranchOptLegacy() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override { return RISCV_LATE_BRANCH_OPT_NAME; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &Fn) override;
};
} // namespace

char RISCVLateBranchOptLegacy::ID = 0;
INITIALIZE_PASS(RISCVLateBranchOptLegacy, "riscv-late-branch-opt",
                RISCV_LATE_BRANCH_OPT_NAME, false, false)

static bool runLateBranchOptOnMBB(MachineBasicBlock &MBB,
                                  const RISCVInstrInfo *RII) {
  MachineBasicBlock *TBB, *FBB;
  SmallVector<MachineOperand, 4> Cond;
  if (RII->analyzeBranch(MBB, TBB, FBB, Cond, /*AllowModify=*/false))
    return false;

  if (!TBB || Cond.size() != 3)
    return false;

  RISCVCC::CondCode CC = RISCVInstrInfo::getCondFromBranchOpc(Cond[0].getImm());
  assert(CC != RISCVCC::COND_INVALID);

  MachineRegisterInfo &MRI = MBB.getParent()->getRegInfo();

  // Try and convert a conditional branch that can be evaluated statically
  // into an unconditional branch.
  int64_t C0, C1;
  if (!RISCVInstrInfo::isFromLoadImm(MRI, Cond[1], C0) ||
      !RISCVInstrInfo::isFromLoadImm(MRI, Cond[2], C1))
    return false;

  MachineBasicBlock *Folded =
      RISCVInstrInfo::evaluateCondBranch(CC, C0, C1) ? TBB : FBB;

  // At this point, its legal to optimize.
  RII->removeBranch(MBB);

  // Only need to insert a branch if we're not falling through.
  if (Folded) {
    DebugLoc DL = MBB.findBranchDebugLoc();
    RII->insertBranch(MBB, Folded, nullptr, {}, DL);
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

static bool runLateBranchOptOnMF(MachineFunction &Fn) {
  auto &ST = Fn.getSubtarget<RISCVSubtarget>();
  const RISCVInstrInfo *RII = ST.getInstrInfo();

  bool Changed = false;
  for (MachineBasicBlock &MBB : Fn)
    Changed |= runLateBranchOptOnMBB(MBB, RII);
  return Changed;
}

FunctionPass *llvm::createRISCVLateBranchOptLegacyPass() {
  return new RISCVLateBranchOptLegacy();
}

bool RISCVLateBranchOptLegacy::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;
  return runLateBranchOptOnMF(MF);
}

PreservedAnalyses
RISCVLateBranchOptPass::run(MachineFunction &MF,
                            MachineFunctionAnalysisManager &MFAM) {
  return runLateBranchOptOnMF(MF) ? getMachineFunctionPassPreservedAnalyses()
                                  : PreservedAnalyses::all();
}
