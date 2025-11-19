//===-- LoongArchLateBranchOpt.cpp - Late Stage Branch Optimization -------===//
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

#include "LoongArchInstrInfo.h"
#include "LoongArchSubtarget.h"

using namespace llvm;

#define LOONGARCH_LATE_BRANCH_OPT_NAME "LoongArch Late Branch Optimisation Pass"

namespace {

struct LoongArchLateBranchOpt : public MachineFunctionPass {
  static char ID;

  LoongArchLateBranchOpt() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override {
    return LOONGARCH_LATE_BRANCH_OPT_NAME;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &Fn) override;

private:
  bool runOnBasicBlock(MachineBasicBlock &MBB) const;

  bool isLoadImm(const MachineInstr *MI, int64_t &Imm) const;
  bool isFromLoadImm(const MachineOperand &Op, int64_t &Imm) const;

  bool evaluateCondBranch(unsigned Opc, int64_t C0, int64_t C1) const;

  const LoongArchSubtarget *ST = nullptr;
  MachineRegisterInfo *MRI;
};
} // namespace

char LoongArchLateBranchOpt::ID = 0;
INITIALIZE_PASS(LoongArchLateBranchOpt, "loongarch-late-branch-opt",
                LOONGARCH_LATE_BRANCH_OPT_NAME, false, false)

// Return true if the instruction is a load immediate instruction.
// TODO: More consideration?
bool LoongArchLateBranchOpt::isLoadImm(const MachineInstr *MI,
                                       int64_t &Imm) const {
  unsigned Shift = 0;
  switch (MI->getOpcode()) {
  default:
    return false;
  case LoongArch::LU52I_D:
    Shift = 52;
    [[fallthrough]];
  case LoongArch::ORI:
  case LoongArch::ADDI_W:
    if (!MI->getOperand(1).isReg() ||
        MI->getOperand(1).getReg() != LoongArch::R0)
      return false;
    Imm = MI->getOperand(2).getImm() << Shift;
    return true;
  case LoongArch::LU12I_W:
    Imm = MI->getOperand(1).getImm() << 12;
    return true;
  }
}

// Return true if the operand is a load immediate instruction and
// sets Imm to the immediate value.
bool LoongArchLateBranchOpt::isFromLoadImm(const MachineOperand &Op,
                                           int64_t &Imm) const {
  // Either a load from immediate instruction or R0.
  if (!Op.isReg())
    return false;

  Register Reg = Op.getReg();
  if (Reg == LoongArch::R0) {
    Imm = 0;
    return true;
  }
  return Reg.isVirtual() && isLoadImm(MRI->getVRegDef(Reg), Imm);
}

// Return the result of the evaluation of 'C0 CC C1', where CC is the
// condition of Opc and C1 is always zero when Opc is B{EQ/NE/CEQ/CNE}Z.
bool LoongArchLateBranchOpt::evaluateCondBranch(unsigned Opc, int64_t C0,
                                                int64_t C1) const {
  switch (Opc) {
  default:
    llvm_unreachable("Unexpected Opcode.");
  case LoongArch::BEQ:
  case LoongArch::BEQZ:
  case LoongArch::BCEQZ:
    return C0 == C1;
  case LoongArch::BNE:
  case LoongArch::BNEZ:
  case LoongArch::BCNEZ:
    return C0 != C1;
  case LoongArch::BLT:
    return C0 < C1;
  case LoongArch::BGE:
    return C0 >= C1;
  case LoongArch::BLTU:
    return (uint64_t)C0 < (uint64_t)C1;
  case LoongArch::BGEU:
    return (uint64_t)C0 >= (uint64_t)C1;
  }
}

bool LoongArchLateBranchOpt::runOnBasicBlock(MachineBasicBlock &MBB) const {
  const LoongArchInstrInfo &TII = *ST->getInstrInfo();
  MachineBasicBlock *TBB, *FBB;
  SmallVector<MachineOperand, 4> Cond;

  if (TII.analyzeBranch(MBB, TBB, FBB, Cond, /*AllowModify=*/false))
    return false;

  // LoongArch conditional branch instructions compare two operands (i.e.
  // Opc C0, C1, TBB) or one operand with immediate zero (i.e. Opc C0, TBB).
  if (!TBB || (Cond.size() != 2 && Cond.size() != 3))
    return false;

  // Try and convert a conditional branch that can be evaluated statically
  // into an unconditional branch.
  int64_t C0 = 0, C1 = 0;
  unsigned Opc = Cond[0].getImm();
  switch (Opc) {
  default:
    llvm_unreachable("Unexpected Opcode.");
  case LoongArch::BEQ:
  case LoongArch::BNE:
  case LoongArch::BLT:
  case LoongArch::BGE:
  case LoongArch::BLTU:
  case LoongArch::BGEU:
    if (!isFromLoadImm(Cond[1], C0) || !isFromLoadImm(Cond[2], C1))
      return false;
    break;
  case LoongArch::BEQZ:
  case LoongArch::BNEZ:
  case LoongArch::BCEQZ:
  case LoongArch::BCNEZ:
    if (!isFromLoadImm(Cond[1], C0))
      return false;
    break;
  }

  MachineBasicBlock *Folded = evaluateCondBranch(Opc, C0, C1) ? TBB : FBB;

  // At this point, its legal to optimize.
  TII.removeBranch(MBB);

  // Only need to insert a branch if we're not falling through.
  if (Folded) {
    DebugLoc DL = MBB.findBranchDebugLoc();
    TII.insertBranch(MBB, Folded, nullptr, {}, DL);
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

bool LoongArchLateBranchOpt::runOnMachineFunction(MachineFunction &Fn) {
  if (skipFunction(Fn.getFunction()))
    return false;

  ST = &Fn.getSubtarget<LoongArchSubtarget>();
  MRI = &Fn.getRegInfo();

  bool Changed = false;
  for (MachineBasicBlock &MBB : Fn)
    Changed |= runOnBasicBlock(MBB);
  return Changed;
}

FunctionPass *llvm::createLoongArchLateBranchOptPass() {
  return new LoongArchLateBranchOpt();
}
