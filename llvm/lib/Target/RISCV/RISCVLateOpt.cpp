//===-- RISCVLateOpt.cpp - Late stage optimization ------------------------===//
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

#include "MCTargetDesc/RISCVMCTargetDesc.h"
#include "RISCV.h"
#include "RISCVInstrInfo.h"
#include "RISCVSubtarget.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineBranchProbabilityInfo.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

#define DEBUG_TYPE "riscv-late-opt"
#define RISCV_LATE_OPT_NAME "RISC-V Late Stage Optimizations"

namespace {

struct RISCVLateOpt : public MachineFunctionPass {
  static char ID;

  RISCVLateOpt() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override { return RISCV_LATE_OPT_NAME; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &Fn) override;

private:
  bool trySimplifyCondBr(MachineBasicBlock &MBB, MachineBasicBlock *TBB,
                         MachineBasicBlock *FBB,
                         SmallVectorImpl<MachineOperand> &Cond) const;

  const RISCVInstrInfo *RII = nullptr;
};
} // namespace

char RISCVLateOpt::ID = 0;
INITIALIZE_PASS(RISCVLateOpt, "riscv-late-opt", RISCV_LATE_OPT_NAME, false,
                false)

static bool evaluateCondBranch(unsigned CC, int64_t C0, int64_t C1) {
  switch (CC) {
  default:
    llvm_unreachable("Unexpected CC");
  case RISCVCC::COND_EQ:
    return C0 == C1;
  case RISCVCC::COND_NE:
    return C0 != C1;
  case RISCVCC::COND_LT:
    return C0 < C1;
  case RISCVCC::COND_GE:
    return C0 >= C1;
  case RISCVCC::COND_LTU:
    return (uint64_t)C0 < (uint64_t)C1;
  case RISCVCC::COND_GEU:
    return (uint64_t)C0 >= (uint64_t)C1;
  }
}

bool RISCVLateOpt::trySimplifyCondBr(
    MachineBasicBlock &MBB, MachineBasicBlock *TBB, MachineBasicBlock *FBB,
    SmallVectorImpl<MachineOperand> &Cond) const {

  if (!TBB || Cond.size() != 3)
    return false;

  RISCVCC::CondCode CC = static_cast<RISCVCC::CondCode>(Cond[0].getImm());
  assert(CC != RISCVCC::COND_INVALID);

  // Right now we only care about LI (i.e. ADDI x0, imm)
  auto isLoadImm = [](const MachineInstr *MI, int64_t &Imm) -> bool {
    if (MI->getOpcode() == RISCV::ADDI && MI->getOperand(1).isReg() &&
        MI->getOperand(1).getReg() == RISCV::X0) {
      Imm = MI->getOperand(2).getImm();
      return true;
    }
    return false;
  };

  MachineRegisterInfo &MRI = MBB.getParent()->getRegInfo();
  // Either a load from immediate instruction or X0.
  auto isFromLoadImm = [&](const MachineOperand &Op, int64_t &Imm) -> bool {
    if (!Op.isReg())
      return false;
    Register Reg = Op.getReg();
    if (Reg == RISCV::X0) {
      Imm = 0;
      return true;
    }
    return Reg.isVirtual() && isLoadImm(MRI.getVRegDef(Reg), Imm);
  };

  // Try and convert a conditional branch that can be evaluated statically
  // into an unconditional branch.
  MachineBasicBlock *Folded = nullptr;
  int64_t C0, C1;
  if (isFromLoadImm(Cond[1], C0) && isFromLoadImm(Cond[2], C1)) {
    Folded = evaluateCondBranch(CC, C0, C1) ? TBB : FBB;

    // At this point, its legal to optimize.
    RII->removeBranch(MBB);
    Cond.clear();

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

    TBB = Folded;
    FBB = nullptr;
    return true;
  }

  return false;
}

bool RISCVLateOpt::runOnMachineFunction(MachineFunction &Fn) {
  if (skipFunction(Fn.getFunction()))
    return false;

  auto &ST = Fn.getSubtarget<RISCVSubtarget>();
  RII = ST.getInstrInfo();

  bool Changed = false;

  for (MachineBasicBlock &MBB : Fn) {
    MachineBasicBlock *TBB, *FBB;
    SmallVector<MachineOperand, 4> Cond;
    if (!RII->analyzeBranch(MBB, TBB, FBB, Cond, /*AllowModify=*/false))
      Changed |= trySimplifyCondBr(MBB, TBB, FBB, Cond);
  }

  return Changed;
}

/// Returns an instance of the Make Compressible Optimization pass.
FunctionPass *llvm::createRISCVLateOptPass() { return new RISCVLateOpt(); }
