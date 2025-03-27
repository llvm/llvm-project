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
  bool trySimplifyCondBr(MachineInstr *MI, MachineBasicBlock *TBB,
                         MachineBasicBlock *FBB,
                         SmallVectorImpl<MachineOperand> &Cond) const;

  const RISCVInstrInfo *RII = nullptr;
};
} // namespace

char RISCVLateOpt::ID = 0;
INITIALIZE_PASS(RISCVLateOpt, "riscv-late-opt", RISCV_LATE_OPT_NAME, false,
                false)

bool RISCVLateOpt::trySimplifyCondBr(
    MachineInstr *MI, MachineBasicBlock *TBB, MachineBasicBlock *FBB,
    SmallVectorImpl<MachineOperand> &Cond) const {

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

  MachineBasicBlock *MBB = MI->getParent();
  MachineRegisterInfo &MRI = MBB->getParent()->getRegInfo();
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
    switch (CC) {
    case RISCVCC::COND_INVALID:
      llvm_unreachable("Unexpected CC");
    case RISCVCC::COND_EQ: {
      Folded = (C0 == C1) ? TBB : FBB;
      break;
    }
    case RISCVCC::COND_NE: {
      Folded = (C0 != C1) ? TBB : FBB;
      break;
    }
    case RISCVCC::COND_LT: {
      Folded = (C0 < C1) ? TBB : FBB;
      break;
    }
    case RISCVCC::COND_GE: {
      Folded = (C0 >= C1) ? TBB : FBB;
      break;
    }
    case RISCVCC::COND_LTU: {
      Folded = ((uint64_t)C0 < (uint64_t)C1) ? TBB : FBB;
      break;
    }
    case RISCVCC::COND_GEU: {
      Folded = ((uint64_t)C0 >= (uint64_t)C1) ? TBB : FBB;
      break;
    }
    }

    // Do the conversion
    // Build the new unconditional branch
    DebugLoc DL = MBB->findBranchDebugLoc();
    if (Folded) {
      BuildMI(*MBB, MI, DL, RII->get(RISCV::PseudoBR)).addMBB(Folded);
    } else {
      MachineFunction::iterator Fallthrough = ++MBB->getIterator();
      if (Fallthrough == MBB->getParent()->end())
        return false;
      BuildMI(*MBB, MI, DL, RII->get(RISCV::PseudoBR)).addMBB(&*Fallthrough);
    }

    // Update successors of MBB->
    if (Folded == TBB) {
      // If we're taking TBB, then the succ to delete is the fallthrough (if
      // it was a succ in the first place), or its the MBB from the
      // unconditional branch.
      if (!FBB) {
        MachineFunction::iterator Fallthrough = ++MBB->getIterator();
        if (Fallthrough != MBB->getParent()->end() &&
            MBB->isSuccessor(&*Fallthrough))
          MBB->removeSuccessor(&*Fallthrough, true);
      } else {
        MBB->removeSuccessor(FBB, true);
      }
    } else if (Folded == FBB) {
      // If we're taking the fallthrough or unconditional branch, then the
      // succ to remove is the one from the conditional branch.
      MBB->removeSuccessor(TBB, true);
    }

    MI->eraseFromParent();
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
    for (MachineBasicBlock::iterator MII = MBB.begin(), MIE = MBB.end();
         MII != MIE;) {
      MachineInstr *MI = &*MII;
      // We may be erasing MI below, increment MII now.
      ++MII;
      if (!MI->isConditionalBranch())
        continue;

      MachineBasicBlock *TBB, *FBB;
      SmallVector<MachineOperand, 4> Cond;
      if (!RII->analyzeBranch(MBB, TBB, FBB, Cond, /*AllowModify=*/false))
        Changed |= trySimplifyCondBr(MI, TBB, FBB, Cond);
    }
  }

  return Changed;
}

/// Returns an instance of the Make Compressible Optimization pass.
FunctionPass *llvm::createRISCVLateOptPass() { return new RISCVLateOpt(); }
