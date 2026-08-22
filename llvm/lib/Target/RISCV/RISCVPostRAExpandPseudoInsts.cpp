//===-- RISCVPostRAExpandPseudoInsts.cpp - Expand pseudo instrs ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains one of the several passes that expand pseudo instructions
// into target instructions. This pass is run after register allocation and
// before post RA scheduling.
//
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVInstrInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"

using namespace llvm;

#define RISCV_POST_RA_EXPAND_PSEUDO_NAME                                       \
  "RISC-V post-regalloc pseudo instruction expansion pass"

namespace {

class RISCVPostRAExpandPseudoImpl {
public:
  const RISCVInstrInfo *TII;
  bool run(MachineFunction &MF);

private:
  bool expandMBB(MachineBasicBlock &MBB);
  bool expandMI(MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
                MachineBasicBlock::iterator &NextMBBI);
  bool expandMovImm(MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI);
  bool expandMovAddr(MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI);
  bool expandMERGE(MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI);
  bool expandAddUpperImm(MachineBasicBlock &MBB,
                         MachineBasicBlock::iterator MBBI);
};

class RISCVPostRAExpandPseudoLegacy : public MachineFunctionPass {
public:
  static char ID;

  RISCVPostRAExpandPseudoLegacy() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override {
    return RISCVPostRAExpandPseudoImpl().run(MF);
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  StringRef getPassName() const override {
    return RISCV_POST_RA_EXPAND_PSEUDO_NAME;
  }
};

char RISCVPostRAExpandPseudoLegacy::ID = 0;

bool RISCVPostRAExpandPseudoImpl::run(MachineFunction &MF) {
  TII = static_cast<const RISCVInstrInfo *>(MF.getSubtarget().getInstrInfo());
  bool Modified = false;
  for (auto &MBB : MF)
    Modified |= expandMBB(MBB);
  return Modified;
}

bool RISCVPostRAExpandPseudoImpl::expandMBB(MachineBasicBlock &MBB) {
  bool Modified = false;

  MachineBasicBlock::iterator MBBI = MBB.begin(), E = MBB.end();
  while (MBBI != E) {
    MachineBasicBlock::iterator NMBBI = std::next(MBBI);
    Modified |= expandMI(MBB, MBBI, NMBBI);
    MBBI = NMBBI;
  }

  return Modified;
}

bool RISCVPostRAExpandPseudoImpl::expandMI(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
    MachineBasicBlock::iterator &NextMBBI) {
  switch (MBBI->getOpcode()) {
  case RISCV::PseudoMovImm:
    return expandMovImm(MBB, MBBI);
  case RISCV::PseudoMovAddr:
    return expandMovAddr(MBB, MBBI);
  case RISCV::PseudoAddUpperImm:
    return expandAddUpperImm(MBB, MBBI);
  case RISCV::PseudoMERGE:
    return expandMERGE(MBB, MBBI);
  default:
    return false;
  }
}

bool RISCVPostRAExpandPseudoImpl::expandMovImm(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI) {
  DebugLoc DL = MBBI->getDebugLoc();

  int64_t Val = MBBI->getOperand(1).getImm();

  Register DstReg = MBBI->getOperand(0).getReg();
  bool DstIsDead = MBBI->getOperand(0).isDead();
  bool Renamable = MBBI->getOperand(0).isRenamable();

  TII->movImm(MBB, MBBI, DL, DstReg, Val, MachineInstr::NoFlags, Renamable,
              DstIsDead);

  MBBI->eraseFromParent();
  return true;
}

bool RISCVPostRAExpandPseudoImpl::expandMovAddr(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI) {
  DebugLoc DL = MBBI->getDebugLoc();

  Register DstReg = MBBI->getOperand(0).getReg();
  bool DstIsDead = MBBI->getOperand(0).isDead();
  bool Renamable = MBBI->getOperand(0).isRenamable();

  BuildMI(MBB, MBBI, DL, TII->get(RISCV::LUI))
      .addReg(DstReg, RegState::Define | getRenamableRegState(Renamable))
      .add(MBBI->getOperand(1));
  BuildMI(MBB, MBBI, DL, TII->get(RISCV::ADDI))
      .addReg(DstReg, RegState::Define | getDeadRegState(DstIsDead) |
                          getRenamableRegState(Renamable))
      .addReg(DstReg, RegState::Kill | getRenamableRegState(Renamable))
      .add(MBBI->getOperand(2));
  MBBI->eraseFromParent();
  return true;
}

bool RISCVPostRAExpandPseudoImpl::expandAddUpperImm(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI) {
  DebugLoc DL = MBBI->getDebugLoc();

  Register DstReg = MBBI->getOperand(0).getReg();
  bool DstIsDead = MBBI->getOperand(0).isDead();
  bool Renamable = MBBI->getOperand(0).isRenamable();
  Register BaseReg = MBBI->getOperand(1).getReg();
  int64_t Hi = MBBI->getOperand(2).getImm();

  // Expand to LUI+ADD: the immediate is already the upper 20-bit value.
  BuildMI(MBB, MBBI, DL, TII->get(RISCV::LUI))
      .addReg(DstReg, RegState::Define | getRenamableRegState(Renamable))
      .addImm(Hi);
  BuildMI(MBB, MBBI, DL, TII->get(RISCV::ADD))
      .addReg(DstReg, RegState::Define | getDeadRegState(DstIsDead) |
                          getRenamableRegState(Renamable))
      .addReg(BaseReg)
      .addReg(DstReg, RegState::Kill | getRenamableRegState(Renamable));

  MBBI->eraseFromParent();
  return true;
}

/// Transfer implicit operands on the pseudo instruction to the
/// instructions created from the expansion.
static void transferImpOps(MachineInstr &OldMI, MachineInstrBuilder &MI) {
  const MCInstrDesc &Desc = OldMI.getDesc();
  for (const MachineOperand &MO :
       llvm::drop_begin(OldMI.operands(), Desc.getNumOperands())) {
    assert(MO.isReg() && MO.getReg());
    MI.add(MO);
  }
}

// Expand PseudoMERGE to MERGE, MVM, or MVMN.
bool RISCVPostRAExpandPseudoImpl::expandMERGE(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI) {
  MachineInstr &MI = *MBBI;
  DebugLoc DL = MI.getDebugLoc();

  Register DstReg = MI.getOperand(0).getReg();
  if (DstReg == MI.getOperand(3).getReg()) {
    // Expand to MVMN
    auto I = BuildMI(MBB, MBBI, DL, TII->get(RISCV::MVMN))
                 .add(MI.getOperand(0))
                 .add(MI.getOperand(3))
                 .add(MI.getOperand(2))
                 .add(MI.getOperand(1));
    transferImpOps(*MBBI, I);
  } else if (DstReg == MBBI->getOperand(2).getReg()) {
    // Expand to MVM
    auto I = BuildMI(MBB, MBBI, DL, TII->get(RISCV::MVM))
                 .add(MI.getOperand(0))
                 .add(MI.getOperand(2))
                 .add(MI.getOperand(3))
                 .add(MI.getOperand(1));
    transferImpOps(*MBBI, I);
  } else if (DstReg == MI.getOperand(1).getReg()) {
    // Expand to MERGE
    auto I = BuildMI(MBB, MBBI, DL, TII->get(RISCV::MERGE))
                 .add(MI.getOperand(0))
                 .add(MI.getOperand(1))
                 .add(MI.getOperand(2))
                 .add(MI.getOperand(3));
    transferImpOps(*MBBI, I);
  } else {
    // Use an additional move.
    RegState RegState =
        getRenamableRegState(MI.getOperand(1).isRenamable()) |
        getKillRegState(MI.getOperand(1).isKill() &&
                        MI.getOperand(1).getReg() !=
                            MI.getOperand(2).getReg() &&
                        MI.getOperand(1).getReg() != MI.getOperand(3).getReg());
    BuildMI(MBB, MBBI, MI.getDebugLoc(), TII->get(RISCV::ADDI))
        .addDef(DstReg, getRenamableRegState(MI.getOperand(0).isRenamable()))
        .addReg(MI.getOperand(1).getReg(), RegState)
        .addImm(0);
    auto I = BuildMI(MBB, MBBI, MI.getDebugLoc(), TII->get(RISCV::MERGE))
                 .add(MI.getOperand(0))
                 .addReg(DstReg,
                         RegState::Kill | getRenamableRegState(
                                              MI.getOperand(0).isRenamable()))
                 .add(MI.getOperand(2))
                 .add(MI.getOperand(3));
    transferImpOps(*MBBI, I);
  }
  MI.eraseFromParent();
  return true;
}

} // end of anonymous namespace

INITIALIZE_PASS(RISCVPostRAExpandPseudoLegacy, "riscv-post-ra-expand-pseudo",
                RISCV_POST_RA_EXPAND_PSEUDO_NAME, false, false)
namespace llvm {

FunctionPass *createRISCVPostRAExpandPseudoLegacyPass() {
  return new RISCVPostRAExpandPseudoLegacy();
}

PreservedAnalyses
RISCVPostRAExpandPseudoPass::run(MachineFunction &MF,
                                 MachineFunctionAnalysisManager &MFAM) {
  bool Changed = RISCVPostRAExpandPseudoImpl().run(MF);
  if (!Changed)
    return PreservedAnalyses::all();

  PreservedAnalyses PA = getMachineFunctionPassPreservedAnalyses();
  PA.preserveSet<CFGAnalyses>();
  return PA;
}

} // end of namespace llvm
