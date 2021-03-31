//===-- RISCVExpandPseudoInsts.cpp - Expand pseudo instructions -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a pass that expands pseudo instructions into target
// instructions. This pass should be run after register allocation but before
// the post-regalloc scheduling pass.
//
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVInstrInfo.h"
#include "RISCVTargetMachine.h"

#include "llvm/CodeGen/LivePhysRegs.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"

using namespace llvm;

#define RISCV_EXPAND_PSEUDO_NAME "RISCV pseudo instruction expansion pass"

namespace {

class RISCVExpandPseudo : public MachineFunctionPass {
public:
  const RISCVInstrInfo *TII;
  static char ID;

  RISCVExpandPseudo() : MachineFunctionPass(ID) {
    initializeRISCVExpandPseudoPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override { return RISCV_EXPAND_PSEUDO_NAME; }

private:
  bool expandMBB(MachineBasicBlock &MBB);
  bool expandMI(MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
                MachineBasicBlock::iterator &NextMBBI);
  bool expandAuipcInstPair(MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator MBBI,
                           MachineBasicBlock::iterator &NextMBBI,
                           unsigned FlagsHi, unsigned SecondOpcode);
  bool expandLoadLocalAddress(MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator MBBI,
                              MachineBasicBlock::iterator &NextMBBI);
  bool expandLoadAddress(MachineBasicBlock &MBB,
                         MachineBasicBlock::iterator MBBI,
                         MachineBasicBlock::iterator &NextMBBI);
  bool expandLoadTLSIEAddress(MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator MBBI,
                              MachineBasicBlock::iterator &NextMBBI);
  bool expandLoadTLSGDAddress(MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator MBBI,
                              MachineBasicBlock::iterator &NextMBBI);
  bool expandVSetVL(MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI);
  bool expandVMSET_VMCLR(MachineBasicBlock &MBB,
                         MachineBasicBlock::iterator MBBI, unsigned Opcode);
  bool expandVSPILL(MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI);
  bool expandVRELOAD(MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI);
};

char RISCVExpandPseudo::ID = 0;

bool RISCVExpandPseudo::runOnMachineFunction(MachineFunction &MF) {
  TII = static_cast<const RISCVInstrInfo *>(MF.getSubtarget().getInstrInfo());
  bool Modified = false;
  for (auto &MBB : MF)
    Modified |= expandMBB(MBB);
  return Modified;
}

bool RISCVExpandPseudo::expandMBB(MachineBasicBlock &MBB) {
  bool Modified = false;

  MachineBasicBlock::iterator MBBI = MBB.begin(), E = MBB.end();
  while (MBBI != E) {
    MachineBasicBlock::iterator NMBBI = std::next(MBBI);
    Modified |= expandMI(MBB, MBBI, NMBBI);
    MBBI = NMBBI;
  }

  return Modified;
}

bool RISCVExpandPseudo::expandMI(MachineBasicBlock &MBB,
                                 MachineBasicBlock::iterator MBBI,
                                 MachineBasicBlock::iterator &NextMBBI) {
  // RISCVInstrInfo::getInstSizeInBytes hard-codes the number of expanded
  // instructions for each pseudo, and must be updated when adding new pseudos
  // or changing existing ones.
  switch (MBBI->getOpcode()) {
  case RISCV::PseudoLLA:
    return expandLoadLocalAddress(MBB, MBBI, NextMBBI);
  case RISCV::PseudoLA:
    return expandLoadAddress(MBB, MBBI, NextMBBI);
  case RISCV::PseudoLA_TLS_IE:
    return expandLoadTLSIEAddress(MBB, MBBI, NextMBBI);
  case RISCV::PseudoLA_TLS_GD:
    return expandLoadTLSGDAddress(MBB, MBBI, NextMBBI);
  case RISCV::PseudoVSETVLI:
  case RISCV::PseudoVSETIVLI:
    return expandVSetVL(MBB, MBBI);
  case RISCV::PseudoVMCLR_M_B1:
  case RISCV::PseudoVMCLR_M_B2:
  case RISCV::PseudoVMCLR_M_B4:
  case RISCV::PseudoVMCLR_M_B8:
  case RISCV::PseudoVMCLR_M_B16:
  case RISCV::PseudoVMCLR_M_B32:
  case RISCV::PseudoVMCLR_M_B64:
    // vmclr.m vd => vmxor.mm vd, vd, vd
    return expandVMSET_VMCLR(MBB, MBBI, RISCV::VMXOR_MM);
  case RISCV::PseudoVMSET_M_B1:
  case RISCV::PseudoVMSET_M_B2:
  case RISCV::PseudoVMSET_M_B4:
  case RISCV::PseudoVMSET_M_B8:
  case RISCV::PseudoVMSET_M_B16:
  case RISCV::PseudoVMSET_M_B32:
  case RISCV::PseudoVMSET_M_B64:
    // vmset.m vd => vmxnor.mm vd, vd, vd
    return expandVMSET_VMCLR(MBB, MBBI, RISCV::VMXNOR_MM);
  case RISCV::PseudoVSPILL2_M1:
  case RISCV::PseudoVSPILL2_M2:
  case RISCV::PseudoVSPILL2_M4:
  case RISCV::PseudoVSPILL3_M1:
  case RISCV::PseudoVSPILL3_M2:
  case RISCV::PseudoVSPILL4_M1:
  case RISCV::PseudoVSPILL4_M2:
  case RISCV::PseudoVSPILL5_M1:
  case RISCV::PseudoVSPILL6_M1:
  case RISCV::PseudoVSPILL7_M1:
  case RISCV::PseudoVSPILL8_M1:
    return expandVSPILL(MBB, MBBI);
  case RISCV::PseudoVRELOAD2_M1:
  case RISCV::PseudoVRELOAD2_M2:
  case RISCV::PseudoVRELOAD2_M4:
  case RISCV::PseudoVRELOAD3_M1:
  case RISCV::PseudoVRELOAD3_M2:
  case RISCV::PseudoVRELOAD4_M1:
  case RISCV::PseudoVRELOAD4_M2:
  case RISCV::PseudoVRELOAD5_M1:
  case RISCV::PseudoVRELOAD6_M1:
  case RISCV::PseudoVRELOAD7_M1:
  case RISCV::PseudoVRELOAD8_M1:
    return expandVRELOAD(MBB, MBBI);
  }

  return false;
}

bool RISCVExpandPseudo::expandAuipcInstPair(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
    MachineBasicBlock::iterator &NextMBBI, unsigned FlagsHi,
    unsigned SecondOpcode) {
  MachineFunction *MF = MBB.getParent();
  MachineInstr &MI = *MBBI;
  DebugLoc DL = MI.getDebugLoc();

  Register DestReg = MI.getOperand(0).getReg();
  const MachineOperand &Symbol = MI.getOperand(1);

  MachineBasicBlock *NewMBB = MF->CreateMachineBasicBlock(MBB.getBasicBlock());

  // Tell AsmPrinter that we unconditionally want the symbol of this label to be
  // emitted.
  NewMBB->setLabelMustBeEmitted();

  MF->insert(++MBB.getIterator(), NewMBB);

  BuildMI(NewMBB, DL, TII->get(RISCV::AUIPC), DestReg)
      .addDisp(Symbol, 0, FlagsHi);
  BuildMI(NewMBB, DL, TII->get(SecondOpcode), DestReg)
      .addReg(DestReg)
      .addMBB(NewMBB, RISCVII::MO_PCREL_LO);

  // Move all the rest of the instructions to NewMBB.
  NewMBB->splice(NewMBB->end(), &MBB, std::next(MBBI), MBB.end());
  // Update machine-CFG edges.
  NewMBB->transferSuccessorsAndUpdatePHIs(&MBB);
  // Make the original basic block fall-through to the new.
  MBB.addSuccessor(NewMBB);

  // Make sure live-ins are correctly attached to this new basic block.
  LivePhysRegs LiveRegs;
  computeAndAddLiveIns(LiveRegs, *NewMBB);

  NextMBBI = MBB.end();
  MI.eraseFromParent();
  return true;
}

bool RISCVExpandPseudo::expandLoadLocalAddress(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
    MachineBasicBlock::iterator &NextMBBI) {
  return expandAuipcInstPair(MBB, MBBI, NextMBBI, RISCVII::MO_PCREL_HI,
                             RISCV::ADDI);
}

bool RISCVExpandPseudo::expandLoadAddress(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
    MachineBasicBlock::iterator &NextMBBI) {
  MachineFunction *MF = MBB.getParent();

  unsigned SecondOpcode;
  unsigned FlagsHi;
  if (MF->getTarget().isPositionIndependent()) {
    const auto &STI = MF->getSubtarget<RISCVSubtarget>();
    SecondOpcode = STI.is64Bit() ? RISCV::LD : RISCV::LW;
    FlagsHi = RISCVII::MO_GOT_HI;
  } else {
    SecondOpcode = RISCV::ADDI;
    FlagsHi = RISCVII::MO_PCREL_HI;
  }
  return expandAuipcInstPair(MBB, MBBI, NextMBBI, FlagsHi, SecondOpcode);
}

bool RISCVExpandPseudo::expandLoadTLSIEAddress(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
    MachineBasicBlock::iterator &NextMBBI) {
  MachineFunction *MF = MBB.getParent();

  const auto &STI = MF->getSubtarget<RISCVSubtarget>();
  unsigned SecondOpcode = STI.is64Bit() ? RISCV::LD : RISCV::LW;
  return expandAuipcInstPair(MBB, MBBI, NextMBBI, RISCVII::MO_TLS_GOT_HI,
                             SecondOpcode);
}

bool RISCVExpandPseudo::expandLoadTLSGDAddress(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
    MachineBasicBlock::iterator &NextMBBI) {
  return expandAuipcInstPair(MBB, MBBI, NextMBBI, RISCVII::MO_TLS_GD_HI,
                             RISCV::ADDI);
}

bool RISCVExpandPseudo::expandVSetVL(MachineBasicBlock &MBB,
                                     MachineBasicBlock::iterator MBBI) {
  assert(MBBI->getNumOperands() == 5 && "Unexpected instruction format");

  DebugLoc DL = MBBI->getDebugLoc();

  assert((MBBI->getOpcode() == RISCV::PseudoVSETVLI ||
          MBBI->getOpcode() == RISCV::PseudoVSETIVLI) &&
         "Unexpected pseudo instruction");
  unsigned Opcode;
  if (MBBI->getOpcode() == RISCV::PseudoVSETVLI)
    Opcode = RISCV::VSETVLI;
  else
    Opcode = RISCV::VSETIVLI;
  const MCInstrDesc &Desc = TII->get(Opcode);
  assert(Desc.getNumOperands() == 3 && "Unexpected instruction format");

  Register DstReg = MBBI->getOperand(0).getReg();
  bool DstIsDead = MBBI->getOperand(0).isDead();
  BuildMI(MBB, MBBI, DL, Desc)
      .addReg(DstReg, RegState::Define | getDeadRegState(DstIsDead))
      .add(MBBI->getOperand(1))  // VL
      .add(MBBI->getOperand(2)); // VType

  MBBI->eraseFromParent(); // The pseudo instruction is gone now.
  return true;
}

bool RISCVExpandPseudo::expandVMSET_VMCLR(MachineBasicBlock &MBB,
                                          MachineBasicBlock::iterator MBBI,
                                          unsigned Opcode) {
  DebugLoc DL = MBBI->getDebugLoc();
  Register DstReg = MBBI->getOperand(0).getReg();
  const MCInstrDesc &Desc = TII->get(Opcode);
  BuildMI(MBB, MBBI, DL, Desc, DstReg)
      .addReg(DstReg, RegState::Undef)
      .addReg(DstReg, RegState::Undef);
  MBBI->eraseFromParent(); // The pseudo instruction is gone now.
  return true;
}

bool RISCVExpandPseudo::expandVSPILL(MachineBasicBlock &MBB,
                                     MachineBasicBlock::iterator MBBI) {
  const TargetRegisterInfo *TRI =
      MBB.getParent()->getSubtarget().getRegisterInfo();
  DebugLoc DL = MBBI->getDebugLoc();
  Register SrcReg = MBBI->getOperand(0).getReg();
  Register Base = MBBI->getOperand(1).getReg();
  Register VL = MBBI->getOperand(2).getReg();
  auto ZvlssegInfo = TII->isRVVSpillForZvlsseg(MBBI->getOpcode());
  if (!ZvlssegInfo)
    return false;
  unsigned NF = ZvlssegInfo->first;
  unsigned LMUL = ZvlssegInfo->second;
  assert(NF * LMUL <= 8 && "Invalid NF/LMUL combinations.");
  unsigned Opcode = RISCV::VS1R_V;
  unsigned SubRegIdx = RISCV::sub_vrm1_0;
  static_assert(RISCV::sub_vrm1_7 == RISCV::sub_vrm1_0 + 7,
                "Unexpected subreg numbering");
  if (LMUL == 2) {
    Opcode = RISCV::VS2R_V;
    SubRegIdx = RISCV::sub_vrm2_0;
    static_assert(RISCV::sub_vrm2_3 == RISCV::sub_vrm2_0 + 3,
                  "Unexpected subreg numbering");
  } else if (LMUL == 4) {
    Opcode = RISCV::VS4R_V;
    SubRegIdx = RISCV::sub_vrm4_0;
    static_assert(RISCV::sub_vrm4_1 == RISCV::sub_vrm4_0 + 1,
                  "Unexpected subreg numbering");
  } else
    assert(LMUL == 1 && "LMUL must be 1, 2, or 4.");

  for (unsigned I = 0; I < NF; ++I) {
    BuildMI(MBB, MBBI, DL, TII->get(Opcode))
        .addReg(TRI->getSubReg(SrcReg, SubRegIdx + I))
        .addReg(Base)
        .addMemOperand(*(MBBI->memoperands_begin()));
    if (I != NF - 1)
      BuildMI(MBB, MBBI, DL, TII->get(RISCV::ADD), Base)
          .addReg(Base)
          .addReg(VL);
  }
  MBBI->eraseFromParent();
  return true;
}

bool RISCVExpandPseudo::expandVRELOAD(MachineBasicBlock &MBB,
                                      MachineBasicBlock::iterator MBBI) {
  const TargetRegisterInfo *TRI =
      MBB.getParent()->getSubtarget().getRegisterInfo();
  DebugLoc DL = MBBI->getDebugLoc();
  Register DestReg = MBBI->getOperand(0).getReg();
  Register Base = MBBI->getOperand(1).getReg();
  Register VL = MBBI->getOperand(2).getReg();
  auto ZvlssegInfo = TII->isRVVSpillForZvlsseg(MBBI->getOpcode());
  if (!ZvlssegInfo)
    return false;
  unsigned NF = ZvlssegInfo->first;
  unsigned LMUL = ZvlssegInfo->second;
  assert(NF * LMUL <= 8 && "Invalid NF/LMUL combinations.");
  unsigned Opcode = RISCV::VL1RE8_V;
  unsigned SubRegIdx = RISCV::sub_vrm1_0;
  static_assert(RISCV::sub_vrm1_7 == RISCV::sub_vrm1_0 + 7,
                "Unexpected subreg numbering");
  if (LMUL == 2) {
    Opcode = RISCV::VL2RE8_V;
    SubRegIdx = RISCV::sub_vrm2_0;
    static_assert(RISCV::sub_vrm2_3 == RISCV::sub_vrm2_0 + 3,
                  "Unexpected subreg numbering");
  } else if (LMUL == 4) {
    Opcode = RISCV::VL4RE8_V;
    SubRegIdx = RISCV::sub_vrm4_0;
    static_assert(RISCV::sub_vrm4_1 == RISCV::sub_vrm4_0 + 1,
                  "Unexpected subreg numbering");
  } else
    assert(LMUL == 1 && "LMUL must be 1, 2, or 4.");

  for (unsigned I = 0; I < NF; ++I) {
    BuildMI(MBB, MBBI, DL, TII->get(Opcode),
            TRI->getSubReg(DestReg, SubRegIdx + I))
        .addReg(Base)
        .addMemOperand(*(MBBI->memoperands_begin()));
    if (I != NF - 1)
      BuildMI(MBB, MBBI, DL, TII->get(RISCV::ADD), Base)
          .addReg(Base)
          .addReg(VL);
  }
  MBBI->eraseFromParent();
  return true;
}

} // end of anonymous namespace

INITIALIZE_PASS(RISCVExpandPseudo, "riscv-expand-pseudo",
                RISCV_EXPAND_PSEUDO_NAME, false, false)
namespace llvm {

FunctionPass *createRISCVExpandPseudoPass() { return new RISCVExpandPseudo(); }

} // end of namespace llvm
