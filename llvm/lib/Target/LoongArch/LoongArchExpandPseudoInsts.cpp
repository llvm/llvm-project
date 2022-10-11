//===-- LoongArchExpandPseudoInsts.cpp - Expand pseudo instructions -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a pass that expands pseudo instructions into target
// instructions.
//
//===----------------------------------------------------------------------===//

#include "LoongArch.h"
#include "LoongArchInstrInfo.h"
#include "LoongArchTargetMachine.h"
#include "MCTargetDesc/LoongArchBaseInfo.h"
#include "MCTargetDesc/LoongArchMCTargetDesc.h"
#include "llvm/CodeGen/LivePhysRegs.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/MC/MCContext.h"

using namespace llvm;

#define LOONGARCH_PRERA_EXPAND_PSEUDO_NAME                                     \
  "LoongArch Pre-RA pseudo instruction expansion pass"

namespace {

class LoongArchPreRAExpandPseudo : public MachineFunctionPass {
public:
  const LoongArchInstrInfo *TII;
  static char ID;

  LoongArchPreRAExpandPseudo() : MachineFunctionPass(ID) {
    initializeLoongArchPreRAExpandPseudoPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
  StringRef getPassName() const override {
    return LOONGARCH_PRERA_EXPAND_PSEUDO_NAME;
  }

private:
  bool expandMBB(MachineBasicBlock &MBB);
  bool expandMI(MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
                MachineBasicBlock::iterator &NextMBBI);
  bool expandPcalau12iInstPair(MachineBasicBlock &MBB,
                               MachineBasicBlock::iterator MBBI,
                               MachineBasicBlock::iterator &NextMBBI,
                               unsigned FlagsHi, unsigned SecondOpcode,
                               unsigned FlagsLo);
  bool expandLoadAddressPcrel(MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator MBBI,
                              MachineBasicBlock::iterator &NextMBBI);
  bool expandLoadAddressGot(MachineBasicBlock &MBB,
                            MachineBasicBlock::iterator MBBI,
                            MachineBasicBlock::iterator &NextMBBI);
  bool expandLoadAddressTLSLE(MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator MBBI,
                              MachineBasicBlock::iterator &NextMBBI);
  bool expandLoadAddressTLSIE(MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator MBBI,
                              MachineBasicBlock::iterator &NextMBBI);
  bool expandLoadAddressTLSLD(MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator MBBI,
                              MachineBasicBlock::iterator &NextMBBI);
  bool expandLoadAddressTLSGD(MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator MBBI,
                              MachineBasicBlock::iterator &NextMBBI);
};

char LoongArchPreRAExpandPseudo::ID = 0;

bool LoongArchPreRAExpandPseudo::runOnMachineFunction(MachineFunction &MF) {
  TII =
      static_cast<const LoongArchInstrInfo *>(MF.getSubtarget().getInstrInfo());
  bool Modified = false;
  for (auto &MBB : MF)
    Modified |= expandMBB(MBB);
  return Modified;
}

bool LoongArchPreRAExpandPseudo::expandMBB(MachineBasicBlock &MBB) {
  bool Modified = false;

  MachineBasicBlock::iterator MBBI = MBB.begin(), E = MBB.end();
  while (MBBI != E) {
    MachineBasicBlock::iterator NMBBI = std::next(MBBI);
    Modified |= expandMI(MBB, MBBI, NMBBI);
    MBBI = NMBBI;
  }

  return Modified;
}

bool LoongArchPreRAExpandPseudo::expandMI(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
    MachineBasicBlock::iterator &NextMBBI) {
  switch (MBBI->getOpcode()) {
  case LoongArch::PseudoLA_PCREL:
    return expandLoadAddressPcrel(MBB, MBBI, NextMBBI);
  case LoongArch::PseudoLA_GOT:
    return expandLoadAddressGot(MBB, MBBI, NextMBBI);
  case LoongArch::PseudoLA_TLS_LE:
    return expandLoadAddressTLSLE(MBB, MBBI, NextMBBI);
  case LoongArch::PseudoLA_TLS_IE:
    return expandLoadAddressTLSIE(MBB, MBBI, NextMBBI);
  case LoongArch::PseudoLA_TLS_LD:
    return expandLoadAddressTLSLD(MBB, MBBI, NextMBBI);
  case LoongArch::PseudoLA_TLS_GD:
    return expandLoadAddressTLSGD(MBB, MBBI, NextMBBI);
  }
  return false;
}

bool LoongArchPreRAExpandPseudo::expandPcalau12iInstPair(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
    MachineBasicBlock::iterator &NextMBBI, unsigned FlagsHi,
    unsigned SecondOpcode, unsigned FlagsLo) {
  MachineFunction *MF = MBB.getParent();
  MachineInstr &MI = *MBBI;
  DebugLoc DL = MI.getDebugLoc();

  Register DestReg = MI.getOperand(0).getReg();
  Register ScratchReg =
      MF->getRegInfo().createVirtualRegister(&LoongArch::GPRRegClass);
  MachineOperand &Symbol = MI.getOperand(1);

  BuildMI(MBB, MBBI, DL, TII->get(LoongArch::PCALAU12I), ScratchReg)
      .addDisp(Symbol, 0, FlagsHi);

  MachineInstr *SecondMI =
      BuildMI(MBB, MBBI, DL, TII->get(SecondOpcode), DestReg)
          .addReg(ScratchReg)
          .addDisp(Symbol, 0, FlagsLo);

  if (MI.hasOneMemOperand())
    SecondMI->addMemOperand(*MF, *MI.memoperands_begin());

  MI.eraseFromParent();
  return true;
}

bool LoongArchPreRAExpandPseudo::expandLoadAddressPcrel(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
    MachineBasicBlock::iterator &NextMBBI) {
  // Code Sequence:
  // pcalau12i $rd, %pc_hi20(sym)
  // addi.w/d $rd, $rd, %pc_lo12(sym)
  MachineFunction *MF = MBB.getParent();
  const auto &STI = MF->getSubtarget<LoongArchSubtarget>();
  unsigned SecondOpcode = STI.is64Bit() ? LoongArch::ADDI_D : LoongArch::ADDI_W;
  return expandPcalau12iInstPair(MBB, MBBI, NextMBBI, LoongArchII::MO_PCREL_HI,
                                 SecondOpcode, LoongArchII::MO_PCREL_LO);
}

bool LoongArchPreRAExpandPseudo::expandLoadAddressGot(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
    MachineBasicBlock::iterator &NextMBBI) {
  // Code Sequence:
  // pcalau12i $rd, %got_pc_hi20(sym)
  // ld.w/d $rd, $rd, %got_pc_lo12(sym)
  MachineFunction *MF = MBB.getParent();
  const auto &STI = MF->getSubtarget<LoongArchSubtarget>();
  unsigned SecondOpcode = STI.is64Bit() ? LoongArch::LD_D : LoongArch::LD_W;
  return expandPcalau12iInstPair(MBB, MBBI, NextMBBI, LoongArchII::MO_GOT_PC_HI,
                                 SecondOpcode, LoongArchII::MO_GOT_PC_LO);
}

bool LoongArchPreRAExpandPseudo::expandLoadAddressTLSLE(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
    MachineBasicBlock::iterator &NextMBBI) {
  // Code Sequence:
  // lu12i.w $rd, %le_hi20(sym)
  // ori $rd, $rd, %le_lo12(sym)
  MachineFunction *MF = MBB.getParent();
  MachineInstr &MI = *MBBI;
  DebugLoc DL = MI.getDebugLoc();

  Register DestReg = MI.getOperand(0).getReg();
  Register ScratchReg =
      MF->getRegInfo().createVirtualRegister(&LoongArch::GPRRegClass);
  MachineOperand &Symbol = MI.getOperand(1);

  BuildMI(MBB, MBBI, DL, TII->get(LoongArch::LU12I_W), ScratchReg)
      .addDisp(Symbol, 0, LoongArchII::MO_LE_HI);

  BuildMI(MBB, MBBI, DL, TII->get(LoongArch::ORI), DestReg)
      .addReg(ScratchReg)
      .addDisp(Symbol, 0, LoongArchII::MO_LE_LO);

  MI.eraseFromParent();
  return true;
}

bool LoongArchPreRAExpandPseudo::expandLoadAddressTLSIE(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
    MachineBasicBlock::iterator &NextMBBI) {
  // Code Sequence:
  // pcalau12i $rd, %ie_pc_hi20(sym)
  // ld.w/d $rd, $rd, %ie_pc_lo12(sym)
  MachineFunction *MF = MBB.getParent();
  const auto &STI = MF->getSubtarget<LoongArchSubtarget>();
  unsigned SecondOpcode = STI.is64Bit() ? LoongArch::LD_D : LoongArch::LD_W;
  return expandPcalau12iInstPair(MBB, MBBI, NextMBBI, LoongArchII::MO_IE_PC_HI,
                                 SecondOpcode, LoongArchII::MO_IE_PC_LO);
}

bool LoongArchPreRAExpandPseudo::expandLoadAddressTLSLD(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
    MachineBasicBlock::iterator &NextMBBI) {
  // Code Sequence:
  // pcalau12i $rd, %ld_pc_hi20(sym)
  // addi.w/d $rd, $rd, %got_pc_lo12(sym)
  MachineFunction *MF = MBB.getParent();
  const auto &STI = MF->getSubtarget<LoongArchSubtarget>();
  unsigned SecondOpcode = STI.is64Bit() ? LoongArch::ADDI_D : LoongArch::ADDI_W;
  return expandPcalau12iInstPair(MBB, MBBI, NextMBBI, LoongArchII::MO_LD_PC_HI,
                                 SecondOpcode, LoongArchII::MO_GOT_PC_LO);
}

bool LoongArchPreRAExpandPseudo::expandLoadAddressTLSGD(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
    MachineBasicBlock::iterator &NextMBBI) {
  // Code Sequence:
  // pcalau12i $rd, %gd_pc_hi20(sym)
  // addi.w/d $rd, $rd, %got_pc_lo12(sym)
  MachineFunction *MF = MBB.getParent();
  const auto &STI = MF->getSubtarget<LoongArchSubtarget>();
  unsigned SecondOpcode = STI.is64Bit() ? LoongArch::ADDI_D : LoongArch::ADDI_W;
  return expandPcalau12iInstPair(MBB, MBBI, NextMBBI, LoongArchII::MO_GD_PC_HI,
                                 SecondOpcode, LoongArchII::MO_GOT_PC_LO);
}

} // end namespace

INITIALIZE_PASS(LoongArchPreRAExpandPseudo, "LoongArch-prera-expand-pseudo",
                LOONGARCH_PRERA_EXPAND_PSEUDO_NAME, false, false)

namespace llvm {

FunctionPass *createLoongArchPreRAExpandPseudoPass() {
  return new LoongArchPreRAExpandPseudo();
}

} // end namespace llvm
