//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains one of the several passes that expand pseudo instructions
// into target instructions. This pass is run before register allocation.
//
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVExpandPseudoBase.h"
#include "RISCVInstrInfo.h"
#include "RISCVSubtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/MC/MCContext.h"

using namespace llvm;

#define RISCV_PRERA_EXPAND_PSEUDO_NAME                                         \
  "RISC-V Pre-RA pseudo instruction expansion pass"

namespace {

class RISCVPreRAExpandPseudoImpl final : public RISCVExpandPseudoImplBase {
  bool expandMI(MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
                MachineBasicBlock::iterator &NextMBBI) const override;

  bool expandAuipcInstPair(MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator MBBI, unsigned FlagsHi,
                           unsigned SecondOpcode) const;

  bool expandLoadLocalAddress(MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator MBBI) const;

  bool expandLoadGlobalAddress(MachineBasicBlock &MBB,
                               MachineBasicBlock::iterator MBBI) const;

  bool expandLoadTLSIEAddress(MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator MBBI) const;

  bool expandLoadTLSGDAddress(MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator MBBI) const;

  bool expandLoadTLSDescAddress(MachineBasicBlock &MBB,
                                MachineBasicBlock::iterator MBBI) const;
};

class RISCVPreRAExpandPseudoLegacy : public MachineFunctionPass {
public:
  static char ID;

  RISCVPreRAExpandPseudoLegacy() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override {
    return RISCVPreRAExpandPseudoImpl().run(MF);
  }

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().setIsSSA();
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
  StringRef getPassName() const override {
    return RISCV_PRERA_EXPAND_PSEUDO_NAME;
  }
};

} // anonymous namespace

bool RISCVPreRAExpandPseudoImpl::expandMI(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
    MachineBasicBlock::iterator &NextMBBI) const {

  switch (MBBI->getOpcode()) {
  case RISCV::PseudoLLA:
    return expandLoadLocalAddress(MBB, MBBI);
  case RISCV::PseudoLGA:
    return expandLoadGlobalAddress(MBB, MBBI);
  case RISCV::PseudoLA_TLS_IE:
    return expandLoadTLSIEAddress(MBB, MBBI);
  case RISCV::PseudoLA_TLS_GD:
    return expandLoadTLSGDAddress(MBB, MBBI);
  case RISCV::PseudoLA_TLSDESC:
    return expandLoadTLSDescAddress(MBB, MBBI);
  }

  return false;
}

bool RISCVPreRAExpandPseudoImpl::expandAuipcInstPair(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI, unsigned FlagsHi,
    unsigned SecondOpcode) const {
  MachineFunction *MF = MBB.getParent();
  MachineInstr &MI = *MBBI;
  DebugLoc DL = MI.getDebugLoc();

  Register DestReg = MI.getOperand(0).getReg();
  Register ScratchReg =
      MF->getRegInfo().createVirtualRegister(&RISCV::GPRRegClass);

  MachineOperand &Symbol = MI.getOperand(1);
  Symbol.setTargetFlags(FlagsHi);
  MCSymbol *AUIPCSymbol = MF->getContext().createNamedTempSymbol("pcrel_hi");

  MachineInstr *MIAUIPC =
      BuildMI(MBB, MBBI, DL, TII->get(RISCV::AUIPC), ScratchReg).add(Symbol);
  MIAUIPC->setPreInstrSymbol(*MF, AUIPCSymbol);

  MachineInstr *SecondMI =
      BuildMI(MBB, MBBI, DL, TII->get(SecondOpcode), DestReg)
          .addReg(ScratchReg)
          .addSym(AUIPCSymbol, RISCVII::MO_PCREL_LO);

  if (MI.hasOneMemOperand())
    SecondMI->addMemOperand(*MF, *MI.memoperands_begin());

  MI.eraseFromParent();
  return true;
}

bool RISCVPreRAExpandPseudoImpl::expandLoadLocalAddress(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI) const {
  return expandAuipcInstPair(MBB, MBBI, RISCVII::MO_PCREL_HI, RISCV::ADDI);
}

bool RISCVPreRAExpandPseudoImpl::expandLoadGlobalAddress(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI) const {
  unsigned SecondOpcode = STI->is64Bit() ? RISCV::LD : RISCV::LW;
  return expandAuipcInstPair(MBB, MBBI, RISCVII::MO_GOT_HI, SecondOpcode);
}

bool RISCVPreRAExpandPseudoImpl::expandLoadTLSIEAddress(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI) const {
  unsigned SecondOpcode = STI->is64Bit() ? RISCV::LD : RISCV::LW;
  return expandAuipcInstPair(MBB, MBBI, RISCVII::MO_TLS_GOT_HI, SecondOpcode);
}

bool RISCVPreRAExpandPseudoImpl::expandLoadTLSGDAddress(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI) const {
  return expandAuipcInstPair(MBB, MBBI, RISCVII::MO_TLS_GD_HI, RISCV::ADDI);
}

bool RISCVPreRAExpandPseudoImpl::expandLoadTLSDescAddress(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI) const {
  MachineFunction *MF = MBB.getParent();
  MachineInstr &MI = *MBBI;
  DebugLoc DL = MI.getDebugLoc();

  const auto &STI = MF->getSubtarget<RISCVSubtarget>();
  unsigned SecondOpcode = STI.is64Bit() ? RISCV::LD : RISCV::LW;

  Register FinalReg = MI.getOperand(0).getReg();
  Register DestReg =
      MF->getRegInfo().createVirtualRegister(&RISCV::GPRRegClass);
  Register ScratchReg =
      MF->getRegInfo().createVirtualRegister(&RISCV::GPRRegClass);

  MachineOperand &Symbol = MI.getOperand(1);
  Symbol.setTargetFlags(RISCVII::MO_TLSDESC_HI);
  MCSymbol *AUIPCSymbol = MF->getContext().createNamedTempSymbol("tlsdesc_hi");

  MachineInstr *MIAUIPC =
      BuildMI(MBB, MBBI, DL, TII->get(RISCV::AUIPC), ScratchReg).add(Symbol);
  MIAUIPC->setPreInstrSymbol(*MF, AUIPCSymbol);

  BuildMI(MBB, MBBI, DL, TII->get(SecondOpcode), DestReg)
      .addReg(ScratchReg)
      .addSym(AUIPCSymbol, RISCVII::MO_TLSDESC_LOAD_LO);

  BuildMI(MBB, MBBI, DL, TII->get(RISCV::ADDI), RISCV::X10)
      .addReg(ScratchReg)
      .addSym(AUIPCSymbol, RISCVII::MO_TLSDESC_ADD_LO);

  BuildMI(MBB, MBBI, DL, TII->get(RISCV::PseudoTLSDESCCall), RISCV::X5)
      .addReg(DestReg)
      .addImm(0)
      .addSym(AUIPCSymbol, RISCVII::MO_TLSDESC_CALL);

  BuildMI(MBB, MBBI, DL, TII->get(RISCV::ADD), FinalReg)
      .addReg(RISCV::X10)
      .addReg(RISCV::X4);

  MI.eraseFromParent();
  return true;
}

char RISCVPreRAExpandPseudoLegacy::ID = 0;

INITIALIZE_PASS(RISCVPreRAExpandPseudoLegacy, "riscv-pre-ra-expand-pseudo",
                RISCV_PRERA_EXPAND_PSEUDO_NAME, false, false)

FunctionPass *llvm::createRISCVPreRAExpandPseudoLegacyPass() {
  return new RISCVPreRAExpandPseudoLegacy();
}

PreservedAnalyses
RISCVPreRAExpandPseudoPass::run(MachineFunction &MF,
                                MachineFunctionAnalysisManager &MFAM) {
  MFPropsModifier _(*this, MF);
  bool Changed = RISCVPreRAExpandPseudoImpl().run(MF);
  if (!Changed)
    return PreservedAnalyses::all();

  PreservedAnalyses PA = getMachineFunctionPassPreservedAnalyses();
  PA.preserveSet<CFGAnalyses>();
  return PA;
}
