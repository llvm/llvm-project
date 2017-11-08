//===-- RISCVRegisterInfo.cpp - RISCV Register Information ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the RISCV implementation of the TargetRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#include "RISCVRegisterInfo.h"
#include "RISCV.h"
#include "RISCVSubtarget.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/CodeGen/TargetFrameLowering.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/Support/ErrorHandling.h"

#define GET_REGINFO_TARGET_DESC
#include "RISCVGenRegisterInfo.inc"

using namespace llvm;

RISCVRegisterInfo::RISCVRegisterInfo(unsigned HwMode)
    : RISCVGenRegisterInfo(RISCV::X1, /*DwarfFlavour*/0, /*EHFlavor*/0,
                           /*PC*/0, HwMode) {}

const MCPhysReg *
RISCVRegisterInfo::getCalleeSavedRegs(const MachineFunction *MF) const {
  return CSR_SaveList;
}

BitVector RISCVRegisterInfo::getReservedRegs(const MachineFunction &MF) const {
  BitVector Reserved(getNumRegs());

  // Use markSuperRegs to ensure any register aliases are also reserved
  markSuperRegs(Reserved, RISCV::X0); // zero
  markSuperRegs(Reserved, RISCV::X1); // ra
  markSuperRegs(Reserved, RISCV::X2); // sp
  markSuperRegs(Reserved, RISCV::X3); // gp
  markSuperRegs(Reserved, RISCV::X4); // tp
  markSuperRegs(Reserved, RISCV::X8); // fp
  assert(checkAllSuperRegsMarked(Reserved));
  return Reserved;
}

const uint32_t *RISCVRegisterInfo::getNoPreservedMask() const {
  return CSR_NoRegs_RegMask;
}

void RISCVRegisterInfo::eliminateFrameIndex(MachineBasicBlock::iterator II,
                                            int SPAdj, unsigned FIOperandNum,
                                            RegScavenger *RS) const {
  // TODO: this implementation is a temporary placeholder which does just
  // enough to allow other aspects of code generation to be tested

  assert(SPAdj == 0 && "Unexpected non-zero SPAdj value");

  MachineInstr &MI = *II;
  MachineFunction &MF = *MI.getParent()->getParent();
  const TargetFrameLowering *TFI = MF.getSubtarget().getFrameLowering();
  DebugLoc DL = MI.getDebugLoc();

  unsigned FrameReg = getFrameRegister(MF);
  int FrameIndex = MI.getOperand(FIOperandNum).getIndex();
  int Offset = TFI->getFrameIndexReference(MF, FrameIndex, FrameReg);
  Offset += MI.getOperand(FIOperandNum + 1).getImm();

  assert(TFI->hasFP(MF) && "eliminateFrameIndex currently requires hasFP");

  // Offsets must be directly encoded in a 12-bit immediate field
  if (!isInt<12>(Offset)) {
    report_fatal_error(
        "Frame offsets outside of the signed 12-bit range not supported");
  }

  MI.getOperand(FIOperandNum).ChangeToRegister(FrameReg, false);
  MI.getOperand(FIOperandNum + 1).ChangeToImmediate(Offset);
  return;
}

unsigned RISCVRegisterInfo::getFrameRegister(const MachineFunction &MF) const {
  return RISCV::X8;
}

const uint32_t *
RISCVRegisterInfo::getCallPreservedMask(const MachineFunction & /*MF*/,
                                        CallingConv::ID /*CC*/) const {
  return CSR_RegMask;
}
