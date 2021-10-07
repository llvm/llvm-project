//===-- M88kRegisterInfo.cpp - M88k Register Information ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the M88k implementation of the TargetRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#include "M88kRegisterInfo.h"
#include "M88k.h"
//#include "M88kMachineFunctionInfo.h"
#include "M88kSubtarget.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

#define GET_REGINFO_TARGET_DESC
#include "M88kGenRegisterInfo.inc"

M88kRegisterInfo::M88kRegisterInfo() : M88kGenRegisterInfo(M88k::R1) {}

const MCPhysReg *
M88kRegisterInfo::getCalleeSavedRegs(const MachineFunction *MF) const {
  return CSR_M88k_SaveList;
}

BitVector M88kRegisterInfo::getReservedRegs(const MachineFunction &MF) const {
  BitVector Reserved(getNumRegs());

  // R0 is always reserved.
  Reserved.set(M88k::R0);

  // R28 and R29 are always reserved according to SYS-V ABI.
  Reserved.set(M88k::R28);
  Reserved.set(M88k::R29);

  // R31 is the stack pointer.
  Reserved.set(M88k::R31);

  return Reserved;
}

const uint32_t *
M88kRegisterInfo::getCallPreservedMask(const MachineFunction &MF,
                                       CallingConv::ID CC) const {
  return CSR_M88k_RegMask;
}

void M88kRegisterInfo::eliminateFrameIndex(MachineBasicBlock::iterator II,
                                           int SPAdj, unsigned FIOperandNum,
                                           RegScavenger *RS) const {
  MachineInstr &MI = *II;
  MachineFunction &MF = *MI.getParent()->getParent();
  MachineFrameInfo &MFI = MF.getFrameInfo();
  int FrameIndex = MI.getOperand(FIOperandNum).getIndex();

  const std::vector<CalleeSavedInfo> &CSI = MFI.getCalleeSavedInfo();
  int MinCSFI = 0;
  int MaxCSFI = -1;

  if (CSI.size()) {
    MinCSFI = CSI[0].getFrameIdx();
    MaxCSFI = CSI[CSI.size() - 1].getFrameIdx();
  }

  Register FrameReg;

  if (FrameIndex >= MinCSFI && FrameIndex <= MaxCSFI)
    FrameReg = M88k::R31;
  else {
    const TargetFrameLowering *TFI = MF.getSubtarget().getFrameLowering();
    if (TFI->hasFP(MF)) {
      FrameReg = M88k::R30;
    } else {
      FrameReg = M88k::R31;
    }
  }

  uint64_t StackSize = MF.getFrameInfo().getStackSize();
  int64_t SPOffset = MF.getFrameInfo().getObjectOffset(FrameIndex);

  int64_t Offset;
  bool IsKill = false;
  Offset = SPOffset + (int64_t)StackSize;
  Offset += MI.getOperand(FIOperandNum + 1).getImm();

  assert(isInt<16>(Offset) && "m88k: Larger offsets not yet supported.");
  MI.getOperand(FIOperandNum).ChangeToRegister(FrameReg, false, false, IsKill);
  MI.getOperand(FIOperandNum + 1).ChangeToImmediate(Offset);
}

Register M88kRegisterInfo::getFrameRegister(const MachineFunction &MF) const {
  return M88k::R30;
}
