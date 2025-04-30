//===-- Next32RegisterInfo.cpp - Next32 Register Information --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the Next32 implementation of the TargetRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#include "Next32RegisterInfo.h"
#include "Next32.h"
#include "Next32Subtarget.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/CodeGen/TargetFrameLowering.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/Support/ErrorHandling.h"

#define GET_REGINFO_TARGET_DESC
#include "Next32GenRegisterInfo.inc"
using namespace llvm;

Next32RegisterInfo::Next32RegisterInfo() : Next32GenRegisterInfo(Next32::R1) {}

const MCPhysReg *
Next32RegisterInfo::getCalleeSavedRegs(const MachineFunction *MF) const {
  static const MCPhysReg CSR_SaveList[] = {0};
  return CSR_SaveList;
}

BitVector Next32RegisterInfo::getReservedRegs(const MachineFunction &MF) const {
  BitVector Reserved(getNumRegs());
  Reserved.set(Next32::STACK_SIZE);
  Reserved.set(Next32::MBB_ADDR);
  Reserved.set(Next32::MBB_ADDR_1);
  Reserved.set(Next32::MBB_ADDR_2);
  Reserved.set(Next32::CALL_ADDR);
  Reserved.set(Next32::CALL_RET_BB);
  Reserved.set(Next32::CALL_RET_FID);
  return Reserved;
}

const uint32_t *Next32RegisterInfo::getNoPreservedMask() const {
  return nullptr;
}

bool Next32RegisterInfo::trackLivenessAfterRegAlloc(
    const MachineFunction &MF) const {
  return true;
}

bool Next32RegisterInfo::eliminateFrameIndex(MachineBasicBlock::iterator II,
                                             int SPAdj, unsigned FIOperandNum,
                                             RegScavenger *RS) const {
  MachineInstr &MI = *II;
  MachineBasicBlock *MBB = MI.getParent();
  const MachineFunction &MF = *(MBB->getParent());
  const MachineFrameInfo &MFI = MF.getFrameInfo();
  MachineOperand &FIOp = MI.getOperand(FIOperandNum);
  int FI = FIOp.getIndex();
  int64_t Offset = MFI.getObjectOffset(FI);
  FIOp.ChangeToImmediate(Offset);
  return false;
}

Register Next32RegisterInfo::getFrameRegister(const MachineFunction &MF) const {
  return Next32::SP_LOW;
}
