//===-- Mips16RegisterInfo.cpp - MIPS16 Register Information -== ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the MIPS16 implementation of the TargetRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#include "Mips16RegisterInfo.h"
#include "Mips16InstrInfo.h"
#include "Mips.h"
#include "MipsAnalyzeImmediate.h"
#include "MipsInstrInfo.h"
#include "MipsSubtarget.h"
#include "MipsMachineFunction.h"
#include "llvm/Constants.h"
#include "llvm/DebugInfo.h"
#include "llvm/Type.h"
#include "llvm/Function.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/Target/TargetFrameLowering.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"

using namespace llvm;

Mips16RegisterInfo::Mips16RegisterInfo(const MipsSubtarget &ST,
    const Mips16InstrInfo &I)
  : MipsRegisterInfo(ST), TII(I) {}

// This function eliminate ADJCALLSTACKDOWN,
// ADJCALLSTACKUP pseudo instructions
void Mips16RegisterInfo::
eliminateCallFramePseudoInstr(MachineFunction &MF, MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator I) const {
  const TargetFrameLowering *TFI = MF.getTarget().getFrameLowering();

  if (!TFI->hasReservedCallFrame(MF)) {
    int64_t Amount = I->getOperand(0).getImm();

    if (I->getOpcode() == Mips::ADJCALLSTACKDOWN)
      Amount = -Amount;

    const Mips16InstrInfo *II = static_cast<const Mips16InstrInfo*>(&TII);

    II->adjustStackPtr(Mips::SP, Amount, MBB, I);
  }

  MBB.erase(I);
}

void Mips16RegisterInfo::eliminateFI(MachineBasicBlock::iterator II,
                                     unsigned OpNo, int FrameIndex,
                                     uint64_t StackSize,
                                     int64_t SPOffset) const {
  MachineInstr &MI = *II;
  MachineFunction &MF = *MI.getParent()->getParent();
  MachineFrameInfo *MFI = MF.getFrameInfo();

  const std::vector<CalleeSavedInfo> &CSI = MFI->getCalleeSavedInfo();
  int MinCSFI = 0;
  int MaxCSFI = -1;

  if (CSI.size()) {
    MinCSFI = CSI[0].getFrameIdx();
    MaxCSFI = CSI[CSI.size() - 1].getFrameIdx();
  }

  // The following stack frame objects are always
  // referenced relative to $sp:
  //  1. Outgoing arguments.
  //  2. Pointer to dynamically allocated stack space.
  //  3. Locations for callee-saved registers.
  // Everything else is referenced relative to whatever register
  // getFrameRegister() returns.
  unsigned FrameReg;

  if (FrameIndex >= MinCSFI && FrameIndex <= MaxCSFI)
    FrameReg = Mips::SP;
  else {
    const TargetFrameLowering *TFI = MF.getTarget().getFrameLowering();
    if (TFI->hasFP(MF)) {
      FrameReg = Mips::S0;
    }
    else {
      if ((MI.getNumOperands()> OpNo+2) && MI.getOperand(OpNo+2).isReg())
        FrameReg = MI.getOperand(OpNo+2).getReg();
      else
        FrameReg = Mips::SP;
    }
  }
  // Calculate final offset.
  // - There is no need to change the offset if the frame object
  //   is one of the
  //   following: an outgoing argument, pointer to a dynamically allocated
  //   stack space or a $gp restore location,
  // - If the frame object is any of the following,
  //   its offset must be adjusted
  //   by adding the size of the stack:
  //   incoming argument, callee-saved register location or local variable.
  int64_t Offset;
  Offset = SPOffset + (int64_t)StackSize;
  Offset += MI.getOperand(OpNo + 1).getImm();


  DEBUG(errs() << "Offset     : " << Offset << "\n" << "<--------->\n");

  MI.getOperand(OpNo).ChangeToRegister(FrameReg, false);
  MI.getOperand(OpNo + 1).ChangeToImmediate(Offset);


}
