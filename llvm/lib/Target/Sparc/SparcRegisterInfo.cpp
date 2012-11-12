//===-- SparcRegisterInfo.cpp - SPARC Register Information ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the SPARC implementation of the TargetRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#include "SparcRegisterInfo.h"
#include "Sparc.h"
#include "SparcSubtarget.h"
#include "llvm/Type.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"

#define GET_REGINFO_TARGET_DESC
#include "SparcGenRegisterInfo.inc"

using namespace llvm;

SparcRegisterInfo::SparcRegisterInfo(SparcSubtarget &st,
                                     const TargetInstrInfo &tii)
  : SparcGenRegisterInfo(SP::I7), Subtarget(st), TII(tii) {
}

const uint16_t* SparcRegisterInfo::getCalleeSavedRegs(const MachineFunction *MF)
                                                                         const {
  static const uint16_t CalleeSavedRegs[] = { 0 };
  return CalleeSavedRegs;
}

BitVector SparcRegisterInfo::getReservedRegs(const MachineFunction &MF) const {
  BitVector Reserved(getNumRegs());
  // FIXME: G1 reserved for now for large imm generation by frame code.
  Reserved.set(SP::G1);
  Reserved.set(SP::G2);
  Reserved.set(SP::G3);
  Reserved.set(SP::G4);
  Reserved.set(SP::O6);
  Reserved.set(SP::I6);
  Reserved.set(SP::I7);
  Reserved.set(SP::G0);
  Reserved.set(SP::G5);
  Reserved.set(SP::G6);
  Reserved.set(SP::G7);
  return Reserved;
}

void SparcRegisterInfo::
eliminateCallFramePseudoInstr(MachineFunction &MF, MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator I) const {
  MachineInstr &MI = *I;
  DebugLoc dl = MI.getDebugLoc();
  int Size = MI.getOperand(0).getImm();
  if (MI.getOpcode() == SP::ADJCALLSTACKDOWN)
    Size = -Size;
  if (Size)
    BuildMI(MBB, I, dl, TII.get(SP::ADDri), SP::O6).addReg(SP::O6).addImm(Size);
  MBB.erase(I);
}

void
SparcRegisterInfo::eliminateFrameIndex(MachineBasicBlock::iterator II,
                                       int SPAdj, RegScavenger *RS) const {
  assert(SPAdj == 0 && "Unexpected");

  unsigned i = 0;
  MachineInstr &MI = *II;
  DebugLoc dl = MI.getDebugLoc();
  while (!MI.getOperand(i).isFI()) {
    ++i;
    assert(i < MI.getNumOperands() && "Instr doesn't have FrameIndex operand!");
  }

  int FrameIndex = MI.getOperand(i).getIndex();

  // Addressable stack objects are accessed using neg. offsets from %fp
  MachineFunction &MF = *MI.getParent()->getParent();
  int Offset = MF.getFrameInfo()->getObjectOffset(FrameIndex) +
               MI.getOperand(i+1).getImm();

  // Replace frame index with a frame pointer reference.
  if (Offset >= -4096 && Offset <= 4095) {
    // If the offset is small enough to fit in the immediate field, directly
    // encode it.
    MI.getOperand(i).ChangeToRegister(SP::I6, false);
    MI.getOperand(i+1).ChangeToImmediate(Offset);
  } else {
    // Otherwise, emit a G1 = SETHI %hi(offset).  FIXME: it would be better to 
    // scavenge a register here instead of reserving G1 all of the time.
    unsigned OffHi = (unsigned)Offset >> 10U;
    BuildMI(*MI.getParent(), II, dl, TII.get(SP::SETHIi), SP::G1).addImm(OffHi);
    // Emit G1 = G1 + I6
    BuildMI(*MI.getParent(), II, dl, TII.get(SP::ADDrr), SP::G1).addReg(SP::G1)
      .addReg(SP::I6);
    // Insert: G1+%lo(offset) into the user.
    MI.getOperand(i).ChangeToRegister(SP::G1, false);
    MI.getOperand(i+1).ChangeToImmediate(Offset & ((1 << 10)-1));
  }
}

unsigned SparcRegisterInfo::getFrameRegister(const MachineFunction &MF) const {
  return SP::I6;
}

unsigned SparcRegisterInfo::getEHExceptionRegister() const {
  llvm_unreachable("What is the exception register");
}

unsigned SparcRegisterInfo::getEHHandlerRegister() const {
  llvm_unreachable("What is the exception handler register");
}
