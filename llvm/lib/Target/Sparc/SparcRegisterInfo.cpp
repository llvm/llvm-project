//===- SparcRegisterInfo.cpp - SPARC Register Information -------*- C++ -*-===//
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

#include "Sparc.h"
#include "SparcRegisterInfo.h"
#include "SparcSubtarget.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineLocation.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Type.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"
using namespace llvm;

SparcRegisterInfo::SparcRegisterInfo(SparcSubtarget &st,
                                     const TargetInstrInfo &tii)
  : SparcGenRegisterInfo(SP::ADJCALLSTACKDOWN, SP::ADJCALLSTACKUP),
    Subtarget(st), TII(tii) {
}

const unsigned* SparcRegisterInfo::getCalleeSavedRegs(const MachineFunction *MF)
                                                                         const {
  static const unsigned CalleeSavedRegs[] = { 0 };
  return CalleeSavedRegs;
}

BitVector SparcRegisterInfo::getReservedRegs(const MachineFunction &MF) const {
  BitVector Reserved(getNumRegs());
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


const TargetRegisterClass* const*
SparcRegisterInfo::getCalleeSavedRegClasses(const MachineFunction *MF) const {
  static const TargetRegisterClass * const CalleeSavedRegClasses[] = { 0 };
  return CalleeSavedRegClasses;
}

bool SparcRegisterInfo::hasFP(const MachineFunction &MF) const {
  return false;
}

void SparcRegisterInfo::
eliminateCallFramePseudoInstr(MachineFunction &MF, MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator I) const {
  MachineInstr &MI = *I;
  int Size = MI.getOperand(0).getImm();
  if (MI.getOpcode() == SP::ADJCALLSTACKDOWN)
    Size = -Size;
  if (Size)
    BuildMI(MBB, I, TII.get(SP::ADDri), SP::O6).addReg(SP::O6).addImm(Size);
  MBB.erase(I);
}

void SparcRegisterInfo::eliminateFrameIndex(MachineBasicBlock::iterator II,
                                            int SPAdj, RegScavenger *RS) const {
  assert(SPAdj == 0 && "Unexpected");

  unsigned i = 0;
  MachineInstr &MI = *II;
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
    BuildMI(*MI.getParent(), II, TII.get(SP::SETHIi), SP::G1).addImm(OffHi);
    // Emit G1 = G1 + I6
    BuildMI(*MI.getParent(), II, TII.get(SP::ADDrr), SP::G1).addReg(SP::G1)
      .addReg(SP::I6);
    // Insert: G1+%lo(offset) into the user.
    MI.getOperand(i).ChangeToRegister(SP::G1, false);
    MI.getOperand(i+1).ChangeToImmediate(Offset & ((1 << 10)-1));
  }
}

void SparcRegisterInfo::
processFunctionBeforeFrameFinalized(MachineFunction &MF) const {}

void SparcRegisterInfo::emitPrologue(MachineFunction &MF) const {
  MachineBasicBlock &MBB = MF.front();
  MachineFrameInfo *MFI = MF.getFrameInfo();

  // Get the number of bytes to allocate from the FrameInfo
  int NumBytes = (int) MFI->getStackSize();

  // Emit the correct save instruction based on the number of bytes in
  // the frame. Minimum stack frame size according to V8 ABI is:
  //   16 words for register window spill
  //    1 word for address of returned aggregate-value
  // +  6 words for passing parameters on the stack
  // ----------
  //   23 words * 4 bytes per word = 92 bytes
  NumBytes += 92;
  // Round up to next doubleword boundary -- a double-word boundary
  // is required by the ABI.
  NumBytes = (NumBytes + 7) & ~7;
  NumBytes = -NumBytes;
  
  if (NumBytes >= -4096) {
    BuildMI(MBB, MBB.begin(), TII.get(SP::SAVEri),
            SP::O6).addReg(SP::O6).addImm(NumBytes);
  } else {
    MachineBasicBlock::iterator InsertPt = MBB.begin();
    // Emit this the hard way.  This clobbers G1 which we always know is 
    // available here.
    unsigned OffHi = (unsigned)NumBytes >> 10U;
    BuildMI(MBB, InsertPt, TII.get(SP::SETHIi), SP::G1).addImm(OffHi);
    // Emit G1 = G1 + I6
    BuildMI(MBB, InsertPt, TII.get(SP::ORri), SP::G1)
      .addReg(SP::G1).addImm(NumBytes & ((1 << 10)-1));
    BuildMI(MBB, InsertPt, TII.get(SP::SAVErr), SP::O6)
      .addReg(SP::O6).addReg(SP::G1);
  }
}

void SparcRegisterInfo::emitEpilogue(MachineFunction &MF,
                                     MachineBasicBlock &MBB) const {
  MachineBasicBlock::iterator MBBI = prior(MBB.end());
  assert(MBBI->getOpcode() == SP::RETL &&
         "Can only put epilog before 'retl' instruction!");
  BuildMI(MBB, MBBI, TII.get(SP::RESTORErr), SP::G0).addReg(SP::G0)
    .addReg(SP::G0);
}

unsigned SparcRegisterInfo::getRARegister() const {
  assert(0 && "What is the return address register");
  return 0;
}

unsigned SparcRegisterInfo::getFrameRegister(MachineFunction &MF) const {
  assert(0 && "What is the frame register");
  return SP::G1;
}

unsigned SparcRegisterInfo::getEHExceptionRegister() const {
  assert(0 && "What is the exception register");
  return 0;
}

unsigned SparcRegisterInfo::getEHHandlerRegister() const {
  assert(0 && "What is the exception handler register");
  return 0;
}

int SparcRegisterInfo::getDwarfRegNum(unsigned RegNum, bool isEH) const {
  assert(0 && "What is the dwarf register number");
  return -1;
}

#include "SparcGenRegisterInfo.inc"

