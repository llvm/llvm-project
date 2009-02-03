//===- PIC16RegisterInfo.cpp - PIC16 Register Information -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source 
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the PIC16 implementation of the TargetRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "pic16-reg-info"

#include "PIC16.h"
#include "PIC16RegisterInfo.h"
#include "llvm/ADT/BitVector.h"


using namespace llvm;

PIC16RegisterInfo::PIC16RegisterInfo(const TargetInstrInfo &tii,
                                     const PIC16Subtarget &st)
  : PIC16GenRegisterInfo(PIC16::ADJCALLSTACKDOWN, PIC16::ADJCALLSTACKUP),
    TII(tii),
    ST(st) {}

#include "PIC16GenRegisterInfo.inc"

/// PIC16 Callee Saved Registers
const unsigned* PIC16RegisterInfo::
getCalleeSavedRegs(const MachineFunction *MF) const {
  static const unsigned CalleeSavedRegs[] = { 0 };
  return CalleeSavedRegs;
}

// PIC16 Callee Saved Reg Classes
const TargetRegisterClass* const*
PIC16RegisterInfo::getCalleeSavedRegClasses(const MachineFunction *MF) const {
  static const TargetRegisterClass * const CalleeSavedRegClasses[] = { 0 };
  return CalleeSavedRegClasses;
}

BitVector PIC16RegisterInfo::getReservedRegs(const MachineFunction &MF) const {
  BitVector Reserved(getNumRegs());
  return Reserved;
}

bool PIC16RegisterInfo::hasFP(const MachineFunction &MF) const {
  return false;
}

void PIC16RegisterInfo::
eliminateFrameIndex(MachineBasicBlock::iterator II, int SPAdj,
                    RegScavenger *RS) const
{    /* NOT YET IMPLEMENTED */  }

void PIC16RegisterInfo::emitPrologue(MachineFunction &MF) const
{    /* NOT YET IMPLEMENTED */  }

void PIC16RegisterInfo::
emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const
{    /* NOT YET IMPLEMENTED */  }

int PIC16RegisterInfo::
getDwarfRegNum(unsigned RegNum, bool isEH) const {
  assert(0 && "Not keeping track of debug information yet!!");
  return -1;
}

unsigned PIC16RegisterInfo::getFrameRegister(MachineFunction &MF) const {
  assert(0 && "PIC16 Does not have any frame register");
  return 0;
}

unsigned PIC16RegisterInfo::getRARegister() const {
  assert(0 && "PIC16 Does not have any return address register");
  return 0;
}

// This function eliminates ADJCALLSTACKDOWN,
// ADJCALLSTACKUP pseudo instructions
void PIC16RegisterInfo::
eliminateCallFramePseudoInstr(MachineFunction &MF, MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator I) const {
  // Simply discard ADJCALLSTACKDOWN,
  // ADJCALLSTACKUP instructions.
  MBB.erase(I);
}

