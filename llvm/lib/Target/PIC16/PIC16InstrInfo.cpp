//===- PIC16InstrInfo.cpp - PIC16 Instruction Information -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source 
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the PIC16 implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "PIC16.h"
#include "PIC16InstrInfo.h"
#include "PIC16TargetMachine.h"
#include "PIC16GenInstrInfo.inc"
#include "llvm/Function.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include <cstdio>


using namespace llvm;

// FIXME: Add the subtarget support on this constructor.
PIC16InstrInfo::PIC16InstrInfo(PIC16TargetMachine &tm)
  : TargetInstrInfoImpl(PIC16Insts, array_lengthof(PIC16Insts)),
    TM(tm), 
    RegInfo(*this, *TM.getSubtargetImpl()) {}


/// isStoreToStackSlot - If the specified machine instruction is a direct
/// store to a stack slot, return the virtual or physical register number of
/// the source reg along with the FrameIndex of the loaded stack slot.  
/// If not, return 0.  This predicate must return 0 if the instruction has
/// any side effects other than storing to the stack slot.
unsigned PIC16InstrInfo::isStoreToStackSlot(const MachineInstr *MI,
                                            int &FrameIndex) const {
  if (MI->getOpcode() == PIC16::movwf 
      && MI->getOperand(0).isReg()
      && MI->getOperand(1).isSymbol()) {
    FrameIndex = MI->getOperand(1).getIndex();
    return MI->getOperand(0).getReg();
  }
  return 0;
}

/// isLoadFromStackSlot - If the specified machine instruction is a direct
/// load from a stack slot, return the virtual or physical register number of
/// the dest reg along with the FrameIndex of the stack slot.  
/// If not, return 0.  This predicate must return 0 if the instruction has
/// any side effects other than storing to the stack slot.
unsigned PIC16InstrInfo::isLoadFromStackSlot(const MachineInstr *MI,
                                            int &FrameIndex) const {
  if (MI->getOpcode() == PIC16::movf 
      && MI->getOperand(0).isReg()
      && MI->getOperand(1).isSymbol()) {
    FrameIndex = MI->getOperand(1).getIndex();
    return MI->getOperand(0).getReg();
  }
  return 0;
}


void PIC16InstrInfo::storeRegToStackSlot(MachineBasicBlock &MBB, 
                                         MachineBasicBlock::iterator I,
                                         unsigned SrcReg, bool isKill, int FI,
                                         const TargetRegisterClass *RC) const {

  const Function *Func = MBB.getParent()->getFunction();
  const std::string FuncName = Func->getName();

  char *tmpName = new char [strlen(FuncName.c_str()) +  6];
  sprintf(tmpName, "%s.tmp", FuncName.c_str());

  // On the order of operands here: think "movwf SrcReg, tmp_slot, offset".
  if (RC == PIC16::GPRRegisterClass) {
    //MachineFunction &MF = *MBB.getParent();
    //MachineRegisterInfo &RI = MF.getRegInfo();
    BuildMI(MBB, I, get(PIC16::movwf))
      .addReg(SrcReg, false, false, isKill)
      .addImm(FI)
      .addExternalSymbol(tmpName)
      .addImm(1); // Emit banksel for it.
  }
  else if (RC == PIC16::FSR16RegisterClass)
    assert(0 && "Don't know yet how to store a FSR16 to stack slot");
  else
    assert(0 && "Can't store this register to stack slot");
}

void PIC16InstrInfo::loadRegFromStackSlot(MachineBasicBlock &MBB, 
                                          MachineBasicBlock::iterator I,
                                          unsigned DestReg, int FI,
                                          const TargetRegisterClass *RC) const {

  const Function *Func = MBB.getParent()->getFunction();
  const std::string FuncName = Func->getName();

  char *tmpName = new char [strlen(FuncName.c_str()) +  6];
  sprintf(tmpName, "%s.tmp", FuncName.c_str());

  // On the order of operands here: think "movf FrameIndex, W".
  if (RC == PIC16::GPRRegisterClass) {
    //MachineFunction &MF = *MBB.getParent();
    //MachineRegisterInfo &RI = MF.getRegInfo();
    BuildMI(MBB, I, get(PIC16::movf), DestReg)
      .addImm(FI)
      .addExternalSymbol(tmpName)
      .addImm(1); // Emit banksel for it.
  }
  else if (RC == PIC16::FSR16RegisterClass)
    assert(0 && "Don't know yet how to load an FSR16 from stack slot");
  else
    assert(0 && "Can't load this register from stack slot");
}

bool PIC16InstrInfo::copyRegToReg (MachineBasicBlock &MBB,
                                   MachineBasicBlock::iterator I,
                                   unsigned DestReg, unsigned SrcReg,
                                   const TargetRegisterClass *DestRC,
                                   const TargetRegisterClass *SrcRC) const {
  if (DestRC == PIC16::FSR16RegisterClass) {
    BuildMI(MBB, I, get(PIC16::copy_fsr), DestReg).addReg(SrcReg);
    return true;
  }

  if (DestRC == PIC16::GPRRegisterClass) {
    BuildMI(MBB, I, get(PIC16::copy_w), DestReg).addReg(SrcReg);
    return true;
  }

  // Not yet supported.
  return false;
}

bool PIC16InstrInfo::isMoveInstr(const MachineInstr &MI,
                                 unsigned &SrcReg, unsigned &DestReg,
                                 unsigned &SrcSubIdx, unsigned &DstSubIdx) const {
  SrcSubIdx = DstSubIdx = 0; // No sub-registers.

  if (MI.getOpcode() == PIC16::copy_fsr
      || MI.getOpcode() == PIC16::copy_w) {
    DestReg = MI.getOperand(0).getReg();
    SrcReg = MI.getOperand(1).getReg();
    return true;
  }

  return false;
}

