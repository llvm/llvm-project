//===- PIC16RegisterInfo.h - PIC16 Register Information Impl ----*- C++ -*-===//
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

#ifndef PIC16REGISTERINFO_H
#define PIC16REGISTERINFO_H

#include "PIC16GenRegisterInfo.h.inc"
#include "llvm/Target/TargetRegisterInfo.h"

namespace llvm {

// Forward Declarations.
  class PIC16Subtarget;
  class TargetInstrInfo;

class PIC16RegisterInfo : public PIC16GenRegisterInfo {
  private:
    const TargetInstrInfo &TII;
    const PIC16Subtarget &ST;
  
  public:
    PIC16RegisterInfo(const TargetInstrInfo &tii, 
                      const PIC16Subtarget &st);


  //------------------------------------------------------
  // Pure virtual functions from TargetRegisterInfo
  //------------------------------------------------------

  // PIC16 callee saved registers
  virtual const unsigned* 
  getCalleeSavedRegs(const MachineFunction *MF = 0) const;

  // PIC16 callee saved register classes
  virtual const TargetRegisterClass* const *
  getCalleeSavedRegClasses(const MachineFunction *MF) const;

  virtual BitVector getReservedRegs(const MachineFunction &MF) const;
  virtual bool hasFP(const MachineFunction &MF) const;

  virtual void eliminateFrameIndex(MachineBasicBlock::iterator MI,
                        int SPAdj, RegScavenger *RS=NULL) const;

  void eliminateCallFramePseudoInstr(MachineFunction &MF,
                                     MachineBasicBlock &MBB,
                                     MachineBasicBlock::iterator I) const;

  virtual void emitPrologue(MachineFunction &MF) const;
  virtual void emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const;
  virtual int getDwarfRegNum(unsigned RegNum, bool isEH) const;
  virtual unsigned getFrameRegister(MachineFunction &MF) const;
  virtual unsigned getRARegister() const;

};

} // end namespace llvm

#endif
