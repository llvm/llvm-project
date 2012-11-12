//===-- MSP430RegisterInfo.h - MSP430 Register Information Impl -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the MSP430 implementation of the MRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_MSP430REGISTERINFO_H
#define LLVM_TARGET_MSP430REGISTERINFO_H

#include "llvm/Target/TargetRegisterInfo.h"

#define GET_REGINFO_HEADER
#include "MSP430GenRegisterInfo.inc"

namespace llvm {

class TargetInstrInfo;
class MSP430TargetMachine;

struct MSP430RegisterInfo : public MSP430GenRegisterInfo {
private:
  MSP430TargetMachine &TM;
  const TargetInstrInfo &TII;

  /// StackAlign - Default stack alignment.
  ///
  unsigned StackAlign;
public:
  MSP430RegisterInfo(MSP430TargetMachine &tm, const TargetInstrInfo &tii);

  /// Code Generation virtual methods...
  const uint16_t *getCalleeSavedRegs(const MachineFunction *MF = 0) const;

  BitVector getReservedRegs(const MachineFunction &MF) const;
  const TargetRegisterClass*
  getPointerRegClass(const MachineFunction &MF, unsigned Kind = 0) const;

  void eliminateCallFramePseudoInstr(MachineFunction &MF,
                                     MachineBasicBlock &MBB,
                                     MachineBasicBlock::iterator I) const;

  void eliminateFrameIndex(MachineBasicBlock::iterator II,
                           int SPAdj, RegScavenger *RS = NULL) const;

  // Debug information queries.
  unsigned getFrameRegister(const MachineFunction &MF) const;
};

} // end namespace llvm

#endif // LLVM_TARGET_MSP430REGISTERINFO_H
