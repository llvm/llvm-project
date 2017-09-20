//===-- Nios2RegisterInfo.h - Nios2 Register Information Impl ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the Nios2 implementation of the TargetRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_NIOS2_NIOS2REGISTERINFO_H
#define LLVM_LIB_TARGET_NIOS2_NIOS2REGISTERINFO_H

#include "Nios2.h"
#include "llvm/Target/TargetRegisterInfo.h"

#define GET_REGINFO_HEADER
#include "Nios2GenRegisterInfo.inc"

namespace llvm {
class Nios2Subtarget;
class TargetInstrInfo;
class Type;

class Nios2RegisterInfo : public Nios2GenRegisterInfo {
protected:
  const Nios2Subtarget &Subtarget;

public:
  Nios2RegisterInfo(const Nios2Subtarget &Subtarget);

  const MCPhysReg *getCalleeSavedRegs(const MachineFunction *MF) const override;

  BitVector getReservedRegs(const MachineFunction &MF) const override;

  /// Stack Frame Processing Methods
  void eliminateFrameIndex(MachineBasicBlock::iterator II, int SPAdj,
                           unsigned FIOperandNum,
                           RegScavenger *RS = nullptr) const override;

  /// Debug information queries.
  unsigned getFrameRegister(const MachineFunction &MF) const override;

  /// Return GPR register class.
  const TargetRegisterClass *intRegClass(unsigned Size) const;
};

} // end namespace llvm
#endif
