//===-- SuperHRegisterInfo.h - SuperH Register Information ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the SuperH implementation of the TargetRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_SUPERH_SUPERHREGISTERINFO_H
#define LLVM_LIB_TARGET_SUPERH_SUPERHREGISTERINFO_H

#include "llvm/CodeGen/TargetRegisterInfo.h"

#define GET_REGINFO_HEADER
#include "SuperHGenRegisterInfo.inc"

namespace llvm {
class SuperHSubtarget;

class SuperHRegisterInfo : public SuperHGenRegisterInfo {
protected:
  const SuperHSubtarget &Subtarget;

public:
  SuperHRegisterInfo(const SuperHSubtarget &Subtarget);

  const MCPhysReg *getCalleeSavedRegs(const MachineFunction *MF) const override;
  const uint32_t *getCallPreservedMask(const MachineFunction &MF, CallingConv::ID CC) const override;
  const TargetRegisterClass *getPointerRegClass(unsigned Kind = 0) const override;
  BitVector getReservedRegs(const MachineFunction &MF) const override;
  const TargetRegisterClass *getLargestLegalSuperClass(const TargetRegisterClass *RC,
                                           const MachineFunction &MF) const override;
  Register getFrameRegister(const MachineFunction &MF) const override;
  bool eliminateFrameIndex(MachineBasicBlock::iterator II, int SPAdj,
                           unsigned FIOperandNum,
                           RegScavenger *RS = nullptr) const override;

  // Helpers
  Register getFrameRegister() const;
  Register getStackRegister() const;
  Register getGOTRegister() const;
};

} // end namespace llvm

#endif // end LLVM_LIB_TARGET_SUPERH_SUPERHREGISTERINFO_H