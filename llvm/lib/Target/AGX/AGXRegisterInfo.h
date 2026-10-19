//===-- AGXRegisterInfo.h - AGX register information ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AGX_AGXREGISTERINFO_H
#define LLVM_LIB_TARGET_AGX_AGXREGISTERINFO_H

#define GET_REGINFO_HEADER
#include "AGXGenRegisterInfo.inc"

namespace llvm {

class AGXSubtarget;

class AGXRegisterInfo : public AGXGenRegisterInfo {
  const AGXSubtarget &ST;

public:
  explicit AGXRegisterInfo(const AGXSubtarget &ST);

  const MCPhysReg *getCalleeSavedRegs(const MachineFunction *MF) const override;
  const uint32_t *getCallPreservedMask(const MachineFunction &MF,
                                       CallingConv::ID CC) const override;
  BitVector getReservedRegs(const MachineFunction &MF) const override;

  bool requiresRegisterScavenging(const MachineFunction &MF) const override {
    return false;
  }
  bool requiresFrameIndexScavenging(const MachineFunction &MF) const override {
    return false;
  }

  bool eliminateFrameIndex(MachineBasicBlock::iterator MI, int SPAdj,
                           unsigned FIOperandNum,
                           RegScavenger *RS = nullptr) const override;

  Register getFrameRegister(const MachineFunction &MF) const override;

  /// The general purpose register class of the same size as \p RC.
  const TargetRegisterClass *getGPRClassForSize(unsigned Bits) const;

  static bool isUniformClass(const TargetRegisterClass *RC);
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_AGX_AGXREGISTERINFO_H
