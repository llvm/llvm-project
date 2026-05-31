//===-- AIRRegisterInfo.h - Define RegisterInfo for AIR -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the AIR specific subclass of TargetRegisterInfo.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AIR_AIRREGISTERINFO_H
#define LLVM_LIB_TARGET_AIR_AIRREGISTERINFO_H

#include "llvm/CodeGen/TargetRegisterInfo.h"

#define GET_REGINFO_HEADER
#include "AIRGenRegisterInfo.inc"

namespace llvm {
struct AIRRegisterInfo : public AIRGenRegisterInfo {
  AIRRegisterInfo() : AIRGenRegisterInfo(0) {}
  ~AIRRegisterInfo();

  const MCPhysReg *getCalleeSavedRegs(const MachineFunction *MF) const override;
  BitVector getReservedRegs(const MachineFunction &MF) const override;
  bool eliminateFrameIndex(MachineBasicBlock::iterator II, int SPAdj,
                           unsigned FIOperandNum,
                           RegScavenger *RS = nullptr) const override;
  Register getFrameRegister(const MachineFunction &MF) const override;
};
} // namespace llvm

#endif // LLVM_LIB_TARGET_AIR_AIRREGISTERINFO_H
