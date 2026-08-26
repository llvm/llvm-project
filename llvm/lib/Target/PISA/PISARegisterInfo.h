//===-- PISARegisterInfo.h - PISA Register Information --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_PISA_PISAREGISTERINFO_H
#define LLVM_LIB_TARGET_PISA_PISAREGISTERINFO_H

#include "llvm/ADT/BitVector.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"

#define GET_REGINFO_HEADER
#include "PISAGenRegisterInfo.inc"

namespace llvm {

class PISARegisterInfo : public PISAGenRegisterInfo {
public:
  PISARegisterInfo();
  const MCPhysReg *getCalleeSavedRegs(const MachineFunction *MF) const override;
  BitVector getReservedRegs(const MachineFunction &MF) const override;
  bool eliminateFrameIndex(MachineBasicBlock::iterator MI, int SPAdj,
                           unsigned FIOperandNum,
                           RegScavenger *RS = nullptr) const override {
    llvm_unreachable("unexpected execution");
  }
  Register getFrameRegister(const MachineFunction &MF) const override {
    return Register();
  }
};
} // namespace llvm

#endif // LLVM_LIB_TARGET_PISA_PISAREGISTERINFO_H
