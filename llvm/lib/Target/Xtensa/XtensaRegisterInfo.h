//===-- XtensaRegisterInfo.h - Xtensa Register Information Impl -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the Xtensa implementation of the TargetRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_XTENSA_XTENSAREGISTERINFO_H
#define LLVM_LIB_TARGET_XTENSA_XTENSAREGISTERINFO_H

#include "Xtensa.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"

#define GET_REGINFO_HEADER
#include "XtensaGenRegisterInfo.inc"

namespace llvm {
class TargetRegisterClass;
class XtensaInstrInfo;
class XtensaSubtarget;

struct XtensaRegisterInfo : public XtensaGenRegisterInfo {
public:
  const XtensaSubtarget &Subtarget;

  XtensaRegisterInfo(const XtensaSubtarget &STI);

  bool requiresRegisterScavenging(const MachineFunction &MF) const override {
    return true;
  }

  bool requiresFrameIndexScavenging(const MachineFunction &MF) const override {
    return true;
  }

  bool trackLivenessAfterRegAlloc(const MachineFunction &) const override {
    return true;
  }

  const uint16_t *
  getCalleeSavedRegs(const MachineFunction *MF = 0) const override;
  const uint32_t *getCallPreservedMask(const MachineFunction &MF,
                                       CallingConv::ID) const override;
  BitVector getReservedRegs(const MachineFunction &MF) const override;
  void eliminateFrameIndex(MachineBasicBlock::iterator MI, int SPAdj,
                           unsigned FIOperandNum,
                           RegScavenger *RS) const override;
  Register getFrameRegister(const MachineFunction &MF) const override;

private:
  void eliminateFI(MachineBasicBlock::iterator II, unsigned OpNo,
                   int FrameIndex, uint64_t StackSize, int64_t SPOffset) const;
};

} // end namespace llvm

#endif /* LLVM_LIB_TARGET_XTENSA_REGISTERINFO_H */
