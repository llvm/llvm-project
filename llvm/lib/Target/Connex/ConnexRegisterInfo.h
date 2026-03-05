//===-- ConnexRegisterInfo.h - Connex Register Information Impl -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the Connex implementation of the TargetRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_CONNEX_CONNEXREGISTERINFO_H
#define LLVM_LIB_TARGET_CONNEX_CONNEXREGISTERINFO_H

#include "llvm/CodeGen/TargetRegisterInfo.h"

#define GET_REGINFO_HEADER
#include "ConnexGenRegisterInfo.inc"

namespace llvm {

struct ConnexRegisterInfo : public ConnexGenRegisterInfo {

  ConnexRegisterInfo();

  // Inspired from lib/Target/Mips/MipsRegisterInfo.cpp
  const TargetRegisterClass *getPointerRegClass(const MachineFunction &MF,
                                                unsigned Kind) const;

  const MCPhysReg *getCalleeSavedRegs(const MachineFunction *MF) const override;

  /*
  From http://llvm.org/doxygen/classllvm_1_1TargetRegisterInfo.html:
    <<Returns a bitset indexed by physical register number indicating if a
      register is a special register that has particular uses and should be
      considered unavailable at all times, e.g. stack pointer, return address.
      A reserved register:
        is not allocatable
        is considered always live
        is ignored by liveness tracking It is often necessary to reserve the
          super registers of a reserved register as well, to avoid them
          getting allocated indirectly. You may use markSuperRegs() and
          checkAllSuperRegsMarked() in this case.>>
  */
  BitVector getReservedRegs(const MachineFunction &MF) const override;

  bool eliminateFrameIndex(MachineBasicBlock::iterator MI, int SPAdj,
                           unsigned FIOperandNum,
                           RegScavenger *RS = nullptr) const override;

  Register getFrameRegister(const MachineFunction &MF) const override;

  /* Addressing bug
      (llc -O0, at pass: "********** FAST REGISTER ALLOCATION **********")
    <<Remaining virtual register operands
    UNREACHABLE executed at llvm/lib/CodeGen/MachineRegisterInfo.cpp:144!>>

    (Using suggestion from at
      https://groups.google.com/forum/#!topic/llvm-dev/fEyD9YREi5M).
  */
  // See http://llvm.org/docs/doxygen/html/classllvm_1_1TargetRegisterInfo.html
  // Returns true if the target requires (and can make use of) the register
  //    scavenger.
  virtual bool
  requiresRegisterScavenging(const MachineFunction &MF) const override {
    return false;
  }

  virtual bool
  requiresFrameIndexScavenging(const MachineFunction &MF) const override {
    return false;
  }
};
} // namespace llvm

#endif
