//===-- VEInstrInfo.h - VE Instruction Information --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the VE implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_VE_VEINSTRINFO_H
#define LLVM_LIB_TARGET_VE_VEINSTRINFO_H

#include "VERegisterInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"

#define GET_INSTRINFO_HEADER
#include "VEGenInstrInfo.inc"

namespace llvm {

class VESubtarget;

class VEInstrInfo : public VEGenInstrInfo {
  const VERegisterInfo RI;
  const VESubtarget &Subtarget;
  virtual void anchor();

public:
  explicit VEInstrInfo(VESubtarget &ST);

  /// getRegisterInfo - TargetInstrInfo is a superset of MRegister info.  As
  /// such, whenever a client has an instance of instruction info, it should
  /// always be able to get register info as well (through this method).
  ///
  const VERegisterInfo &getRegisterInfo() const { return RI; }

  void copyPhysReg(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
                   const DebugLoc &DL, MCRegister DestReg, MCRegister SrcReg,
                   bool KillSrc) const override;

  // Lower pseudo instructions after register allocation.
  bool expandPostRAPseudo(MachineInstr &MI) const override;

  bool expandExtendStackPseudo(MachineInstr &MI) const;
};

} // namespace llvm

#endif
