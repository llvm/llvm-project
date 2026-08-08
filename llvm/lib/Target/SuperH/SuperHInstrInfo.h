//===-- SuperHInstrInfo.h - SuperH Instruction Information ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the SuperH implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_SUPERH_SUPERHINSTRINFO_H
#define LLVM_LIB_TARGET_SUPERH_SUPERHINSTRINFO_H

#include "SuperHRegisterInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/TargetInstrInfo.h"

#define GET_INSTRINFO_HEADER
#include "SuperHGenInstrInfo.inc"

namespace llvm {

class SuperHInstrInfo : public SuperHGenInstrInfo {
  const SuperHRegisterInfo RI;
  const SuperHSubtarget &Subtarget;

public:
  explicit SuperHInstrInfo(const SuperHSubtarget &STI);

  /// getRegisterInfo - TargetInstrInfo is a superset of MRegister info.  As
  /// such, whenever a client has an instance of instruction info, it should
  /// always be able to get register info as well (through this method).
  const SuperHRegisterInfo &getRegisterInfo() const { return RI; }

  void copyPhysReg(MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator MI, const DebugLoc &DL,
                           Register DestReg, Register SrcReg, bool KillSrc,
                           bool RenamableDest = false,
                           bool RenamableSrc = false) const override;
};

const SuperHInstrInfo *createSuperHInstrInfo(const SuperHSubtarget &STI);
}

#endif // end LLVM_LIB_TARGET_SUPERH_SUPERHINSTRINFO_H