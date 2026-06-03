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
  virtual void anchor();
public:
  explicit SuperHInstrInfo(const SuperHSubtarget &STI);
};
}

#endif // end LLVM_LIB_TARGET_SUPERH_SUPERHINSTRINFO_H