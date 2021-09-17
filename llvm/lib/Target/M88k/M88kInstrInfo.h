//===-- M88kInstrInfo.h - M88k instruction information ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the M88k implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_M88K_M88KINSTRINFO_H
#define LLVM_LIB_TARGET_M88K_M88KINSTRINFO_H

#include "M88k.h"
#include "M88kRegisterInfo.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include <cstdint>

#define GET_INSTRINFO_HEADER
#include "M88kGenInstrInfo.inc"

namespace llvm {

class M88kSubtarget;

class M88kInstrInfo : public M88kGenInstrInfo {
  const M88kRegisterInfo RI;
  M88kSubtarget &STI;

  virtual void anchor();

public:
  explicit M88kInstrInfo(M88kSubtarget &STI);

  // Return the M88kRegisterInfo, which this class owns.
  const M88kRegisterInfo &getRegisterInfo() const { return RI; }
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_M88K_M88KINSTRINFO_H
