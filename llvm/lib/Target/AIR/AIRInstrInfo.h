//===-- AIRInstrInfo.h - Define InstrInfo for AIR -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the AIR specific subclass of TargetInstrInfo.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AIR_AIRINSTRINFO_H
#define LLVM_LIB_TARGET_AIR_AIRINSTRINFO_H

#include "AIRRegisterInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"

#define GET_INSTRINFO_HEADER
#include "AIRGenInstrInfo.inc"

namespace llvm {
class AIRSubtarget;

struct AIRInstrInfo : public AIRGenInstrInfo {
  const AIRRegisterInfo RI;
  explicit AIRInstrInfo(const AIRSubtarget &STI);
  const AIRRegisterInfo &getRegisterInfo() const { return RI; }
  ~AIRInstrInfo() override;
};
} // namespace llvm

#endif // LLVM_LIB_TARGET_AIR_AIRINSTRINFO_H
