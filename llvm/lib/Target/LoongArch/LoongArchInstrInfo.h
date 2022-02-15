//=- LoongArchInstrInfo.h - LoongArch Instruction Information ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the LoongArch implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_LOONGARCH_LOONGARCHINSTRINFO_H
#define LLVM_LIB_TARGET_LOONGARCH_LOONGARCHINSTRINFO_H

#include "llvm/CodeGen/TargetInstrInfo.h"

#define GET_INSTRINFO_HEADER
#include "LoongArchGenInstrInfo.inc"

namespace llvm {

class LoongArchSubtarget;

class LoongArchInstrInfo : public LoongArchGenInstrInfo {
  const LoongArchSubtarget &STI;

public:
  explicit LoongArchInstrInfo(LoongArchSubtarget &STI);
};

} // end namespace llvm
#endif // LLVM_LIB_TARGET_LOONGARCH_LOONGARCHINSTRINFO_H
