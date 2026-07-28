//===-- PISAInstrInfo.h - PISA Instruction Information --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_PISA_PISAINSTRINFO_H
#define LLVM_LIB_TARGET_PISA_PISAINSTRINFO_H

#include "PISARegisterInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"

#define GET_INSTRINFO_HEADER
#include "PISAGenInstrInfo.inc"

namespace llvm {
class PISASubtarget;

class PISAInstrInfo : public PISAGenInstrInfo {
  const PISARegisterInfo RI;

public:
  PISAInstrInfo(const PISASubtarget &STI);

  const PISARegisterInfo &getRegisterInfo() const { return RI; }
};
} // namespace llvm

#endif // LLVM_LIB_TARGET_PISA_PISAINSTRINFO_H
