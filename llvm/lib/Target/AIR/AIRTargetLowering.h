//===-- AIRTargetLowering.h - Define AIR TargetLowering -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the AIR specific subclass of TargetLowering.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AIR_AIRTARGETLOWERING_H
#define LLVM_LIB_TARGET_AIR_AIRTARGETLOWERING_H

#include "llvm/CodeGen/TargetLowering.h"

namespace llvm {

class AIRSubtarget;
class AIRTargetMachine;

class AIRTargetLowering : public TargetLowering {
public:
  explicit AIRTargetLowering(const AIRTargetMachine &TM,
                               const AIRSubtarget &STI);

  bool isLegalAddressingMode(const DataLayout &DL, const AddrMode &AM, Type *Ty,
                             unsigned AS,
                             Instruction *I = nullptr) const override;
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_AIR_AIRTARGETLOWERING_H
