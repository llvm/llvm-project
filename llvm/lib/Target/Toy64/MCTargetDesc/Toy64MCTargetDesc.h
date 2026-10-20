//===-- Toy64MCTargetDesc.h - Toy64 Target Descriptions -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_TOY64_MCTARGETDESC_TOY64MCTARGETDESC_H
#define LLVM_LIB_TARGET_TOY64_MCTARGETDESC_TOY64MCTARGETDESC_H

#include "llvm/MC/MCRegisterInfo.h"

// Register enum values:
//
//   Toy64::X0
//   Toy64::X1
//   ...
//   Toy64::X31
//
#define GET_REGINFO_ENUM
#include "Toy64GenRegisterInfo.inc"

// Instruction enum values:
//
//   Toy64::ADD
//   Toy64::SUB
//   Toy64::ADDI
//
#define GET_INSTRINFO_ENUM
#include "Toy64GenInstrInfo.inc"

// Subtarget-related enums.
#define GET_SUBTARGETINFO_ENUM
#include "Toy64GenSubtargetInfo.inc"

namespace llvm {

class Target;

} // namespace llvm

#endif // LLVM_LIB_TARGET_TOY64_MCTARGETDESC_TOY64MCTARGETDESC_H