//===-- EZHBaseInfo.h - Top level definitions for EZH MC ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_EZH_MCTARGETDESC_EZHBASEINFO_H
#define LLVM_LIB_TARGET_EZH_MCTARGETDESC_EZHBASEINFO_H

#include "EZHMCTargetDesc.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/Support/ErrorHandling.h"

namespace llvm {

namespace EZHII {
enum TOF { MO_NO_FLAG, MO_HI16, MO_LO16 };
} // namespace EZHII

static inline unsigned getEZHRegisterNumbering(MCRegister Reg) {
  switch (Reg.id()) {
  case EZH::R0:
    return 0;
  case EZH::R1:
    return 1;
  case EZH::R2:
    return 2;
  case EZH::R3:
    return 3;
  case EZH::R4:
    return 4;
  case EZH::R5:
    return 5;
  case EZH::R6:
    return 6;
  case EZH::R7:
    return 7;
  case EZH::GPO:
    return 8;
  case EZH::GPD:
    return 9;
  case EZH::CFS:
    return 10;
  case EZH::CFM:
    return 11;
  case EZH::SP:
    return 12;
  case EZH::PC:
    return 13;
  case EZH::GPI:
    return 14;
  case EZH::RA:
    return 15;
  default:
    llvm_unreachable("Unknown register number!");
  }
}
} // namespace llvm
#endif // LLVM_LIB_TARGET_EZH_MCTARGETDESC_EZHBASEINFO_H
