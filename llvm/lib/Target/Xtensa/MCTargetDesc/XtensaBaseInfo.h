//===-- XtensaBaseInfo.h - Top level definitions for Xtensa MC ----*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains small standalone helper functions and enum definitions
// for the Xtensa target useful for the compiler back-end and the MC libraries.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_LIB_TARGET_XTENSA_MCTARGETDESC_XTENSABASEINFO_H
#define LLVM_LIB_TARGET_XTENSA_MCTARGETDESC_XTENSABASEINFO_H

#include "XtensaFixupKinds.h"
#include "XtensaMCTargetDesc.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/ErrorHandling.h"

namespace llvm {

/// XtensaII - This namespace holds all of the target specific flags that
/// instruction info tracks.
///
namespace XtensaII {
/// Target Operand Flag enum.
enum TOF {
  // Xtensa Specific MachineOperand flags.
  MO_NO_FLAG,
  MO_TPOFF, // Represents the offset from the thread pointer
  MO_PLT
};

} // namespace XtensaII
} // namespace llvm

#endif
