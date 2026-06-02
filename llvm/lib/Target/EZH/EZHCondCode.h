//===-- EZHCondCode.h - EZH Condition Code Enumeration --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The encoding used for conditional codes used in EZH instructions
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_EZH_EZHCONDCODE_H
#define LLVM_LIB_TARGET_EZH_EZHCONDCODE_H

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/ErrorHandling.h"

namespace llvm {
namespace LPCC {
enum CondCode {
  ICC_EU = 0,   // Execute Unconditionally
  ICC_ZE = 1,   // Zero (EQ)
  ICC_NZ = 2,   // Not Zero (NE)
  ICC_PO = 3,   // Positive (PL/GE)
  ICC_NE = 4,   // Negative (MI)
  ICC_AZ = 5,   // Above zero (GT)
  ICC_ZB = 6,   // Zero or below (LE)
  ICC_CA = 7,   // Carry set (CS)
  ICC_NC = 8,   // Carry not set (CC)
  ICC_CZ = 9,   // Carry set and zero
  ICC_SPO = 10, // Shift-only-when-Positive
  ICC_SNE = 11, // Shift-only-when-Negative
  ICC_NBS = 12, // Not Boolean-expression set
  ICC_NEX = 13, // External flag is not set
  ICC_BS = 14,  // Boolean-expression set
  ICC_EX = 15,  // External flag is set
  UNKNOWN
};

} // namespace LPCC
} // namespace llvm

#endif // LLVM_LIB_TARGET_EZH_EZHCONDCODE_H
