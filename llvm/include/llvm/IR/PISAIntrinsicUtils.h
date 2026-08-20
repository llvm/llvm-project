//===-- PISAIntrinsicUtils.h ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_PISAINTRINSICUTILS_H
#define LLVM_IR_PISAINTRINSICUTILS_H
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsPISA.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {
namespace pisa {

// The enum values must match the Clang preprocessor definitions in
// lib/Frontend/InitPreprocessor.cpp.
// See the PISA memory-scope specification:
// https://intel.github.io/pisa/virtual_machine.html#memory-scope
namespace MemoryScope {
enum : unsigned {
  none = 255,
  system = 0,    // __MEMORY_SCOPE_SYSTEM
  gpu = 1,       // __MEMORY_SCOPE_DEVICE
  workgroup = 2, // __MEMORY_SCOPE_WRKGRP
  subgroup = 3,  // __MEMORY_SCOPE_WVFRNT
  total_scopes
};
} // namespace MemoryScope

// Integer reduction operations:
// https://intel.github.io/pisa/instructions_cross_lane.html#ired
namespace IRedOp {
enum : unsigned {
  SUM = 0,
  SMIN = 1,
  SMAX = 2,
  UMIN = 3,
  UMAX = 4,
  AND = 5,
  OR = 6,
  XOR = 7,
  ABSMAX = 8,
  Last
};
} // namespace IRedOp

// Floating-point reduction operations:
// https://intel.github.io/pisa/instructions_cross_lane.html#fred
namespace FRedOp {
enum : unsigned { MIN = 0, MAX = 1, ABSMAX = 2, Last };
} // namespace FRedOp

// Sub-group shuffle modes:
// https://intel.github.io/pisa/instructions_cross_lane.html#shfl
namespace SHFLMode {
enum : unsigned { UP = 0, DOWN = 1, XOR = 2, IDX = 3, Last };
} // namespace SHFLMode

// Print a string corresponding to various immediate arguments to OS.
//
// If the value is invalid/unsupported, the functions print nothing; no errors
// are raised. This is because these functions may be called during printing of
// invalid IR, which should not crash the compiler. Other code (like the PISA
// Verifier) is responsible for reporting errors on invalid IR.

LLVM_ABI void printMemoryOrdering(raw_ostream &OS, const Constant *ImmArgVal);
LLVM_ABI void printRoundingMode(raw_ostream &OS, const Constant *ImmArgVal);
LLVM_ABI void printIRedOp(raw_ostream &OS, const Constant *ImmArgVal);
LLVM_ABI void printFRedOp(raw_ostream &OS, const Constant *ImmArgVal);
LLVM_ABI void printSHFLMode(raw_ostream &OS, const Constant *ImmArgVal);

} // namespace pisa
} // namespace llvm
#endif // LLVM_IR_PISAINTRINSICUTILS_H
