//===-- AGXSoftF64.h - Double precision in software for AGX -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AGX_AGXSOFTF64_H
#define LLVM_LIB_TARGET_AGX_AGXSOFTF64_H

namespace llvm {
class Instruction;
class Value;

namespace AGX {

/// True if I computes with double precision values (beyond moving them).
bool isDoubleArithmetic(const Instruction &I);

/// Replaces I, a double precision operation, with integer arithmetic on the
/// bits of its operands, correctly rounded in the direction Rounding (an i32
/// value numbered like FLT_ROUNDS). Returns the replacement, or null (and
/// leaves I) if the operation is not supported.
Value *lowerDoubleArithmetic(Instruction &I, Value *Rounding,
                             bool CanonicalNaN);

} // namespace AGX
} // namespace llvm

#endif
