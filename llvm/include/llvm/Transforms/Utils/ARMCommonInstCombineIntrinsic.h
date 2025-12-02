//===- ARMCommonInstCombineIntrinsic.h - Shared ARM/AArch64 opts *- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains optimizations for ARM and AArch64 intrinsics that
/// are shared between both architectures. These functions can be called from:
/// - ARM TTI's instCombineIntrinsic (for arm_neon_* intrinsics)
/// - AArch64 TTI's instCombineIntrinsic (for aarch64_neon_* and aarch64_sve_*
///   intrinsics)
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_ARMCOMMONINSTCOMBINEINTRINSIC_H
#define LLVM_TRANSFORMS_UTILS_ARMCOMMONINSTCOMBINEINTRINSIC_H

#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Value.h"
#include "llvm/Transforms/InstCombine/InstCombiner.h"

namespace llvm {

namespace ARMCommon {

/// Convert a table lookup to shufflevector if the mask is constant.
/// This could benefit tbl1 if the mask is { 7,6,5,4,3,2,1,0 }, in
/// which case we could lower the shufflevector with rev64 instructions
/// as it's actually a byte reverse.
Instruction *simplifyNeonTbl1(IntrinsicInst &II, InstCombiner &IC);

/// Simplify NEON multiply-long intrinsics (smull, umull).
/// These intrinsics perform widening multiplies: they multiply two vectors of
/// narrow integers and produce a vector of wider integers. This function
/// performs algebraic simplifications:
/// 1. Multiply by zero => zero vector
/// 2. Multiply by one => zero/sign-extend the non-one operand
/// 3. Both operands constant => regular multiply that can be constant-folded
///    later
Instruction *simplifyNeonMultiply(IntrinsicInst &II, InstCombiner &IC,
                                  bool IsSigned);

/// Simplify AES encryption/decryption intrinsics (AESE, AESD).
///
/// ARM's AES instructions (AESE/AESD) XOR the data and the key, provided as
/// separate arguments, before performing the encryption/decryption operation.
/// We can fold that "internal" XOR with a previous one.
Instruction *simplifyAES(IntrinsicInst &II, InstCombiner &IC);

} // namespace ARMCommon
} // namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_ARMCOMMONINSTCOMBINEINTRINSIC_H
