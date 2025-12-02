//===- ARMCommonInstCombineIntrinsic.cpp - Shared ARM/AArch64 opts  -------===//
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

#include "llvm/Transforms/Utils/ARMCommonInstCombineIntrinsic.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Value.h"
#include "llvm/Transforms/InstCombine/InstCombiner.h"

using namespace llvm;
using namespace llvm::PatternMatch;

namespace llvm {
namespace ARMCommon {

/// Convert a table lookup to shufflevector if the mask is constant.
/// This could benefit tbl1 if the mask is { 7,6,5,4,3,2,1,0 }, in
/// which case we could lower the shufflevector with rev64 instructions
/// as it's actually a byte reverse.
Instruction *simplifyNeonTbl1(IntrinsicInst &II, InstCombiner &IC) {
  // Bail out if the mask is not a constant.
  auto *C = dyn_cast<Constant>(II.getArgOperand(1));
  if (!C)
    return nullptr;

  auto *VecTy = cast<FixedVectorType>(II.getType());
  unsigned NumElts = VecTy->getNumElements();

  // Only perform this transformation for <8 x i8> vector types.
  if (!VecTy->getElementType()->isIntegerTy(8) || NumElts != 8)
    return nullptr;

  int Indexes[8];

  for (unsigned I = 0; I < NumElts; ++I) {
    Constant *COp = C->getAggregateElement(I);

    if (!COp || !isa<ConstantInt>(COp))
      return nullptr;

    Indexes[I] = cast<ConstantInt>(COp)->getLimitedValue();

    // Make sure the mask indices are in range.
    if ((unsigned)Indexes[I] >= NumElts)
      return nullptr;
  }

  auto *V1 = II.getArgOperand(0);
  auto *V2 = Constant::getNullValue(V1->getType());
  Value *Shuf = IC.Builder.CreateShuffleVector(V1, V2, ArrayRef(Indexes));
  return IC.replaceInstUsesWith(II, Shuf);
}

/// Simplify NEON multiply-long intrinsics (smull, umull).
/// These intrinsics perform widening multiplies: they multiply two vectors of
/// narrow integers and produce a vector of wider integers. This function
/// performs algebraic simplifications:
/// 1. Multiply by zero => zero vector
/// 2. Multiply by one => zero/sign-extend the non-one operand
/// 3. Both operands constant => regular multiply that can be constant-folded
///    later
Instruction *simplifyNeonMultiply(IntrinsicInst &II, InstCombiner &IC,
                                  bool IsSigned) {
  Value *Arg0 = II.getArgOperand(0);
  Value *Arg1 = II.getArgOperand(1);

  // Handle mul by zero first:
  if (isa<ConstantAggregateZero>(Arg0) || isa<ConstantAggregateZero>(Arg1)) {
    return IC.replaceInstUsesWith(II, ConstantAggregateZero::get(II.getType()));
  }

  // Check for constant LHS & RHS - in this case we just simplify.
  VectorType *NewVT = cast<VectorType>(II.getType());
  if (Constant *CV0 = dyn_cast<Constant>(Arg0)) {
    if (Constant *CV1 = dyn_cast<Constant>(Arg1)) {
      Value *V0 = IC.Builder.CreateIntCast(CV0, NewVT, IsSigned);
      Value *V1 = IC.Builder.CreateIntCast(CV1, NewVT, IsSigned);
      return IC.replaceInstUsesWith(II, IC.Builder.CreateMul(V0, V1));
    }

    // Couldn't simplify - canonicalize constant to the RHS.
    std::swap(Arg0, Arg1);
  }

  // Handle mul by one:
  if (Constant *CV1 = dyn_cast<Constant>(Arg1))
    if (ConstantInt *Splat =
            dyn_cast_or_null<ConstantInt>(CV1->getSplatValue()))
      if (Splat->isOne())
        return CastInst::CreateIntegerCast(Arg0, II.getType(), IsSigned);

  return nullptr;
}

/// Simplify AES encryption/decryption intrinsics (AESE, AESD).
///
/// ARM's AES instructions (AESE/AESD) XOR the data and the key, provided as
/// separate arguments, before performing the encryption/decryption operation.
/// We can fold that "internal" XOR with a previous one.
Instruction *simplifyAES(IntrinsicInst &II, InstCombiner &IC) {
  Value *DataArg = II.getArgOperand(0);
  Value *KeyArg = II.getArgOperand(1);

  // Accept zero on either operand.
  if (!match(KeyArg, m_ZeroInt()))
    std::swap(KeyArg, DataArg);

  // Try to use the builtin XOR in AESE and AESD to eliminate a prior XOR
  Value *Data, *Key;
  if (match(KeyArg, m_ZeroInt()) &&
      match(DataArg, m_Xor(m_Value(Data), m_Value(Key)))) {
    IC.replaceOperand(II, 0, Data);
    IC.replaceOperand(II, 1, Key);
    return &II;
  }

  return nullptr;
}

} // namespace ARMCommon
} // namespace llvm
