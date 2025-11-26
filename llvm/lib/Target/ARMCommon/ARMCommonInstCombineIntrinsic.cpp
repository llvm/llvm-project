//===- ARMCommonInstCombineIntrinsic.cpp -
//                  instCombineIntrinsic opts for both ARM and AArch64  ---===//
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

#include "ARMCommonInstCombineIntrinsic.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Value.h"
#include "llvm/Transforms/InstCombine/InstCombiner.h"

using namespace llvm;
using namespace llvm::PatternMatch;

namespace llvm {
namespace ARMCommon {

/// Convert `tbl`/`tbx` intrinsics to shufflevector if the mask is constant, and
/// at most two source operands are actually referenced.
Instruction *simplifyNeonTbl(IntrinsicInst &II, InstCombiner &IC,
                             bool IsExtension) {
  // Bail out if the mask is not a constant.
  auto *C = dyn_cast<Constant>(II.getArgOperand(II.arg_size() - 1));
  if (!C)
    return nullptr;

  auto *RetTy = cast<FixedVectorType>(II.getType());
  unsigned NumIndexes = RetTy->getNumElements();

  // Only perform this transformation for <8 x i8> and <16 x i8> vector types.
  if (!(RetTy->getElementType()->isIntegerTy(8) &&
        (NumIndexes == 8 || NumIndexes == 16)))
    return nullptr;

  // For tbx instructions, the first argument is the "fallback" vector, which
  // has the same length as the mask and return type.
  unsigned int StartIndex = (unsigned)IsExtension;
  auto *SourceTy =
      cast<FixedVectorType>(II.getArgOperand(StartIndex)->getType());
  // Note that the element count of each source vector does *not* need to be the
  // same as the element count of the return type and mask! All source vectors
  // must have the same element count as each other, though.
  unsigned NumElementsPerSource = SourceTy->getNumElements();

  // There are no tbl/tbx intrinsics for which the destination size exceeds the
  // source size. However, our definitions of the intrinsics, at least in
  // IntrinsicsAArch64.td, allow for arbitrary destination vector sizes, so it
  // *could* technically happen.
  if (NumIndexes > NumElementsPerSource) {
    return nullptr;
  }

  // The tbl/tbx intrinsics take several source operands followed by a mask
  // operand.
  unsigned int NumSourceOperands = II.arg_size() - 1 - (unsigned)IsExtension;

  // Map input operands to shuffle indices. This also helpfully deduplicates the
  // input arguments, in case the same value is passed as an argument multiple
  // times.
  SmallDenseMap<Value *, unsigned, 2> ValueToShuffleSlot;
  Value *ShuffleOperands[2] = {PoisonValue::get(SourceTy),
                               PoisonValue::get(SourceTy)};

  int Indexes[16];
  for (unsigned I = 0; I < NumIndexes; ++I) {
    Constant *COp = C->getAggregateElement(I);

    if (!COp || (!isa<UndefValue>(COp) && !isa<ConstantInt>(COp)))
      return nullptr;

    if (isa<UndefValue>(COp)) {
      Indexes[I] = -1;
      continue;
    }

    uint64_t Index = cast<ConstantInt>(COp)->getZExtValue();
    // The index of the input argument that this index references (0 = first
    // source argument, etc).
    unsigned SourceOperandIndex = Index / NumElementsPerSource;
    // The index of the element at that source operand.
    unsigned SourceOperandElementIndex = Index % NumElementsPerSource;

    Value *SourceOperand;
    if (SourceOperandIndex >= NumSourceOperands) {
      // This index is out of bounds. Map it to index into either the fallback
      // vector (tbx) or vector of zeroes (tbl).
      SourceOperandIndex = NumSourceOperands;
      if (IsExtension) {
        // For out-of-bounds indices in tbx, choose the `I`th element of the
        // fallback.
        SourceOperand = II.getArgOperand(0);
        SourceOperandElementIndex = I;
      } else {
        // Otherwise, choose some element from the dummy vector of zeroes (we'll
        // always choose the first).
        SourceOperand = Constant::getNullValue(SourceTy);
        SourceOperandElementIndex = 0;
      }
    } else {
      SourceOperand = II.getArgOperand(SourceOperandIndex + StartIndex);
    }

    // The source operand may be the fallback vector, which may not have the
    // same number of elements as the source vector. In that case, we *could*
    // choose to extend its length with another shufflevector, but it's simpler
    // to just bail instead.
    if (cast<FixedVectorType>(SourceOperand->getType())->getNumElements() !=
        NumElementsPerSource) {
      return nullptr;
    }

    // We now know the source operand referenced by this index. Make it a
    // shufflevector operand, if it isn't already.
    unsigned NumSlots = ValueToShuffleSlot.size();
    // This shuffle references more than two sources, and hence cannot be
    // represented as a shufflevector.
    if (NumSlots == 2 && !ValueToShuffleSlot.contains(SourceOperand)) {
      return nullptr;
    }
    auto [It, Inserted] =
        ValueToShuffleSlot.try_emplace(SourceOperand, NumSlots);
    if (Inserted) {
      ShuffleOperands[It->getSecond()] = SourceOperand;
    }

    unsigned RemappedIndex =
        (It->getSecond() * NumElementsPerSource) + SourceOperandElementIndex;
    Indexes[I] = RemappedIndex;
  }

  Value *Shuf = IC.Builder.CreateShuffleVector(
      ShuffleOperands[0], ShuffleOperands[1], ArrayRef(Indexes, NumIndexes));
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
