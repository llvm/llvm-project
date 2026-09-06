//===- SLPTypeUtils.cpp - SLP Vectorizer type/width helpers ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SLPTypeUtils.h"
#include "SLPUtils.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/bit.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/VectorTypeUtils.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/MathExtras.h"

#include <cassert>

using namespace llvm;

namespace llvm::slpvectorizer {

bool isValidElementType(Type *Ty, bool ReVec) {
  // TODO: Support ScalableVectorType.
  if (ReVec && isVectorizedTy(Ty) && !getVectorizedTypeVF(Ty).isScalable())
    Ty = toScalarizedTy(Ty);
  return canVectorizeTy(Ty) && !Ty->isX86_FP80Ty() && !Ty->isPPC_FP128Ty() &&
         !Ty->isVoidTy();
}

Type *getValueType(Value *V, bool ReVec, bool LookThroughCmp) {
  if (auto *SI = dyn_cast<StoreInst>(V))
    return SI->getValueOperand()->getType();
  if (LookThroughCmp)
    if (auto *CI = dyn_cast<CmpInst>(V))
      return CI->getOperand(0)->getType();
  if (!ReVec)
    if (auto *IE = dyn_cast<InsertElementInst>(V))
      return IE->getOperand(1)->getType();
  if (auto *IV = dyn_cast<InsertValueInst>(V))
    return IV->getOperand(1)->getType();
  return V->getType();
}

Type *getWidenedType(Type *ScalarTy, unsigned VF) {
  if (VF == 1 && !isVectorizedTy(ScalarTy)) {
    // Workaround for 1 x vector types: toVectorizedTy returns the type
    // unchanged when EC is scalar, but BoUpSLP relies on widening to
    // <1 x ScalarTy> (or struct of <1 x ElTy>) to keep the rest of the
    // pipeline operating on vector types.
    if (auto *StructTy = dyn_cast<StructType>(ScalarTy)) {
      assert(isUnpackedStructLiteral(StructTy) &&
             "expected unpacked struct literal");
      assert(all_of(StructTy->elements(), VectorType::isValidElementType) &&
             "expected all element types to be valid vector element types");
      return StructType::get(
          StructTy->getContext(),
          map_to_vector(StructTy->elements(), [&](Type *ElTy) -> Type * {
            return FixedVectorType::get(ElTy, 1);
          }));
    }
    return FixedVectorType::get(ScalarTy, 1);
  }
  return toVectorizedTy(toScalarizedTy(ScalarTy),
                        ElementCount::getFixed(VF * getNumElements(ScalarTy)));
}

unsigned getFullVectorNumberOfElements(const TargetTransformInfo &TTI, Type *Ty,
                                       unsigned Sz, bool ReVec) {
  if (!isValidElementType(Ty, ReVec) || isa<StructType>(Ty))
    return bit_ceil(Sz);
  // Find the number of elements, which forms full vectors.
  const unsigned NumParts = TTI.getNumberOfParts(getWidenedType(Ty, Sz));
  if (NumParts == 0 || NumParts >= Sz)
    return bit_ceil(Sz);
  return bit_ceil(divideCeil(Sz, NumParts)) * NumParts;
}

unsigned getFloorFullVectorNumberOfElements(const TargetTransformInfo &TTI,
                                            Type *Ty, unsigned Sz, bool ReVec) {
  if (!isValidElementType(Ty, ReVec) || isa<StructType>(Ty))
    return bit_floor(Sz);
  // Find the number of elements, which forms full vectors.
  unsigned NumParts = TTI.getNumberOfParts(getWidenedType(Ty, Sz));
  if (NumParts == 0 || NumParts >= Sz)
    return bit_floor(Sz);
  unsigned RegVF = bit_ceil(divideCeil(Sz, NumParts));
  if (RegVF > Sz)
    return bit_floor(Sz);
  return (Sz / RegVF) * RegVF;
}

FixedVectorType *getMaskedDivRemType(const TargetTransformInfo &TTI,
                                     unsigned Opcode, Type *ScalarTy,
                                     unsigned NumElts, bool ReVec) {
  if (!Instruction::isIntDivRem(Opcode) || has_single_bit(NumElts))
    return nullptr;
  unsigned PaddedNumElts =
      getFullVectorNumberOfElements(TTI, ScalarTy, NumElts, ReVec);
  if (PaddedNumElts == NumElts)
    return nullptr;
  return cast<FixedVectorType>(getWidenedType(ScalarTy, PaddedNumElts));
}

bool hasFullVectorsOrPowerOf2(const TargetTransformInfo &TTI, Type *Ty,
                              unsigned Sz, bool ReVec) {
  if (Sz <= 1)
    return false;
  if (!isValidElementType(Ty, ReVec) && !isa<FixedVectorType>(Ty))
    return false;
  if (has_single_bit(Sz))
    return true;
  if (isa<StructType>(Ty))
    return false;
  const unsigned NumParts = TTI.getNumberOfParts(getWidenedType(Ty, Sz));
  return NumParts > 0 && NumParts < Sz && has_single_bit(Sz / NumParts) &&
         Sz % NumParts == 0;
}

} // namespace llvm::slpvectorizer
