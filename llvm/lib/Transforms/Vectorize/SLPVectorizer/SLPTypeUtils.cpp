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
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/VectorTypeUtils.h"
#include "llvm/Support/Casting.h"

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

} // namespace llvm::slpvectorizer
