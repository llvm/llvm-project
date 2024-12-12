//===- StructWideningUtils.cpp - Utils for widening/narrowing struct types ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/StructWideningUtils.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/IR/VectorUtils.h"

using namespace llvm;

static bool isUnpackedStructLiteral(StructType *StructTy) {
  return StructTy->isLiteral() && !StructTy->isPacked();
}

/// A helper for converting structs of scalar types to structs of vector types.
/// Note: Only unpacked literal struct types are supported.
Type *llvm::ToWideStructTy(StructType *StructTy, ElementCount EC) {
  if (EC.isScalar())
    return StructTy;
  assert(isUnpackedStructLiteral(StructTy) &&
         "expected unpacked struct literal");
  return StructType::get(
      StructTy->getContext(),
      map_to_vector(StructTy->elements(), [&](Type *ElTy) -> Type * {
        return VectorType::get(ElTy, EC);
      }));
}

/// A helper for converting structs of vector types to structs of scalar types.
/// Note: Only unpacked literal struct types are supported.
Type *llvm::ToNarrowStructTy(StructType *StructTy) {
  assert(isUnpackedStructLiteral(StructTy) &&
         "expected unpacked struct literal");
  return StructType::get(
      StructTy->getContext(),
      map_to_vector(StructTy->elements(), [](Type *ElTy) -> Type * {
        return ElTy->getScalarType();
      }));
}

/// Returns true if `StructTy` is an unpacked literal struct where all elements
/// are vectors of matching element count. This does not include empty structs.
bool llvm::isWideStructTy(StructType *StructTy) {
  if (!isUnpackedStructLiteral(StructTy))
    return false;
  auto ElemTys = StructTy->elements();
  if (ElemTys.empty() || !ElemTys.front()->isVectorTy())
    return false;
  ElementCount VF = cast<VectorType>(ElemTys.front())->getElementCount();
  return all_of(ElemTys, [&](Type *Ty) {
    return Ty->isVectorTy() && cast<VectorType>(Ty)->getElementCount() == VF;
  });
}
