//===------- VectorTypeUtils.cpp - Vector type utility functions ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/VectorTypeUtils.h"
#include "llvm/ADT/SmallVectorExtras.h"

using namespace llvm;

/// A helper for converting structs of scalar types to structs of vector types.
/// Note: Only unpacked literal struct types are supported.
Type *llvm::toVectorizedStructTy(StructType *StructTy, ElementCount EC) {
  if (EC.isScalar())
    return StructTy;
  assert(isUnpackedStructLiteral(StructTy) &&
         "expected unpacked struct literal");
  assert(all_of(StructTy->elements(), VectorType::isValidElementType) &&
         "expected all element types to be valid vector element types");
  return StructType::get(
      StructTy->getContext(),
      map_to_vector(StructTy->elements(), [&](Type *ElTy) -> Type * {
        return VectorType::get(ElTy, EC);
      }));
}

/// A helper for converting structs of vector types to structs of scalar types.
/// Note: Only unpacked literal struct types are supported.
Type *llvm::toScalarizedStructTy(StructType *StructTy) {
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
bool llvm::isVectorizedStructTy(StructType *StructTy) {
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

/// Returns true if `StructTy` is an unpacked literal struct where all elements
/// are scalars that can be used as vector element types.
bool llvm::canVectorizeStructTy(StructType *StructTy) {
  auto ElemTys = StructTy->elements();
  return !ElemTys.empty() && isUnpackedStructLiteral(StructTy) &&
         all_of(ElemTys, VectorType::isValidElementType);
}
