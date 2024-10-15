//===----------- VectorUtils.cpp - Vector type utility functions ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/VectorUtils.h"
#include "llvm/ADT/SmallVectorExtras.h"

using namespace llvm;

/// A helper for converting to wider (vector) types. For scalar types, this is
/// equivalent to calling `ToVectorTy`. For struct types, this returns a new
/// struct where each element type has been widened to a vector type. Note: Only
/// unpacked literal struct types are supported.
Type *llvm::ToWideTy(Type *Ty, ElementCount EC) {
  if (EC.isScalar())
    return Ty;
  auto *StructTy = dyn_cast<StructType>(Ty);
  if (!StructTy)
    return ToVectorTy(Ty, EC);
  assert(StructTy->isLiteral() && !StructTy->isPacked() &&
         "expected unpacked struct literal");
  return StructType::get(
      Ty->getContext(),
      map_to_vector(StructTy->elements(), [&](Type *ElTy) -> Type * {
        return VectorType::get(ElTy, EC);
      }));
}

/// A helper for converting wide types to narrow (non-vector) types. For vector
/// types, this is equivalent to calling .getScalarType(). For struct types,
/// this returns a new struct where each element type has been converted to a
/// scalar type. Note: Only unpacked literal struct types are supported.
Type *llvm::ToNarrowTy(Type *Ty) {
  auto *StructTy = dyn_cast<StructType>(Ty);
  if (!StructTy)
    return Ty->getScalarType();
  assert(StructTy->isLiteral() && !StructTy->isPacked() &&
         "expected unpacked struct literal");
  return StructType::get(
      Ty->getContext(),
      map_to_vector(StructTy->elements(), [](Type *ElTy) -> Type * {
        return ElTy->getScalarType();
      }));
}

/// Returns the types contained in `Ty`. For struct types, it returns the
/// elements, all other types are returned directly.
SmallVector<Type *, 2> llvm::getContainedTypes(Type *Ty) {
  auto *StructTy = dyn_cast<StructType>(Ty);
  if (StructTy)
    return to_vector<2>(StructTy->elements());
  return {Ty};
}

/// Returns true if `Ty` is a vector type or a struct of vector types where all
/// vector types share the same VF.
bool llvm::isWideTy(Type *Ty) {
  auto ContainedTys = getContainedTypes(Ty);
  if (ContainedTys.empty() || !ContainedTys.front()->isVectorTy())
    return false;
  ElementCount VF = cast<VectorType>(ContainedTys.front())->getElementCount();
  return all_of(ContainedTys, [&](Type *Ty) {
    return Ty->isVectorTy() && cast<VectorType>(Ty)->getElementCount() == VF;
  });
}
