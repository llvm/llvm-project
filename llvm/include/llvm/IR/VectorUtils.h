//===----------- VectorUtils.h -  Vector type utility functions -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/DerivedTypes.h"

namespace llvm {

/// A helper function for converting Scalar types to vector types. If
/// the incoming type is void, we return void. If the EC represents a
/// scalar, we return the scalar type.
inline Type *ToVectorTy(Type *Scalar, ElementCount EC) {
  if (Scalar->isVoidTy() || Scalar->isMetadataTy() || EC.isScalar())
    return Scalar;
  return VectorType::get(Scalar, EC);
}

inline Type *ToVectorTy(Type *Scalar, unsigned VF) {
  return ToVectorTy(Scalar, ElementCount::getFixed(VF));
}

/// A helper for converting to wider (vector) types. For scalar types, this is
/// equivalent to calling `ToVectorTy`. For struct types, this returns a new
/// struct where each element type has been widened to a vector type. Note: Only
/// unpacked literal struct types are supported.
Type *ToWideTy(Type *Ty, ElementCount EC);

/// A helper for converting wide types to narrow (non-vector) types. For vector
/// types, this is equivalent to calling .getScalarType(). For struct types,
/// this returns a new struct where each element type has been converted to a
/// scalar type. Note: Only unpacked literal struct types are supported.
Type *ToNarrowTy(Type *Ty);

/// Returns the types contained in `Ty`. For struct types, it returns the
/// elements, all other types are returned directly.
SmallVector<Type *, 2> getContainedTypes(Type *Ty);

/// Returns true if `Ty` is a vector type or a struct of vector types where all
/// vector types share the same VF.
bool isWideTy(Type *Ty);

/// Returns the vectorization factor for a widened type.
inline ElementCount getWideTypeVF(Type *Ty) {
  assert(isWideTy(Ty) && "expected widened type!");
  return cast<VectorType>(getContainedTypes(Ty).front())->getElementCount();
}

} // namespace llvm
