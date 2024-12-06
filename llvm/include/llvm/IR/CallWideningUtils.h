//===---- CallWideningUtils.h - Utils for widening scalar to vector calls --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_CALLWIDENINGUTILS_H
#define LLVM_IR_CALLWIDENINGUTILS_H

#include "llvm/IR/DerivedTypes.h"

namespace llvm {

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
  assert(isWideTy(Ty) && "expected widened type");
  return cast<VectorType>(getContainedTypes(Ty).front())->getElementCount();
}

} // namespace llvm

#endif
