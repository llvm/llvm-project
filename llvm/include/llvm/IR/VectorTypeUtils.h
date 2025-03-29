//===------- VectorTypeUtils.h - Vector type utility functions -*- C++ -*-====//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_VECTORTYPEUTILS_H
#define LLVM_IR_VECTORTYPEUTILS_H

#include "llvm/IR/DerivedTypes.h"

namespace llvm {

/// A helper function for converting Scalar types to vector types. If
/// the incoming type is void, we return void. If the EC represents a
/// scalar, we return the scalar type.
inline Type *toVectorTy(Type *Scalar, ElementCount EC) {
  if (Scalar->isVoidTy() || Scalar->isMetadataTy() || EC.isScalar())
    return Scalar;
  return VectorType::get(Scalar, EC);
}

inline Type *toVectorTy(Type *Scalar, unsigned VF) {
  return toVectorTy(Scalar, ElementCount::getFixed(VF));
}

/// A helper for converting structs of scalar types to structs of vector types.
/// Note:
///   - If \p EC is scalar, \p StructTy is returned unchanged
///   - Only unpacked literal struct types are supported
Type *toVectorizedStructTy(StructType *StructTy, ElementCount EC);

/// A helper for converting structs of vector types to structs of scalar types.
/// Note: Only unpacked literal struct types are supported.
Type *toScalarizedStructTy(StructType *StructTy);

/// Returns true if `StructTy` is an unpacked literal struct where all elements
/// are vectors of matching element count. This does not include empty structs.
bool isVectorizedStructTy(StructType *StructTy);

/// Returns true if `StructTy` is an unpacked literal struct where all elements
/// are scalars that can be used as vector element types.
bool canVectorizeStructTy(StructType *StructTy);

/// A helper for converting to vectorized types. For scalar types, this is
/// equivalent to calling `toVectorTy`. For struct types, this returns a new
/// struct where each element type has been widened to a vector type.
/// Note:
///   - If the incoming type is void, we return void
///   - If \p EC is scalar, \p Ty is returned unchanged
///   - Only unpacked literal struct types are supported
inline Type *toVectorizedTy(Type *Ty, ElementCount EC) {
  if (StructType *StructTy = dyn_cast<StructType>(Ty))
    return toVectorizedStructTy(StructTy, EC);
  return toVectorTy(Ty, EC);
}

/// A helper for converting vectorized types to scalarized (non-vector) types.
/// For vector types, this is equivalent to calling .getScalarType(). For struct
/// types, this returns a new struct where each element type has been converted
/// to a scalar type. Note: Only unpacked literal struct types are supported.
inline Type *toScalarizedTy(Type *Ty) {
  if (StructType *StructTy = dyn_cast<StructType>(Ty))
    return toScalarizedStructTy(StructTy);
  return Ty->getScalarType();
}

/// Returns true if `Ty` is a vector type or a struct of vector types where all
/// vector types share the same VF.
inline bool isVectorizedTy(Type *Ty) {
  if (StructType *StructTy = dyn_cast<StructType>(Ty))
    return isVectorizedStructTy(StructTy);
  return Ty->isVectorTy();
}

/// Returns true if `Ty` is a valid vector element type, void, or an unpacked
/// literal struct where all elements are valid vector element types.
/// Note: Even if a type can be vectorized that does not mean it is valid to do
/// so in all cases. For example, a vectorized struct (as returned by
/// toVectorizedTy) does not perform (de)interleaving, so it can't be used for
/// vectorizing loads/stores.
inline bool canVectorizeTy(Type *Ty) {
  if (StructType *StructTy = dyn_cast<StructType>(Ty))
    return canVectorizeStructTy(StructTy);
  return Ty->isVoidTy() || VectorType::isValidElementType(Ty);
}

/// Returns the types contained in `Ty`. For struct types, it returns the
/// elements, all other types are returned directly.
inline ArrayRef<Type *> getContainedTypes(Type *const &Ty) {
  if (auto *StructTy = dyn_cast<StructType>(Ty))
    return StructTy->elements();
  return ArrayRef<Type *>(&Ty, 1);
}

/// Returns the number of vector elements for a vectorized type.
inline ElementCount getVectorizedTypeVF(Type *Ty) {
  assert(isVectorizedTy(Ty) && "expected vectorized type");
  return cast<VectorType>(getContainedTypes(Ty).front())->getElementCount();
}

inline bool isUnpackedStructLiteral(StructType *StructTy) {
  return StructTy->isLiteral() && !StructTy->isPacked();
}

} // namespace llvm

#endif
