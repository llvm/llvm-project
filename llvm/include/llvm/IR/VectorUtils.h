//===----------- VectorUtils.h -  Vector type utility functions -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_VECTORUTILS_H
#define LLVM_IR_VECTORUTILS_H

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

} // namespace llvm

#endif
