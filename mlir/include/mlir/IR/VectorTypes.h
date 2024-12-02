//===- VectorTypes.h - MLIR Vector Types ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Convenience wrappers for `VectorType` to allow idiomatic code like
//  * isa<vector::ScalableVectorType>(type)
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_VECTORTYPES_H
#define MLIR_IR_VECTORTYPES_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"

namespace mlir {
namespace vector {

/// A vector type containing at least one scalable dimension.
class ScalableVectorType : public VectorType {
public:
  using VectorType::VectorType;

  static bool classof(Type type) {
    auto vecTy = llvm::dyn_cast<VectorType>(type);
    if (!vecTy)
      return false;
    return vecTy.isScalable();
  }
};

/// A vector type with no scalable dimensions.
class FixedVectorType : public VectorType {
public:
  using VectorType::VectorType;
  static bool classof(Type type) {
    auto vecTy = llvm::dyn_cast<VectorType>(type);
    if (!vecTy)
      return false;
    return !vecTy.isScalable();
  }
};

} // namespace vector
} // namespace mlir

#endif // MLIR_IR_VECTORTYPES_H
