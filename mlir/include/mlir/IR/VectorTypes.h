//===- VectorTypes.h - MLIR Vector Types ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_VECTOR_IR_VECTORTYPES_H_
#define MLIR_DIALECT_VECTOR_IR_VECTORTYPES_H_

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Types.h"

namespace mlir {
namespace vector {

class ScalableVectorType : public VectorType {
public:
  using VectorType::VectorType;

  static bool classof(Type type);
};

class FixedWidthVectorType : public VectorType {
public:
  using VectorType::VectorType;
  static bool classof(Type type);
};

} // namespace vector
} // namespace mlir

#endif // MLIR_DIALECT_VECTOR_IR_VECTORTYPES_H_
