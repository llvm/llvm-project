//===- GENXTypes.h - MLIR GENX Types ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the types in the GENX dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LLVMIR_GENXTYPES_H_
#define MLIR_DIALECT_LLVMIR_GENXTYPES_H_

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"

#include "mlir/Dialect/LLVMIR/GENXOpsEnums.h.inc"

namespace mlir {
namespace GENX {

namespace detail {
struct JointMatrixTypeStorage;
} // namespace detail

class GENXType : public Type {
public:
  using Type::Type;

  static bool classof(Type type);

  std::optional<int64_t> getSizeInBytes() const;
};

class JointMatrixType : public Type::TypeBase<JointMatrixType, GENXType,
                                              detail::JointMatrixTypeStorage> {
public:
  using Base::Base;

  static JointMatrixType get(Type elementType, unsigned rows, unsigned columns,
                             MatrixLayout matrixLayout);

  Type getElementType() const;
  MatrixLayout getMatrixLayout() const;
  unsigned getNumRows() const;
  unsigned getNumColumns() const;
};

} // namespace GENX
} // namespace mlir

#endif // MLIR_DIALECT_LLVMIR_GENXTYPES_H_
