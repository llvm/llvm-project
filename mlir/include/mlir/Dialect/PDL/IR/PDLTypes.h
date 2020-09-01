//===- PDL.h - Pattern Descriptor Language Types ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the types for the Pattern Descriptor Language dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PDL_IR_PDLTYPES_H_
#define MLIR_DIALECT_PDL_IR_PDLTYPES_H_

#include "mlir/IR/Types.h"

namespace mlir {
namespace pdl {
//===----------------------------------------------------------------------===//
// PDL Dialect Types
//===----------------------------------------------------------------------===//

/// This type represents a handle to an `mlir::Attribute`.
struct AttributeType : public Type::TypeBase<AttributeType, Type, TypeStorage> {
  using Base::Base;
};

/// This type represents a handle to an `mlir::Operation*`.
struct OperationType : public Type::TypeBase<OperationType, Type, TypeStorage> {
  using Base::Base;
};

/// This type represents a handle to an `mlir::Type`.
struct TypeType : public Type::TypeBase<TypeType, Type, TypeStorage> {
  using Base::Base;
};

/// This type represents a handle to an `mlir::Value`.
struct ValueType : public Type::TypeBase<ValueType, Type, TypeStorage> {
  using Base::Base;
};

} // end namespace pdl
} // end namespace mlir

#endif // MLIR_DIALECT_PDL_IR_PDLTYPES_H_
