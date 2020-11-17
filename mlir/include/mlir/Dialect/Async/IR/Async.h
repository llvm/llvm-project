//===- Async.h - MLIR Async dialect -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the async dialect that is used for modeling asynchronous
// execution.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ASYNC_IR_ASYNC_H
#define MLIR_DIALECT_ASYNC_IR_ASYNC_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
namespace async {

namespace detail {
struct ValueTypeStorage;
} // namespace detail

/// The token type to represent asynchronous operation completion.
class TokenType : public Type::TypeBase<TokenType, Type, TypeStorage> {
public:
  using Base::Base;
};

/// The value type to represent values returned from asynchronous operations.
class ValueType
    : public Type::TypeBase<ValueType, Type, detail::ValueTypeStorage> {
public:
  using Base::Base;

  /// Get or create an async ValueType with the provided value type.
  static ValueType get(Type valueType);

  Type getValueType();
};

/// The group type to represent async tokens or values grouped together.
class GroupType : public Type::TypeBase<GroupType, Type, TypeStorage> {
public:
  using Base::Base;
};

} // namespace async
} // namespace mlir

#define GET_OP_CLASSES
#include "mlir/Dialect/Async/IR/AsyncOps.h.inc"

#include "mlir/Dialect/Async/IR/AsyncOpsDialect.h.inc"

#endif // MLIR_DIALECT_ASYNC_IR_ASYNC_H
