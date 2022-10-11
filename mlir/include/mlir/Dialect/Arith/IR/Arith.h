//===- Arith.h - Arith dialect ------------------------------------*- C++-*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ARITH_IR_ARITH_H_
#define MLIR_DIALECT_ARITH_IR_ARITH_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/VectorInterfaces.h"

//===----------------------------------------------------------------------===//
// ArithDialect
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/ArithOpsDialect.h.inc"

//===----------------------------------------------------------------------===//
// Arith Dialect Enum Attributes
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/ArithOpsEnums.h.inc"

//===----------------------------------------------------------------------===//
// Arith Dialect Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/Arith/IR/ArithOps.h.inc"

namespace mlir {
namespace arith {

/// Specialization of `arith.constant` op that returns an integer value.
class ConstantIntOp : public arith::ConstantOp {
public:
  using arith::ConstantOp::ConstantOp;

  /// Build a constant int op that produces an integer of the specified width.
  static void build(OpBuilder &builder, OperationState &result, int64_t value,
                    unsigned width);

  /// Build a constant int op that produces an integer of the specified type,
  /// which must be an integer type.
  static void build(OpBuilder &builder, OperationState &result, int64_t value,
                    Type type);

  inline int64_t value() {
    return arith::ConstantOp::getValue().cast<IntegerAttr>().getInt();
  }

  static bool classof(Operation *op);
};

/// Specialization of `arith.constant` op that returns a floating point value.
class ConstantFloatOp : public arith::ConstantOp {
public:
  using arith::ConstantOp::ConstantOp;

  /// Build a constant float op that produces a float of the specified type.
  static void build(OpBuilder &builder, OperationState &result,
                    const APFloat &value, FloatType type);

  inline APFloat value() {
    return arith::ConstantOp::getValue().cast<FloatAttr>().getValue();
  }

  static bool classof(Operation *op);
};

/// Specialization of `arith.constant` op that returns an integer of index type.
class ConstantIndexOp : public arith::ConstantOp {
public:
  using arith::ConstantOp::ConstantOp;

  /// Build a constant int op that produces an index.
  static void build(OpBuilder &builder, OperationState &result, int64_t value);

  inline int64_t value() {
    return arith::ConstantOp::getValue().cast<IntegerAttr>().getInt();
  }

  static bool classof(Operation *op);
};

} // namespace arith
} // namespace mlir

//===----------------------------------------------------------------------===//
// Utility Functions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace arith {

/// Compute `lhs` `pred` `rhs`, where `pred` is one of the known integer
/// comparison predicates.
bool applyCmpPredicate(arith::CmpIPredicate predicate, const APInt &lhs,
                       const APInt &rhs);

/// Compute `lhs` `pred` `rhs`, where `pred` is one of the known floating point
/// comparison predicates.
bool applyCmpPredicate(arith::CmpFPredicate predicate, const APFloat &lhs,
                       const APFloat &rhs);

/// Returns the identity value attribute associated with an AtomicRMWKind op.
Attribute getIdentityValueAttr(AtomicRMWKind kind, Type resultType,
                               OpBuilder &builder, Location loc);

/// Returns the identity value associated with an AtomicRMWKind op.
Value getIdentityValue(AtomicRMWKind op, Type resultType, OpBuilder &builder,
                       Location loc);

/// Returns the value obtained by applying the reduction operation kind
/// associated with a binary AtomicRMWKind op to `lhs` and `rhs`.
Value getReductionOp(AtomicRMWKind op, OpBuilder &builder, Location loc,
                     Value lhs, Value rhs);

arith::CmpIPredicate invertPredicate(arith::CmpIPredicate pred);
} // namespace arith
} // namespace mlir

#endif // MLIR_DIALECT_ARITH_IR_ARITH_H_
