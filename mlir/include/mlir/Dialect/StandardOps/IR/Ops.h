//===- Ops.h - Standard MLIR Operations -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines convenience types for working with standard operations
// in the MLIR operation set.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_STANDARDOPS_IR_OPS_H
#define MLIR_DIALECT_STANDARDOPS_IR_OPS_H

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/VectorInterfaces.h"

// Pull in all enum type definitions and utility function declarations.
#include "mlir/Dialect/StandardOps/IR/OpsEnums.h.inc"

namespace mlir {
class AffineMap;
class Builder;
class FuncOp;
class OpBuilder;
class PatternRewriter;
} // namespace mlir

#define GET_OP_CLASSES
#include "mlir/Dialect/StandardOps/IR/Ops.h.inc"

#include "mlir/Dialect/StandardOps/IR/OpsDialect.h.inc"

namespace mlir {

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

} // end namespace mlir

#endif // MLIR_DIALECT_IR_STANDARDOPS_IR_OPS_H
