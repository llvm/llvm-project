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

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/VectorInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

// Pull in all enum type definitions and utility function declarations.
#include "mlir/Dialect/StandardOps/IR/OpsEnums.h.inc"

namespace mlir {
class AffineMap;
class Builder;
class FuncOp;
class OpBuilder;

/// Return the list of Range (i.e. offset, size, stride). Each Range
/// entry contains either the dynamic value or a ConstantIndexOp constructed
/// with `b` at location `loc`.
SmallVector<Range, 8> getOrCreateRanges(OffsetSizeAndStrideOpInterface op,
                                        OpBuilder &b, Location loc);

#define GET_OP_CLASSES
#include "mlir/Dialect/StandardOps/IR/Ops.h.inc"

#include "mlir/Dialect/StandardOps/IR/OpsDialect.h.inc"

/// This is a refinement of the "constant" op for the case where it is
/// returning a float value of FloatType.
///
///   %1 = "std.constant"(){value: 42.0} : bf16
///
class ConstantFloatOp : public ConstantOp {
public:
  using ConstantOp::ConstantOp;

  /// Builds a constant float op producing a float of the specified type.
  static void build(OpBuilder &builder, OperationState &result,
                    const APFloat &value, FloatType type);

  APFloat getValue() {
    return (*this)->getAttrOfType<FloatAttr>("value").getValue();
  }

  static bool classof(Operation *op);
};

/// This is a refinement of the "constant" op for the case where it is
/// returning an integer value of IntegerType.
///
///   %1 = "std.constant"(){value: 42} : i32
///
class ConstantIntOp : public ConstantOp {
public:
  using ConstantOp::ConstantOp;
  /// Build a constant int op producing an integer of the specified width.
  static void build(OpBuilder &builder, OperationState &result, int64_t value,
                    unsigned width);

  /// Build a constant int op producing an integer with the specified type,
  /// which must be an integer type.
  static void build(OpBuilder &builder, OperationState &result, int64_t value,
                    Type type);

  int64_t getValue() {
    return (*this)->getAttrOfType<IntegerAttr>("value").getInt();
  }

  static bool classof(Operation *op);
};

/// This is a refinement of the "constant" op for the case where it is
/// returning an integer value of Index type.
///
///   %1 = "std.constant"(){value: 99} : () -> index
///
class ConstantIndexOp : public ConstantOp {
public:
  using ConstantOp::ConstantOp;

  /// Build a constant int op producing an index.
  static void build(OpBuilder &builder, OperationState &result, int64_t value);

  int64_t getValue() {
    return (*this)->getAttrOfType<IntegerAttr>("value").getInt();
  }

  static bool classof(Operation *op);
};

/// Given an `originalShape` and a `reducedShape` assumed to be a subset of
/// `originalShape` with some `1` entries erased, return the set of indices
/// that specifies which of the entries of `originalShape` are dropped to obtain
/// `reducedShape`. The returned mask can be applied as a projection to
/// `originalShape` to obtain the `reducedShape`. This mask is useful to track
/// which dimensions must be kept when e.g. compute MemRef strides under
/// rank-reducing operations. Return None if reducedShape cannot be obtained
/// by dropping only `1` entries in `originalShape`.
llvm::Optional<llvm::SmallDenseSet<unsigned>>
computeRankReductionMask(ArrayRef<int64_t> originalShape,
                         ArrayRef<int64_t> reducedShape);

/// Compute `lhs` `pred` `rhs`, where `pred` is one of the known integer
/// comparison predicates.
bool applyCmpPredicate(CmpIPredicate predicate, const APInt &lhs,
                       const APInt &rhs);

/// Compute `lhs` `pred` `rhs`, where `pred` is one of the known floating point
/// comparison predicates.
bool applyCmpPredicate(CmpFPredicate predicate, const APFloat &lhs,
                       const APFloat &rhs);

/// Return true if ofr1 and ofr2 are the same integer constant attribute values
/// or the same SSA value.
/// Ignore integer bitwitdh and type mismatch that come from the fact there is
/// no IndexAttr and that IndexType have no bitwidth.
bool isEqualConstantIntOrValue(OpFoldResult ofr1, OpFoldResult ofr2);
} // end namespace mlir

#endif // MLIR_DIALECT_IR_STANDARDOPS_IR_OPS_H
