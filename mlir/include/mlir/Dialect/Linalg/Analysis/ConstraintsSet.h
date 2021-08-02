//===- ConstraintsSet.h - Extensions for FlatAffineConstraints --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Linalg-specific constraints set extensions.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LINALG_ANALYSIS_CONSTRAINTS_SET_H_
#define MLIR_DIALECT_LINALG_ANALYSIS_CONSTRAINTS_SET_H_

#include "mlir/Analysis/AffineStructures.h"
#include "mlir/IR/AffineMap.h"

namespace mlir {
class ValueRange;

/// Linalg-specific constraints set extensions.
class ConstraintsSet : public FlatAffineConstraints {
public:
  ConstraintsSet() : FlatAffineConstraints() {}

  /// Assuming `val` is defined by `val = affine.min map (operands)`, introduce
  /// all the constraints `val <= expr_i(operands)`, where expr_i are all the
  /// results of `map`.
  // This API avoids taking a dependence on the AffineMinOp definition.
  LogicalResult composeMin(Value val, AffineMap map, ValueRange operands) {
    return composeMinOrMaxMapAndOperands(val, map, operands, /*min=*/true);
  }

  /// Assuming `val` is defined by `val = affine.max map (operands)`, introduce
  /// all the constraints `val >= expr_i(operands)`, where expr_i are all the
  /// results of `map`.
  // This API avoids taking a dependence on the AffineMaxOp definition.
  LogicalResult composeMax(Value val, AffineMap map, ValueRange operands) {
    return composeMinOrMaxMapAndOperands(val, map, operands, /*min=*/false);
  }

  /// Assuming `val` is defined by `val = affine.apply map (operands)`, call
  /// composeMap.
  // This API avoids taking a dependence on the AffineMApplyOp definition.
  LogicalResult composeAffineApply(Value val, AffineMap map,
                                   ValueRange operands);

  /// Asserts the identifier `id` is in the constraints set and returns it.
  unsigned lookupPos(Value id) const;

  /// If v is not in the constraint set, insert it as a dim or symbol depending
  /// on `asDim`.
  /// Return success if v is of dim id type when `asDim` is true and of symbol
  /// id type when `asDim` is false.
  /// Return failure otherwise.
  LogicalResult ensureIdOfType(Value v, bool asDim);

private:
  /// Implementation detail for composeMin/Max.
  LogicalResult composeMinOrMaxMapAndOperands(Value val, AffineMap map,
                                              ValueRange operands, bool min);
};

} // namespace mlir

#endif // MLIR_DIALECT_LINALG_ANALYSIS_CONSTRAINTS_SET_H_
