//===- Utils.h - Utils for APFloat Conversion - C++ -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_ARITHANDMATHTOAPFLOAT_UTILS_H_
#define MLIR_CONVERSION_ARITHANDMATHTOAPFLOAT_UTILS_H_

#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
class Value;
class OpBuilder;
class Location;
class FloatType;

Value getAPFloatSemanticsValue(OpBuilder &b, Location loc, FloatType floatTy);

/// Given two operands of vector type and vector result type (with the same
/// shape), call the given function for each pair of scalar operands and
/// package the result into a vector. If the given operands and result type are
/// not vectors, call the function directly. The second operand is optional.
template <typename Fn, typename... Values>
Value forEachScalarValue(mlir::RewriterBase &rewriter, Location loc,
                         Value operand1, Value operand2, Type resultType,
                         Fn fn) {
  auto vecTy1 = dyn_cast<VectorType>(operand1.getType());
  if (operand2) {
    // Sanity check: Operand types must match.
    assert(vecTy1 == dyn_cast<VectorType>(operand2.getType()) &&
           "expected same vector types");
  }
  if (!vecTy1) {
    // Not a vector. Call the function directly.
    return fn(operand1, operand2, resultType);
  }

  // Prepare scalar operands.
  ResultRange sclars1 =
      vector::ToElementsOp::create(rewriter, loc, operand1)->getResults();
  SmallVector<Value> scalars2;
  if (!operand2) {
    // No second operand. Create a vector of empty values.
    scalars2.assign(vecTy1.getNumElements(), Value());
  } else {
    llvm::append_range(
        scalars2,
        vector::ToElementsOp::create(rewriter, loc, operand2)->getResults());
  }

  // Call the function for each pair of scalar operands.
  auto resultVecType = cast<VectorType>(resultType);
  SmallVector<Value> results;
  for (auto [scalar1, scalar2] : llvm::zip_equal(sclars1, scalars2)) {
    Value result = fn(scalar1, scalar2, resultVecType.getElementType());
    results.push_back(result);
  }

  // Package the results into a vector.
  return vector::FromElementsOp::create(
      rewriter, loc,
      vecTy1.cloneWith(/*shape=*/std::nullopt, results.front().getType()),
      results);
}

/// Check preconditions for the conversion:
/// 1. All operands / results must be integers or floats (or vectors thereof).
/// 2. The bitwidth of the operands / results must be <= 64.
LogicalResult checkPreconditions(RewriterBase &rewriter, Operation *op);

} // namespace mlir

#endif // MLIR_CONVERSION_ARITHANDMATHTOAPFLOAT_UTILS_H_
