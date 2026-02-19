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
Value forEachScalarValue(mlir::RewriterBase &rewriter, Location loc,
                         Value operand1, Value operand2, Type resultType,
                         llvm::function_ref<Value(Value, Value, Type)> fn);

/// Check preconditions for the conversion:
/// 1. All operands / results must be integers or floats (or vectors thereof).
/// 2. The bitwidth of the operands / results must be <= 64.
LogicalResult checkPreconditions(RewriterBase &rewriter, Operation *op);

} // namespace mlir

#endif // MLIR_CONVERSION_ARITHANDMATHTOAPFLOAT_UTILS_H_
