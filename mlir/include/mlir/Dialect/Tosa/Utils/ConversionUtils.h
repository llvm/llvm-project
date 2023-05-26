//===- ConversionUtils.h - Helper functions for tosa conversion -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utility functions for TOSA lowering
//
//===----------------------------------------------------------------------===//

#ifndef DIALECT_TOSA_UTILS_COVERSION_UTILS_H_
#define DIALECT_TOSA_UTILS_COVERSION_UTILS_H_

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/PatternMatch.h"
#include <optional>

namespace mlir {
namespace tosa {

// Creates a SmallVector of Stringrefs for N parallel loops
SmallVector<utils::IteratorType>
getNParallelLoopsAttrs(unsigned nParallelLoops);

// Takes a vector of values and condenses them to a vector with no gaps.
SmallVector<Value> condenseValues(const SmallVector<Value> &values);

// Takes the parameters for a clamp and turns it into a series of ops for float
// inputs.
Value clampFloatHelper(Location loc, Value arg, Value min, Value max,
                       OpBuilder &rewriter);

// Takes the parameters for a clamp and turns it into a series of ops for
// integer inputs.
Value clampIntHelper(Location loc, Value arg, Value min, Value max,
                     OpBuilder &rewriter);

// Determines whether the integer value falls witin the range of integer type.
bool validIntegerRange(IntegerType ty, int64_t value);

// Checks for a dynamic batch dim in any of the passed parameters of an op.
// The batch dimention must be #0 and the rest of the dimensions must be static.
template <typename Op>
std::optional<SmallVector<Value>>
checkHasDynamicBatchDims(PatternRewriter &rewriter, Op op,
                         ArrayRef<Value> params) {
  SmallVector<ShapedType> dynTypes;
  SmallVector<Value> dynamicDims;
  for (const Value &param : params) {
    auto paramTy = cast<ShapedType>(param.getType());
    if (!paramTy.hasStaticShape())
      dynTypes.push_back(paramTy);
  }

  if (dynTypes.empty())
    return dynamicDims;

  for (const ShapedType &dynTy : dynTypes) {
    if (llvm::any_of(dynTy.getShape().drop_front(), ShapedType::isDynamic)) {
      (void)rewriter.notifyMatchFailure(
          op, "input can only be dynamic for batch size");
      return std::nullopt;
    }
  }

  dynamicDims.push_back(
      rewriter.create<tensor::DimOp>(op->getLoc(), params[0], 0));
  return dynamicDims;
}

/// Common code to create the reshape op where necessary to make the rank of two
/// values equal. input1 and input2 will be updated when the rank has
/// changed. The caller is expected to use these to rewrite the original
/// operator with the RESHAPE now in the graph.
LogicalResult EqualizeRanks(PatternRewriter &rewriter, Location loc,
                            Value &input1, Value &input2);

} // namespace tosa
} // namespace mlir

#endif // DIALECT_TOSA_UTILS_COVERSION_UTILS_H_
