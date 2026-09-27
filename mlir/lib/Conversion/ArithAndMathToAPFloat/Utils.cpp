//===- Utils.cpp - Utils for APFloat Conversion ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Utils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"

using namespace mlir;

Value mlir::getAPFloatSemanticsValue(OpBuilder &b, Location loc,
                                     FloatType floatTy) {
  int32_t sem = llvm::APFloatBase::SemanticsToEnum(floatTy.getFloatSemantics());
  return arith::ConstantOp::create(b, loc, b.getI32Type(),
                                   b.getIntegerAttr(b.getI32Type(), sem));
}

Value mlir::forEachScalarValue(
    mlir::RewriterBase &rewriter, Location loc, Value operand1, Value operand2,
    Type resultType, llvm::function_ref<Value(Value, Value, Type)> fn) {
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

LogicalResult mlir::checkPreconditions(RewriterBase &rewriter, Operation *op) {
  for (Value value : llvm::concat<Value>(op->getOperands(), op->getResults())) {
    Type type = value.getType();
    if (auto vecTy = dyn_cast<VectorType>(type)) {
      type = vecTy.getElementType();
    }
    if (!type.isIntOrFloat()) {
      return rewriter.notifyMatchFailure(
          op, "only integers and floats (or vectors thereof) are supported");
    }
    if (type.getIntOrFloatBitWidth() > 64)
      return rewriter.notifyMatchFailure(op,
                                         "bitwidth > 64 bits is not supported");
  }
  return success();
}
