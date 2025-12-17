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
