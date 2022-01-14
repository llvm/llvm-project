//===- MathOps.cpp - MLIR operations for math implementation --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Builders.h"

using namespace mlir;
using namespace mlir::math;

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/Math/IR/MathOps.cpp.inc"

//===----------------------------------------------------------------------===//
// CeilOp folder
//===----------------------------------------------------------------------===//

OpFoldResult math::CeilOp::fold(ArrayRef<Attribute> operands) {
  auto constOperand = operands.front();
  if (!constOperand)
    return {};

  auto attr = constOperand.dyn_cast<FloatAttr>();
  if (!attr)
    return {};

  APFloat sourceVal = attr.getValue();
  sourceVal.roundToIntegral(llvm::RoundingMode::TowardPositive);

  return FloatAttr::get(getType(), sourceVal);
}

//===----------------------------------------------------------------------===//
// Log2Op folder
//===----------------------------------------------------------------------===//

OpFoldResult math::Log2Op::fold(ArrayRef<Attribute> operands) {
  auto constOperand = operands.front();
  if (!constOperand)
    return {};

  auto attr = constOperand.dyn_cast<FloatAttr>();
  if (!attr)
    return {};

  auto ft = getType().cast<FloatType>();

  APFloat apf = attr.getValue();

  if (apf.isNegative())
    return {};

  if (ft.getWidth() == 64)
    return FloatAttr::get(getType(), log2(apf.convertToDouble()));

  if (ft.getWidth() == 32)
    return FloatAttr::get(getType(), log2f(apf.convertToDouble()));

  return {};
}

/// Materialize an integer or floating point constant.
Operation *math::MathDialect::materializeConstant(OpBuilder &builder,
                                                  Attribute value, Type type,
                                                  Location loc) {
  return builder.create<arith::ConstantOp>(loc, value, type);
}
