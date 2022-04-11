//===- MathOps.cpp - MLIR operations for math implementation --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/CommonFolders.h"
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
// AbsOp folder
//===----------------------------------------------------------------------===//

OpFoldResult math::AbsOp::fold(ArrayRef<Attribute> operands) {
  return constFoldUnaryOp<FloatAttr>(operands, [](const APFloat &a) {
    APFloat result(a);
    return abs(result);
  });
}

//===----------------------------------------------------------------------===//
// CeilOp folder
//===----------------------------------------------------------------------===//

OpFoldResult math::CeilOp::fold(ArrayRef<Attribute> operands) {
  return constFoldUnaryOp<FloatAttr>(operands, [](const APFloat &a) {
    APFloat result(a);
    result.roundToIntegral(llvm::RoundingMode::TowardPositive);
    return result;
  });
}

//===----------------------------------------------------------------------===//
// CopySignOp folder
//===----------------------------------------------------------------------===//

OpFoldResult math::CopySignOp::fold(ArrayRef<Attribute> operands) {
  return constFoldBinaryOp<FloatAttr>(operands,
                                      [](const APFloat &a, const APFloat &b) {
                                        APFloat result(a);
                                        result.copySign(b);
                                        return result;
                                      });
}

//===----------------------------------------------------------------------===//
// CountLeadingZerosOp folder
//===----------------------------------------------------------------------===//

OpFoldResult math::CountLeadingZerosOp::fold(ArrayRef<Attribute> operands) {
  return constFoldUnaryOp<IntegerAttr>(operands, [](const APInt &a) {
    return APInt(a.getBitWidth(), a.countLeadingZeros());
  });
}

//===----------------------------------------------------------------------===//
// CountTrailingZerosOp folder
//===----------------------------------------------------------------------===//

OpFoldResult math::CountTrailingZerosOp::fold(ArrayRef<Attribute> operands) {
  return constFoldUnaryOp<IntegerAttr>(operands, [](const APInt &a) {
    return APInt(a.getBitWidth(), a.countTrailingZeros());
  });
}

//===----------------------------------------------------------------------===//
// CtPopOp folder
//===----------------------------------------------------------------------===//

OpFoldResult math::CtPopOp::fold(ArrayRef<Attribute> operands) {
  return constFoldUnaryOp<IntegerAttr>(operands, [](const APInt &a) {
    return APInt(a.getBitWidth(), a.countPopulation());
  });
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
    return FloatAttr::get(getType(), log2f(apf.convertToFloat()));

  return {};
}

//===----------------------------------------------------------------------===//
// PowFOp folder
//===----------------------------------------------------------------------===//

OpFoldResult math::PowFOp::fold(ArrayRef<Attribute> operands) {
  auto ft = getType().dyn_cast<FloatType>();
  if (!ft)
    return {};

  APFloat vals[2]{APFloat(ft.getFloatSemantics()),
                  APFloat(ft.getFloatSemantics())};
  for (int i = 0; i < 2; ++i) {
    if (!operands[i])
      return {};

    auto attr = operands[i].dyn_cast<FloatAttr>();
    if (!attr)
      return {};

    vals[i] = attr.getValue();
  }

  if (ft.getWidth() == 64)
    return FloatAttr::get(
        getType(), pow(vals[0].convertToDouble(), vals[1].convertToDouble()));

  if (ft.getWidth() == 32)
    return FloatAttr::get(
        getType(), powf(vals[0].convertToFloat(), vals[1].convertToFloat()));

  return {};
}

OpFoldResult math::SqrtOp::fold(ArrayRef<Attribute> operands) {
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
    return FloatAttr::get(getType(), sqrt(apf.convertToDouble()));

  if (ft.getWidth() == 32)
    return FloatAttr::get(getType(), sqrtf(apf.convertToFloat()));

  return {};
}

/// Materialize an integer or floating point constant.
Operation *math::MathDialect::materializeConstant(OpBuilder &builder,
                                                  Attribute value, Type type,
                                                  Location loc) {
  return builder.create<arith::ConstantOp>(loc, value, type);
}
