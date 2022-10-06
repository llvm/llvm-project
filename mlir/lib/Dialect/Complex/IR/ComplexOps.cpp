//===- ComplexOps.cpp - MLIR Complex Operations ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"

using namespace mlir;
using namespace mlir::complex;

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

OpFoldResult ConstantOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.empty() && "constant has no operands");
  return getValue();
}

void ConstantOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "cst");
}

bool ConstantOp::isBuildableWith(Attribute value, Type type) {
  if (auto arrAttr = value.dyn_cast<ArrayAttr>()) {
    auto complexTy = type.dyn_cast<ComplexType>();
    if (!complexTy || arrAttr.size() != 2)
      return false;
    auto complexEltTy = complexTy.getElementType();
    auto re = arrAttr[0].dyn_cast<FloatAttr>();
    auto im = arrAttr[1].dyn_cast<FloatAttr>();
    return re && im && re.getType() == complexEltTy &&
           im.getType() == complexEltTy;
  }
  return false;
}

LogicalResult ConstantOp::verify() {
  ArrayAttr arrayAttr = getValue();
  if (arrayAttr.size() != 2) {
    return emitOpError(
        "requires 'value' to be a complex constant, represented as array of "
        "two values");
  }

  auto complexEltTy = getType().getElementType();
  auto re = arrayAttr[0].dyn_cast<FloatAttr>();
  auto im = arrayAttr[1].dyn_cast<FloatAttr>();
  if (!re || !im)
    return emitOpError("requires attribute's elements to be float attributes");
  if (complexEltTy != re.getType() || complexEltTy != im.getType()) {
    return emitOpError()
           << "requires attribute's element types (" << re.getType() << ", "
           << im.getType()
           << ") to match the element type of the op's return type ("
           << complexEltTy << ")";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// CreateOp
//===----------------------------------------------------------------------===//

OpFoldResult CreateOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.size() == 2 && "binary op takes two operands");
  // Fold complex.create(complex.re(op), complex.im(op)).
  if (auto reOp = getOperand(0).getDefiningOp<ReOp>()) {
    if (auto imOp = getOperand(1).getDefiningOp<ImOp>()) {
      if (reOp.getOperand() == imOp.getOperand()) {
        return reOp.getOperand();
      }
    }
  }
  return {};
}

//===----------------------------------------------------------------------===//
// ImOp
//===----------------------------------------------------------------------===//

OpFoldResult ImOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.size() == 1 && "unary op takes 1 operand");
  ArrayAttr arrayAttr = operands[0].dyn_cast_or_null<ArrayAttr>();
  if (arrayAttr && arrayAttr.size() == 2)
    return arrayAttr[1];
  if (auto createOp = getOperand().getDefiningOp<CreateOp>())
    return createOp.getOperand(1);
  return {};
}

//===----------------------------------------------------------------------===//
// ReOp
//===----------------------------------------------------------------------===//

OpFoldResult ReOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.size() == 1 && "unary op takes 1 operand");
  ArrayAttr arrayAttr = operands[0].dyn_cast_or_null<ArrayAttr>();
  if (arrayAttr && arrayAttr.size() == 2)
    return arrayAttr[0];
  if (auto createOp = getOperand().getDefiningOp<CreateOp>())
    return createOp.getOperand(0);
  return {};
}

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

OpFoldResult AddOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.size() == 2 && "binary op takes 2 operands");

  // complex.add(complex.sub(a, b), b) -> a
  if (auto sub = getLhs().getDefiningOp<SubOp>())
    if (getRhs() == sub.getRhs())
      return sub.getLhs();

  // complex.add(b, complex.sub(a, b)) -> a
  if (auto sub = getRhs().getDefiningOp<SubOp>())
    if (getLhs() == sub.getRhs())
      return sub.getLhs();

  // complex.add(a, complex.constant<0.0, 0.0>) -> a
  if (auto constantOp = getRhs().getDefiningOp<ConstantOp>()) {
    auto arrayAttr = constantOp.getValue();
    if (arrayAttr[0].cast<FloatAttr>().getValue().isZero() &&
        arrayAttr[1].cast<FloatAttr>().getValue().isZero()) {
      return getLhs();
    }
  }

  return {};
}

//===----------------------------------------------------------------------===//
// SubOp
//===----------------------------------------------------------------------===//

OpFoldResult SubOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.size() == 2 && "binary op takes 2 operands");

  // complex.sub(complex.add(a, b), b) -> a
  if (auto add = getLhs().getDefiningOp<AddOp>())
    if (getRhs() == add.getRhs())
      return add.getLhs();

  return {};
}

//===----------------------------------------------------------------------===//
// NegOp
//===----------------------------------------------------------------------===//

OpFoldResult NegOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.size() == 1 && "unary op takes 1 operand");

  // complex.neg(complex.neg(a)) -> a
  if (auto negOp = getOperand().getDefiningOp<NegOp>())
    return negOp.getOperand();

  return {};
}

//===----------------------------------------------------------------------===//
// LogOp
//===----------------------------------------------------------------------===//

OpFoldResult LogOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.size() == 1 && "unary op takes 1 operand");

  // complex.log(complex.exp(a)) -> a
  if (auto expOp = getOperand().getDefiningOp<ExpOp>())
    return expOp.getOperand();

  return {};
}

//===----------------------------------------------------------------------===//
// ExpOp
//===----------------------------------------------------------------------===//

OpFoldResult ExpOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.size() == 1 && "unary op takes 1 operand");

  // complex.exp(complex.log(a)) -> a
  if (auto logOp = getOperand().getDefiningOp<LogOp>())
    return logOp.getOperand();

  return {};
}

//===----------------------------------------------------------------------===//
// ConjOp
//===----------------------------------------------------------------------===//

OpFoldResult ConjOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.size() == 1 && "unary op takes 1 operand");

  // complex.conj(complex.conj(a)) -> a
  if (auto conjOp = getOperand().getDefiningOp<ConjOp>())
    return conjOp.getOperand();

  return {};
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/Complex/IR/ComplexOps.cpp.inc"
