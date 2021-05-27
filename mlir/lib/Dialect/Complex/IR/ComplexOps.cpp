//===- ComplexOps.cpp - MLIR Complex Operations ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/IR/Builders.h"

using namespace mlir;
using namespace mlir::complex;

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/Complex/IR/ComplexOps.cpp.inc"

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

OpFoldResult ImOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.size() == 1 && "unary op takes 1 operand");
  ArrayAttr arrayAttr = operands[0].dyn_cast_or_null<ArrayAttr>();
  if (arrayAttr && arrayAttr.size() == 2)
    return arrayAttr[1];
  if (auto createOp = getOperand().getDefiningOp<CreateOp>())
    return createOp.getOperand(1);
  return {};
}

OpFoldResult ReOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.size() == 1 && "unary op takes 1 operand");
  ArrayAttr arrayAttr = operands[0].dyn_cast_or_null<ArrayAttr>();
  if (arrayAttr && arrayAttr.size() == 2)
    return arrayAttr[0];
  if (auto createOp = getOperand().getDefiningOp<CreateOp>())
    return createOp.getOperand(0);
  return {};
}
