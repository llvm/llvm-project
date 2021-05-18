//===- ComplexOps.cpp - MLIR Complex Operations ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Complex/IR/Complex.h"

using namespace mlir;
using namespace mlir::complex;

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/Complex/IR/ComplexOps.cpp.inc"

OpFoldResult ReOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.size() == 1 && "unary op takes 1 operand");
  ArrayAttr arrayAttr = operands[0].dyn_cast_or_null<ArrayAttr>();
  if (arrayAttr && arrayAttr.size() == 2)
    return arrayAttr[0];
  return {};
}

OpFoldResult ImOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.size() == 1 && "unary op takes 1 operand");
  ArrayAttr arrayAttr = operands[0].dyn_cast_or_null<ArrayAttr>();
  if (arrayAttr && arrayAttr.size() == 2)
    return arrayAttr[1];
  return {};
}
