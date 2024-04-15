//===- GENTraits.cpp - GEN dialect traits ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/GEN/IR/GENTraits.h"

#include "mlir/IR/Matchers.h"

using namespace mlir;

LogicalResult mlir::OpTrait::GEN::detail::verifyGEN3DNDRange(Operation *op) {
  llvm::APInt value;
  if (matchPattern(op->getOperand(0), m_ConstantInt(&value)) &&
      !(/*value in [0, 3)*/ value.sge(0) && value.slt(3))) {
    return op->emitOpError()
           << "input dimension must be in the range [0, 3). Got "
           << value.getSExtValue();
  }
  return success();
}
