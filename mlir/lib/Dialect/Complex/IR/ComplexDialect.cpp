//===- ComplexDialect.cpp - MLIR Complex Dialect --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

using namespace mlir;

#include "mlir/Dialect/Complex/IR/ComplexOpsDialect.cpp.inc"

void complex::ComplexDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Complex/IR/ComplexOps.cpp.inc"
      >();
}

Operation *complex::ComplexDialect::materializeConstant(OpBuilder &builder,
                                                        Attribute value,
                                                        Type type,
                                                        Location loc) {
  // TODO complex.constant
  if (type.isa<ComplexType>())
    return builder.create<ConstantOp>(loc, type, value);
  if (arith::ConstantOp::isBuildableWith(value, type))
    return builder.create<arith::ConstantOp>(loc, type, value);
  return nullptr;
}
