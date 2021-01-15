//===- ComplexDialect.cpp - MLIR Complex Dialect --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Complex/IR/Complex.h"

void mlir::complex::ComplexDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Complex/IR/ComplexOps.cpp.inc"
      >();
}
