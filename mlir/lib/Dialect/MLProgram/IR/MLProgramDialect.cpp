//===- MLProgramDialect.cpp - MLProgram dialect implementation ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MLProgram/IR/MLProgram.h"

using namespace mlir;
using namespace mlir::ml_program;

#include "mlir/Dialect/MLProgram/IR/MLProgramOpsDialect.cpp.inc"

void ml_program::MLProgramDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/MLProgram/IR/MLProgramOps.cpp.inc"
      >();
}
