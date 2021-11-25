//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"

using namespace mlir;
using namespace mlir::bufferization;

#include "mlir/Dialect/Bufferization/IR/BufferizationOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Bufferization Dialect
//===----------------------------------------------------------------------===//

void mlir::bufferization::BufferizationDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Bufferization/IR/BufferizationOps.cpp.inc"
      >();
}
