//===- IndexOps.cpp - Index operation definitions --------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"

using namespace mlir;
using namespace mlir::index;

//===----------------------------------------------------------------------===//
// IndexDialect
//===----------------------------------------------------------------------===//

void IndexDialect::registerOperations() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Index/IR/IndexOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/Index/IR/IndexOps.cpp.inc"
