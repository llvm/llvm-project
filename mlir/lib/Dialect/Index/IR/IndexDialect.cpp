//===- IndexDialect.cpp - Index dialect definition -------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"

using namespace mlir;
using namespace mlir::index;

//===----------------------------------------------------------------------===//
// IndexDialect
//===----------------------------------------------------------------------===//

void IndexDialect::initialize() {
  registerAttributes();
  registerOperations();
  declarePromisedInterface<IndexDialect, ConvertToLLVMPatternInterface>();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Index/IR/IndexOpsDialect.cpp.inc"
