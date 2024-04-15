//===- GENDialect.cpp - MLIR GEN Dialect implementation -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/GEN/IR/GENDialect.h"

#include "mlir/Dialect/GEN/IR/GENOps.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::GEN;

#include "mlir/Dialect/GEN/IR/GENOpsDialect.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/GEN/IR/GENOpsAttrDefs.cpp.inc"

//===----------------------------------------------------------------------===//
// GEN dialect.
//===----------------------------------------------------------------------===//

void GENDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/GEN/IR/GENOps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/GEN/IR/GENOpsAttrDefs.cpp.inc"
      >();
}
