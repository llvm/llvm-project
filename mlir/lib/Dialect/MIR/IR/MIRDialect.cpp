//===- MIRDialect.cpp - MIR dialect implementation ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MIR/IR/MIROps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "mlir/Dialect/MIR/IR/MIROpsDialect.cpp.inc"

using namespace mlir;
using namespace mlir::mir;

//===----------------------------------------------------------------------===//
// MIRDialect
//===----------------------------------------------------------------------===//

void MIRDialect::initialize() {
  registerTypes();
  registerAttributes();
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/MIR/IR/MIROps.cpp.inc"
      >();
}

void MIRDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/MIR/IR/MIROpsTypes.cpp.inc"
      >();
}

void MIRDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/MIR/IR/MIROpsAttributes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Generated definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/MIR/IR/MIROpsTypes.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/MIR/IR/MIROpsAttributes.cpp.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/MIR/IR/MIROps.cpp.inc"
