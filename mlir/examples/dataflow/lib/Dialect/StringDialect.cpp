//===- StringDialect.cpp - String ops implementation ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the String dialect and its operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "StringDialect.h"

using namespace mlir;
using namespace string;

#include "StringOpsDialect.cpp.inc"

void StringDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "StringOpsTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "StringOps.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "StringOps.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "StringOpsTypes.cpp.inc"
