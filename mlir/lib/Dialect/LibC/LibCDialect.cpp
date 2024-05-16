//===- LibCDialect.cpp - MLIR LibC ops implementation -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the LibC dialect and its operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LibC/LibCDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"

using namespace mlir;

#include "mlir/Dialect/LibC/LibCDialect.cpp.inc"

void libc::LibCDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/LibC/LibC.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "mlir/Dialect/LibC/LibC.cpp.inc"
