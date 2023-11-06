//===- VCIXDialect.cpp - MLIR VCIX ops implementation ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the VCIX dialect and its operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/VCIX/VCIXDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"

using namespace mlir;

#include "mlir/Dialect/VCIX/VCIXDialect.cpp.inc"

void vcix::VCIXDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/VCIX/VCIX.cpp.inc"
      >();
}
