//===- ArmSMEDialect.cpp - MLIR ArmSME dialect implementation -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the ArmSME dialect and its operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/ArmSME/IR/ArmSMEDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"

using namespace mlir;
using namespace mlir::arm_sme;

//===----------------------------------------------------------------------===//
// Tablegen Definitions
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/ArmSME/IR/ArmSMEDialect.cpp.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/ArmSME/IR/ArmSME.cpp.inc"

void ArmSMEDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/ArmSME/IR/ArmSME.cpp.inc"
      >();
}
