//===- ArmNeonOps.cpp - MLIRArmNeon ops implementation --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the ArmNeon dialect and its operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/ArmNeon/ArmNeonDialect.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"

using namespace mlir;

void arm_neon::ArmNeonDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/ArmNeon/ArmNeon.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "mlir/Dialect/ArmNeon/ArmNeon.cpp.inc"
