//===- AArch64MIRDialect.cpp - AArch64 MIR dialect impl -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/AArch64MIR/IR/AArch64MIROps.h"
#include "mlir/Dialect/MIR/IR/MIRDialect.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/TypeSwitch.h"

#include "mlir/Dialect/AArch64MIR/IR/AArch64MIROpsDialect.cpp.inc"

using namespace mlir;
using namespace mlir::aarch64_mir;

void AArch64MIRDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/AArch64MIR/IR/AArch64MIROps.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "mlir/Dialect/AArch64MIR/IR/AArch64MIROps.cpp.inc"
