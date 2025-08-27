//===- WebAssemblyDialect.cpp - MLIR WebAssembly dialect implementation ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/WasmSSA/IR/WasmSSA.h"

#include "llvm/ADT/TypeSwitch.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Support/LLVM.h"

using namespace mlir;
using namespace mlir::wasmssa;

#include "mlir/Dialect/WasmSSA/IR/WasmSSAOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// TableGen'd types definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/WasmSSA/IR/WasmSSAOpsTypes.cpp.inc"

void wasmssa::WasmSSADialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/WasmSSA/IR/WasmSSAOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/WasmSSA/IR/WasmSSAOpsTypes.cpp.inc"
      >();
}
