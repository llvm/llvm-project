//===- WebAssemblyDialect.cpp - MLIR WebAssembly dialect implementation ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/WebAssemblySSA/IR/WebAssemblySSA.h"

#include "llvm/ADT/TypeSwitch.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Support/LLVM.h"

using namespace mlir;
using namespace mlir::wasmssa;

#include "mlir/Dialect/WebAssemblySSA/IR/WebAssemblySSAOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// TableGen'd types definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/WebAssemblySSA/IR/WebAssemblySSAOpsTypes.cpp.inc"

void wasmssa::WasmSSADialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/WebAssemblySSA/IR/WebAssemblySSAOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/WebAssemblySSA/IR/WebAssemblySSAOpsTypes.cpp.inc"
      >();
}
