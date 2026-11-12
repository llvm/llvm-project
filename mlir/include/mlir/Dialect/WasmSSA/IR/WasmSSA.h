//===- WasmSSA.h - WasmSSA dialect ------------------*- C++-*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_WasmSSA_IR_WasmSSA_H_
#define MLIR_DIALECT_WasmSSA_IR_WasmSSA_H_

//===----------------------------------------------------------------------===//
// WebAssemblyDialect
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/WasmSSA/IR/WasmSSAOpsDialect.h.inc"

//===----------------------------------------------------------------------===//
// WebAssembly Dialect Types
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/WasmSSA/IR/WasmSSAOpsTypes.h.inc"

//===----------------------------------------------------------------------===//
// WebAssembly Interfaces
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/WasmSSA/IR/WasmSSAInterfaces.h"

//===----------------------------------------------------------------------===//
// WebAssembly Dialect Operations
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// WebAssembly Constraints
//===----------------------------------------------------------------------===//

namespace mlir {
namespace wasmssa {
#include "mlir/Dialect/WasmSSA/IR/WasmSSATypeConstraints.h.inc"
}
} // namespace mlir

#define GET_OP_CLASSES
#include "mlir/Dialect/WasmSSA/IR/WasmSSAOps.h.inc"

#endif // MLIR_DIALECT_WasmSSA_IR_WasmSSA_H_
