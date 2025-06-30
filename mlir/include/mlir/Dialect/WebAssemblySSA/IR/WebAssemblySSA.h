//===- WebAssemblySSA.h - WebAssemblySSA dialect ------------------*- C++-*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_WEBASSEMBLYSSA_IR_WEBASSEMBLYSSA_H_
#define MLIR_DIALECT_WEBASSEMBLYSSA_IR_WEBASSEMBLYSSA_H_

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"

//===----------------------------------------------------------------------===//
// WebAssemblyDialect
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/WebAssemblySSA/IR/WebAssemblySSAOpsDialect.h.inc"

//===----------------------------------------------------------------------===//
// WebAssembly Dialect Types
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/WebAssemblySSA/IR/WebAssemblySSAOpsTypes.h.inc"

//===----------------------------------------------------------------------===//
// WebAssembly Interfaces
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/WebAssemblySSA/IR/WebAssemblySSAInterfaces.h"

//===----------------------------------------------------------------------===//
// WebAssembly Dialect Operations
//===----------------------------------------------------------------------===//
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"

//===----------------------------------------------------------------------===//
// WebAssembly Constraints
//===----------------------------------------------------------------------===//

namespace mlir {
namespace wasmssa {
#include "mlir/Dialect/WebAssemblySSA/IR/WebAssemblySSATypeConstraints.h.inc"
}
} // namespace mlir

#define GET_OP_CLASSES
#include "mlir/Dialect/WebAssemblySSA/IR/WebAssemblySSAOps.h.inc"

#endif // MLIR_DIALECT_WEBASSEMBLYSSA_IR_WEBASSEMBLYSSA_H_
