//===- WasmImporter.h - Helpers to create WebAssembly emitter ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines helpers to import WebAssembly code using the WebAssembly
// dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TARGET_WASM_WASMIMPORTER_H
#define MLIR_TARGET_WASM_WASMIMPORTER_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "llvm/Support/SourceMgr.h"

namespace mlir {
namespace wasm {

/// Translates the given operation to C++ code. The operation or operations in
/// the region of 'op' need almost all be in EmitC dialect. The parameter
/// 'declareVariablesAtTop' enforces that all variables for op results and block
/// arguments are declared at the beginning of the function.
/// If parameter 'fileId' is non-empty, then body of `emitc.file` ops
/// with matching id are emitted.
OwningOpRef<ModuleOp> importWebAssemblyToModule(llvm::SourceMgr &source, MLIRContext* context);
} // namespace wasm
} // namespace mlir

#endif // MLIR_TARGET_WASM_WASMIMPORTER_H
