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

namespace mlir::wasm {

/// If `source` contains a valid Wasm binary file, this function returns a
/// a ModuleOp containing the representation of the Wasm module encoded in
/// the source file in the `wasmssa` dialect.
OwningOpRef<ModuleOp> importWebAssemblyToModule(llvm::SourceMgr &source,
                                                MLIRContext *context);
} // namespace mlir::wasm

#endif // MLIR_TARGET_WASM_WASMIMPORTER_H
