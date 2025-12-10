//===- TranslateRegistration.cpp - Register translation -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/WasmSSA/IR/WasmSSA.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Target/Wasm/WasmImporter.h"
#include "mlir/Tools/mlir-translate/Translation.h"

using namespace mlir;

namespace mlir {
void registerFromWasmTranslation() {
  TranslateToMLIRRegistration registration{
      "import-wasm", "Translate WASM to MLIR",
      [](llvm::SourceMgr &sourceMgr,
         MLIRContext *context) -> OwningOpRef<Operation *> {
        return wasm::importWebAssemblyToModule(sourceMgr, context);
      },
      [](DialectRegistry &registry) {
        registry.insert<wasmssa::WasmSSADialect>();
      }};
}
} // namespace mlir
