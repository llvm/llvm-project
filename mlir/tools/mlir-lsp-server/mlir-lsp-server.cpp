//===- mlir-lsp-server.cpp - MLIR Language Server -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Tools/mlir-lsp-server/MlirLspServerMain.h"

using namespace mlir;

#ifdef MLIR_INCLUDE_TESTS
namespace test {
void registerTestDialect(DialectRegistry &);
void registerTestTransformDialectExtension(DialectRegistry &);
} // namespace test
#endif

int main(int argc, char **argv) {
  DialectRegistry registry;
  registerAllDialects(registry);
#ifdef MLIR_INCLUDE_TESTS
  ::test::registerTestDialect(registry);
  ::test::registerTestTransformDialectExtension(registry);
#endif
  return failed(MlirLspServerMain(argc, argv, registry));
}
