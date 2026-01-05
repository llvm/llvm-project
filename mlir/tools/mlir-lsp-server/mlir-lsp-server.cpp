//===- mlir-lsp-server.cpp - MLIR Language Server -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/Tools/mlir-lsp-server/MlirLspServerMain.h"
#include "llvm/Support/LSP/Protocol.h"

using namespace mlir;

#ifdef MLIR_INCLUDE_TESTS
namespace test {
void registerTestDialect(DialectRegistry &);
void registerTestDynDialect(DialectRegistry &);
void registerTestTransformDialectExtension(DialectRegistry &);
} // namespace test
#endif

int main(int argc, char **argv) {
  DialectRegistry registry, empty;
  registerAllDialects(registry);
  registerAllExtensions(registry);

#ifdef MLIR_INCLUDE_TESTS
  ::test::registerTestDialect(registry);
  ::test::registerTestTransformDialectExtension(registry);
  ::test::registerTestDynDialect(registry);
#endif

  // Returns the registry, except in testing mode when the URI contains
  // "-disable-lsp-registration". Testing for/example of registering dialects
  // based on URI.
  auto registryFn = [&registry, &empty](
                        const llvm::lsp::URIForFile &uri) -> DialectRegistry & {
    (void)empty;
#ifdef MLIR_INCLUDE_TESTS
    if (uri.uri().contains("-disable-lsp-registration"))
      return empty;
#endif
    return registry;
  };
  return failed(MlirLspServerMain(argc, argv, registryFn));
}
