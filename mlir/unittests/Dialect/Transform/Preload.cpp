//===- Preload.cpp - Test MlirOptMain parameterization ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/Transforms/TransformInterpreterUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/TypeID.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

using namespace mlir;

namespace mlir {
namespace test {
std::unique_ptr<Pass> createTestTransformDialectInterpreterPass();
} // namespace test
} // namespace mlir
namespace test {
void registerTestTransformDialectExtension(DialectRegistry &registry);
} // namespace test

const static llvm::StringLiteral library = R"MLIR(
module attributes {transform.with_named_sequence} {
  transform.named_sequence private @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    transform.test_print_remark_at_operand %arg0, "from external symbol" : !transform.any_op
    transform.yield
  }
})MLIR";

const static llvm::StringLiteral input = R"MLIR(
module attributes {transform.with_named_sequence} {
  transform.named_sequence private @__transform_main(%arg0: !transform.any_op {transform.readonly})

  transform.sequence failures(propagate) {
  ^bb0(%arg0: !transform.any_op):
    include @__transform_main failures(propagate) (%arg0) : (!transform.any_op) -> ()
  }
})MLIR";

TEST(Preload, ContextPreloadConstructedLibrary) {
  registerPassManagerCLOptions();

  MLIRContext context;
  auto *dialect = context.getOrLoadDialect<transform::TransformDialect>();
  DialectRegistry registry;
  ::test::registerTestTransformDialectExtension(registry);
  registry.applyExtensions(&context);
  ParserConfig parserConfig(&context);

  OwningOpRef<ModuleOp> inputModule =
      parseSourceString<ModuleOp>(input, parserConfig, "<input>");
  EXPECT_TRUE(inputModule) << "failed to parse input module";

  OwningOpRef<ModuleOp> transformLibrary =
      parseSourceString<ModuleOp>(library, parserConfig, "<transform-library>");
  EXPECT_TRUE(transformLibrary) << "failed to parse transform module";
  dialect->registerLibraryModule(std::move(transformLibrary));

  ModuleOp retrievedTransformLibrary =
      transform::detail::getPreloadedTransformModule(&context);
  EXPECT_TRUE(retrievedTransformLibrary)
      << "failed to retrieve transform module";

  transform::TransformOpInterface entryPoint =
      transform::detail::findTransformEntryPoint(inputModule->getOperation(),
                                                 retrievedTransformLibrary);
  EXPECT_TRUE(entryPoint) << "failed to find entry point";

  OwningOpRef<Operation *> clonedTransformModule(
      retrievedTransformLibrary->clone());
  LogicalResult res = transform::detail::mergeSymbolsInto(
      inputModule->getOperation(), std::move(clonedTransformModule));
  EXPECT_TRUE(succeeded(res)) << "failed to define declared symbols";

  transform::TransformOptions options;
  res = transform::applyTransformNamedSequence(
      inputModule->getOperation(), retrievedTransformLibrary, options);
  EXPECT_TRUE(succeeded(res)) << "failed to apply named sequence";
}
