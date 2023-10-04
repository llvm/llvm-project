//===- MlirOptContextPreload.cpp - Test MlirOptMain parameterization ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Transform/IR/TransformDialect.h"
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

namespace {
/// A pass that fails if the transform dialect is not loaded in the context.
/// Sets the flag, the reference to which is passed into the constructor, when
/// runs to check for lack-of-failure because of the pass not running at all.
struct CheckIfTransformIsLoadedPass
    : public PassWrapper<CheckIfTransformIsLoadedPass, OperationPass<>> {
  explicit CheckIfTransformIsLoadedPass(bool &hasRun) : hasRun(hasRun) {}

  void runOnOperation() override {
    hasRun = true;
    if (!getContext().getLoadedDialect<transform::TransformDialect>())
      return signalPassFailure();
  }

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CheckIfTransformIsLoadedPass)

private:
  bool &hasRun;
};
} // namespace

TEST(MlirOptMain, ContextPreloadDialect) {
  registerPassManagerCLOptions();

  MlirOptMainConfig config;

  // Make sure the transform dialect is unconditionally loaded.
  config.setContextConfigurationFn([](MLIRContext &context) {
    context.loadDialect<transform::TransformDialect>();
    return success();
  });

  // Configure the pass pipeline.
  bool hasRun = false;
  config.setPassPipelineSetupFn([&](PassManager &pm) {
    pm.addPass(std::make_unique<CheckIfTransformIsLoadedPass>(hasRun));
    return success();
  });

  // Main.
  DialectRegistry registry;
  LogicalResult mainResult = MlirOptMain(
      llvm::nulls(), llvm::MemoryBuffer::getMemBuffer("", "<empty_buffer>"),
      registry, config);

  EXPECT_TRUE(succeeded(mainResult));
  EXPECT_TRUE(hasRun) << "pass has not run";
}

TEST(MlirOptMain, ContextPreloadDialectNotLoaded) {
  registerPassManagerCLOptions();

  // Configure the pass pipeline, but do not load the transform dialect
  // unconditionally. The pass should run and fail.
  MlirOptMainConfig config;
  bool hasRun = false;
  config.setPassPipelineSetupFn([&](PassManager &pm) {
    pm.addPass(std::make_unique<CheckIfTransformIsLoadedPass>(hasRun));
    return success();
  });

  // Main.
  DialectRegistry registry;
  LogicalResult mainResult = MlirOptMain(
      llvm::nulls(), llvm::MemoryBuffer::getMemBuffer("", "<empty_buffer>"),
      registry, config);

  EXPECT_FALSE(succeeded(mainResult));
  EXPECT_TRUE(hasRun) << "pass has not run";
}

TEST(MlirOptMain, ContextPreloadDialectFailure) {
  registerPassManagerCLOptions();

  // Return failure when configuring the context. The pass should not run.
  MlirOptMainConfig config;
  config.setContextConfigurationFn(
      [](MLIRContext &context) { return failure(); });
  bool hasRun = false;
  config.setPassPipelineSetupFn([&](PassManager &pm) {
    pm.addPass(std::make_unique<CheckIfTransformIsLoadedPass>(hasRun));
    return success();
  });

  // Main.
  DialectRegistry registry;
  LogicalResult mainResult = MlirOptMain(
      llvm::nulls(), llvm::MemoryBuffer::getMemBuffer("", "<empty_buffer>"),
      registry, config);

  EXPECT_FALSE(succeeded(mainResult));
  EXPECT_FALSE(hasRun) << "pass was not expected to run";
}

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
  transform.named_sequence public @print_remark(%arg0: !transform.any_op {transform.readonly}) {
    transform.test_print_remark_at_operand %arg0, "from external symbol" : !transform.any_op
    transform.yield
  }
})MLIR";

const static llvm::StringLiteral input = R"MLIR(
module attributes {transform.with_named_sequence} {
  transform.named_sequence private @print_remark(%arg0: !transform.any_op {transform.readonly})
  transform.sequence failures(propagate) {
  ^bb0(%arg0: !transform.any_op):
    include @print_remark failures(propagate) (%arg0) : (!transform.any_op) -> ()
  }
})MLIR";

TEST(MlirOptMain, ContextPreloadConstructedLibrary) {
  registerPassManagerCLOptions();

  // Make sure the transform dialect is always loaded and make it own a library
  // module that will be used by the pass.
  bool emittedDiagnostic = false;
  MlirOptMainConfig config;
  config.setContextConfigurationFn([&](MLIRContext &context) {
    auto *dialect = context.getOrLoadDialect<transform::TransformDialect>();

    ParserConfig parserConfig(&context);
    OwningOpRef<ModuleOp> transformLibrary = parseSourceString<ModuleOp>(
        library, parserConfig, "<transform-library>");
    if (!transformLibrary)
      return failure();

    dialect->registerLibraryModule(std::move(transformLibrary));

    context.getDiagEngine().registerHandler([&](Diagnostic &diag) {
      if (diag.getSeverity() == DiagnosticSeverity::Remark &&
          diag.str() == "from external symbol") {
        emittedDiagnostic = true;
      }
    });

    return success();
  });

  // Pass pipeline configuration.
  config.setPassPipelineSetupFn([](PassManager &pm) {
    pm.addPass(mlir::test::createTestTransformDialectInterpreterPass());
    return success();
  });

  // We need to register the test extension since the input contains its ops.
  DialectRegistry registry;
  ::test::registerTestTransformDialectExtension(registry);
  std::string fileErrorMessage;
  std::string output;
  llvm::raw_string_ostream os(output);
  LogicalResult mainResult = MlirOptMain(
      os, llvm::MemoryBuffer::getMemBuffer(input, "<input>"), registry, config);

  ASSERT_TRUE(fileErrorMessage.empty()) << fileErrorMessage;
  EXPECT_TRUE(succeeded(mainResult));
  EXPECT_TRUE(emittedDiagnostic)
      << "did not produce the expected diagnostic from external symbol";
}
