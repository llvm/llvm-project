//===- TestComprehensiveBufferize.cpp - Test Comprehensive Bufferize ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements logic for testing Comprehensive Bufferize.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Arithmetic/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Linalg/ComprehensiveBufferize/AffineInterfaceImpl.h"
#include "mlir/Dialect/Linalg/ComprehensiveBufferize/LinalgInterfaceImpl.h"
#include "mlir/Dialect/Linalg/ComprehensiveBufferize/SCFInterfaceImpl.h"
#include "mlir/Dialect/Linalg/ComprehensiveBufferize/StdInterfaceImpl.h"
#include "mlir/Dialect/Linalg/ComprehensiveBufferize/VectorInterfaceImpl.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::linalg;
using namespace mlir::linalg::comprehensive_bufferize;
using namespace mlir::bufferization;

namespace {
/// A helper struct for FunctionBufferize and ModuleBufferize. Both passes are
/// mostly identical.
struct TestComprehensiveFunctionBufferize
    : public PassWrapper<TestComprehensiveFunctionBufferize,
                         OperationPass<FuncOp>> {
  StringRef getArgument() const final {
    return "test-comprehensive-function-bufferize";
  }

  StringRef getDescription() const final {
    return "Test Comprehensive Bufferize of FuncOps (body only).";
  }

  TestComprehensiveFunctionBufferize() = default;
  TestComprehensiveFunctionBufferize(
      const TestComprehensiveFunctionBufferize &pass)
      : PassWrapper(pass) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<bufferization::BufferizationDialect, linalg::LinalgDialect,
                    memref::MemRefDialect, tensor::TensorDialect,
                    vector::VectorDialect, scf::SCFDialect, StandardOpsDialect,
                    arith::ArithmeticDialect, AffineDialect>();
    affine_ext::registerBufferizableOpInterfaceExternalModels(registry);
    arith::registerBufferizableOpInterfaceExternalModels(registry);
    linalg_ext::registerBufferizableOpInterfaceExternalModels(registry);
    scf_ext::registerBufferizableOpInterfaceExternalModels(registry);
    std_ext::registerBufferizableOpInterfaceExternalModels(registry);
    tensor::registerBufferizableOpInterfaceExternalModels(registry);
    vector_ext::registerBufferizableOpInterfaceExternalModels(registry);
  }

  void runOnOperation() override;

  Option<bool> allowReturnMemref{
      *this, "allow-return-memref",
      llvm::cl::desc("Allow returning/yielding memrefs from functions/blocks"),
      llvm::cl::init(false)};
  Option<bool> allowUnknownOps{
      *this, "allow-unknown-ops",
      llvm::cl::desc(
          "Allows the return of memrefs (for testing purposes only)"),
      llvm::cl::init(false)};
  Option<bool> testAnalysisOnly{
      *this, "test-analysis-only",
      llvm::cl::desc(
          "Only runs inplaceability analysis (for testing purposes only)"),
      llvm::cl::init(false)};
  Option<unsigned> analysisFuzzerSeed{
      *this, "analysis-fuzzer-seed",
      llvm::cl::desc("Analyze ops in random order with a given seed (fuzzer)"),
      llvm::cl::init(0)};
  ListOption<std::string> dialectFilter{
      *this, "dialect-filter",
      llvm::cl::desc("Bufferize only ops from the specified dialects"),
      llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated};
  Option<bool> fullyDynamicLayoutMaps{
      *this, "fully-dynamic-layout-maps",
      llvm::cl::desc("Use fully dynamic layout maps on memref types"),
      llvm::cl::init(true)};
  Option<bool> createDeallocs{
      *this, "create-deallocs",
      llvm::cl::desc("Specify if buffers should be deallocated"),
      llvm::cl::init(true)};
};
} // namespace

void TestComprehensiveFunctionBufferize::runOnOperation() {
  auto options = std::make_unique<AnalysisBufferizationOptions>();

  if (!allowReturnMemref)
    options->addPostAnalysisStep<scf_ext::AssertScfForAliasingProperties>();

  options->allowReturnMemref = allowReturnMemref;
  options->allowUnknownOps = allowUnknownOps;
  options->testAnalysisOnly = testAnalysisOnly;
  options->analysisFuzzerSeed = analysisFuzzerSeed;
  options->fullyDynamicLayoutMaps = fullyDynamicLayoutMaps;
  options->createDeallocs = createDeallocs;

  if (dialectFilter.hasValue()) {
    options->dialectFilter.emplace();
    for (const std::string &dialectNamespace : dialectFilter)
      options->dialectFilter->insert(dialectNamespace);
  }

  Operation *op = getOperation();
  if (failed(runOneShotBufferize(op, std::move(options))))
    return;

  if (testAnalysisOnly)
    return;

  OpPassManager cleanupPipeline("builtin.func");
  cleanupPipeline.addPass(createCanonicalizerPass());
  cleanupPipeline.addPass(createCSEPass());
  cleanupPipeline.addPass(createLoopInvariantCodeMotionPass());
  (void)this->runPipeline(cleanupPipeline, op);
}

namespace mlir {
namespace test {
void registerTestComprehensiveFunctionBufferize() {
  PassRegistration<TestComprehensiveFunctionBufferize>();
}
} // namespace test
} // namespace mlir
