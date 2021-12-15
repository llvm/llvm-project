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
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/ComprehensiveBufferize/AffineInterfaceImpl.h"
#include "mlir/Dialect/Linalg/ComprehensiveBufferize/ArithInterfaceImpl.h"
#include "mlir/Dialect/Linalg/ComprehensiveBufferize/BufferizableOpInterface.h"
#include "mlir/Dialect/Linalg/ComprehensiveBufferize/BufferizationInterfaceImpl.h"
#include "mlir/Dialect/Linalg/ComprehensiveBufferize/ComprehensiveBufferize.h"
#include "mlir/Dialect/Linalg/ComprehensiveBufferize/LinalgInterfaceImpl.h"
#include "mlir/Dialect/Linalg/ComprehensiveBufferize/ModuleBufferization.h"
#include "mlir/Dialect/Linalg/ComprehensiveBufferize/SCFInterfaceImpl.h"
#include "mlir/Dialect/Linalg/ComprehensiveBufferize/TensorInterfaceImpl.h"
#include "mlir/Dialect/Linalg/ComprehensiveBufferize/VectorInterfaceImpl.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::linalg;
using namespace mlir::linalg::comprehensive_bufferize;

namespace {
/// A helper struct for FunctionBufferize and ModuleBufferize. Both passes are
/// mostly identical.
struct TestComprehensiveFunctionBufferize
    : public PassWrapper<TestComprehensiveFunctionBufferize, FunctionPass> {
  StringRef getArgument() const final {
    return "test-comprehensive-function-bufferize";
  }

  StringRef getDescription() const final {
    return "Test Comprehensive Bufferize of FuncOps (body only).";
  }

  TestComprehensiveFunctionBufferize() = default;
  TestComprehensiveFunctionBufferize(
      const TestComprehensiveFunctionBufferize &pass) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<bufferization::BufferizationDialect, linalg::LinalgDialect,
                    memref::MemRefDialect, tensor::TensorDialect,
                    vector::VectorDialect, scf::SCFDialect,
                    arith::ArithmeticDialect, AffineDialect>();
    affine_ext::registerBufferizableOpInterfaceExternalModels(registry);
    arith_ext::registerBufferizableOpInterfaceExternalModels(registry);
    bufferization_ext::registerBufferizableOpInterfaceExternalModels(registry);
    linalg_ext::registerBufferizableOpInterfaceExternalModels(registry);
    scf_ext::registerBufferizableOpInterfaceExternalModels(registry);
    tensor_ext::registerBufferizableOpInterfaceExternalModels(registry);
    vector_ext::registerBufferizableOpInterfaceExternalModels(registry);
  }

  void runOnFunction() override;

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
};
} // namespace

void TestComprehensiveFunctionBufferize::runOnFunction() {
  BufferizationOptions options;

  // Enable InitTensorOp elimination.
  options.addPostAnalysisStep<
      linalg_ext::InsertSliceAnchoredInitTensorEliminationStep>();
  // TODO: Find a way to enable this step automatically when bufferizing
  // tensor dialect ops.
  options.addPostAnalysisStep<tensor_ext::InplaceInsertSliceOpAnalysis>();
  if (!allowReturnMemref)
    options.addPostAnalysisStep<scf_ext::AssertDestinationPassingStyle>();

  options.allowReturnMemref = allowReturnMemref;
  options.allowUnknownOps = allowUnknownOps;
  options.testAnalysisOnly = testAnalysisOnly;
  options.analysisFuzzerSeed = analysisFuzzerSeed;

  if (dialectFilter.hasValue()) {
    options.dialectFilter.emplace();
    for (const std::string &dialectNamespace : dialectFilter)
      options.dialectFilter->insert(dialectNamespace);
  }

  Operation *op = getFunction().getOperation();
  if (failed(runComprehensiveBufferize(op, options)))
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
