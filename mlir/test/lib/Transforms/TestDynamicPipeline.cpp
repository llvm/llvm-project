//===------ TestDynamicPipeline.cpp --- dynamic pipeline test pass --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to test the dynamic pipeline feature.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/LoopUtils.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace {

class TestDynamicPipelinePass
    : public PassWrapper<TestDynamicPipelinePass, OperationPass<>> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    OpPassManager pm(ModuleOp::getOperationName(), false);
    parsePassPipeline(pipeline, pm, llvm::errs());
    pm.getDependentDialects(registry);
  }

  TestDynamicPipelinePass(){};
  TestDynamicPipelinePass(const TestDynamicPipelinePass &) {}

  void runOnOperation() override {
    llvm::errs() << "Dynamic execute '" << pipeline << "' on "
                 << getOperation()->getName() << "\n";
    if (pipeline.empty()) {
      llvm::errs() << "Empty pipeline\n";
      return;
    }
    auto symbolOp = dyn_cast<SymbolOpInterface>(getOperation());
    if (!symbolOp) {
      getOperation()->emitWarning()
          << "Ignoring because not implementing SymbolOpInterface\n";
      return;
    }

    auto opName = symbolOp.getName();
    if (!opNames.empty() && !llvm::is_contained(opNames, opName)) {
      llvm::errs() << "dynamic-pipeline skip op name: " << opName << "\n";
      return;
    }
    if (!pm) {
      pm = std::make_unique<OpPassManager>(
          getOperation()->getName().getIdentifier(), false);
      parsePassPipeline(pipeline, *pm, llvm::errs());
    }

    // Check that running on the parent operation always immediately fails.
    if (runOnParent) {
      if (getOperation()->getParentOp())
        if (!failed(runPipeline(*pm, getOperation()->getParentOp())))
          signalPassFailure();
      return;
    }

    if (runOnNestedOp) {
      llvm::errs() << "Run on nested op\n";
      getOperation()->walk([&](Operation *op) {
        if (op == getOperation() || !op->isKnownIsolatedFromAbove())
          return;
        llvm::errs() << "Run on " << *op << "\n";
        // Run on the current operation
        if (failed(runPipeline(*pm, op)))
          signalPassFailure();
      });
    } else {
      // Run on the current operation
      if (failed(runPipeline(*pm, getOperation())))
        signalPassFailure();
    }
  }

  std::unique_ptr<OpPassManager> pm;

  Option<bool> runOnNestedOp{
      *this, "run-on-nested-operations",
      llvm::cl::desc("This will apply the pipeline on nested operations under "
                     "the visited operation.")};
  Option<bool> runOnParent{
      *this, "run-on-parent",
      llvm::cl::desc("This will apply the pipeline on the parent operation if "
                     "it exist, this is expected to fail.")};
  Option<std::string> pipeline{
      *this, "dynamic-pipeline",
      llvm::cl::desc("The pipeline description that "
                     "will run on the filtered function.")};
  ListOption<std::string> opNames{
      *this, "op-name", llvm::cl::MiscFlags::CommaSeparated,
      llvm::cl::desc("List of function name to apply the pipeline to")};
};
} // end namespace

namespace mlir {
void registerTestDynamicPipelinePass() {
  PassRegistration<TestDynamicPipelinePass>(
      "test-dynamic-pipeline", "Tests the dynamic pipeline feature by applying "
                               "a pipeline on a selected set of functions");
}
} // namespace mlir
