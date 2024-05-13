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

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

using namespace mlir;

namespace {

class TestDynamicPipelinePass
    : public PassWrapper<TestDynamicPipelinePass, OperationPass<>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestDynamicPipelinePass)

  StringRef getArgument() const final { return "test-dynamic-pipeline"; }
  StringRef getDescription() const final {
    return "Tests the dynamic pipeline feature by applying "
           "a pipeline on a selected set of functions";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    OpPassManager pm(ModuleOp::getOperationName(),
                     OpPassManager::Nesting::Implicit);
    (void)parsePassPipeline(pipeline, pm, llvm::errs());
    pm.getDependentDialects(registry);
  }

  TestDynamicPipelinePass() = default;
  TestDynamicPipelinePass(const TestDynamicPipelinePass &) {}

  void runOnOperation() override {
    Operation *currentOp = getOperation();

    llvm::errs() << "Dynamic execute '" << pipeline << "' on "
                 << currentOp->getName() << "\n";
    if (pipeline.empty()) {
      llvm::errs() << "Empty pipeline\n";
      return;
    }
    auto symbolOp = dyn_cast<SymbolOpInterface>(currentOp);
    if (!symbolOp) {
      currentOp->emitWarning()
          << "Ignoring because not implementing SymbolOpInterface\n";
      return;
    }

    auto opName = symbolOp.getName();
    if (!opNames.empty() && !llvm::is_contained(opNames, opName)) {
      llvm::errs() << "dynamic-pipeline skip op name: " << opName << "\n";
      return;
    }
    OpPassManager pm(currentOp->getName().getIdentifier(),
                     OpPassManager::Nesting::Implicit);
    (void)parsePassPipeline(pipeline, pm, llvm::errs());

    // Check that running on the parent operation always immediately fails.
    if (runOnParent) {
      if (currentOp->getParentOp())
        if (!failed(runPipeline(pm, currentOp->getParentOp())))
          signalPassFailure();
      return;
    }

    if (runOnNestedOp) {
      llvm::errs() << "Run on nested op\n";
      currentOp->walk([&](Operation *op) {
        if (op == currentOp || !op->hasTrait<OpTrait::IsIsolatedFromAbove>() ||
            op->getName() != currentOp->getName())
          return;
        llvm::errs() << "Run on " << *op << "\n";
        // Run on the current operation
        if (failed(runPipeline(pm, op)))
          signalPassFailure();
      });
    } else {
      // Run on the current operation
      if (failed(runPipeline(pm, currentOp)))
        signalPassFailure();
    }
  }

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
      *this, "op-name",
      llvm::cl::desc("List of function name to apply the pipeline to")};
};
} // namespace

namespace mlir {
namespace test {
void registerTestDynamicPipelinePass() {
  PassRegistration<TestDynamicPipelinePass>();
}
} // namespace test
} // namespace mlir
