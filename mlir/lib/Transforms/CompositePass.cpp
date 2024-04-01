//===- CompositePass.cpp - Composite pass code ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// CompositePass allows to run set of passes until fixed point is reached.
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/Passes.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace mlir {
#define GEN_PASS_DEF_COMPOSITEFIXEDPOINTPASS
#include "mlir/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {
struct CompositeFixedPointPass final
    : public impl::CompositeFixedPointPassBase<CompositeFixedPointPass> {
  using CompositeFixedPointPassBase::CompositeFixedPointPassBase;

  CompositeFixedPointPass(
      std::string name_, llvm::function_ref<void(OpPassManager &)> populateFunc,
      unsigned maxIterations) {
    name = std::move(name_);
    maxIter = maxIterations;
    populateFunc(dynamicPM);
    std::string pipeline;
    llvm::raw_string_ostream os(pipeline);
    dynamicPM.printAsTextualPipeline(os);
    os.flush();
    pipelineStr = pipeline;
  }

  LogicalResult initializeOptions(StringRef options) override {
    if (failed(CompositeFixedPointPassBase::initializeOptions(options)))
      return failure();

    if (failed(parsePassPipeline(pipelineStr, dynamicPM))) {
      llvm::errs() << "Failed to parse composite pass pipeline\n";
      return failure();
    }

    return success();
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    dynamicPM.getDependentDialects(registry);
  }

  void runOnOperation() override {
    auto op = getOperation();
    OperationFingerPrint fp(op);

    unsigned currentIter = 0;
    unsigned maxIterVal = maxIter;
    while (true) {
      if (failed(runPipeline(dynamicPM, op)))
        return signalPassFailure();

      if (currentIter++ >= maxIterVal) {
        op->emitWarning("Composite pass \"" + llvm::Twine(name) +
                        "\"+ didn't converge in " + llvm::Twine(maxIterVal) +
                        " iterations");
        break;
      }

      OperationFingerPrint newFp(op);
      if (newFp == fp)
        break;

      fp = newFp;
    }
  }

protected:
  llvm::StringRef getName() const override { return name; }

private:
  OpPassManager dynamicPM;
};
} // namespace

std::unique_ptr<Pass> mlir::createCompositeFixedPointPass(
    std::string name, llvm::function_ref<void(OpPassManager &)> populateFunc,
    unsigned maxIterations) {

  return std::make_unique<CompositeFixedPointPass>(std::move(name),
                                                   populateFunc, maxIterations);
}
