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

using namespace mlir;

namespace {
struct CompositePass final
    : public PassWrapper<CompositePass, OperationPass<void>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CompositePass)

  CompositePass(std::string name_, std::string argument_,
                llvm::function_ref<void(OpPassManager &)> populateFunc,
                unsigned maxIterations)
      : name(std::move(name_)), argument(std::move(argument_)),
        dynamicPM(std::make_shared<OpPassManager>()), maxIters(maxIterations) {
    populateFunc(*dynamicPM);
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    dynamicPM->getDependentDialects(registry);
  }

  void runOnOperation() override {
    auto op = getOperation();
    OperationFingerPrint fp(op);

    unsigned currentIter = 0;
    while (true) {
      if (failed(runPipeline(*dynamicPM, op)))
        return signalPassFailure();

      if (currentIter++ >= maxIters) {
        op->emitWarning("Composite pass \"" + llvm::Twine(name) +
                        "\"+ didn't converge in " + llvm::Twine(maxIters) +
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

  llvm::StringRef getArgument() const override { return argument; }

private:
  std::string name;
  std::string argument;
  std::shared_ptr<OpPassManager> dynamicPM;
  unsigned maxIters;
};
} // namespace

std::unique_ptr<Pass> mlir::createCompositePass(
    std::string name, std::string argument,
    llvm::function_ref<void(OpPassManager &)> populateFunc,
    unsigned maxIterations) {

  return std::make_unique<CompositePass>(std::move(name), std::move(argument),
                                         populateFunc, maxIterations);
}
