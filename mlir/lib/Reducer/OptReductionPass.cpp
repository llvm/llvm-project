//===- OptReductionPass.cpp - Optimization Reduction Pass Wrapper ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the Opt Reduction Pass Wrapper. It creates a MLIR pass to
// run any optimization pass within it and only replaces the output module with
// the transformed version if it is smaller and interesting.
//
//===----------------------------------------------------------------------===//

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Reducer/Passes.h"
#include "mlir/Reducer/Tester.h"
#include "llvm/Support/Debug.h"

namespace mlir {
#define GEN_PASS_DEF_OPTREDUCTIONPASS
#include "mlir/Reducer/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "mlir-reduce"

using namespace mlir;

namespace {

class OptReductionPass
    : public mlir::impl::OptReductionPassBase<OptReductionPass> {
public:
  /// Runs the pass instance in the pass pipeline.
  void runOnOperation() override;
};

} // namespace

/// Runs the pass instance in the pass pipeline.
void OptReductionPass::runOnOperation() {
  LLVM_DEBUG(llvm::dbgs() << "\nOptimization Reduction pass: ");

  Tester test(testerName, testerArgs);

  ModuleOp module = this->getOperation();
  ModuleOp moduleVariant = module.clone();

  OpPassManager passManager("builtin.module");
  if (failed(parsePassPipeline(optPass, passManager))) {
    module.emitError() << "\nfailed to parse pass pipeline";
    return signalPassFailure();
  }

  std::pair<Tester::Interestingness, int> original = test.isInteresting(module);
  if (original.first != Tester::Interestingness::True) {
    module.emitError() << "\nthe original input is not interested";
    return signalPassFailure();
  }

  // Temporarily push the variant under the main module and execute the pipeline
  // on it.
  module.getBody()->push_back(moduleVariant);
  LogicalResult pipelineResult = runPipeline(passManager, moduleVariant);
  moduleVariant->remove();

  if (failed(pipelineResult)) {
    module.emitError() << "\nfailed to run pass pipeline";
    return signalPassFailure();
  }

  std::pair<Tester::Interestingness, int> reduced =
      test.isInteresting(moduleVariant);

  if (reduced.first == Tester::Interestingness::True &&
      reduced.second < original.second) {
    module.getBody()->clear();
    module.getBody()->getOperations().splice(
        module.getBody()->begin(), moduleVariant.getBody()->getOperations());
    LLVM_DEBUG(llvm::dbgs() << "\nSuccessful Transformed version\n\n");
  } else {
    LLVM_DEBUG(llvm::dbgs() << "\nUnsuccessful Transformed version\n\n");
  }

  moduleVariant->destroy();

  LLVM_DEBUG(llvm::dbgs() << "Pass Complete\n\n");
}

std::unique_ptr<Pass> mlir::createOptReductionPass() {
  return std::make_unique<OptReductionPass>();
}
