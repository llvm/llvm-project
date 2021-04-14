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

#include "mlir/Reducer/OptReductionPass.h"

#define DEBUG_TYPE "mlir-reduce"

using namespace mlir;

OptReductionPass::OptReductionPass(const Tester &test, MLIRContext *context,
                                   std::unique_ptr<Pass> optPass)
    : context(context), test(test), optPass(std::move(optPass)) {}

OptReductionPass::OptReductionPass(const OptReductionPass &srcPass)
    : OptReductionBase<OptReductionPass>(srcPass), test(srcPass.test),
      optPass(srcPass.optPass.get()) {}

/// Runs the pass instance in the pass pipeline.
void OptReductionPass::runOnOperation() {
  LLVM_DEBUG(llvm::dbgs() << "\nOptimization Reduction pass: ");
  LLVM_DEBUG(llvm::dbgs() << optPass.get()->getName() << "\nTesting:\n");

  ModuleOp module = this->getOperation();
  ModuleOp moduleVariant = module.clone();
  PassManager pmTransform(context);
  pmTransform.addPass(std::move(optPass));

  std::pair<Tester::Interestingness, int> original = test.isInteresting(module);

  if (failed(pmTransform.run(moduleVariant)))
    return;

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
