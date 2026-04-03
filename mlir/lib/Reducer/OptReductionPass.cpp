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

#include "llvm/Support/DebugLog.h"

namespace mlir {
#define GEN_PASS_DEF_OPTREDUCTIONPASS
#include "mlir/Reducer/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "mlir-reduce"

using namespace mlir;

namespace {

class OptReductionPass : public impl::OptReductionPassBase<OptReductionPass> {
public:
  using Base::Base;

  /// Runs the pass instance in the pass pipeline.
  void runOnOperation() override;
};

} // namespace

/// Runs the pass instance in the pass pipeline.
void OptReductionPass::runOnOperation() {
  LDBG() << "\nOptimization Reduction pass: ";

  Tester test(testerName, testerArgs);

  Operation *topOp = this->getOperation();
  Operation *topOpVariant = topOp->clone();

  PassManager passManager(topOp->getName());
  if (failed(parsePassPipeline(optPass, passManager))) {
    topOp->emitError() << "\nfailed to parse pass pipeline";
    return signalPassFailure();
  }

  std::pair<Tester::Interestingness, int> original = test.isInteresting(topOp);
  if (original.first != Tester::Interestingness::True) {
    topOp->emitError() << "\nthe original input is not interested";
    return signalPassFailure();
  }

  LogicalResult pipelineResult = passManager.run(topOpVariant);
  if (failed(pipelineResult)) {
    topOp->emitError() << "\nfailed to run pass pipeline";
    return signalPassFailure();
  }

  std::pair<Tester::Interestingness, int> reduced =
      test.isInteresting(topOpVariant);

  if (reduced.first == Tester::Interestingness::True &&
      reduced.second < original.second) {
    topOp->getRegion(0).getBlocks().clear();
    topOp->getRegion(0).getBlocks().splice(
        topOp->getRegion(0).getBlocks().begin(),
        topOpVariant->getRegion(0).getBlocks());

    LDBG() << "\nSuccessful Transformed version\n";
  } else {
    LDBG() << "\nUnsuccessful Transformed version\n";
  }

  topOpVariant->destroy();

  LDBG() << "Pass Complete\n";
}
