//===- ComprehensiveBufferize.cpp - Single pass bufferization -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::bufferization;
using namespace mlir::linalg;

namespace {
struct LinalgInitTensorElimination
    : public LinalgInitTensorEliminationBase<LinalgInitTensorElimination> {
  LinalgInitTensorElimination() = default;

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, tensor::TensorDialect>();
  }
};
} // namespace

void LinalgInitTensorElimination::runOnOperation() {
  Operation *op = getOperation();
  OneShotBufferizationOptions options;
  OneShotAnalysisState state(op, options);
  if (failed(analyzeOp(op, state))) {
    signalPassFailure();
    return;
  }

  IRRewriter rewriter(op->getContext());
  if (failed(insertSliceAnchoredInitTensorEliminationStep(rewriter, op, state)))
    signalPassFailure();
}

std::unique_ptr<Pass> mlir::createLinalgInitTensorEliminationPass() {
  return std::make_unique<LinalgInitTensorElimination>();
}
