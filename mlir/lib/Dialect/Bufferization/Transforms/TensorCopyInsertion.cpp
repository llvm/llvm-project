//===- TensorCopyInsertion.cpp - Resolve Bufferization Conflicts w/ Copies ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Bufferization/Transforms/TensorCopyInsertion.h"

#include "PassDetail.h"

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotModuleBufferize.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::bufferization;

LogicalResult mlir::bufferization::insertTensorCopies(
    Operation *op, const OneShotBufferizationOptions &options) {
  OneShotAnalysisState state(op, options);
  // Run normal One-Shot Bufferize analysis or One-Shot Module Bufferize
  // analysis depending on whether function boundary bufferization is enabled or
  // not.
  if (options.bufferizeFunctionBoundaries) {
    if (failed(analyzeModuleOp(cast<ModuleOp>(op), state)))
      return failure();
  } else {
    if (failed(analyzeOp(op, state)))
      return failure();
  }

  if (options.testAnalysisOnly)
    return success();

  return insertTensorCopies(op, state);
}

LogicalResult
mlir::bufferization::insertTensorCopies(Operation *op,
                                        const AnalysisState &state) {
  IRRewriter rewriter(op->getContext());
  WalkResult result = op->walk([&](Operation *op) {
    auto bufferizableOp = state.getOptions().dynCastBufferizableOp(op);
    if (!bufferizableOp)
      return WalkResult::skip();

    // Find AllocTensorOps without an `escape` attribute and add the attribute
    // based on analysis results.
    if (auto allocTensorOp = dyn_cast<AllocTensorOp>(op)) {
      if (allocTensorOp.escape())
        return WalkResult::advance();
      bool escape = !state.getOptions().createDeallocs ||
                    state.isTensorYielded(allocTensorOp.result());
      allocTensorOp.escapeAttr(rewriter.getBoolAttr(escape));
      return WalkResult::advance();
    }

    // Find inplacability conflicts and resolve them. (Typically with explicit
    // tensor copies in the form of AllocTensorOps.)
    rewriter.setInsertionPoint(op);
    if (failed(bufferizableOp.resolveConflicts(rewriter, state)))
      return WalkResult::interrupt();

    return WalkResult::advance();
  });

  return failure(result.wasInterrupted());
}

namespace {
struct TensorCopyInsertionPass
    : TensorCopyInsertionBase<TensorCopyInsertionPass> {
  TensorCopyInsertionPass()
      : TensorCopyInsertionBase<TensorCopyInsertionPass>(),
        options(llvm::None) {}
  TensorCopyInsertionPass(const OneShotBufferizationOptions &options)
      : TensorCopyInsertionBase<TensorCopyInsertionPass>(), options(options) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<bufferization::BufferizationDialect>();
  }

  void runOnOperation() override {
    if (options.hasValue()) {
      if (failed(insertTensorCopies(getOperation(), *options)))
        signalPassFailure();
    } else {
      OneShotBufferizationOptions options;
      options.allowReturnAllocs = allowReturnAllocs;
      options.bufferizeFunctionBoundaries = bufferizeFunctionBoundaries;
      options.createDeallocs = createDeallocs;
      if (failed(insertTensorCopies(getOperation(), options)))
        signalPassFailure();
    }
  }

private:
  Optional<OneShotBufferizationOptions> options;
};
} // namespace

std::unique_ptr<Pass> mlir::bufferization::createTensorCopyInsertionPass() {
  return std::make_unique<TensorCopyInsertionPass>();
}

std::unique_ptr<Pass> mlir::bufferization::createTensorCopyInsertionPass(
    const OneShotBufferizationOptions &options) {
  return std::make_unique<TensorCopyInsertionPass>(options);
}
