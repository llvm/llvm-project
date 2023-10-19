//===- TensorCopyInsertion.cpp - Resolve Bufferization Conflicts w/ Copies ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Bufferization/Transforms/Passes.h"

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotModuleBufferize.h"
#include "mlir/Dialect/Bufferization/Transforms/Transforms.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir {
namespace bufferization {
#define GEN_PASS_DEF_TENSORCOPYINSERTION
#include "mlir/Dialect/Bufferization/Transforms/Passes.h.inc"
} // namespace bufferization
} // namespace mlir

using namespace mlir;
using namespace mlir::bufferization;

LogicalResult mlir::bufferization::insertTensorCopies(
    Operation *op, const OneShotBufferizationOptions &options,
    BufferizationStatistics *statistics) {
  OneShotAnalysisState state(op, options);
  // Run normal One-Shot Bufferize analysis or One-Shot Module Bufferize
  // analysis depending on whether function boundary bufferization is enabled or
  // not.
  if (options.bufferizeFunctionBoundaries) {
    if (failed(analyzeModuleOp(cast<ModuleOp>(op), state, statistics)))
      return failure();
  } else {
    if (failed(analyzeOp(op, state, statistics)))
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

    // Find inplacability conflicts and resolve them. (Typically with explicit
    // tensor copies in the form of AllocTensorOps.)
    rewriter.setInsertionPoint(op);
    if (failed(bufferizableOp.resolveConflicts(rewriter, state)))
      return WalkResult::interrupt();

    return WalkResult::advance();
  });

  return failure(result.wasInterrupted());
}
