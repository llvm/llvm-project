//===- TensorCopyInsertion.cpp - Resolve Bufferization Conflicts w/ Copies ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Bufferization/Transforms/Passes.h"

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotModuleBufferize.h"
#include "mlir/Dialect/Bufferization/Transforms/Transforms.h"

using namespace mlir;
using namespace mlir::bufferization;

LogicalResult mlir::bufferization::insertTensorCopies(
    Operation *op, const OneShotBufferizationOptions &options,
    const BufferizationState &bufferizationState,
    BufferizationStatistics *statistics) {
  OneShotAnalysisState analysisState(op, options);
  // Run normal One-Shot Bufferize analysis or One-Shot Module Bufferize
  // analysis depending on whether function boundary bufferization is enabled or
  // not.
  if (options.bufferizeFunctionBoundaries) {
    if (failed(analyzeModuleOp(op, analysisState, statistics)))
      return failure();
  } else {
    if (failed(analyzeOp(op, analysisState, statistics)))
      return failure();
  }

  if (options.testAnalysisOnly)
    return success();

  return insertTensorCopies(op, analysisState, bufferizationState);
}

LogicalResult mlir::bufferization::insertTensorCopies(
    Operation *op, const AnalysisState &analysisState,
    const BufferizationState &bufferizationState) {
  IRRewriter rewriter(op->getContext());

  // It may be more efficient to walk in pre-order here, but the current
  // implementation visits regions of ops even if they are not allowed or
  // bufferizable, and existing tests rely on this behavior.
  // For now, only exclude nested operations if they are in a different symbol
  // table scope.
  WalkResult result = op->walk([&](Operation *nestedOp) {
    if (op->hasTrait<OpTrait::SymbolTable>() &&
        nestedOp->getParentWithTrait<OpTrait::SymbolTable>() != op)
      return WalkResult::skip();

    auto bufferizableOp =
        analysisState.getOptions().dynCastBufferizableOp(nestedOp);
    if (!bufferizableOp)
      return WalkResult::skip();

    // Find inplacability conflicts and resolve them. (Typically with explicit
    // tensor copies in the form of AllocTensorOps.)
    rewriter.setInsertionPoint(nestedOp);
    if (failed(bufferizableOp.resolveConflicts(rewriter, analysisState,
                                               bufferizationState)))
      return WalkResult::interrupt();

    return WalkResult::advance();
  });

  return failure(result.wasInterrupted());
}
