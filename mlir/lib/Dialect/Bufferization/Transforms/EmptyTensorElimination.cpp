//===- EmptyTensorElimination.cpp - tensor.empty op elimination -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Bufferization/Transforms/Passes.h"

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotModuleBufferize.h"
#include "mlir/Dialect/Bufferization/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/SubsetOpInterface.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace bufferization {
#define GEN_PASS_DEF_EMPTYTENSORELIMINATION
#include "mlir/Dialect/Bufferization/Transforms/Passes.h.inc"
} // namespace bufferization
} // namespace mlir

using namespace mlir;
using namespace mlir::bufferization;

/// Return true if all `neededValues` are in scope at the given
/// `insertionPoint`.
static bool
neededValuesDominateInsertionPoint(const DominanceInfo &domInfo,
                                   Operation *insertionPoint,
                                   const SmallVector<Value> &neededValues) {
  for (Value val : neededValues) {
    if (auto bbArg = dyn_cast<BlockArgument>(val)) {
      Block *owner = bbArg.getOwner();
      if (!owner->findAncestorOpInBlock(*insertionPoint))
        return false;
    } else {
      auto opResult = cast<OpResult>(val);
      if (!domInfo.properlyDominates(opResult.getOwner(), insertionPoint))
        return false;
    }
  }
  return true;
}

/// Return true if the given `insertionPoint` dominates all uses of
/// `emptyTensorOp`.
static bool insertionPointDominatesUses(const DominanceInfo &domInfo,
                                        Operation *insertionPoint,
                                        Operation *emptyTensorOp) {
  return llvm::all_of(emptyTensorOp->getUsers(), [&](Operation *user) {
    return domInfo.dominates(insertionPoint, user);
  });
}

/// Find a valid insertion point for a replacement of `emptyTensorOp`, assuming
/// that the replacement may use any value from `neededValues`.
static Operation *
findValidInsertionPoint(Operation *emptyTensorOp,
                        const SmallVector<Value> &neededValues) {
  DominanceInfo domInfo;

  // Gather all possible insertion points: the location of `emptyTensorOp` and
  // right after the definition of each value in `neededValues`.
  SmallVector<Operation *> insertionPointCandidates;
  insertionPointCandidates.push_back(emptyTensorOp);
  for (Value val : neededValues) {
    // Note: The anchor op is using all of `neededValues`, so:
    // * in case of a block argument: There must be at least one op in the block
    //                                (the anchor op or one of its parents).
    // * in case of an OpResult: There must be at least one op right after the
    //                           defining op (the anchor op or one of its
    //                           parents).
    if (auto bbArg = dyn_cast<BlockArgument>(val)) {
      insertionPointCandidates.push_back(
          &bbArg.getOwner()->getOperations().front());
    } else {
      insertionPointCandidates.push_back(val.getDefiningOp()->getNextNode());
    }
  }

  // Select first matching insertion point.
  for (Operation *insertionPoint : insertionPointCandidates) {
    // Check if all needed values are in scope.
    if (!neededValuesDominateInsertionPoint(domInfo, insertionPoint,
                                            neededValues))
      continue;
    // Check if the insertion point is before all uses.
    if (!insertionPointDominatesUses(domInfo, insertionPoint, emptyTensorOp))
      continue;
    return insertionPoint;
  }

  // No suitable insertion point was found.
  return nullptr;
}

LogicalResult mlir::bufferization::eliminateEmptyTensors(
    RewriterBase &rewriter, Operation *op, OneShotAnalysisState &state) {
  OpBuilder::InsertionGuard g(rewriter);

  op->walk([&](SubsetInsertionOpInterface op) {
    OpOperand &source = op.getSourceOperand();
    // Skip operands that do not bufferize inplace. "tensor.empty" could still
    // be replaced, but the transformation may not be beneficial.
    if (!state.isInPlace(source))
      return WalkResult::skip();

    // All values that are needed to create the replacement op.
    SmallVector<Value> neededValues =
        op.getValuesNeededToBuildSubsetExtraction();

    // Find tensor.empty ops on the reverse SSA use-def chain. Only follow
    // equivalent tensors. I.e., stop when there are ops such as extract_slice
    // on the path.
    TraversalConfig config;
    config.followEquivalentOnly = true;
    config.alwaysIncludeLeaves = false;
    // Replace only if the types match or are static <-> dynamic casts. We do
    // not support slices or reshapes.
    // TODO: This could be extended to support IR such as:
    // %0 = tensor.empty() : tensor<128xf32>
    // %1 = "some_op"(%0) : (tensor<128xf32>) -> (tensor<128xf32>)
    // %2 = tensor.expand_shape %1 ...
    // %3 = tensor.insert_slice %2 into ...
    config.followSameTypeOrCastsOnly = true;
    SetVector<Value> emptyTensors = state.findValueInReverseUseDefChain(
        source.get(), /*condition=*/
        [&](Value val) { return val.getDefiningOp<tensor::EmptyOp>(); },
        config);

    for (Value v : emptyTensors) {
      Operation *emptyTensorOp = v.getDefiningOp();

      // Find a suitable insertion point. If no suitable insertion point for
      // the replacement can be found, skip this replacement.
      Operation *insertionPoint =
          findValidInsertionPoint(emptyTensorOp, neededValues);
      if (!insertionPoint)
        continue;

      rewriter.setInsertionPoint(insertionPoint);
      Value replacement =
          op.buildSubsetExtraction(rewriter, emptyTensorOp->getLoc());
      if (!replacement)
        continue;
      if (emptyTensorOp == replacement.getDefiningOp())
        continue;
      if (replacement.getType() != v.getType()) {
        rewriter.setInsertionPointAfterValue(replacement);
        replacement = rewriter.create<tensor::CastOp>(v.getLoc(), v.getType(),
                                                      replacement);
      }
      // Replace the tensor::EmptyOp.
      rewriter.replaceOp(emptyTensorOp, replacement);
      state.resetCache();
    }

    return WalkResult::advance();
  });

  return success();
}

namespace {
struct EmptyTensorElimination
    : public bufferization::impl::EmptyTensorEliminationBase<
          EmptyTensorElimination> {
  EmptyTensorElimination() = default;

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<bufferization::BufferizationDialect, tensor::TensorDialect>();
  }
};
} // namespace

LogicalResult mlir::bufferization::eliminateEmptyTensors(RewriterBase &rewriter,
                                                         Operation *op) {
  auto moduleOp = dyn_cast<ModuleOp>(op);
  OneShotBufferizationOptions options;
  options.allowReturnAllocsFromLoops = true;
  if (moduleOp)
    options.bufferizeFunctionBoundaries = true;
  OneShotAnalysisState state(op, options);
  if (moduleOp) {
    // Module analysis takes into account function boundaries.
    if (failed(analyzeModuleOp(moduleOp, state)))
      return failure();
  } else {
    // Regular One-Shot Bufferize ignores func.func block arguments, func.call,
    // func.return.
    if (failed(analyzeOp(op, state)))
      return failure();
  }

  return bufferization::eliminateEmptyTensors(rewriter, op, state);
}

void EmptyTensorElimination::runOnOperation() {
  IRRewriter rewriter(getOperation()->getContext());
  if (failed(bufferization::eliminateEmptyTensors(rewriter, getOperation())))
    signalPassFailure();
}

std::unique_ptr<Pass> mlir::bufferization::createEmptyTensorEliminationPass() {
  return std::make_unique<EmptyTensorElimination>();
}
