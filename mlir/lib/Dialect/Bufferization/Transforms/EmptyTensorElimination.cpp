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
#include "mlir/Dialect/Bufferization/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Dominance.h"
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
  for (Operation *user : emptyTensorOp->getUsers())
    if (!domInfo.dominates(insertionPoint, user))
      return false;
  return true;
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

/// Try to eliminate tensor::EmptyOps inside `op`. A tensor::EmptyOp is replaced
/// with the result of `rewriteFunc` if it is anchored on a matching
/// OpOperand. "Anchored" means that there is a path on the reverse SSA use-def
/// chain, starting from the OpOperand and always following the aliasing
/// OpOperand, that eventually ends at the tensor::EmptyOp.
///
/// E.g.:
/// %0 = tensor.empty() : tensor<10xf32>
/// %1 = linalg.fill ... outs(%0 : tensor<10xf32>)
/// %2 = tensor.insert_slice %0 into %t ...
///
/// In the above example, the anchor is the source operand of the insert_slice
/// op. When tracing back the reverse use-def chain, we end up at a
/// tensor.empty op.
LogicalResult mlir::bufferization::eliminateEmptyTensors(
    RewriterBase &rewriter, Operation *op, OneShotAnalysisState &state,
    AnchorMatchFn anchorMatchFunc, RewriteFn rewriteFunc) {
  OpBuilder::InsertionGuard g(rewriter);

  op->walk([&](Operation *op) {
    for (OpOperand &operand : op->getOpOperands()) {
      // Skip operands that do not bufferize inplace.
      if (!state.isInPlace(operand))
        continue;
      // All values that are needed to create the replacement op.
      SmallVector<Value> neededValues;
      // Is this an anchor?
      if (!anchorMatchFunc(operand, neededValues))
        continue;

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
          operand.get(), /*condition=*/
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
            rewriteFunc(rewriter, emptyTensorOp->getLoc(), operand);
        if (!replacement)
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
    }
  });

  return success();
}

/// Try to eliminate tensor::EmptyOps inside `op`. An tensor::EmptyOp can be
/// eliminated if it is eventually inserted into another tensor (and some other
/// conditions are met).
///
/// E.g.:
/// %0 = tensor.empty()
/// %1 = linalg.fill(%cst, %0) {inplace = [true]}
/// %2 = tensor.insert_slice %1 into %t[10][20][1]
///
/// tensor::EmptyOp elimination will try to fill %t inplace instead of filling a
/// new allocation %0 and inserting it into %t. This is done by replacing the
/// tensor::EmptyOp with:
///
/// %0 = tensor.extract_slice %t[10][20][1]
///
/// The analysis looks for matching ExtractSliceOp/InsertSliceOp pairs and lets
/// those bufferize inplace in the absence of other conflicts.
///
/// Starting from an InsertSliceOp, an tensor::EmptyOp at the end of the insert
/// source's reverse use-def chain is eliminated if:
/// * On the reverse use-def chain path from the InsertSliceOp to the
///   tensor::EmptyOp, all ops were decided to bufferize inplace and the buffer
///   relation is "equivalent" (TODO: can be relaxed if needed).
/// * The reverse use-def chain has exactly one end, which is the
///   tensor::EmptyOp.
template <typename OpTy>
static LogicalResult insertSliceLikeAnchoredEmptyTensorEliminationStep(
    RewriterBase &rewriter, Operation *op, OneShotAnalysisState &state) {
  return eliminateEmptyTensors(
      rewriter, op, state,
      /*anchorMatchFunc=*/
      [&](OpOperand &operand, SmallVector<Value> &neededValues) {
        auto insertSliceOp = dyn_cast<OpTy>(operand.getOwner());
        if (!insertSliceOp)
          return false;
        if (&operand != &insertSliceOp->getOpOperand(0) /*source*/)
          return false;

        // Collect all values that are needed to construct the replacement op.
        neededValues.append(insertSliceOp.getOffsets().begin(),
                            insertSliceOp.getOffsets().end());
        neededValues.append(insertSliceOp.getSizes().begin(),
                            insertSliceOp.getSizes().end());
        neededValues.append(insertSliceOp.getStrides().begin(),
                            insertSliceOp.getStrides().end());
        neededValues.push_back(insertSliceOp.getDest());

        return true;
      },
      /*rewriteFunc=*/
      [](OpBuilder &b, Location loc, OpOperand &operand) {
        auto insertOp = cast<OpTy>(operand.getOwner());
        auto extractOp = b.create<tensor::ExtractSliceOp>(
            loc, insertOp.getSourceType(), insertOp.getDest(),
            insertOp.getMixedOffsets(), insertOp.getMixedSizes(),
            insertOp.getMixedStrides());
        return extractOp.getResult();
      });
}

LogicalResult
mlir::bufferization::insertSliceAnchoredEmptyTensorEliminationStep(
    RewriterBase &rewriter, Operation *op, OneShotAnalysisState &state) {
  if (failed(insertSliceLikeAnchoredEmptyTensorEliminationStep<
             tensor::InsertSliceOp>(rewriter, op, state)))
    return failure();
  if (failed(insertSliceLikeAnchoredEmptyTensorEliminationStep<
             tensor::ParallelInsertSliceOp>(rewriter, op, state)))
    return failure();
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

void EmptyTensorElimination::runOnOperation() {
  Operation *op = getOperation();
  OneShotBufferizationOptions options;
  OneShotAnalysisState state(op, options);
  if (failed(analyzeOp(op, state))) {
    signalPassFailure();
    return;
  }

  IRRewriter rewriter(op->getContext());
  if (failed(bufferization::insertSliceAnchoredEmptyTensorEliminationStep(
          rewriter, op, state)))
    signalPassFailure();
}

std::unique_ptr<Pass> mlir::bufferization::createEmptyTensorEliminationPass() {
  return std::make_unique<EmptyTensorElimination>();
}
