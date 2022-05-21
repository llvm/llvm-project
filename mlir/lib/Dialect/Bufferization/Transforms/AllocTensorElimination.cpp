//===- AllocTensorElimination.cpp - alloc_tensor op elimination -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/AllocTensorElimination.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::bufferization;

/// Return true if all `neededValues` are in scope at the given
/// `insertionPoint`.
static bool
neededValuesDominateInsertionPoint(const DominanceInfo &domInfo,
                                   Operation *insertionPoint,
                                   const SmallVector<Value> &neededValues) {
  for (Value val : neededValues) {
    if (auto bbArg = val.dyn_cast<BlockArgument>()) {
      Block *owner = bbArg.getOwner();
      if (!owner->findAncestorOpInBlock(*insertionPoint))
        return false;
    } else {
      auto opResult = val.cast<OpResult>();
      if (!domInfo.dominates(opResult.getOwner(), insertionPoint))
        return false;
    }
  }
  return true;
}

/// Return true if the given `insertionPoint` dominates all uses of
/// `allocTensorOp`.
static bool insertionPointDominatesUses(const DominanceInfo &domInfo,
                                        Operation *insertionPoint,
                                        Operation *allocTensorOp) {
  for (Operation *user : allocTensorOp->getUsers())
    if (!domInfo.dominates(insertionPoint, user))
      return false;
  return true;
}

/// Find a valid insertion point for a replacement of `allocTensorOp`, assuming
/// that the replacement may use any value from `neededValues`.
static Operation *
findValidInsertionPoint(Operation *allocTensorOp,
                        const SmallVector<Value> &neededValues) {
  DominanceInfo domInfo;

  // Gather all possible insertion points: the location of `allocTensorOp` and
  // right after the definition of each value in `neededValues`.
  SmallVector<Operation *> insertionPointCandidates;
  insertionPointCandidates.push_back(allocTensorOp);
  for (Value val : neededValues) {
    // Note: The anchor op is using all of `neededValues`, so:
    // * in case of a block argument: There must be at least one op in the block
    //                                (the anchor op or one of its parents).
    // * in case of an OpResult: There must be at least one op right after the
    //                           defining op (the anchor op or one of its
    //                           parents).
    if (auto bbArg = val.dyn_cast<BlockArgument>()) {
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
    if (!insertionPointDominatesUses(domInfo, insertionPoint, allocTensorOp))
      continue;
    return insertionPoint;
  }

  // No suitable insertion point was found.
  return nullptr;
}

/// Try to eliminate AllocTensorOps inside `op`. An AllocTensorOp is replaced
/// with the result of `rewriteFunc` if it is anchored on a matching
/// OpOperand. "Anchored" means that there is a path on the reverse SSA use-def
/// chain, starting from the OpOperand and always following the aliasing
/// OpOperand, that eventually ends at a single AllocTensorOp.
LogicalResult mlir::bufferization::eliminateAllocTensors(
    RewriterBase &rewriter, Operation *op, AnalysisState &state,
    AnchorMatchFn anchorMatchFunc, RewriteFn rewriteFunc) {
  OpBuilder::InsertionGuard g(rewriter);

  WalkResult status = op->walk([&](Operation *op) {
    for (OpOperand &operand : op->getOpOperands()) {
      // Skip operands that do not bufferize inplace.
      if (!state.isInPlace(operand))
        continue;
      // All values that are needed to create the replacement op.
      SmallVector<Value> neededValues;
      // Is this a matching OpOperand?
      if (!anchorMatchFunc(operand, neededValues))
        continue;
      SetVector<Value> maybeAllocTensor =
          state.findValueInReverseUseDefChain(operand.get(), [&](Value val) {
            // Continue traversal until this function returns true.
            OpResult opResult = val.dyn_cast<OpResult>();
            if (!opResult)
              return true;
            SmallVector<OpOperand *> opOperands =
                state.getAliasingOpOperand(opResult);
            if (!llvm::all_of(opOperands, [&](OpOperand *operand) {
                  return state.isInPlace(*operand);
                }))
              return true;
            // Only equivalent tensors are supported at the moment.
            // TODO: Support cases such as extract_slice(alloc_tensor)
            return !llvm::all_of(opOperands, [&](OpOperand *operand) {
              return state.areEquivalentBufferizedValues(operand->get(),
                                                         opResult);
            });
          });

      // Replace only if the reverse use-def chain ends at exactly one
      // AllocTensorOp.
      if (maybeAllocTensor.size() != 1 ||
          !maybeAllocTensor.front().getDefiningOp<AllocTensorOp>())
        return WalkResult::skip();
      Value allocTensor = maybeAllocTensor.front();

      // Find a suitable insertion point.
      Operation *insertionPoint =
          findValidInsertionPoint(allocTensor.getDefiningOp(), neededValues);
      if (!insertionPoint)
        continue;

      // Create a replacement for the AllocTensorOp.
      rewriter.setInsertionPoint(insertionPoint);
      Value replacement = rewriteFunc(rewriter, allocTensor.getLoc(), operand);
      if (!replacement)
        continue;

      // Replace the AllocTensorOp.
      rewriter.replaceOp(allocTensor.getDefiningOp(), replacement);
    }

    // Advance to the next operation.
    return WalkResult::advance();
  });

  return failure(status.wasInterrupted());
}

/// Try to eliminate AllocTensorOps inside `op`. An AllocTensorOp can be
/// eliminated if it is eventually inserted into another tensor (and some other
/// conditions are met).
///
/// E.g.:
/// %0 = linalg.alloc_tensor
/// %1 = linalg.fill(%cst, %0) {inplace = [true]}
/// %2 = tensor.insert_slice %1 into %t[10][20][1]
///
/// AllocTensorOp elimination will try to fill %t inplace instead of filling a
/// new allocation %0 and inserting it into %t. This is done by replacing the
/// AllocTensorOp with:
///
/// %0 = tensor.extract_slice %t[10][20][1]
///
/// The analysis looks for matching ExtractSliceOp/InsertSliceOp pairs and lets
/// those bufferize inplace in the absence of other conflicts.
///
/// Starting from an InsertSliceOp, an AllocTensorOp at the end of the insert
/// source's reverse use-def chain is eliminated if:
/// * On the reverse use-def chain path from the InsertSliceOp to the
///   AllocTensorOp, all ops were decided to bufferize inplace and the buffer
///   relation is "equivalent" (TODO: can be relaxed if needed).
/// * The reverse use-def chain has exactly one end, which is the AllocTensorOp.
LogicalResult
mlir::bufferization::insertSliceAnchoredAllocTensorEliminationStep(
    RewriterBase &rewriter, Operation *op, AnalysisState &state) {
  return eliminateAllocTensors(
      rewriter, op, state,
      /*anchorMatchFunc=*/
      [&](OpOperand &operand, SmallVector<Value> &neededValues) {
        auto insertSliceOp =
            dyn_cast<tensor::InsertSliceOp>(operand.getOwner());
        if (!insertSliceOp)
          return false;
        if (&operand != &insertSliceOp->getOpOperand(0) /*source*/)
          return false;

        // Collect all values that are needed to construct the replacement op.
        neededValues.append(insertSliceOp.offsets().begin(),
                            insertSliceOp.offsets().end());
        neededValues.append(insertSliceOp.sizes().begin(),
                            insertSliceOp.sizes().end());
        neededValues.append(insertSliceOp.strides().begin(),
                            insertSliceOp.strides().end());
        neededValues.push_back(insertSliceOp.dest());

        return true;
      },
      /*rewriteFunc=*/
      [](OpBuilder &b, Location loc, OpOperand &operand) {
        auto insertOp = cast<tensor::InsertSliceOp>(operand.getOwner());
        // Expand offsets, sizes and strides to the full rank to handle the
        // rank-reducing case.
        SmallVector<OpFoldResult> mixedOffsets = insertOp.getMixedOffsets();
        SmallVector<OpFoldResult> mixedSizes = insertOp.getMixedSizes();
        SmallVector<OpFoldResult> mixedStrides = insertOp.getMixedStrides();
        OffsetSizeAndStrideOpInterface::expandToRank(
            insertOp.dest(), mixedOffsets, mixedSizes, mixedStrides,
            [&](Value target, int64_t dim) -> OpFoldResult {
              auto shapedType = target.getType().cast<ShapedType>();
              if (shapedType.isDynamicDim(dim))
                return b.create<tensor::DimOp>(loc, target, dim).result();
              return b.getIndexAttr(shapedType.getDimSize(dim));
            });
        auto t = tensor::ExtractSliceOp::inferRankReducedResultType(
            insertOp.getSourceType().getRank(),
            insertOp.dest().getType().cast<RankedTensorType>(), mixedOffsets,
            mixedSizes, mixedStrides);
        auto extractOp = b.create<tensor::ExtractSliceOp>(
            loc, t, insertOp.dest(), mixedOffsets, mixedSizes, mixedStrides);
        return extractOp.result();
      });
}

namespace {
struct AllocTensorElimination
    : public AllocTensorEliminationBase<AllocTensorElimination> {
  AllocTensorElimination() = default;

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<bufferization::BufferizationDialect, tensor::TensorDialect>();
  }
};
} // namespace

void AllocTensorElimination::runOnOperation() {
  Operation *op = getOperation();
  OneShotBufferizationOptions options;
  OneShotAnalysisState state(op, options);
  if (failed(analyzeOp(op, state))) {
    signalPassFailure();
    return;
  }

  IRRewriter rewriter(op->getContext());
  if (failed(bufferization::insertSliceAnchoredAllocTensorEliminationStep(
          rewriter, op, state)))
    signalPassFailure();
}

std::unique_ptr<Pass> mlir::bufferization::createAllocTensorEliminationPass() {
  return std::make_unique<AllocTensorElimination>();
}
