//===- EmptyTensorElimination.cpp - tensor.empty op elimination -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/Transforms/Transforms.h"

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

using namespace mlir;
using namespace mlir::bufferization;
using namespace mlir::linalg;

/// Get an output operand that matches the given input operand and can be used
/// to eliminate a tensor.empty op.
static OpOperand *getUnusedOutOperand(LinalgOp op, OpOperand *in) {
  for (OpOperand &operand : op.getDpsInitsMutable()) {
    // Operand must be unused.
    if (op.payloadUsesValueFromOperand(&operand))
      continue;
    // Types must match.
    if (operand.get().getType() != in->get().getType())
      continue;
    // Indexing maps must match.
    if (op.getMatchingIndexingMap(&operand) != op.getMatchingIndexingMap(in))
      continue;
    return &operand;
  }
  return nullptr;
}

LogicalResult linalg::linalgOpAnchoredEmptyTensorEliminationStep(
    RewriterBase &rewriter, Operation *op, OneShotAnalysisState &state) {
  OpBuilder::InsertionGuard g(rewriter);
  DominanceInfo domInfo;

  op->walk([&](LinalgOp op) {
    // Only ops with all "parallel" iterator types are supported.
    if (op.getNumParallelLoops() != op.getNumLoops())
      return WalkResult::skip();

    for (OpOperand *in : op.getDpsInputOperands()) {
      // Skip non-tensor operands.
      if (!in->get().getType().isa<RankedTensorType>())
        continue;

      // Find tensor.empty ops on the reverse SSA use-def chain. Only follow
      // equivalent tensors. I.e., stop when there are ops such as extract_slice
      // on the path.
      TraversalConfig config;
      config.followEquivalentOnly = true;
      config.alwaysIncludeLeaves = false;
      SetVector<Value> emptyTensors = state.findValueInReverseUseDefChain(
          in->get(), /*condition=*/
          [&](Value val) { return val.getDefiningOp<tensor::EmptyOp>(); },
          config);
      if (emptyTensors.empty())
        continue;

      // Find matching out operand.
      OpOperand *out = getUnusedOutOperand(op, in);
      if (!out)
        continue;

      // Check if this transform would violate dominance.
      if (!llvm::all_of(emptyTensors, [&](Value v) {
            return domInfo.properlyDominates(out->get(), v.getDefiningOp());
          }))
        continue;

      // Replace all uses of the tensor.empty, but do not delete it yet. It will
      // fold away later (to not invalidate DominanceInfo).
      for (Value v : emptyTensors) {
        assert(v.getDefiningOp<tensor::EmptyOp>() && "expected tensor.empty");
        rewriter.replaceAllUsesWith(v, out->get());
      }

      // Turn the "in" into an "out".
      rewriter.updateRootInPlace(op, [&]() {
        out->set(in->get());
        // The original "in" could be removed entirely here (because it will no
        // longer have any uses in the payload), but we delegate this to
        // existing cleanup patterns that remove unused operands.
        in->set(emptyTensors.front());
        BlockArgument outArg = op.getMatchingBlockArgument(out);
        assert(outArg.getUses().empty() && "expected that out has no uses");
        BlockArgument inArg = op.getMatchingBlockArgument(in);
        rewriter.replaceAllUsesWith(inArg, outArg);
        assert(!op.payloadUsesValueFromOperand(in) &&
               "expected that the in operand is now unused");
      });

      state.resetCache();
    }

    return WalkResult::advance();
  });
  return success();
}
