//===- VectorInferInBounds.cpp - Infer in_bounds for transfer ops ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass that infers the `in_bounds` attribute of
// `vector.transfer_read` and `vector.transfer_write` operations whose indices
// are not constants, using value-bounds analysis.
//
// The canonicalizer already marks a transfer in-bounds when its indices are
// constants. That covers very little real code: after tiling and vectorization
// the index of a transfer is typically a loop induction variable, or an affine
// expression of one, and the canonicalizer gives up. Value-bounds analysis can
// still bound such an index, but running it from a folder would impose the cost
// on every canonicalization of every function, so it lives here, in an opt-in
// pass, instead.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/Passes.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir {
namespace vector {
#define GEN_PASS_DEF_VECTORINFERINBOUNDS
#include "mlir/Dialect/Vector/Transforms/Passes.h.inc"
} // namespace vector
} // namespace mlir

#define DEBUG_TYPE "vector-infer-in-bounds"

using namespace mlir;
using namespace mlir::vector;

namespace {

/// Returns "true" if the transfer of `op` along vector dimension `resultIdx`,
/// which indexes the source at `indicesIdx`, is provably within the bounds of
/// the source.
template <typename TransferOp>
static bool isProvablyInBounds(TransferOp op, int64_t resultIdx,
                               int64_t indicesIdx) {
  // A dynamic source dimension has no static size to compare against.
  if (op.getShapedType().isDynamicDim(indicesIdx))
    return false;
  // A scalable vector dimension holds `vscale * N` elements, so its static size
  // is only a lower bound and cannot prove that the transfer fits.
  if (op.getVectorType().getScalableDims()[resultIdx])
    return false;

  int64_t sourceSize = op.getShapedType().getDimSize(indicesIdx);
  int64_t vectorSize = op.getVectorType().getDimSize(resultIdx);
  // Largest index at which a full vector still fits. Computed as a subtraction
  // rather than adding to the index, which could overflow.
  int64_t maxStart = sourceSize - vectorSize;
  if (maxStart < 0)
    return false;

  Value index = op.getIndices()[indicesIdx];

  // The transfer is in bounds if even the largest index the enclosing loops can
  // produce still leaves room for a full vector. `closedUB` is required because
  // the bound wanted here is the largest attainable index, not one past it.
  FailureOr<int64_t> maxIndex = ValueBoundsConstraintSet::computeConstantBound(
      presburger::BoundType::UB, index, /*stopCondition=*/nullptr,
      ValueBoundsOptions{/*closedUB=*/true});
  if (failed(maxIndex) || *maxIndex > maxStart)
    return false;

  // `in_bounds` promises that the transfer stays within the source *including
  // its starting point*, so the smallest attainable index must be non-negative
  // as well. Queried only once the upper bound holds, so that an index that
  // fails that pays for one query rather than two.
  FailureOr<int64_t> minIndex = ValueBoundsConstraintSet::computeConstantBound(
      presburger::BoundType::LB, index);
  return succeeded(minIndex) && *minIndex >= 0;
}

/// Recomputes the `in_bounds` attribute of `op`, marking a dimension in-bounds
/// when `isProvablyInBounds` can prove it. Dimensions already marked in-bounds
/// are left alone: this only ever adds information.
template <typename TransferOp>
static void inferInBounds(TransferOp op) {
  // TODO: Support the 0-d corner case, which has no vector dimension to mark.
  if (op.getTransferRank() == 0)
    return;

  AffineMap permutationMap = op.getPermutationMap();
  bool changed = false;
  SmallVector<bool, 4> newInBounds;
  newInBounds.reserve(op.getTransferRank());
  // Indices of the non-broadcast dims, needed when handling broadcast dims.
  SmallVector<unsigned> nonBcastDims;

  // 1. Process the non-broadcast dims.
  for (unsigned i = 0; i < op.getTransferRank(); ++i) {
    // 1.1. Already in-bounds, nothing to prove.
    if (op.isDimInBounds(i)) {
      newInBounds.push_back(true);
      continue;
    }
    // 1.2. Marked out-of-bounds; try to prove otherwise.
    bool inBounds = false;
    if (auto dimExpr = dyn_cast<AffineDimExpr>(permutationMap.getResult(i))) {
      inBounds = isProvablyInBounds(op, /*resultIdx=*/i,
                                    /*indicesIdx=*/dimExpr.getPosition());
      nonBcastDims.push_back(i);
    }
    newInBounds.push_back(inBounds);
    changed |= inBounds;
  }

  // 2. Handle the broadcast dims. A broadcast dim reads the same element for
  // every lane, so it is in-bounds exactly when every non-broadcast dim is.
  bool allNonBcastDimsInBounds = llvm::all_of(
      nonBcastDims, [&newInBounds](unsigned idx) { return newInBounds[idx]; });
  if (allNonBcastDimsInBounds) {
    for (size_t idx : permutationMap.getBroadcastDims()) {
      changed |= !newInBounds[idx];
      newInBounds[idx] = true;
    }
  }

  if (!changed)
    return;

  // OpBuilder is only used as a helper to build a BoolArrayAttr.
  OpBuilder b(op.getContext());
  op.setInBoundsAttr(b.getBoolArrayAttr(newInBounds));
}

struct VectorInferInBoundsPass
    : public vector::impl::VectorInferInBoundsBase<VectorInferInBoundsPass> {

  void runOnOperation() override {
    // A `walk` rather than a greedy pattern set: the attribute of one transfer
    // never affects that of another, so there is no fixpoint to reach and each
    // op needs to be visited exactly once. Value-bounds queries are expensive
    // enough that re-running them to a fixpoint would be wasteful.
    getOperation().walk([](Operation *op) {
      if (auto readOp = dyn_cast<vector::TransferReadOp>(op))
        inferInBounds(readOp);
      else if (auto writeOp = dyn_cast<vector::TransferWriteOp>(op))
        inferInBounds(writeOp);
    });
  }
};

} // namespace
