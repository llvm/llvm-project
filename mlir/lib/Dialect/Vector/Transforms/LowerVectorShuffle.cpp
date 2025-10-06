//===- LowerVectorShuffle.cpp - Lower 'vector.shuffle' operation ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the lowering of complex `vector.shuffle` operation to a
// set of simpler operations supported by LLVM/SPIR-V.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/IR/PatternMatch.h"

#define DEBUG_TYPE "vector-shuffle-lowering"

using namespace mlir;
using namespace mlir::vector;

namespace {

/// Lowers a `vector.shuffle` operation with mixed-size inputs to a new
/// `vector.shuffle` which promotes the smaller input to the larger vector size
/// and an updated version of the original `vector.shuffle`.
///
/// Example:
///
///     %0 = vector.shuffle %v1, %v2 [0, 2, 1, 3] : vector<2xf32>, vector<4xf32>
///
///   is lowered to:
///
///     %0 = vector.shuffle %v1, %v1 [0, 1, -1, -1] :
///       vector<2xf32>, vector<2xf32>
///     %1 = vector.shuffle %0, %v2 [0, 4, 1, 5] :
///       vector<4xf32>, vector<4xf32>
///
/// Note: This transformation helps legalize vector.shuffle ops when lowering
/// to SPIR-V/LLVM, which don't support shuffle operations with mixed-size
/// inputs.
///
struct MixedSizeInputShuffleOpRewrite final
    : OpRewritePattern<vector::ShuffleOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(vector::ShuffleOp shuffleOp,
                                PatternRewriter &rewriter) const override {
    auto v1Type = shuffleOp.getV1VectorType();
    auto v2Type = shuffleOp.getV2VectorType();

    // Only support 1-D shuffle for now.
    if (v1Type.getRank() != 1 || v2Type.getRank() != 1)
      return failure();

    // Bail out if inputs don't have mixed sizes.
    int64_t v1OrigNumElems = v1Type.getNumElements();
    int64_t v2OrigNumElems = v2Type.getNumElements();
    if (v1OrigNumElems == v2OrigNumElems)
      return failure();

    // Determine which input needs promotion.
    bool promoteV1 = v1OrigNumElems < v2OrigNumElems;
    Value inputToPromote = promoteV1 ? shuffleOp.getV1() : shuffleOp.getV2();
    VectorType promotedType = promoteV1 ? v2Type : v1Type;
    int64_t origNumElems = promoteV1 ? v1OrigNumElems : v2OrigNumElems;
    int64_t promotedNumElems = promoteV1 ? v2OrigNumElems : v1OrigNumElems;

    // Create a shuffle with a mask that preserves existing elements and fills
    // up with poison.
    SmallVector<int64_t> promoteMask(promotedNumElems, ShuffleOp::kPoisonIndex);
    for (int64_t i = 0; i < origNumElems; ++i)
      promoteMask[i] = i;

    Value promotedInput = rewriter.create<vector::ShuffleOp>(
        shuffleOp.getLoc(), promotedType, inputToPromote, inputToPromote,
        promoteMask);

    // Create the final shuffle with the promoted inputs.
    Value promotedV1 = promoteV1 ? promotedInput : shuffleOp.getV1();
    Value promotedV2 = promoteV1 ? shuffleOp.getV2() : promotedInput;

    SmallVector<int64_t> newMask;
    if (!promoteV1) {
      newMask = to_vector(shuffleOp.getMask());
    } else {
      // Adjust V2 indices to account for the new V1 size.
      for (auto idx : shuffleOp.getMask()) {
        int64_t newIdx = idx;
        if (idx >= v1OrigNumElems) {
          newIdx += promotedNumElems - v1OrigNumElems;
        }
        newMask.push_back(newIdx);
      }
    }

    rewriter.replaceOpWithNewOp<vector::ShuffleOp>(
        shuffleOp, shuffleOp.getResultVectorType(), promotedV1, promotedV2,
        newMask);
    return success();
  }
};
} // namespace

void mlir::vector::populateVectorShuffleLoweringPatterns(
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<MixedSizeInputShuffleOpRewrite>(patterns.getContext(), benefit);
}
