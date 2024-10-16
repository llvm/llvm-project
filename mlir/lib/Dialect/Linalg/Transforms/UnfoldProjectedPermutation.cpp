//===- UnfoldProjectedPermutation.cpp - extract projected projections   ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements pattern to decompose the operand of a GenericOp that
// has `transpose+broadcast` juxtaposed via its affine map into separate
// transpose and broadcast ops.
//
//===----------------------------------------------------------------------===//
//
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include <utility>

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include <map>
#include <optional>
#include <vector>

using namespace mlir;
using namespace mlir::linalg;

namespace {

/// Projected permutation are effectively folding in of a mixture of
/// transpose and broadcast into the affine map of the operand.
/// While folding of transpose and broadcast into the affine map of the
/// linalg.generic operand is a very effective optimization, sometimes
/// we may want to unfold that, for instance when recognizing named ops.
///
///  Example
///
/// ```mlir
///
/// #projection = affine_map<(d0, d1, d2, d3, d4) -> (d2, d3, d1)>
/// #identity   = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
/// ...
///    %res = linalg.generic
///       { indexing_maps = [#projection, #identity, #identity],
///       iterator_types = ["parallel", "parallel", "parallel",
///                         "parallel", "parallel"]}
///       ins(%x, %y : tensor<7x8x9xf32>, tensor<5x9x7x8x10xf32>)
///       outs(%z : tensor<5x9x7x8x10xf32>) {
///         ^bb0(%in: f32, %in_1: f32, %out: f32):
///              %div = arith.divf %in, %in_1 : f32
///              linalg.yield %div : f32
///    } -> tensor<5x9x7x8x10xf32>
/// ```
///
/// In the above IR operand `%x` map is a projected-permutation. This can be
/// unfolded as:
///
/// ```mlir
///   ...
///   %transposed = linalg.transpose ins(%x : tensor<7x8x9xf32>)
///                    outs(%e1 : tensor<9x7x8xf32>) permutation = [2, 0, 1]
///   ...
///   %broadcasted = linalg.broadcast ins(%transposed : tensor<9x7x8xf32>)
///                    outs(%e2 : tensor<5x9x7x8x10xf32>) dimensions = [0, 4]
///   %2 = linalg.div
///           ins(%broadcasted, %y :
///                  tensor<5x9x7x8x10xf32>, tensor<5x9x7x8x10xf32>)
///           outs(%arg2 : tensor<5x9x7x8x10xf32>) -> tensor<5x9x7x8x10xf32>
///
/// Note that linalg.generic has been 'specialized' to linalg.div.
/// To unfold it is more effective to transpose first and then do the broadcast.
/// However, if transpose is done first, the permutation map needs to be
/// expressed in terms of reduced dimension (as broadcast hasn't happened yet).
/// Also, the broadcast dimensions in a linalg.generic come from other operands
/// (those not broadcasted along that particular dimension). We work this out
/// by computing the polytope shape of the linalg.gneric from shapes of all the
/// operands (inputs and outputs).

struct UnfoldProjectedPermutation : public OpRewritePattern<GenericOp> {
  using OpRewritePattern<GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GenericOp genericOp,
                                PatternRewriter &rewriter) const override;
};

/// Calculate shape (dimensions) of the iteration space polytope.
/// This is calculated by concatenating the indexing maps of all operands
/// of the generic; inverting the concatenation; concatenating all the
/// shapes of the operands; and then doing `apply map` to those two.
SmallVector<int64_t> getPolytopeDims(GenericOp op) {
  assert(op.hasPureTensorSemantics() && "works only on tensors");

  /// Concat indexing maps of all operands and invert the mapping.
  auto maps = op.getIndexingMapsArray();
  auto concat = concatAffineMaps(maps);
  auto inverse = inversePermutation(concat);

  /// Concat the size of each dims of all operands.
  SmallVector<int64_t> dims;
  for (auto &operand : op->getOpOperands()) {
    auto rankedType = cast<RankedTensorType>(operand.get().getType());
    for (auto size : rankedType.getShape())
      dims.push_back(size);
  }

  /// Match the inverse map with dims to get polytope dimensions.
  /// Note that some maybe 'kDynamic'.
  return applyPermutationMap<int64_t>(inverse, dims);
}

/// For the given `map` determine what dimensions are transposed
/// and what dimensions are broadcasted.
/// Returns :
///  `isTransposed, isBroadcast,
///   transpose-permutation, broadcast-dimensions`
///
std::tuple<bool, bool, SmallVector<int64_t>, SmallVector<int64_t>>
computeTransposeBroadcast(AffineMap &map) {
  assert(map.isProjectedPermutation(false) && "not a projection");

  // Dimensions that don't appear on result are broadcast.
  int64_t minorSize = map.getNumResults();

  // Convert affine expr to int64_t.
  SmallVector<int64_t> minorResult;
  for (int64_t i = 0; i < minorSize; ++i) {
    auto expr = cast<AffineDimExpr>(map.getResults()[i]);
    minorResult.push_back(expr.getPosition());
  }

  // If dims are not monotonically increasing then transpose is present.
  SmallVector<int64_t> sorted(minorResult);
  std::sort(sorted.begin(), sorted.end());
  bool hasTranspose = !std::equal(minorResult.begin(), minorResult.end(),
                                  sorted.begin(), sorted.end());

  // Walk the sorted map result to determine which dimensions are broadcasted.
  SmallVector<int64_t> broadcast;
  for (int64_t i = 0, j = 0; i < map.getNumInputs(); ++i) {
    if (j < minorSize && sorted[j] == i) {
      j++;
      continue;
    }
    broadcast.push_back(i);
  }
  bool hasBroadcast = broadcast.size();

  /// Consider an operand `x : tensor<7x8x9>` of a genericOp that has
  /// affine map `affine_map<(d0, d1, d2, d3, d4) -> (d2, d3, d1)>`
  /// `x`s access is both transposed and brodcast. But when specifying
  /// the `linalg.transpose(x : tensor<7x8x9>)` the dimensions need to be
  /// specified as `affine_map<(d0,d1,d2) -> (d1, d2, d0)` instead of
  /// refering to d3, d4. Therefore, re-base the transpose dimensions so
  /// that they start from d0.
  std::map<int64_t, int64_t> minorMap;
  for (int64_t i = 0; i < minorSize; ++i)
    minorMap.insert({sorted[i], i});

  // Re-map the dimensions.
  SmallVector<int64_t> remappedResult(minorSize);
  for (int64_t i = 0; i < minorSize; ++i)
    remappedResult[i] = minorMap[minorResult[i]];

  /// Calculate the permutation for the transpose.
  SmallVector<int64_t> permutation(minorSize);
  for (unsigned i = 0; i < minorSize; ++i) {
    permutation[remappedResult[i]] = i;
  }

  return {hasTranspose, hasBroadcast, permutation, broadcast};
}

LogicalResult
UnfoldProjectedPermutation::matchAndRewrite(GenericOp op,
                                            PatternRewriter &rewriter) const {
  if (!op.hasPureTensorSemantics() || op.isSingleInputOutput() ||
      op.isSingleYieldOp() || !op.isAllParallelLoops())
    return failure();

  // All maps need to be projected permutations.
  for (auto &opOperand : op->getOpOperands()) {
    auto map = op.getMatchingIndexingMap(&opOperand);
    if (!map.isProjectedPermutation(false))
      return failure();
  }

  // Currently we handle only static shapes.
  for (auto &operand : op->getOpOperands()) {
    auto rankedType = cast<RankedTensorType>(operand.get().getType());
    for (auto size : rankedType.getShape())
      if (size == ShapedType::kDynamic)
        return failure();
  }

  // Calculate polytope bounds from affine maps and operand(s) shapes.
  auto polytope = getPolytopeDims(op);

  auto loc = op.getLoc();
  bool isChanged = false;
  SmallVector<Value> newInitValues = op.getDpsInputs();
  SmallVector<AffineMap> newMap = op.getIndexingMapsArray();

  // Walk over each input operand and unfold if it is transposed, broadcast
  // or mix of two via operand's affine-map.
  for (int64_t i = 0; i < op.getNumDpsInputs(); ++i) {
    auto &map = newMap[i];
    auto inputRTType = cast<RankedTensorType>(newInitValues[i].getType());
    auto elType = inputRTType.getElementType();

    /// Nothing to do if map is already an identity.
    if (map.isIdentity())
      continue;

    auto [hasTranspose, hasBroadcast, permutation, broadcastedDims] =
        computeTransposeBroadcast(map);

    if (hasTranspose) {
      /// linalg.transpose permutes the dimensions of input using
      /// rule: dim(result, i) = dim(input, permutation[i])
      SmallVector<int64_t> transposedShape(map.getNumResults());
      for (int64_t i = 0; i < map.getNumResults(); ++i)
        transposedShape[i] = inputRTType.getShape()[permutation[i]];

      Value emptyTensor =
          rewriter.create<tensor::EmptyOp>(loc, transposedShape, elType);

      auto transposeOp = rewriter.create<TransposeOp>(loc, newInitValues[i],
                                                      emptyTensor, permutation);
      newInitValues[i] = transposeOp->getResult(0);
      isChanged = true;
    }

    // Does it require broadcast
    if (hasBroadcast) {
      assert(broadcastedDims.size() && "should have non size broadcast");
      Value emptyTensor = rewriter.create<tensor::EmptyOp>(
          loc, polytope, inputRTType.getElementType());

      auto broadcastOp = rewriter.create<linalg::BroadcastOp>(
          loc, newInitValues[i], emptyTensor, broadcastedDims);

      newInitValues[i] = broadcastOp->getResult(0);
      isChanged = true;
    }
    newMap[i] = rewriter.getMultiDimIdentityMap(map.getNumDims());
  }

  if (isChanged) {
    SmallVector<Value> operands = op->getOperands();
    ValueRange operandsRef(operands);

    auto newOp = rewriter.create<linalg::GenericOp>(
        /*location=*/op.getLoc(),
        /*resultTensorTypes=*/op->getResultTypes(),
        /*inputs=*/newInitValues,
        /*outputs=*/operandsRef.drop_front(op.getNumDpsInputs()),
        /*indexingMaps=*/newMap,
        /*iteratorTypes=*/op.getIteratorTypesArray());

    newOp.getRegion().takeBody(op->getRegion(0));
    rewriter.replaceOp(op, newOp->getResults());
  }
  return success();
}

} // namespace

void mlir::linalg::populateUnfoldProjectedPermutationPatterns(
    RewritePatternSet &patterns) {
  patterns.insert<UnfoldProjectedPermutation>(patterns.getContext());
}
