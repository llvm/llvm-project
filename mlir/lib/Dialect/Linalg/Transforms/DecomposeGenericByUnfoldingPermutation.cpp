//===- DecomposeGenericByUnfoldingPermutation.cpp                   -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include <map>
#include <optional>
#include <utility>

using namespace mlir;
using namespace mlir::linalg;

namespace {

/// This pattern decomposes the input operand(s) of a linalg.generic that has
/// a `transpose`, `broadcast`, or a mixture of two, into explicit transpose
/// and broadcast. Having them folded into the linalg.generic is a good
/// optimization but sometimes we may want to unwrap, i.e., `unfold` them as
/// explicit transpose and broadcast. This rewrite pattern helps do it for
/// each input operand. This is useful for instance when trying to recognize
/// named ops.
///
/// The transpose, broadcast, or mixture of both, are expressed in the affine
/// map of the operand. Technically it is essentially `projected permutation`.
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
///   %x_trans = linalg.transpose
///                   ins(%x : tensor<7x8x9xf32>)
///                   outs(%e1 : tensor<9x7x8xf32>) permutation = [2, 0, 1]
///   ...
///   %x_trans_bc = linalg.broadcast
///                   ins(%x_trans : tensor<9x7x8xf32>)
///                   outs(%e2 : tensor<5x9x7x8x10xf32>) dimensions = [0, 4]
///   %2 = linalg.div
///           ins(%x_trans_bc, %y :
///                  tensor<5x9x7x8x10xf32>, tensor<5x9x7x8x10xf32>)
///           outs(%arg2 : tensor<5x9x7x8x10xf32>) -> tensor<5x9x7x8x10xf32>
///
/// Note that linalg.generic has been 'specialized' to linalg.div.
///
/// To unfold it, it is more optimal to transpose first and then do the
/// broadcast. However, if transpose is done first, the permutation map needs
/// to be expressed in terms of reduced dimension as broadcast hasn't happened
/// yet. Also, the broadcast dimensions in a linalg.generic come from other
/// operands (those not broadcasted along that particular dimension). We work
/// this out by computing the convex-polyhedron shape of the linalg.generic
/// iteration space from shapes of all the operands, both inputs and outputs.
///
struct DecomposeProjectedPermutation : public OpRewritePattern<GenericOp> {
  using OpRewritePattern<GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GenericOp genericOp,
                                PatternRewriter &rewriter) const override;
};

/// For the given `map`, determine what dimensions are transposed and what
/// dimensions are broadcasted.
/// Returns :
///   transpose-permutation, broadcast-dimensions` (empty if not needed)
///
std::pair<SmallVector<int64_t>, SmallVector<int64_t>>
computeTransposeBroadcast(AffineMap &map) {
  assert(map.isProjectedPermutation(false) && "not a projection");

  // As the map is a projection it likely operates on a smaller set of
  // dimensions as far as the transpose is concerned (rest are broadcast).
  int64_t minorSize = map.getNumResults();

  SmallVector<int64_t> minorResult;
  for (int64_t i = 0; i < minorSize; ++i) {
    auto expr = cast<AffineDimExpr>(map.getResults()[i]);
    minorResult.push_back(expr.getPosition());
  }

  // If dims are not monotonically increasing then transpose is present.
  SmallVector<int64_t> sortedResMap(minorResult);
  std::sort(sortedResMap.begin(), sortedResMap.end());
  bool hasTranspose = !std::equal(minorResult.begin(), minorResult.end(),
                                  sortedResMap.begin(), sortedResMap.end());

  // Walk the sorted map result to determine which dimensions are broadcasted.
  SmallVector<int64_t> broadcast;
  for (int64_t i = 0, j = 0; i < map.getNumInputs(); ++i) {
    if (j < minorSize && sortedResMap[j] == i) {
      j++;
      continue;
    }
    broadcast.push_back(i);
  }

  SmallVector<int64_t> permutation;
  if (hasTranspose) {
    // Consider an operand `x : tensor<7x8x9>` of a genericOp that has
    // affine map `affine_map<(d0, d1, d2, d3, d4) -> (d2, d3, d1)>`
    // `x`s access is both transposed and broadcast. But when specifying
    // the `linalg.transpose(x : tensor<7x8x9>)` the dimensions need to be
    // specified as `affine_map<(d0,d1,d2) -> (d1, d2, d0)` instead of
    // refering to d3, d4. Therefore, re-base the transpose dimensions so
    // that they start from d0.
    permutation.resize(minorSize);
    std::map<int64_t, int64_t> minorMap;
    for (int64_t i = 0; i < minorSize; ++i)
      minorMap.insert({sortedResMap[i], i});

    // Re-map the dimensions.
    SmallVector<int64_t> remappedResult(minorSize);
    for (int64_t i = 0; i < minorSize; ++i)
      remappedResult[i] = minorMap[minorResult[i]];

    /// Calculate the permutation for the transpose.
    for (unsigned i = 0; i < minorSize; ++i) {
      permutation[remappedResult[i]] = i;
    }
  }
  return {permutation, broadcast};
}

LogicalResult DecomposeProjectedPermutation::matchAndRewrite(
    GenericOp op, PatternRewriter &rewriter) const {
  if (!op.hasPureTensorSemantics() || op.isSingleInputOutput() ||
      op.isSingleYieldOp() || !op.isAllParallelLoops())
    return failure();

  // If the map of an operand is not a `projected permutation` then
  // it cannot be decomposed to mere transpose and broadcast.
  // The requirement that all maps be `projected permutation` may be
  // over-restrictive but since we need to determine shape of the
  // iteration space as well, reject if any map violates assumption.
  for (auto &opOperand : op->getOpOperands()) {
    auto map = op.getMatchingIndexingMap(&opOperand);
    if (!map.isProjectedPermutation(false))
      return failure();
  }

  // Decomposing linalg.generic involves creating `tensor.empty`
  // which can have dynamic shapes but then we would have to work
  // out which operand can supply that runtime-value (tensor.dim).
  // Leaving it as a future TODO.
  if (llvm::any_of(op->getOpOperands(), [](OpOperand &oper) {
        auto opType = cast<RankedTensorType>(oper.get().getType());
        return ShapedType::isDynamicShape(opType.getShape());
      }))
    return failure();

  auto outputShape = op.getStaticLoopRanges();

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

    auto [permutation, broadcastedDims] = computeTransposeBroadcast(map);

    // Does it need transpose?
    if (!permutation.empty()) {
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

    // Does it require broadcast?
    if (!broadcastedDims.empty()) {
      assert(broadcastedDims.size() && "should have non size broadcast");
      Value emptyTensor = rewriter.create<tensor::EmptyOp>(
          loc, outputShape, inputRTType.getElementType());

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

void mlir::linalg::populateDecomposeProjectedPermutationPatterns(
    RewritePatternSet &patterns) {
  patterns.insert<DecomposeProjectedPermutation>(patterns.getContext());
}
