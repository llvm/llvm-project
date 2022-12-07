//===- DataLayoutPropagation.cpp -----------------------------------------===///
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_LINALGDATALAYOUTPROPAGATION
#include "mlir/Dialect/Linalg/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::linalg;

#define DEBUG_TYPE "linalg-data-layout-propagation"

namespace {

/// Returns a tuple for packed operand and indexing_map with the assumptions:
///   1) The generic op is the producer of the pack op.
///   2) The generic op has only one result.
///   3) The indexing map of the output operand is identity.
/// If the operand is a scalar or packing dimensions are all irrelevant to the
/// operand, the opreand and the updated indexing map will be returned.
/// Otherwise, it returns the packed operand and the updated indexing map. E.g.,
///
///   #map0 = affine_map<(d0, d1) -> (d0, d1)>
///   #map1 = affine_map<(d0, d1) -> (d0)>
///   #map2 = affine_map<(d0, d1) -> (d1)>
///   %0 = linalg.generic {indexing_maps = [#map1, #map2, #map0],
///                        iterator_types = ["parallel", "parallel"]}
///      ins(%arg0, %arg1 : tensor<?xf32>, tensor<?xf32>)
///      outs(%init : tensor<?x?xf32>) {
///    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
///      %4 = arith.addf %arg3, %arg4 : f32
///      linalg.yield %4 : f32
///  } -> tensor<?x?xf32>
///  %1 = tensor.pack %0
///    inner_dims_pos = [0, 1]
///    inner_tiles = [8, 2]
///    into %dest : tensor<?x?xf32> -> tensor<?x?x8x2xf32>
///
///  Taking the first input operand as an example, the inner tile size of d1 is
///  8. Thus, the below operation and `affine_map<(d0, d1, d2, d3)> ->
///  affine_map<(d1, d3)>` will be returned.
///
///  %pack = tensor.pack %arg0
///    inner_dims_pos = [0]
///    inner_tiles = [8]
///    into %init : tensor<?xf32> -> tensor<?x8xf32>
static std::tuple<Value, AffineMap>
getOrCreatePackedViewOfOperand(OpBuilder &b, Location loc,
                               tensor::PackOp packOp, GenericOp genericOp,
                               OpOperand *opOperand) {
  int numOrigLoops = genericOp.getNumLoops();
  int64_t numInnerLoops = packOp.getInnerDimsPos().size();
  int64_t numLoops = numOrigLoops + numInnerLoops;
  AffineMap origIndexingMap = genericOp.getMatchingIndexingMap(opOperand);
  SmallVector<AffineExpr> exprs(origIndexingMap.getResults());

  if (genericOp.isScalar(opOperand))
    return std::make_tuple(
        opOperand->get(),
        AffineMap::get(numLoops, 0, exprs, packOp.getContext()));

  llvm::SetVector<int64_t> innerDimsPosSet(packOp.getInnerDimsPos().begin(),
                                           packOp.getInnerDimsPos().end());
  // Mapping from AffinDimExpr of indexing maps to the operand shape dimension.
  DenseMap<int64_t, int64_t> iterMapToDim;
  for (auto [index, expr] : llvm::enumerate(origIndexingMap.getResults())) {
    int64_t dimPos = expr.cast<AffineDimExpr>().getPosition();
    if (!innerDimsPosSet.contains(dimPos))
      continue;
    iterMapToDim[dimPos] = index;
  }

  // Construct the information of packing data dimensions and new indexing maps
  // for the operand.
  SmallVector<int64_t> innerDimsPos;
  SmallVector<OpFoldResult> innerTileSizes;
  for (auto [index, value] : llvm::enumerate(
           llvm::zip(packOp.getInnerDimsPos(), packOp.getMixedTiles()))) {
    int64_t dimPos = std::get<0>(value);
    if (!iterMapToDim.count(dimPos))
      continue;
    innerDimsPos.push_back(iterMapToDim[dimPos]);
    innerTileSizes.push_back(std::get<1>(value));
    exprs.push_back(b.getAffineDimExpr(numOrigLoops + index));
  }
  auto indexingMap = AffineMap::get(numLoops, 0, exprs, packOp.getContext());

  SmallVector<int64_t> outerDimsPerm;
  for (auto outDim : packOp.getOuterDimsPerm()) {
    if (!iterMapToDim.count(outDim))
      continue;
    outerDimsPerm.push_back(iterMapToDim[outDim]);
  }

  // The operand does not have dimensions that relates to pack op.
  if (innerDimsPos.empty() && outerDimsPerm.empty())
    return std::make_tuple(opOperand->get(), indexingMap);

  auto empty = tensor::PackOp::createDestinationTensor(
      b, loc, opOperand->get(), innerTileSizes, innerDimsPos, outerDimsPerm);
  auto packedOperand = b.create<tensor::PackOp>(
      loc, opOperand->get(), empty, innerDimsPos, innerTileSizes,
      packOp.getPaddingValue(), outerDimsPerm);
  return std::make_tuple(packedOperand, indexingMap);
}

/// Bubbles up tensor.pack op through elementwise generic op. This
/// swap pack(generic) to generic(pack). The new generic op works on packed
/// domain; pack ops are created for input and output operands. E.g.,
///
///     #map0 = affine_map<(d0, d1) -> (d0, d1)>
///     %0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
///     %1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
///     %2 = tensor.empty(%0, %1) : tensor<?x?xf32>
///     %3 = linalg.generic {indexing_maps = [#map0, #map0],
///                          iterator_types = ["parallel", "parallel"]}
///         ins(%arg0 : tensor<?x?xf32>)
///         outs(%2 : tensor<?x?xf32>) {
///       ^bb0(%arg3: f32, %arg4: f32):
///         %4 = arith.addf %arg3, %arg3 : f32
///         linalg.yield %4 : f32
///     } -> tensor<?x?xf32>
///     %4 = tensor.pack %3
///       inner_dims_pos = [0, 1]
///       inner_tiles = [8, 2]
///       into %dest : tensor<?x?xf32> -> tensor<?x?x8x2xf32>
///
/// will be converted to
///
///     #map = affine_map<()[s0] -> (s0 ceildiv 8)>
///     #map1 = affine_map<()[s0] -> (s0 ceildiv 2)>
///     #map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
///     %dim = tensor.dim %arg0, %c0 : tensor<?x?xf32>
///     %dim_0 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
///     %0 = affine.apply #map()[%dim]
///     %1 = affine.apply #map1()[%dim_0]
///     %2 = tensor.empty(%0, %1) : tensor<?x?x8x2xf32>
///     %pack = tensor.pack %arg0
///       inner_dims_pos = [0, 1]
///       inner_tiles = [8, 2]
///       into %2 : tensor<?x?xf32> -> tensor<?x?x8x2xf32>
///     %3 = linalg.generic {indexing_maps = [#map2, #map2],
///       iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
///       ins(%pack : tensor<?x?x8x2xf32>)
///       outs(%arg1 : tensor<?x?x8x2xf32>) {
///     ^bb0(%in: f32, %out: f32):
///       %4 = arith.addf %in, %in : f32
///       linalg.yield %4 : f32
///     } -> tensor<?x?x8x2xf32>
static FailureOr<GenericOp>
bubbleUpPackOpThroughElemGenericOp(RewriterBase &rewriter,
                                   tensor::PackOp packOp) {
  auto genericOp = packOp.getSource().getDefiningOp<GenericOp>();
  if (!genericOp)
    return failure();

  if (!isElementwise(genericOp))
    return failure();

  // TODO: Relax the restriction. We are able to bubble up the pack op through
  // multi-result generic op. It just needs more work.
  if (genericOp.getNumResults() != 1)
    return failure();

  // TODO: Add an option for allowing padding values. It could introduce
  // undefined behavior if we unconditionally propagate pack op through all
  // the ops. E.g., if the padding value is zero and there are division ops in
  // a generic op. Some values of padding area could be NaN (0/0).
  if (packOp.getPaddingValue())
    return failure();

  OpOperand *opOperand = genericOp.getDpsInitOperand(0);
  // TODO: Add support for all permutation indexing maps.
  if (!genericOp.getMatchingIndexingMap(opOperand).isIdentity())
    return rewriter.notifyMatchFailure(
        packOp, "the result of generic op does not have identity indexing_map");

  Location loc = packOp.getLoc();
  SmallVector<Value> inputOperands;
  SmallVector<AffineMap> indexingMaps;
  for (OpOperand *inputOperand : genericOp.getDpsInputOperands()) {
    auto [packedOperand, packedIndexingMap] = getOrCreatePackedViewOfOperand(
        rewriter, loc, packOp, genericOp, inputOperand);
    inputOperands.push_back(packedOperand);
    indexingMaps.push_back(packedIndexingMap);
  }

  int64_t numLoops = genericOp.getNumLoops();
  int64_t numInnerLoops = packOp.getInnerDimsPos().size();
  int64_t newNumLoops = numLoops + numInnerLoops;
  SmallVector<utils::IteratorType> iterTypes =
      genericOp.getIteratorTypesArray();
  iterTypes.append(numInnerLoops, utils::IteratorType::parallel);

  SmallVector<AffineExpr> outExprs(
      genericOp.getMatchingIndexingMap(opOperand).getResults());
  for (int i = 0; i < numInnerLoops; ++i)
    outExprs.push_back(rewriter.getAffineDimExpr(numLoops + i));
  indexingMaps.push_back(
      AffineMap::get(newNumLoops, 0, outExprs, rewriter.getContext()));

  auto newGenericOp = rewriter.create<linalg::GenericOp>(
      loc, packOp.getDestType(), inputOperands, packOp.getDest(), indexingMaps,
      iterTypes, /*bodyBuild=*/nullptr,
      linalg::getPrunedAttributeList(genericOp));
  rewriter.cloneRegionBefore(genericOp.getRegion(), newGenericOp.getRegion(),
                             newGenericOp.getRegion().begin());
  return newGenericOp;
}

// Wrapper pattern that applies bubbleUpPackOpThroughElemGenericOp method.
struct BubbleUpPackOpThroughElemGenericOpPattern
    : public OpRewritePattern<tensor::PackOp> {
  using OpRewritePattern<tensor::PackOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::PackOp packOp,
                                PatternRewriter &rewriter) const override {
    auto genericOp = bubbleUpPackOpThroughElemGenericOp(rewriter, packOp);
    if (failed(genericOp))
      return failure();
    rewriter.replaceOp(packOp, genericOp.value().getResults());
    return success();
  }
};
} // namespace

void mlir::linalg::populateDataLayoutPropagationPatterns(
    RewritePatternSet &patterns) {
  patterns.insert<BubbleUpPackOpThroughElemGenericOpPattern>(
      patterns.getContext());
}
