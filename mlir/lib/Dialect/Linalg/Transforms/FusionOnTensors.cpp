//===- FusionOnTensors.cpp - Implementation of linalg Fusion --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements linalg fusion on tensors
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Support/LLVM.h"

using namespace mlir;
using namespace linalg;

//===----------------------------------------------------------------------===//
// StructuredOp specific helpers.
//===----------------------------------------------------------------------===//

/// Returns the tiled slice dimensions given the tiled consumer loop dimensions.
/// The slice defines a hyper rectangular iteration space and fusing the
/// producer is always possible. However, depending on the consumer indexing
/// map, not all slice elements may be consumed and the tiles may overlap. In
/// these cases, fusion introduces redundant computation.
static SmallVector<int64_t> getTiledSliceDims(OpOperand *consumerOperand,
                                              ArrayRef<int64_t> tiledLoopDims) {
  // Get the consumer operand indexing map.
  LinalgOp consumerOp = consumerOperand->getOwner();
  AffineMap indexingMap = consumerOp.getMatchingIndexingMap(consumerOperand);

  // Search the slice dimensions tiled by a tile loop dimension.
  DenseSet<int64_t> tiledSliceDimIndices;
  for (const auto &en : enumerate(indexingMap.getResults())) {
    for (auto tiledLoopDim : tiledLoopDims) {
      if (en.value().isFunctionOfDim(tiledLoopDim))
        tiledSliceDimIndices.insert(en.index());
    }
  }
  return {tiledSliceDimIndices.begin(), tiledSliceDimIndices.end()};
}

/// Given a vector of `tiledSliceDimIndices` that represent the tiled dimensions
/// of the producer result slice returns the tiled producer loop dimensions.
/// Example:
/// ```
/// %res = linalg.fill(%cst, %input)
/// scf.for %i
///   scf.for %j
///     %slice = tensor.extract_slice %res[%i, %j]
/// ```
/// getTiledProducerLoops(%res, [0, 1]) returns the loop indices [0, 1].
static SmallVector<int64_t>
getTiledProducerLoops(OpResult producerResult,
                      ArrayRef<int64_t> tiledSliceDimIndices) {
  LinalgOp producerOp = producerResult.getOwner();

  // Get the indexing map of the `producerOp` output operand that matches
  // ´producerResult´.
  AffineMap producerIndexingMap = producerOp.getMatchingIndexingMap(
      producerOp.getOutputOperand(producerResult.getResultNumber()));

  // Keep only the tiled result slice dimensions of `producerIndexingMap`.
  AffineMap tiledProducerIndexingSubMap =
      producerIndexingMap.getSubMap(SmallVector<unsigned>(
          tiledSliceDimIndices.begin(), tiledSliceDimIndices.end()));

  // Compute the producer loop indices mapped to the tiled result slice
  // dimensions. As the output indexing map of structured operations are
  // projected permutations, `tiledProducerIndexingSubMap` has to be a
  // projected permutation as well. We can thus obtain the producer loop indices
  // by getting the positions of the result dimensions.
  // Example:
  // (d0, d1, d2) -> (d0, d2) has the result positions [0, 2].
  assert(tiledProducerIndexingSubMap.isProjectedPermutation() &&
         "expect slice and producer loop dimensions map one-to-one");
  SmallVector<int64_t> tiledProducerLoopIndices;
  llvm::transform(
      llvm::seq<unsigned>(0, tiledProducerIndexingSubMap.getNumResults()),
      std::back_inserter(tiledProducerLoopIndices), [&](unsigned idx) {
        return tiledProducerIndexingSubMap.getDimPosition(idx);
      });

  return tiledProducerLoopIndices;
}

/// Returns the producer fused in place of `sliceOp`. Tile the producer operands
/// along the `tiledSliceDimIndices` and clone the producer. Consider the case
/// of fusion of an output tensor:
/// ```
/// %1 = producer ins(...) outs(%0)
/// %2 = consumer ins(...) outs(%1)
/// ```
/// When consumer is tiled, %1 appears in the loop iter_args:
/// ```
/// %1 = producer ins(...) outs(%0)
/// %2 = scf.for ... iter_args(%1) .. (%bbarg) {
///   %t1 = tensor.extract_slice %bbarg[..]
///   %t2 = consumer ins(...) outs(%t1)
///   %r = tensor.insert_slice %t2, %bbarg[...]
/// }
/// ```
/// Fusing %1 into the loop requires updating iter_args(%1) to iter_args(%0):
/// ```
/// %2 = scf.for ... iter_args(%0) .. (%bbarg) {
///   %t0 = tensor.extract_slice %bbarg[..]
///   %t1 = producer ins(...) outs(%t0)
///   %t2 = consumer ins(...) outs(%t1)
///   %r = tensor.insert_slice %t2, %bbarg[...]
/// }
/// ```
/// This transformation is only valid if %bbarg is exclusively used by the
/// output ExtractSliceOp / InsertSliceOp pair, which is checked by the
/// `fuseProducer` method.
/// TODO: instead of check and failure, insert new iter_args each time a
/// producer is fused into a consumer and fold away unused iter_args.
static LinalgOp getTiledProducer(OpBuilder &b, OpResult producerResult,
                                 tensor::ExtractSliceOp sliceOp,
                                 ArrayRef<int64_t> tiledSliceDimIndices,
                                 ArrayRef<int64_t> tiledProducerLoopIndices,
                                 OpOperand *iterArg) {
  // Clone the producer after `sliceOp` since the slice may be reused to pass in
  // the producer result.
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointAfter(sliceOp);

  // Get the producer.
  LinalgOp producerOp = producerResult.getOwner();
  Location loc = producerOp.getLoc();

  // Obtain the `producerOp` loop bounds and the `sliceOp` ranges.
  SmallVector<OpFoldResult> producerLoopBounds;
  llvm::transform(producerOp.createLoopRanges(b, loc),
                  std::back_inserter(producerLoopBounds),
                  [&](Range range) { return range.size; });
  SmallVector<Range> sliceOpRanges = sliceOp.getOrCreateRanges(b, loc);

  // Tile the producer operands given the `sliceOp` ranges. Iterate the
  // `tiledSliceDimIndices` and store the tile offset and size for the tiled
  // slice dimension.
  SmallVector<OpFoldResult> tileIvs(producerOp.getNumLoops(), nullptr);
  SmallVector<OpFoldResult> tileSizes(producerOp.getNumLoops(),
                                      b.getIndexAttr(0));
  SmallVector<OpFoldResult> allIvs(producerOp.getNumLoops(), nullptr);
  for (auto it : zip(tiledSliceDimIndices, tiledProducerLoopIndices)) {
    int64_t tiledSliceDim = std::get<0>(it);
    int64_t tiledProducerLoop = std::get<1>(it);
    tileIvs[tiledProducerLoop] = sliceOpRanges[tiledSliceDim].offset;
    tileSizes[tiledProducerLoop] = sliceOpRanges[tiledSliceDim].size;
    allIvs[tiledProducerLoop] = tileIvs[tiledProducerLoop];
  }
  erase_value(tileIvs, OpFoldResult());
  SmallVector<Value> tiledOperands = producerOp->getOperands();
  tiledOperands = makeTiledShapes(b, loc, producerOp, tiledOperands, tileIvs,
                                  tileSizes, producerLoopBounds,
                                  /**omitPartialTileCheck=*/false);

  // Output fusion has to update the iteration arguments of the tile loop nest.
  // In particular, the iteration argument of the outermost tile loop needs to
  // be set to the producer output instead of the producer result and `clonedOp`
  // shall use the existing `sliceOp` result instead of the tiled producer
  // output operand.
  if (iterArg) {
    OpOperand *outputOperand =
        producerOp.getOutputOperand(producerResult.getResultNumber());
    iterArg->set(outputOperand->get());
    tiledOperands[outputOperand->getOperandNumber()] = sliceOp.getResult();
  }

  // Clone the producer using the tiled producer operands.
  TypeRange resultTypes = ValueRange(tiledOperands)
                              .take_back(producerOp.getNumOutputs())
                              .getTypes();
  LinalgOp clonedOp = producerOp.clone(b, loc, resultTypes, tiledOperands);

  // Shift all IndexOp results by the tile offset.
  offsetIndices(b, clonedOp, allIvs);

  return clonedOp;
}

//===----------------------------------------------------------------------===//
// TileLoopNest specific helpers.
//===----------------------------------------------------------------------===//

bool TileLoopNest::isEmpty() { return tileLoopOps.empty(); }

bool TileLoopNest::isValid() {
  // Check if `rootOp` has been tiled at least once.
  if (isEmpty() || tiledRootAndFusedOpsLoops.count(rootOp) == 0)
    return false;

  // Check if the number of loop operations and dimensions match.
  if (tileLoopOps.size() != tiledRootAndFusedOpsLoops[rootOp].size())
    return false;

  // Check if the innermost tile loop is the parent of `tiledOp`.
  if (rootOp->getParentOp() != tileLoopOps.back())
    return false;

  // Check if the tile loops are directly nested.
  return std::adjacent_find(tileLoopOps.begin(), tileLoopOps.end(),
                            [](Operation *op1, Operation *op2) {
                              return op1 != op2->getParentOp();
                            }) == tileLoopOps.end();
}

SmallVector<BlockArgument> TileLoopNest::getTiedBBArgs(BlockArgument bbArg) {
  assert(bbArg && "expect the block argument to be non-zero");
  SmallVector<BlockArgument> bbArgs;

  // Search all tile loop block arguments from inner to outer.
  for (auto tileLoop : reverse(tileLoopOps)) {
    if (bbArg.getOwner()->getParentOp() != tileLoop)
      return {};
    bbArgs.push_back(bbArg);
    OpOperand *iterArg = &tileLoop.getOpOperandForRegionIterArg(bbArg);
    bbArg = iterArg->get().dyn_cast<BlockArgument>();
  }

  // Reverse the block arguments to order them from outer to inner.
  return {bbArgs.rbegin(), bbArgs.rend()};
}

OpOperand *TileLoopNest::getTiedIterArg(BlockArgument bbArg) {
  // Search all block arguments and return the matching iteration argument.
  SmallVector<BlockArgument> bbArgs = getTiedBBArgs(bbArg);
  if (bbArgs.size() != tileLoopOps.size())
    return nullptr;
  return &tileLoopOps.front().getOpOperandForRegionIterArg(bbArgs.front());
}

bool TileLoopNest::hasOtherUses(BlockArgument bbArg,
                                tensor::ExtractSliceOp sliceOp) {
  // Check the innermost block argument is either used by the ExtractSliceOp
  // `sliceOp`, the matching InsertSliceOp, or by a DimOp. Handle other uses
  // conservatively.
  for (Operation *op : bbArg.getUsers()) {
    if (!isa<tensor::DimOp, tensor::InsertSliceOp, tensor::ExtractSliceOp>(op))
      return false;
    if (auto extractSliceOp = dyn_cast<tensor::ExtractSliceOp>(op)) {
      if (extractSliceOp != sliceOp)
        return false;
    }
    if (auto insertSliceOp = dyn_cast<tensor::InsertSliceOp>(op)) {
      SetVector<Operation *> backwardSlice;
      getBackwardSlice(insertSliceOp.getSource(), &backwardSlice,
                       [](Operation *op) {
                         return isa<LinalgOp, tensor::InsertSliceOp>(op);
                       });
      if (backwardSlice.empty() || backwardSlice.front() != sliceOp)
        return false;
    }
  }

  // Check the block arguments, except for the innermost one, have one use.
  SmallVector<BlockArgument> bbArgs = getTiedBBArgs(bbArg);
  return !all_of(bbArgs, [&](BlockArgument bbArg) {
    return bbArg.hasOneUse() || bbArg == bbArgs.back();
  });
}

LogicalResult TileLoopNest::tileRootOp(
    OpBuilder &b, ArrayRef<int64_t> tileSizes,
    ArrayRef<int64_t> tileInterchange,
    Optional<LinalgLoopDistributionOptions> tileDistribution) {
  // Exit if all tile sizes are zero.
  if (tileSizes.size() == static_cast<size_t>(count(tileSizes, 0)))
    return success();

  // Tile the root operation.
  LinalgTilingOptions tilingOptions;
  tilingOptions = tilingOptions
                      .setInterchange(SmallVector<unsigned>(
                          tileInterchange.begin(), tileInterchange.end()))
                      .setTileSizes(tileSizes)
                      .setLoopType(LinalgTilingLoopType::Loops);
  if (tileDistribution)
    tilingOptions = tilingOptions.setDistributionOptions(*tileDistribution);

  // TODO: Propagate RewriterBase everywhere.
  IRRewriter rewriter(b);
  FailureOr<TiledLinalgOp> tiledRootOp =
      tileLinalgOp(rewriter, rootOp, tilingOptions);

  // Exit if tiling the root operation fails.
  if (failed(tiledRootOp))
    return failure();

  // Replace all uses of the root operation if it has been tiled before. All
  // uses of the original untiled root operation are updated by the calling pass
  // or pattern.
  if (!isEmpty())
    rootOp->replaceAllUsesWith(tiledRootOp->tensorResults);

  // Transfer the stored `rootOp` loop dimensions if it has been tiled before.
  if (tiledRootAndFusedOpsLoops.count(rootOp) != 0) {
    tiledRootAndFusedOpsLoops[tiledRootOp->op] =
        tiledRootAndFusedOpsLoops[rootOp];
  }

  // Update the root operation and append the loops and tile loop dimensions.
  rootOp = tiledRootOp->op;
  tileLoopOps.append(tiledRootOp->loops.begin(), tiledRootOp->loops.end());
  for (const auto &en : enumerate(tileSizes)) {
    // Copy only the tiled loop dimensions with non-zero tile size.
    if (en.value() == 0)
      continue;
    tiledRootAndFusedOpsLoops[rootOp].push_back(tileInterchange[en.index()]);
  }
  assert(isValid() && "expect tile loop nest to be valid after tiling");
  return success();
}

FailureOr<LinalgOp> TileLoopNest::fuseProducer(OpBuilder &b,
                                               OpOperand *consumerOpOperand) {
  // Check if the consumer has been tiled before. For example, it may not have
  // been tiled if the outermost tile loop is a reduction loop.
  if (tiledRootAndFusedOpsLoops.count(consumerOpOperand->getOwner()) == 0)
    return failure();

  assert(this->isValid() &&
         "expect the tile loop nest to satisfy all invariants");

  // Check the tile loop nest is non-empty.
  if (isEmpty())
    return failure();

  // Check `consumerOpOperand` is defined by an ExtractSliceOp.
  auto sliceOp =
      consumerOpOperand->get().getDefiningOp<tensor::ExtractSliceOp>();
  if (!sliceOp)
    return failure();

  // Check `sliceOp` and `consumerOp` are in the same block.
  LinalgOp consumerOp = consumerOpOperand->getOwner();
  if (sliceOp->getBlock() != rootOp->getBlock() ||
      consumerOp->getBlock() != rootOp->getBlock())
    return failure();

  // Check `consumerOpOperand` is not shape-only to avoid fusion if the data is
  // not used by the `consumerOp` computation.
  BlockArgument bbArg = consumerOp.getMatchingBlockArgument(consumerOpOperand);
  if (bbArg.getUses().empty())
    return failure();

  // Check if the producer is a LinalgOp possibly passed by iteration argument.
  OpOperand *iterArg = nullptr;
  auto producerResult = sliceOp.getSource().dyn_cast<OpResult>();
  if (auto bbArg = sliceOp.getSource().dyn_cast<BlockArgument>()) {
    iterArg = getTiedIterArg(bbArg);
    // Check the iteration argument may be used to pass in the producer output.
    if (!iterArg || hasOtherUses(bbArg, sliceOp))
      return failure();
    producerResult = iterArg->get().dyn_cast<OpResult>();
  }
  if (!producerResult || !isa<LinalgOp>(producerResult.getOwner()))
    return failure();

  // Compute the tiled producer slice dimensions given the tiled consumer loops.
  SmallVector<int64_t> tiledSliceDimIndices = getTiledSliceDims(
      consumerOpOperand, tiledRootAndFusedOpsLoops[consumerOp]);
  if (tiledSliceDimIndices.empty())
    return failure();

  // Compute the tiled producer loop indices.
  SmallVector<int64_t> tiledProducerLoopIndices =
      getTiledProducerLoops(producerResult, tiledSliceDimIndices);

  // Tile the producer operands and clone the producer in place of `sliceOp`.
  LinalgOp clonedOp =
      getTiledProducer(b, producerResult, sliceOp, tiledSliceDimIndices,
                       tiledProducerLoopIndices, iterArg);
  tiledRootAndFusedOpsLoops[clonedOp] = tiledProducerLoopIndices;

  // Cast the `clonedOp` result to gap type mismatches before canonicalization.
  Type consumerOperandType = consumerOpOperand->get().getType();
  Value newResult = clonedOp->getResult(producerResult.getResultNumber());
  if (newResult.getType() != consumerOperandType) {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointAfter(clonedOp);
    newResult = b.create<tensor::CastOp>(producerResult.getLoc(),
                                         consumerOperandType, newResult);
  }

  // Replace the `sliceOp` uses except for the `clonedOp` output uses.
  sliceOp.getResult().replaceAllUsesExcept(newResult, clonedOp);
  return clonedOp;
}

ValueRange TileLoopNest::getRootOpReplacementResults() {
  assert(!isEmpty() && "expect tile loop nest to be non-empty");
  return tileLoopOps.front()->getOpResults();
}

SmallVector<LinalgOp> TileLoopNest::getAllTiledAndFusedOps() {
  SmallVector<LinalgOp> result;
  for (const auto &kvp : tiledRootAndFusedOpsLoops) {
    auto linalgOp = dyn_cast<LinalgOp>(kvp.getFirst());
    assert(linalgOp &&
           "expect all tiled and fused operations are linalg operations");
    result.push_back(linalgOp);
  }
  return result;
}
