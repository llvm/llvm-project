//===- Fusion.cpp - Implementation of linalg Fusion -----------------------===//
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

#include "PassDetail.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgTypes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
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
SmallVector<int64_t> getTiledSliceDims(OpOperand *consumerOperand,
                                       ArrayRef<int64_t> tiledLoopDims) {
  // Get the consumer operand indexing map.
  LinalgOp consumerOp = consumerOperand->getOwner();
  AffineMap indexingMap = consumerOp.getTiedIndexingMap(consumerOperand);

  // Search the slice dimensions tiled by a tile loop dimension.
  DenseSet<int64_t> tiledSliceDims;
  for (auto en : enumerate(indexingMap.getResults())) {
    for (auto tiledLoopDim : tiledLoopDims) {
      if (en.value().isFunctionOfDim(tiledLoopDim))
        tiledSliceDims.insert(en.index());
    }
  }
  return {tiledSliceDims.begin(), tiledSliceDims.end()};
}

/// Returns the producer fused in place of `sliceOp`. Tile the producer operands
/// along the `tiledSliceDims` and clone the producer. Consider the case of
/// fusion of an output tensor:
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
                                 ArrayRef<int64_t> tiledSliceDims,
                                 OpOperand *iterArg) {
  // Clone the producer after `sliceOp` since the slice may be reused to pass in
  // the producer result.
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointAfter(sliceOp);

  // Get the producer.
  LinalgOp producerOp = producerResult.getOwner();
  Location loc = producerOp.getLoc();

  // Obtain the `producerOp` loop bounds and the `sliceOp` ranges.
  SmallVector<Value> producerLoopBounds;
  transform(producerOp.createLoopRanges(b, loc),
            std::back_inserter(producerLoopBounds),
            [](Range range) { return range.size; });
  SmallVector<Range> sliceOpRanges = sliceOp.getOrCreateRanges(b, loc);

  // Get the producer result indexing map.
  AffineMap producerIndexingMap = producerOp.getTiedIndexingMap(
      producerOp.getOutputOperand(producerResult.getResultNumber()));

  // Tile the producer operands given the `sliceOp` ranges. Iterate the
  // `tiledSliceDims` and store the tile offset and size for the tiled slice
  // dimension. Assumes the mapping from slice dimensions to producer loops is a
  // permutation.
  auto zero = b.create<arith::ConstantIndexOp>(loc, 0);
  SmallVector<Value> tileIvs(producerOp.getNumLoops(), nullptr);
  SmallVector<Value> tileSizes(producerOp.getNumLoops(), zero);
  SmallVector<Value> allIvs(producerOp.getNumLoops(), nullptr);
  for (int64_t tiledSliceDim : tiledSliceDims) {
    AffineExpr result = producerIndexingMap.getResults()[tiledSliceDim];
    assert(result.isa<AffineDimExpr>() &&
           "expect producer indexing map is a projected permutation");
    int64_t tiledProducerLoop = result.cast<AffineDimExpr>().getPosition();
    tileIvs[tiledProducerLoop] = sliceOpRanges[tiledSliceDim].offset;
    tileSizes[tiledProducerLoop] = sliceOpRanges[tiledSliceDim].size;
    allIvs[tiledProducerLoop] = tileIvs[tiledProducerLoop];
  }
  erase_value(tileIvs, nullptr);
  SmallVector<Value> tiledOperands = producerOp.getInputAndOutputOperands();
  tiledOperands = makeTiledShapes(b, loc, producerOp, tiledOperands, tileIvs,
                                  tileSizes, producerLoopBounds);

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
  addTileLoopIvsToIndexOpResults(b, clonedOp, allIvs);

  return clonedOp;
}

//===----------------------------------------------------------------------===//
// TileLoopNest specific helpers.
//===----------------------------------------------------------------------===//

bool TileLoopNest::isEmpty() { return loopOps.empty(); }

bool TileLoopNest::isValid() {
  // Check if the number of `tileLoopOps` and `tileLoopDims` match.
  if (loopOps.size() != loopDims.size())
    return false;

  // Check if the innermost tile loop is the parent of `tiledOp`.
  if (rootOp->getParentOp() != loopOps.back())
    return false;

  // Check if the tile loops are directly nested.
  return std::adjacent_find(loopOps.begin(), loopOps.end(),
                            [](Operation *op1, Operation *op2) {
                              return op1 != op2->getParentOp();
                            }) == loopOps.end();
}

SmallVector<BlockArgument> TileLoopNest::getTiedBBArgs(BlockArgument bbArg) {
  assert(bbArg && "expect the block argument to be non-zero");
  SmallVector<BlockArgument> bbArgs;

  // Search all tile loop block arguments from inner to outer.
  for (auto tileLoop : reverse(loopOps)) {
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
  if (bbArgs.size() != loopOps.size())
    return nullptr;
  return &loopOps.front().getOpOperandForRegionIterArg(bbArgs.front());
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
      getBackwardSlice(insertSliceOp.source(), &backwardSlice,
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

LogicalResult TileLoopNest::tileRootOp(OpBuilder &b,
                                       ArrayRef<int64_t> tileSizes,
                                       ArrayRef<int64_t> tileInterchange) {
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
  Optional<TiledLinalgOp> tiledRootOp = tileLinalgOp(b, rootOp, tilingOptions);

  // Exit if tiling the root operation fails.
  if (!tiledRootOp.hasValue())
    return failure();

  // Replace all uses of the root operation if it has been tiled before. All
  // uses of the original untiled root operation are updated by the calling pass
  // or pattern.
  if (!isEmpty())
    rootOp->replaceAllUsesWith(tiledRootOp->tensorResults);

  // Update the root operation and append the loops and tile loop dimensions.
  rootOp = tiledRootOp->op;
  loopOps.append(tiledRootOp->loops.begin(), tiledRootOp->loops.end());
  for (auto en : enumerate(tileSizes)) {
    // Copy only the tiled loop dimensions with non-zero tile size.
    if (en.value() == 0)
      continue;
    loopDims.push_back(tileInterchange[en.index()]);
  }
  assert(isValid() && "expect tile loop nest to be valid after tiling");

  return success();
}

FailureOr<LinalgOp> TileLoopNest::fuseProducer(OpBuilder &b,
                                               OpOperand *rootOpOperand) {
  assert(rootOpOperand->getOwner() == rootOp &&
         "expect the root op to be the owner of the operand to fuse");
  assert(this->isValid() &&
         "expect the tile loop nest to satisfy all invariants");

  // Check the tile loop nest is non-empty.
  if (isEmpty())
    return failure();

  // Check `rootOpOperand` is defined by an ExtractSliceOp.
  auto sliceOp = rootOpOperand->get().getDefiningOp<tensor::ExtractSliceOp>();
  if (!sliceOp)
    return failure();

  // Check `sliceOp` is tiled by the tile loop nest.
  if (sliceOp->getParentOp() != rootOp->getParentOp())
    return failure();

  // Check if the producer is a LinalgOp possibly passed by iteration argument.
  OpOperand *iterArg = nullptr;
  auto producerResult = sliceOp.source().dyn_cast<OpResult>();
  if (auto bbArg = sliceOp.source().dyn_cast<BlockArgument>()) {
    iterArg = getTiedIterArg(bbArg);
    // Check the iteration argument may be used to pass in the producer output.
    if (!iterArg || hasOtherUses(bbArg, sliceOp))
      return failure();
    producerResult = iterArg->get().dyn_cast<OpResult>();
  }
  if (!producerResult || !isa<LinalgOp>(producerResult.getOwner()))
    return failure();

  // Compute the tiled producer slice dimensions given the tiled root operation
  // loop dimensions `loopDims`.
  SmallVector<int64_t> tiledSliceDims =
      getTiledSliceDims(rootOpOperand, loopDims);
  if (tiledSliceDims.empty())
    return failure();

  // Tile the producer operands and clone the producer in place of `sliceOp`.
  LinalgOp clonedOp =
      getTiledProducer(b, producerResult, sliceOp, tiledSliceDims, iterArg);

  // Cast the `clonedOp` result to gap type mismatches before canonicalization.
  Type consumerOperandType = rootOpOperand->get().getType();
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
  return loopOps.front()->getOpResults();
}

//===----------------------------------------------------------------------===//
// Tile and fuse entry-points.
//===----------------------------------------------------------------------===//

FailureOr<TileLoopNest>
mlir::linalg::tileConsumerAndFuseProducers(OpBuilder &b, LinalgOp consumerOp,
                                           ArrayRef<int64_t> tileSizes,
                                           ArrayRef<int64_t> tileInterchange) {
  assert(tileSizes.size() == tileInterchange.size() &&
         "expect the number of tile sizes and interchange dims to match");
  assert(isPermutation(tileInterchange) &&
         "expect tile interchange is a permutation");

  // Create an empty tile loop nest.
  TileLoopNest tileLoopNest(consumerOp);

  // Search the number of outer parallel loops to separate them from possible
  // inner reduction dimensions.
  SmallVector<StringAttr> iterTypes =
      llvm::to_vector<6>(consumerOp.iterator_types().getAsRange<StringAttr>());
  applyPermutationToVector(iterTypes, tileInterchange);
  auto *it = find_if(iterTypes, [&](StringAttr iterType) {
    return !isParallelIterator(iterType);
  });
  int64_t split = std::distance(iterTypes.begin(), it);

  // Tile the outer parallel loops and fuse the output operands.
  SmallVector<int64_t> outerTileSizes;
  outerTileSizes.append(tileSizes.begin(), tileSizes.begin() + split);
  outerTileSizes.append(tileSizes.size() - split, 0);
  if (failed(tileLoopNest.tileRootOp(b, outerTileSizes, tileInterchange)))
    return failure();
  for (OpOperand *opOperand : tileLoopNest.getRootOp().getOutputOperands())
    (void)tileLoopNest.fuseProducer(b, opOperand);

  // Tile the remaining loops and fuse the input operands.
  SmallVector<int64_t> innerTileSizes;
  innerTileSizes.append(split, 0);
  innerTileSizes.append(tileSizes.begin() + split, tileSizes.end());
  if (failed(tileLoopNest.tileRootOp(b, innerTileSizes, tileInterchange)))
    return failure();
  SmallVector<OpOperand *> inputOperands =
      tileLoopNest.getRootOp().getInputOperands();
  for (OpOperand *opOperand : tileLoopNest.getRootOp().getInputOperands())
    (void)tileLoopNest.fuseProducer(b, opOperand);

  return tileLoopNest;
}

namespace {
struct LinalgTileAndFuseTensorOps
    : public LinalgTileAndFuseTensorOpsBase<LinalgTileAndFuseTensorOps> {

  void notifyFailure(StringRef message) {
    llvm::errs() << " - LinalgTileAndFuseTensorOps: " << message << "\n";
    signalPassFailure();
  }

  void runOnFunction() override {
    FuncOp funcOp = getFunction();
    OpBuilder b(funcOp.getContext());

    // Heuristic to find a good operation to tile and start fusion. Walk all
    // operations and select the one with the maximal backward slice of fusion
    // candidates.
    LinalgOp rootOp = nullptr;
    int64_t numFusionCandidates = -1;
    funcOp.walk([&](LinalgOp linalgOp) {
      SetVector<Operation *> backwardSlice;
      getBackwardSlice(linalgOp, &backwardSlice);
      int64_t backwardSliceSize = count_if(
          backwardSlice, [](Operation *op) { return isa<LinalgOp>(op); });
      if (backwardSliceSize > numFusionCandidates) {
        rootOp = linalgOp;
        numFusionCandidates = backwardSliceSize;
      }
    });
    if (!rootOp)
      return notifyFailure("expect to find a root operation");

    // Check `tileSizes` contains a tile size for every `rootOp` loop dimension.
    if (tileSizes.size() < rootOp.getNumLoops())
      return notifyFailure("expect #tile sizes >= #loops");

    // Check `tileInterchange` contains no entries or as many as `tileSizes`.
    if (!tileInterchange.empty() &&
        tileInterchange.size() != tileSizes.size()) {
      return notifyFailure(
          "expect the number of tile sizes and interchange dims to match");
    }

    // Copy the `tileSizes` and `tileInterchange` prefixes needed to tile
    // `rootOp` or use the identity interchange if `tileInterchange` is empty.
    SmallVector<int64_t> rootTileSizes(
        tileSizes.begin(), tileSizes.begin() + rootOp.getNumLoops());
    SmallVector<int64_t> rootInterchange =
        tileInterchange.empty()
            ? llvm::to_vector<6>(llvm::seq<int64_t>(0, rootOp.getNumLoops()))
            : SmallVector<int64_t>(tileInterchange.begin(),
                                   tileInterchange.begin() +
                                       rootOp.getNumLoops());

    // Check `rootInterchange` is a permutation of the `rootOp` loop dimensions.
    // It has to be a permutation since the tiling cannot tile the same loop
    // dimension multiple times.
    if (!isPermutation(rootInterchange))
      return notifyFailure(
          "expect the tile interchange permutes the root loops");

    // Tile `rootOp` and fuse its producers.
    FailureOr<TileLoopNest> tileLoopNest =
        tileConsumerAndFuseProducers(b, rootOp, rootTileSizes, rootInterchange);
    if (failed(tileLoopNest))
      return notifyFailure("tileConsumerAndFuseProducers failed unexpectedly");

    // Replace all uses of the tiled loop operation.
    rootOp->replaceAllUsesWith(tileLoopNest->getRootOpReplacementResults());
  }
};
} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::createLinalgTileAndFuseTensorOpsPass() {
  return std::make_unique<LinalgTileAndFuseTensorOps>();
}
