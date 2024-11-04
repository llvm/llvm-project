//===- Tiling.cpp - Implementation of tiling using TilingInterface -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the tiling using TilingInterface.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "llvm/Support/Debug.h"
#include <optional>

#define DEBUG_TYPE "tile-using-interface"

using namespace mlir;

scf::SCFTilingOptions &
scf::SCFTilingOptions::setTileSizes(ArrayRef<OpFoldResult> ts) {
  assert(!tileSizeComputationFunction && "tile sizes already set");
  auto tileSizes = llvm::to_vector(ts);
  tileSizeComputationFunction = [tileSizes](OpBuilder &b, Operation *op) {
    return tileSizes;
  };
  return *this;
}

/// Helper method to adjust the interchange vector to match the iteration
/// domain.
static SmallVector<int64_t>
fillInterchangeVector(ArrayRef<int64_t> interchangeVector,
                      size_t iterationDomainSize) {
  SmallVector<int64_t> filledVector = llvm::to_vector(interchangeVector);
  if (filledVector.size() < iterationDomainSize) {
    auto range = llvm::seq<int64_t>(filledVector.size(), iterationDomainSize);
    filledVector.append(range.begin(), range.end());
  }
  if (filledVector.size() > iterationDomainSize)
    filledVector.resize(iterationDomainSize);
  return filledVector;
}

/// Convert a list of ops of type `SrcOpTy` to list of `Operation *`.
template <typename SrcOpTy>
static SmallVector<Operation *> getAsOperations(ArrayRef<SrcOpTy> ops) {
  return llvm::to_vector(
      llvm::map_range(ops, [](auto op) -> Operation * { return op; }));
}
template <typename SrcOpTy>
static SmallVector<Operation *>
getAsOperations(const SmallVector<SrcOpTy> &ops) {
  return getAsOperations(ArrayRef<SrcOpTy>(ops));
}

/// Convert a list of `Operation *` to a list of `DstOpTy.
template <typename DstOpTy>
static SmallVector<DstOpTy> castToTypedOperations(ArrayRef<Operation *> ops) {
  return llvm::to_vector(
      llvm::map_range(ops, [](Operation *op) { return cast<DstOpTy>(op); }));
}
template <typename DstOpTy>
static SmallVector<DstOpTy>
castToTypedOperations(const SmallVector<Operation *> &ops) {
  return castToTypedOperations<DstOpTy>(ArrayRef<Operation *>(ops));
}

//===----------------------------------------------------------------------===//
// tileUsingSCFForOp implementation.
//===----------------------------------------------------------------------===//

// Check if `stride` evenly divides the trip count `size - offset`.
static bool tileDividesIterationDomain(Range loopRange) {
  std::optional<int64_t> offsetAsInt = getConstantIntValue(loopRange.offset);
  if (!offsetAsInt)
    return false;
  std::optional<int64_t> sizeAsInt = getConstantIntValue(loopRange.size);
  if (!sizeAsInt)
    return false;
  std::optional<int64_t> strideAsInt = getConstantIntValue(loopRange.stride);
  if (!strideAsInt)
    return false;
  return ((sizeAsInt.value() - offsetAsInt.value()) % strideAsInt.value() == 0);
}

/// Returns the bounded tile size given the current `iv`, `loopRange` and
/// `tileSize`, i.e., `min(tileSize, range.end() - iv)`.
static OpFoldResult getBoundedTileSize(OpBuilder &b, Location loc,
                                       Range loopRange, Value iv,
                                       OpFoldResult tileSize) {
  std::optional<int64_t> ts = getConstantIntValue(tileSize);
  if (ts && ts.value() == 1)
    return tileSize;

  if (tileDividesIterationDomain(
          Range{loopRange.offset, loopRange.size, tileSize}))
    return tileSize;

  // The tile size to use (to avoid out of bounds access) is  minimum of
  // `tileSize` and `ub - iv`, where `iv` is the induction variable of the tiled
  // loop.
  AffineExpr s0, s1, d0;
  bindDims(b.getContext(), d0);
  bindSymbols(b.getContext(), s0, s1);
  AffineMap minMap = AffineMap::get(1, 2, {s0, s1 - d0}, b.getContext());
  Value size = getValueOrCreateConstantIndexOp(b, loc, loopRange.size);
  return affine::makeComposedFoldedAffineMin(
      b, loc, minMap, SmallVector<OpFoldResult>{iv, tileSize, size});
}

/// Clones the operation and updates the destination if the operation
/// implements the `DestinationStyleOpInterface`.
static Operation *cloneOpAndUpdateDestinationArgs(RewriterBase &rewriter,
                                                  Operation *op,
                                                  ValueRange newDestArgs) {
  Operation *clonedOp = rewriter.clone(*op);
  if (newDestArgs.empty())
    return clonedOp;
  if (auto destinationStyleOp = dyn_cast<DestinationStyleOpInterface>(clonedOp))
    destinationStyleOp.getDpsInitsMutable().assign(newDestArgs);
  return clonedOp;
}

/// Generate an empty loop nest that represents the tiled loop nest shell.
/// - `loopRanges` specifies the lb, ub and step of the untiled iteration space.
/// - `tileSizes` is the tile sizes to use. Zero represent untiled loops.
/// - In `offsets` and `sizes` return the multi-dimensional offset and size of
///   the tile processed within the inner most loop.
/// Note that this methods adds `scf.yield` operation for all but the innermost
/// loop. These yield the value returned by the immediately inner loop. The
/// caller is expected to add the scf.yield operation for the innermost loop.
static SmallVector<scf::ForOp> generateTileLoopNest(
    OpBuilder &builder, Location loc, ArrayRef<Range> loopRanges,
    ArrayRef<OpFoldResult> tileSizes, SmallVector<OpFoldResult> &offsets,
    SmallVector<OpFoldResult> &sizes, ValueRange destinationTensors = {}) {
  if (loopRanges.empty())
    return {};
  assert(loopRanges.size() == tileSizes.size() &&
         "expected as many tile sizes as loop ranges");
  OpBuilder::InsertionGuard guard(builder);
  SmallVector<scf::ForOp> loops;
  offsets.resize(loopRanges.size());
  sizes.resize(loopRanges.size());

  for (auto loopRange : llvm::enumerate(loopRanges)) {
    Value offset =
        getValueOrCreateConstantIndexOp(builder, loc, loopRange.value().offset);
    Value size =
        getValueOrCreateConstantIndexOp(builder, loc, loopRange.value().size);
    Value tileSize = getValueOrCreateConstantIndexOp(
        builder, loc, tileSizes[loopRange.index()]);
    // No loops if tile size is zero. Set offset and size to the loop
    // offset and size.
    if (matchPattern(tileSize, m_Zero())) {
      offsets[loopRange.index()] = offset;
      sizes[loopRange.index()] = size;
      continue;
    }

    auto loop = builder.create<scf::ForOp>(
        loc, offset, size, tileSize, destinationTensors,
        [&](OpBuilder &bodyBuilder, Location bodyLoc, Value iv,
            ValueRange /*iterArgs*/) {
          sizes[loopRange.index()] =
              getBoundedTileSize(bodyBuilder, bodyLoc, loopRange.value(), iv,
                                 getAsOpFoldResult(tileSize));
        });
    offsets[loopRange.index()] = loop.getInductionVar();
    loops.push_back(loop);
    builder.setInsertionPointToEnd(loop.getBody());
    destinationTensors = loop.getRegionIterArgs();
  }

  // Add the scf.yield operations for all the outer loops.
  if (!loops.empty()) {
    for (auto [outerLoop, innerLoop] :
         llvm::zip_equal(MutableArrayRef(loops).drop_back(),
                         MutableArrayRef(loops).drop_front())) {
      builder.setInsertionPointToEnd(outerLoop.getBody());
      builder.create<scf::YieldOp>(outerLoop.getLoc(), innerLoop.getResults());
    }
  }
  return loops;
}

/// Method to add new init values to a loop nest. Updates `loops` in-place with
/// new loops that use the `newInitValues`.
/// The outer-loops are updated to yield the new result values of the inner
/// loop. For the innermost loop, the call back `getNewYields` is invoked to get
/// the additional values to yield form the innermost loop.
static void addInitOperandsToLoopNest(
    RewriterBase &rewriter, MutableArrayRef<scf::ForOp> loops,
    ValueRange newInitValues,
    llvm::function_ref<SmallVector<Value>(RewriterBase &rewriter, Value iv,
                                          ValueRange newRegionIterArgs)>
        getNewYieldValsFn) {
  SmallVector<scf::ForOp> newLoops;
  if (loops.empty())
    return;
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(loops.front());
  for (auto &loop : loops) {
    rewriter.setInsertionPoint(loop);

    // Create a new loop with the new init values for this loop.
    SmallVector<Value> newInits = llvm::to_vector(loop.getInitArgs());
    newInits.append(newInitValues.begin(), newInitValues.end());
    auto newLoop = rewriter.create<scf::ForOp>(
        loop.getLoc(), loop.getLowerBound(), loop.getUpperBound(),
        loop.getStep(), newInits,
        [&](OpBuilder &b, Location loc, Value iv, ValueRange iterArgs) {});

    // Merge the body of the new loop with the body of the old loops.
    SmallVector<Value> sourceBlockArgs;
    sourceBlockArgs.push_back(newLoop.getInductionVar());
    auto newRegionIterArgs = newLoop.getRegionIterArgs();
    sourceBlockArgs.append(
        newRegionIterArgs.begin(),
        std::next(newRegionIterArgs.begin(), loop.getNumResults()));
    rewriter.mergeBlocks(loop.getBody(), newLoop.getBody(), sourceBlockArgs);
    rewriter.replaceOp(loop,
                       newLoop.getResults().take_front(loop.getNumResults()));
    loop = newLoop;
    newInitValues = newLoop.getRegionIterArgs().take_back(newInitValues.size());
  }

  // Update the loop body of the innermost loop to get new yield values.
  scf::ForOp innerMostLoop = loops.back();
  auto innerMostYieldOp =
      cast<scf::YieldOp>(innerMostLoop.getBody()->getTerminator());
  rewriter.setInsertionPoint(innerMostYieldOp);
  SmallVector<Value> newYieldVals =
      getNewYieldValsFn(rewriter, innerMostLoop.getInductionVar(),
                        innerMostLoop.getRegionIterArgs());
  SmallVector<Value> newYieldOperands =
      llvm::to_vector(innerMostYieldOp->getOperands());
  newYieldOperands.append(newYieldVals);
  rewriter.replaceOpWithNewOp<scf::YieldOp>(innerMostYieldOp, newYieldOperands);

  // Make all other loops except the innermost loops yield the values returned
  // by the inner loop.
  for (auto [outerLoop, innerLoop] :
       llvm::zip_equal(loops.drop_back(), loops.drop_front())) {
    auto outerLoopYield =
        cast<scf::YieldOp>(outerLoop.getBody()->getTerminator());
    SmallVector<Value> newYields =
        llvm::to_vector(outerLoopYield.getOperands());
    ValueRange additionalYields =
        innerLoop.getResults().take_back(newInitValues.size());
    newYields.append(additionalYields.begin(), additionalYields.end());
    rewriter.setInsertionPoint(outerLoopYield);
    rewriter.replaceOpWithNewOp<scf::YieldOp>(outerLoopYield, newYields);
  }
}

/// Implementation of tiling transformation of `op` that implements the
/// `TilingInterface` using `scf.for` to iterate over the tiles.
FailureOr<scf::SCFTilingResult>
mlir::scf::tileUsingSCFForOp(RewriterBase &rewriter, TilingInterface op,
                             const scf::SCFTilingOptions &options) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointAfter(op);

  if (!options.tileSizeComputationFunction) {
    return rewriter.notifyMatchFailure(
        op, "missing tile size computation function");
  }

  // 1. Get the range of the loops that are represented by the operation.
  SmallVector<Range> iterationDomain = op.getIterationDomain(rewriter);
  size_t numLoops = iterationDomain.size();
  if (numLoops == 0) {
    return rewriter.notifyMatchFailure(
        op, "unable to tile op with no iteration domain");
  }
  // 2. Materialize the tile sizes. Enforce the convention that "tiling by zero"
  // skips tiling a particular dimension. This convention is significantly
  // simpler to handle instead of adjusting affine maps to account for missing
  // dimensions.
  SmallVector<OpFoldResult> tileSizeVector =
      options.tileSizeComputationFunction(rewriter, op);
  if (tileSizeVector.size() < iterationDomain.size()) {
    auto zero = rewriter.getIndexAttr(0);
    tileSizeVector.append(numLoops - tileSizeVector.size(), zero);
  }

  // 3. Find the destination tensors to use for the operation.
  SmallVector<Value> destinationTensors;
  if (failed(tensor::getOrCreateDestinations(rewriter, op.getLoc(), op,
                                             destinationTensors))) {
    return rewriter.notifyMatchFailure(op,
                                       "unable to create destination tensors");
  }

  SmallVector<OpFoldResult> offsets, sizes;
  SmallVector<scf::ForOp> forLoops;
  {
    // If there is an interchange specified, permute the iteration domain and
    // the tile sizes.
    SmallVector<int64_t> interchangeVector;
    if (!options.interchangeVector.empty()) {
      interchangeVector = fillInterchangeVector(options.interchangeVector,
                                                iterationDomain.size());
    }
    if (!interchangeVector.empty()) {
      if (!isPermutationVector(interchangeVector)) {
        return rewriter.notifyMatchFailure(
            op, "invalid intechange vector, not a permutation of the entire "
                "iteration space");
      }

      applyPermutationToVector(iterationDomain, interchangeVector);
      applyPermutationToVector(tileSizeVector, interchangeVector);
    }

    // 4. Materialize an empty loop nest that iterates over the tiles. These
    // loops for now do not return any values even if the original operation has
    // results.
    forLoops = generateTileLoopNest(rewriter, op.getLoc(), iterationDomain,
                                    tileSizeVector, offsets, sizes,
                                    destinationTensors);

    if (!interchangeVector.empty()) {
      auto inversePermutation = invertPermutationVector(interchangeVector);
      applyPermutationToVector(offsets, inversePermutation);
      applyPermutationToVector(sizes, inversePermutation);
    }
  }

  LLVM_DEBUG({
    if (!forLoops.empty()) {
      llvm::dbgs() << "LoopNest shell :\n";
      forLoops.front().dump();
      llvm::dbgs() << "\n";
    }
  });

  // 5. Generate the tiled implementation within the inner most loop.
  SmallVector<Value> clonedOpDestination = destinationTensors;
  if (!forLoops.empty()) {
    rewriter.setInsertionPointToEnd(forLoops.back().getBody());
    clonedOpDestination =
        llvm::map_to_vector(forLoops.back().getRegionIterArgs(),
                            [](BlockArgument b) -> Value { return b; });
  }

  // 5a. Clone the operation within the loop body.
  auto clonedOp = cast<TilingInterface>(
      cloneOpAndUpdateDestinationArgs(rewriter, op, clonedOpDestination));

  // 5b. Tile the cloned operation.
  FailureOr<TilingResult> tiledImplementation =
      clonedOp.getTiledImplementation(rewriter, offsets, sizes);
  if (failed(tiledImplementation)) {
    return rewriter.notifyMatchFailure(op, "failed to tile operation");
  }

  // 5c. Delete the cloned operation.
  rewriter.eraseOp(clonedOp);

  // If loops are empty, the tiled op is used as the replacement for the untiled
  // op.
  if (forLoops.empty()) {
    return scf::SCFTilingResult{tiledImplementation->tiledOps,
                                getAsOperations(forLoops),
                                tiledImplementation->tiledValues};
  }

  if (op->getNumResults() == 0) {
    // The innermost loop does not have a `scf.yield` yet. There is nothing to
    // return, so generate an empty `scf.yield` operation.
    rewriter.setInsertionPointToEnd(forLoops.back().getBody());
    rewriter.create<scf::YieldOp>(op->getLoc());
    return scf::SCFTilingResult{
        tiledImplementation->tiledOps, getAsOperations(forLoops), {}};
  }

  // 6. Yield all the results of the tiled operation.
  int64_t numResults = op->getNumResults();
  SmallVector<SmallVector<OpFoldResult>> resultOffsetsList(numResults),
      resultSizesList(numResults);
  SmallVector<Value> yieldedValues;
  for (auto [index, tiledValue] :
       llvm::enumerate(tiledImplementation->tiledValues)) {
    SmallVector<OpFoldResult> resultOffsets, resultSizes;
    if (failed(op.getResultTilePosition(rewriter, index, offsets, sizes,
                                        resultOffsets, resultSizes))) {
      return rewriter.notifyMatchFailure(
          op, "failed to get slice of result produced");
    }
    SmallVector<OpFoldResult> resultStrides(resultOffsets.size(),
                                            rewriter.getIndexAttr(1));
    auto insertSlice = rewriter.create<tensor::InsertSliceOp>(
        op->getLoc(), tiledValue, clonedOpDestination[index], resultOffsets,
        resultSizes, resultStrides);
    yieldedValues.push_back(insertSlice);
  }
  rewriter.create<scf::YieldOp>(op->getLoc(), yieldedValues);

  SmallVector<Value> replacements = llvm::map_to_vector(
      forLoops.front().getResults(), [](OpResult r) -> Value { return r; });
  LLVM_DEBUG({
    if (!forLoops.empty()) {
      llvm::dbgs() << "After tiled implementation :\n";
      forLoops.front().dump();
      llvm::dbgs() << "\n";
    }
  });
  return scf::SCFTilingResult{tiledImplementation->tiledOps,
                              getAsOperations(forLoops), replacements};
}

FailureOr<scf::SCFReductionTilingResult>
mlir::scf::tileReductionUsingScf(RewriterBase &b,
                                 PartialReductionOpInterface op,
                                 ArrayRef<OpFoldResult> tileSizes) {
  Location loc = op.getLoc();
  // Ops implementing PartialReductionOpInterface are expected to implement
  // TilingInterface.
  auto tilingInterfaceOp = cast<TilingInterface>(op.getOperation());
  SmallVector<Range> iterationDomain = tilingInterfaceOp.getIterationDomain(b);
  auto tileSizesVector = llvm::to_vector(tileSizes);
  if (tileSizesVector.size() < iterationDomain.size()) {
    auto zero = b.getIndexAttr(0);
    tileSizesVector.append(iterationDomain.size() - tileSizesVector.size(),
                           zero);
  }
  if (op->getNumResults() != 1)
    return b.notifyMatchFailure(
        op, "don't support ops with multiple results for now");
  SmallVector<utils::IteratorType> iterators =
      tilingInterfaceOp.getLoopIteratorTypes();

  SmallVector<int> reductionDims;
  for (auto [idx, iteratorType] :
       llvm::enumerate(tilingInterfaceOp.getLoopIteratorTypes())) {
    if (iteratorType == utils::IteratorType::reduction)
      reductionDims.push_back(idx);
  }

  // 2. create the inital tensor value.
  FailureOr<Operation *> identityTensor =
      op.generateInitialTensorForPartialReduction(b, loc, tileSizesVector,
                                                  reductionDims);
  if (failed(identityTensor))
    return b.notifyMatchFailure(op,
                                "cannot create a tensor of identity value.");
  // 3. Create the nested loops.
  SmallVector<OpFoldResult> offsets, sizes;
  SmallVector<scf::ForOp> loops =
      generateTileLoopNest(b, loc, iterationDomain, tileSizesVector, offsets,
                           sizes, identityTensor.value()->getResults());

  // 4. Generate the tiled implementation within the inner most loop.
  // 4a. Clone the operation within the loop body.
  SmallVector<Value> clonedOpDestination =
      llvm::map_to_vector(identityTensor.value()->getResults(),
                          [](OpResult res) -> Value { return res; });
  if (!loops.empty()) {
    b.setInsertionPointToEnd(loops.back().getBody());
    clonedOpDestination =
        llvm::map_to_vector(loops.back().getRegionIterArgs(),
                            [](BlockArgument b) -> Value { return b; });
  }
  auto clonedOp = cast<PartialReductionOpInterface>(
      cloneOpAndUpdateDestinationArgs(b, op, clonedOpDestination));

  // 4b. Tile the cloned operation.
  Operation *parallelOp = clonedOp.tileToPartialReduction(
      b, loc, clonedOpDestination, offsets, sizes, reductionDims);
  // 4c. Delete the cloned operation.
  b.eraseOp(clonedOp);

  SmallVector<OpFoldResult> outSizes;
  for (size_t i = 0; i < offsets.size(); i++) {
    outSizes.push_back(
        tensor::getMixedSize(b, loc, parallelOp->getResult(0), i));
  }
  SmallVector<OpFoldResult> outOffsets(offsets.size(), b.getIndexAttr(0));
  SmallVector<OpFoldResult> outStrides(outOffsets.size(), b.getIndexAttr(1));
  SmallVector<Value> yieldedVals;
  auto bbArgs = loops.back().getRegionIterArgs();
  for (auto [result, bbArg] : llvm::zip(parallelOp->getResults(), bbArgs)) {
    Value insert = b.create<tensor::InsertSliceOp>(
        loc, result, bbArg, outOffsets, outSizes, outStrides);
    yieldedVals.push_back(insert);
  }
  b.create<scf::YieldOp>(loc, yieldedVals);

  SmallVector<Value> replacements = llvm::map_to_vector(
      loops.front().getResults(), [](OpResult r) -> Value { return r; });

  // 5. Apply the merge reduction to combine all the partial values.
  b.setInsertionPointAfter(*loops.begin());
  Operation *mergeOp = op.mergeReductions(b, loc, replacements, reductionDims);
  b.replaceOp(op, mergeOp->getResults());

  SCFReductionTilingResult results;
  results.initialOp = *identityTensor;
  results.loops = std::move(loops);
  results.parallelTiledOp = parallelOp;
  results.mergeOp = mergeOp;
  return results;
}

//===----------------------------------------------------------------------===//
// tileConsumerAndFuseProducerGreedilyUsingSCFForOp implementation.
//===----------------------------------------------------------------------===//

/// Return the untiled producer whose slice is used in a tiled consumer. The
/// method traverses the tile loop nest (`loops`) if needed, and returns the
/// `iter_args` of the outer most that is encountered. Traversing the iter_args
/// indicates that this is a destination operand of the consumer. If there was
/// no loop traversal needed, the second value of the returned tuple is empty.
static std::tuple<OpResult, std::optional<OpOperand *>>
getUntiledProducerFromSliceSource(OpOperand *source,
                                  ArrayRef<scf::ForOp> loops) {
  std::optional<OpOperand *> destinationIterArg;
  auto loopIt = loops.rbegin();
  while (auto iterArg = dyn_cast<BlockArgument>(source->get())) {
    scf::ForOp loop = *loopIt;
    if (iterArg.getOwner()->getParentOp() != loop)
      break;
    source = loop.getTiedLoopInit(iterArg);
    loopIt++;
  }
  if (loopIt == loops.rend())
    destinationIterArg = source;
  return {dyn_cast<OpResult>(source->get()), destinationIterArg};
}

/// Implementation of fusing producer of a single slice by computing the
/// slice of the producer in-place.
std::optional<scf::SCFFuseProducerOfSliceResult>
mlir::scf::tileAndFuseProducerOfSlice(RewriterBase &rewriter,
                                      tensor::ExtractSliceOp candidateSliceOp,
                                      MutableArrayRef<scf::ForOp> loops) {
  // 1. Get the producer of the source (potentially walking through
  // `iter_args` of nested `scf.for`)
  auto [fusableProducer, destinationInitArg] =
      getUntiledProducerFromSliceSource(&candidateSliceOp.getSourceMutable(),
                                        loops);
  if (!fusableProducer)
    return std::nullopt;
  unsigned resultNumber = fusableProducer.getResultNumber();

  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(candidateSliceOp);

  // 2. Clone the fused producer
  // 2a. Compute the destination operands to use for the cloned operation.
  SmallVector<Value> origDestinationTensors, clonedOpDestinationTensors;
  Operation *fusableProducerOp = fusableProducer.getOwner();
  if (isa<DestinationStyleOpInterface>(fusableProducerOp) &&
      failed(tensor::getOrCreateDestinations(
          rewriter, fusableProducerOp->getLoc(), fusableProducerOp,
          origDestinationTensors)))
    return std::nullopt;

  clonedOpDestinationTensors = origDestinationTensors;
  if (destinationInitArg &&
      isa<DestinationStyleOpInterface>(fusableProducerOp)) {
    // 2b. If the producer is also destination style, then to maintain the
    // destination passing style, update the destination of the producer to be
    // the source of the slice.
    clonedOpDestinationTensors[resultNumber] = candidateSliceOp.getSource();
  }
  // 2c. Clone the fused producer.
  Operation *clonedProducerOp = cloneOpAndUpdateDestinationArgs(
      rewriter, fusableProducerOp, clonedOpDestinationTensors);
  // 2d. Update the source of the candidateSlice to be the cloned producer.
  //     Easier to just clone the slice with different source since replacements
  //     and DCE of cloned ops becomes easier
  SmallVector<Value> candidateSliceOpOperands =
      llvm::to_vector(candidateSliceOp->getOperands());
  candidateSliceOpOperands[0] = clonedProducerOp->getResult(resultNumber);
  tensor::ExtractSliceOp clonedCandidateSliceOp =
      mlir::clone(rewriter, candidateSliceOp,
                  candidateSliceOp->getResultTypes(), candidateSliceOpOperands);

  // 3. Generate the tiled implementation of the producer of the source
  FailureOr<TilingResult> tileAndFuseResult =
      tensor::replaceExtractSliceWithTiledProducer(
          rewriter, clonedCandidateSliceOp,
          clonedProducerOp->getResult(resultNumber));
  if (failed(tileAndFuseResult))
    return std::nullopt;
  // Note: Do not delete the candidateSliceOp, since its passed in from the
  // caller.
  rewriter.replaceAllUsesWith(candidateSliceOp,
                              tileAndFuseResult->tiledValues[0]);
  rewriter.eraseOp(clonedCandidateSliceOp);
  rewriter.eraseOp(clonedProducerOp);

  // 3. If the slice is for a destination operand, for example,
  //
  // ```mlir
  // %0 = linalg.init
  // %1 = linalg.fill .. outs(%0 : )
  // %2 = scf.for .. iter_args(%arg0 = %1) {
  //   %3 = scf.for .. iter_args(%arg1 = %arg0) {
  //     %4 = tensor.extract_slice %arg1 [..]
  //     .. = linalg.matmul .. outs(%4 : )
  //   }
  // }
  // ```
  //
  // the IR is currently
  //
  // ```
  // %0 = linalg.init
  // %1 = linalg.fill
  // %2 = scf.for .. iter_args(%arg0 = %1 /* incorrect value */ ) {
  //   %3 = scf.for .. iter_args(%arg1 = %arg0) {
  //     %4 = tensor.extract_slice %arg1[..]
  //     %5 = linalg.fill .. outs(%4 : )
  //     .. = linalg.matmul .. outs(%5 : )
  //   }
  // }
  // ```
  //
  // The untiled `linalg.fill` is still used as the `init_value` since it
  // was originally a destination operand of the untiled `linalg.matmul`.
  // When fusing an operand that is a destination operand, the iter_arg of
  // the outer most loop should be changed to use the destination of the
  // fused operation. With this the IR will be.
  //
  // ```
  // %0 = linalg.init
  // %1 = scf.for .. iter_args(%arg0 = %0 /* corrected value */ ) {
  //   %2 = scf.for .. iter_args(%arg1 = %arg0) {
  //     %3 = tensor.extract_slice %arg1[..]
  //     %4 = linalg.fill .. outs(%3 : )
  //     .. = linalg.matmul .. outs(%4 : )
  //   }
  // }
  // ```
  if (destinationInitArg &&
      isa<DestinationStyleOpInterface>(fusableProducerOp) && !loops.empty()) {
    loops.front()
        ->getOpOperands()[destinationInitArg.value()->getOperandNumber()]
        .set(origDestinationTensors[resultNumber]);
  }
  return scf::SCFFuseProducerOfSliceResult{fusableProducer,
                                           tileAndFuseResult->tiledValues[0],
                                           tileAndFuseResult->tiledOps};
}

/// Reconstruct the fused producer from within the tiled-and-fused code.
void mlir::scf::yieldReplacementForFusedProducer(
    RewriterBase &rewriter, tensor::ExtractSliceOp sliceOp,
    scf::SCFFuseProducerOfSliceResult fusedProducerInfo,
    MutableArrayRef<scf::ForOp> loops) {
  if (loops.empty())
    return;

  OpResult fusableProducer = fusedProducerInfo.origProducer;
  Value tiledAndFusedProducer = fusedProducerInfo.tiledAndFusedProducer;
  FailureOr<Value> initValue = tensor::getOrCreateDestination(
      rewriter, fusableProducer.getOwner()->getLoc(), fusableProducer);
  if (succeeded(initValue)) {

    auto newYieldValuesFn =
        [&](RewriterBase &innerRewriter, Value iv,
            ValueRange newRegionIterArgs) -> SmallVector<Value> {
      OpBuilder::InsertionGuard g(innerRewriter);
      if (auto tiledDestStyleOp =
              tiledAndFusedProducer
                  .getDefiningOp<DestinationStyleOpInterface>()) {
        rewriter.setInsertionPoint(tiledDestStyleOp);
        BlockArgument newRegionArg = loops.back().getRegionIterArgs().back();
        auto destSlice = rewriter.create<tensor::ExtractSliceOp>(
            sliceOp.getLoc(), newRegionArg, sliceOp.getMixedOffsets(),
            sliceOp.getMixedSizes(), sliceOp.getMixedStrides());
        unsigned resultNumber = fusableProducer.getResultNumber();
        rewriter.updateRootInPlace(tiledDestStyleOp, [&]() {
          tiledDestStyleOp.getDpsInitsMutable()[resultNumber].set(destSlice);
        });
      }
      Block *block = rewriter.getInsertionPoint()->getBlock();
      rewriter.setInsertionPoint(block->getTerminator());
      Value replacement = rewriter.create<tensor::InsertSliceOp>(
          fusedProducerInfo.origProducer.getLoc(),
          fusedProducerInfo.tiledAndFusedProducer,
          loops.back().getRegionIterArgs().back(), sliceOp.getMixedOffsets(),
          sliceOp.getMixedSizes(), sliceOp.getMixedStrides());
      return {replacement};
    };

    addInitOperandsToLoopNest(rewriter, loops,
                              SmallVector<Value>{initValue.value()},
                              newYieldValuesFn);
  }
}

/// Implementation of tile consumer and fuse producer greedily.
FailureOr<scf::SCFTileAndFuseResult>
mlir::scf::tileConsumerAndFuseProducerGreedilyUsingSCFForOp(
    RewriterBase &rewriter, TilingInterface consumer,
    const scf::SCFTileAndFuseOptions &options) {
  // This transformation is only valid for ops that return values (i.e. not
  // valid to use with operations that have memref operands).
  if (!consumer->getNumResults()) {
    return rewriter.notifyMatchFailure(
        consumer, "invalid pattern for op with no results");
  }

  // 1. First tile the consumer.
  SmallVector<scf::ForOp> forLoops;
  SetVector<Operation *> fusedProducers, tiledAndFusedOps;
  DenseMap<Value, Value> replacements;
  llvm::SmallDenseMap<Value, int64_t> yieldedValueToResultNumber;
  {
    FailureOr<scf::SCFTilingResult> tilingResult =
        tileUsingSCFForOp(rewriter, consumer, options.tilingOptions);
    if (failed(tilingResult))
      return rewriter.notifyMatchFailure(consumer, "failed to tile consumer");
    for (auto *tiledOp : tilingResult->tiledOps)
      tiledAndFusedOps.insert(tiledOp);
    forLoops = castToTypedOperations<scf::ForOp>(tilingResult->loops);
    for (auto [index, origValue, replacement] :
         llvm::enumerate(consumer->getResults(), tilingResult->replacements)) {
      replacements[origValue] = replacement;
      yieldedValueToResultNumber[tilingResult->tiledOps.back()->getResult(
          index)] = index;
    }
  }

  // If there are no loops generated, fusion is immaterial.
  if (forLoops.empty()) {
    return scf::SCFTileAndFuseResult{fusedProducers, tiledAndFusedOps,
                                     getAsOperations(forLoops), replacements};
  }

  // 2. Typically, the operands of the tiled operation are slices of the
  //    operands of the untiled operation. These are expressed in IR using
  //    `tensor.extract_slice` operations with source being the operands of the
  //    untiled operation. Create a worklist of these `tensor.extract_slice`
  //    operations. If the producers of the source of the `tensor.extract_slice`
  //    can be tiled such that the tiled value is generated in-place, that
  //    effectively tiles + fuses the operations.
  auto addCandidateSlices = [](Operation *fusedOp,
                               std::deque<tensor::ExtractSliceOp> &candidates) {
    for (Value operand : fusedOp->getOperands())
      if (auto sliceOp = operand.getDefiningOp<tensor::ExtractSliceOp>())
        candidates.push_back(sliceOp);
  };

  std::deque<tensor::ExtractSliceOp> candidates;
  addCandidateSlices(tiledAndFusedOps.back(), candidates);
  OpBuilder::InsertionGuard g(rewriter);
  while (!candidates.empty()) {
    // Traverse the slices in BFS fashion.
    tensor::ExtractSliceOp candidateSliceOp = candidates.front();
    candidates.pop_front();

    // The operands of the fused producer might themselved be slices of
    // values produced by operations that implement the `TilingInterface`.
    // Add these operations to the worklist.
    std::optional<scf::SCFFuseProducerOfSliceResult> fusedResult =
        tileAndFuseProducerOfSlice(rewriter, candidateSliceOp, forLoops);
    if (!fusedResult)
      continue;

    if (Operation *tiledAndFusedOp =
            fusedResult->tiledAndFusedProducer.getDefiningOp()) {
      fusedProducers.insert(fusedResult->origProducer.getDefiningOp());
      tiledAndFusedOps.insert(tiledAndFusedOp);
      addCandidateSlices(tiledAndFusedOp, candidates);
    }
  }
  return scf::SCFTileAndFuseResult{fusedProducers, tiledAndFusedOps,
                                   getAsOperations(forLoops), replacements};
}

//===----------------------------------------------------------------------===//
// tileUsingSCFForAllOp implementation.
//===----------------------------------------------------------------------===//

FailureOr<scf::SCFTilingResult>
mlir::scf::tileUsingSCFForallOp(RewriterBase &rewriter, TilingInterface op,
                                const scf::SCFTilingOptions &options) {
  Location loc = op->getLoc();
  OpBuilder::InsertionGuard g(rewriter);

  // 1. Get the range of loops that are represented by the operation.
  SmallVector<Range> loopRanges = op.getIterationDomain(rewriter);
  if (loopRanges.empty())
    return op->emitOpError("expected non-empty loop ranges");
  auto hasStrideOne = [](Range r) { return !isConstantIntValue(r.stride, 1); };
  if (llvm::any_of(loopRanges, hasStrideOne))
    return op->emitOpError("only stride-1 supported atm");

  // 2. Get the tile sizes. If tile size is 0, it is not tiled and distributed.
  // To make it easier, pad the tile sizes to loopRanges.size with value 0.
  SmallVector<OpFoldResult> tileSizeVector =
      options.tileSizeComputationFunction(rewriter, op);
  tileSizeVector.resize(loopRanges.size(), rewriter.getIndexAttr(0));

  // 3. Build the offsets, sizes and steps for the tile and distributed loops.
  SmallVector<OpFoldResult> lbs, ubs, steps;
  for (auto [tileSize, loopRange] : llvm::zip(tileSizeVector, loopRanges)) {
    if (isConstantIntValue(tileSize, 0))
      continue;
    lbs.push_back(loopRange.offset);
    ubs.push_back(loopRange.size);
    steps.push_back(tileSize);
  }

  // 4. Gather destination tensors.
  SmallVector<Value> dest;
  if (failed(tensor::getOrCreateDestinations(rewriter, loc, op, dest)))
    return op->emitOpError("failed to get destination tensors");

  // 5. Build the device mapping attribute.
  std::optional<ArrayAttr> mappingAttr;
  if (!options.mappingVector.empty()) {
    mappingAttr = rewriter.getArrayAttr(ArrayRef(options.mappingVector));
  }

  // 6. Create the ForallOp. We don't use the lambda body-builder
  // version because we require the use of RewriterBase in the body, so we
  // manually move the insertion point to the body below.
  auto forallOp =
      rewriter.create<scf::ForallOp>(loc, lbs, ubs, steps, dest, mappingAttr);

  // 7. Get the tile offset and sizes.
  rewriter.setInsertionPoint(forallOp.getTerminator());
  SmallVector<OpFoldResult> tiledOffsets, tiledSizes;
  ValueRange ivs = forallOp.getInductionVars();
  {
    int materializedLoopNum = 0;
    for (auto [tileSize, loopRange] : llvm::zip(tileSizeVector, loopRanges)) {
      if (isConstantIntValue(tileSize, 0)) {
        tiledOffsets.push_back(loopRange.offset);
        tiledSizes.push_back(loopRange.size);
        continue;
      }
      Value iv = ivs[materializedLoopNum++];
      tiledOffsets.push_back(iv);
      tiledSizes.push_back(
          getBoundedTileSize(rewriter, loc, loopRange, iv, tileSize));
    }
  }

  // 8. Tile the operation. Clone the operation to allow fix up of destination
  // operands.
  ArrayRef<BlockArgument> destBbArgs = forallOp.getOutputBlockArguments();
  Operation *clonedOp =
      cloneOpAndUpdateDestinationArgs(rewriter, op, destBbArgs);
  FailureOr<TilingResult> tilingResult =
      cast<TilingInterface>(clonedOp).getTiledImplementation(
          rewriter, tiledOffsets, tiledSizes);
  if (failed(tilingResult))
    return clonedOp->emitError("failed to tile op: ");
  rewriter.eraseOp(clonedOp);

  // 9. Parallel insert back into the result tensor.
  for (auto [index, tiledValue, destBBArg] :
       llvm::enumerate(tilingResult->tiledValues, destBbArgs)) {
    // 9.a. Partial subset information is inserted just before the terminator.
    rewriter.setInsertionPoint(forallOp.getTerminator());

    SmallVector<OpFoldResult> resultOffsets, resultSizes;
    if (failed(op.getResultTilePosition(rewriter, index, tiledOffsets,
                                        tiledSizes, resultOffsets,
                                        resultSizes))) {
      return op->emitOpError("output offsets couldn't be calculated");
    }

    SmallVector<OpFoldResult> strides(resultSizes.size(),
                                      rewriter.getIndexAttr(1));
    // 9.b. Parallel insertions are inserted at the end of the combining
    // terminator.
    rewriter.setInsertionPointToEnd(forallOp.getTerminator().getBody());
    rewriter.create<tensor::ParallelInsertSliceOp>(
        loc, tiledValue, destBBArg, resultOffsets, resultSizes, strides);
  }

  // 10. Return the tiling result.
  return scf::SCFTilingResult{
      tilingResult->tiledOps,
      {forallOp.getOperation()},
      llvm::map_to_vector(forallOp.getResults(),
                          [](auto val) -> Value { return val; })};
}

//===----------------------------------------------------------------------===//
// lowerToLoopsUsingSCFForOp implementation.
//===----------------------------------------------------------------------===//

FailureOr<SmallVector<scf::ForOp>>
mlir::scf::lowerToLoopsUsingSCFForOp(RewriterBase &rewriter,
                                     TilingInterface op) {
  // TODO: Handle cases where the op has results if needed.
  if (op->getNumResults() > 0) {
    return rewriter.notifyMatchFailure(
        op, "unable to lower to loops operations with return values");
  }

  SmallVector<Range> domain = op.getIterationDomain(rewriter);
  SmallVector<Value> ivs;
  SmallVector<scf::ForOp> loops;
  Location loc = op.getLoc();
  for (auto loopRange : domain) {
    Value offsetVal =
        getValueOrCreateConstantIndexOp(rewriter, loc, loopRange.offset);
    Value sizeVal =
        getValueOrCreateConstantIndexOp(rewriter, loc, loopRange.size);
    Value strideVal =
        getValueOrCreateConstantIndexOp(rewriter, loc, loopRange.stride);
    auto loop = rewriter.create<scf::ForOp>(op.getLoc(), offsetVal, sizeVal,
                                            strideVal, ValueRange{});
    loops.push_back(loop);
    ivs.push_back(loop.getInductionVar());
    rewriter.setInsertionPoint(loop.getBody()->getTerminator());
  }
  if (failed(op.generateScalarImplementation(rewriter, op.getLoc(), ivs))) {
    return failure();
  }
  return loops;
}
