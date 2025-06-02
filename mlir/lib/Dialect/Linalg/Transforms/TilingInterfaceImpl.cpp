//===- TilingInterfaceImpl.cpp - Implementation of TilingInterface -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include <optional>

using namespace mlir;
using namespace mlir::linalg;

//===----------------------------------------------------------------------===//
// Utility methods for implementation of Tiling Interface for Linalg ops
//===----------------------------------------------------------------------===//

/// Return the SSA values that represent the data point accessed using a given
/// `indexingMap` for a given point in the iteration space represented by `ivs`.
static SmallVector<Value> getIndicesForAccess(OpBuilder &b, Location loc,
                                              AffineMap indexingMap,
                                              ValueRange ivs) {
  SmallVector<Value> indices;
  indices.reserve(indexingMap.getNumResults());
  for (auto result : indexingMap.getResults()) {
    AffineMap m = AffineMap::get(indexingMap.getNumDims(),
                                 indexingMap.getNumSymbols(), result);
    Value v = b.create<affine::AffineApplyOp>(loc, m, ivs);
    indices.push_back(v);
  }
  return indices;
}

/// Method to inline the payload of a `linalgOp` given the iteration space
/// point and values for the arguments of the payload.
static LogicalResult inlinePayload(OpBuilder &b, LinalgOp linalgOp,
                                   ValueRange ivs, ValueRange argValues) {
  Block *body = linalgOp.getBlock();
  IRMapping map;
  map.map(body->getArguments(), argValues);
  for (auto &op : body->without_terminator()) {
    if (auto indexOp = dyn_cast<IndexOp>(&op)) {
      map.map(indexOp.getResult(), ivs[indexOp.getDim()]);
      continue;
    }
    b.clone(op, map);
  }

  Operation *terminator = body->getTerminator();
  Location loc = terminator->getLoc();
  for (const auto &operand : llvm::enumerate(terminator->getOperands())) {
    Value toStore = map.lookupOrDefault(operand.value());
    OpOperand *storeInto = linalgOp.getDpsInitOperand(operand.index());
    auto indices = getIndicesForAccess(
        b, loc, linalgOp.getMatchingIndexingMap(storeInto), ivs);
    b.create<memref::StoreOp>(
        loc, toStore, linalgOp.getDpsInitOperand(operand.index())->get(),
        indices);
  }
  return success();
}

//===----------------------------------------------------------------------===//
// External Model for implementing `TilingInterface` for `LinalgOp`s.
//===----------------------------------------------------------------------===//

namespace {
/// External model implementation of TilingInterface for LinalgOps. An external
/// model implementation is used for now till the use of `TilingInterface` is
/// on-par with the current Linalg tiling + fusion patterns. Once it is
/// maybe possible to move this into the op-definition (though there are
/// advantages to leaving it as an external model)
template <typename LinalgOpTy>
struct LinalgOpTilingInterface
    : public TilingInterface::ExternalModel<LinalgOpTilingInterface<LinalgOpTy>,
                                            LinalgOpTy> {
  /// Return the loop iterator type.
  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *op) const {
    LinalgOpTy concreteOp = cast<LinalgOpTy>(op);
    return concreteOp.getIteratorTypesArray();
  }

  /// Return the iteration domain range.
  SmallVector<Range> getIterationDomain(Operation *op, OpBuilder &b) const {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPoint(op);
    Location loc = op->getLoc();
    LinalgOp linalgOp = cast<LinalgOp>(op);
    SmallVector<OpFoldResult> allShapesSizes =
        linalgOp.createFlatListOfOperandDims(b, loc);
    AffineMap map = linalgOp.getShapesToLoopsMap();

    return llvm::to_vector(
        llvm::map_range(map.getResults(), [&](AffineExpr loopExpr) {
          OpFoldResult ofr = affine::makeComposedFoldedAffineApply(
              b, loc, loopExpr, allShapesSizes);
          return Range{b.getIndexAttr(0), ofr, b.getIndexAttr(1)};
        }));
  }

  /// Instantiate the tiled implementation of the operation.
  FailureOr<TilingResult>
  getTiledImplementation(Operation *op, OpBuilder &b,
                         ArrayRef<OpFoldResult> offsets,
                         ArrayRef<OpFoldResult> sizes) const {
    // Leave the `sizeBounds` value empty. That is only needed when the `sizes`
    // specified could lead to out of bounds accesses.
    Location loc = op->getLoc();
    LinalgOp linalgOp = cast<LinalgOp>(op);
    SmallVector<Value> valuesToTile = linalgOp->getOperands();
    SmallVector<Value> tiledOperands = makeTiledShapes(
        b, loc, linalgOp, valuesToTile, offsets, sizes, {}, true);
    SmallVector<Operation *> generatedSlices = llvm::map_to_vector(
        llvm::make_filter_range(
            tiledOperands,
            [](Value v) -> bool {
              return isa_and_nonnull<tensor::ExtractSliceOp, memref::SubViewOp>(
                  v.getDefiningOp());
            }),
        [](Value v) -> Operation * { return v.getDefiningOp(); });

    SmallVector<Type> resultTensorTypes =
        getTensorOutputTypes(linalgOp, tiledOperands);

    Operation *tiledOp = clone(b, linalgOp, resultTensorTypes, tiledOperands);
    offsetIndices(b, cast<LinalgOp>(tiledOp), offsets);

    return TilingResult{
        {tiledOp}, SmallVector<Value>(tiledOp->getResults()), generatedSlices};
  }

  /// Utility to fetch the offsets and sizes when applied as per the indexing
  /// map of the linalg op. This helps in fusing the linalg op as a consumer of
  /// a given slice op.
  void
  getMappedOffsetAndSize(LinalgOp linalgOp, OpBuilder &b, AffineMap indexingMap,
                         ArrayRef<OpFoldResult> offsets,
                         ArrayRef<OpFoldResult> sizes,
                         SmallVectorImpl<OpFoldResult> &mappedOffsets,
                         SmallVectorImpl<OpFoldResult> &mappedSizes) const {
    unsigned numLoops = linalgOp.getNumLoops();
    auto tilingInterfaceOp = cast<TilingInterface>(linalgOp.getOperation());
    mappedOffsets.resize(numLoops);
    mappedSizes.resize(numLoops);
    if (!indexingMap.isPermutation()) {
      SmallVector<Range> iterationDomain =
          tilingInterfaceOp.getIterationDomain(b);
      for (const auto &&[index, value] : llvm::enumerate(iterationDomain)) {
        mappedOffsets[index] = value.offset;
        mappedSizes[index] = value.size;
      }
    }
    for (const auto &&[index, value] :
         llvm::enumerate(indexingMap.getResults())) {
      unsigned dimPosition = cast<AffineDimExpr>(value).getPosition();
      mappedOffsets[dimPosition] = offsets[index];
      mappedSizes[dimPosition] = sizes[index];
    }
  }

  /// Method to return the position of the result tile computed by the tiled
  /// operation.
  LogicalResult getIterationDomainTileFromOperandTile(
      Operation *op, OpBuilder &b, unsigned operandNumber,
      ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes,
      SmallVectorImpl<OpFoldResult> &iterDomainOffsets,
      SmallVectorImpl<OpFoldResult> &iterDomainSizes) const {
    auto linalgOp = cast<LinalgOp>(op);

    // Check that the indexing map used for the operand is a projected
    // permutation. This could be relaxed with a more general approach that can
    // map the offsets and sizes from the operand to iteration space tiles
    // (filling in full extent for dimensions not used to access the result).
    AffineMap indexingMap =
        linalgOp.getMatchingIndexingMap(&op->getOpOperand(operandNumber));
    if (!indexingMap.isProjectedPermutation()) {
      return op->emitError()
             << "unhandled get iter domain position when operand is not "
                "accessed using a permuted projection";
    }

    getMappedOffsetAndSize(linalgOp, b, indexingMap, offsets, sizes,
                           iterDomainOffsets, iterDomainSizes);
    return success();
  }

  /// Return the details of the output tile generated by the tiled
  /// implementation.
  LogicalResult
  getResultTilePosition(Operation *op, OpBuilder &b, unsigned resultNumber,
                        ArrayRef<OpFoldResult> offsets,
                        ArrayRef<OpFoldResult> sizes,
                        SmallVector<OpFoldResult> &resultOffsets,
                        SmallVector<OpFoldResult> &resultSizes) const {
    Location loc = op->getLoc();
    LinalgOp linalgOp = cast<LinalgOp>(op);

    AffineExpr d0;
    bindDims(b.getContext(), d0);
    SmallVector<OpFoldResult> subShapeSizes =
        llvm::to_vector(llvm::map_range(sizes, [&](OpFoldResult ofr) {
          return affine::makeComposedFoldedAffineApply(b, loc, d0 - 1, ofr);
        }));

    OpOperand *outOperand = linalgOp.getDpsInitOperand(resultNumber);
    SliceParameters sliceParams = computeSliceParameters(
        b, loc, outOperand->get(), sizes,
        linalgOp.getMatchingIndexingMap(outOperand), offsets,
        /*ubs*/ {}, subShapeSizes, true);
    resultOffsets = sliceParams.offsets;
    resultSizes = sliceParams.sizes;
    return success();
  }

  LogicalResult getIterationDomainTileFromResultTile(
      Operation *op, OpBuilder &b, unsigned resultNumber,
      ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes,
      SmallVectorImpl<OpFoldResult> &iterDomainOffsets,
      SmallVectorImpl<OpFoldResult> &iterDomainSizes) const {
    auto linalgOp = cast<LinalgOp>(op);

    // Check that the indexing map used for the output is a projected
    // permutation. This could be relaxed with a more general approach that can
    // map the offsets and sizes from the result to iteration space tiles
    // (filling in full extent for dimensions not used to access the result).
    AffineMap indexingMap =
        linalgOp.getIndexingMapMatchingResult(op->getResult(resultNumber));
    if (!indexingMap.isProjectedPermutation()) {
      return op->emitOpError(
          "unhandled tiled implementation generation when result is not "
          "accessed using a permuted projection");
    }

    getMappedOffsetAndSize(linalgOp, b, indexingMap, offsets, sizes,
                           iterDomainOffsets, iterDomainSizes);
    return success();
  }

  FailureOr<TilingResult>
  generateResultTileValue(Operation *op, OpBuilder &b, unsigned resultNumber,
                          ArrayRef<OpFoldResult> offsets,
                          ArrayRef<OpFoldResult> sizes) const {
    SmallVector<OpFoldResult> mappedOffsets, mappedSizes;
    if (failed(getIterationDomainTileFromResultTile(
            op, b, resultNumber, offsets, sizes, mappedOffsets, mappedSizes))) {
      return failure();
    }
    auto tilingInterfaceOp = cast<TilingInterface>(op);
    FailureOr<TilingResult> tilingResult =
        tilingInterfaceOp.getTiledImplementation(b, mappedOffsets, mappedSizes);

    if (failed(tilingResult))
      return failure();

    if (tilingResult->tiledOps.size() != 1)
      return op->emitOpError("failed to generate tiled implementation");

    return TilingResult{
        tilingResult->tiledOps,
        SmallVector<Value>{tilingResult->tiledValues[resultNumber]},
        tilingResult->generatedSlices};
  }

  /// Method to generate the tiled implementation of an operation from the tile
  /// of the operand.
  FailureOr<TilingResult> getTiledImplementationFromOperandTile(
      Operation *op, OpBuilder &b, unsigned operandNumber,
      ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes) const {
    SmallVector<OpFoldResult> mappedOffsets, mappedSizes;
    if (failed(getIterationDomainTileFromOperandTile(
            op, b, operandNumber, offsets, sizes, mappedOffsets,
            mappedSizes))) {
      return failure();
    }
    return getTiledImplementation(op, b, mappedOffsets, mappedSizes);
  }

  LogicalResult generateScalarImplementation(Operation *op, OpBuilder &builder,
                                             Location loc,
                                             ValueRange ivs) const {
    auto linalgOp = cast<LinalgOp>(op);
    if (!linalgOp.hasPureBufferSemantics())
      return op->emitOpError("expected operation to have buffer semantics");

    SmallVector<Value> indexedValues;
    indexedValues.reserve(linalgOp->getNumOperands());
    Location linalgOpLoc = op->getLoc();
    /// Load the data corresponding to the block arguments that
    /// represent input operands.
    for (OpOperand &operand : linalgOp->getOpOperands()) {
      if (!linalgOp.payloadUsesValueFromOperand(&operand)) {
        indexedValues.push_back(nullptr);
        continue;
      }
      if (linalgOp.isScalar(&operand)) {
        indexedValues.push_back(operand.get());
        continue;
      }
      SmallVector<Value> indices = getIndicesForAccess(
          builder, linalgOpLoc, linalgOp.getMatchingIndexingMap(&operand), ivs);
      Value load =
          builder.create<memref::LoadOp>(linalgOpLoc, operand.get(), indices);
      indexedValues.push_back(load);
    }

    /// Inline the op payload and store the result.
    return inlinePayload(builder, linalgOp, ivs, indexedValues);
  }
};

//===----------------------------------------------------------------------===//
// External Model for implementing `PartialReductionInterface` for `LinalgOp`s.
//===----------------------------------------------------------------------===//

/// Return an AffineMap for a partial result for the given result number,
/// assuming the partial tiling strategy is outer-reduction loop +
/// inner-parallel tile. The returned AffineMap can be used as the replacement
/// AffineMap for the inner-parallel tile linalg op for the given result number.
///
/// The new AffineMap is the old AffineMap with reduction dimensions appended
/// at end.
static AffineMap getPartialResultAffineMap(LinalgOp linalgOp,
                                           ArrayRef<int> reductionDims,
                                           unsigned resultNumber) {
  AffineMap map =
      linalgOp.getMatchingIndexingMap(linalgOp.getDpsInitOperand(resultNumber));
  for (int redPos : reductionDims) {
    map = map.insertResult(getAffineDimExpr(redPos, linalgOp.getContext()),
                           map.getNumResults());
  }
  return map;
}

/// External model implementation of PartialReductionInterface for
/// LinalgOps.
template <typename LinalgOpTy>
struct LinalgOpPartialReductionInterface
    : public PartialReductionOpInterface::ExternalModel<
          LinalgOpPartialReductionInterface<LinalgOpTy>, LinalgOpTy> {
  FailureOr<SmallVector<Value>> generateInitialTensorForPartialReduction(
      Operation *op, OpBuilder &b, Location loc, ArrayRef<OpFoldResult> sizes,
      ArrayRef<int> reductionDims) const {
    auto linalgOp = cast<LinalgOp>(op);
    OpBuilder::InsertionGuard guard(b);

    if (linalgOp.hasPureBufferSemantics())
      return op->emitOpError("expected operation to have tensor semantics");

    // LinalgOp implements TilingInterface.
    auto tilingInterfaceOp = cast<TilingInterface>(linalgOp.getOperation());
    SmallVector<OpFoldResult> shape =
        llvm::map_to_vector(tilingInterfaceOp.getIterationDomain(b),
                            [](Range x) { return x.size; });

    SmallVector<OpFoldResult> tiledShape;
    for (auto [tileSize, dimSize] : llvm::zip_equal(sizes, shape)) {
      if (isZeroInteger(tileSize)) {
        tiledShape.push_back(dimSize);
      } else {
        tiledShape.push_back(tileSize);
      }
    }

    SmallVector<Value> inits;
    for (int initIdx = 0, e = linalgOp.getNumDpsInits(); initIdx < e;
         ++initIdx) {
      SmallVector<Operation *, 4> combinerOps;
      if (!matchReduction(linalgOp.getRegionOutputArgs(), initIdx,
                          combinerOps) ||
          combinerOps.size() != 1)
        return op->emitOpError("Failed to anaysis the reduction operation.");

      Operation *reductionOp = combinerOps[0];
      std::optional<TypedAttr> identity = arith::getNeutralElement(reductionOp);
      if (!identity.has_value())
        return op->emitOpError(
            "Failed to get an identity value for the reduction operation.");

      // Append the new partial result dimensions.
      AffineMap partialMap =
          getPartialResultAffineMap(linalgOp, reductionDims, initIdx);
      SmallVector<OpFoldResult> partialResultShape;
      for (AffineExpr dimExpr : partialMap.getResults()) {
        auto dim = cast<AffineDimExpr>(dimExpr);
        partialResultShape.push_back(tiledShape[dim.getPosition()]);
      }

      Type elType =
          getElementTypeOrSelf(linalgOp->getResult(initIdx).getType());
      Value emptyTensor =
          b.create<tensor::EmptyOp>(loc, partialResultShape, elType);
      Value constantOp = b.create<arith::ConstantOp>(loc, *identity);
      auto identityTensor =
          b.create<linalg::FillOp>(loc, constantOp, emptyTensor);
      inits.push_back(identityTensor.getResult(0));
    }

    return inits;
  }

  FailureOr<TilingResult>
  tileToPartialReduction(Operation *op, OpBuilder &b, Location loc,
                         ValueRange init, ArrayRef<OpFoldResult> offsets,
                         ArrayRef<OpFoldResult> sizes,
                         ArrayRef<int> reductionDims) const {
    OpBuilder::InsertionGuard guard(b);
    auto linalgOp = cast<LinalgOp>(op);

    // Step 1. Extend init maps to have reduction dimension dims, since we
    // are converting them to parallel dimensions.
    SmallVector<AffineMap> newInitMaps;
    newInitMaps.reserve(linalgOp.getNumDpsInits());
    for (int idx : llvm::seq<int>(0, linalgOp.getNumDpsInits())) {
      // TODO: linalg::Generic doesn't have getDpsInitOperands. Can replace
      // this with a for range loop when we have it.
      AffineMap newMap =
          getPartialResultAffineMap(linalgOp, reductionDims, idx);
      newInitMaps.push_back(newMap);
    }

    // Step 2a: Extract a slice of the input operands.
    SmallVector<Value> tiledInputs = makeTiledShapes(
        b, loc, linalgOp, linalgOp.getDpsInputs(), offsets, sizes, {}, true);
    SmallVector<Operation *> generatedSlices = llvm::map_to_vector(
        llvm::make_filter_range(
            tiledInputs, [](Value v) -> bool { return v.getDefiningOp(); }),
        [](Value v) -> Operation * { return v.getDefiningOp(); });

    // Step 2b: Extract a slice of the init operands.
    SmallVector<Value, 1> tiledInits;
    for (auto [valueMap, valueToTile] : llvm::zip_equal(newInitMaps, init)) {
      int64_t initRank = valueMap.getNumResults();
      SmallVector<OpFoldResult> initOffset(initRank, b.getIndexAttr(0));
      SmallVector<OpFoldResult> initStride(initRank, b.getIndexAttr(1));
      SmallVector<OpFoldResult> initSizes;
      for (AffineExpr dimExpr : valueMap.getResults()) {
        auto dim = cast<AffineDimExpr>(dimExpr);
        initSizes.push_back(sizes[dim.getPosition()]);
      }
      // TODO: Use SubsetExtractOpInterface here once available.
      auto extractSlice = b.create<tensor::ExtractSliceOp>(
          loc, valueToTile, initOffset, initSizes, initStride);
      tiledInits.push_back(extractSlice);
      generatedSlices.push_back(extractSlice);
    }

    // Update the indexing maps.
    SmallVector<AffineMap> newMaps = linalgOp.getIndexingMapsArray();
    // Change the init maps.
    for (int idx : llvm::seq<int>(0, linalgOp.getNumDpsInits())) {
      // TODO: linalg::Generic doesn't have getDpsInitOperands. Can replace
      // this with a for range loop when we have it.
      OpOperand *initOperand = linalgOp.getDpsInitOperand(idx);
      int64_t mapIdx = linalgOp.getIndexingMapIndex(initOperand);
      newMaps[mapIdx] = newInitMaps[idx];
    }

    // Step 3. Change the reduction dim iterator types.
    SmallVector<utils::IteratorType> newIteratorTypes =
        linalgOp.getIteratorTypesArray();
    for (int dim : reductionDims)
      newIteratorTypes[dim] = utils::IteratorType::parallel;

    // Step 4. Create the new generic op.
    auto genericOp =
        b.create<GenericOp>(loc, ValueRange(tiledInits).getTypes(), tiledInputs,
                            tiledInits, newMaps, newIteratorTypes);
    IRMapping mapping;
    op->getRegion(0).cloneInto(&genericOp.getRegion(),
                               genericOp.getRegion().begin(), mapping);
    return TilingResult{
        {genericOp.getOperation()},
        llvm::map_to_vector(genericOp->getResults(),
                            [](OpResult r) -> Value { return r; }),
        generatedSlices};
  }

  FailureOr<MergeResult> mergeReductions(Operation *op, OpBuilder &b,
                                         Location loc, ValueRange partialReduce,
                                         ArrayRef<int> reductionDims) const {
    auto linalgOp = cast<LinalgOp>(op);

    // Permute the reduction dims as permuted by the partial result map.

    int64_t numInits = linalgOp.getNumDpsInits();
    SmallVector<Operation *> mergeOperations;
    SmallVector<Value> replacements;
    for (int idx : llvm::seq(numInits)) {
      // linalg.reduce's iteration space is the tiled result's iteration space
      // (and not the tiled operation's iteration space). To account for this,
      // permute the reduction dimensions based on the partial result map of the
      // tiled result.
      AffineMap partialMap =
          getPartialResultAffineMap(linalgOp, reductionDims, idx);
      SmallVector<int64_t> partialReductionDims;
      for (auto [resultNum, dimExpr] :
           llvm::enumerate(partialMap.getResults())) {
        unsigned dim = cast<AffineDimExpr>(dimExpr).getPosition();
        if (llvm::is_contained(reductionDims, dim)) {
          partialReductionDims.push_back(resultNum);
        }
      }

      Value partialResult = partialReduce[idx];
      Value init = linalgOp.getDpsInits()[idx];

      auto reduction = b.create<linalg::ReduceOp>(
          loc, partialResult, init, partialReductionDims,
          [&linalgOp, &idx](OpBuilder &b, Location loc, ValueRange inputs) {
            // Get the combiner op.
            SmallVector<Operation *, 4> combinerOps;
            matchReduction(linalgOp.getRegionOutputArgs(), idx, combinerOps);
            Operation *clonedReductionOp = b.clone(*combinerOps[0]);
            // Combine the input at idx and output at numInits + idx.
            clonedReductionOp->setOperand(0, inputs[0]);
            clonedReductionOp->setOperand(1, inputs[1]);
            b.create<linalg::YieldOp>(loc, clonedReductionOp->getResult(0));
          });

      mergeOperations.push_back(reduction);
      replacements.push_back(reduction->getResult(0));
    }

    return MergeResult{mergeOperations, replacements};
  }

  LogicalResult getPartialResultTilePosition(
      Operation *op, OpBuilder &b, unsigned resultNumber,
      ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes,
      SmallVector<OpFoldResult> &resultOffsets,
      SmallVector<OpFoldResult> &resultSizes,
      ArrayRef<int> reductionDims) const {
    auto linalgOp = cast<LinalgOp>(op);

    AffineMap partialMap =
        getPartialResultAffineMap(linalgOp, reductionDims, resultNumber);
    for (AffineExpr dimExpr : partialMap.getResults()) {
      unsigned dim = cast<AffineDimExpr>(dimExpr).getPosition();
      resultSizes.push_back(sizes[dim]);

      if (llvm::is_contained(reductionDims, dim)) {
        // Reduction dims are reduced, and are always outputed in the same
        // place. So use offset 0 for them.
        resultOffsets.push_back(b.getIndexAttr(0));
      } else {
        resultOffsets.push_back(offsets[dim]);
      }
    }

    return success();
  }
};

template <typename OpTy>
static SmallVector<Range> getPackUnPackIterationDomain(OpTy op,
                                                       OpBuilder &builder) {
  static_assert(llvm::is_one_of<OpTy, PackOp, UnPackOp>::value,
                "applies to only pack or unpack operations");
  OpBuilder::InsertionGuard g(builder);
  int64_t rank = (std::is_same<OpTy, PackOp>::value) ? op.getSourceRank()
                                                     : op.getDestRank();
  OpFoldResult zero = builder.getIndexAttr(0);
  OpFoldResult one = builder.getIndexAttr(1);
  ReifiedRankedShapedTypeDims resultShape;
  (void)reifyResultShapes(builder, op, resultShape);
  SmallVector<Range> loopBounds(rank);
  for (auto dim : llvm::seq<int64_t>(0, rank)) {
    loopBounds[dim].offset = zero;
    loopBounds[dim].stride = one;
    loopBounds[dim].size = resultShape[0][dim];
  }
  return loopBounds;
}

static void applyPermToRange(SmallVector<OpFoldResult> &offsets,
                             SmallVector<OpFoldResult> &sizes,
                             ArrayRef<int64_t> permutation) {
  if (permutation.empty())
    return;
  applyPermutationToVector<OpFoldResult>(offsets, permutation);
  applyPermutationToVector<OpFoldResult>(sizes, permutation);
}

struct PackOpTiling
    : public TilingInterface::ExternalModel<PackOpTiling, linalg::PackOp> {

  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *op) const {
    // Note that here we only consider untiled dimensions and outer tiled data
    // dimensions, the inner tiled data dimensions are materialized when
    // building the body of the operation.
    auto packOp = cast<PackOp>(op);
    SmallVector<utils::IteratorType> iteratorTypes(
        packOp.getSourceRank(), utils::IteratorType::parallel);
    return iteratorTypes;
  }

  SmallVector<Range> getIterationDomain(Operation *op, OpBuilder &b) const {
    return getPackUnPackIterationDomain<PackOp>(cast<PackOp>(op), b);
  }

  FailureOr<TilingResult>
  getTiledImplementation(Operation *op, OpBuilder &b,
                         ArrayRef<OpFoldResult> offsets,
                         ArrayRef<OpFoldResult> sizes) const {
    auto packOp = cast<PackOp>(op);
    Location loc = packOp.getLoc();

    // The tiling is applied on interchanged dimensions. We have to undo the
    // interchange to map sizes and offsets to the original input.
    int64_t inputRank = packOp.getSourceRank();
    SmallVector<OpFoldResult> origOffsets(offsets);
    SmallVector<OpFoldResult> origSizes(sizes);
    applyPermToRange(origOffsets, origSizes,
                     invertPermutationVector(packOp.getOuterDimsPerm()));

    DenseMap<int64_t, OpFoldResult> dimAndTileMapping =
        packOp.getDimAndTileMapping();
    SmallVector<OpFoldResult> srcDimValues =
        tensor::getMixedSizes(b, loc, packOp.getSource());
    SmallVector<OpFoldResult> inputIndices, inputSizes;
    for (auto dim : llvm::seq<int64_t>(0, inputRank)) {
      using AV = affine::AffineValueExpr;
      affine::AffineBuilder ab(b, loc);
      AffineExpr dim0, dim1, sym;
      bindDims(b.getContext(), dim0, dim1);
      bindSymbols(b.getContext(), sym);
      if (dimAndTileMapping.count(dim)) {
        // If the data dimension is tiled, the i-th index is the product of
        // offset_i and tile_i, and the i-th size is the product of sizes_i and
        // tile_i.
        auto avOffset = AV(dim0).bind(origOffsets[dim]);
        auto avSize = AV(dim0).bind(origSizes[dim]);
        auto avTileSize = AV(sym).bind(dimAndTileMapping[dim]);
        inputIndices.push_back(ab.mul(avOffset, avTileSize));
        inputSizes.push_back(ab.mul(avSize, avTileSize));
      } else {
        inputIndices.push_back(origOffsets[dim]);
        inputSizes.push_back(origSizes[dim]);
      }

      // Limit the size of the input operand for incomplete tiles.
      if (packOp.getPaddingValue()) {
        OpFoldResult dimSize = srcDimValues[dim];
        auto avDimSize = AV(dim0).bind(dimSize);
        auto avInputIdx = AV(dim1).bind(inputIndices.back());
        inputSizes.back() =
            ab.min({inputSizes.back(), ab.sub(avDimSize, avInputIdx)});
      }
    }

    auto oneAttr = b.getI64IntegerAttr(1);
    SmallVector<OpFoldResult> strides(inputRank, oneAttr);

    SmallVector<Value> tiledOperands;
    auto sourceSlice = b.create<tensor::ExtractSliceOp>(
        loc, packOp.getSource(), inputIndices, inputSizes, strides);
    tiledOperands.push_back(sourceSlice);

    SmallVector<OpFoldResult> outputOffsets, outputSizes;
    if (failed(getResultTilePosition(op, b, 0, offsets, sizes, outputOffsets,
                                     outputSizes)))
      return {};

    strides.append(packOp.getDestRank() - inputRank, oneAttr);
    auto outSlice = b.create<tensor::ExtractSliceOp>(
        loc, packOp.getDest(), outputOffsets, outputSizes, strides);
    tiledOperands.push_back(outSlice);

    if (auto val = packOp.getPaddingValue())
      tiledOperands.push_back(val);
    for (auto tile : packOp.getInnerTiles())
      tiledOperands.push_back(tile);

    Operation *tiledPackOp = b.create<PackOp>(
        loc, TypeRange{outSlice.getType()}, tiledOperands, op->getAttrs());

    return TilingResult{
        {tiledPackOp},
        SmallVector<Value>(tiledPackOp->getResults()),
        llvm::to_vector(ArrayRef<Operation *>{sourceSlice, outSlice})};
  }

  LogicalResult
  getResultTilePosition(Operation *op, OpBuilder &b, unsigned resultNumber,
                        ArrayRef<OpFoldResult> offsets,
                        ArrayRef<OpFoldResult> sizes,
                        SmallVector<OpFoldResult> &resultOffsets,
                        SmallVector<OpFoldResult> &resultSizes) const {
    // The iteration domain is over outer dimensions of packed layout. In this
    // context, the outer dimensions of `resultOffsets` are `offsets`. The
    // inner dimensions of `resultOffsets` are zeros because tiling is not
    // applied to them.
    auto packOp = cast<PackOp>(op);
    int64_t inputRank = packOp.getSourceRank();
    int64_t outputRank = packOp.getDestRank();
    auto zeroAttr = b.getI64IntegerAttr(0);
    resultOffsets.assign(offsets.begin(), offsets.end());
    resultOffsets.append(outputRank - inputRank, zeroAttr);

    ReifiedRankedShapedTypeDims outputShape;
    (void)reifyResultShapes(b, packOp, outputShape);
    resultSizes.assign(sizes.begin(), sizes.end());
    for (auto dataTileDim : llvm::seq<unsigned>(inputRank, outputRank))
      resultSizes.push_back(outputShape[0][dataTileDim]);

    return success();
  }

  FailureOr<TilingResult>
  generateResultTileValue(Operation *op, OpBuilder &b, unsigned resultNumber,
                          ArrayRef<OpFoldResult> offsets,
                          ArrayRef<OpFoldResult> sizes) const {
    auto packOp = cast<PackOp>(op);
    int64_t numTiles = packOp.getInnerDimsPos().size();

    // tensor.pack op is fusible (as a producer) only if full inner tiles are
    // iterated or inner dims are not tiled. Otherwise, it will generate a
    // sequence of non-trivial ops (for partial tiles).
    for (auto offset : offsets.take_back(numTiles))
      if (!isZeroInteger(offset))
        return failure();

    for (auto iter :
         llvm::zip_equal(packOp.getMixedTiles(), sizes.take_back(numTiles)))
      if (!isEqualConstantIntOrValue(std::get<0>(iter), std::get<1>(iter)))
        return failure();

    FailureOr<TilingResult> tilingResult = getTiledImplementation(
        op, b, offsets.drop_back(numTiles), sizes.drop_back(numTiles));
    if (failed(tilingResult))
      return failure();
    return tilingResult.value();
  }

  /// Method to return the position of iteration domain tile computed by the
  /// tiled operation. In current `tensor.pack` context, the `resultOffsets` and
  /// `resultSizes` only cover outer dimensions.
  LogicalResult getIterationDomainTileFromOperandTile(
      Operation *op, OpBuilder &b, unsigned operandNumber,
      ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes,
      SmallVectorImpl<OpFoldResult> &resultOffsets,
      SmallVectorImpl<OpFoldResult> &resultSizes) const {
    if (operandNumber != 0)
      return failure();

    auto packOp = cast<PackOp>(op);
    // It is not trivial to infer dest tile from source tile if `packOp` has
    // padding semantic.
    if (packOp.getPaddingValue())
      return failure();

    Location loc = packOp.getLoc();

    SmallVector<OpFoldResult> outerDimOffsets, outerDimSizes;
    DenseMap<int64_t, OpFoldResult> dimAndTileMapping =
        packOp.getDimAndTileMapping();
    for (auto dim : llvm::seq<int64_t>(packOp.getSourceRank())) {
      if (dimAndTileMapping.count(dim)) {
        FailureOr<int64_t> cstSize =
            ValueBoundsConstraintSet::computeConstantBound(
                presburger::BoundType::UB, sizes[dim],
                /*stopCondition=*/nullptr, /*closedUB=*/true);
        std::optional<int64_t> cstInnerSize =
            getConstantIntValue(dimAndTileMapping[dim]);
        // Currently fusing `packOp` as consumer only expects perfect tiling
        // scenario because even if without padding semantic, the `packOp` may
        // also yield incomplete tiles. E.g. tensor<30xf32> -> tensor<5x6xf32>,
        // where the `tileSize` from operand of `packOp` is 5, which is not
        // exactly divided by `innerTile`(=6) of `packOp`. As the result:
        // 1. the first slice is extracted from (0) to (4) and inserted into
        // (0,0)~(0,4) at first row.
        // 2. the second slice is extracted from (5) to (9) and SHOULD BE
        // respectively inserted into two rows with different length, including
        // first row: (0,5) and second row (1,0)~(1,3). It is hard to coordinate
        // them, thus adding below constraint to bypass them temporarily. In
        // another word, we can only support tiling with consumer if the tile
        // size for the producer is a multiple of the inner tile size for the
        // packed dimensions at this moment.
        if (failed(cstSize) || !cstInnerSize || *cstSize % *cstInnerSize != 0) {
          return failure();
        }

        using AV = affine::AffineValueExpr;
        affine::AffineBuilder ab(b, loc);
        AffineExpr dim0, sym;
        bindDims(b.getContext(), dim0);
        bindSymbols(b.getContext(), sym);
        auto avOffset = AV(dim0).bind(offsets[dim]);
        auto avSize = AV(dim0).bind(sizes[dim]);
        auto avTileSize = AV(sym).bind(dimAndTileMapping[dim]);
        outerDimOffsets.push_back(ab.floor(avOffset, avTileSize));
        outerDimSizes.push_back(ab.ceil(avSize, avTileSize));
      } else {
        outerDimOffsets.push_back(offsets[dim]);
        outerDimSizes.push_back(sizes[dim]);
      }
    }
    applyPermToRange(outerDimOffsets, outerDimSizes, packOp.getOuterDimsPerm());
    resultOffsets = outerDimOffsets;
    resultSizes = outerDimSizes;
    return success();
  }

  /// Method to return the tiled implementation of tensor.pack as a consumer.
  FailureOr<TilingResult> getTiledImplementationFromOperandTile(
      Operation *op, OpBuilder &b, unsigned operandNumber,
      ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes) const {
    if (operandNumber != 0)
      return failure();

    auto packOp = cast<PackOp>(op);
    Location loc = packOp.getLoc();

    int64_t inputRank = packOp.getSourceRank();
    auto oneAttr = b.getI64IntegerAttr(1);
    SmallVector<OpFoldResult> strides(inputRank, oneAttr);

    SmallVector<Value> tiledOperands;
    auto sourceSlice = b.create<tensor::ExtractSliceOp>(
        loc, packOp.getSource(), offsets, sizes, strides);
    tiledOperands.push_back(sourceSlice);

    SmallVector<OpFoldResult> outerDimOffsets, outerDimSizes;
    if (failed(getIterationDomainTileFromOperandTile(
            op, b, /*operandNumber=*/0, offsets, sizes, outerDimOffsets,
            outerDimSizes)))
      return failure();

    SmallVector<OpFoldResult> outputOffsets, outputSizes;
    if (failed(getResultTilePosition(op, b, 0, outerDimOffsets, outerDimSizes,
                                     outputOffsets, outputSizes)))
      return failure();

    strides.append(packOp.getDestRank() - inputRank, oneAttr);
    auto outSlice = b.create<tensor::ExtractSliceOp>(
        loc, packOp.getDest(), outputOffsets, outputSizes, strides);
    tiledOperands.push_back(outSlice);

    assert(!packOp.getPaddingValue() && "Expect no padding semantic");
    for (auto tile : packOp.getInnerTiles())
      tiledOperands.push_back(tile);

    Operation *tiledPackOp = b.create<PackOp>(
        loc, TypeRange{outSlice.getType()}, tiledOperands, op->getAttrs());

    return TilingResult{
        {tiledPackOp},
        SmallVector<Value>(tiledPackOp->getResults()),
        llvm::to_vector(ArrayRef<Operation *>{sourceSlice, outSlice})};
  }
};

struct UnpackTileDimInfo {
  bool isAlignedToInnerTileSize;
  OpFoldResult sourceOffset;
  OpFoldResult sourceSize;
  OpFoldResult resultOffset;
  OpFoldResult destExpandedSize;
};

/// Returns the needed information for tiling unpack op on `tileDim` with given
/// `tileOffset` and `tileSize`. For more details, see the comment of the
/// `getTiledImplementation`.
static UnpackTileDimInfo getUnpackTileDimInfo(OpBuilder &b, UnPackOp unpackOp,
                                              int64_t tileDim,
                                              OpFoldResult tileOffset,
                                              OpFoldResult tileSize) {
  UnpackTileDimInfo info;
  Attribute zeroAttr = b.getIndexAttr(0);
  Attribute oneAttr = b.getIndexAttr(1);
  DenseMap<int64_t, OpFoldResult> dimAndTileMapping =
      unpackOp.getDimAndTileMapping();
  // The dimension is not one of packed data dimension.
  if (!dimAndTileMapping.count(tileDim)) {
    info.isAlignedToInnerTileSize = true;
    info.sourceOffset = tileOffset;
    info.sourceSize = tileSize;
    info.resultOffset = zeroAttr;
    info.destExpandedSize = tileSize;
    return info;
  }

  Location loc = unpackOp.getLoc();
  using AV = affine::AffineValueExpr;
  affine::AffineBuilder ab(b, loc);
  AffineExpr dim0, dim1, sym0;
  bindDims(b.getContext(), dim0, dim1);
  bindSymbols(b.getContext(), sym0);

  OpFoldResult innerTileSize = dimAndTileMapping[tileDim];

  info.isAlignedToInnerTileSize = false;
  FailureOr<int64_t> cstSize = ValueBoundsConstraintSet::computeConstantBound(
      presburger::BoundType::UB, tileSize,
      /*stopCondition=*/nullptr, /*closedUB=*/true);
  std::optional<int64_t> cstInnerSize = getConstantIntValue(innerTileSize);
  if (!failed(cstSize) && cstInnerSize) {
    if (*cstSize % *cstInnerSize == 0)
      info.isAlignedToInnerTileSize = true;

    // If the tiling size equals to the inner tiling size, the outer dims are
    // always 1.
    if (*cstInnerSize == *cstSize) {
      auto lhs = AV(dim0).bind(tileOffset);
      auto rhs = AV(dim1).bind(innerTileSize);
      info.sourceOffset = ab.floor(lhs, rhs);
      info.sourceSize = oneAttr;
      info.resultOffset = zeroAttr;
      info.destExpandedSize = tileSize;
      return info;
    }
  }

  if (info.isAlignedToInnerTileSize) {
    info.sourceOffset =
        ab.floor(AV(dim0).bind(tileOffset), AV(dim1).bind(innerTileSize));
    info.resultOffset = zeroAttr;
    info.destExpandedSize = tileSize;

    // The ceilDiv is needed here because there could be incomplete tile even
    // it is perfect tiling cases. E.g.,
    //   %0 = unpack tensor<33x2xf32> into tensor<64xf32>
    // If the tiling size is 32, there will be 3 tiles. Two of them have
    // size=32; one of them have size=2. The size is represented using
    // affine_min op; we need ceilDiv.
    info.sourceSize =
        ab.ceil(AV(dim0).bind(tileSize), AV(dim1).bind(innerTileSize));
    return info;
  }

  affine::DivModValue firstCoord = affine::getDivMod(
      b, loc, getValueOrCreateConstantIndexOp(b, loc, tileOffset),
      getValueOrCreateConstantIndexOp(b, loc, innerTileSize));
  OpFoldResult tileExclusiveBound =
      ab.add(AV(dim0).bind(tileOffset), AV(dim1).bind(tileSize));
  affine::DivModValue lastCoord = affine::getDivMod(
      b, loc,
      getValueOrCreateConstantIndexOp(
          b, loc,
          ab.sub(AV(dim0).bind(tileExclusiveBound), AV(dim1).bind(oneAttr))),
      getValueOrCreateConstantIndexOp(b, loc, innerTileSize));

  OpFoldResult lengthMinusOne = ab.sub(AV(dim0).bind(lastCoord.quotient),
                                       AV(dim1).bind(firstCoord.quotient));
  info.sourceSize =
      ab.add(AV(dim0).bind(lengthMinusOne), AV(dim1).bind(oneAttr));
  info.sourceOffset = firstCoord.quotient;
  info.resultOffset = firstCoord.remainder;
  // Do not create an Affine ops for expanded size because the affine op is too
  // complicated which would trigger an issue in affine ops simplification.
  info.destExpandedSize = b.createOrFold<arith::MulIOp>(
      loc, getValueOrCreateConstantIndexOp(b, loc, info.sourceSize),
      getValueOrCreateConstantIndexOp(b, loc, innerTileSize));
  return info;
}

struct UnPackOpTiling
    : public TilingInterface::ExternalModel<UnPackOpTiling, linalg::UnPackOp> {

  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *op) const {
    auto unpackOp = cast<UnPackOp>(op);
    SmallVector<utils::IteratorType> iteratorTypes(
        unpackOp.getDestRank(), utils::IteratorType::parallel);
    return iteratorTypes;
  }

  SmallVector<Range> getIterationDomain(Operation *op, OpBuilder &b) const {
    return getPackUnPackIterationDomain<UnPackOp>(cast<UnPackOp>(op), b);
  }

  /// There are two cases in tiling unpack ops. If the tiling size is aligned to
  /// the inner tile size, the corresponding tiles of source are all complete.
  /// Otherwise, there are in-complete tiles. We will need to expand the slice
  /// of source for getting complete tiles. The tiled unpack op unpacks more
  /// data from source, so We'll need an extract_slice op to shift and truncate
  /// the output.
  /// Take Nn_to_N as an example. Say that N=32, n=8, and tiling_size=15. The
  /// coordinates of second tile (i.e., result[15..31]) are
  /// [(1, 7), (2, 0,), (2, 1) ... (3, 6), (3, 7)]. The first row and the last
  /// row are incomplete tiles. To represent the unpack op, we have to complete
  /// the rows. I.e., the input coordinates would start with (1, 0); end with
  /// (3, 7). In this context, the tiled unpack produces a (3 * n) elements
  /// because there are 3 rows in total. Follow by a tensor.extract_slice op, we
  /// can get the actual result.
  FailureOr<TilingResult>
  getTiledImplementation(Operation *op, OpBuilder &b,
                         ArrayRef<OpFoldResult> offsets,
                         ArrayRef<OpFoldResult> sizes) const {
    auto unpackOp = cast<UnPackOp>(op);
    int64_t srcRank = unpackOp.getSourceRank();
    int64_t destRank = unpackOp.getDestRank();
    int64_t numInnerTiles = srcRank - destRank;
    Location loc = unpackOp.getLoc();

    // The perfect tiling case indicates that the tiling sizes are multiple of
    // inner_tile_size. In this context, no extra data is needed when
    // representing the tiled unpack op.
    bool isPerfectTilingCase = true;
    Attribute oneAttr = b.getIndexAttr(1);
    SmallVector<OpFoldResult> sliceSrcStrides(destRank, oneAttr);
    SmallVector<OpFoldResult> sliceSrcIndices, sliceSrcSizes;
    SmallVector<OpFoldResult> destExpandedSizes, resultOffsetsFromDest;
    for (auto dim : llvm::seq<int64_t>(0, destRank)) {
      UnpackTileDimInfo info =
          getUnpackTileDimInfo(b, unpackOp, dim, offsets[dim], sizes[dim]);
      if (!info.isAlignedToInnerTileSize)
        isPerfectTilingCase = false;
      sliceSrcIndices.push_back(info.sourceOffset);
      sliceSrcSizes.push_back(info.sourceSize);
      destExpandedSizes.push_back(info.destExpandedSize);
      resultOffsetsFromDest.push_back(info.resultOffset);
    }

    // The tiling is applied on destination dimensions. We have to apply the
    // interchange on source dimensions if outer_dims_perm is set.
    applyPermToRange(sliceSrcIndices, sliceSrcSizes,
                     unpackOp.getOuterDimsPerm());
    Attribute zeroAttr = b.getIndexAttr(0);
    sliceSrcIndices.append(numInnerTiles, zeroAttr);
    sliceSrcSizes.append(unpackOp.getMixedTiles());
    sliceSrcStrides.append(numInnerTiles, oneAttr);
    SmallVector<Operation *> generatedSlices;
    tensor::ExtractSliceOp sliceSource = b.create<tensor::ExtractSliceOp>(
        loc, unpackOp.getSource(), sliceSrcIndices, sliceSrcSizes,
        sliceSrcStrides);
    generatedSlices.push_back(sliceSource);

    SmallVector<OpFoldResult> destStrides(destRank, oneAttr);
    Value sliceDest;
    if (isPerfectTilingCase) {
      auto destSliceOp = b.create<tensor::ExtractSliceOp>(
          loc, unpackOp.getDest(), offsets, sizes, destStrides);
      sliceDest = destSliceOp;
      generatedSlices.push_back(destSliceOp);
    } else {
      sliceDest = b.create<tensor::EmptyOp>(
          loc, destExpandedSizes, unpackOp.getDestType().getElementType());
    }

    SmallVector<Value> tiledOperands = {sliceSource.getResult(), sliceDest};
    for (auto tile : unpackOp.getInnerTiles())
      tiledOperands.push_back(tile);

    Operation *tiledUnpackOp = b.create<UnPackOp>(
        loc, TypeRange{sliceDest.getType()}, tiledOperands, op->getAttrs());

    if (isPerfectTilingCase)
      return TilingResult{{tiledUnpackOp},
                          SmallVector<Value>(tiledUnpackOp->getResults()),
                          generatedSlices};

    auto extractSlice = b.create<tensor::ExtractSliceOp>(
        loc, tiledUnpackOp->getResult(0), resultOffsetsFromDest, sizes,
        destStrides);
    return TilingResult{
        {tiledUnpackOp}, {extractSlice.getResult()}, generatedSlices};
  }

  LogicalResult
  getResultTilePosition(Operation *op, OpBuilder &b, unsigned resultNumber,
                        ArrayRef<OpFoldResult> offsets,
                        ArrayRef<OpFoldResult> sizes,
                        SmallVector<OpFoldResult> &resultOffsets,
                        SmallVector<OpFoldResult> &resultSizes) const {
    resultOffsets = llvm::to_vector(offsets);
    resultSizes = llvm::to_vector(sizes);
    return success();
  }

  FailureOr<TilingResult>
  generateResultTileValue(Operation *op, OpBuilder &b, unsigned resultNumber,
                          ArrayRef<OpFoldResult> offsets,
                          ArrayRef<OpFoldResult> sizes) const {
    FailureOr<TilingResult> tilingResult =
        getTiledImplementation(op, b, offsets, sizes);
    if (failed(tilingResult))
      return failure();
    return tilingResult.value();
  }

  /// Method to return the position of iteration domain tile computed by the
  /// tiled operation.
  LogicalResult getIterationDomainTileFromOperandTile(
      Operation *op, OpBuilder &b, unsigned operandNumber,
      ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes,
      SmallVectorImpl<OpFoldResult> &resultOffsets,
      SmallVectorImpl<OpFoldResult> &resultSizes) const {
    auto unPackOp = cast<UnPackOp>(op);
    // If the operand tile is the dest, then no adjustment is needed.
    if (operandNumber == unPackOp.getDestMutable().getOperandNumber()) {
      resultOffsets = llvm::to_vector(offsets);
      resultSizes = llvm::to_vector(sizes);
      return success();
    }
    Location loc = unPackOp.getLoc();

    int64_t numTiles = unPackOp.getInnerDimsPos().size();
    auto destOffsets = offsets.drop_back(numTiles);
    auto destSizes = sizes.drop_back(numTiles);
    // The tiling is applied on interchanged dimensions. We have to undo the
    // interchange to map sizes and offsets to the original input.
    int64_t outputRank = unPackOp.getDestRank();
    ReifiedRankedShapedTypeDims reifiedReturnShapes;
    if (failed(reifyResultShapes(b, unPackOp, reifiedReturnShapes)))
      return failure();
    SmallVector<OpFoldResult> outputMixedSizes = reifiedReturnShapes.front();
    SmallVector<OpFoldResult> origOffsets(destOffsets);
    SmallVector<OpFoldResult> origSizes(destSizes);
    applyPermToRange(origOffsets, origSizes,
                     invertPermutationVector(unPackOp.getOuterDimsPerm()));

    DenseMap<int64_t, OpFoldResult> dimAndTileMapping =
        unPackOp.getDimAndTileMapping();

    for (auto dim : llvm::seq<int64_t>(0, outputRank)) {
      using AV = affine::AffineValueExpr;
      affine::AffineBuilder ab(b, loc);
      AffineExpr dim0, dim1, sym0;
      bindDims(b.getContext(), dim0, dim1);
      bindSymbols(b.getContext(), sym0);
      if (dimAndTileMapping.count(dim)) {
        // If the data dimension is tiled, the i-th index is the product of
        // offset_i and tile_i, and the i-th size is the product of sizes_i and
        // tile_i. The sizes must be clamped to the sizes of the unpack result.
        auto avOffset = AV(dim0).bind(origOffsets[dim]);
        auto avSize = AV(dim0).bind(origSizes[dim]);
        auto avTileSize = AV(sym0).bind(dimAndTileMapping[dim]);
        auto avResultSize = AV(dim0).bind(outputMixedSizes[dim]);
        resultOffsets.push_back(ab.mul(avOffset, avTileSize));
        auto avResultOffset = AV(dim1).bind(resultOffsets.back());
        resultSizes.push_back(ab.min({ab.mul(avSize, avTileSize),
                                      ab.sub(avResultSize, avResultOffset)}));
      } else {
        resultOffsets.push_back(origOffsets[dim]);
        resultSizes.push_back(origSizes[dim]);
      }
    }
    return success();
  }

  /// Method to return the tiled implementation of tensor.unpack as a consumer.
  FailureOr<TilingResult> getTiledImplementationFromOperandTile(
      Operation *op, OpBuilder &b, unsigned operandNumber,
      ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes) const {
    auto unPackOp = cast<UnPackOp>(op);
    // tensor.unpack op is fusible (as a consumer) only if inner dims are not
    // tiled.
    int64_t numTiles = unPackOp.getInnerDimsPos().size();
    for (auto iter :
         llvm::zip_equal(unPackOp.getMixedTiles(), sizes.take_back(numTiles))) {
      if (!isEqualConstantIntOrValue(std::get<0>(iter), std::get<1>(iter)))
        return failure();
    }

    Location loc = unPackOp.getLoc();

    // Fetch offset/size for creating the slice of the dest operand of
    // unpack op.
    SmallVector<OpFoldResult> outputOffsets, outputSizes;
    if (failed(getIterationDomainTileFromOperandTile(
            op, b, /*operandNumber=*/0, offsets, sizes, outputOffsets,
            outputSizes)))
      return failure();

    auto oneAttr = b.getI64IntegerAttr(1);
    int64_t outputRank = unPackOp.getDestRank();
    SmallVector<OpFoldResult> strides(outputRank, oneAttr);

    SmallVector<Value> tiledOperands;
    // Create slice of the dest operand.
    auto extractDestSlice = b.create<tensor::ExtractSliceOp>(
        loc, unPackOp.getDest(), outputOffsets, outputSizes, strides);
    tiledOperands.push_back(extractDestSlice);

    strides.append(unPackOp.getSourceRank() - outputRank, oneAttr);
    // Create slice of the source operand.
    auto extractSourceSlice = b.create<tensor::ExtractSliceOp>(
        loc, unPackOp.getSource(), offsets, sizes, strides);
    tiledOperands.insert(tiledOperands.begin(), extractSourceSlice);
    for (auto tile : unPackOp.getInnerTiles())
      tiledOperands.push_back(tile);

    // Create tiled unpack op.
    Operation *tiledUnPackOp =
        b.create<UnPackOp>(loc, TypeRange{extractDestSlice.getType()},
                           tiledOperands, op->getAttrs());

    return TilingResult{{tiledUnPackOp},
                        SmallVector<Value>(tiledUnPackOp->getResults()),
                        llvm::to_vector(ArrayRef<Operation *>{
                            extractSourceSlice, extractDestSlice})};
  }
};

} // namespace

template <typename OpType>
static void registerOne(MLIRContext *ctx) {
  OpType::template attachInterface<LinalgOpTilingInterface<OpType>>(*ctx);
  OpType::template attachInterface<LinalgOpPartialReductionInterface<OpType>>(
      *ctx);
}

/// Variadic helper function.
template <typename... OpTypes>
static void registerAll(MLIRContext *ctx) {
  (registerOne<OpTypes>(ctx), ...);
}

#define GET_OP_LIST

void mlir::linalg::registerTilingInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, linalg::LinalgDialect *dialect) {
    registerOne<linalg::GenericOp>(ctx);
    linalg::PackOp::attachInterface<PackOpTiling>(*ctx);
    linalg::UnPackOp::attachInterface<UnPackOpTiling>(*ctx);
    registerAll<
#include "mlir/Dialect/Linalg/IR/LinalgStructuredOps.cpp.inc"
        >(ctx);
  });
}

void mlir::linalg::registerTilingInterfaceExternalModelsForPackUnPackOps(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, LinalgDialect *dialect) {
    linalg::PackOp::attachInterface<PackOpTiling>(*ctx);
    linalg::UnPackOp::attachInterface<UnPackOpTiling>(*ctx);
  });
}
