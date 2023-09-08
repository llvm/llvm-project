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
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Interfaces/TilingInterface.h"
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

  // Instantiate the tiled implementation of the operation.
  FailureOr<TilingResult>
  getTiledImplementation(Operation *op, OpBuilder &b,
                         ArrayRef<OpFoldResult> offsets,
                         ArrayRef<OpFoldResult> sizes) const {
    // Leave the `sizeBounds` value empty. That is only needed when the `sizes`
    // specified could lead to out of bounds accesses.
    Location loc = op->getLoc();
    LinalgOp linalgOp = cast<LinalgOp>(op);
    SmallVector<Value> valuesToTile = linalgOp->getOperands();
    SmallVector<Value, 4> tiledOperands = makeTiledShapes(
        b, loc, linalgOp, valuesToTile, offsets, sizes, {}, true);

    SmallVector<Type> resultTensorTypes =
        getTensorOutputTypes(linalgOp, tiledOperands);

    Operation *tiledOp = clone(b, linalgOp, resultTensorTypes, tiledOperands);
    offsetIndices(b, cast<LinalgOp>(tiledOp), offsets);

    return TilingResult{{tiledOp}, SmallVector<Value>(tiledOp->getResults())};
  }

  // Return the details of the output tile generated by the tiled
  // implementation.
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

  FailureOr<TilingResult>
  generateResultTileValue(Operation *op, OpBuilder &b, unsigned resultNumber,
                          ArrayRef<OpFoldResult> offsets,
                          ArrayRef<OpFoldResult> sizes) const {
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

    auto numLoops = linalgOp.getNumLoops();
    auto tilingInterfaceOp = cast<TilingInterface>(op);
    SmallVector<OpFoldResult> iterationTileOffsets(numLoops),
        iterationTileSizes(numLoops);
    if (!indexingMap.isPermutation()) {
      SmallVector<Range> iterationDomain =
          tilingInterfaceOp.getIterationDomain(b);
      for (const auto &range : llvm::enumerate(iterationDomain)) {
        iterationTileOffsets[range.index()] = range.value().offset;
        iterationTileSizes[range.index()] = range.value().size;
      }
    }
    for (const auto &resultExpr : llvm::enumerate(indexingMap.getResults())) {
      unsigned dimPosition =
          resultExpr.value().template cast<AffineDimExpr>().getPosition();
      iterationTileOffsets[dimPosition] = offsets[resultExpr.index()];
      iterationTileSizes[dimPosition] = sizes[resultExpr.index()];
    }

    FailureOr<TilingResult> tilingResult =
        tilingInterfaceOp.getTiledImplementation(b, iterationTileOffsets,
                                                 iterationTileSizes);
    if (tilingResult->tiledOps.size() != 1)
      return op->emitOpError("failed to generate tiled implementation");

    return TilingResult{
        tilingResult->tiledOps,
        SmallVector<Value>{tilingResult->tiledValues[resultNumber]}};
  }

  LogicalResult generateScalarImplementation(Operation *op, OpBuilder &builder,
                                             Location loc,
                                             ValueRange ivs) const {
    auto linalgOp = cast<LinalgOp>(op);
    if (!linalgOp.hasBufferSemantics())
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

/// External model implementation of PartialReductionInterface for LinalgOps.
template <typename LinalgOpTy>
struct LinalgOpPartialReductionInterface
    : public PartialReductionOpInterface::ExternalModel<
          LinalgOpPartialReductionInterface<LinalgOpTy>, LinalgOpTy> {
  FailureOr<Operation *> generateInitialTensorForPartialReduction(
      Operation *op, OpBuilder &b, Location loc, ArrayRef<OpFoldResult> sizes,
      ArrayRef<int> reductionDims) const {
    auto linalgOp = cast<LinalgOp>(op);
    OpBuilder::InsertionGuard guard(b);

    if (linalgOp.hasBufferSemantics())
      return op->emitOpError("expected operation to have tensor semantics");
    // Insert the new parallel dimension based on the index of the reduction
    // loops. This could be controlled by user for more flexibility.

    SmallVector<Operation *, 4> combinerOps;
    if (!matchReduction(linalgOp.getRegionOutputArgs(), 0, combinerOps) ||
        combinerOps.size() != 1)
      return op->emitOpError("Failed to anaysis the reduction operation.");

    Operation *reductionOp = combinerOps[0];
    std::optional<TypedAttr> identity = arith::getNeutralElement(reductionOp);
    if (!identity.has_value())
      return op->emitOpError(
          "Failed to get an identity value for the reduction operation.");

    ArrayRef<int64_t> oldShape =
        linalgOp.getShape(linalgOp.getDpsInitOperand(0));

    // Extend tile size vector to the rank of the output tensor.
    SmallVector<Value> tileSizeVector =
        getValueOrCreateConstantIndexOp(b, loc, sizes);
    if (tileSizeVector.size() < oldShape.size()) {
      auto zero = b.create<arith::ConstantIndexOp>(loc, 0);
      tileSizeVector.append(oldShape.size() - tileSizeVector.size(), zero);
    }

    // Calculate the new shape, we insert the new dimensions based on the index
    // of the reduction dimensions.
    SmallVector<int64_t> newOutputShape;
    SmallVector<Value> dynamicDims;
    int64_t currReductionDims = 0;
    DenseSet<int> reductionDimsSet(reductionDims.begin(), reductionDims.end());
    for (int64_t idx :
         llvm::seq<int64_t>(0, oldShape.size() + reductionDims.size())) {
      if (reductionDimsSet.contains(idx)) {
        dispatchIndexOpFoldResults(sizes[idx], dynamicDims, newOutputShape);
        currReductionDims++;
        continue;
      }
      int64_t oldIdx = idx - currReductionDims;
      int64_t dim = oldShape[oldIdx];
      newOutputShape.push_back(dim);
      if (ShapedType::isDynamic(dim))
        dynamicDims.push_back(b.create<tensor::DimOp>(
            loc, linalgOp.getDpsInitOperand(0)->get(), oldIdx));
    }
    Value emptyTensor = b.create<tensor::EmptyOp>(
        loc, newOutputShape, linalgOp.getRegionOutputArgs()[0].getType(),
        dynamicDims);
    Value constantOp = b.create<arith::ConstantOp>(loc, *identity);
    auto identityTensor =
        b.create<linalg::FillOp>(loc, constantOp, emptyTensor);
    return identityTensor.getOperation();
  }

  Operation *tileToPartialReduction(Operation *op, OpBuilder &b, Location loc,
                                    ValueRange init,
                                    ArrayRef<OpFoldResult> offsets,
                                    ArrayRef<OpFoldResult> sizes,
                                    ArrayRef<int> reductionDims) const {
    OpBuilder::InsertionGuard guard(b);
    auto linalgOp = cast<LinalgOp>(op);

    AffineMap oldOutputMap =
        linalgOp.getMatchingIndexingMap(linalgOp.getDpsInitOperand(0));
    SmallVector<AffineExpr> outputExpr(oldOutputMap.getNumResults() +
                                       reductionDims.size());

    for (int idx : reductionDims)
      outputExpr[idx] = b.getAffineDimExpr(idx);
    int currExpr = 0;
    for (int idx : llvm::seq<int>(0, outputExpr.size())) {
      if (outputExpr[idx])
        continue;
      outputExpr[idx] = oldOutputMap.getResult(currExpr++);
    }

    // Step 1: Extract a slice of the input operands.
    SmallVector<Value> valuesToTile = linalgOp.getDpsInputOperands();
    SmallVector<Value, 4> tiledOperands = makeTiledShapes(
        b, loc, linalgOp, valuesToTile, offsets, sizes, {}, true);

    // Step 2: Extract the accumulator operands
    SmallVector<OpFoldResult> strides(offsets.size(), b.getIndexAttr(1));
    SmallVector<OpFoldResult> outOffsets(offsets.size(), b.getIndexAttr(0));
    // TODO: use SubsetExtractOpInterface once it is available.
    Value out = b.create<tensor::ExtractSliceOp>(loc, init[0], outOffsets,
                                                 sizes, strides);

    // Step3. Create a generic op where the reduction dimensions are replaced
    // by a parallel dimension of the size of reduction.
    SmallVector<utils::IteratorType> newIteratorTypes =
        linalgOp.getIteratorTypesArray();
    for (int dim : reductionDims)
      newIteratorTypes[dim] = utils::IteratorType::parallel;
    SmallVector<AffineMap> newMaps = linalgOp.getIndexingMapsArray();
    newMaps.back() = AffineMap::get(newMaps.back().getNumDims(), 0, outputExpr,
                                    linalgOp.getContext());
    auto genericOp =
        b.create<GenericOp>(loc, TypeRange({out.getType()}), tiledOperands,
                            ValueRange({out}), newMaps, newIteratorTypes);
    IRMapping mapping;
    op->getRegion(0).cloneInto(&genericOp.getRegion(),
                               genericOp.getRegion().begin(), mapping);
    return genericOp.getOperation();
  }

  Operation *mergeReductions(Operation *op, OpBuilder &b, Location loc,
                             ValueRange partialReduce,
                             ArrayRef<int> reductionDims) const {
    auto linalgOp = cast<LinalgOp>(op);

    DenseSet<int> reductionDimsSet(reductionDims.begin(), reductionDims.end());

    // Then create a new reduction that only reduce the newly added dimensions
    // from the previous op.
    int64_t intermRank = cast<ShapedType>(partialReduce[0].getType()).getRank();
    AffineMap inputMap = b.getMultiDimIdentityMap(intermRank);
    SmallVector<utils::IteratorType> reductionIteratorTypes;
    SmallVector<AffineExpr> exprs;

    for (int64_t i : llvm::seq<int64_t>(0, intermRank)) {
      if (reductionDimsSet.contains(i)) {
        reductionIteratorTypes.push_back(utils::IteratorType::reduction);
      } else {
        exprs.push_back(b.getAffineDimExpr(i));
        reductionIteratorTypes.push_back(utils::IteratorType::parallel);
      }
    }

    AffineMap outputMap =
        AffineMap::get(intermRank, 0, exprs, op->getContext());
    SmallVector<AffineMap> reductionMaps = {inputMap, outputMap};

    SmallVector<Operation *, 4> combinerOps;
    matchReduction(linalgOp.getRegionOutputArgs(), 0, combinerOps);
    Operation *reductionOp = combinerOps[0];

    auto reduction = b.create<GenericOp>(
        loc, op->getResultTypes(), ValueRange({partialReduce[0]}),
        SmallVector<Value>{linalgOp.getDpsInitOperands()}, reductionMaps,
        reductionIteratorTypes,
        [reductionOp](OpBuilder &b, Location loc, ValueRange inputs) {
          Operation *clonedReductionOp = b.clone(*reductionOp);
          clonedReductionOp->setOperand(0, inputs[0]);
          clonedReductionOp->setOperand(1, inputs[1]);
          b.create<linalg::YieldOp>(loc, clonedReductionOp->getResult(0));
        });
    return reduction.getOperation();
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
    registerAll<
#include "mlir/Dialect/Linalg/IR/LinalgStructuredOps.cpp.inc"
        >(ctx);
  });
}
