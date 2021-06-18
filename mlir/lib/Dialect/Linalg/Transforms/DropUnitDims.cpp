//===- DropUnitDims.cpp - Pass to drop use of unit-extent for broadcasting ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements patterns/pass to remove usage of unit-extent dimensions
// to specify broadcasting in favor of more canonical representation of the
// computation
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgTypes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "linalg-drop-unit-dims"

using namespace mlir;
using namespace mlir::linalg;

/// Implements a pass that canonicalizes the uses of unit-extent dimensions for
/// broadcasting. For example,
///
/// ```mlir
/// #accesses = [
///   affine_map<(d0, d1) -> (0, d1)>,
///   affine_map<(d0, d1) -> (d0, 0)>,
///   affine_map<(d0, d1) -> (d0, d1)>
/// ]
///
/// #trait = {
///   args_in = 2,
///   args_out = 1,
///   indexing_maps = #accesses,
///   iterator_types = ["parallel", "parallel"],
///   library_call = "some_external_fn"
/// }
///
/// func @broadcast_test(%arg0 : tensor<5xf32>, %arg1 : tensor<5xf32>) ->
/// tensor<5x5xf32>
/// {
///   %0 = linalg.tensor_reshape %arg0 [affine_map<(d0, d1) -> (d0, d1)>] :
///        tensor<5xf32> into tensor<1x5xf32>
///   %1 = linalg.tensor_reshape %arg1 [affine_map<(d0, d1) -> (d0, d1)>] :
///        tensor<5xf32> into tensor<5x1xf32>
///   %2 = linalg.generic #trait %0, %1 {
///        ^bb0(%arg2: f32, %arg3: f32):
///          %3 = addf %arg2, %arg3 : f32
///          linalg.yield %3 : f32
///        } : tensor<1x5xf32>, tensor<5x1xf32> -> tensor<5x5xf32>
///   return %2 : tensor<5x5xf32>
/// }
///
/// would canonicalize to
///
/// ```mlir
/// #accesses = [
///   affine_map<(d0, d1) -> (d1)>,
///   affine_map<(d0, d1) -> (d0)>,
///   affine_map<(d0, d1) -> (d0, d1)>
/// ]
///
/// #trait = {
///   args_in = 2,
///   args_out = 1,
///   indexing_maps = #accesses,
///   iterator_types = ["parallel", "parallel"],
///   library_call = "some_external_fn"
/// }
///
/// func @broadcast_test(%arg0 : tensor<5xf32>, %arg1 : tensor<5xf32>) ->
/// tensor<5x5xf32>
/// {
///   %0 = linalg.generic #trait %arg0, %arg1 {
///        ^bb0(%arg2: f32, %arg3: f32):
///          %3 = addf %arg2, %arg3 : f32
///          linalg.yield %3 : f32
///        } : tensor<5xf32>, tensor<5xf32> -> tensor<5x5xf32>
///   return %0 : tensor<5x5xf32>
/// }

/// Given dims of the iteration space of a structured op that are known to be
/// single trip count (`unitDims`), return the indexing maps to use in the
/// canonicalized op with these dims removed, given the original `indexingMaps`.
static ArrayAttr replaceUnitDims(DenseSet<unsigned> &unitDims,
                                 ArrayRef<AffineMap> indexingMaps,
                                 MLIRContext *context) {
  if (indexingMaps.empty())
    return nullptr;
  unsigned numIterationDims = indexingMaps.front().getNumDims();
  unsigned numSymbols = indexingMaps.front().getNumSymbols();

  // Compute the replacement for each dim expr.
  SmallVector<AffineExpr, 4> dimReplacements;
  dimReplacements.reserve(numIterationDims);
  unsigned numKeptDims = 0;
  for (unsigned dim : llvm::seq<unsigned>(0, numIterationDims)) {
    if (unitDims.count(dim))
      dimReplacements.push_back(getAffineConstantExpr(0, context));
    else
      dimReplacements.push_back(getAffineDimExpr(numKeptDims++, context));
  }

  // Symbols remain the same.
  SmallVector<AffineExpr, 4> symReplacements;
  symReplacements.reserve(numSymbols);
  for (unsigned symbol : llvm::seq<unsigned>(0, numSymbols))
    symReplacements.push_back(getAffineSymbolExpr(symbol, context));

  SmallVector<AffineMap, 4> newIndexingMaps;
  newIndexingMaps.reserve(indexingMaps.size());
  for (AffineMap operandMap : indexingMaps) {
    // Expected indexing maps to have no symbols.
    if (operandMap.getNumSymbols())
      return nullptr;
    newIndexingMaps.push_back(simplifyAffineMap(
        operandMap.replaceDimsAndSymbols(dimReplacements, symReplacements,
                                         numIterationDims - unitDims.size(),
                                         numSymbols)));
  }

  // Check that the new index maps are invertible. If not, something went
  // wrong, so abort.
  if (!inversePermutation(concatAffineMaps(newIndexingMaps)))
    return nullptr;
  return ArrayAttr::get(context,
                        llvm::to_vector<4>(llvm::map_range(
                            newIndexingMaps, [](AffineMap map) -> Attribute {
                              return AffineMapAttr::get(map);
                            })));
}

/// Update the index accesses of linalg operations having index semantics.
static void replaceUnitDimIndexOps(GenericOp genericOp,
                                   const DenseSet<unsigned> &unitDims,
                                   PatternRewriter &rewriter) {
  assert(genericOp->getNumRegions() == 1 &&
         genericOp->getRegion(0).getBlocks().size() == 1 &&
         "expected generic operation to have one block.");
  Block &block = genericOp->getRegion(0).front();

  for (IndexOp indexOp : llvm::make_early_inc_range(block.getOps<IndexOp>())) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(indexOp);
    if (unitDims.count(indexOp.dim()) != 0) {
      rewriter.replaceOpWithNewOp<ConstantIndexOp>(indexOp, 0);
    } else {
      // Update the dimension of the index operation if needed.
      unsigned droppedDims = llvm::count_if(
          unitDims, [&](unsigned dim) { return dim < indexOp.dim(); });
      if (droppedDims != 0)
        rewriter.replaceOpWithNewOp<IndexOp>(indexOp,
                                             indexOp.dim() - droppedDims);
    }
  }
}

namespace {
/// Pattern to fold unit-trip count loops in GenericOps.
struct FoldUnitDimLoops : public OpRewritePattern<GenericOp> {
  using OpRewritePattern<GenericOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    SmallVector<AffineMap, 4> indexingMaps = genericOp.getIndexingMaps();
    if (indexingMaps.empty())
      return failure();

    // Check if any of the iteration dimensions are unit-trip count. They will
    // end up being unit-trip count if they are used to index into a unit-dim
    // tensor/memref.
    AffineMap invertedMap = inversePermutation(concatAffineMaps(indexingMaps));
    if (!invertedMap)
      return failure();
    SmallVector<int64_t> dims = genericOp.getStaticShape();

    // Find all the reduction iterators. Those need some special consideration
    // (see below).
    auto getLoopDimsOfType =
        [&](StringRef iteratorTypeName) -> SmallVector<unsigned, 4> {
      SmallVector<AffineExpr> dimExprs;
      getDimsOfType(genericOp, iteratorTypeName, dimExprs);
      return llvm::to_vector<4>(llvm::map_range(dimExprs, [](AffineExpr expr) {
        return expr.cast<AffineDimExpr>().getPosition();
      }));
    };
    auto reductionDims = getLoopDimsOfType(getReductionIteratorTypeName());

    DenseSet<unsigned> unitDims;
    SmallVector<unsigned, 4> unitDimsReductionLoops;
    ArrayAttr iteratorTypes = genericOp.iterator_types();
    for (auto expr : enumerate(invertedMap.getResults())) {
      if (AffineDimExpr dimExpr = expr.value().dyn_cast<AffineDimExpr>())
        if (dims[dimExpr.getPosition()] == 1) {
          if (isParallelIterator(iteratorTypes[expr.index()]))
            unitDims.insert(expr.index());
          else if (isReductionIterator(iteratorTypes[expr.index()]))
            unitDimsReductionLoops.push_back(expr.index());
        }
    }

    // Reduction loops can be dropped if there is at least one other reduction
    // loop that is not dropped. This accounts for the initial value read in the
    // reduction loop.
    if (!unitDimsReductionLoops.empty() && reductionDims.size() > 1) {
      if (unitDimsReductionLoops.size() == reductionDims.size())
        unitDims.insert(reductionDims.begin(), std::prev(reductionDims.end()));
      else
        unitDims.insert(unitDimsReductionLoops.begin(),
                        unitDimsReductionLoops.end());
    }

    if (unitDims.empty())
      return failure();

    // Compute the modified indexing maps.
    MLIRContext *context = rewriter.getContext();
    ArrayAttr newIndexingMapAttr =
        replaceUnitDims(unitDims, indexingMaps, context);
    if (!newIndexingMapAttr)
      return genericOp.emitError("unable to compute modified indexing_maps");

    // Compute the iterator types of the modified op by dropping the one-trip
    // count loops.
    SmallVector<Attribute, 4> newIteratorTypes;
    for (auto attr : llvm::enumerate(iteratorTypes)) {
      if (!unitDims.count(attr.index()))
        newIteratorTypes.push_back(attr.value());
    }

    rewriter.startRootUpdate(genericOp);
    genericOp.indexing_mapsAttr(newIndexingMapAttr);
    genericOp.iterator_typesAttr(ArrayAttr::get(context, newIteratorTypes));
    replaceUnitDimIndexOps(genericOp, unitDims, rewriter);
    rewriter.finalizeRootUpdate(genericOp);
    return success();
  }
};

struct UnitExtentReplacementInfo {
  RankedTensorType type;
  AffineMap indexMap;
  ArrayAttr reassociation;
};
} // namespace

/// Utility function for replacing operands/results to a linalg generic
/// operation on tensors with unit-extent dimensions. These can be replaced with
/// an operand/result with the unit-extent dimension removed. This is only done
/// if the indexing map used to access that didimensionmension has a
/// AffineConstantExpr of value 0. Given the `type` of an result/operand of a
/// Linalg op, and its `indexMap` the utility function returns:
/// - the new type with dimensions of size 1 removed.
/// - modified index map that can be used to access the replaced result/operand
/// - the reassociation that converts from the original tensor type to the
///   modified tensor type.
static UnitExtentReplacementInfo replaceUnitExtents(GenericOp genericOp,
                                                    OpOperand *opOperand,
                                                    MLIRContext *context) {
  AffineMap indexingMap = genericOp.getTiedIndexingMap(opOperand);
  ArrayRef<int64_t> shape = genericOp.getShape(opOperand);
  ArrayRef<AffineExpr> exprs = indexingMap.getResults();
  SmallVector<AffineExpr, 2> reassociations;
  SmallVector<Attribute, 4> reassociationMaps;
  SmallVector<AffineExpr, 4> newIndexExprs;
  SmallVector<int64_t, 4> newShape;

  int64_t origRank = genericOp.getRank(opOperand);
  AffineExpr zeroExpr = getAffineConstantExpr(0, context);
  auto isUnitExtent = [&](int64_t dim) -> bool {
    return shape[dim] == 1 && exprs[dim] == zeroExpr;
  };

  unsigned dim = 0;
  // Fold dimensions that are unit-extent at the beginning of the tensor.
  while (dim < origRank && isUnitExtent(dim))
    reassociations.push_back(getAffineDimExpr(dim++, context));
  while (dim < origRank) {
    reassociations.push_back(getAffineDimExpr(dim, context));
    newIndexExprs.push_back(exprs[dim]);
    newShape.push_back(shape[dim]);
    // Fold all following dimensions that are unit-extent.
    while (dim + 1 < origRank && isUnitExtent(dim + 1)) {
      ++dim;
      reassociations.push_back(getAffineDimExpr(dim, context));
    }
    reassociationMaps.push_back(AffineMapAttr::get(AffineMap::get(
        origRank, /*symbolCount = */ 0, reassociations, context)));
    reassociations.clear();
    ++dim;
  }
  UnitExtentReplacementInfo info = {
      RankedTensorType::get(newShape,
                            getElementTypeOrSelf(opOperand->get().getType())),
      AffineMap::get(indexingMap.getNumDims(), indexingMap.getNumSymbols(),
                     newIndexExprs, context),
      ArrayAttr::get(context, reassociationMaps)};
  return info;
}

namespace {

SmallVector<ReassociationExprs, 2>
convertAffineMapArrayToExprs(ArrayAttr affineMapArrayAttr) {
  SmallVector<ReassociationExprs, 2> reassociationExprs;
  for (auto attr : affineMapArrayAttr)
    reassociationExprs.push_back(
        llvm::to_vector<4>(attr.cast<AffineMapAttr>().getValue().getResults()));
  return reassociationExprs;
}

/// Pattern to replace tensors operands/results that are unit extents.
struct ReplaceUnitExtentTensors : public OpRewritePattern<GenericOp> {
  using OpRewritePattern<GenericOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    if (!genericOp.hasTensorSemantics())
      return failure();

    MLIRContext *context = rewriter.getContext();
    Location loc = genericOp.getLoc();

    SmallVector<AffineMap, 4> newIndexingMaps;
    SmallVector<ArrayAttr, 4> reassociationMaps;
    SmallVector<ShapedType, 4> newInputOutputTypes;
    bool doCanonicalization = false;

    for (OpOperand *opOperand : genericOp.getInputAndOutputOperands()) {
      auto replacementInfo = replaceUnitExtents(genericOp, opOperand, context);
      reassociationMaps.push_back(replacementInfo.reassociation);
      newIndexingMaps.push_back(replacementInfo.indexMap);
      newInputOutputTypes.push_back(replacementInfo.type);
      doCanonicalization |= replacementInfo.type != opOperand->get().getType();
    }

    // If the indexing maps of the result operation are not invertible (i.e. not
    // legal), abort.
    if (!doCanonicalization ||
        !inversePermutation(concatAffineMaps(newIndexingMaps)))
      return failure();

    // If any operand type change, insert a reshape to convert from the original
    // type to the new type.
    // TODO: get rid of flattenedIdx which assumes operand order and contiguity.
    unsigned flattenedIdx = 0;
    auto insertReshapes = [&](ValueRange values) {
      SmallVector<Value, 4> res;
      res.reserve(values.size());
      for (auto operand : llvm::enumerate(values)) {
        if (operand.value().getType() == newInputOutputTypes[flattenedIdx])
          res.push_back(operand.value());
        else {
          res.push_back(rewriter.create<TensorCollapseShapeOp>(
              loc, newInputOutputTypes[flattenedIdx], operand.value(),
              convertAffineMapArrayToExprs(reassociationMaps[flattenedIdx])));
        }
        ++flattenedIdx;
      }
      return res;
    };

    SmallVector<Value, 4> newInputs = insertReshapes(genericOp.inputs());
    SmallVector<Value, 4> newOutputs = insertReshapes(genericOp.outputs());

    // If any result type changes, insert a reshape to convert from the original
    // type to the new type.
    SmallVector<Type, 4> resultTypes;
    resultTypes.reserve(genericOp.getNumResults());
    for (unsigned i : llvm::seq<unsigned>(0, genericOp.getNumResults()))
      resultTypes.push_back(newInputOutputTypes[i + genericOp.getNumInputs()]);
    GenericOp replacementOp = rewriter.create<GenericOp>(
        loc, resultTypes, newInputs, newOutputs, newIndexingMaps,
        llvm::to_vector<4>(
            genericOp.iterator_types().template getAsValueRange<StringAttr>()));
    rewriter.inlineRegionBefore(genericOp.region(), replacementOp.region(),
                                replacementOp.region().begin());

    // If any result tensor has a modified shape, then add reshape to recover
    // the original shape.
    SmallVector<Value, 4> resultReplacements;
    for (auto result : llvm::enumerate(replacementOp.getResults())) {
      unsigned index = result.index() + replacementOp.getNumInputs();
      RankedTensorType origResultType = genericOp.getResult(result.index())
                                            .getType()
                                            .template cast<RankedTensorType>();
      if (origResultType != result.value().getType()) {
        resultReplacements.push_back(rewriter.create<TensorExpandShapeOp>(
            loc, origResultType, result.value(),
            convertAffineMapArrayToExprs(reassociationMaps[index])));
      } else
        resultReplacements.push_back(result.value());
    }
    rewriter.replaceOp(genericOp, resultReplacements);
    return success();
  }
};
} // namespace

/// Get the reassociation maps to fold the result of a subtensor (or source of a
/// subtensor_insert) operation with given offsets, and sizes to its
/// rank-reduced version. This is only done for the cases where the size is 1
/// and offset is 0. Strictly speaking the offset 0 is not required in general,
/// but non-zero offsets are not handled by SPIR-V backend at this point (and
/// potentially cannot be handled).
static Optional<SmallVector<ReassociationIndices>>
getReassociationMapForFoldingUnitDims(ArrayRef<OpFoldResult> mixedSizes) {
  SmallVector<ReassociationIndices> reassociation;
  ReassociationIndices curr;
  for (auto it : llvm::enumerate(mixedSizes)) {
    auto dim = it.index();
    auto size = it.value();
    curr.push_back(dim);
    auto attr = size.dyn_cast<Attribute>();
    if (attr && attr.cast<IntegerAttr>().getInt() == 1)
      continue;
    reassociation.emplace_back(ReassociationIndices{});
    std::swap(reassociation.back(), curr);
  }
  // When the reassociations are not empty, then fold the remaining
  // unit-dimensions into the last dimension.  If the reassociations so far is
  // empty, then leave it emtpy. This will fold everything to a rank-0 tensor.
  if (!curr.empty() && !reassociation.empty())
    reassociation.back().append(curr.begin(), curr.end());
  return reassociation;
}

namespace {
/// Convert `subtensor` operations to rank-reduced versions.
struct UseRankReducedSubTensorOp : public OpRewritePattern<SubTensorOp> {
  using OpRewritePattern<SubTensorOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SubTensorOp subTensorOp,
                                PatternRewriter &rewriter) const override {
    RankedTensorType resultType = subTensorOp.getType();
    SmallVector<OpFoldResult> offsets = subTensorOp.getMixedOffsets();
    SmallVector<OpFoldResult> sizes = subTensorOp.getMixedSizes();
    SmallVector<OpFoldResult> strides = subTensorOp.getMixedStrides();
    auto reassociation = getReassociationMapForFoldingUnitDims(sizes);
    if (!reassociation ||
        reassociation->size() == static_cast<size_t>(resultType.getRank()))
      return failure();
    auto rankReducedType =
        SubTensorOp::inferRankReducedResultType(reassociation->size(),
                                                subTensorOp.getSourceType(),
                                                offsets, sizes, strides)
            .cast<RankedTensorType>();

    Location loc = subTensorOp.getLoc();
    Value newSubTensor = rewriter.create<SubTensorOp>(
        loc, rankReducedType, subTensorOp.source(), offsets, sizes, strides);
    rewriter.replaceOpWithNewOp<TensorExpandShapeOp>(
        subTensorOp, resultType, newSubTensor, *reassociation);
    return success();
  }
};

/// Convert `subtensor_insert` operations to rank-reduced versions.
struct UseRankReducedSubTensorInsertOp
    : public OpRewritePattern<SubTensorInsertOp> {
  using OpRewritePattern<SubTensorInsertOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SubTensorInsertOp insertOp,
                                PatternRewriter &rewriter) const override {
    RankedTensorType sourceType = insertOp.getSourceType();
    SmallVector<OpFoldResult> offsets = insertOp.getMixedOffsets();
    SmallVector<OpFoldResult> sizes = insertOp.getMixedSizes();
    SmallVector<OpFoldResult> strides = insertOp.getMixedStrides();
    auto reassociation = getReassociationMapForFoldingUnitDims(sizes);
    if (!reassociation ||
        reassociation->size() == static_cast<size_t>(sourceType.getRank()))
      return failure();
    Location loc = insertOp.getLoc();
    auto reshapedSource = rewriter.create<TensorCollapseShapeOp>(
        loc, insertOp.source(), *reassociation);
    rewriter.replaceOpWithNewOp<SubTensorInsertOp>(
        insertOp, reshapedSource, insertOp.dest(), insertOp.getMixedOffsets(),
        insertOp.getMixedSizes(), insertOp.getMixedStrides());
    return success();
  }
};
} // namespace

/// Patterns that are used to canonicalize the use of unit-extent dims for
/// broadcasting.
void mlir::linalg::populateFoldUnitExtentDimsPatterns(
    RewritePatternSet &patterns) {
  auto *context = patterns.getContext();
  patterns.add<FoldUnitDimLoops, ReplaceUnitExtentTensors,
               UseRankReducedSubTensorOp, UseRankReducedSubTensorInsertOp>(
      context);
  TensorCollapseShapeOp::getCanonicalizationPatterns(patterns, context);
  TensorExpandShapeOp::getCanonicalizationPatterns(patterns, context);
}

namespace {
/// Pass that removes unit-extent dims within generic ops.
struct LinalgFoldUnitExtentDimsPass
    : public LinalgFoldUnitExtentDimsBase<LinalgFoldUnitExtentDimsPass> {
  void runOnFunction() override {
    FuncOp funcOp = getFunction();
    MLIRContext *context = funcOp.getContext();
    RewritePatternSet patterns(context);
    if (foldOneTripLoopsOnly)
      patterns.add<FoldUnitDimLoops>(context);
    else
      populateFoldUnitExtentDimsPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(funcOp.getBody(), std::move(patterns));
  }
};
} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::createLinalgFoldUnitExtentDimsPass() {
  return std::make_unique<LinalgFoldUnitExtentDimsPass>();
}
