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
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/StandardOps/EDSC/Intrinsics.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "linalg-drop-unit-dims"

using namespace mlir;
using namespace mlir::edsc;
using namespace mlir::edsc::intrinsics;
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

/// Modify the region of indexed generic op to drop arguments corresponding to
/// loops that are unit trip count.
template <typename OpTy>
static LogicalResult
replaceBlockArgForUnitDimLoops(OpTy op, const DenseSet<unsigned> &unitDims,
                               PatternRewriter &rewriterp) {
  return success();
}

template <>
LogicalResult replaceBlockArgForUnitDimLoops<IndexedGenericOp>(
    IndexedGenericOp op, const DenseSet<unsigned> &unitDims,
    PatternRewriter &rewriter) {
  OpBuilder::InsertionGuard guard(rewriter);
  Block *entryBlock = &op->getRegion(0).front();
  rewriter.setInsertionPointToStart(entryBlock);
  Value zero = rewriter.create<ConstantIndexOp>(op.getLoc(), 0);
  for (unsigned unitDimLoop : unitDims) {
    entryBlock->getArgument(unitDimLoop).replaceAllUsesWith(zero);
  }
  SmallVector<unsigned, 8> unitDimsToErase(unitDims.begin(), unitDims.end());
  entryBlock->eraseArguments(unitDimsToErase);
  return success();
}

namespace {
/// Pattern to fold unit-trip count loops in GenericOps.
// TODO: Generalize this to indexed-generic as well by modifying the region args
// as well.
template <typename GenericOpTy>
struct FoldUnitDimLoops : public OpRewritePattern<GenericOpTy> {
  using OpRewritePattern<GenericOpTy>::OpRewritePattern;
  LogicalResult matchAndRewrite(GenericOpTy op,
                                PatternRewriter &rewriter) const override {
    SmallVector<AffineMap, 4> indexingMaps = op.getIndexingMaps();
    if (indexingMaps.empty())
      return failure();

    // Check if any of the iteration dimensions are unit-trip count. They will
    // end up being unit-trip count if they are used to index into a unit-dim
    // tensor/memref.
    AffineMap invertedMap = inversePermutation(concatAffineMaps(indexingMaps));
    if (!invertedMap)
      return failure();
    SmallVector<int64_t, 4> dims;
    for (ShapedType shapedType : op.getShapedOperandTypes())
      dims.append(shapedType.getShape().begin(), shapedType.getShape().end());
    DenseSet<unsigned> unitDims;
    ArrayAttr iteratorTypes = op.iterator_types();
    for (auto expr : enumerate(invertedMap.getResults())) {
      if (AffineDimExpr dimExpr = expr.value().dyn_cast<AffineDimExpr>())
        if (dims[dimExpr.getPosition()] == 1 &&
            iteratorTypes[expr.index()].dyn_cast<StringAttr>().getValue() ==
                getParallelIteratorTypeName())
          unitDims.insert(expr.index());
    }
    if (unitDims.empty())
      return failure();

    // Compute the modified indexing maps.
    MLIRContext *context = rewriter.getContext();
    ArrayAttr newIndexingMapAttr =
        replaceUnitDims(unitDims, indexingMaps, context);
    if (!newIndexingMapAttr)
      return op.emitError("unable to compute modified indexing_maps");

    // Compute the iterator types of the modified op by dropping the one-trip
    // count loops.
    SmallVector<Attribute, 4> newIteratorTypes;
    for (auto attr : llvm::enumerate(iteratorTypes)) {
      if (!unitDims.count(attr.index()))
        newIteratorTypes.push_back(attr.value());
    }

    rewriter.startRootUpdate(op);
    op.indexing_mapsAttr(newIndexingMapAttr);
    op.iterator_typesAttr(ArrayAttr::get(context, newIteratorTypes));
    (void)replaceBlockArgForUnitDimLoops(op, unitDims, rewriter);
    rewriter.finalizeRootUpdate(op);
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
static UnitExtentReplacementInfo replaceUnitExtents(AffineMap indexMap,
                                                    RankedTensorType type,
                                                    MLIRContext *context) {
  ArrayRef<int64_t> shape = type.getShape();
  ArrayRef<AffineExpr> exprs = indexMap.getResults();
  SmallVector<AffineExpr, 2> reassociations;
  SmallVector<Attribute, 4> reassociationMaps;
  SmallVector<AffineExpr, 4> newIndexExprs;
  SmallVector<int64_t, 4> newShape;

  int64_t origRank = type.getRank();
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
        origRank, /*numSymbols = */ 0, reassociations, context)));
    reassociations.clear();
    ++dim;
  }
  UnitExtentReplacementInfo info = {
      RankedTensorType::get(newShape, type.getElementType()),
      AffineMap::get(indexMap.getNumDims(), indexMap.getNumSymbols(),
                     newIndexExprs, context),
      ArrayAttr::get(context, reassociationMaps)};
  return info;
}

namespace {

/// Pattern to replace tensors operands/results that are unit extents.
template <typename GenericOpTy>
struct ReplaceUnitExtentTensors : public OpRewritePattern<GenericOpTy> {
  using OpRewritePattern<GenericOpTy>::OpRewritePattern;
  LogicalResult matchAndRewrite(GenericOpTy op,
                                PatternRewriter &rewriter) const override {
    // TODO: support init_tensors and reductions.
    if (!op.hasTensorSemantics() || op.getNumInitTensors() != 0)
      return failure();

    MLIRContext *context = rewriter.getContext();
    Location loc = op.getLoc();

    SmallVector<AffineMap, 4> newIndexingMaps;
    SmallVector<ArrayAttr, 4> reassociationMaps;
    SmallVector<ShapedType, 4> newInputOutputTypes;
    bool doCanonicalization = false;
    for (auto it :
         llvm::zip(op.getIndexingMaps(), op.getShapedOperandTypes())) {
      auto replacementInfo = replaceUnitExtents(
          std::get<0>(it), std::get<1>(it).template cast<RankedTensorType>(),
          context);
      reassociationMaps.push_back(replacementInfo.reassociation);
      newIndexingMaps.push_back(replacementInfo.indexMap);
      newInputOutputTypes.push_back(replacementInfo.type);
      doCanonicalization |= replacementInfo.type != std::get<1>(it);
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
        else
          res.push_back(rewriter.create<linalg::TensorReshapeOp>(
              loc, newInputOutputTypes[flattenedIdx], operand.value(),
              reassociationMaps[flattenedIdx]));
        ++flattenedIdx;
      }
      return res;
    };

    SmallVector<Value, 4> newInputs = insertReshapes(op.inputs());
    SmallVector<Value, 4> newOutputs = insertReshapes(op.outputs());

    // If any result type changes, insert a reshape to convert from the original
    // type to the new type.
    SmallVector<Type, 4> resultTypes;
    resultTypes.reserve(op.getNumResults());
    for (unsigned i : llvm::seq<unsigned>(0, op.getNumResults()))
      resultTypes.push_back(newInputOutputTypes[i + op.getNumInputs()]);
    GenericOpTy replacementOp = rewriter.create<GenericOpTy>(
        loc, resultTypes, newInputs, newOutputs, newIndexingMaps,
        llvm::to_vector<4>(
            op.iterator_types().template getAsValueRange<StringAttr>()));
    rewriter.inlineRegionBefore(op.region(), replacementOp.region(),
                                replacementOp.region().begin());

    // If any result tensor has a modified shape, then add reshape to recover
    // the original shape.
    SmallVector<Value, 4> resultReplacements;
    for (auto result : llvm::enumerate(replacementOp.getResults())) {
      unsigned index = result.index() + replacementOp.getNumInputs();
      RankedTensorType origResultType = op.getResult(result.index())
                                            .getType()
                                            .template cast<RankedTensorType>();
      if (origResultType != result.value().getType())
        resultReplacements.push_back(rewriter.create<linalg::TensorReshapeOp>(
            loc, origResultType, result.value(), reassociationMaps[index]));
      else
        resultReplacements.push_back(result.value());
    }
    rewriter.replaceOp(op, resultReplacements);
    return success();
  }
};
} // namespace

namespace {
/// Pattern to fold pair of reshape ops where the intermediate has unit-dims for
/// example:
///
///  %0 = linalg.tensor_reshape %arg0
///    [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>]
///    : tensor<2048xf32> into tensor<1x4x1x512xf32>
///  %1 = linalg.tensor_reshape %0
///    [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>,
///     affine_map<(d0, d1, d2, d3) -> (d3)>]
///    : tensor<1x4x1x512xf32> into tensor<4x512xf32>
///
/// can be replaced with
///
///  %0 = linalg.tensor_reshape %arg0 [affine_map<(d0, d1) -> (d0, d1)>]
///    : tensor<2048xf32> into tensor<4x512xf32>
///
/// Similarly,
///
///  %0 = linalg.tensor_reshape %arg0
///    [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>,
///     affine_map<(d0, d1, d2, d3) -> (d3)>]
///    : tensor<4x512xf32> into tensor<1x4x1x512xf32>
///  %1 = linalg.tensor_reshape %0
///   [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>]
///    : tensor<1x4x1x512xf32> into tensor<2048xf32>
///
/// can be replaced with
///
///  %0 = linalg.tensor_reshape %arg0 [affine_map<(d0, d1) -> (d0, d1)>]
///    : tensor<4x512xf32> into tensor<2048xf32>
struct FoldReshapeOpWithUnitExtent : OpRewritePattern<TensorReshapeOp> {
  using OpRewritePattern<TensorReshapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TensorReshapeOp reshapeOp,
                                PatternRewriter &rewriter) const override {
    // Check that the source operand is created from a reshape as well.
    TensorReshapeOp parentReshapeOp =
        reshapeOp.src().getDefiningOp<TensorReshapeOp>();
    if (!parentReshapeOp)
      return failure();

    RankedTensorType srcType = reshapeOp.getSrcType(),
                     dstType = reshapeOp.getResultType(),
                     parentSrcType = parentReshapeOp.getSrcType();
    if (!srcType.hasStaticShape() || !dstType.hasStaticShape() ||
        !parentSrcType.hasStaticShape() ||
        srcType.getRank() < dstType.getRank() ||
        parentSrcType.getRank() == dstType.getRank())
      return failure();

    // Check if the result tensor_reshape after folding the reshapeOp and
    // parentReshapeOp are combined.
    // If the final tensor_reshape is folding, the parentReshapeOp is
    // introducing unit-dims, and the reshapeOp does an actual reshape.
    // If the final tensor_reshape op is expanding, the reshapeOp is
    // introducing unit-dims, and the parentReshapeOp does an actual reshape.
    bool isFoldingPattern = parentSrcType.getRank() > dstType.getRank();
    ArrayRef<int64_t> expandedShape =
        isFoldingPattern ? parentSrcType.getShape() : dstType.getShape();
    ArrayRef<int64_t> foldedShape =
        isFoldingPattern ? dstType.getShape() : parentSrcType.getShape();

    unsigned expandedDim = 0, foldedDim = 0;
    SmallVector<SmallVector<AffineExpr, 4>, 4> reassociationExprs(
        foldedShape.size());
    while (expandedDim < expandedShape.size() &&
           foldedDim < foldedShape.size()) {
      int64_t dstSize = foldedShape[foldedDim];
      int64_t srcSize = expandedShape[expandedDim];
      while (srcSize < dstSize && expandedDim < expandedShape.size()) {
        reassociationExprs[foldedDim].push_back(
            rewriter.getAffineDimExpr(expandedDim++));
        srcSize *= expandedShape[expandedDim];
      }
      if (srcSize == dstSize) {
        reassociationExprs[foldedDim].push_back(
            rewriter.getAffineDimExpr(expandedDim++));
        // If the next dim in foldedShape is not 1, treat subsequent dims in
        // expandedShape which are 1 to be collapsed.
        if (foldedDim == foldedShape.size() - 1 ||
            foldedShape[foldedDim + 1] != 1) {
          while (expandedDim < expandedShape.size() &&
                 expandedShape[expandedDim] == 1) {
            reassociationExprs[foldedDim].push_back(
                rewriter.getAffineDimExpr(expandedDim++));
          }
        }
      } else {
        return failure();
      }
      foldedDim++;
    }
    if (expandedDim != expandedShape.size())
      return failure();

    SmallVector<AffineMap, 4> reassociationMaps =
        llvm::to_vector<4>(llvm::map_range(
            reassociationExprs, [&](ArrayRef<AffineExpr> exprs) -> AffineMap {
              return AffineMap::get(expandedShape.size(), 0, exprs,
                                    rewriter.getContext());
            }));
    rewriter.replaceOpWithNewOp<TensorReshapeOp>(
        reshapeOp, dstType, parentReshapeOp.src(),
        rewriter.getAffineMapArrayAttr(reassociationMaps));
    return success();
  }
};
} // namespace

/// Patterns that are used to canonicalize the use of unit-extent dims for
/// broadcasting.
void mlir::populateLinalgFoldUnitExtentDimsPatterns(
    RewritePatternSet &patterns) {
  auto *context = patterns.getContext();
  patterns.add<FoldUnitDimLoops<GenericOp>, FoldUnitDimLoops<IndexedGenericOp>,
               ReplaceUnitExtentTensors<GenericOp>,
               ReplaceUnitExtentTensors<IndexedGenericOp>>(context);
  TensorReshapeOp::getCanonicalizationPatterns(patterns, context);
  patterns.add<FoldReshapeOpWithUnitExtent>(context);
  populateFoldUnitDimsReshapeOpsByLinearizationPatterns(patterns);
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
      patterns
          .add<FoldUnitDimLoops<GenericOp>, FoldUnitDimLoops<IndexedGenericOp>>(
              context);
    else
      populateLinalgFoldUnitExtentDimsPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(funcOp.getBody(), std::move(patterns));
  }
};
} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::createLinalgFoldUnitExtentDimsPass() {
  return std::make_unique<LinalgFoldUnitExtentDimsPass>();
}
