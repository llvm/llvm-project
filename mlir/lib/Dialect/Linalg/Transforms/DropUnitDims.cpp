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

#include "mlir/Dialect/Linalg/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

namespace mlir {
#define GEN_PASS_DEF_LINALGFOLDUNITEXTENTDIMS
#include "mlir/Dialect/Linalg/Passes.h.inc"
} // namespace mlir

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
///          %3 = arith.addf %arg2, %arg3 : f32
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
///          %3 = arith.addf %arg2, %arg3 : f32
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
  for (IndexOp indexOp :
       llvm::make_early_inc_range(genericOp.getBody()->getOps<IndexOp>())) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(indexOp);
    if (unitDims.count(indexOp.getDim()) != 0) {
      rewriter.replaceOpWithNewOp<arith::ConstantIndexOp>(indexOp, 0);
    } else {
      // Update the dimension of the index operation if needed.
      unsigned droppedDims = llvm::count_if(
          unitDims, [&](unsigned dim) { return dim < indexOp.getDim(); });
      if (droppedDims != 0)
        rewriter.replaceOpWithNewOp<IndexOp>(indexOp,
                                             indexOp.getDim() - droppedDims);
    }
  }
}

namespace {
/// Pattern to fold unit-trip count loops in GenericOps.
struct FoldUnitDimLoops : public OpRewritePattern<GenericOp> {
  using OpRewritePattern<GenericOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    SmallVector<AffineMap, 4> indexingMaps = genericOp.getIndexingMapsArray();
    if (indexingMaps.empty())
      return failure();

    // Check if any of the iteration dimensions are unit-trip count. They will
    // end up being unit-trip count if they are used to index into a unit-dim
    // tensor/memref.
    AffineMap invertedMap = inversePermutation(concatAffineMaps(indexingMaps));
    if (!invertedMap)
      return failure();
    SmallVector<int64_t> dims = genericOp.getStaticShape();

    DenseSet<unsigned> unitDims;
    SmallVector<unsigned, 4> unitDimsReductionLoops;
    ArrayAttr iteratorTypes = genericOp.getIteratorTypes();
    for (const auto &expr : enumerate(invertedMap.getResults())) {
      if (AffineDimExpr dimExpr = expr.value().dyn_cast<AffineDimExpr>())
        if (dims[dimExpr.getPosition()] == 1)
          unitDims.insert(expr.index());
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
    for (const auto &attr : llvm::enumerate(iteratorTypes)) {
      if (!unitDims.count(attr.index()))
        newIteratorTypes.push_back(attr.value());
    }

    rewriter.startRootUpdate(genericOp);
    genericOp.setIndexingMapsAttr(newIndexingMapAttr);
    genericOp.setIteratorTypesAttr(ArrayAttr::get(context, newIteratorTypes));
    replaceUnitDimIndexOps(genericOp, unitDims, rewriter);
    rewriter.finalizeRootUpdate(genericOp);
    return success();
  }
};

/// Pattern to add init operands to ins when all the loops are parallel and
/// blockArgument corresponding to init is used in the region. This is a fix-up
/// when unit reduction dimensions are all folded away. In this context, it
/// becomes a elementwise generic op. E.g., it converts
///
///  %0 = tensor.empty() : tensor<1x1xf32>
///  %1 = linalg.fill
///    ins(%cst : f32)
///    outs(%0 : tensor<1x1xf32>) -> tensor<1x1xf32>
///  %2 = linalg.generic {indexing_maps = [affine_map<(d0) -> (0, d0, 0, 0)>,
///                                        affine_map<(d0) -> (0, d0)>],
///                       iterator_types = ["parallel"]}
///    ins(%arg0 : tensor<1x?x1x1xf32>)
///    outs(%1 : tensor<1x1xf32>) {
///  ^bb0(%in: f32, %out: f32):
///    %3 = arith.addf %in, %out : f32
///    linalg.yield %3 : f32
///  } -> tensor<1x1xf32>
///
///  into
///
///  %0 = tensor.empty() : tensor<1x1xf32>
///  %1 = linalg.fill
///    ins(%cst : f32)
///    outs(%0 : tensor<1x1xf32>) -> tensor<1x1xf32>
///  %2 = tensor.empty() : tensor<1x1xf32>
///  %3 = linalg.generic {indexing_maps = [affine_map<(d0) -> (0, d0, 0, 0)>,
///                                        affine_map<(d0) -> (0, d0)>,
///                                        affine_map<(d0) -> (0, d0)>],
///                       iterator_types = ["parallel"]}
///   ins(%arg0, %1 : tensor<1x?x1x1xf32>, tensor<1x1xf32>)
///   outs(%2 : tensor<1x1xf32>) {
///  ^bb0(%in: f32, %in_0: f32, %out: f32):
///    %4 = arith.addf %in, %in_0 : f32
///    linalg.yield %4 : f32
///  } -> tensor<1x1xf32>
struct AddInitOperandsToInput : public OpRewritePattern<GenericOp> {
  using OpRewritePattern<GenericOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    if (!genericOp.hasTensorSemantics())
      return failure();
    if (genericOp.getNumParallelLoops() != genericOp.getNumLoops())
      return failure();

    auto outputOperands = genericOp.getDpsInitOperands();
    SetVector<OpOperand *> candidates;
    for (OpOperand *op : outputOperands) {
      if (genericOp.getMatchingBlockArgument(op).use_empty())
        continue;
      candidates.insert(op);
    }

    if (candidates.empty())
      return failure();

    // Compute the modified indexing maps.
    int64_t origNumInput = genericOp.getNumDpsInputs();
    SmallVector<Value> newInputOperands = genericOp.getDpsInputOperands();
    SmallVector<AffineMap> indexingMaps = genericOp.getIndexingMapsArray();
    SmallVector<AffineMap> newIndexingMaps;
    newIndexingMaps.append(indexingMaps.begin(),
                           std::next(indexingMaps.begin(), origNumInput));
    for (OpOperand *op : candidates) {
      newInputOperands.push_back(op->get());
      newIndexingMaps.push_back(genericOp.getMatchingIndexingMap(op));
    }
    newIndexingMaps.append(std::next(indexingMaps.begin(), origNumInput),
                           indexingMaps.end());

    Location loc = genericOp.getLoc();
    SmallVector<Value> newOutputOperands = outputOperands;
    for (OpOperand *op : candidates) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointAfterValue(op->get());
      auto elemType = op->get().getType().cast<ShapedType>().getElementType();
      auto empty = rewriter.create<tensor::EmptyOp>(
          loc, tensor::createDimValues(rewriter, loc, op->get()), elemType);

      auto [start, end] = genericOp.getDpsInitsPositionRange();
      newOutputOperands[op->getOperandNumber() - start] = empty.getResult();
    }

    auto newOp = rewriter.create<GenericOp>(
        loc, genericOp.getResultTypes(), newInputOperands, newOutputOperands,
        newIndexingMaps, genericOp.getIteratorTypesArray(),
        /*bodyBuild=*/nullptr, linalg::getPrunedAttributeList(genericOp));

    Region &region = newOp.getRegion();
    Block *block = new Block();
    region.push_back(block);
    BlockAndValueMapping mapper;
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(block);
    for (auto bbarg : genericOp.getRegionInputArgs())
      mapper.map(bbarg, block->addArgument(bbarg.getType(), loc));

    for (OpOperand *op : candidates) {
      BlockArgument bbarg = genericOp.getMatchingBlockArgument(op);
      mapper.map(bbarg, block->addArgument(bbarg.getType(), loc));
    }

    for (OpOperand *op : outputOperands) {
      BlockArgument bbarg = genericOp.getMatchingBlockArgument(op);
      if (candidates.count(op))
        block->addArgument(bbarg.getType(), loc);
      else
        mapper.map(bbarg, block->addArgument(bbarg.getType(), loc));
    }

    for (auto &op : genericOp.getBody()->getOperations()) {
      rewriter.clone(op, mapper);
    }
    rewriter.replaceOp(genericOp, newOp.getResults());

    return success();
  }
};

struct UnitExtentReplacementInfo {
  Type type;
  AffineMap indexMap;
  ArrayAttr reassociation;
};
} // namespace

/// Utility function for replacing operands/results to a linalg generic
/// operation with unit-extent dimensions. These can be replaced with
/// an operand/result with the unit-extent dimension removed. This is only done
/// if the indexing map used to access that dimension has a
/// AffineConstantExpr of value 0. Given the `type` of an result/operand of a
/// Linalg op, and its `indexMap` the utility function returns:
/// - the new type with dimensions of size 1 removed.
/// - modified index map that can be used to access the replaced result/operand
/// - the reassociation that converts from the original tensor type to the
///   modified tensor type.
static std::optional<UnitExtentReplacementInfo>
replaceUnitExtents(GenericOp genericOp, OpOperand *opOperand,
                   MLIRContext *context) {
  AffineMap indexingMap = genericOp.getMatchingIndexingMap(opOperand);
  ArrayRef<int64_t> shape = genericOp.getShape(opOperand);
  ArrayRef<AffineExpr> exprs = indexingMap.getResults();
  SmallVector<AffineExpr> reassociations;
  SmallVector<Attribute> reassociationMaps;
  SmallVector<AffineExpr> newIndexExprs;
  SmallVector<int64_t> newShape;

  int64_t origRank = genericOp.getRank(opOperand);
  AffineExpr zeroExpr = getAffineConstantExpr(0, context);
  auto isUnitExtent = [&](int64_t dim) -> bool {
    return shape[dim] == 1 && exprs[dim] == zeroExpr;
  };

  // Early return for memrefs with affine maps to represent that we will always
  // leave them unchanged.
  Type actualType = opOperand->get().getType();
  if (auto memref = actualType.dyn_cast<MemRefType>()) {
    if (!memref.getLayout().isIdentity())
      return std::nullopt;
  }

  int64_t dim = 0;
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

  // Compute the tensor or scalar replacement type.
  Type elementType = getElementTypeOrSelf(opOperand->get());
  Type replacementType;
  if (elementType == opOperand->get().getType()) {
    replacementType = elementType;
  } else if (actualType.isa<RankedTensorType>()) {
    replacementType = RankedTensorType::get(newShape, elementType);
  } else {
    auto memrefType = actualType.cast<MemRefType>();
    replacementType = MemRefType::get(newShape, elementType, {},
                                      memrefType.getMemorySpaceAsInt());
  }
  UnitExtentReplacementInfo info = {replacementType,
                                    AffineMap::get(indexingMap.getNumDims(),
                                                   indexingMap.getNumSymbols(),
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

/// Pattern to replace tensor/buffer operands/results that are unit extents.
struct ReplaceUnitExtents : public OpRewritePattern<GenericOp> {
  using OpRewritePattern<GenericOp>::OpRewritePattern;

  // Return the original value if the type is unchanged, or reshape it. Return a
  // nullptr if this is an unsupported type.
  Value maybeExpand(Value result, Type origResultType,
                    ArrayAttr reassociationMap, Location loc,
                    PatternRewriter &rewriter) const {
    if (origResultType == result.getType())
      return result;
    if (origResultType.isa<RankedTensorType>()) {
      return rewriter.create<tensor::ExpandShapeOp>(
          loc, origResultType, result,
          convertAffineMapArrayToExprs(reassociationMap));
    }
    if (origResultType.isa<MemRefType>()) {
      return rewriter.create<memref::ExpandShapeOp>(
          loc, origResultType, result,
          convertAffineMapArrayToExprs(reassociationMap));
    }
    return nullptr;
  };

  // Return the original value if the type is unchanged, or reshape it. Return a
  // nullptr if this is an unsupported type.
  Value maybeCollapse(Value operand, Type newInputOutputType,
                      ArrayAttr reassociationMap, Location loc,
                      PatternRewriter &rewriter) const {
    auto operandType = operand.getType();
    if (operandType == newInputOutputType)
      return operand;
    if (operandType.isa<MemRefType>()) {
      return rewriter.create<memref::CollapseShapeOp>(
          loc, newInputOutputType, operand,
          convertAffineMapArrayToExprs(reassociationMap));
    }
    if (operandType.isa<RankedTensorType>()) {
      return rewriter.create<tensor::CollapseShapeOp>(
          loc, newInputOutputType, operand,
          convertAffineMapArrayToExprs(reassociationMap));
    }
    return nullptr;
  };

  LogicalResult matchAndRewrite(GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    // Skip the pattern if the op has any tensor with special encoding.
    if (llvm::any_of(genericOp->getOperandTypes(), [](Type type) {
          auto tensorType = type.dyn_cast<RankedTensorType>();
          return tensorType && tensorType.getEncoding() != nullptr;
        }))
      return failure();
    MLIRContext *context = rewriter.getContext();
    Location loc = genericOp.getLoc();

    SmallVector<AffineMap> newIndexingMaps;
    SmallVector<ArrayAttr> reassociationMaps;
    SmallVector<Type> newInputOutputTypes;
    bool doCanonicalization = false;
    for (OpOperand &opOperand : genericOp->getOpOperands()) {
      auto replacementInfo = replaceUnitExtents(genericOp, &opOperand, context);
      if (replacementInfo) {
        reassociationMaps.push_back(replacementInfo->reassociation);
        newIndexingMaps.push_back(replacementInfo->indexMap);
        newInputOutputTypes.push_back(replacementInfo->type);
        doCanonicalization |=
            replacementInfo->type != opOperand.get().getType();
      } else {
        // If replaceUnitExtents cannot handle this case, maintain the same
        // type, indexing map, and create a set of mappings representing an
        // identity matrix.
        newInputOutputTypes.push_back(opOperand.get().getType());
        newIndexingMaps.push_back(genericOp.getMatchingIndexingMap(&opOperand));
        int64_t origRank = genericOp.getRank(&opOperand);
        auto maps = llvm::to_vector<8>(llvm::map_range(
            llvm::seq<int64_t>(0, origRank), [&](int64_t dim) -> Attribute {
              return AffineMapAttr::get(
                  AffineMap::get(origRank, /*symbolCount = */ 0,
                                 getAffineDimExpr(dim, context), context));
            }));
        reassociationMaps.push_back(ArrayAttr::get(context, maps));
      }
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
      for (auto operand : values) {
        auto reshapedValue =
            maybeCollapse(operand, newInputOutputTypes[flattenedIdx],
                          reassociationMaps[flattenedIdx], loc, rewriter);
        assert(reshapedValue &&
               "expected ranked MemRef or Tensor operand type");
        res.push_back(reshapedValue);
        ++flattenedIdx;
      }
      return res;
    };

    SmallVector<Value, 4> newInputs = insertReshapes(genericOp.getInputs());
    SmallVector<Value, 4> newOutputs = insertReshapes(genericOp.getOutputs());

    // If any result type changes, insert a reshape to convert from the original
    // type to the new type.
    SmallVector<Type, 4> resultTypes;
    resultTypes.reserve(genericOp.getNumResults());
    for (unsigned i : llvm::seq<unsigned>(0, genericOp.getNumResults()))
      resultTypes.push_back(
          newInputOutputTypes[i + genericOp.getNumDpsInputs()]);
    GenericOp replacementOp = rewriter.create<GenericOp>(
        loc, resultTypes, newInputs, newOutputs, newIndexingMaps,
        genericOp.getIteratorTypesArray());
    rewriter.inlineRegionBefore(genericOp.getRegion(),
                                replacementOp.getRegion(),
                                replacementOp.getRegion().begin());

    // If any result tensor has a modified shape, then add reshape to recover
    // the original shape.
    SmallVector<Value, 4> resultReplacements;
    for (const auto &result : llvm::enumerate(replacementOp.getResults())) {
      unsigned index = result.index() + replacementOp.getNumDpsInputs();
      auto origResultType = genericOp.getResult(result.index()).getType();

      auto newResult = maybeExpand(result.value(), origResultType,
                                   reassociationMaps[index], loc, rewriter);
      assert(newResult &&
             "unexpected output type other than ranked MemRef or Tensor");
      resultReplacements.push_back(newResult);
    }
    rewriter.replaceOp(genericOp, resultReplacements);
    return success();
  }
};
} // namespace

namespace {
/// Convert `extract_slice` operations to rank-reduced versions.
struct RankReducedExtractSliceOp
    : public OpRewritePattern<tensor::ExtractSliceOp> {
  using OpRewritePattern<tensor::ExtractSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractSliceOp sliceOp,
                                PatternRewriter &rewriter) const override {
    RankedTensorType resultType = sliceOp.getType();
    SmallVector<OpFoldResult> offsets = sliceOp.getMixedOffsets();
    SmallVector<OpFoldResult> sizes = sliceOp.getMixedSizes();
    SmallVector<OpFoldResult> strides = sliceOp.getMixedStrides();
    auto reassociation = getReassociationMapForFoldingUnitDims(sizes);
    if (!reassociation ||
        reassociation->size() == static_cast<size_t>(resultType.getRank()))
      return failure();
    auto rankReducedType =
        tensor::ExtractSliceOp::inferCanonicalRankReducedResultType(
            reassociation->size(), sliceOp.getSourceType(), offsets, sizes,
            strides)
            .cast<RankedTensorType>();

    Location loc = sliceOp.getLoc();
    Value newSlice = rewriter.create<tensor::ExtractSliceOp>(
        loc, rankReducedType, sliceOp.getSource(), offsets, sizes, strides);
    rewriter.replaceOpWithNewOp<tensor::ExpandShapeOp>(
        sliceOp, resultType, newSlice, *reassociation);
    return success();
  }
};

/// Convert `insert_slice` operations to rank-reduced versions.
/// This patterns works with both InsertSliceOp and ParallelInsertSliceOp.
template <typename InsertOpTy>
struct RankReducedInsertSliceOp : public OpRewritePattern<InsertOpTy> {
  using OpRewritePattern<InsertOpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(InsertOpTy insertSliceOp,
                                PatternRewriter &rewriter) const override {
    RankedTensorType sourceType = insertSliceOp.getSourceType();
    SmallVector<OpFoldResult> offsets = insertSliceOp.getMixedOffsets();
    SmallVector<OpFoldResult> sizes = insertSliceOp.getMixedSizes();
    SmallVector<OpFoldResult> strides = insertSliceOp.getMixedStrides();
    auto reassociation = getReassociationMapForFoldingUnitDims(sizes);
    if (!reassociation ||
        reassociation->size() == static_cast<size_t>(sourceType.getRank()))
      return failure();
    Location loc = insertSliceOp.getLoc();
    tensor::CollapseShapeOp reshapedSource;
    {
      OpBuilder::InsertionGuard g(rewriter);
      // The only difference between InsertSliceOp and ParallelInsertSliceOp is
      // the insertion point is just before the ParallelCombiningOp in the
      // parallel case.
      if (std::is_same<InsertOpTy, tensor::ParallelInsertSliceOp>::value)
        rewriter.setInsertionPoint(insertSliceOp->getParentOp());
      reshapedSource = rewriter.create<tensor::CollapseShapeOp>(
          loc, insertSliceOp.getSource(), *reassociation);
    }
    rewriter.replaceOpWithNewOp<InsertOpTy>(
        insertSliceOp, reshapedSource, insertSliceOp.getDest(),
        insertSliceOp.getMixedOffsets(), insertSliceOp.getMixedSizes(),
        insertSliceOp.getMixedStrides());
    return success();
  }
};
} // namespace

/// Patterns that are used to canonicalize the use of unit-extent dims for
/// broadcasting.
void mlir::linalg::populateFoldUnitExtentDimsPatterns(
    RewritePatternSet &patterns) {
  auto *context = patterns.getContext();
  patterns.add<FoldUnitDimLoops, AddInitOperandsToInput, ReplaceUnitExtents,
               RankReducedExtractSliceOp,
               RankReducedInsertSliceOp<tensor::InsertSliceOp>,
               RankReducedInsertSliceOp<tensor::ParallelInsertSliceOp>>(
      context);
  linalg::FillOp::getCanonicalizationPatterns(patterns, context);
  tensor::CollapseShapeOp::getCanonicalizationPatterns(patterns, context);
  tensor::EmptyOp::getCanonicalizationPatterns(patterns, context);
  tensor::ExpandShapeOp::getCanonicalizationPatterns(patterns, context);
  tensor::populateFoldTensorEmptyPatterns(patterns);
  memref::populateResolveRankedShapeTypeResultDimsPatterns(patterns);
  memref::populateResolveShapedTypeResultDimsPatterns(patterns);
}

namespace {
/// Pass that removes unit-extent dims within generic ops.
struct LinalgFoldUnitExtentDimsPass
    : public impl::LinalgFoldUnitExtentDimsBase<LinalgFoldUnitExtentDimsPass> {
  void runOnOperation() override {
    Operation *op = getOperation();
    MLIRContext *context = op->getContext();
    RewritePatternSet patterns(context);
    if (foldOneTripLoopsOnly)
      patterns.add<FoldUnitDimLoops, AddInitOperandsToInput>(context);
    else
      populateFoldUnitExtentDimsPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(op, std::move(patterns));
  }
};
} // namespace

std::unique_ptr<Pass> mlir::createLinalgFoldUnitExtentDimsPass() {
  return std::make_unique<LinalgFoldUnitExtentDimsPass>();
}
