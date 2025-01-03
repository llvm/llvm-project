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
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

namespace mlir {
#define GEN_PASS_DEF_LINALGFOLDUNITEXTENTDIMSPASS
#include "mlir/Dialect/Linalg/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "linalg-drop-unit-dims"

using namespace mlir;
using namespace mlir::linalg;

namespace {
/// Pattern to move init operands to ins when all the loops are parallel and
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
struct MoveInitOperandsToInput : public OpRewritePattern<GenericOp> {
  using OpRewritePattern<GenericOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    if (!genericOp.hasPureTensorSemantics())
      return failure();
    if (genericOp.getNumParallelLoops() != genericOp.getNumLoops())
      return failure();

    auto outputOperands = genericOp.getDpsInitsMutable();
    SetVector<OpOperand *> candidates;
    for (OpOperand &op : outputOperands) {
      if (genericOp.getMatchingBlockArgument(&op).use_empty())
        continue;
      candidates.insert(&op);
    }

    if (candidates.empty())
      return failure();

    // Compute the modified indexing maps.
    int64_t origNumInput = genericOp.getNumDpsInputs();
    SmallVector<Value> newInputOperands = genericOp.getDpsInputs();
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
    SmallVector<Value> newOutputOperands =
        llvm::to_vector(genericOp.getDpsInits());
    for (OpOperand *op : candidates) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointAfterValue(op->get());
      auto elemType = cast<ShapedType>(op->get().getType()).getElementType();
      auto empty = rewriter.create<tensor::EmptyOp>(
          loc, tensor::getMixedSizes(rewriter, loc, op->get()), elemType);

      unsigned start = genericOp.getDpsInits().getBeginOperandIndex();
      newOutputOperands[op->getOperandNumber() - start] = empty.getResult();
    }

    auto newOp = rewriter.create<GenericOp>(
        loc, genericOp.getResultTypes(), newInputOperands, newOutputOperands,
        newIndexingMaps, genericOp.getIteratorTypesArray(),
        /*bodyBuild=*/nullptr, linalg::getPrunedAttributeList(genericOp));

    OpBuilder::InsertionGuard guard(rewriter);
    Region &region = newOp.getRegion();
    Block *block = rewriter.createBlock(&region);
    IRMapping mapper;
    for (auto bbarg : genericOp.getRegionInputArgs())
      mapper.map(bbarg, block->addArgument(bbarg.getType(), loc));

    for (OpOperand *op : candidates) {
      BlockArgument bbarg = genericOp.getMatchingBlockArgument(op);
      mapper.map(bbarg, block->addArgument(bbarg.getType(), loc));
    }

    for (OpOperand &op : outputOperands) {
      BlockArgument bbarg = genericOp.getMatchingBlockArgument(&op);
      if (candidates.count(&op))
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
} // namespace

//===---------------------------------------------------------------------===//
// Drop loops that are unit-extents within Linalg operations.
//===---------------------------------------------------------------------===//

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

/// Update the index accesses of linalg operations having index semantics.
static void
replaceUnitDimIndexOps(GenericOp genericOp,
                       const llvm::SmallDenseSet<unsigned> &unitDims,
                       RewriterBase &rewriter) {
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

/// Expand the given `value` so that the type matches the type of `origDest`.
/// The `reassociation` is used when `rankReductionStrategy` is set to
/// `RankReductionStrategy::ReassociativeReshape`.
static Value
expandValue(RewriterBase &rewriter, Location loc, Value result, Value origDest,
            ArrayRef<ReassociationIndices> reassociation,
            ControlDropUnitDims::RankReductionStrategy rankReductionStrategy) {
  // There are no results for memref outputs.
  auto origResultType = cast<RankedTensorType>(origDest.getType());
  if (rankReductionStrategy ==
      ControlDropUnitDims::RankReductionStrategy::ExtractInsertSlice) {
    unsigned rank = origResultType.getRank();
    SmallVector<OpFoldResult> offsets(rank, rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> sizes =
        tensor::getMixedSizes(rewriter, loc, origDest);
    SmallVector<OpFoldResult> strides(rank, rewriter.getIndexAttr(1));
    return rewriter.createOrFold<tensor::InsertSliceOp>(
        loc, result, origDest, offsets, sizes, strides);
  }

  assert(rankReductionStrategy ==
             ControlDropUnitDims::RankReductionStrategy::ReassociativeReshape &&
         "unknown rank reduction strategy");
  return rewriter
      .create<tensor::ExpandShapeOp>(loc, origResultType, result, reassociation)
      .getResult();
}

/// Collapse the given `value` so that the type matches the type of
/// `origOutput`. The `reassociation` is used when `rankReductionStrategy` is
/// set to `RankReductionStrategy::ReassociativeReshape`.
static Value collapseValue(
    RewriterBase &rewriter, Location loc, Value operand,
    ArrayRef<int64_t> targetShape, ArrayRef<ReassociationIndices> reassociation,
    ControlDropUnitDims::RankReductionStrategy rankReductionStrategy) {
  if (auto memrefType = dyn_cast<MemRefType>(operand.getType())) {
    if (rankReductionStrategy ==
        ControlDropUnitDims::RankReductionStrategy::ExtractInsertSlice) {
      FailureOr<Value> rankReducingExtract =
          memref::SubViewOp::rankReduceIfNeeded(rewriter, loc, operand,
                                                targetShape);
      assert(succeeded(rankReducingExtract) && "not a unit-extent collapse");
      return *rankReducingExtract;
    }

    assert(
        rankReductionStrategy ==
            ControlDropUnitDims::RankReductionStrategy::ReassociativeReshape &&
        "unknown rank reduction strategy");
    MemRefLayoutAttrInterface layout;
    auto targetType = MemRefType::get(targetShape, memrefType.getElementType(),
                                      layout, memrefType.getMemorySpace());
    return rewriter.create<memref::CollapseShapeOp>(loc, targetType, operand,
                                                    reassociation);
  }
  if (auto tensorType = dyn_cast<RankedTensorType>(operand.getType())) {
    if (rankReductionStrategy ==
        ControlDropUnitDims::RankReductionStrategy::ExtractInsertSlice) {
      FailureOr<Value> rankReducingExtract =
          tensor::ExtractSliceOp::rankReduceIfNeeded(rewriter, loc, operand,
                                                     targetShape);
      assert(succeeded(rankReducingExtract) && "not a unit-extent collapse");
      return *rankReducingExtract;
    }

    assert(
        rankReductionStrategy ==
            ControlDropUnitDims::RankReductionStrategy::ReassociativeReshape &&
        "unknown rank reduction strategy");
    auto targetType =
        RankedTensorType::get(targetShape, tensorType.getElementType());
    return rewriter.create<tensor::CollapseShapeOp>(loc, targetType, operand,
                                                    reassociation);
  }
  llvm_unreachable("unsupported operand type");
}

/// Compute the modified metadata for an operands of operation
/// whose unit dims are being dropped. Return the new indexing map
/// to use, the shape of the operand in the replacement op
/// and the `reassocation` to use to go from original operand shape
/// to modified operand shape.
struct UnitExtentReplacementInfo {
  AffineMap indexMap;
  SmallVector<ReassociationIndices> reassociation;
  SmallVector<int64_t> targetShape;
};
static UnitExtentReplacementInfo dropUnitExtentFromOperandMetadata(
    MLIRContext *context, GenericOp genericOp, OpOperand *opOperand,
    llvm::SmallDenseMap<unsigned, unsigned> &oldDimsToNewDimsMap,
    ArrayRef<AffineExpr> dimReplacements) {
  UnitExtentReplacementInfo info;
  ReassociationIndices reassociationGroup;
  SmallVector<AffineExpr> newIndexExprs;
  AffineMap indexingMap = genericOp.getMatchingIndexingMap(opOperand);
  ArrayRef<int64_t> operandShape = genericOp.getShape(opOperand);
  ArrayRef<AffineExpr> exprs = indexingMap.getResults();

  auto isUnitDim = [&](unsigned dim) {
    if (auto dimExpr = dyn_cast<AffineDimExpr>(exprs[dim])) {
      unsigned oldPosition = dimExpr.getPosition();
      return !oldDimsToNewDimsMap.count(oldPosition) &&
             (operandShape[dim] == 1);
    }
    // Handle the other case where the shape is 1, and is accessed using a
    // constant 0.
    if (operandShape[dim] == 1) {
      auto constAffineExpr = dyn_cast<AffineConstantExpr>(exprs[dim]);
      return constAffineExpr && constAffineExpr.getValue() == 0;
    }
    return false;
  };

  unsigned dim = 0;
  while (dim < operandShape.size() && isUnitDim(dim))
    reassociationGroup.push_back(dim++);
  while (dim < operandShape.size()) {
    assert(!isUnitDim(dim) && "expected non unit-extent");
    reassociationGroup.push_back(dim);
    AffineExpr newExpr = exprs[dim].replaceDims(dimReplacements);
    newIndexExprs.push_back(newExpr);
    info.targetShape.push_back(operandShape[dim]);
    ++dim;
    // Fold all following dimensions that are unit-extent.
    while (dim < operandShape.size() && isUnitDim(dim)) {
      reassociationGroup.push_back(dim++);
    }
    info.reassociation.push_back(reassociationGroup);
    reassociationGroup.clear();
  }
  info.indexMap =
      AffineMap::get(oldDimsToNewDimsMap.size(), indexingMap.getNumSymbols(),
                     newIndexExprs, context);
  return info;
}

FailureOr<DropUnitDimsResult>
linalg::dropUnitDims(RewriterBase &rewriter, GenericOp genericOp,
                     const ControlDropUnitDims &options) {
  SmallVector<AffineMap> indexingMaps = genericOp.getIndexingMapsArray();
  if (indexingMaps.empty())
    return failure();

  // 1. Check if any of the iteration dimensions are unit-trip count. They will
  //    end up being unit-trip count if they are used to index into a unit-dim
  //    tensor/memref.
  AffineMap invertedMap =
      inversePermutation(concatAffineMaps(indexingMaps, rewriter.getContext()));
  if (!invertedMap) {
    return rewriter.notifyMatchFailure(genericOp,
                                       "invalid indexing maps for operation");
  }
  SmallVector<int64_t> dims = genericOp.getStaticShape();

  // 1a. Get the allowed list of dimensions to drop from the `options`.
  SmallVector<unsigned> allowedUnitDims = options.controlFn(genericOp);
  if (allowedUnitDims.empty()) {
    return rewriter.notifyMatchFailure(
        genericOp, "control function returns no allowed unit dims to prune");
  }
  llvm::SmallDenseSet<unsigned> unitDimsFilter(allowedUnitDims.begin(),
                                               allowedUnitDims.end());
  llvm::SmallDenseSet<unsigned> unitDims;
  for (const auto &expr : enumerate(invertedMap.getResults())) {
    if (AffineDimExpr dimExpr = dyn_cast<AffineDimExpr>(expr.value())) {
      if (dims[dimExpr.getPosition()] == 1 &&
          unitDimsFilter.count(expr.index()))
        unitDims.insert(expr.index());
    }
  }

  // 2. Compute the iterator types of the modified op by dropping the one-trip
  //    count loops.
  SmallVector<utils::IteratorType> newIteratorTypes;
  llvm::SmallDenseMap<unsigned, unsigned> oldDimToNewDimMap;
  SmallVector<AffineExpr> dimReplacements;
  unsigned newDims = 0;
  for (auto [index, attr] :
       llvm::enumerate(genericOp.getIteratorTypesArray())) {
    if (unitDims.count(index)) {
      dimReplacements.push_back(
          getAffineConstantExpr(0, rewriter.getContext()));
    } else {
      newIteratorTypes.push_back(attr);
      oldDimToNewDimMap[index] = newDims;
      dimReplacements.push_back(
          getAffineDimExpr(newDims, rewriter.getContext()));
      newDims++;
    }
  }

  // 3. For each of the operands, find the
  //    - modified affine map to use.
  //    - shape of the operands after the unit-dims are dropped.
  //    - the reassociation indices used to convert from the original
  //      operand type to modified operand (needed only when using reshapes
  //      for rank reduction strategy)
  // Note that the indexing maps might need changing even if there are no
  // unit dimensions that are dropped to handle cases where `0` is used to
  // access a unit-extent tensor. Consider moving this out of this specific
  // transformation as a stand-alone transformation. Kept here right now due
  // to legacy.
  SmallVector<AffineMap> newIndexingMaps;
  SmallVector<SmallVector<ReassociationIndices>> reassociations;
  SmallVector<SmallVector<int64_t>> targetShapes;
  SmallVector<bool> collapsed;
  auto hasCollapsibleType = [](OpOperand &operand) {
    Type operandType = operand.get().getType();
    if (auto memrefOperandType = dyn_cast_or_null<MemRefType>(operandType)) {
      return memrefOperandType.getLayout().isIdentity();
    }
    if (auto tensorOperandType = dyn_cast<RankedTensorType>(operandType)) {
      return tensorOperandType.getEncoding() == nullptr;
    }
    return false;
  };
  for (OpOperand &opOperand : genericOp->getOpOperands()) {
    auto indexingMap = genericOp.getMatchingIndexingMap(&opOperand);
    ArrayRef<int64_t> shape = genericOp.getShape(&opOperand);
    if (!hasCollapsibleType(opOperand)) {
      AffineMap newIndexingMap = indexingMap.replaceDimsAndSymbols(
          dimReplacements, ArrayRef<AffineExpr>{}, oldDimToNewDimMap.size(), 0);
      newIndexingMaps.push_back(newIndexingMap);
      targetShapes.push_back(llvm::to_vector(shape));
      collapsed.push_back(false);
      reassociations.push_back({});
      continue;
    }
    auto replacementInfo = dropUnitExtentFromOperandMetadata(
        rewriter.getContext(), genericOp, &opOperand, oldDimToNewDimMap,
        dimReplacements);
    reassociations.push_back(replacementInfo.reassociation);
    newIndexingMaps.push_back(replacementInfo.indexMap);
    targetShapes.push_back(replacementInfo.targetShape);
    collapsed.push_back(!(replacementInfo.indexMap.getNumResults() ==
                          indexingMap.getNumResults()));
  }

  // Abort if the indexing maps of the result operation are not invertible
  // (i.e. not legal) or if no dimension was reduced.
  if (newIndexingMaps == indexingMaps ||
      !inversePermutation(
          concatAffineMaps(newIndexingMaps, rewriter.getContext())))
    return failure();

  Location loc = genericOp.getLoc();
  // 4. For each of the operands, collapse the operand to convert
  //    from original shape to shape in the modified operation if needed,
  //    either through use of reshapes or rank-reducing slices as
  //    specified in `options`.
  SmallVector<Value> newOperands;
  for (OpOperand &opOperand : genericOp->getOpOperands()) {
    int64_t idx = opOperand.getOperandNumber();
    if (!collapsed[idx]) {
      newOperands.push_back(opOperand.get());
      continue;
    }
    newOperands.push_back(collapseValue(rewriter, loc, opOperand.get(),
                                        targetShapes[idx], reassociations[idx],
                                        options.rankReductionStrategy));
  }

  // 5. Create the `linalg.generic` operation with the new operands,
  //    indexing maps, iterator types and result types.
  ArrayRef<Value> newInputs =
      ArrayRef<Value>(newOperands).take_front(genericOp.getNumDpsInputs());
  ArrayRef<Value> newOutputs =
      ArrayRef<Value>(newOperands).take_back(genericOp.getNumDpsInits());
  SmallVector<Type> resultTypes;
  resultTypes.reserve(genericOp.getNumResults());
  for (unsigned i : llvm::seq<unsigned>(0, genericOp.getNumResults()))
    resultTypes.push_back(newOutputs[i].getType());
  GenericOp replacementOp =
      rewriter.create<GenericOp>(loc, resultTypes, newInputs, newOutputs,
                                 newIndexingMaps, newIteratorTypes);
  rewriter.inlineRegionBefore(genericOp.getRegion(), replacementOp.getRegion(),
                              replacementOp.getRegion().begin());
  // 5a. Replace `linalg.index` operations that refer to the dropped unit
  //     dimensions.
  replaceUnitDimIndexOps(replacementOp, unitDims, rewriter);

  // 6. If any result type changes, insert a reshape/slice to convert from the
  //    original type to the new type.
  SmallVector<Value> resultReplacements;
  for (auto [index, result] : llvm::enumerate(replacementOp.getResults())) {
    unsigned opOperandIndex = index + replacementOp.getNumDpsInputs();
    Value origDest = genericOp.getDpsInitOperand(index)->get();
    if (!collapsed[opOperandIndex]) {
      resultReplacements.push_back(result);
      continue;
    }
    Value expandedValue = expandValue(rewriter, loc, result, origDest,
                                      reassociations[opOperandIndex],
                                      options.rankReductionStrategy);
    resultReplacements.push_back(expandedValue);
  }

  return DropUnitDimsResult{replacementOp, resultReplacements};
}

namespace {
struct DropUnitDims : public OpRewritePattern<GenericOp> {
  DropUnitDims(MLIRContext *context, ControlDropUnitDims options = {},
               PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), options(std::move(options)) {}

  LogicalResult matchAndRewrite(GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    FailureOr<DropUnitDimsResult> result =
        dropUnitDims(rewriter, genericOp, options);
    if (failed(result)) {
      return failure();
    }
    rewriter.replaceOp(genericOp, result->replacements);
    return success();
  }

private:
  ControlDropUnitDims options;
};
} // namespace

//===---------------------------------------------------------------------===//
// Drop dimensions that are unit-extents within tensor operations.
//===---------------------------------------------------------------------===//

namespace {
struct DropPadUnitDims : public OpRewritePattern<tensor::PadOp> {
  DropPadUnitDims(MLIRContext *context, ControlDropUnitDims options = {},
                  PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), options(std::move(options)) {}

  LogicalResult matchAndRewrite(tensor::PadOp padOp,
                                PatternRewriter &rewriter) const override {
    // 1a. Get the allowed list of dimensions to drop from the `options`.
    SmallVector<unsigned> allowedUnitDims = options.controlFn(padOp);
    if (allowedUnitDims.empty()) {
      return rewriter.notifyMatchFailure(
          padOp, "control function returns no allowed unit dims to prune");
    }

    if (padOp.getSourceType().getEncoding()) {
      return rewriter.notifyMatchFailure(
          padOp, "cannot collapse dims of tensor with encoding");
    }

    // Fail for non-constant padding values. The body of the pad could
    // depend on the padding indices and/or properties of the padded
    // tensor so for now we fail.
    // TODO: Support non-constant padding values.
    Value paddingVal = padOp.getConstantPaddingValue();
    if (!paddingVal) {
      return rewriter.notifyMatchFailure(
          padOp, "unimplemented: non-constant padding value");
    }

    ArrayRef<int64_t> sourceShape = padOp.getSourceType().getShape();
    int64_t padRank = sourceShape.size();

    auto isStaticZero = [](OpFoldResult f) {
      std::optional<int64_t> maybeInt = getConstantIntValue(f);
      return maybeInt && *maybeInt == 0;
    };

    llvm::SmallDenseSet<unsigned> unitDimsFilter(allowedUnitDims.begin(),
                                                 allowedUnitDims.end());
    llvm::SmallDenseSet<unsigned> unitDims;
    SmallVector<int64_t> newShape;
    SmallVector<OpFoldResult> newLowPad;
    SmallVector<OpFoldResult> newHighPad;
    for (const auto [dim, size, low, high] :
         zip_equal(llvm::seq(static_cast<int64_t>(0), padRank), sourceShape,
                   padOp.getMixedLowPad(), padOp.getMixedHighPad())) {
      if (unitDimsFilter.contains(dim) && size == 1 && isStaticZero(low) &&
          isStaticZero(high)) {
        unitDims.insert(dim);
      } else {
        newShape.push_back(size);
        newLowPad.push_back(low);
        newHighPad.push_back(high);
      }
    }

    if (unitDims.empty()) {
      return rewriter.notifyMatchFailure(padOp, "no unit dims to collapse");
    }

    ReassociationIndices reassociationGroup;
    SmallVector<ReassociationIndices> reassociationMap;
    int64_t dim = 0;
    while (dim < padRank && unitDims.contains(dim))
      reassociationGroup.push_back(dim++);
    while (dim < padRank) {
      assert(!unitDims.contains(dim) && "expected non unit-extent");
      reassociationGroup.push_back(dim);
      dim++;
      // Fold all following dimensions that are unit-extent.
      while (dim < padRank && unitDims.contains(dim))
        reassociationGroup.push_back(dim++);
      reassociationMap.push_back(reassociationGroup);
      reassociationGroup.clear();
    }

    Value collapsedSource =
        collapseValue(rewriter, padOp.getLoc(), padOp.getSource(), newShape,
                      reassociationMap, options.rankReductionStrategy);

    auto newPadOp = rewriter.create<tensor::PadOp>(
        padOp.getLoc(), /*result=*/Type(), collapsedSource, newLowPad,
        newHighPad, paddingVal, padOp.getNofold());

    Value dest = padOp.getResult();
    if (options.rankReductionStrategy ==
        ControlDropUnitDims::RankReductionStrategy::ExtractInsertSlice) {
      SmallVector<OpFoldResult> expandedSizes;
      int64_t numUnitDims = 0;
      for (auto dim : llvm::seq(static_cast<int64_t>(0), padRank)) {
        if (unitDims.contains(dim)) {
          expandedSizes.push_back(rewriter.getIndexAttr(1));
          numUnitDims++;
          continue;
        }
        expandedSizes.push_back(tensor::getMixedSize(
            rewriter, padOp.getLoc(), newPadOp, dim - numUnitDims));
      }
      dest = rewriter.create<tensor::EmptyOp>(
          padOp.getLoc(), expandedSizes,
          padOp.getResultType().getElementType());
    }

    Value expandedValue =
        expandValue(rewriter, padOp.getLoc(), newPadOp.getResult(), dest,
                    reassociationMap, options.rankReductionStrategy);
    rewriter.replaceOp(padOp, expandedValue);
    return success();
  }

private:
  ControlDropUnitDims options;
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
    SmallVector<OpFoldResult> targetShape;
    for (auto size : resultType.getShape())
      targetShape.push_back(rewriter.getIndexAttr(size));
    auto reassociation = getReassociationMapForFoldingUnitDims(targetShape);
    if (!reassociation ||
        reassociation->size() == static_cast<size_t>(resultType.getRank()))
      return failure();

    SmallVector<OpFoldResult> offsets = sliceOp.getMixedOffsets();
    SmallVector<OpFoldResult> strides = sliceOp.getMixedStrides();
    SmallVector<OpFoldResult> sizes = sliceOp.getMixedSizes();
    auto rankReducedType = cast<RankedTensorType>(
        tensor::ExtractSliceOp::inferCanonicalRankReducedResultType(
            reassociation->size(), sliceOp.getSourceType(), offsets, sizes,
            strides));

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
    SmallVector<OpFoldResult> targetShape;
    for (auto size : sourceType.getShape())
      targetShape.push_back(rewriter.getIndexAttr(size));
    auto reassociation = getReassociationMapForFoldingUnitDims(targetShape);
    if (!reassociation ||
        reassociation->size() == static_cast<size_t>(sourceType.getRank()))
      return failure();

    Location loc = insertSliceOp.getLoc();
    tensor::CollapseShapeOp reshapedSource;
    {
      OpBuilder::InsertionGuard g(rewriter);
      // The only difference between InsertSliceOp and ParallelInsertSliceOp
      // is the insertion point is just before the ParallelCombiningOp in the
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
static void
populateFoldUnitExtentDimsViaReshapesPatterns(RewritePatternSet &patterns,
                                              ControlDropUnitDims &options) {
  auto *context = patterns.getContext();
  patterns.add<DropUnitDims>(context, options);
  patterns.add<DropPadUnitDims>(context, options);
  // TODO: Patterns unrelated to unit dim folding should be factored out.
  patterns.add<RankReducedExtractSliceOp,
               RankReducedInsertSliceOp<tensor::InsertSliceOp>,
               RankReducedInsertSliceOp<tensor::ParallelInsertSliceOp>>(
      context);
  linalg::FillOp::getCanonicalizationPatterns(patterns, context);
  tensor::CollapseShapeOp::getCanonicalizationPatterns(patterns, context);
  tensor::EmptyOp::getCanonicalizationPatterns(patterns, context);
  tensor::ExpandShapeOp::getCanonicalizationPatterns(patterns, context);
  tensor::populateFoldTensorEmptyPatterns(patterns);
  memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
  memref::populateResolveShapedTypeResultDimsPatterns(patterns);
}

static void
populateFoldUnitExtentDimsViaSlicesPatterns(RewritePatternSet &patterns,
                                            ControlDropUnitDims &options) {
  auto *context = patterns.getContext();
  patterns.add<DropUnitDims>(context, options);
  patterns.add<DropPadUnitDims>(context, options);
  // TODO: Patterns unrelated to unit dim folding should be factored out.
  linalg::FillOp::getCanonicalizationPatterns(patterns, context);
  tensor::EmptyOp::getCanonicalizationPatterns(patterns, context);
  tensor::populateFoldTensorEmptyPatterns(patterns);
  memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
  memref::populateResolveShapedTypeResultDimsPatterns(patterns);
}

void mlir::linalg::populateFoldUnitExtentDimsPatterns(
    RewritePatternSet &patterns, linalg::ControlDropUnitDims &options) {
  if (options.rankReductionStrategy ==
      linalg::ControlDropUnitDims::RankReductionStrategy::ExtractInsertSlice) {
    populateFoldUnitExtentDimsViaSlicesPatterns(patterns, options);
  } else if (options.rankReductionStrategy ==
             linalg::ControlDropUnitDims::RankReductionStrategy::
                 ReassociativeReshape) {
    populateFoldUnitExtentDimsViaReshapesPatterns(patterns, options);
  }
}

void mlir::linalg::populateMoveInitOperandsToInputPattern(
    RewritePatternSet &patterns) {
  patterns.add<MoveInitOperandsToInput>(patterns.getContext());
}

namespace {
/// Pass that removes unit-extent dims within generic ops.
struct LinalgFoldUnitExtentDimsPass
    : public impl::LinalgFoldUnitExtentDimsPassBase<
          LinalgFoldUnitExtentDimsPass> {
  using impl::LinalgFoldUnitExtentDimsPassBase<
      LinalgFoldUnitExtentDimsPass>::LinalgFoldUnitExtentDimsPassBase;
  void runOnOperation() override {
    Operation *op = getOperation();
    MLIRContext *context = op->getContext();
    RewritePatternSet patterns(context);
    ControlDropUnitDims options;
    if (useRankReducingSlices) {
      options.rankReductionStrategy = linalg::ControlDropUnitDims::
          RankReductionStrategy::ExtractInsertSlice;
    }
    linalg::populateFoldUnitExtentDimsPatterns(patterns, options);
    populateMoveInitOperandsToInputPattern(patterns);
    (void)applyPatternsGreedily(op, std::move(patterns));
  }
};

} // namespace

namespace {

/// Returns reassociation indices for collapsing/expanding a
/// tensor of rank `rank` at position `pos`.
static SmallVector<ReassociationIndices>
getReassociationForReshapeAtDim(int64_t rank, int64_t pos) {
  SmallVector<ReassociationIndices> reassociation(rank - 1, {0, 1});
  bool lastDim = pos == rank - 1;
  if (rank > 2) {
    for (int64_t i = 0; i < rank - 1; i++) {
      if (i == pos || (lastDim && i == pos - 1))
        reassociation[i] = ReassociationIndices{i, i + 1};
      else if (i < pos)
        reassociation[i] = ReassociationIndices{i};
      else
        reassociation[i] = ReassociationIndices{i + 1};
    }
  }
  return reassociation;
}

/// Returns a collapsed `val` where the collapsing occurs at dim `pos`.
/// If `pos < 0`, then don't collapse.
static Value collapseSingletonDimAt(PatternRewriter &rewriter, Value val,
                                    int64_t pos) {
  if (pos < 0)
    return val;
  auto valType = cast<ShapedType>(val.getType());
  SmallVector<int64_t> collapsedShape(valType.getShape());
  collapsedShape.erase(collapsedShape.begin() + pos);
  return collapseValue(
      rewriter, val.getLoc(), val, collapsedShape,
      getReassociationForReshapeAtDim(valType.getRank(), pos),
      ControlDropUnitDims::RankReductionStrategy::ReassociativeReshape);
}

/// Base class for all rank reduction patterns for contraction ops
/// with unit dimensions.  All patterns should convert one named op
/// to another named op.  Intended to reduce only one iteration space dim
/// at a time.
/// Reducing multiple dims will happen with recusive application of
/// pattern rewrites.
template <typename FromOpTy, typename ToOpTy>
struct RankReduceContractionOps : OpRewritePattern<FromOpTy> {
  using OpRewritePattern<FromOpTy>::OpRewritePattern;

  /// Collapse all collapsable operands.
  SmallVector<Value>
  collapseOperands(PatternRewriter &rewriter, ArrayRef<Value> operands,
                   ArrayRef<int64_t> operandCollapseDims) const {
    assert(operandCollapseDims.size() == 3 && operands.size() == 3 &&
           "expected 3 operands and dims");
    return llvm::map_to_vector(
        llvm::zip(operands, operandCollapseDims), [&](auto pair) {
          return collapseSingletonDimAt(rewriter, std::get<0>(pair),
                                        std::get<1>(pair));
        });
  }

  /// Expand result tensor.
  Value expandResult(PatternRewriter &rewriter, Value result,
                     RankedTensorType expandedType, int64_t dim) const {
    return rewriter.create<tensor::ExpandShapeOp>(
        result.getLoc(), expandedType, result,
        getReassociationForReshapeAtDim(expandedType.getRank(), dim));
  }

  LogicalResult matchAndRewrite(FromOpTy contractionOp,
                                PatternRewriter &rewriter) const override {

    auto loc = contractionOp.getLoc();
    auto inputs = contractionOp.getDpsInputs();
    auto inits = contractionOp.getDpsInits();
    if (inputs.size() != 2 || inits.size() != 1)
      return rewriter.notifyMatchFailure(contractionOp,
                                         "expected 2 inputs and 1 init");
    auto lhs = inputs[0];
    auto rhs = inputs[1];
    auto init = inits[0];
    SmallVector<Value> operands{lhs, rhs, init};

    SmallVector<int64_t> operandUnitDims;
    if (failed(getOperandUnitDims(contractionOp, operandUnitDims)))
      return rewriter.notifyMatchFailure(contractionOp,
                                         "no reducable dims found");

    SmallVector<Value> collapsedOperands =
        collapseOperands(rewriter, operands, operandUnitDims);
    Value collapsedLhs = collapsedOperands[0];
    Value collapsedRhs = collapsedOperands[1];
    Value collapsedInit = collapsedOperands[2];
    SmallVector<Type, 1> collapsedResultTy;
    if (isa<RankedTensorType>(collapsedInit.getType()))
      collapsedResultTy.push_back(collapsedInit.getType());
    auto collapsedOp = rewriter.create<ToOpTy>(
        loc, collapsedResultTy, ValueRange{collapsedLhs, collapsedRhs},
        ValueRange{collapsedInit});
    for (auto attr : contractionOp->getAttrs()) {
      if (attr.getName() == LinalgDialect::kMemoizedIndexingMapsAttrName)
        continue;
      collapsedOp->setAttr(attr.getName(), attr.getValue());
    }

    auto results = contractionOp.getResults();
    assert(results.size() < 2 && "expected at most one result");
    if (results.empty()) {
      rewriter.replaceOp(contractionOp, collapsedOp);
    } else {
      rewriter.replaceOp(
          contractionOp,
          expandResult(rewriter, collapsedOp.getResultTensors()[0],
                       cast<RankedTensorType>(results[0].getType()),
                       operandUnitDims[2]));
    }

    return success();
  }

  /// Populate `operandUnitDims` with 3 indices indicating the unit dim
  /// for each operand that should be collapsed in this pattern.  If an
  /// operand shouldn't be collapsed, the index should be negative.
  virtual LogicalResult
  getOperandUnitDims(LinalgOp op,
                     SmallVectorImpl<int64_t> &operandUnitDims) const = 0;
};

/// Patterns for unbatching batched contraction ops
template <typename FromOpTy, typename ToOpTy>
struct RankReduceToUnBatched : RankReduceContractionOps<FromOpTy, ToOpTy> {
  using RankReduceContractionOps<FromOpTy, ToOpTy>::RankReduceContractionOps;

  /// Look for unit batch dims to collapse.
  LogicalResult
  getOperandUnitDims(LinalgOp op,
                     SmallVectorImpl<int64_t> &operandUnitDims) const override {
    FailureOr<ContractionDimensions> maybeContractionDims =
        inferContractionDims(op);
    if (failed(maybeContractionDims)) {
      LLVM_DEBUG(llvm::dbgs() << "could not infer contraction dims");
      return failure();
    }
    ContractionDimensions contractionDims = maybeContractionDims.value();

    if (contractionDims.batch.size() != 1)
      return failure();
    auto batchDim = contractionDims.batch[0];
    SmallVector<std::pair<Value, unsigned>, 3> bOperands;
    op.mapIterationSpaceDimToAllOperandDims(batchDim, bOperands);
    if (bOperands.size() != 3 || llvm::any_of(bOperands, [](auto pair) {
          return cast<ShapedType>(std::get<0>(pair).getType())
                     .getShape()[std::get<1>(pair)] != 1;
        })) {
      LLVM_DEBUG(llvm::dbgs() << "specified unit dims not found");
      return failure();
    }

    operandUnitDims = SmallVector<int64_t>{std::get<1>(bOperands[0]),
                                           std::get<1>(bOperands[1]),
                                           std::get<1>(bOperands[2])};
    return success();
  }
};

/// Patterns for reducing non-batch dimensions
template <typename FromOpTy, typename ToOpTy>
struct RankReduceMatmul : RankReduceContractionOps<FromOpTy, ToOpTy> {
  using RankReduceContractionOps<FromOpTy, ToOpTy>::RankReduceContractionOps;

  /// Helper for determining whether the lhs/init or rhs/init are reduced.
  static bool constexpr reduceLeft =
      (std::is_same_v<FromOpTy, BatchMatmulOp> &&
       std::is_same_v<ToOpTy, BatchVecmatOp>) ||
      (std::is_same_v<FromOpTy, BatchMatmulTransposeAOp> &&
       std::is_same_v<ToOpTy, BatchVecmatOp>) ||
      (std::is_same_v<FromOpTy, MatmulOp> &&
       std::is_same_v<ToOpTy, VecmatOp>) ||
      (std::is_same_v<FromOpTy, MatmulTransposeAOp> &&
       std::is_same_v<ToOpTy, VecmatOp>) ||
      (std::is_same_v<FromOpTy, MatvecOp> && std::is_same_v<ToOpTy, DotOp>);

  /// Look for non-batch spatial dims to collapse.
  LogicalResult
  getOperandUnitDims(LinalgOp op,
                     SmallVectorImpl<int64_t> &operandUnitDims) const override {
    FailureOr<ContractionDimensions> maybeContractionDims =
        inferContractionDims(op);
    if (failed(maybeContractionDims)) {
      LLVM_DEBUG(llvm::dbgs() << "could not infer contraction dims");
      return failure();
    }
    ContractionDimensions contractionDims = maybeContractionDims.value();

    if constexpr (reduceLeft) {
      auto m = contractionDims.m[0];
      SmallVector<std::pair<Value, unsigned>, 2> mOperands;
      op.mapIterationSpaceDimToAllOperandDims(m, mOperands);
      if (mOperands.size() != 2)
        return failure();
      if (llvm::all_of(mOperands, [](auto pair) {
            return cast<ShapedType>(std::get<0>(pair).getType())
                       .getShape()[std::get<1>(pair)] == 1;
          })) {
        operandUnitDims = SmallVector<int64_t>{std::get<1>(mOperands[0]), -1,
                                               std::get<1>(mOperands[1])};
        return success();
      }
    } else {
      auto n = contractionDims.n[0];
      SmallVector<std::pair<Value, unsigned>, 2> nOperands;
      op.mapIterationSpaceDimToAllOperandDims(n, nOperands);
      if (nOperands.size() != 2)
        return failure();
      if (llvm::all_of(nOperands, [](auto pair) {
            return cast<ShapedType>(std::get<0>(pair).getType())
                       .getShape()[std::get<1>(pair)] == 1;
          })) {
        operandUnitDims = SmallVector<int64_t>{-1, std::get<1>(nOperands[0]),
                                               std::get<1>(nOperands[1])};
        return success();
      }
    }
    LLVM_DEBUG(llvm::dbgs() << "specified unit dims not found");
    return failure();
  }
};

} // namespace

void mlir::linalg::populateContractionOpRankReducingPatterns(
    RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  // Unbatching patterns for unit batch size
  patterns.add<RankReduceToUnBatched<BatchMatmulOp, MatmulOp>>(context);
  patterns
      .add<RankReduceToUnBatched<BatchMatmulTransposeAOp, MatmulTransposeAOp>>(
          context);
  patterns
      .add<RankReduceToUnBatched<BatchMatmulTransposeBOp, MatmulTransposeBOp>>(
          context);
  patterns.add<RankReduceToUnBatched<BatchMatvecOp, MatvecOp>>(context);
  patterns.add<RankReduceToUnBatched<BatchVecmatOp, VecmatOp>>(context);

  // Non-batch rank 1 reducing patterns
  patterns.add<RankReduceMatmul<MatmulOp, VecmatOp>>(context);
  patterns.add<RankReduceMatmul<MatmulOp, MatvecOp>>(context);
  patterns.add<RankReduceMatmul<MatmulTransposeAOp, VecmatOp>>(context);
  patterns.add<RankReduceMatmul<MatmulTransposeBOp, MatvecOp>>(context);
  // Batch rank 1 reducing patterns
  patterns.add<RankReduceMatmul<BatchMatmulOp, BatchVecmatOp>>(context);
  patterns.add<RankReduceMatmul<BatchMatmulOp, BatchMatvecOp>>(context);
  patterns.add<RankReduceMatmul<BatchMatmulTransposeAOp, BatchVecmatOp>>(
      context);
  patterns.add<RankReduceMatmul<BatchMatmulTransposeBOp, BatchMatvecOp>>(
      context);

  // Non-batch rank 0 reducing patterns
  patterns.add<RankReduceMatmul<MatvecOp, DotOp>>(context);
  patterns.add<RankReduceMatmul<VecmatOp, DotOp>>(context);
}
