//===-------- SplitReduction.cpp - Split reduction dimesion ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements linalg transformation to break a reduction dimension
// between a parallel and a reduction dimension.
//
//===----------------------------------------------------------------------===//

#include <utility>

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::linalg;

/// Return the identity numeric value associated to the give op.
static Attribute getNeutralElement(Operation *op) {
  // Builder only used as helper for attribute creation.
  OpBuilder b(op->getContext());
  Type resultType = op->getResult(0).getType();
  if (auto floatType = resultType.dyn_cast<FloatType>()) {
    const llvm::fltSemantics &semantic = floatType.getFloatSemantics();
    if (isa<arith::AddFOp>(op))
      return b.getFloatAttr(resultType, llvm::APFloat::getZero(semantic));
    if (isa<arith::MulFOp>(op))
      return b.getFloatAttr(resultType, llvm::APFloat(semantic, 1));
    if (isa<arith::MaxFOp>(op))
      return b.getFloatAttr(resultType,
                            llvm::APFloat::getLargest(semantic, true));
    if (isa<arith::MinFOp>(op))
      return b.getFloatAttr(resultType,
                            llvm::APFloat::getLargest(semantic, true));
    return Attribute();
  }
  if (isa<arith::AddIOp, arith::OrIOp, arith::XOrIOp>(op))
    return b.getIntegerAttr(resultType, 0);
  if (isa<arith::AndIOp>(op))
    return b.getIntegerAttr(resultType, -1);
  if (isa<arith::MaxSIOp>(op))
    return b.getIntegerAttr(resultType, std::numeric_limits<int64_t>::min());
  if (isa<arith::MinSIOp>(op))
    return b.getIntegerAttr(resultType, std::numeric_limits<int64_t>::max());
  if (isa<arith::MulIOp>(op))
    return b.getIntegerAttr(resultType, 1);
  return Attribute();
}

FailureOr<LinalgOp> mlir::linalg::splitReduction(
    PatternRewriter &b, LinalgOp op,
    const ControlSplitReductionFn &controlSplitReductionFn,
    const LinalgTransformationFilter &filter, bool useAlloc) {
  if (failed(filter.checkAndNotify(b, op)) || !op.hasTensorSemantics() ||
      op.getNumReductionLoops() != 1 || op.getNumOutputs() != 1 ||
      !op.hasOnlyProjectedPermutations())
    return b.notifyMatchFailure(op, "precondition not met");

  FailureOr<SplitReductionResult> res =
      splitReduction(b, op, controlSplitReductionFn, useAlloc);
  if (failed(res))
    return failure();

  filter.replaceLinalgTransformationFilter(b, res->splitLinalgOp);
  filter.replaceLinalgTransformationFilter(b, res->resultCombiningLinalgOp);

  return res->splitLinalgOp;
}

FailureOr<SplitReductionResult> mlir::linalg::splitReduction(
    PatternRewriter &b, LinalgOp op,
    const ControlSplitReductionFn &controlSplitReductionFn, bool useAlloc) {
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPoint(op);

  SplitReductionOptions control = controlSplitReductionFn(op);
  int64_t ratio = control.ratio;
  unsigned insertSplitDimension = control.index;
  if (ratio <= 1)
    return b.notifyMatchFailure(op, "split ratio needs to be greater than 1");

  SmallVector<unsigned> dims;
  op.getReductionDims(dims);
  assert(dims.size() == 1);
  unsigned reductionDim = dims[0];
  SmallVector<int64_t, 4> loopRanges = op.getStaticLoopRanges();
  int64_t reductionDimSize = loopRanges[reductionDim];
  if (reductionDimSize == ShapedType::kDynamicSize ||
      reductionDimSize % ratio != 0 ||
      insertSplitDimension >= loopRanges.size())
    return b.notifyMatchFailure(
        op, "Reduction dimension not divisible by split ratio");

  SmallVector<Operation *, 4> combinerOps;
  if (!matchReduction(op.getRegionOutputArgs(), 0, combinerOps) ||
      combinerOps.size() != 1)
    return b.notifyMatchFailure(op, "Cannot match the reduction pattern");

  Operation *reductionOp = combinerOps[0];
  Attribute identity = getNeutralElement(reductionOp);
  if (!identity)
    return b.notifyMatchFailure(op, "Unknown identity value for the reduction");

  Location loc = op->getLoc();
  SmallVector<Value> newInputs;
  SmallVector<AffineMap> newMaps;
  // Calculate the new shapes and indexing maps of the input operands.
  for (OpOperand *operand : op.getInputOperands()) {
    AffineMap map = op.getTiedIndexingMap(operand);
    SmallVector<int64_t> newShape;
    SmallVector<AffineExpr> exprs;
    SmallVector<ReassociationIndices> reassociation;
    unsigned index = 0;
    for (unsigned idx : llvm::seq<unsigned>(0, map.getNumResults())) {
      unsigned dim = map.getDimPosition(idx);
      if (reductionDim == dim) {
        if (control.innerParallel) {
          newShape.push_back(op.getShape(operand)[idx] / ratio);
          newShape.push_back(ratio);
        } else {
          newShape.push_back(ratio);
          newShape.push_back(op.getShape(operand)[idx] / ratio);
        }
        reassociation.push_back({index++, index++});
        if (control.innerParallel) {
          exprs.push_back(b.getAffineDimExpr(reductionDim));
          exprs.push_back(b.getAffineDimExpr(reductionDim + 1));
        } else {
          exprs.push_back(b.getAffineDimExpr(insertSplitDimension));
          exprs.push_back(
              b.getAffineDimExpr(dim < insertSplitDimension ? dim : dim + 1));
        }
        continue;
      }
      newShape.push_back(op.getShape(operand)[idx]);
      if (control.innerParallel) {
        exprs.push_back(
            b.getAffineDimExpr(dim <= reductionDim ? dim : dim + 1));
      } else {
        exprs.push_back(
            b.getAffineDimExpr(dim < insertSplitDimension ? dim : dim + 1));
      }
      reassociation.push_back({index++});
    }
    newMaps.push_back(
        AffineMap::get(map.getNumDims() + 1, 0, exprs, op.getContext()));
    // If the shape is unchanged the input doesn't change.
    if (newShape == op.getShape(operand)) {
      newInputs.push_back(operand->get());
      continue;
    }
    Type newType = RankedTensorType::get(
        newShape,
        operand->get().getType().cast<RankedTensorType>().getElementType());
    Value newInput = b.create<tensor::ExpandShapeOp>(
        loc, newType, operand->get(), reassociation);
    newInputs.push_back(newInput);
  }

  // Calculate the new output map and shape, we insert the new dimension based
  // on the index returned by `controlSplitReductionFn`.
  SmallVector<int64_t> newOutputShape;
  AffineMap oldOutputMap = op.getTiedIndexingMap(op.getOutputOperand(0));
  ArrayRef<int64_t> oldShape = op.getShape(op.getOutputOperand(0));
  SmallVector<AffineExpr> outputExpr;
  for (unsigned idx :
       llvm::seq<unsigned>(0, oldOutputMap.getNumResults() + 1)) {
    if (idx == insertSplitDimension) {
      newOutputShape.push_back(ratio);
      if (control.innerParallel) {
        outputExpr.push_back(b.getAffineDimExpr(reductionDim + 1));
      } else {
        outputExpr.push_back(b.getAffineDimExpr(insertSplitDimension));
      }
      continue;
    }
    unsigned oldIdx = idx < insertSplitDimension ? idx : idx - 1;
    newOutputShape.push_back(oldShape[oldIdx]);
    unsigned dim = oldOutputMap.getDimPosition(oldIdx);
    if (control.innerParallel) {
      outputExpr.push_back(
          b.getAffineDimExpr(dim <= reductionDim ? dim : dim + 1));
    } else {
      outputExpr.push_back(
          b.getAffineDimExpr(dim < insertSplitDimension ? dim : dim + 1));
    }
  }
  Value initOrAllocTensor;
  if (useAlloc) {
    initOrAllocTensor = b.create<bufferization::AllocTensorOp>(
        loc,
        RankedTensorType::get(newOutputShape,
                              op.getRegionOutputArgs()[0].getType()),
        ValueRange{});
  } else {
    initOrAllocTensor = b.create<linalg::InitTensorOp>(
        loc, newOutputShape, op.getRegionOutputArgs()[0].getType());
  }
  Value constantOp = b.create<arith::ConstantOp>(loc, identity);
  Value identityTensor =
      b.create<linalg::FillOp>(op->getLoc(), constantOp, initOrAllocTensor)
          .getResult(0);

  newMaps.push_back(AffineMap::get(oldOutputMap.getNumDims() + 1, 0, outputExpr,
                                   op.getContext()));
  SmallVector<StringRef> newIteratorTypes;
  for (auto &it : llvm::enumerate(op.iterator_types())) {
    if (insertSplitDimension == it.index() && !control.innerParallel)
      newIteratorTypes.push_back(getParallelIteratorTypeName());
    newIteratorTypes.push_back(it.value().cast<StringAttr>().getValue());
    if (insertSplitDimension == it.index() && control.innerParallel)
      newIteratorTypes.push_back(getParallelIteratorTypeName());
  }
  // Create the new op matching the original op with an extra parallel
  // dimension.
  GenericOp genericOp = b.create<GenericOp>(
      loc, TypeRange({initOrAllocTensor.getType()}), newInputs,
      ValueRange({identityTensor}), newMaps, newIteratorTypes);
  b.inlineRegionBefore(op->getRegion(0), genericOp.getRegion(),
                       genericOp.getRegion().begin());

  // Then create a new reduction that only reduce the newly added dimension
  // from the previous op.
  unsigned intermRank = newOutputShape.size();
  AffineMap inputMap = b.getMultiDimIdentityMap(intermRank);
  SmallVector<Value> outputOperands = op.getOutputOperands();
  SmallVector<StringRef> reductionIteratorTypes;
  SmallVector<AffineExpr> exprs;
  for (unsigned i : llvm::seq<unsigned>(0, intermRank)) {
    if (insertSplitDimension == i) {
      reductionIteratorTypes.push_back(getReductionIteratorTypeName());
    } else {
      exprs.push_back(b.getAffineDimExpr(i));
      reductionIteratorTypes.push_back(getParallelIteratorTypeName());
    }
  }
  AffineMap outputMap = AffineMap::get(intermRank, 0, exprs, op.getContext());
  SmallVector<AffineMap> reductionMaps = {inputMap, outputMap};

  auto reduction = b.create<GenericOp>(
      loc, op->getResultTypes(), ValueRange({genericOp.getResult(0)}),
      outputOperands, reductionMaps, reductionIteratorTypes,
      [reductionOp](OpBuilder &b, Location loc, ValueRange inputs) {
        Operation *clonedReductionOp = b.clone(*reductionOp);
        clonedReductionOp->setOperand(0, inputs[0]);
        clonedReductionOp->setOperand(1, inputs[1]);
        b.create<linalg::YieldOp>(loc, clonedReductionOp->getResult(0));
      });
  b.replaceOp(op, reduction.getResults());

  return SplitReductionResult{
      initOrAllocTensor.getDefiningOp(), identityTensor.getDefiningOp<FillOp>(),
      cast<LinalgOp>(genericOp.getOperation()), reduction};
}

/// Rewrite f(i, j, k, ...) into f(i, j, k * ratio + kk, ...)
/// TODO: Additional pattern to rewrite f(i, j, k * ratio + kk, ...) into
/// f(i, j, k, kk, ...) with a proper ExpandShapeOp. This is probably better
/// done as a transform to enable better vectorization.
static AffineMap scaleReductionDim(LinalgOp op, OpOperand &opOperand,
                                   unsigned reductionDimPos,
                                   int64_t reductionRatio) {
  auto reductionDim = getAffineDimExpr(reductionDimPos, op.getContext());
  auto reductionDimP1 = getAffineDimExpr(reductionDimPos + 1, op.getContext());
  AffineMap map = op.getTiedIndexingMap(&opOperand);
  AffineMap idMap =
      AffineMap::getMultiDimIdentityMap(map.getNumDims(), op.getContext());
  AffineMap shiftedIdMap = idMap.shiftDims(1, /*offset=*/reductionDimPos + 1);
  AffineMap composeMap = shiftedIdMap.replace(
      reductionDim, reductionDim * reductionRatio + reductionDimP1,
      shiftedIdMap.getNumDims(), /*numSymbols=*/0);
  return map.compose(composeMap);
}

static AffineMap insertParallelDim(LinalgOp op, OpOperand &opOperand,
                                   unsigned reductionDimPos, int64_t size) {
  auto reductionDim = getAffineDimExpr(reductionDimPos, op.getContext());
  AffineMap map = op.getTiedIndexingMap(&opOperand);
  AffineMap idMap =
      AffineMap::getMultiDimIdentityMap(map.getNumDims(), op.getContext());
  AffineMap shiftedIdMap = idMap.shiftDims(1, /*offset=*/reductionDimPos + 1);
  return map.compose(shiftedIdMap).insertResult(reductionDim, reductionDimPos);
}

/// Core rewrite implementation.
FailureOr<SplitReductionResult> mlir::linalg::splitReductionByScaling(
    PatternRewriter &b, LinalgOp op,
    const ControlSplitReductionFn &controlSplitReductionFn, bool useAlloc) {
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPoint(op);

  // Matcher part, enforce preconditions.
  SplitReductionOptions control = controlSplitReductionFn(op);
  if (control.innerParallel)
    return b.notifyMatchFailure(op, "innerParallel not supported");

  int64_t splitFactor = control.ratio;
  unsigned insertSplitDimension = control.index;
  if (splitFactor <= 1)
    return b.notifyMatchFailure(op, "split factor needs to be greater than 1");

  SmallVector<unsigned> dims;
  op.getReductionDims(dims);
  if (dims.empty())
    return b.notifyMatchFailure(op, "needs at least 1 reduction dimension");

  unsigned reductionDimPos = dims[0];
  SmallVector<int64_t> loopRanges = op.getStaticLoopRanges();
  int64_t reductionDimSize = loopRanges[reductionDimPos];
  if (reductionDimSize == ShapedType::kDynamicSize ||
      reductionDimSize % splitFactor != 0 ||
      insertSplitDimension >= loopRanges.size())
    return b.notifyMatchFailure(
        op, "first reduction dimension not divisible by split factor");

  SmallVector<Operation *> combinerOps;
  if (!matchReduction(op.getRegionOutputArgs(), 0, combinerOps))
    return b.notifyMatchFailure(op, "cannot match a reduction pattern");

  SmallVector<Attribute> neutralElements = llvm::to_vector<4>(
      llvm::map_range(combinerOps, [&](Operation *reductionOp) {
        return getNeutralElement(reductionOp);
      }));
  if (!llvm::all_of(neutralElements, [](Attribute attr) { return attr; }))
    return b.notifyMatchFailure(op, "unknown reduction neutral");

  // TODO: relax this when multi-reduction support is available.
  if (op.getNumOutputs() != static_cast<int64_t>(neutralElements.size()))
    return b.notifyMatchFailure(op, "expect one reduction per output");

  // Rewrite part.
  // Step 1. Build the intermediate outputs filled with the proper
  // neutralElements. Such outputs are of the same shape with an extra dimension
  // inserted at `insertSplitDimension`.
  //
  // Consider a minimal example where `k` is reduced:
  //     O(i, j) += I(i, j, k)
  // Assume i=3, j=5, k=128, splitFactor=16 and insertSplitDimension=0.
  // The compute is rewritten as:
  //   a. O_i(kk, i, j) += I(i, j, 16 * k + kk)
  //   b. O(i, j) += O_i(kk, i, j)
  // The intermediate tensor O_i is of shape (128/16)x3x5 == 8x3x5.
  Location loc = op->getLoc();
  MLIRContext *context = op.getContext();
  // For now assume outputs are 1-1 with reduction neutralElements.
  // TODO: generalize when multi-reduction support is available.
  SmallVector<Value> newOutputs;
  newOutputs.reserve(op.getNumOutputs());
  SmallVector<Operation *> initOrAllocTensorOps;
  SmallVector<linalg::FillOp> fillOps;
  fillOps.reserve(op.getNumOutputs());
  for (auto it : llvm::zip(op.outputs(), neutralElements)) {
    Value rankedTensor = std::get<0>(it);
    auto t = rankedTensor.getType().cast<RankedTensorType>();
    RankedTensorType newT = RankedTensorType::Builder(t).insertDim(
        reductionDimSize / splitFactor, insertSplitDimension);
    SmallVector<Value> dims =
        tensor::createDynamicDimValues(b, loc, rankedTensor);
    Value initOrAllocTensor;
    if (useAlloc) {
      initOrAllocTensor =
          b.create<bufferization::AllocTensorOp>(loc, newT, dims);
    } else {
      initOrAllocTensor = b.create<linalg::InitTensorOp>(
          loc, dims, newT.getShape(), t.getElementType());
    }
    Value constantOp = b.create<arith::ConstantOp>(loc, std::get<1>(it));
    fillOps.push_back(
        b.create<linalg::FillOp>(op->getLoc(), constantOp, initOrAllocTensor));
    newOutputs.push_back(fillOps.back().getResult(0));
    initOrAllocTensorOps.push_back(initOrAllocTensor.getDefiningOp());
  }

  // Step 2. Reindex / expand indexing maps.
  // Reindex existing input indexings: k -> k * splitFactor + k'.
  SmallVector<AffineMap> newMaps;
  newMaps.reserve(op.getNumInputsAndOutputs() + 1);
  for (OpOperand *o : op.getInputOperands())
    newMaps.push_back(scaleReductionDim(op, *o, reductionDimPos, splitFactor));
  // Provision a new indexing for the shape-only tensor.
  auto nDims = op.getNumLoops() + 1;
  auto redDim = getAffineDimExpr(reductionDimPos, context);
  auto redDimP1 = getAffineDimExpr(reductionDimPos + 1, context);
  newMaps.push_back(AffineMap::get(nDims, 0, {redDim, redDimP1}, context));
  // Expand existing output indexings.
  // TODO: a subset of these may not reduce along reducePos and should be
  // reindexed: k -> k * splitFactor + k', when multi-reduction support is
  // available.
  for (OpOperand *o : op.getOutputOperands())
    newMaps.push_back(insertParallelDim(op, *o, reductionDimPos,
                                        reductionDimSize / splitFactor));

  // Step 3. Handle operands.
  // Compute the new input tensors.
  auto newInputs = llvm::to_vector<4>(op.inputs());
  // Add a single shape-only tensor to carry the dimensions without resorting to
  // more complex inversions.
  newInputs.push_back(b.create<linalg::InitTensorOp>(
      loc, ArrayRef<int64_t>{reductionDimSize / splitFactor, splitFactor},
      b.getIntegerType(1)));
  // Output tensors are already good to go.

  // Step 4. Create the new op matching the original op with an extra parallel
  // dimension.
  SmallVector<StringRef> iteratorTypes =
      llvm::to_vector<4>(op.getIteratorTypes().getAsValueRange<StringAttr>());
  iteratorTypes.insert(iteratorTypes.begin() + reductionDimPos,
                       getParallelIteratorTypeName());
  GenericOp genericOp =
      b.create<GenericOp>(loc, ValueRange(newOutputs).getTypes(), newInputs,
                          newOutputs, newMaps, iteratorTypes);
  b.inlineRegionBefore(op->getRegion(0), genericOp.getRegion(),
                       genericOp.getRegion().begin());
  genericOp.getRegion().front().insertArgument(reductionDimPos,
                                               b.getIntegerType(1), loc);

  // Step 5. Create new reduction ops that only reduce the newly added
  // dimensions from the previous op.
  // For now assume outputs are 1-1 with reduction ops.
  // TODO: a subset of these may not reduce in the first place and do not
  // require a new op, when multi-reduction support is available.
  // TODO: all results can be handled in a single GenericOp, when
  // multi-reduction support is available.
  SmallVector<LinalgOp> results;
  for (auto it :
       llvm::zip(genericOp->getResults(), op.outputs(), combinerOps)) {
    Value reindexedOutput = std::get<0>(it);
    Value originalOutput = std::get<1>(it);
    auto originalOutputType = originalOutput.getType().cast<RankedTensorType>();
    Operation *combinerOp = std::get<2>(it);

    AffineMap map = b.getMultiDimIdentityMap(originalOutputType.getRank() + 1);
    SmallVector<AffineMap> indexingMaps = {
        map, map.dropResult(insertSplitDimension)};
    SmallVector<StringRef> reductionIteratorTypes(
        originalOutputType.getRank() + 1, getParallelIteratorTypeName());
    reductionIteratorTypes[insertSplitDimension] =
        getReductionIteratorTypeName();

    // clang-format off
    auto reductionOp = b.create<GenericOp>(
        loc,
        originalOutputType,
        reindexedOutput,
        originalOutput,
        indexingMaps,
        reductionIteratorTypes,
        [combinerOp](OpBuilder &b, Location loc, ValueRange bbArgs) {
          Operation *clonedReductionOp = b.clone(*combinerOp);
          clonedReductionOp->setOperand(0, bbArgs[0]);
          clonedReductionOp->setOperand(1, bbArgs[1]);
          b.create<linalg::YieldOp>(loc, clonedReductionOp->getResult(0));
        });
    // clang-format on

    results.push_back(reductionOp);
  }

  // TODO: extend when multi-reduction support is available.
  assert(fillOps.size() == results.size() && results.size() == 1);
  b.replaceOp(op, results.front()->getResults());
  return SplitReductionResult{initOrAllocTensorOps.front(), fillOps.front(),
                              cast<LinalgOp>(genericOp.getOperation()),
                              results.front()};
}

namespace {

struct LinalgSplitReduction : public OpInterfaceRewritePattern<LinalgOp> {
  /// Construct a generic pattern applied to all LinalgOp that verify `filter`.
  LinalgSplitReduction(MLIRContext *context,
                       ControlSplitReductionFn controlSplitReductionFn,
                       LinalgTransformationFilter f, bool useAlloc = false,
                       PatternBenefit benefit = 1)
      : OpInterfaceRewritePattern<LinalgOp>(context, benefit),
        controlSplitReductionFn(std::move(controlSplitReductionFn)),
        useAlloc(useAlloc), filter(std::move(f)) {}

  LogicalResult matchAndRewrite(LinalgOp op,
                                PatternRewriter &rewriter) const override {
    return splitReduction(rewriter, op, controlSplitReductionFn, filter,
                          useAlloc);
  }

private:
  ControlSplitReductionFn controlSplitReductionFn;
  bool useAlloc;
  LinalgTransformationFilter filter;
};

} // namespace

void linalg::populateSplitReductionPattern(
    RewritePatternSet &patterns,
    const ControlSplitReductionFn &controlSplitReductionFn,
    const LinalgTransformationFilter &f, bool useAlloc) {
  patterns.add<LinalgSplitReduction>(patterns.getContext(),
                                     controlSplitReductionFn, f, useAlloc);
}
