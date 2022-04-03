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
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::linalg;

/// Return the identity numeric value associated to the give op.
static Optional<Attribute> getIdentity(Operation *op) {
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
    return llvm::None;
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
  return llvm::None;
}

FailureOr<LinalgOp> mlir::linalg::splitReduction(
    PatternRewriter &b, LinalgOp op,
    const ControlSplitReductionFn &controlSplitReductionFn,
    const LinalgTransformationFilter &filter) {
  if (failed(filter.checkAndNotify(b, op)) || !op.hasTensorSemantics() ||
      op.getNumReductionLoops() != 1 || op.getNumOutputs() != 1 ||
      !op.hasOnlyProjectedPermutations())
    return b.notifyMatchFailure(op, "precondition not met");
  std::pair<int64_t, unsigned> control = controlSplitReductionFn(op);
  int64_t ratio = control.first;
  unsigned insertDimIndex = control.second;
  if (ratio <= 1)
    return b.notifyMatchFailure(op, "split ratio needs to be greater than 1");
  SmallVector<unsigned> dims;
  op.getReductionDims(dims);
  assert(dims.size() == 1);
  unsigned reductionDim = dims[0];
  Optional<SmallVector<int64_t, 4>> loopRanges = op.getStaticLoopRanges();
  if (!loopRanges)
    return b.notifyMatchFailure(op, "Cannot analyze loops");
  int64_t reductionDimSize = (*loopRanges)[reductionDim];
  if (reductionDimSize == ShapedType::kDynamicSize ||
      reductionDimSize % ratio != 0 || insertDimIndex >= loopRanges->size())
    return b.notifyMatchFailure(
        op, "Reduction dimension not divisible by split ratio");
  SmallVector<Operation *, 4> combinerOps;
  if (!matchReduction(op.getRegionOutputArgs(), 0, combinerOps) ||
      combinerOps.size() != 1)
    return b.notifyMatchFailure(op, "Cannot match the reduction pattern");
  Operation *reductionOp = combinerOps[0];
  Optional<Attribute> identity = getIdentity(reductionOp);
  if (!identity)
    return b.notifyMatchFailure(op, "Unknown identity value for the redution");

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
        newShape.push_back(ratio);
        newShape.push_back(op.getShape(operand)[idx] / ratio);
        reassociation.push_back({index++, index++});
        exprs.push_back(b.getAffineDimExpr(insertDimIndex));
        exprs.push_back(
            b.getAffineDimExpr(dim < insertDimIndex ? dim : dim + 1));
        continue;
      }
      newShape.push_back(op.getShape(operand)[idx]);
      exprs.push_back(b.getAffineDimExpr(dim < insertDimIndex ? dim : dim + 1));
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
    if (idx == insertDimIndex) {
      newOutputShape.push_back(ratio);
      outputExpr.push_back(b.getAffineDimExpr(insertDimIndex));
      continue;
    }
    unsigned oldDim = idx < insertDimIndex ? idx : idx - 1;
    newOutputShape.push_back(oldShape[oldDim]);
    unsigned dim = oldOutputMap.getDimPosition(oldDim);
    outputExpr.push_back(
        b.getAffineDimExpr(dim < insertDimIndex ? dim : dim + 1));
  }
  Value initTensor = b.create<linalg::InitTensorOp>(
      loc, newOutputShape, op.getRegionOutputArgs()[0].getType());
  Value constantOp = b.create<arith::ConstantOp>(loc, *identity);
  Value identityTensor =
      b.create<linalg::FillOp>(op->getLoc(), constantOp, initTensor)
          .getResult(0);

  newMaps.push_back(AffineMap::get(oldOutputMap.getNumDims() + 1, 0, outputExpr,
                                   op.getContext()));
  SmallVector<StringRef> newIteratorTypes;
  for (auto &it : llvm::enumerate(op.iterator_types())) {
    if (insertDimIndex == it.index())
      newIteratorTypes.push_back(getParallelIteratorTypeName());
    newIteratorTypes.push_back(it.value().cast<StringAttr>().getValue());
  }
  // Create the new op matching the original op with an extra parallel
  // dimension.
  GenericOp genericOp = b.create<GenericOp>(
      loc, TypeRange({initTensor.getType()}), newInputs,
      ValueRange({identityTensor}), newMaps, newIteratorTypes);
  b.inlineRegionBefore(op->getRegion(0), genericOp.region(),
                       genericOp.region().begin());

  // Then create a new reduction that only reduce the newly added dimension from
  // the previous op.
  unsigned intermRank = newOutputShape.size();
  AffineMap inputMap = b.getMultiDimIdentityMap(intermRank);
  SmallVector<Value> outputOperands = op.getOutputOperands();
  SmallVector<StringRef> reductionIteratorTypes;
  SmallVector<AffineExpr> exprs;
  for (unsigned i : llvm::seq<unsigned>(0, intermRank)) {
    if (insertDimIndex == i) {
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
  filter.replaceLinalgTransformationFilter(b, genericOp);
  filter.replaceLinalgTransformationFilter(b, reduction);
  return cast<LinalgOp>(genericOp.getOperation());
}

namespace {

struct LinalgSplitReduction : public OpInterfaceRewritePattern<LinalgOp> {
  /// Construct a generic pattern applied to all LinalgOp that verify `filter`.
  LinalgSplitReduction(MLIRContext *context,
                       ControlSplitReductionFn controlSplitReductionFn,
                       LinalgTransformationFilter f, PatternBenefit benefit = 1)
      : OpInterfaceRewritePattern<LinalgOp>(context, benefit),
        controlSplitReductionFn(std::move(controlSplitReductionFn)),
        filter(std::move(f)) {}

  LogicalResult matchAndRewrite(LinalgOp op,
                                PatternRewriter &rewriter) const override {
    return splitReduction(rewriter, op, controlSplitReductionFn, filter);
  }

private:
  ControlSplitReductionFn controlSplitReductionFn;
  LinalgTransformationFilter filter;
};

} // namespace

void linalg::populateSplitReductionPattern(
    RewritePatternSet &patterns,
    const ControlSplitReductionFn &controlSplitReductionFn,
    const LinalgTransformationFilter &f) {
  patterns.add<LinalgSplitReduction>(patterns.getContext(),
                                     controlSplitReductionFn, f);
}
