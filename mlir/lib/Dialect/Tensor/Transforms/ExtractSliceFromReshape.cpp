//===- ExtractSliceFromReshape.cpp - Slice reshape rewrites-------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements rewrites that replace slices of reshape results with
// aggregated slices of the reshape source.
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Arithmetic/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/TransformUtils.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::tensor;

/// Get the dimension size of a value of RankedTensor type at the
OpFoldResult getShapeDimSize(OpBuilder &b, Location loc, Value rankedTensor,
                             int64_t dimIdx) {
  RankedTensorType tensorType = rankedTensor.getType().cast<RankedTensorType>();
  if (!tensorType.isDynamicDim(dimIdx)) {
    return b.getIndexAttr(tensorType.getDimSize(dimIdx));
  }
  Value idxValue = b.create<arith::ConstantIndexOp>(loc, dimIdx);
  return b.createOrFold<tensor::DimOp>(loc, rankedTensor, idxValue);
}

/// Get all the dimension sizes of a value of RankedTensor type.
static SmallVector<OpFoldResult> getShapeDimSizes(OpBuilder &b, Location loc,
                                                  Value rankedTensor) {
  SmallVector<OpFoldResult> dimSizes;
  RankedTensorType tensorType = rankedTensor.getType().cast<RankedTensorType>();
  for (unsigned i = 0; i < tensorType.getRank(); i++)
    dimSizes.push_back(getShapeDimSize(b, loc, rankedTensor, i));
  return dimSizes;
}

/// A tuple that represents (dimension number, dimension value).
using DimAndIndex = std::tuple<unsigned, Value>;

/// Transform `dimAndIndex` from the output index space of a (non-rank-reducing)
/// slice described by `sliceParams` into the input index space.
static DimAndIndex invertSliceIndexing(OpBuilder &b, Location loc,
                                       ArrayRef<Range> sliceParams,
                                       const DimAndIndex &dimAndIndex) {
  AffineExpr d0, s0, s1;
  bindDims(b.getContext(), d0);
  bindSymbols(b.getContext(), s0, s1);
  auto [dim, indexValue] = dimAndIndex;
  assert(dim < sliceParams.size() && "slice should be non rank-reducing");
  return std::make_pair(
      dim,
      makeComposedAffineApply(
          b, loc, s0 + d0 * s1,
          {indexValue,
           getValueOrCreateConstantIndexOp(b, loc, sliceParams[dim].offset),
           getValueOrCreateConstantIndexOp(b, loc, sliceParams[dim].stride)}));
}

/// Transform `dimAndIndex` from the result tensor index space of a
/// CollapseShapeOp to the source tensor index space.
static ValueRange invertCollapseShapeIndexing(
    OpBuilder &b, Location loc, ArrayRef<ReassociationIndices> reassociation,
    ArrayRef<OpFoldResult> reshapeSourceShape, const DimAndIndex &dimAndIndex) {
  const auto &[dim, indexValue] = dimAndIndex;
  SmallVector<OpFoldResult> basis;
  for (int64_t i : reassociation[dim])
    basis.push_back(reshapeSourceShape[i]);
  auto delinearized =
      b.create<AffineDelinearizeIndexOp>(loc, indexValue, basis);
  return delinearized->getResults();
}

FailureOr<ExtractSliceFromCollapseHelper>
tensor::ExtractSliceFromCollapseHelper::create(
    OpBuilder &b, tensor::CollapseShapeOp collapseOp,
    tensor::ExtractSliceOp extractOp) {
  if (extractOp.getSource().getDefiningOp<tensor::CollapseShapeOp>() !=
      collapseOp)
    return failure();
  SmallVector<Range> ranges;
  ranges.reserve(extractOp.getSourceType().getRank());
  for (const auto &[o, s, st] :
       llvm::zip(extractOp.getMixedOffsets(), extractOp.getMixedSizes(),
                 extractOp.getMixedStrides())) {
    ranges.push_back({o, s, st});
  }
  return ExtractSliceFromCollapseHelper::create(b, collapseOp, ranges);
}

FailureOr<ExtractSliceFromCollapseHelper>
tensor::ExtractSliceFromCollapseHelper::create(OpBuilder &b,
                                               tensor::CollapseShapeOp op,
                                               ArrayRef<Range> sliceParams) {

  // Materialize the output shape of the collapse_shape operation. This will
  // create IR describing the output shape in terms of the input shape.
  ReifiedRankedShapedTypeDims reifiedShapes;
  ReifyRankedShapedTypeOpInterface reifyShapedTypeInterface =
      dyn_cast<ReifyRankedShapedTypeOpInterface>(op.getOperation());
  if (failed(reifyShapedTypeInterface.reifyResultShapes(b, reifiedShapes)))
    return failure();
  SmallVector<OpFoldResult> collapseShapeOutputShape =
      getAsOpFoldResult(reifiedShapes[0]);
  SmallVector<ReassociationIndices> reassociationIndices =
      op.getReassociationIndices();

  // Determine which of the CollapseShapeOp's result dimensions are sliced
  // and/or linearized.
  llvm::SmallBitVector linearizedDimensions =
      getLinearizedDimensions(reassociationIndices);
  llvm::SmallBitVector slicedDimensions =
      getSlicedDimensions(collapseShapeOutputShape, sliceParams);

  auto collapseShapeInputShape = getShapeDimSizes(b, op.getLoc(), op.getSrc());

  SmallVector<OpFoldResult> srcShape =
      getShapeDimSizes(b, op->getLoc(), op.getSrc());

  SmallVector<Value> tileSizes;
  for (unsigned i = 0; i < sliceParams.size(); i++) {
    if (slicedDimensions[i] && linearizedDimensions[i])
      tileSizes.push_back(
          getValueOrCreateConstantIndexOp(b, op.getLoc(), sliceParams[i].size));
  }

  return ExtractSliceFromCollapseHelper(
      op, collapseShapeInputShape, collapseShapeOutputShape, sliceParams,
      linearizedDimensions, slicedDimensions, tileSizes);
}

std::pair<Value, SmallVector<Range>>
tensor::ExtractSliceFromCollapseHelper::emitLoopNestBody(
    OpBuilder &builder, Location loc, ValueRange tileInductionVars) {
  // Create the helper class for forming the slice parameters.
  const SmallVector<ReassociationIndices> reassociationIndices =
      collapseShapeOp.getReassociationIndices();
  SliceFromCollapseHelper helper(reassociationIndices, collapseShapeInputShape,
                                 collapseShapeOutputShape, sliceParams);

  // Get the indices of the tiled dims (linearized by the collapse_shape
  // and sliced by the extract_slice) invert the index spaces
  // transformations.
  SmallVector<ValueRange> multiIndices;
  unsigned loopIdx = 0;
  for (unsigned i = 0, e = linearizedDimensions.size(); i < e; i++) {
    if (linearizedDimensions[i] && slicedDimensions[i]) {
      DimAndIndex tb =
          invertSliceIndexing(builder, loc, sliceParams,
                              std::make_tuple(i, tileInductionVars[loopIdx++]));
      multiIndices.push_back(invertCollapseShapeIndexing(
          builder, loc, reassociationIndices, collapseShapeInputShape, tb));
    }
  }

  SmallVector<Range> extractParams =
      helper.getExtractSliceParams(builder.getContext(), multiIndices);

  Value subTileResult = builder.create<tensor::ExtractSliceOp>(
      loc, collapseShapeOp.getSrc(), extractParams);

  SmallVector<Range> insertParams =
      helper.getInsertSliceParams(builder.getContext(), tileInductionVars);

  // Collapse the dimensions of the source slice back down.
  Value collapsedResult = builder.create<tensor::CollapseShapeOp>(
      loc, subTileResult, reassociationIndices);
  return std::make_pair(collapsedResult, insertParams);
}
