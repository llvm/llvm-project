//===- IndependenceTransforms.cpp - Make ops independent of values --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tensor/Transforms/Transforms.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"

using namespace mlir;
using namespace mlir::tensor;

/// Make the given OpFoldResult independent of all independencies.
static FailureOr<OpFoldResult> makeIndependent(OpBuilder &b, Location loc,
                                               OpFoldResult ofr,
                                               ValueRange independencies) {
  if (isa<Attribute>(ofr))
    return ofr;
  Value value = cast<Value>(ofr);
  AffineMap boundMap;
  ValueDimList mapOperands;
  if (failed(ValueBoundsConstraintSet::computeIndependentBound(
          boundMap, mapOperands, presburger::BoundType::UB, value,
          independencies,
          /*closedUB=*/true)))
    return failure();
  return mlir::affine::materializeComputedBound(b, loc, boundMap, mapOperands);
}

FailureOr<Value> tensor::buildIndependentOp(OpBuilder &b, tensor::PadOp padOp,
                                            ValueRange independencies) {
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(padOp);
  Location loc = padOp.getLoc();

  // Non-constant padding not supported.
  Value constantPadding = padOp.getConstantPaddingValue();
  if (!constantPadding)
    return failure();

  SmallVector<OpFoldResult> newMixedLow, newMixedHigh;
  for (OpFoldResult ofr : padOp.getMixedLowPad()) {
    auto ub = makeIndependent(b, loc, ofr, independencies);
    if (failed(ub))
      return failure();
    newMixedLow.push_back(*ub);
  }
  for (OpFoldResult ofr : padOp.getMixedHighPad()) {
    auto ub = makeIndependent(b, loc, ofr, independencies);
    if (failed(ub))
      return failure();
    newMixedHigh.push_back(*ub);
  }

  // Return existing tensor::PadOp if nothing has changed.
  if (llvm::equal(padOp.getMixedLowPad(), newMixedLow) &&
      llvm::equal(padOp.getMixedHighPad(), newMixedHigh))
    return padOp.getResult();

  // Create a new tensor::PadOp.
  auto newPadOp = b.create<PadOp>(
      loc, padOp.getResultType(), padOp.getSource(), newMixedLow, newMixedHigh,
      constantPadding, padOp.getNofold(), /*attrs=*/ArrayRef<NamedAttribute>{});

  // Create a tensor::ExtractSliceOp.
  // Reify the result sizes of the old tensor::PadOp.
  ReifiedRankedShapedTypeDims reifiedSizes;
  ReifyRankedShapedTypeOpInterface reifyShapedTypeInterface =
      dyn_cast<ReifyRankedShapedTypeOpInterface>(padOp.getOperation());
  if (failed(reifyShapedTypeInterface.reifyResultShapes(b, reifiedSizes)))
    return failure();
  SmallVector<OpFoldResult> offsets, sizes, strides;
  for (int64_t i = 0, e = padOp.getResultType().getRank(); i < e; ++i) {
    // offset = ub(low_padding) - low_padding
    OpFoldResult prevLow = padOp.getMixedLowPad()[i];
    if (isa<Attribute>(prevLow)) {
      offsets.push_back(b.getIndexAttr(0));
    } else {
      offsets.push_back(
          b.create<affine::AffineApplyOp>(
               loc, b.getAffineDimExpr(0) - b.getAffineDimExpr(1),
               std::initializer_list<Value>{cast<Value>(newMixedLow[i]),
                                            cast<Value>(prevLow)})
              .getResult());
    }
    // size = reified result size
    if (!padOp.getResultType().isDynamicDim(i)) {
      sizes.push_back(b.getIndexAttr(padOp.getResultType().getDimSize(i)));
    } else {
      sizes.push_back(reifiedSizes[0][i]);
    }
    // stride = 1
    strides.push_back(b.getIndexAttr(1));
  }

  return b.create<ExtractSliceOp>(loc, newPadOp, offsets, sizes, strides)
      .getResult();
}

FailureOr<Value> tensor::buildIndependentOp(OpBuilder &b,
                                            tensor::EmptyOp emptyOp,
                                            ValueRange independencies) {
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(emptyOp);
  Location loc = emptyOp.getLoc();

  SmallVector<OpFoldResult> newSizes;
  for (OpFoldResult ofr : emptyOp.getMixedSizes()) {
    auto ub = makeIndependent(b, loc, ofr, independencies);
    if (failed(ub))
      return failure();
    newSizes.push_back(*ub);
  }

  // Return existing tensor::EmptyOp if nothing has changed.
  if (llvm::equal(emptyOp.getMixedSizes(), newSizes))
    return emptyOp.getResult();

  // Create a new tensor::EmptyOp.
  Value newEmptyOp =
      b.create<EmptyOp>(loc, newSizes, emptyOp.getType().getElementType());

  // Create a tensor::ExtractSliceOp.
  SmallVector<OpFoldResult> offsets(newSizes.size(), b.getIndexAttr(0));
  SmallVector<OpFoldResult> strides(newSizes.size(), b.getIndexAttr(1));
  return b
      .create<ExtractSliceOp>(loc, newEmptyOp, offsets, emptyOp.getMixedSizes(),
                              strides)
      .getResult();
}
