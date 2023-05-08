//===- Utils.cpp - Utilities to support the Tensor dialect ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements utilities for the Tensor dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tensor/Utils/Utils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"

using namespace mlir;
using namespace mlir::tensor;

PadOp mlir::tensor::createPadHighOp(RankedTensorType type, Value source,
                                    Value pad, bool nofold, Location loc,
                                    OpBuilder &b) {
  auto zero = b.createOrFold<arith::ConstantIndexOp>(loc, 0);
  SmallVector<OpFoldResult> low(type.getRank(), zero);
  SmallVector<OpFoldResult> high(type.getRank(), zero);
  for (const auto &en : enumerate(type.getShape())) {
    // Pad only the static dimensions of the result tensor type.
    if (ShapedType::isDynamic(en.value()))
      continue;
    // Compute the padding width.
    AffineExpr d0;
    bindDims(b.getContext(), d0);
    auto dimOp = b.createOrFold<tensor::DimOp>(loc, source, en.index());
    high[en.index()] =
        affine::makeComposedAffineApply(b, loc, en.value() - d0, {dimOp})
            .getResult();
  }
  return b.create<PadOp>(loc, type, source, low, high, pad, nofold);
}

SmallVector<Value> mlir::tensor::createDynamicDimValues(OpBuilder &b,
                                                        Location loc,
                                                        Value rankedTensor) {
  auto tensorTy = cast<RankedTensorType>(rankedTensor.getType());
  SmallVector<Value> dynamicDims;
  for (const auto &en : llvm::enumerate(tensorTy.getShape())) {
    if (en.value() == ShapedType::kDynamic)
      dynamicDims.push_back(
          b.create<tensor::DimOp>(loc, rankedTensor, en.index()));
  }
  return dynamicDims;
}

FailureOr<OpFoldResult> mlir::tensor::createDimValue(OpBuilder &b, Location loc,
                                                     Value rankedTensor,
                                                     int64_t dim) {
  auto tensorTy = dyn_cast<RankedTensorType>(rankedTensor.getType());
  if (!tensorTy)
    return failure();
  auto shape = tensorTy.getShape();
  if (dim >= static_cast<int64_t>(shape.size()))
    return failure();
  if (ShapedType::isDynamic(shape[dim]))
    return OpFoldResult(b.createOrFold<tensor::DimOp>(loc, rankedTensor, dim));
  return OpFoldResult(b.getIndexAttr(shape[dim]));
}

SmallVector<OpFoldResult>
mlir::tensor::createDimValues(OpBuilder &b, Location loc, Value rankedTensor) {
  auto tensorTy = cast<RankedTensorType>(rankedTensor.getType());
  SmallVector<OpFoldResult> dims;
  for (const auto &en : llvm::enumerate(tensorTy.getShape())) {
    if (ShapedType::isDynamic(en.value())) {
      dims.push_back(
          b.createOrFold<tensor::DimOp>(loc, rankedTensor, en.index()));
    } else {
      dims.push_back(b.getIndexAttr(en.value()));
    }
  }
  return dims;
}

FailureOr<RankedTensorType>
mlir::tensor::computeTransposedType(RankedTensorType rankedTensorType,
                                    ArrayRef<int64_t> transposeVector) {
  if (transposeVector.empty())
    return rankedTensorType;

  if (!isPermutationVector(transposeVector) ||
      transposeVector.size() != static_cast<size_t>(rankedTensorType.getRank()))
    return failure();

  SmallVector<int64_t> transposedShape(rankedTensorType.getShape().begin(),
                                       rankedTensorType.getShape().end());
  applyPermutationToVector(transposedShape, transposeVector);

  using RTTBuilder = RankedTensorType::Builder;
  RankedTensorType transposedTensorType =
      RTTBuilder(rankedTensorType).setShape(transposedShape);
  return transposedTensorType;
}
