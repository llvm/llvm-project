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

using namespace mlir;
using namespace mlir::tensor;

PadOp mlir::tensor::createPadScalarOp(Type type, Value source, Value pad,
                                      ArrayRef<OpFoldResult> low,
                                      ArrayRef<OpFoldResult> high, bool nofold,
                                      Location loc, OpBuilder &builder) {
  auto padTensorOp =
      builder.create<PadOp>(loc, type, source, low, high, nofold);
  int rank = padTensorOp.getResultType().getRank();
  SmallVector<Type> blockArgTypes(rank, builder.getIndexType());
  SmallVector<Location> blockArgLocs(rank, loc);
  auto &region = padTensorOp.getRegion();
  // `builder.createBlock` changes the insertion point within the block. Create
  // a guard to reset the insertion point of the builder after it is destroyed.
  OpBuilder::InsertionGuard guard(builder);
  builder.createBlock(&region, region.end(), blockArgTypes, blockArgLocs);
  builder.create<YieldOp>(loc, pad);
  return padTensorOp;
}

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
        makeComposedAffineApply(b, loc, en.value() - d0, {dimOp}).getResult();
  }
  return createPadScalarOp(type, source, pad, low, high, nofold, loc, b);
}

SmallVector<Value> mlir::tensor::createDynamicDimValues(OpBuilder &b,
                                                        Location loc,
                                                        Value rankedTensor) {
  auto tensorTy = rankedTensor.getType().cast<RankedTensorType>();
  SmallVector<Value> dynamicDims;
  for (const auto &en : llvm::enumerate(tensorTy.getShape())) {
    if (en.value() == ShapedType::kDynamicSize)
      dynamicDims.push_back(
          b.create<tensor::DimOp>(loc, rankedTensor, en.index()));
  }
  return dynamicDims;
}

SmallVector<Value> mlir::tensor::createDimValues(OpBuilder &b, Location loc,
                                                 Value rankedTensor) {
  auto tensorTy = rankedTensor.getType().cast<RankedTensorType>();
  SmallVector<Value> dims;
  for (const auto &en : llvm::enumerate(tensorTy.getShape())) {
    dims.push_back(
        b.createOrFold<tensor::DimOp>(loc, rankedTensor, en.index()));
  }
  return dims;
}
