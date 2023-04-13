//===- IndexingUtils.cpp - Indexing utilities supporting Linalg -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements indexing utilities for the Linalg dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/Utils/Utils.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include <optional>

#define DEBUG_TYPE "linalg-utils"

namespace mlir {
namespace linalg {
Value createOrFoldDimOp(OpBuilder &b, Location loc, Value val, int64_t dim) {
  if (val.getType().isa<UnrankedMemRefType, MemRefType>())
    return b.createOrFold<memref::DimOp>(loc, val, dim);
  if (val.getType().isa<UnrankedTensorType, RankedTensorType>())
    return b.createOrFold<tensor::DimOp>(loc, val, dim);
  llvm_unreachable("Expected MemRefType or TensorType");
}

OpFoldResult createFoldedDimOp(OpBuilder &b, Location loc, Value val,
                               int64_t dim) {
  auto shapedType = val.getType().cast<ShapedType>();
  if (!shapedType.hasRank() || shapedType.isDynamicDim(dim))
    return createOrFoldDimOp(b, loc, val, dim);
  return b.getIndexAttr(shapedType.getDimSize(dim));
}

SmallVector<Value> createDynamicDimensions(OpBuilder &b, Location loc,
                                           Value val) {
  auto shapedType = val.getType().cast<ShapedType>();
  assert(shapedType.hasRank() && "`val` must have a static rank");
  SmallVector<Value> res;
  res.reserve(shapedType.getRank());
  for (const auto &dim : llvm::enumerate(shapedType.getShape())) {
    if (dim.value() == ShapedType::kDynamic)
      res.push_back(createOrFoldDimOp(b, loc, val, dim.index()));
  }
  return res;
}

SmallVector<OpFoldResult> getMixedDimensions(OpBuilder &b, Location loc,
                                             Value val) {
  auto shapedType = val.getType().cast<ShapedType>();
  assert(shapedType.hasRank() && "`val` must have a static rank");
  SmallVector<Value> dynamicDims = createDynamicDimensions(b, loc, val);
  return getMixedValues(shapedType.getShape(), dynamicDims, b);
}
} // namespace linalg
} // namespace mlir
