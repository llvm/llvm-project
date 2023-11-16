//===- ReifyValueBounds.cpp --- Reify value bounds with affine ops ------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/Transforms/Transforms.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"

using namespace mlir;
using namespace mlir::affine;

static FailureOr<OpFoldResult>
reifyValueBound(OpBuilder &b, Location loc, presburger::BoundType type,
                Value value, std::optional<int64_t> dim,
                ValueBoundsConstraintSet::StopConditionFn stopCondition,
                bool closedUB) {
  // Compute bound.
  AffineMap boundMap;
  ValueDimList mapOperands;
  if (failed(ValueBoundsConstraintSet::computeBound(
          boundMap, mapOperands, type, value, dim, stopCondition, closedUB)))
    return failure();

  // Reify bound.
  return affine::materializeComputedBound(b, loc, boundMap, mapOperands);
}

OpFoldResult affine::materializeComputedBound(
    OpBuilder &b, Location loc, AffineMap boundMap,
    ArrayRef<std::pair<Value, std::optional<int64_t>>> mapOperands) {
  // Materialize tensor.dim/memref.dim ops.
  SmallVector<Value> operands;
  for (auto valueDim : mapOperands) {
    Value value = valueDim.first;
    std::optional<int64_t> dim = valueDim.second;

    if (!dim.has_value()) {
      // This is an index-typed value.
      assert(value.getType().isIndex() && "expected index type");
      operands.push_back(value);
      continue;
    }

    assert(cast<ShapedType>(value.getType()).isDynamicDim(*dim) &&
           "expected dynamic dim");
    if (isa<RankedTensorType>(value.getType())) {
      // A tensor dimension is used: generate a tensor.dim.
      operands.push_back(b.create<tensor::DimOp>(loc, value, *dim));
    } else if (isa<MemRefType>(value.getType())) {
      // A memref dimension is used: generate a memref.dim.
      operands.push_back(b.create<memref::DimOp>(loc, value, *dim));
    } else {
      llvm_unreachable("cannot generate DimOp for unsupported shaped type");
    }
  }

  // Simplify and return bound.
  affine::canonicalizeMapAndOperands(&boundMap, &operands);
  // Check for special cases where no affine.apply op is needed.
  if (boundMap.isSingleConstant()) {
    // Bound is a constant: return an IntegerAttr.
    return static_cast<OpFoldResult>(
        b.getIndexAttr(boundMap.getSingleConstantResult()));
  }
  // No affine.apply op is needed if the bound is a single SSA value.
  if (auto expr = dyn_cast<AffineDimExpr>(boundMap.getResult(0)))
    return static_cast<OpFoldResult>(operands[expr.getPosition()]);
  if (auto expr = dyn_cast<AffineSymbolExpr>(boundMap.getResult(0)))
    return static_cast<OpFoldResult>(
        operands[expr.getPosition() + boundMap.getNumDims()]);
  // General case: build affine.apply op.
  return static_cast<OpFoldResult>(
      b.create<affine::AffineApplyOp>(loc, boundMap, operands).getResult());
}

FailureOr<OpFoldResult> mlir::affine::reifyShapedValueDimBound(
    OpBuilder &b, Location loc, presburger::BoundType type, Value value,
    int64_t dim, ValueBoundsConstraintSet::StopConditionFn stopCondition,
    bool closedUB) {
  auto reifyToOperands = [&](Value v, std::optional<int64_t> d) {
    // We are trying to reify a bound for `value` in terms of the owning op's
    // operands. Construct a stop condition that evaluates to "true" for any SSA
    // value except for `value`. I.e., the bound will be computed in terms of
    // any SSA values except for `value`. The first such values are operands of
    // the owner of `value`.
    return v != value;
  };
  return reifyValueBound(b, loc, type, value, dim,
                         stopCondition ? stopCondition : reifyToOperands,
                         closedUB);
}

FailureOr<OpFoldResult> mlir::affine::reifyIndexValueBound(
    OpBuilder &b, Location loc, presburger::BoundType type, Value value,
    ValueBoundsConstraintSet::StopConditionFn stopCondition, bool closedUB) {
  auto reifyToOperands = [&](Value v, std::optional<int64_t> d) {
    return v != value;
  };
  return reifyValueBound(b, loc, type, value, /*dim=*/std::nullopt,
                         stopCondition ? stopCondition : reifyToOperands,
                         closedUB);
}
