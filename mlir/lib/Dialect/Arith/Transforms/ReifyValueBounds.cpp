//===- ReifyValueBounds.cpp --- Reify value bounds with arith ops -------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/Transforms/Transforms.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"

using namespace mlir;
using namespace mlir::arith;

/// Build Arith IR for the given affine map and its operands.
static Value buildArithValue(OpBuilder &b, Location loc, AffineMap map,
                             ValueRange operands) {
  assert(map.getNumResults() == 1 && "multiple results not supported yet");
  std::function<Value(AffineExpr)> buildExpr = [&](AffineExpr e) -> Value {
    switch (e.getKind()) {
    case AffineExprKind::Constant:
      return b.create<ConstantIndexOp>(loc,
                                       cast<AffineConstantExpr>(e).getValue());
    case AffineExprKind::DimId:
      return operands[cast<AffineDimExpr>(e).getPosition()];
    case AffineExprKind::SymbolId:
      return operands[cast<AffineSymbolExpr>(e).getPosition() +
                      map.getNumDims()];
    case AffineExprKind::Add: {
      auto binaryExpr = cast<AffineBinaryOpExpr>(e);
      return b.create<AddIOp>(loc, buildExpr(binaryExpr.getLHS()),
                              buildExpr(binaryExpr.getRHS()));
    }
    case AffineExprKind::Mul: {
      auto binaryExpr = cast<AffineBinaryOpExpr>(e);
      return b.create<MulIOp>(loc, buildExpr(binaryExpr.getLHS()),
                              buildExpr(binaryExpr.getRHS()));
    }
    case AffineExprKind::FloorDiv: {
      auto binaryExpr = cast<AffineBinaryOpExpr>(e);
      return b.create<DivSIOp>(loc, buildExpr(binaryExpr.getLHS()),
                               buildExpr(binaryExpr.getRHS()));
    }
    case AffineExprKind::CeilDiv: {
      auto binaryExpr = cast<AffineBinaryOpExpr>(e);
      return b.create<CeilDivSIOp>(loc, buildExpr(binaryExpr.getLHS()),
                                   buildExpr(binaryExpr.getRHS()));
    }
    case AffineExprKind::Mod: {
      auto binaryExpr = cast<AffineBinaryOpExpr>(e);
      return b.create<RemSIOp>(loc, buildExpr(binaryExpr.getLHS()),
                               buildExpr(binaryExpr.getRHS()));
    }
    }
    llvm_unreachable("unsupported AffineExpr kind");
  };
  return buildExpr(map.getResult(0));
}

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

  // Check for special cases where no arith ops are needed.
  if (boundMap.isSingleConstant()) {
    // Bound is a constant: return an IntegerAttr.
    return static_cast<OpFoldResult>(
        b.getIndexAttr(boundMap.getSingleConstantResult()));
  }
  // No arith ops are needed if the bound is a single SSA value.
  if (auto expr = dyn_cast<AffineDimExpr>(boundMap.getResult(0)))
    return static_cast<OpFoldResult>(operands[expr.getPosition()]);
  if (auto expr = dyn_cast<AffineSymbolExpr>(boundMap.getResult(0)))
    return static_cast<OpFoldResult>(
        operands[expr.getPosition() + boundMap.getNumDims()]);
  // General case: build Arith ops.
  return static_cast<OpFoldResult>(buildArithValue(b, loc, boundMap, operands));
}

FailureOr<OpFoldResult> mlir::arith::reifyShapedValueDimBound(
    OpBuilder &b, Location loc, presburger::BoundType type, Value value,
    int64_t dim, ValueBoundsConstraintSet::StopConditionFn stopCondition,
    bool closedUB) {
  auto reifyToOperands = [&](Value v, std::optional<int64_t> d) {
    // We are trying to reify a bound for `value` in terms of the owning op's
    // operands. Construct a stop condition that evaluates to "true" for any SSA
    // value expect for `value`. I.e., the bound will be computed in terms of
    // any SSA values expect for `value`. The first such values are operands of
    // the owner of `value`.
    return v != value;
  };
  return reifyValueBound(b, loc, type, value, dim,
                         stopCondition ? stopCondition : reifyToOperands,
                         closedUB);
}

FailureOr<OpFoldResult> mlir::arith::reifyIndexValueBound(
    OpBuilder &b, Location loc, presburger::BoundType type, Value value,
    ValueBoundsConstraintSet::StopConditionFn stopCondition, bool closedUB) {
  auto reifyToOperands = [&](Value v, std::optional<int64_t> d) {
    return v != value;
  };
  return reifyValueBound(b, loc, type, value, /*dim=*/std::nullopt,
                         stopCondition ? stopCondition : reifyToOperands,
                         closedUB);
}
