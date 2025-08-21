//===- ScalableValueBoundsConstraintSet.cpp - Scalable Value Bounds -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Vector/IR/ScalableValueBoundsConstraintSet.h"

namespace mlir::vector {

FailureOr<ConstantOrScalableBound::BoundSize>
ConstantOrScalableBound::getSize() const {
  if (map.isSingleConstant())
    return BoundSize{map.getSingleConstantResult(), /*scalable=*/false};
  if (map.getNumResults() != 1 || map.getNumInputs() != 1)
    return failure();
  auto binop = dyn_cast<AffineBinaryOpExpr>(map.getResult(0));
  if (!binop || binop.getKind() != AffineExprKind::Mul)
    return failure();
  auto matchConstant = [&](AffineExpr expr, int64_t &constant) -> bool {
    if (auto cst = dyn_cast<AffineConstantExpr>(expr)) {
      constant = cst.getValue();
      return true;
    }
    return false;
  };
  // Match `s0 * cst` or `cst * s0`:
  int64_t cst = 0;
  auto lhs = binop.getLHS();
  auto rhs = binop.getRHS();
  if ((matchConstant(lhs, cst) && isa<AffineSymbolExpr>(rhs)) ||
      (matchConstant(rhs, cst) && isa<AffineSymbolExpr>(lhs))) {
    return BoundSize{cst, /*scalable=*/true};
  }
  return failure();
}

char ScalableValueBoundsConstraintSet::ID = 0;

FailureOr<ConstantOrScalableBound>
ScalableValueBoundsConstraintSet::computeScalableBound(
    Value value, std::optional<int64_t> dim, unsigned vscaleMin,
    unsigned vscaleMax, presburger::BoundType boundType, bool closedUB,
    const StopConditionFn &stopCondition) {
  using namespace presburger;
  assert(vscaleMin <= vscaleMax);

  // No stop condition specified: Keep adding constraints until the worklist
  // is empty.
  auto defaultStopCondition = [&](Value v, std::optional<int64_t> dim,
                                  mlir::ValueBoundsConstraintSet &cstr) {
    return false;
  };

  ScalableValueBoundsConstraintSet scalableCstr(
      value.getContext(), stopCondition ? stopCondition : defaultStopCondition,
      vscaleMin, vscaleMax);
  int64_t pos = scalableCstr.insert(value, dim, /*isSymbol=*/false);
  scalableCstr.processWorklist();

  // Check the resulting constraints set is valid.
  if (scalableCstr.cstr.isEmpty()) {
    return failure();
  }

  // Project out all columns apart from vscale and the starting point
  // (value/dim). This should result in constraints in terms of vscale only.
  auto projectOutFn = [&](ValueDim p) {
    bool isStartingPoint =
        p.first == value &&
        p.second == dim.value_or(ValueBoundsConstraintSet::kIndexValue);
    return p.first != scalableCstr.getVscaleValue() && !isStartingPoint;
  };
  scalableCstr.projectOut(projectOutFn);
  scalableCstr.projectOutAnonymous(/*except=*/pos);
  // Also project out local variables (these are not tracked by the
  // ValueBoundsConstraintSet).
  for (unsigned i = 0, e = scalableCstr.cstr.getNumLocalVars(); i < e; ++i) {
    scalableCstr.cstr.projectOut(scalableCstr.cstr.getNumDimAndSymbolVars());
  }

  assert(scalableCstr.cstr.getNumDimAndSymbolVars() ==
             scalableCstr.positionToValueDim.size() &&
         "inconsistent mapping state");

  // Check that the only columns left are vscale and the starting point.
  for (int64_t i = 0; i < scalableCstr.cstr.getNumDimAndSymbolVars(); ++i) {
    if (i == pos)
      continue;
    if (scalableCstr.positionToValueDim[i] !=
        ValueDim(scalableCstr.getVscaleValue(),
                 ValueBoundsConstraintSet::kIndexValue)) {
      return failure();
    }
  }

  SmallVector<AffineMap, 1> lowerBound(1), upperBound(1);
  scalableCstr.cstr.getSliceBounds(pos, 1, value.getContext(), &lowerBound,
                                   &upperBound, closedUB);

  auto invalidBound = [](auto &bound) {
    return !bound[0] || bound[0].getNumResults() != 1;
  };

  AffineMap bound = [&] {
    if (boundType == BoundType::EQ && !invalidBound(lowerBound) &&
        lowerBound[0] == upperBound[0]) {
      return lowerBound[0];
    }
    if (boundType == BoundType::LB && !invalidBound(lowerBound)) {
      return lowerBound[0];
    } else if (boundType == BoundType::UB && !invalidBound(upperBound)) {
      return upperBound[0];
    }
    return AffineMap{};
  }();

  if (!bound)
    return failure();

  return ConstantOrScalableBound{bound};
}

} // namespace mlir::vector
