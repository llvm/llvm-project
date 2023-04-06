//===- ValueBoundsOpInterface.cpp - Value Bounds  -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Interfaces/ValueBoundsOpInterface.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/APSInt.h"

using namespace mlir;
using presburger::BoundType;
using presburger::VarKind;

namespace mlir {
#include "mlir/Interfaces/ValueBoundsOpInterface.cpp.inc"
} // namespace mlir

/// If ofr is a constant integer or an IntegerAttr, return the integer.
static std::optional<int64_t> getConstantIntValue(OpFoldResult ofr) {
  // Case 1: Check for Constant integer.
  if (auto val = ofr.dyn_cast<Value>()) {
    APSInt intVal;
    if (matchPattern(val, m_ConstantInt(&intVal)))
      return intVal.getSExtValue();
    return std::nullopt;
  }
  // Case 2: Check for IntegerAttr.
  Attribute attr = ofr.dyn_cast<Attribute>();
  if (auto intAttr = attr.dyn_cast_or_null<IntegerAttr>())
    return intAttr.getValue().getSExtValue();
  return std::nullopt;
}

ValueBoundsConstraintSet::ValueBoundsConstraintSet(Value value,
                                                   std::optional<int64_t> dim)
    : builder(value.getContext()) {
  insert(value, dim, /*isSymbol=*/false);
}

#ifndef NDEBUG
static void assertValidValueDim(Value value, std::optional<int64_t> dim) {
  if (value.getType().isIndex()) {
    assert(!dim.has_value() && "invalid dim value");
  } else if (auto shapedType = dyn_cast<ShapedType>(value.getType())) {
    assert(*dim >= 0 && "invalid dim value");
    if (shapedType.hasRank())
      assert(*dim < shapedType.getRank() && "invalid dim value");
  } else {
    llvm_unreachable("unsupported type");
  }
}
#endif // NDEBUG

void ValueBoundsConstraintSet::addBound(BoundType type, int64_t pos,
                                        AffineExpr expr) {
  LogicalResult status = cstr.addBound(
      type, pos,
      AffineMap::get(cstr.getNumDimVars(), cstr.getNumSymbolVars(), expr));
  (void)status;
  assert(succeeded(status) && "failed to add bound to constraint system");
}

AffineExpr ValueBoundsConstraintSet::getExpr(Value value,
                                             std::optional<int64_t> dim) {
#ifndef NDEBUG
  assertValidValueDim(value, dim);
#endif // NDEBUG

  auto shapedType = dyn_cast<ShapedType>(value.getType());
  if (shapedType) {
    // Static dimension: return constant directly.
    if (shapedType.hasRank() && !shapedType.isDynamicDim(*dim))
      return builder.getAffineConstantExpr(shapedType.getDimSize(*dim));
  } else {
    // Constant index value: return directly.
    if (auto constInt = getConstantIntValue(value))
      return builder.getAffineConstantExpr(*constInt);
  }

  // Dynamic value: add to constraint set.
  ValueDim valueDim = std::make_pair(value, dim.value_or(kIndexValue));
  if (valueDimToPosition.find(valueDim) == valueDimToPosition.end())
    (void)insert(value, dim);
  int64_t pos = getPos(value, dim);
  return pos < cstr.getNumDimVars()
             ? builder.getAffineDimExpr(pos)
             : builder.getAffineSymbolExpr(pos - cstr.getNumDimVars());
}

AffineExpr ValueBoundsConstraintSet::getExpr(OpFoldResult ofr) {
  if (Value value = ofr.dyn_cast<Value>())
    return getExpr(value, /*dim=*/std::nullopt);
  auto constInt = getConstantIntValue(ofr);
  assert(constInt.has_value() && "expected Integer constant");
  return builder.getAffineConstantExpr(*constInt);
}

AffineExpr ValueBoundsConstraintSet::getExpr(int64_t constant) {
  return builder.getAffineConstantExpr(constant);
}

int64_t ValueBoundsConstraintSet::insert(Value value,
                                         std::optional<int64_t> dim,
                                         bool isSymbol) {
#ifndef NDEBUG
  assertValidValueDim(value, dim);
#endif // NDEBUG

  ValueDim valueDim = std::make_pair(value, dim.value_or(kIndexValue));
  assert((valueDimToPosition.find(valueDim) == valueDimToPosition.end()) &&
         "already mapped");
  int64_t pos = isSymbol ? cstr.appendVar(VarKind::Symbol)
                         : cstr.appendVar(VarKind::SetDim);
  positionToValueDim.insert(positionToValueDim.begin() + pos, valueDim);
  // Update reverse mapping.
  for (int64_t i = pos, e = positionToValueDim.size(); i < e; ++i)
    valueDimToPosition[positionToValueDim[i]] = i;

  worklist.insert(pos);
  return pos;
}

int64_t ValueBoundsConstraintSet::getPos(Value value,
                                         std::optional<int64_t> dim) const {
#ifndef NDEBUG
  assertValidValueDim(value, dim);
  assert((value.isa<OpResult>() ||
          value.cast<BlockArgument>().getOwner()->isEntryBlock()) &&
         "unstructured control flow is not supported");
#endif // NDEBUG

  auto it =
      valueDimToPosition.find(std::make_pair(value, dim.value_or(kIndexValue)));
  assert(it != valueDimToPosition.end() && "expected mapped entry");
  return it->second;
}

static Operation *getOwnerOfValue(Value value) {
  if (auto bbArg = value.dyn_cast<BlockArgument>())
    return bbArg.getOwner()->getParentOp();
  return value.getDefiningOp();
}

void ValueBoundsConstraintSet::processWorklist(StopConditionFn stopCondition) {
  while (!worklist.empty()) {
    int64_t pos = worklist.pop_back_val();
    ValueDim valueDim = positionToValueDim[pos];
    Value value = valueDim.first;
    int64_t dim = valueDim.second;

    // Check for static dim size.
    if (dim != kIndexValue) {
      auto shapedType = cast<ShapedType>(value.getType());
      if (shapedType.hasRank() && !shapedType.isDynamicDim(dim)) {
        bound(value)[dim] == getExpr(shapedType.getDimSize(dim));
        continue;
      }
    }

    // Do not process any further if the stop condition is met.
    if (stopCondition(value))
      continue;

    // Query `ValueBoundsOpInterface` for constraints. New items may be added to
    // the worklist.
    auto valueBoundsOp =
        dyn_cast<ValueBoundsOpInterface>(getOwnerOfValue(value));
    if (!valueBoundsOp)
      continue;
    if (dim == kIndexValue) {
      valueBoundsOp.populateBoundsForIndexValue(value, *this);
    } else {
      valueBoundsOp.populateBoundsForShapedValueDim(value, dim, *this);
    }
  }
}

void ValueBoundsConstraintSet::projectOut(int64_t pos) {
  assert(pos >= 0 && pos < static_cast<int64_t>(positionToValueDim.size()) &&
         "invalid position");
  cstr.projectOut(pos);
  bool erased = valueDimToPosition.erase(positionToValueDim[pos]);
  (void)erased;
  assert(erased && "inconsistent reverse mapping");
  positionToValueDim.erase(positionToValueDim.begin() + pos);
  // Update reverse mapping.
  for (int64_t i = pos, e = positionToValueDim.size(); i < e; ++i)
    valueDimToPosition[positionToValueDim[i]] = i;
}

void ValueBoundsConstraintSet::projectOut(
    function_ref<bool(ValueDim)> condition) {
  int64_t nextPos = 0;
  while (nextPos < static_cast<int64_t>(positionToValueDim.size())) {
    if (condition(positionToValueDim[nextPos])) {
      projectOut(nextPos);
      // The column was projected out so another column is now at that position.
      // Do not increase the counter.
    } else {
      ++nextPos;
    }
  }
}

LogicalResult ValueBoundsConstraintSet::computeBound(
    AffineMap &resultMap, ValueDimList &mapOperands, presburger::BoundType type,
    Value value, std::optional<int64_t> dim, StopConditionFn stopCondition) {
#ifndef NDEBUG
  assertValidValueDim(value, dim);
#endif // NDEBUG

  // Only EQ bounds are supported at the moment.
  assert(type == BoundType::EQ && "unsupported bound type");

  Builder b(value.getContext());
  mapOperands.clear();

  if (stopCondition(value)) {
    // Special case: If the stop condition is satisfied for the input
    // value/dimension, directly return it.
    mapOperands.push_back(std::make_pair(value, dim));
    resultMap = AffineMap::get(/*dimCount=*/1, /*symbolCount=*/0,
                               b.getAffineDimExpr(0));
    return success();
  }

  // Process the backward slice of `value` (i.e., reverse use-def chain) until
  // `stopCondition` is met.
  ValueDim valueDim = std::make_pair(value, dim.value_or(kIndexValue));
  ValueBoundsConstraintSet cstr(value, dim);
  cstr.processWorklist(stopCondition);

  // Project out all variables (apart from `valueDim`) that do not match the
  // stop condition.
  cstr.projectOut([&](ValueDim p) {
    // Do not project out `valueDim`.
    if (valueDim == p)
      return false;
    return !stopCondition(p.first);
  });

  // Compute lower and upper bounds for `valueDim`.
  int64_t pos = cstr.getPos(value, dim);
  SmallVector<AffineMap> lb(1), ub(1);
  cstr.cstr.getSliceBounds(pos, 1, value.getContext(), &lb, &ub,
                           /*getClosedUB=*/true);
  // Note: There are TODOs in the implementation of `getSliceBounds`. In such a
  // case, no lower/upper bound can be computed at the moment.
  if (lb.empty() || !lb[0] || ub.empty() || !ub[0] ||
      lb[0].getNumResults() != 1 || ub[0].getNumResults() != 1)
    return failure();

  // Look for same lower and upper bound: EQ bound.
  if (ub[0] != lb[0])
    return failure();

  // Gather all SSA values that are used in the computed bound.
  assert(cstr.cstr.getNumDimAndSymbolVars() == cstr.positionToValueDim.size() &&
         "inconsistent mapping state");
  SmallVector<AffineExpr> replacementDims, replacementSymbols;
  int64_t numDims = 0, numSymbols = 0;
  for (int64_t i = 0; i < cstr.cstr.getNumDimAndSymbolVars(); ++i) {
    // Skip `value`.
    if (i == pos)
      continue;
    // Check if the position `i` is used in the generated bound. If so, it must
    // be included in the generated affine.apply op.
    bool used = false;
    bool isDim = i < cstr.cstr.getNumDimVars();
    if (isDim) {
      if (lb[0].isFunctionOfDim(i))
        used = true;
    } else {
      if (lb[0].isFunctionOfSymbol(i - cstr.cstr.getNumDimVars()))
        used = true;
    }

    if (!used) {
      // Not used: Remove dim/symbol from the result.
      if (isDim) {
        replacementDims.push_back(b.getAffineConstantExpr(0));
      } else {
        replacementSymbols.push_back(b.getAffineConstantExpr(0));
      }
      continue;
    }

    if (isDim) {
      replacementDims.push_back(b.getAffineDimExpr(numDims++));
    } else {
      replacementSymbols.push_back(b.getAffineSymbolExpr(numSymbols++));
    }

    ValueBoundsConstraintSet::ValueDim valueDim = cstr.positionToValueDim[i];
    Value value = valueDim.first;
    int64_t dim = valueDim.second;
    if (dim == ValueBoundsConstraintSet::kIndexValue) {
      // An index-type value is used: can be used directly in the affine.apply
      // op.
      assert(value.getType().isIndex() && "expected index type");
      mapOperands.push_back(std::make_pair(value, std::nullopt));
      continue;
    }

    assert(cast<ShapedType>(value.getType()).isDynamicDim(dim) &&
           "expected dynamic dim");
    mapOperands.push_back(std::make_pair(value, dim));
  }

  resultMap = lb[0].replaceDimsAndSymbols(replacementDims, replacementSymbols,
                                          numDims, numSymbols);
  return success();
}

ValueBoundsConstraintSet::BoundBuilder &
ValueBoundsConstraintSet::BoundBuilder::operator[](int64_t dim) {
  assert(!this->dim.has_value() && "dim was already set");
  this->dim = dim;
#ifndef NDEBUG
  assertValidValueDim(value, this->dim);
#endif // NDEBUG
  return *this;
}

void ValueBoundsConstraintSet::BoundBuilder::operator<(AffineExpr expr) {
#ifndef NDEBUG
  assertValidValueDim(value, this->dim);
#endif // NDEBUG
  cstr.addBound(BoundType::UB, cstr.getPos(value, this->dim), expr);
}

void ValueBoundsConstraintSet::BoundBuilder::operator<=(AffineExpr expr) {
  operator<(expr + 1);
}

void ValueBoundsConstraintSet::BoundBuilder::operator>(AffineExpr expr) {
  operator>=(expr + 1);
}

void ValueBoundsConstraintSet::BoundBuilder::operator>=(AffineExpr expr) {
#ifndef NDEBUG
  assertValidValueDim(value, this->dim);
#endif // NDEBUG
  cstr.addBound(BoundType::LB, cstr.getPos(value, this->dim), expr);
}

void ValueBoundsConstraintSet::BoundBuilder::operator==(AffineExpr expr) {
#ifndef NDEBUG
  assertValidValueDim(value, this->dim);
#endif // NDEBUG
  cstr.addBound(BoundType::EQ, cstr.getPos(value, this->dim), expr);
}

void ValueBoundsConstraintSet::BoundBuilder::operator<(OpFoldResult ofr) {
  operator<(cstr.getExpr(ofr));
}

void ValueBoundsConstraintSet::BoundBuilder::operator<=(OpFoldResult ofr) {
  operator<=(cstr.getExpr(ofr));
}

void ValueBoundsConstraintSet::BoundBuilder::operator>(OpFoldResult ofr) {
  operator>(cstr.getExpr(ofr));
}

void ValueBoundsConstraintSet::BoundBuilder::operator>=(OpFoldResult ofr) {
  operator>=(cstr.getExpr(ofr));
}

void ValueBoundsConstraintSet::BoundBuilder::operator==(OpFoldResult ofr) {
  operator==(cstr.getExpr(ofr));
}

void ValueBoundsConstraintSet::BoundBuilder::operator<(int64_t i) {
  operator<(cstr.getExpr(i));
}

void ValueBoundsConstraintSet::BoundBuilder::operator<=(int64_t i) {
  operator<=(cstr.getExpr(i));
}

void ValueBoundsConstraintSet::BoundBuilder::operator>(int64_t i) {
  operator>(cstr.getExpr(i));
}

void ValueBoundsConstraintSet::BoundBuilder::operator>=(int64_t i) {
  operator>=(cstr.getExpr(i));
}

void ValueBoundsConstraintSet::BoundBuilder::operator==(int64_t i) {
  operator==(cstr.getExpr(i));
}
