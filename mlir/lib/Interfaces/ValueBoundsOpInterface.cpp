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
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "value-bounds-op-interface"

using namespace mlir;
using presburger::BoundType;
using presburger::VarKind;

namespace mlir {
#include "mlir/Interfaces/ValueBoundsOpInterface.cpp.inc"
} // namespace mlir

static Operation *getOwnerOfValue(Value value) {
  if (auto bbArg = dyn_cast<BlockArgument>(value))
    return bbArg.getOwner()->getParentOp();
  return value.getDefiningOp();
}

HyperrectangularSlice::HyperrectangularSlice(ArrayRef<OpFoldResult> offsets,
                                             ArrayRef<OpFoldResult> sizes,
                                             ArrayRef<OpFoldResult> strides)
    : mixedOffsets(offsets), mixedSizes(sizes), mixedStrides(strides) {
  assert(offsets.size() == sizes.size() &&
         "expected same number of offsets, sizes, strides");
  assert(offsets.size() == strides.size() &&
         "expected same number of offsets, sizes, strides");
}

HyperrectangularSlice::HyperrectangularSlice(ArrayRef<OpFoldResult> offsets,
                                             ArrayRef<OpFoldResult> sizes)
    : mixedOffsets(offsets), mixedSizes(sizes) {
  assert(offsets.size() == sizes.size() &&
         "expected same number of offsets and sizes");
  // Assume that all strides are 1.
  if (offsets.empty())
    return;
  MLIRContext *ctx = offsets.front().getContext();
  mixedStrides.append(offsets.size(), Builder(ctx).getIndexAttr(1));
}

HyperrectangularSlice::HyperrectangularSlice(OffsetSizeAndStrideOpInterface op)
    : HyperrectangularSlice(op.getMixedOffsets(), op.getMixedSizes(),
                            op.getMixedStrides()) {}

/// If ofr is a constant integer or an IntegerAttr, return the integer.
static std::optional<int64_t> getConstantIntValue(OpFoldResult ofr) {
  // Case 1: Check for Constant integer.
  if (auto val = llvm::dyn_cast_if_present<Value>(ofr)) {
    APSInt intVal;
    if (matchPattern(val, m_ConstantInt(&intVal)))
      return intVal.getSExtValue();
    return std::nullopt;
  }
  // Case 2: Check for IntegerAttr.
  Attribute attr = llvm::dyn_cast_if_present<Attribute>(ofr);
  if (auto intAttr = dyn_cast_or_null<IntegerAttr>(attr))
    return intAttr.getValue().getSExtValue();
  return std::nullopt;
}

ValueBoundsConstraintSet::Variable::Variable(OpFoldResult ofr)
    : Variable(ofr, std::nullopt) {}

ValueBoundsConstraintSet::Variable::Variable(Value indexValue)
    : Variable(static_cast<OpFoldResult>(indexValue)) {}

ValueBoundsConstraintSet::Variable::Variable(Value shapedValue, int64_t dim)
    : Variable(static_cast<OpFoldResult>(shapedValue), std::optional(dim)) {}

ValueBoundsConstraintSet::Variable::Variable(OpFoldResult ofr,
                                             std::optional<int64_t> dim) {
  Builder b(ofr.getContext());
  if (auto constInt = ::getConstantIntValue(ofr)) {
    assert(!dim && "expected no dim for index-typed values");
    map = AffineMap::get(/*dimCount=*/0, /*symbolCount=*/0,
                         b.getAffineConstantExpr(*constInt));
    return;
  }
  Value value = cast<Value>(ofr);
#ifndef NDEBUG
  if (dim) {
    assert(isa<ShapedType>(value.getType()) && "expected shaped type");
  } else {
    assert(value.getType().isIndex() && "expected index type");
  }
#endif // NDEBUG
  map = AffineMap::get(/*dimCount=*/0, /*symbolCount=*/1,
                       b.getAffineSymbolExpr(0));
  mapOperands.emplace_back(value, dim);
}

ValueBoundsConstraintSet::Variable::Variable(AffineMap map,
                                             ArrayRef<Variable> mapOperands) {
  assert(map.getNumResults() == 1 && "expected single result");

  // Turn all dims into symbols.
  Builder b(map.getContext());
  SmallVector<AffineExpr> dimReplacements, symReplacements;
  for (int64_t i = 0, e = map.getNumDims(); i < e; ++i)
    dimReplacements.push_back(b.getAffineSymbolExpr(i));
  for (int64_t i = 0, e = map.getNumSymbols(); i < e; ++i)
    symReplacements.push_back(b.getAffineSymbolExpr(i + map.getNumDims()));
  AffineMap tmpMap = map.replaceDimsAndSymbols(
      dimReplacements, symReplacements, /*numResultDims=*/0,
      /*numResultSyms=*/map.getNumSymbols() + map.getNumDims());

  // Inline operands.
  DenseMap<AffineExpr, AffineExpr> replacements;
  for (auto [index, var] : llvm::enumerate(mapOperands)) {
    assert(var.map.getNumResults() == 1 && "expected single result");
    assert(var.map.getNumDims() == 0 && "expected only symbols");
    SmallVector<AffineExpr> symReplacements;
    for (auto valueDim : var.mapOperands) {
      auto it = llvm::find(this->mapOperands, valueDim);
      if (it != this->mapOperands.end()) {
        // There is already a symbol for this operand.
        symReplacements.push_back(b.getAffineSymbolExpr(
            std::distance(this->mapOperands.begin(), it)));
      } else {
        // This is a new operand: add a new symbol.
        symReplacements.push_back(
            b.getAffineSymbolExpr(this->mapOperands.size()));
        this->mapOperands.push_back(valueDim);
      }
    }
    replacements[b.getAffineSymbolExpr(index)] =
        var.map.getResult(0).replaceSymbols(symReplacements);
  }
  this->map = tmpMap.replace(replacements, /*numResultDims=*/0,
                             /*numResultSyms=*/this->mapOperands.size());
}

ValueBoundsConstraintSet::Variable::Variable(AffineMap map,
                                             ArrayRef<Value> mapOperands)
    : Variable(map, llvm::map_to_vector(mapOperands,
                                        [](Value v) { return Variable(v); })) {}

ValueBoundsConstraintSet::ValueBoundsConstraintSet(
    MLIRContext *ctx, StopConditionFn stopCondition,
    bool addConservativeSemiAffineBounds)
    : builder(ctx), stopCondition(stopCondition),
      addConservativeSemiAffineBounds(addConservativeSemiAffineBounds) {
  assert(stopCondition && "expected non-null stop condition");
}

char ValueBoundsConstraintSet::ID = 0;

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
  // Note: If `addConservativeSemiAffineBounds` is true then the bound
  // computation function needs to handle the case that the constraints set
  // could become empty. This is because the conservative bounds add assumptions
  // (e.g. for `mod` it assumes `rhs > 0`). If these constraints are later found
  // not to hold, then the bound is invalid.
  LogicalResult status = cstr.addBound(
      type, pos,
      AffineMap::get(cstr.getNumDimVars(), cstr.getNumSymbolVars(), expr),
      addConservativeSemiAffineBounds
          ? FlatLinearConstraints::AddConservativeSemiAffineBounds::Yes
          : FlatLinearConstraints::AddConservativeSemiAffineBounds::No);
  if (failed(status)) {
    // Not all semi-affine expressions are not yet supported by
    // FlatLinearConstraints. However, we can just ignore such failures here.
    // Even without this bound, there may be enough information in the
    // constraint system to compute the requested bound. In case this bound is
    // actually needed, `computeBound` will return `failure`.
    LLVM_DEBUG(llvm::dbgs() << "Failed to add bound: " << expr << "\n");
  }
}

AffineExpr ValueBoundsConstraintSet::getExpr(Value value,
                                             std::optional<int64_t> dim) {
#ifndef NDEBUG
  assertValidValueDim(value, dim);
#endif // NDEBUG

  // Check if the value/dim is statically known. In that case, an affine
  // constant expression should be returned. This allows us to support
  // multiplications with constants. (Multiplications of two columns in the
  // constraint set is not supported.)
  std::optional<int64_t> constSize = std::nullopt;
  auto shapedType = dyn_cast<ShapedType>(value.getType());
  if (shapedType) {
    if (shapedType.hasRank() && !shapedType.isDynamicDim(*dim))
      constSize = shapedType.getDimSize(*dim);
  } else if (auto constInt = ::getConstantIntValue(value)) {
    constSize = *constInt;
  }

  // If the value/dim is already mapped, return the corresponding expression
  // directly.
  ValueDim valueDim = std::make_pair(value, dim.value_or(kIndexValue));
  if (valueDimToPosition.contains(valueDim)) {
    // If it is a constant, return an affine constant expression. Otherwise,
    // return an affine expression that represents the respective column in the
    // constraint set.
    if (constSize)
      return builder.getAffineConstantExpr(*constSize);
    return getPosExpr(getPos(value, dim));
  }

  if (constSize) {
    // Constant index value/dim: add column to the constraint set, add EQ bound
    // and return an affine constant expression without pushing the newly added
    // column to the worklist.
    (void)insert(value, dim, /*isSymbol=*/true, /*addToWorklist=*/false);
    if (shapedType)
      bound(value)[*dim] == *constSize;
    else
      bound(value) == *constSize;
    return builder.getAffineConstantExpr(*constSize);
  }

  // Dynamic value/dim: insert column to the constraint set and put it on the
  // worklist. Return an affine expression that represents the newly inserted
  // column in the constraint set.
  return getPosExpr(insert(value, dim, /*isSymbol=*/true));
}

AffineExpr ValueBoundsConstraintSet::getExpr(OpFoldResult ofr) {
  if (Value value = llvm::dyn_cast_if_present<Value>(ofr))
    return getExpr(value, /*dim=*/std::nullopt);
  auto constInt = ::getConstantIntValue(ofr);
  assert(constInt.has_value() && "expected Integer constant");
  return builder.getAffineConstantExpr(*constInt);
}

AffineExpr ValueBoundsConstraintSet::getExpr(int64_t constant) {
  return builder.getAffineConstantExpr(constant);
}

int64_t ValueBoundsConstraintSet::insert(Value value,
                                         std::optional<int64_t> dim,
                                         bool isSymbol, bool addToWorklist) {
#ifndef NDEBUG
  assertValidValueDim(value, dim);
#endif // NDEBUG

  ValueDim valueDim = std::make_pair(value, dim.value_or(kIndexValue));
  assert(!valueDimToPosition.contains(valueDim) && "already mapped");
  int64_t pos = isSymbol ? cstr.appendVar(VarKind::Symbol)
                         : cstr.appendVar(VarKind::SetDim);
  LLVM_DEBUG(llvm::dbgs() << "Inserting constraint set column " << pos
                          << " for: " << value
                          << " (dim: " << dim.value_or(kIndexValue)
                          << ", owner: " << getOwnerOfValue(value)->getName()
                          << ")\n");
  positionToValueDim.insert(positionToValueDim.begin() + pos, valueDim);
  // Update reverse mapping.
  for (int64_t i = pos, e = positionToValueDim.size(); i < e; ++i)
    if (positionToValueDim[i].has_value())
      valueDimToPosition[*positionToValueDim[i]] = i;

  if (addToWorklist) {
    LLVM_DEBUG(llvm::dbgs() << "Push to worklist: " << value
                            << " (dim: " << dim.value_or(kIndexValue) << ")\n");
    worklist.push(pos);
  }

  return pos;
}

int64_t ValueBoundsConstraintSet::insert(bool isSymbol) {
  int64_t pos = isSymbol ? cstr.appendVar(VarKind::Symbol)
                         : cstr.appendVar(VarKind::SetDim);
  LLVM_DEBUG(llvm::dbgs() << "Inserting anonymous constraint set column " << pos
                          << "\n");
  positionToValueDim.insert(positionToValueDim.begin() + pos, std::nullopt);
  // Update reverse mapping.
  for (int64_t i = pos, e = positionToValueDim.size(); i < e; ++i)
    if (positionToValueDim[i].has_value())
      valueDimToPosition[*positionToValueDim[i]] = i;
  return pos;
}

int64_t ValueBoundsConstraintSet::insert(AffineMap map, ValueDimList operands,
                                         bool isSymbol) {
  assert(map.getNumResults() == 1 && "expected affine map with one result");
  int64_t pos = insert(isSymbol);

  // Add map and operands to the constraint set. Dimensions are converted to
  // symbols. All operands are added to the worklist (unless they were already
  // processed).
  auto mapper = [&](std::pair<Value, std::optional<int64_t>> v) {
    return getExpr(v.first, v.second);
  };
  SmallVector<AffineExpr> dimReplacements = llvm::to_vector(
      llvm::map_range(ArrayRef(operands).take_front(map.getNumDims()), mapper));
  SmallVector<AffineExpr> symReplacements = llvm::to_vector(
      llvm::map_range(ArrayRef(operands).drop_front(map.getNumDims()), mapper));
  addBound(
      presburger::BoundType::EQ, pos,
      map.getResult(0).replaceDimsAndSymbols(dimReplacements, symReplacements));

  return pos;
}

int64_t ValueBoundsConstraintSet::insert(const Variable &var, bool isSymbol) {
  return insert(var.map, var.mapOperands, isSymbol);
}

int64_t ValueBoundsConstraintSet::getPos(Value value,
                                         std::optional<int64_t> dim) const {
#ifndef NDEBUG
  assertValidValueDim(value, dim);
  assert((isa<OpResult>(value) ||
          cast<BlockArgument>(value).getOwner()->isEntryBlock()) &&
         "unstructured control flow is not supported");
#endif // NDEBUG
  LLVM_DEBUG(llvm::dbgs() << "Getting pos for: " << value
                          << " (dim: " << dim.value_or(kIndexValue)
                          << ", owner: " << getOwnerOfValue(value)->getName()
                          << ")\n");
  auto it =
      valueDimToPosition.find(std::make_pair(value, dim.value_or(kIndexValue)));
  assert(it != valueDimToPosition.end() && "expected mapped entry");
  return it->second;
}

AffineExpr ValueBoundsConstraintSet::getPosExpr(int64_t pos) {
  assert(pos >= 0 && pos < cstr.getNumDimAndSymbolVars() && "invalid position");
  return pos < cstr.getNumDimVars()
             ? builder.getAffineDimExpr(pos)
             : builder.getAffineSymbolExpr(pos - cstr.getNumDimVars());
}

bool ValueBoundsConstraintSet::isMapped(Value value,
                                        std::optional<int64_t> dim) const {
  auto it =
      valueDimToPosition.find(std::make_pair(value, dim.value_or(kIndexValue)));
  return it != valueDimToPosition.end();
}

void ValueBoundsConstraintSet::processWorklist() {
  LLVM_DEBUG(llvm::dbgs() << "Processing value bounds worklist...\n");
  while (!worklist.empty()) {
    int64_t pos = worklist.front();
    worklist.pop();
    assert(positionToValueDim[pos].has_value() &&
           "did not expect std::nullopt on worklist");
    ValueDim valueDim = *positionToValueDim[pos];
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
    auto maybeDim = dim == kIndexValue ? std::nullopt : std::make_optional(dim);
    if (stopCondition(value, maybeDim, *this)) {
      LLVM_DEBUG(llvm::dbgs() << "Stop condition met for: " << value
                              << " (dim: " << maybeDim << ")\n");
      continue;
    }

    // Query `ValueBoundsOpInterface` for constraints. New items may be added to
    // the worklist.
    auto valueBoundsOp =
        dyn_cast<ValueBoundsOpInterface>(getOwnerOfValue(value));
    LLVM_DEBUG(llvm::dbgs()
               << "Query value bounds for: " << value
               << " (owner: " << getOwnerOfValue(value)->getName() << ")\n");
    if (valueBoundsOp) {
      if (dim == kIndexValue) {
        valueBoundsOp.populateBoundsForIndexValue(value, *this);
      } else {
        valueBoundsOp.populateBoundsForShapedValueDim(value, dim, *this);
      }
      continue;
    }
    LLVM_DEBUG(llvm::dbgs() << "--> ValueBoundsOpInterface not implemented\n");

    // If the op does not implement `ValueBoundsOpInterface`, check if it
    // implements the `DestinationStyleOpInterface`. OpResults of such ops are
    // tied to OpOperands. Tied values have the same shape.
    auto dstOp = value.getDefiningOp<DestinationStyleOpInterface>();
    if (!dstOp || dim == kIndexValue)
      continue;
    Value tiedOperand = dstOp.getTiedOpOperand(cast<OpResult>(value))->get();
    bound(value)[dim] == getExpr(tiedOperand, dim);
  }
}

void ValueBoundsConstraintSet::projectOut(int64_t pos) {
  assert(pos >= 0 && pos < static_cast<int64_t>(positionToValueDim.size()) &&
         "invalid position");
  cstr.projectOut(pos);
  if (positionToValueDim[pos].has_value()) {
    bool erased = valueDimToPosition.erase(*positionToValueDim[pos]);
    (void)erased;
    assert(erased && "inconsistent reverse mapping");
  }
  positionToValueDim.erase(positionToValueDim.begin() + pos);
  // Update reverse mapping.
  for (int64_t i = pos, e = positionToValueDim.size(); i < e; ++i)
    if (positionToValueDim[i].has_value())
      valueDimToPosition[*positionToValueDim[i]] = i;
}

void ValueBoundsConstraintSet::projectOut(
    function_ref<bool(ValueDim)> condition) {
  int64_t nextPos = 0;
  while (nextPos < static_cast<int64_t>(positionToValueDim.size())) {
    if (positionToValueDim[nextPos].has_value() &&
        condition(*positionToValueDim[nextPos])) {
      projectOut(nextPos);
      // The column was projected out so another column is now at that position.
      // Do not increase the counter.
    } else {
      ++nextPos;
    }
  }
}

void ValueBoundsConstraintSet::projectOutAnonymous(
    std::optional<int64_t> except) {
  int64_t nextPos = 0;
  while (nextPos < static_cast<int64_t>(positionToValueDim.size())) {
    if (positionToValueDim[nextPos].has_value() || except == nextPos) {
      ++nextPos;
    } else {
      projectOut(nextPos);
      // The column was projected out so another column is now at that position.
      // Do not increase the counter.
    }
  }
}

LogicalResult ValueBoundsConstraintSet::computeBound(
    AffineMap &resultMap, ValueDimList &mapOperands, presburger::BoundType type,
    const Variable &var, StopConditionFn stopCondition, bool closedUB) {
  MLIRContext *ctx = var.getContext();
  int64_t ubAdjustment = closedUB ? 0 : 1;
  Builder b(ctx);
  mapOperands.clear();

  // Process the backward slice of `value` (i.e., reverse use-def chain) until
  // `stopCondition` is met.
  ValueBoundsConstraintSet cstr(ctx, stopCondition);
  int64_t pos = cstr.insert(var, /*isSymbol=*/false);
  assert(pos == 0 && "expected first column");
  cstr.processWorklist();

  // Project out all variables (apart from `valueDim`) that do not match the
  // stop condition.
  cstr.projectOut([&](ValueDim p) {
    auto maybeDim =
        p.second == kIndexValue ? std::nullopt : std::make_optional(p.second);
    return !stopCondition(p.first, maybeDim, cstr);
  });
  cstr.projectOutAnonymous(/*except=*/pos);

  // Compute lower and upper bounds for `valueDim`.
  SmallVector<AffineMap> lb(1), ub(1);
  cstr.cstr.getSliceBounds(pos, 1, ctx, &lb, &ub,
                           /*closedUB=*/true);

  // Note: There are TODOs in the implementation of `getSliceBounds`. In such a
  // case, no lower/upper bound can be computed at the moment.
  // EQ, UB bounds: upper bound is needed.
  if ((type != BoundType::LB) &&
      (ub.empty() || !ub[0] || ub[0].getNumResults() == 0))
    return failure();
  // EQ, LB bounds: lower bound is needed.
  if ((type != BoundType::UB) &&
      (lb.empty() || !lb[0] || lb[0].getNumResults() == 0))
    return failure();

  // TODO: Generate an affine map with multiple results.
  if (type != BoundType::LB)
    assert(ub.size() == 1 && ub[0].getNumResults() == 1 &&
           "multiple bounds not supported");
  if (type != BoundType::UB)
    assert(lb.size() == 1 && lb[0].getNumResults() == 1 &&
           "multiple bounds not supported");

  // EQ bound: lower and upper bound must match.
  if (type == BoundType::EQ && ub[0] != lb[0])
    return failure();

  AffineMap bound;
  if (type == BoundType::EQ || type == BoundType::LB) {
    bound = lb[0];
  } else {
    // Computed UB is a closed bound.
    bound = AffineMap::get(ub[0].getNumDims(), ub[0].getNumSymbols(),
                           ub[0].getResult(0) + ubAdjustment);
  }

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
      if (bound.isFunctionOfDim(i))
        used = true;
    } else {
      if (bound.isFunctionOfSymbol(i - cstr.cstr.getNumDimVars()))
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

    assert(cstr.positionToValueDim[i].has_value() &&
           "cannot build affine map in terms of anonymous column");
    ValueBoundsConstraintSet::ValueDim valueDim = *cstr.positionToValueDim[i];
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

  resultMap = bound.replaceDimsAndSymbols(replacementDims, replacementSymbols,
                                          numDims, numSymbols);
  return success();
}

LogicalResult ValueBoundsConstraintSet::computeDependentBound(
    AffineMap &resultMap, ValueDimList &mapOperands, presburger::BoundType type,
    const Variable &var, ValueDimList dependencies, bool closedUB) {
  return computeBound(
      resultMap, mapOperands, type, var,
      [&](Value v, std::optional<int64_t> d, ValueBoundsConstraintSet &cstr) {
        return llvm::is_contained(dependencies, std::make_pair(v, d));
      },
      closedUB);
}

LogicalResult ValueBoundsConstraintSet::computeIndependentBound(
    AffineMap &resultMap, ValueDimList &mapOperands, presburger::BoundType type,
    const Variable &var, ValueRange independencies, bool closedUB) {
  // Return "true" if the given value is independent of all values in
  // `independencies`. I.e., neither the value itself nor any value in the
  // backward slice (reverse use-def chain) is contained in `independencies`.
  auto isIndependent = [&](Value v) {
    SmallVector<Value> worklist;
    DenseSet<Value> visited;
    worklist.push_back(v);
    while (!worklist.empty()) {
      Value next = worklist.pop_back_val();
      if (!visited.insert(next).second)
        continue;
      if (llvm::is_contained(independencies, next))
        return false;
      // TODO: DominanceInfo could be used to stop the traversal early.
      Operation *op = next.getDefiningOp();
      if (!op)
        continue;
      worklist.append(op->getOperands().begin(), op->getOperands().end());
    }
    return true;
  };

  // Reify bounds in terms of any independent values.
  return computeBound(
      resultMap, mapOperands, type, var,
      [&](Value v, std::optional<int64_t> d, ValueBoundsConstraintSet &cstr) {
        return isIndependent(v);
      },
      closedUB);
}

FailureOr<int64_t> ValueBoundsConstraintSet::computeConstantBound(
    presburger::BoundType type, const Variable &var,
    StopConditionFn stopCondition, bool closedUB) {
  // Default stop condition if none was specified: Keep adding constraints until
  // a bound could be computed.
  int64_t pos = 0;
  auto defaultStopCondition = [&](Value v, std::optional<int64_t> dim,
                                  ValueBoundsConstraintSet &cstr) {
    return cstr.cstr.getConstantBound64(type, pos).has_value();
  };

  ValueBoundsConstraintSet cstr(
      var.getContext(), stopCondition ? stopCondition : defaultStopCondition);
  pos = cstr.populateConstraints(var.map, var.mapOperands);
  assert(pos == 0 && "expected `map` is the first column");

  // Compute constant bound for `valueDim`.
  int64_t ubAdjustment = closedUB ? 0 : 1;
  if (auto bound = cstr.cstr.getConstantBound64(type, pos))
    return type == BoundType::UB ? *bound + ubAdjustment : *bound;
  return failure();
}

void ValueBoundsConstraintSet::populateConstraints(Value value,
                                                   std::optional<int64_t> dim) {
#ifndef NDEBUG
  assertValidValueDim(value, dim);
#endif // NDEBUG

  // `getExpr` pushes the value/dim onto the worklist (unless it was already
  // analyzed).
  (void)getExpr(value, dim);
  // Process all values/dims on the worklist. This may traverse and analyze
  // additional IR, depending the current stop function.
  processWorklist();
}

int64_t ValueBoundsConstraintSet::populateConstraints(AffineMap map,
                                                      ValueDimList operands) {
  int64_t pos = insert(map, operands, /*isSymbol=*/false);
  // Process the backward slice of `operands` (i.e., reverse use-def chain)
  // until `stopCondition` is met.
  processWorklist();
  return pos;
}

FailureOr<int64_t>
ValueBoundsConstraintSet::computeConstantDelta(Value value1, Value value2,
                                               std::optional<int64_t> dim1,
                                               std::optional<int64_t> dim2) {
#ifndef NDEBUG
  assertValidValueDim(value1, dim1);
  assertValidValueDim(value2, dim2);
#endif // NDEBUG

  Builder b(value1.getContext());
  AffineMap map = AffineMap::get(/*dimCount=*/2, /*symbolCount=*/0,
                                 b.getAffineDimExpr(0) - b.getAffineDimExpr(1));
  return computeConstantBound(presburger::BoundType::EQ,
                              Variable(map, {{value1, dim1}, {value2, dim2}}));
}

bool ValueBoundsConstraintSet::comparePos(int64_t lhsPos,
                                          ComparisonOperator cmp,
                                          int64_t rhsPos) {
  // This function returns "true" if "lhs CMP rhs" is proven to hold.
  //
  // Example for ComparisonOperator::LE and index-typed values: We would like to
  // prove that lhs <= rhs. Proof by contradiction: add the inverse
  // relation (lhs > rhs) to the constraint set and check if the resulting
  // constraint set is "empty" (i.e. has no solution). In that case,
  // lhs > rhs must be incorrect and we can deduce that lhs <= rhs holds.

  // We cannot prove anything if the constraint set is already empty.
  if (cstr.isEmpty()) {
    LLVM_DEBUG(
        llvm::dbgs()
        << "cannot compare value/dims: constraint system is already empty");
    return false;
  }

  // EQ can be expressed as LE and GE.
  if (cmp == EQ)
    return comparePos(lhsPos, ComparisonOperator::LE, rhsPos) &&
           comparePos(lhsPos, ComparisonOperator::GE, rhsPos);

  // Construct inequality.
  SmallVector<int64_t> eq(cstr.getNumCols(), 0);
  if (cmp == LT || cmp == LE) {
    ++eq[lhsPos];
    --eq[rhsPos];
  } else if (cmp == GT || cmp == GE) {
    --eq[lhsPos];
    ++eq[rhsPos];
  } else {
    llvm_unreachable("unsupported comparison operator");
  }
  if (cmp == LE || cmp == GE)
    eq[cstr.getNumCols() - 1] -= 1;

  // Add inequality to the constraint set and check if it made the constraint
  // set empty.
  int64_t ineqPos = cstr.getNumInequalities();
  cstr.addInequality(eq);
  bool isEmpty = cstr.isEmpty();
  cstr.removeInequality(ineqPos);
  return isEmpty;
}

bool ValueBoundsConstraintSet::populateAndCompare(const Variable &lhs,
                                                  ComparisonOperator cmp,
                                                  const Variable &rhs) {
  int64_t lhsPos = populateConstraints(lhs.map, lhs.mapOperands);
  int64_t rhsPos = populateConstraints(rhs.map, rhs.mapOperands);
  return comparePos(lhsPos, cmp, rhsPos);
}

bool ValueBoundsConstraintSet::compare(const Variable &lhs,
                                       ComparisonOperator cmp,
                                       const Variable &rhs) {
  int64_t lhsPos = -1, rhsPos = -1;
  auto stopCondition = [&](Value v, std::optional<int64_t> dim,
                           ValueBoundsConstraintSet &cstr) {
    // Keep processing as long as lhs/rhs were not processed.
    if (size_t(lhsPos) >= cstr.positionToValueDim.size() ||
        size_t(rhsPos) >= cstr.positionToValueDim.size())
      return false;
    // Keep processing as long as the relation cannot be proven.
    return cstr.comparePos(lhsPos, cmp, rhsPos);
  };
  ValueBoundsConstraintSet cstr(lhs.getContext(), stopCondition);
  lhsPos = cstr.populateConstraints(lhs.map, lhs.mapOperands);
  rhsPos = cstr.populateConstraints(rhs.map, rhs.mapOperands);
  return cstr.comparePos(lhsPos, cmp, rhsPos);
}

FailureOr<bool> ValueBoundsConstraintSet::areEqual(const Variable &var1,
                                                   const Variable &var2) {
  if (ValueBoundsConstraintSet::compare(var1, ComparisonOperator::EQ, var2))
    return true;
  if (ValueBoundsConstraintSet::compare(var1, ComparisonOperator::LT, var2) ||
      ValueBoundsConstraintSet::compare(var1, ComparisonOperator::GT, var2))
    return false;
  return failure();
}

FailureOr<bool>
ValueBoundsConstraintSet::areOverlappingSlices(MLIRContext *ctx,
                                               HyperrectangularSlice slice1,
                                               HyperrectangularSlice slice2) {
  assert(slice1.getMixedOffsets().size() == slice2.getMixedOffsets().size() &&
         "expected slices of same rank");
  assert(slice1.getMixedSizes().size() == slice2.getMixedSizes().size() &&
         "expected slices of same rank");
  assert(slice1.getMixedStrides().size() == slice2.getMixedStrides().size() &&
         "expected slices of same rank");

  Builder b(ctx);
  bool foundUnknownBound = false;
  for (int64_t i = 0, e = slice1.getMixedOffsets().size(); i < e; ++i) {
    AffineMap map =
        AffineMap::get(/*dimCount=*/0, /*symbolCount=*/4,
                       b.getAffineSymbolExpr(0) +
                           b.getAffineSymbolExpr(1) * b.getAffineSymbolExpr(2) -
                           b.getAffineSymbolExpr(3));
    {
      // Case 1: Slices are guaranteed to be non-overlapping if
      // offset1 + size1 * stride1 <= offset2 (for at least one dimension).
      SmallVector<OpFoldResult> ofrOperands;
      ofrOperands.push_back(slice1.getMixedOffsets()[i]);
      ofrOperands.push_back(slice1.getMixedSizes()[i]);
      ofrOperands.push_back(slice1.getMixedStrides()[i]);
      ofrOperands.push_back(slice2.getMixedOffsets()[i]);
      SmallVector<Value> valueOperands;
      AffineMap foldedMap =
          foldAttributesIntoMap(b, map, ofrOperands, valueOperands);
      FailureOr<int64_t> constBound = computeConstantBound(
          presburger::BoundType::EQ, Variable(foldedMap, valueOperands));
      foundUnknownBound |= failed(constBound);
      if (succeeded(constBound) && *constBound <= 0)
        return false;
    }
    {
      // Case 2: Slices are guaranteed to be non-overlapping if
      // offset2 + size2 * stride2 <= offset1 (for at least one dimension).
      SmallVector<OpFoldResult> ofrOperands;
      ofrOperands.push_back(slice2.getMixedOffsets()[i]);
      ofrOperands.push_back(slice2.getMixedSizes()[i]);
      ofrOperands.push_back(slice2.getMixedStrides()[i]);
      ofrOperands.push_back(slice1.getMixedOffsets()[i]);
      SmallVector<Value> valueOperands;
      AffineMap foldedMap =
          foldAttributesIntoMap(b, map, ofrOperands, valueOperands);
      FailureOr<int64_t> constBound = computeConstantBound(
          presburger::BoundType::EQ, Variable(foldedMap, valueOperands));
      foundUnknownBound |= failed(constBound);
      if (succeeded(constBound) && *constBound <= 0)
        return false;
    }
  }

  // If at least one bound could not be computed, we cannot be certain that the
  // slices are really overlapping.
  if (foundUnknownBound)
    return failure();

  // All bounds could be computed and none of the above cases applied.
  // Therefore, the slices are guaranteed to overlap.
  return true;
}

FailureOr<bool>
ValueBoundsConstraintSet::areEquivalentSlices(MLIRContext *ctx,
                                              HyperrectangularSlice slice1,
                                              HyperrectangularSlice slice2) {
  assert(slice1.getMixedOffsets().size() == slice2.getMixedOffsets().size() &&
         "expected slices of same rank");
  assert(slice1.getMixedSizes().size() == slice2.getMixedSizes().size() &&
         "expected slices of same rank");
  assert(slice1.getMixedStrides().size() == slice2.getMixedStrides().size() &&
         "expected slices of same rank");

  // The two slices are equivalent if all of their offsets, sizes and strides
  // are equal. If equality cannot be determined for at least one of those
  // values, equivalence cannot be determined and this function returns
  // "failure".
  for (auto [offset1, offset2] :
       llvm::zip_equal(slice1.getMixedOffsets(), slice2.getMixedOffsets())) {
    FailureOr<bool> equal = areEqual(offset1, offset2);
    if (failed(equal))
      return failure();
    if (!equal.value())
      return false;
  }
  for (auto [size1, size2] :
       llvm::zip_equal(slice1.getMixedSizes(), slice2.getMixedSizes())) {
    FailureOr<bool> equal = areEqual(size1, size2);
    if (failed(equal))
      return failure();
    if (!equal.value())
      return false;
  }
  for (auto [stride1, stride2] :
       llvm::zip_equal(slice1.getMixedStrides(), slice2.getMixedStrides())) {
    FailureOr<bool> equal = areEqual(stride1, stride2);
    if (failed(equal))
      return failure();
    if (!equal.value())
      return false;
  }
  return true;
}

void ValueBoundsConstraintSet::dump() const {
  llvm::errs() << "==========\nColumns:\n";
  llvm::errs() << "(column\tdim\tvalue)\n";
  for (auto [index, valueDim] : llvm::enumerate(positionToValueDim)) {
    llvm::errs() << " " << index << "\t";
    if (valueDim) {
      if (valueDim->second == kIndexValue) {
        llvm::errs() << "n/a\t";
      } else {
        llvm::errs() << valueDim->second << "\t";
      }
      llvm::errs() << getOwnerOfValue(valueDim->first)->getName() << " ";
      if (OpResult result = dyn_cast<OpResult>(valueDim->first)) {
        llvm::errs() << "(result " << result.getResultNumber() << ")";
      } else {
        llvm::errs() << "(bbarg "
                     << cast<BlockArgument>(valueDim->first).getArgNumber()
                     << ")";
      }
      llvm::errs() << "\n";
    } else {
      llvm::errs() << "n/a\tn/a\n";
    }
  }
  llvm::errs() << "\nConstraint set:\n";
  cstr.dump();
  llvm::errs() << "==========\n";
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
