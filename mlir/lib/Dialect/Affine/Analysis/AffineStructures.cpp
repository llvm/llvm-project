//===- AffineStructures.cpp - MLIR Affine Structures Class-----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Structures for affine/polyhedral analysis of affine dialect ops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Analysis/Presburger/LinearTransform.h"
#include "mlir/Analysis/Presburger/Simplex.h"
#include "mlir/Analysis/Presburger/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/MathExtras.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>

#define DEBUG_TYPE "affine-structures"

using namespace mlir;
using namespace affine;
using namespace presburger;


void FlatAffineValueConstraints::addInductionVarOrTerminalSymbol(Value val) {
  if (containsVar(val))
    return;

  // Caller is expected to fully compose map/operands if necessary.
  assert((isTopLevelValue(val) || isAffineInductionVar(val)) &&
         "non-terminal symbol / loop IV expected");
  // Outer loop IVs could be used in forOp's bounds.
  if (auto loop = getForInductionVarOwner(val)) {
    appendDimVar(val);
    if (failed(this->addAffineForOpDomain(loop)))
      LLVM_DEBUG(
          loop.emitWarning("failed to add domain info to constraint system"));
    return;
  }
  if (auto parallel = getAffineParallelInductionVarOwner(val)) {
    appendDimVar(parallel.getIVs());
    if (failed(this->addAffineParallelOpDomain(parallel)))
      LLVM_DEBUG(parallel.emitWarning(
          "failed to add domain info to constraint system"));
    return;
  }

  // Add top level symbol.
  appendSymbolVar(val);
  // Check if the symbol is a constant.
  if (std::optional<int64_t> constOp = getConstantIntValue(val))
    addBound(BoundType::EQ, val, constOp.value());
}

LogicalResult
FlatAffineValueConstraints::addAffineForOpDomain(AffineForOp forOp) {
  unsigned pos;
  // Pre-condition for this method.
  if (!findVar(forOp.getInductionVar(), &pos)) {
    assert(false && "Value not found");
    return failure();
  }

  int64_t step = forOp.getStep();
  if (step != 1) {
    if (!forOp.hasConstantLowerBound())
      LLVM_DEBUG(forOp.emitWarning("domain conservatively approximated"));
    else {
      // Add constraints for the stride.
      // (iv - lb) % step = 0 can be written as:
      // (iv - lb) - step * q = 0 where q = (iv - lb) / step.
      // Add local variable 'q' and add the above equality.
      // The first constraint is q = (iv - lb) floordiv step
      SmallVector<int64_t, 8> dividend(getNumCols(), 0);
      int64_t lb = forOp.getConstantLowerBound();
      dividend[pos] = 1;
      dividend.back() -= lb;
      addLocalFloorDiv(dividend, step);
      // Second constraint: (iv - lb) - step * q = 0.
      SmallVector<int64_t, 8> eq(getNumCols(), 0);
      eq[pos] = 1;
      eq.back() -= lb;
      // For the local var just added above.
      eq[getNumCols() - 2] = -step;
      addEquality(eq);
    }
  }

  if (forOp.hasConstantLowerBound()) {
    addBound(BoundType::LB, pos, forOp.getConstantLowerBound());
  } else {
    // Non-constant lower bound case.
    if (failed(addBound(BoundType::LB, pos, forOp.getLowerBoundMap(),
                        forOp.getLowerBoundOperands())))
      return failure();
  }

  if (forOp.hasConstantUpperBound()) {
    addBound(BoundType::UB, pos, forOp.getConstantUpperBound() - 1);
    return success();
  }
  // Non-constant upper bound case.
  return addBound(BoundType::UB, pos, forOp.getUpperBoundMap(),
                  forOp.getUpperBoundOperands());
}

LogicalResult FlatAffineValueConstraints::addAffineParallelOpDomain(
    AffineParallelOp parallelOp) {
  size_t ivPos = 0;
  for (Value iv : parallelOp.getIVs()) {
    unsigned pos;
    if (!findVar(iv, &pos)) {
      assert(false && "variable expected for the IV value");
      return failure();
    }

    AffineMap lowerBound = parallelOp.getLowerBoundMap(ivPos);
    if (lowerBound.isConstant())
      addBound(BoundType::LB, pos, lowerBound.getSingleConstantResult());
    else if (failed(addBound(BoundType::LB, pos, lowerBound,
                             parallelOp.getLowerBoundsOperands())))
      return failure();

    auto upperBound = parallelOp.getUpperBoundMap(ivPos);
    if (upperBound.isConstant())
      addBound(BoundType::UB, pos, upperBound.getSingleConstantResult() - 1);
    else if (failed(addBound(BoundType::UB, pos, upperBound,
                             parallelOp.getUpperBoundsOperands())))
      return failure();
    ++ivPos;
  }
  return success();
}

LogicalResult
FlatAffineValueConstraints::addDomainFromSliceMaps(ArrayRef<AffineMap> lbMaps,
                                                   ArrayRef<AffineMap> ubMaps,
                                                   ArrayRef<Value> operands) {
  assert(lbMaps.size() == ubMaps.size());
  assert(lbMaps.size() <= getNumDimVars());

  for (unsigned i = 0, e = lbMaps.size(); i < e; ++i) {
    AffineMap lbMap = lbMaps[i];
    AffineMap ubMap = ubMaps[i];
    assert(!lbMap || lbMap.getNumInputs() == operands.size());
    assert(!ubMap || ubMap.getNumInputs() == operands.size());

    // Check if this slice is just an equality along this dimension. If so,
    // retrieve the existing loop it equates to and add it to the system.
    if (lbMap && ubMap && lbMap.getNumResults() == 1 &&
        ubMap.getNumResults() == 1 &&
        lbMap.getResult(0) + 1 == ubMap.getResult(0) &&
        // The condition above will be true for maps describing a single
        // iteration (e.g., lbMap.getResult(0) = 0, ubMap.getResult(0) = 1).
        // Make sure we skip those cases by checking that the lb result is not
        // just a constant.
        !lbMap.getResult(0).isa<AffineConstantExpr>()) {
      // Limited support: we expect the lb result to be just a loop dimension.
      // Not supported otherwise for now.
      AffineDimExpr result = lbMap.getResult(0).dyn_cast<AffineDimExpr>();
      if (!result)
        return failure();

      AffineForOp loop =
          getForInductionVarOwner(operands[result.getPosition()]);
      if (!loop)
        return failure();

      if (failed(addAffineForOpDomain(loop)))
        return failure();
      continue;
    }

    // This slice refers to a loop that doesn't exist in the IR yet. Add its
    // bounds to the system assuming its dimension variable position is the
    // same as the position of the loop in the loop nest.
    if (lbMap && failed(addBound(BoundType::LB, i, lbMap, operands)))
      return failure();
    if (ubMap && failed(addBound(BoundType::UB, i, ubMap, operands)))
      return failure();
  }
  return success();
}

void FlatAffineValueConstraints::addAffineIfOpDomain(AffineIfOp ifOp) {
  IntegerSet set = ifOp.getIntegerSet();
  // Canonicalize set and operands to ensure unique values for
  // FlatAffineValueConstraints below and for early simplification.
  SmallVector<Value> operands(ifOp.getOperands());
  canonicalizeSetAndOperands(&set, &operands);

  // Create the base constraints from the integer set attached to ifOp.
  FlatAffineValueConstraints cst(set, operands);

  // Merge the constraints from ifOp to the current domain. We need first merge
  // and align the IDs from both constraints, and then append the constraints
  // from the ifOp into the current one.
  mergeAndAlignVarsWithOther(0, &cst);
  append(cst);
}

LogicalResult FlatAffineValueConstraints::addBound(BoundType type, unsigned pos,
                                                   AffineMap boundMap,
                                                   ValueRange boundOperands) {
  // Fully compose map and operands; canonicalize and simplify so that we
  // transitively get to terminal symbols or loop IVs.
  auto map = boundMap;
  SmallVector<Value, 4> operands(boundOperands.begin(), boundOperands.end());
  fullyComposeAffineMapAndOperands(&map, &operands);
  map = simplifyAffineMap(map);
  canonicalizeMapAndOperands(&map, &operands);
  for (auto operand : operands)
    addInductionVarOrTerminalSymbol(operand);
  return addBound(type, pos, computeAlignedMap(map, operands));
}

// Adds slice lower bounds represented by lower bounds in 'lbMaps' and upper
// bounds in 'ubMaps' to each value in `values' that appears in the constraint
// system. Note that both lower/upper bounds share the same operand list
// 'operands'.
// This function assumes 'values.size' == 'lbMaps.size' == 'ubMaps.size', and
// skips any null AffineMaps in 'lbMaps' or 'ubMaps'.
// Note that both lower/upper bounds use operands from 'operands'.
// Returns failure for unimplemented cases such as semi-affine expressions or
// expressions with mod/floordiv.
LogicalResult FlatAffineValueConstraints::addSliceBounds(
    ArrayRef<Value> values, ArrayRef<AffineMap> lbMaps,
    ArrayRef<AffineMap> ubMaps, ArrayRef<Value> operands) {
  assert(values.size() == lbMaps.size());
  assert(lbMaps.size() == ubMaps.size());

  for (unsigned i = 0, e = lbMaps.size(); i < e; ++i) {
    unsigned pos;
    if (!findVar(values[i], &pos))
      continue;

    AffineMap lbMap = lbMaps[i];
    AffineMap ubMap = ubMaps[i];
    assert(!lbMap || lbMap.getNumInputs() == operands.size());
    assert(!ubMap || ubMap.getNumInputs() == operands.size());

    // Check if this slice is just an equality along this dimension.
    if (lbMap && ubMap && lbMap.getNumResults() == 1 &&
        ubMap.getNumResults() == 1 &&
        lbMap.getResult(0) + 1 == ubMap.getResult(0)) {
      if (failed(addBound(BoundType::EQ, pos, lbMap, operands)))
        return failure();
      continue;
    }

    // If lower or upper bound maps are null or provide no results, it implies
    // that the source loop was not at all sliced, and the entire loop will be a
    // part of the slice.
    if (lbMap && lbMap.getNumResults() != 0 && ubMap &&
        ubMap.getNumResults() != 0) {
      if (failed(addBound(BoundType::LB, pos, lbMap, operands)))
        return failure();
      if (failed(addBound(BoundType::UB, pos, ubMap, operands)))
        return failure();
    } else {
      auto loop = getForInductionVarOwner(values[i]);
      if (failed(this->addAffineForOpDomain(loop)))
        return failure();
    }
  }
  return success();
}

LogicalResult
FlatAffineValueConstraints::composeMap(const AffineValueMap *vMap) {
  return composeMatchingMap(
      computeAlignedMap(vMap->getAffineMap(), vMap->getOperands()));
}

// Turn a symbol into a dimension.
static void turnSymbolIntoDim(FlatAffineValueConstraints *cst, Value value) {
  unsigned pos;
  if (cst->findVar(value, &pos) && pos >= cst->getNumDimVars() &&
      pos < cst->getNumDimAndSymbolVars()) {
    cst->swapVar(pos, cst->getNumDimVars());
    cst->setDimSymbolSeparation(cst->getNumSymbolVars() - 1);
  }
}

// Changes all symbol variables which are loop IVs to dim variables.
void FlatAffineValueConstraints::convertLoopIVSymbolsToDims() {
  // Gather all symbols which are loop IVs.
  SmallVector<Value, 4> loopIVs;
  for (unsigned i = getNumDimVars(), e = getNumDimAndSymbolVars(); i < e; i++) {
    if (hasValue(i) && getForInductionVarOwner(getValue(i)))
      loopIVs.push_back(getValue(i));
  }
  // Turn each symbol in 'loopIVs' into a dim variable.
  for (auto iv : loopIVs) {
    turnSymbolIntoDim(this, iv);
  }
}

void FlatAffineValueConstraints::getIneqAsAffineValueMap(
    unsigned pos, unsigned ineqPos, AffineValueMap &vmap,
    MLIRContext *context) const {
  unsigned numDims = getNumDimVars();
  unsigned numSyms = getNumSymbolVars();

  assert(pos < numDims && "invalid position");
  assert(ineqPos < getNumInequalities() && "invalid inequality position");

  // Get expressions for local vars.
  SmallVector<AffineExpr, 8> memo(getNumVars(), AffineExpr());
  if (failed(computeLocalVars(memo, context)))
    assert(false &&
           "one or more local exprs do not have an explicit representation");
  auto localExprs = ArrayRef<AffineExpr>(memo).take_back(getNumLocalVars());

  // Compute the AffineExpr lower/upper bound for this inequality.
  SmallVector<int64_t, 8> inequality = getInequality64(ineqPos);
  SmallVector<int64_t, 8> bound;
  bound.reserve(getNumCols() - 1);
  // Everything other than the coefficient at `pos`.
  bound.append(inequality.begin(), inequality.begin() + pos);
  bound.append(inequality.begin() + pos + 1, inequality.end());

  if (inequality[pos] > 0)
    // Lower bound.
    std::transform(bound.begin(), bound.end(), bound.begin(),
                   std::negate<int64_t>());
  else
    // Upper bound (which is exclusive).
    bound.back() += 1;

  // Convert to AffineExpr (tree) form.
  auto boundExpr = getAffineExprFromFlatForm(bound, numDims - 1, numSyms,
                                             localExprs, context);

  // Get the values to bind to this affine expr (all dims and symbols).
  SmallVector<Value, 4> operands;
  getValues(0, pos, &operands);
  SmallVector<Value, 4> trailingOperands;
  getValues(pos + 1, getNumDimAndSymbolVars(), &trailingOperands);
  operands.append(trailingOperands.begin(), trailingOperands.end());
  vmap.reset(AffineMap::get(numDims - 1, numSyms, boundExpr), operands);
}

FlatAffineValueConstraints FlatAffineRelation::getDomainSet() const {
  FlatAffineValueConstraints domain = *this;
  // Convert all range variables to local variables.
  domain.convertToLocal(VarKind::SetDim, getNumDomainDims(),
                        getNumDomainDims() + getNumRangeDims());
  return domain;
}

FlatAffineValueConstraints FlatAffineRelation::getRangeSet() const {
  FlatAffineValueConstraints range = *this;
  // Convert all domain variables to local variables.
  range.convertToLocal(VarKind::SetDim, 0, getNumDomainDims());
  return range;
}

void FlatAffineRelation::compose(const FlatAffineRelation &other) {
  assert(getNumDomainDims() == other.getNumRangeDims() &&
         "Domain of this and range of other do not match");
  assert(std::equal(values.begin(), values.begin() + getNumDomainDims(),
                    other.values.begin() + other.getNumDomainDims()) &&
         "Domain of this and range of other do not match");

  FlatAffineRelation rel = other;

  // Convert `rel` from
  //    [otherDomain] -> [otherRange]
  // to
  //    [otherDomain] -> [otherRange thisRange]
  // and `this` from
  //    [thisDomain] -> [thisRange]
  // to
  //    [otherDomain thisDomain] -> [thisRange].
  unsigned removeDims = rel.getNumRangeDims();
  insertDomainVar(0, rel.getNumDomainDims());
  rel.appendRangeVar(getNumRangeDims());

  // Merge symbol and local variables.
  mergeSymbolVars(rel);
  mergeLocalVars(rel);

  // Convert `rel` from [otherDomain] -> [otherRange thisRange] to
  // [otherDomain] -> [thisRange] by converting first otherRange range vars
  // to local vars.
  rel.convertToLocal(VarKind::SetDim, rel.getNumDomainDims(),
                     rel.getNumDomainDims() + removeDims);
  // Convert `this` from [otherDomain thisDomain] -> [thisRange] to
  // [otherDomain] -> [thisRange] by converting last thisDomain domain vars
  // to local vars.
  convertToLocal(VarKind::SetDim, getNumDomainDims() - removeDims,
                 getNumDomainDims());

  auto thisMaybeValues = getMaybeValues(VarKind::SetDim);
  auto relMaybeValues = rel.getMaybeValues(VarKind::SetDim);

  // Add and match domain of `rel` to domain of `this`.
  for (unsigned i = 0, e = rel.getNumDomainDims(); i < e; ++i)
    if (relMaybeValues[i].has_value())
      setValue(i, *relMaybeValues[i]);
  // Add and match range of `this` to range of `rel`.
  for (unsigned i = 0, e = getNumRangeDims(); i < e; ++i) {
    unsigned rangeIdx = rel.getNumDomainDims() + i;
    if (thisMaybeValues[rangeIdx].has_value())
      rel.setValue(rangeIdx, *thisMaybeValues[rangeIdx]);
  }

  // Append `this` to `rel` and simplify constraints.
  rel.append(*this);
  rel.removeRedundantLocalVars();

  *this = rel;
}

void FlatAffineRelation::inverse() {
  unsigned oldDomain = getNumDomainDims();
  unsigned oldRange = getNumRangeDims();
  // Add new range vars.
  appendRangeVar(oldDomain);
  // Swap new vars with domain.
  for (unsigned i = 0; i < oldDomain; ++i)
    swapVar(i, oldDomain + oldRange + i);
  // Remove the swapped domain.
  removeVarRange(0, oldDomain);
  // Set domain and range as inverse.
  numDomainDims = oldRange;
  numRangeDims = oldDomain;
}

void FlatAffineRelation::insertDomainVar(unsigned pos, unsigned num) {
  assert(pos <= getNumDomainDims() &&
         "Var cannot be inserted at invalid position");
  insertDimVar(pos, num);
  numDomainDims += num;
}

void FlatAffineRelation::insertRangeVar(unsigned pos, unsigned num) {
  assert(pos <= getNumRangeDims() &&
         "Var cannot be inserted at invalid position");
  insertDimVar(getNumDomainDims() + pos, num);
  numRangeDims += num;
}

void FlatAffineRelation::appendDomainVar(unsigned num) {
  insertDimVar(getNumDomainDims(), num);
  numDomainDims += num;
}

void FlatAffineRelation::appendRangeVar(unsigned num) {
  insertDimVar(getNumDimVars(), num);
  numRangeDims += num;
}

void FlatAffineRelation::removeVarRange(VarKind kind, unsigned varStart,
                                        unsigned varLimit) {
  assert(varLimit <= getNumVarKind(kind));
  if (varStart >= varLimit)
    return;

  FlatAffineValueConstraints::removeVarRange(kind, varStart, varLimit);

  // If kind is not SetDim, domain and range don't need to be updated.
  if (kind != VarKind::SetDim)
    return;

  // Compute number of domain and range variables to remove. This is done by
  // intersecting the range of domain/range vars with range of vars to remove.
  unsigned intersectDomainLHS = std::min(varLimit, getNumDomainDims());
  unsigned intersectDomainRHS = varStart;
  unsigned intersectRangeLHS = std::min(varLimit, getNumDimVars());
  unsigned intersectRangeRHS = std::max(varStart, getNumDomainDims());

  if (intersectDomainLHS > intersectDomainRHS)
    numDomainDims -= intersectDomainLHS - intersectDomainRHS;
  if (intersectRangeLHS > intersectRangeRHS)
    numRangeDims -= intersectRangeLHS - intersectRangeRHS;
}

LogicalResult mlir::affine::getRelationFromMap(AffineMap &map,
                                               FlatAffineRelation &rel) {
  // Get flattened affine expressions.
  std::vector<SmallVector<int64_t, 8>> flatExprs;
  FlatAffineValueConstraints localVarCst;
  if (failed(getFlattenedAffineExprs(map, &flatExprs, &localVarCst)))
    return failure();

  unsigned oldDimNum = localVarCst.getNumDimVars();
  unsigned oldCols = localVarCst.getNumCols();
  unsigned numRangeVars = map.getNumResults();
  unsigned numDomainVars = map.getNumDims();

  // Add range as the new expressions.
  localVarCst.appendDimVar(numRangeVars);

  // Add equalities between source and range.
  SmallVector<int64_t, 8> eq(localVarCst.getNumCols());
  for (unsigned i = 0, e = map.getNumResults(); i < e; ++i) {
    // Zero fill.
    std::fill(eq.begin(), eq.end(), 0);
    // Fill equality.
    for (unsigned j = 0, f = oldDimNum; j < f; ++j)
      eq[j] = flatExprs[i][j];
    for (unsigned j = oldDimNum, f = oldCols; j < f; ++j)
      eq[j + numRangeVars] = flatExprs[i][j];
    // Set this dimension to -1 to equate lhs and rhs and add equality.
    eq[numDomainVars + i] = -1;
    localVarCst.addEquality(eq);
  }

  // Create relation and return success.
  rel = FlatAffineRelation(numDomainVars, numRangeVars, localVarCst);
  return success();
}

LogicalResult mlir::affine::getRelationFromMap(const AffineValueMap &map,
                                               FlatAffineRelation &rel) {

  AffineMap affineMap = map.getAffineMap();
  if (failed(getRelationFromMap(affineMap, rel)))
    return failure();

  // Set symbol values for domain dimensions and symbols.
  for (unsigned i = 0, e = rel.getNumDomainDims(); i < e; ++i)
    rel.setValue(i, map.getOperand(i));
  for (unsigned i = rel.getNumDimVars(), e = rel.getNumDimAndSymbolVars();
       i < e; ++i)
    rel.setValue(i, map.getOperand(i - rel.getNumRangeDims()));

  return success();
}
