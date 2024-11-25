//===- FlatLinearValueConstraints.cpp - Linear Constraint -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis//FlatLinearValueConstraints.h"

#include "mlir/Analysis/Presburger/LinearTransform.h"
#include "mlir/Analysis/Presburger/PresburgerSpace.h"
#include "mlir/Analysis/Presburger/Simplex.h"
#include "mlir/Analysis/Presburger/Utils.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>

#define DEBUG_TYPE "flat-value-constraints"

using namespace mlir;
using namespace presburger;

//===----------------------------------------------------------------------===//
// AffineExprFlattener
//===----------------------------------------------------------------------===//

namespace {

// See comments for SimpleAffineExprFlattener.
// An AffineExprFlattenerWithLocalVars extends a SimpleAffineExprFlattener by
// recording constraint information associated with mod's, floordiv's, and
// ceildiv's in FlatLinearConstraints 'localVarCst'.
struct AffineExprFlattener : public SimpleAffineExprFlattener {
  using SimpleAffineExprFlattener::SimpleAffineExprFlattener;

  // Constraints connecting newly introduced local variables (for mod's and
  // div's) to existing (dimensional and symbolic) ones. These are always
  // inequalities.
  IntegerPolyhedron localVarCst;

  AffineExprFlattener(unsigned nDims, unsigned nSymbols)
      : SimpleAffineExprFlattener(nDims, nSymbols),
        localVarCst(PresburgerSpace::getSetSpace(nDims, nSymbols)) {};

private:
  // Add a local variable (needed to flatten a mod, floordiv, ceildiv expr).
  // The local variable added is always a floordiv of a pure add/mul affine
  // function of other variables, coefficients of which are specified in
  // `dividend' and with respect to the positive constant `divisor'. localExpr
  // is the simplified tree expression (AffineExpr) corresponding to the
  // quantifier.
  void addLocalFloorDivId(ArrayRef<int64_t> dividend, int64_t divisor,
                          AffineExpr localExpr) override {
    SimpleAffineExprFlattener::addLocalFloorDivId(dividend, divisor, localExpr);
    // Update localVarCst.
    localVarCst.addLocalFloorDiv(dividend, divisor);
  }

  LogicalResult addLocalIdSemiAffine(ArrayRef<int64_t> lhs,
                                     ArrayRef<int64_t> rhs,
                                     AffineExpr localExpr) override {
    // AffineExprFlattener does not support semi-affine expressions.
    return failure();
  }
};

// A SemiAffineExprFlattener is an AffineExprFlattenerWithLocalVars that adds
// conservative bounds for semi-affine expressions (given assumptions hold). If
// the assumptions required to add the semi-affine bounds are found not to hold
// the final constraints set will be empty/inconsistent. If the assumptions are
// never contradicted the final bounds still only will be correct if the
// assumptions hold.
struct SemiAffineExprFlattener : public AffineExprFlattener {
  using AffineExprFlattener::AffineExprFlattener;

  LogicalResult addLocalIdSemiAffine(ArrayRef<int64_t> lhs,
                                     ArrayRef<int64_t> rhs,
                                     AffineExpr localExpr) override {
    auto result =
        SimpleAffineExprFlattener::addLocalIdSemiAffine(lhs, rhs, localExpr);
    assert(succeeded(result) &&
           "unexpected failure in SimpleAffineExprFlattener");
    (void)result;

    if (localExpr.getKind() == AffineExprKind::Mod) {
      // Given two numbers a and b, division is defined as:
      //
      // a = bq + r
      // 0 <= r < |b| (where |x| is the absolute value of x)
      //
      // q = a floordiv b
      // r = a mod b

      // Add a new local variable (r) to represent the mod.
      unsigned rPos = localVarCst.appendVar(VarKind::Local);

      // r >= 0 (Can ALWAYS be added)
      localVarCst.addBound(BoundType::LB, rPos, 0);

      // r < b (Can be added if b > 0, which we assume here)
      ArrayRef<int64_t> b = rhs;
      SmallVector<int64_t> bSubR(b);
      bSubR.insert(bSubR.begin() + rPos, -1);
      // Note: bSubR = b - r
      // So this adds the bound b - r >= 1 (equivalent to r < b)
      localVarCst.addBound(BoundType::LB, bSubR, 1);

      // Note: The assumption of b > 0 is based on the affine expression docs,
      // which state "RHS of mod is always a constant or a symbolic expression
      // with a positive value." (see AffineExprKind in AffineExpr.h). If this
      // assumption does not hold constraints (added above) are a contradiction.

      return success();
    }

    // TODO: Support other semi-affine expressions.
    return failure();
  }
};

} // namespace

// Flattens the expressions in map. Returns failure if 'expr' was unable to be
// flattened. For example two specific cases:
// 1. an unhandled semi-affine expressions is found.
// 2. has poison expression (i.e., division by zero).
static LogicalResult
getFlattenedAffineExprs(ArrayRef<AffineExpr> exprs, unsigned numDims,
                        unsigned numSymbols,
                        std::vector<SmallVector<int64_t, 8>> *flattenedExprs,
                        FlatLinearConstraints *localVarCst,
                        bool addConservativeSemiAffineBounds = false) {
  if (exprs.empty()) {
    if (localVarCst)
      *localVarCst = FlatLinearConstraints(numDims, numSymbols);
    return success();
  }

  auto flattenExprs = [&](AffineExprFlattener &flattener) -> LogicalResult {
    // Use the same flattener to simplify each expression successively. This way
    // local variables / expressions are shared.
    for (auto expr : exprs) {
      auto flattenResult = flattener.walkPostOrder(expr);
      if (failed(flattenResult))
        return failure();
    }

    assert(flattener.operandExprStack.size() == exprs.size());
    flattenedExprs->clear();
    flattenedExprs->assign(flattener.operandExprStack.begin(),
                           flattener.operandExprStack.end());

    if (localVarCst)
      localVarCst->clearAndCopyFrom(flattener.localVarCst);

    return success();
  };

  if (addConservativeSemiAffineBounds) {
    SemiAffineExprFlattener flattener(numDims, numSymbols);
    return flattenExprs(flattener);
  }

  AffineExprFlattener flattener(numDims, numSymbols);
  return flattenExprs(flattener);
}

// Flattens 'expr' into 'flattenedExpr'. Returns failure if 'expr' was unable to
// be flattened (an unhandled semi-affine was found).
LogicalResult mlir::getFlattenedAffineExpr(
    AffineExpr expr, unsigned numDims, unsigned numSymbols,
    SmallVectorImpl<int64_t> *flattenedExpr, FlatLinearConstraints *localVarCst,
    bool addConservativeSemiAffineBounds) {
  std::vector<SmallVector<int64_t, 8>> flattenedExprs;
  LogicalResult ret =
      ::getFlattenedAffineExprs({expr}, numDims, numSymbols, &flattenedExprs,
                                localVarCst, addConservativeSemiAffineBounds);
  *flattenedExpr = flattenedExprs[0];
  return ret;
}

/// Flattens the expressions in map. Returns failure if 'expr' was unable to be
/// flattened (i.e., an unhandled semi-affine was found).
LogicalResult mlir::getFlattenedAffineExprs(
    AffineMap map, std::vector<SmallVector<int64_t, 8>> *flattenedExprs,
    FlatLinearConstraints *localVarCst, bool addConservativeSemiAffineBounds) {
  if (map.getNumResults() == 0) {
    if (localVarCst)
      *localVarCst =
          FlatLinearConstraints(map.getNumDims(), map.getNumSymbols());
    return success();
  }
  return ::getFlattenedAffineExprs(
      map.getResults(), map.getNumDims(), map.getNumSymbols(), flattenedExprs,
      localVarCst, addConservativeSemiAffineBounds);
}

LogicalResult mlir::getFlattenedAffineExprs(
    IntegerSet set, std::vector<SmallVector<int64_t, 8>> *flattenedExprs,
    FlatLinearConstraints *localVarCst) {
  if (set.getNumConstraints() == 0) {
    if (localVarCst)
      *localVarCst =
          FlatLinearConstraints(set.getNumDims(), set.getNumSymbols());
    return success();
  }
  return ::getFlattenedAffineExprs(set.getConstraints(), set.getNumDims(),
                                   set.getNumSymbols(), flattenedExprs,
                                   localVarCst);
}

//===----------------------------------------------------------------------===//
// FlatLinearConstraints
//===----------------------------------------------------------------------===//

// Similar to `composeMap` except that no Values need be associated with the
// constraint system nor are they looked at -- the dimensions and symbols of
// `other` are expected to correspond 1:1 to `this` system.
LogicalResult FlatLinearConstraints::composeMatchingMap(AffineMap other) {
  assert(other.getNumDims() == getNumDimVars() && "dim mismatch");
  assert(other.getNumSymbols() == getNumSymbolVars() && "symbol mismatch");

  std::vector<SmallVector<int64_t, 8>> flatExprs;
  if (failed(flattenAlignedMapAndMergeLocals(other, &flatExprs)))
    return failure();
  assert(flatExprs.size() == other.getNumResults());

  // Add dimensions corresponding to the map's results.
  insertDimVar(/*pos=*/0, /*num=*/other.getNumResults());

  // We add one equality for each result connecting the result dim of the map to
  // the other variables.
  // E.g.: if the expression is 16*i0 + i1, and this is the r^th
  // iteration/result of the value map, we are adding the equality:
  // d_r - 16*i0 - i1 = 0. Similarly, when flattening (i0 + 1, i0 + 8*i2), we
  // add two equalities: d_0 - i0 - 1 == 0, d1 - i0 - 8*i2 == 0.
  for (unsigned r = 0, e = flatExprs.size(); r < e; r++) {
    const auto &flatExpr = flatExprs[r];
    assert(flatExpr.size() >= other.getNumInputs() + 1);

    SmallVector<int64_t, 8> eqToAdd(getNumCols(), 0);
    // Set the coefficient for this result to one.
    eqToAdd[r] = 1;

    // Dims and symbols.
    for (unsigned i = 0, f = other.getNumInputs(); i < f; i++) {
      // Negate `eq[r]` since the newly added dimension will be set to this one.
      eqToAdd[e + i] = -flatExpr[i];
    }
    // Local columns of `eq` are at the beginning.
    unsigned j = getNumDimVars() + getNumSymbolVars();
    unsigned end = flatExpr.size() - 1;
    for (unsigned i = other.getNumInputs(); i < end; i++, j++) {
      eqToAdd[j] = -flatExpr[i];
    }

    // Constant term.
    eqToAdd[getNumCols() - 1] = -flatExpr[flatExpr.size() - 1];

    // Add the equality connecting the result of the map to this constraint set.
    addEquality(eqToAdd);
  }

  return success();
}

// Determine whether the variable at 'pos' (say var_r) can be expressed as
// modulo of another known variable (say var_n) w.r.t a constant. For example,
// if the following constraints hold true:
// ```
// 0 <= var_r <= divisor - 1
// var_n - (divisor * q_expr) = var_r
// ```
// where `var_n` is a known variable (called dividend), and `q_expr` is an
// `AffineExpr` (called the quotient expression), `var_r` can be written as:
//
// `var_r = var_n mod divisor`.
//
// Additionally, in a special case of the above constaints where `q_expr` is an
// variable itself that is not yet known (say `var_q`), it can be written as a
// floordiv in the following way:
//
// `var_q = var_n floordiv divisor`.
//
// First 'num' dimensional variables starting at 'offset' are
// derived/to-be-derived in terms of the remaining variables. The remaining
// variables are assigned trivial affine expressions in `memo`. For example,
// memo is initilized as follows for a `cst` with 5 dims, when offset=2, num=2:
// memo ==>  d0  d1  .   .   d2 ...
// cst  ==>  c0  c1  c2  c3  c4 ...
//
// Returns true if the above mod or floordiv are detected, updating 'memo' with
// these new expressions. Returns false otherwise.
static bool detectAsMod(const FlatLinearConstraints &cst, unsigned pos,
                        unsigned offset, unsigned num, int64_t lbConst,
                        int64_t ubConst, MLIRContext *context,
                        SmallVectorImpl<AffineExpr> &memo) {
  assert(pos < cst.getNumVars() && "invalid position");

  // Check if a divisor satisfying the condition `0 <= var_r <= divisor - 1` can
  // be determined.
  if (lbConst != 0 || ubConst < 1)
    return false;
  int64_t divisor = ubConst + 1;

  // Check for the aforementioned conditions in each equality.
  for (unsigned curEquality = 0, numEqualities = cst.getNumEqualities();
       curEquality < numEqualities; curEquality++) {
    int64_t coefficientAtPos = cst.atEq64(curEquality, pos);
    // If current equality does not involve `var_r`, continue to the next
    // equality.
    if (coefficientAtPos == 0)
      continue;

    // Constant term should be 0 in this equality.
    if (cst.atEq64(curEquality, cst.getNumCols() - 1) != 0)
      continue;

    // Traverse through the equality and construct the dividend expression
    // `dividendExpr`, to contain all the variables which are known and are
    // not divisible by `(coefficientAtPos * divisor)`. Hope here is that the
    // `dividendExpr` gets simplified into a single variable `var_n` discussed
    // above.
    auto dividendExpr = getAffineConstantExpr(0, context);

    // Track the terms that go into quotient expression, later used to detect
    // additional floordiv.
    unsigned quotientCount = 0;
    int quotientPosition = -1;
    int quotientSign = 1;

    // Consider each term in the current equality.
    unsigned curVar, e;
    for (curVar = 0, e = cst.getNumDimAndSymbolVars(); curVar < e; ++curVar) {
      // Ignore var_r.
      if (curVar == pos)
        continue;
      int64_t coefficientOfCurVar = cst.atEq64(curEquality, curVar);
      // Ignore vars that do not contribute to the current equality.
      if (coefficientOfCurVar == 0)
        continue;
      // Check if the current var goes into the quotient expression.
      if (coefficientOfCurVar % (divisor * coefficientAtPos) == 0) {
        quotientCount++;
        quotientPosition = curVar;
        quotientSign = (coefficientOfCurVar * coefficientAtPos) > 0 ? 1 : -1;
        continue;
      }
      // Variables that are part of dividendExpr should be known.
      if (!memo[curVar])
        break;
      // Append the current variable to the dividend expression.
      dividendExpr = dividendExpr + memo[curVar] * coefficientOfCurVar;
    }

    // Can't construct expression as it depends on a yet uncomputed var.
    if (curVar < e)
      continue;

    // Express `var_r` in terms of the other vars collected so far.
    if (coefficientAtPos > 0)
      dividendExpr = (-dividendExpr).floorDiv(coefficientAtPos);
    else
      dividendExpr = dividendExpr.floorDiv(-coefficientAtPos);

    // Simplify the expression.
    dividendExpr = simplifyAffineExpr(dividendExpr, cst.getNumDimVars(),
                                      cst.getNumSymbolVars());
    // Only if the final dividend expression is just a single var (which we call
    // `var_n`), we can proceed.
    // TODO: Handle AffineSymbolExpr as well. There is no reason to restrict it
    // to dims themselves.
    auto dimExpr = dyn_cast<AffineDimExpr>(dividendExpr);
    if (!dimExpr)
      continue;

    // Express `var_r` as `var_n % divisor` and store the expression in `memo`.
    if (quotientCount >= 1) {
      // Find the column corresponding to `dimExpr`. `num` columns starting at
      // `offset` correspond to previously unknown variables. The column
      // corresponding to the trivially known `dimExpr` can be on either side
      // of these.
      unsigned dimExprPos = dimExpr.getPosition();
      unsigned dimExprCol = dimExprPos < offset ? dimExprPos : dimExprPos + num;
      auto ub = cst.getConstantBound64(BoundType::UB, dimExprCol);
      // If `var_n` has an upperbound that is less than the divisor, mod can be
      // eliminated altogether.
      if (ub && *ub < divisor)
        memo[pos] = dimExpr;
      else
        memo[pos] = dimExpr % divisor;
      // If a unique quotient `var_q` was seen, it can be expressed as
      // `var_n floordiv divisor`.
      if (quotientCount == 1 && !memo[quotientPosition])
        memo[quotientPosition] = dimExpr.floorDiv(divisor) * quotientSign;

      return true;
    }
  }
  return false;
}

/// Check if the pos^th variable can be expressed as a floordiv of an affine
/// function of other variables (where the divisor is a positive constant)
/// given the initial set of expressions in `exprs`. If it can be, the
/// corresponding position in `exprs` is set as the detected affine expr. For
/// eg: 4q <= i + j <= 4q + 3   <=>   q = (i + j) floordiv 4. An equality can
/// also yield a floordiv: eg.  4q = i + j <=> q = (i + j) floordiv 4. 32q + 28
/// <= i <= 32q + 31 => q = i floordiv 32.
static bool detectAsFloorDiv(const FlatLinearConstraints &cst, unsigned pos,
                             MLIRContext *context,
                             SmallVectorImpl<AffineExpr> &exprs) {
  assert(pos < cst.getNumVars() && "invalid position");

  // Get upper-lower bound pair for this variable.
  SmallVector<bool, 8> foundRepr(cst.getNumVars(), false);
  for (unsigned i = 0, e = cst.getNumVars(); i < e; ++i)
    if (exprs[i])
      foundRepr[i] = true;

  SmallVector<int64_t, 8> dividend(cst.getNumCols());
  unsigned divisor;
  auto ulPair = computeSingleVarRepr(cst, foundRepr, pos, dividend, divisor);

  // No upper-lower bound pair found for this var.
  if (ulPair.kind == ReprKind::None || ulPair.kind == ReprKind::Equality)
    return false;

  // Construct the dividend expression.
  auto dividendExpr = getAffineConstantExpr(dividend.back(), context);
  for (unsigned c = 0, f = cst.getNumVars(); c < f; c++)
    if (dividend[c] != 0)
      dividendExpr = dividendExpr + dividend[c] * exprs[c];

  // Successfully detected the floordiv.
  exprs[pos] = dividendExpr.floorDiv(divisor);
  return true;
}

std::pair<AffineMap, AffineMap> FlatLinearConstraints::getLowerAndUpperBound(
    unsigned pos, unsigned offset, unsigned num, unsigned symStartPos,
    ArrayRef<AffineExpr> localExprs, MLIRContext *context,
    bool closedUB) const {
  assert(pos + offset < getNumDimVars() && "invalid dim start pos");
  assert(symStartPos >= (pos + offset) && "invalid sym start pos");
  assert(getNumLocalVars() == localExprs.size() &&
         "incorrect local exprs count");

  SmallVector<unsigned, 4> lbIndices, ubIndices, eqIndices;
  getLowerAndUpperBoundIndices(pos + offset, &lbIndices, &ubIndices, &eqIndices,
                               offset, num);

  /// Add to 'b' from 'a' in set [0, offset) U [offset + num, symbStartPos).
  auto addCoeffs = [&](ArrayRef<int64_t> a, SmallVectorImpl<int64_t> &b) {
    b.clear();
    for (unsigned i = 0, e = a.size(); i < e; ++i) {
      if (i < offset || i >= offset + num)
        b.push_back(a[i]);
    }
  };

  SmallVector<int64_t, 8> lb, ub;
  SmallVector<AffineExpr, 4> lbExprs;
  unsigned dimCount = symStartPos - num;
  unsigned symCount = getNumDimAndSymbolVars() - symStartPos;
  lbExprs.reserve(lbIndices.size() + eqIndices.size());
  // Lower bound expressions.
  for (auto idx : lbIndices) {
    auto ineq = getInequality64(idx);
    // Extract the lower bound (in terms of other coeff's + const), i.e., if
    // i - j + 1 >= 0 is the constraint, 'pos' is for i the lower bound is j
    // - 1.
    addCoeffs(ineq, lb);
    std::transform(lb.begin(), lb.end(), lb.begin(), std::negate<int64_t>());
    auto expr =
        getAffineExprFromFlatForm(lb, dimCount, symCount, localExprs, context);
    // expr ceildiv divisor is (expr + divisor - 1) floordiv divisor
    int64_t divisor = std::abs(ineq[pos + offset]);
    expr = (expr + divisor - 1).floorDiv(divisor);
    lbExprs.push_back(expr);
  }

  SmallVector<AffineExpr, 4> ubExprs;
  ubExprs.reserve(ubIndices.size() + eqIndices.size());
  // Upper bound expressions.
  for (auto idx : ubIndices) {
    auto ineq = getInequality64(idx);
    // Extract the upper bound (in terms of other coeff's + const).
    addCoeffs(ineq, ub);
    auto expr =
        getAffineExprFromFlatForm(ub, dimCount, symCount, localExprs, context);
    expr = expr.floorDiv(std::abs(ineq[pos + offset]));
    int64_t ubAdjustment = closedUB ? 0 : 1;
    ubExprs.push_back(expr + ubAdjustment);
  }

  // Equalities. It's both a lower and a upper bound.
  SmallVector<int64_t, 4> b;
  for (auto idx : eqIndices) {
    auto eq = getEquality64(idx);
    addCoeffs(eq, b);
    if (eq[pos + offset] > 0)
      std::transform(b.begin(), b.end(), b.begin(), std::negate<int64_t>());

    // Extract the upper bound (in terms of other coeff's + const).
    auto expr =
        getAffineExprFromFlatForm(b, dimCount, symCount, localExprs, context);
    expr = expr.floorDiv(std::abs(eq[pos + offset]));
    // Upper bound is exclusive.
    ubExprs.push_back(expr + 1);
    // Lower bound.
    expr =
        getAffineExprFromFlatForm(b, dimCount, symCount, localExprs, context);
    expr = expr.ceilDiv(std::abs(eq[pos + offset]));
    lbExprs.push_back(expr);
  }

  auto lbMap = AffineMap::get(dimCount, symCount, lbExprs, context);
  auto ubMap = AffineMap::get(dimCount, symCount, ubExprs, context);

  return {lbMap, ubMap};
}

/// Computes the lower and upper bounds of the first 'num' dimensional
/// variables (starting at 'offset') as affine maps of the remaining
/// variables (dimensional and symbolic variables). Local variables are
/// themselves explicitly computed as affine functions of other variables in
/// this process if needed.
void FlatLinearConstraints::getSliceBounds(unsigned offset, unsigned num,
                                           MLIRContext *context,
                                           SmallVectorImpl<AffineMap> *lbMaps,
                                           SmallVectorImpl<AffineMap> *ubMaps,
                                           bool closedUB) {
  assert(offset + num <= getNumDimVars() && "invalid range");

  // Basic simplification.
  normalizeConstraintsByGCD();

  LLVM_DEBUG(llvm::dbgs() << "getSliceBounds for first " << num
                          << " variables\n");
  LLVM_DEBUG(dump());

  // Record computed/detected variables.
  SmallVector<AffineExpr, 8> memo(getNumVars());
  // Initialize dimensional and symbolic variables.
  for (unsigned i = 0, e = getNumDimVars(); i < e; i++) {
    if (i < offset)
      memo[i] = getAffineDimExpr(i, context);
    else if (i >= offset + num)
      memo[i] = getAffineDimExpr(i - num, context);
  }
  for (unsigned i = getNumDimVars(), e = getNumDimAndSymbolVars(); i < e; i++)
    memo[i] = getAffineSymbolExpr(i - getNumDimVars(), context);

  bool changed;
  do {
    changed = false;
    // Identify yet unknown variables as constants or mod's / floordiv's of
    // other variables if possible.
    for (unsigned pos = 0; pos < getNumVars(); pos++) {
      if (memo[pos])
        continue;

      auto lbConst = getConstantBound64(BoundType::LB, pos);
      auto ubConst = getConstantBound64(BoundType::UB, pos);
      if (lbConst.has_value() && ubConst.has_value()) {
        // Detect equality to a constant.
        if (*lbConst == *ubConst) {
          memo[pos] = getAffineConstantExpr(*lbConst, context);
          changed = true;
          continue;
        }

        // Detect a variable as modulo of another variable w.r.t a
        // constant.
        if (detectAsMod(*this, pos, offset, num, *lbConst, *ubConst, context,
                        memo)) {
          changed = true;
          continue;
        }
      }

      // Detect a variable as a floordiv of an affine function of other
      // variables (divisor is a positive constant).
      if (detectAsFloorDiv(*this, pos, context, memo)) {
        changed = true;
        continue;
      }

      // Detect a variable as an expression of other variables.
      unsigned idx;
      if (!findConstraintWithNonZeroAt(pos, /*isEq=*/true, &idx)) {
        continue;
      }

      // Build AffineExpr solving for variable 'pos' in terms of all others.
      auto expr = getAffineConstantExpr(0, context);
      unsigned j, e;
      for (j = 0, e = getNumVars(); j < e; ++j) {
        if (j == pos)
          continue;
        int64_t c = atEq64(idx, j);
        if (c == 0)
          continue;
        // If any of the involved IDs hasn't been found yet, we can't proceed.
        if (!memo[j])
          break;
        expr = expr + memo[j] * c;
      }
      if (j < e)
        // Can't construct expression as it depends on a yet uncomputed
        // variable.
        continue;

      // Add constant term to AffineExpr.
      expr = expr + atEq64(idx, getNumVars());
      int64_t vPos = atEq64(idx, pos);
      assert(vPos != 0 && "expected non-zero here");
      if (vPos > 0)
        expr = (-expr).floorDiv(vPos);
      else
        // vPos < 0.
        expr = expr.floorDiv(-vPos);
      // Successfully constructed expression.
      memo[pos] = expr;
      changed = true;
    }
    // This loop is guaranteed to reach a fixed point - since once an
    // variable's explicit form is computed (in memo[pos]), it's not updated
    // again.
  } while (changed);

  int64_t ubAdjustment = closedUB ? 0 : 1;

  // Set the lower and upper bound maps for all the variables that were
  // computed as affine expressions of the rest as the "detected expr" and
  // "detected expr + 1" respectively; set the undetected ones to null.
  std::optional<FlatLinearConstraints> tmpClone;
  for (unsigned pos = 0; pos < num; pos++) {
    unsigned numMapDims = getNumDimVars() - num;
    unsigned numMapSymbols = getNumSymbolVars();
    AffineExpr expr = memo[pos + offset];
    if (expr)
      expr = simplifyAffineExpr(expr, numMapDims, numMapSymbols);

    AffineMap &lbMap = (*lbMaps)[pos];
    AffineMap &ubMap = (*ubMaps)[pos];

    if (expr) {
      lbMap = AffineMap::get(numMapDims, numMapSymbols, expr);
      ubMap = AffineMap::get(numMapDims, numMapSymbols, expr + ubAdjustment);
    } else {
      // TODO: Whenever there are local variables in the dependence
      // constraints, we'll conservatively over-approximate, since we don't
      // always explicitly compute them above (in the while loop).
      if (getNumLocalVars() == 0) {
        // Work on a copy so that we don't update this constraint system.
        if (!tmpClone) {
          tmpClone.emplace(FlatLinearConstraints(*this));
          // Removing redundant inequalities is necessary so that we don't get
          // redundant loop bounds.
          tmpClone->removeRedundantInequalities();
        }
        std::tie(lbMap, ubMap) = tmpClone->getLowerAndUpperBound(
            pos, offset, num, getNumDimVars(), /*localExprs=*/{}, context,
            closedUB);
      }

      // If the above fails, we'll just use the constant lower bound and the
      // constant upper bound (if they exist) as the slice bounds.
      // TODO: being conservative for the moment in cases that
      // lead to multiple bounds - until getConstDifference in LoopFusion.cpp is
      // fixed (b/126426796).
      if (!lbMap || lbMap.getNumResults() > 1) {
        LLVM_DEBUG(llvm::dbgs()
                   << "WARNING: Potentially over-approximating slice lb\n");
        auto lbConst = getConstantBound64(BoundType::LB, pos + offset);
        if (lbConst.has_value()) {
          lbMap = AffineMap::get(numMapDims, numMapSymbols,
                                 getAffineConstantExpr(*lbConst, context));
        }
      }
      if (!ubMap || ubMap.getNumResults() > 1) {
        LLVM_DEBUG(llvm::dbgs()
                   << "WARNING: Potentially over-approximating slice ub\n");
        auto ubConst = getConstantBound64(BoundType::UB, pos + offset);
        if (ubConst.has_value()) {
          ubMap = AffineMap::get(
              numMapDims, numMapSymbols,
              getAffineConstantExpr(*ubConst + ubAdjustment, context));
        }
      }
    }
    LLVM_DEBUG(llvm::dbgs()
               << "lb map for pos = " << Twine(pos + offset) << ", expr: ");
    LLVM_DEBUG(lbMap.dump(););
    LLVM_DEBUG(llvm::dbgs()
               << "ub map for pos = " << Twine(pos + offset) << ", expr: ");
    LLVM_DEBUG(ubMap.dump(););
  }
}

LogicalResult FlatLinearConstraints::flattenAlignedMapAndMergeLocals(
    AffineMap map, std::vector<SmallVector<int64_t, 8>> *flattenedExprs,
    bool addConservativeSemiAffineBounds) {
  FlatLinearConstraints localCst;
  if (failed(getFlattenedAffineExprs(map, flattenedExprs, &localCst,
                                     addConservativeSemiAffineBounds))) {
    LLVM_DEBUG(llvm::dbgs()
               << "composition unimplemented for semi-affine maps\n");
    return failure();
  }

  // Add localCst information.
  if (localCst.getNumLocalVars() > 0) {
    unsigned numLocalVars = getNumLocalVars();
    // Insert local dims of localCst at the beginning.
    insertLocalVar(/*pos=*/0, /*num=*/localCst.getNumLocalVars());
    // Insert local dims of `this` at the end of localCst.
    localCst.appendLocalVar(/*num=*/numLocalVars);
    // Dimensions of localCst and this constraint set match. Append localCst to
    // this constraint set.
    append(localCst);
  }

  return success();
}

LogicalResult FlatLinearConstraints::addBound(
    BoundType type, unsigned pos, AffineMap boundMap, bool isClosedBound,
    AddConservativeSemiAffineBounds addSemiAffineBounds) {
  assert(boundMap.getNumDims() == getNumDimVars() && "dim mismatch");
  assert(boundMap.getNumSymbols() == getNumSymbolVars() && "symbol mismatch");
  assert(pos < getNumDimAndSymbolVars() && "invalid position");
  assert((type != BoundType::EQ || isClosedBound) &&
         "EQ bound must be closed.");

  // Equality follows the logic of lower bound except that we add an equality
  // instead of an inequality.
  assert((type != BoundType::EQ || boundMap.getNumResults() == 1) &&
         "single result expected");
  bool lower = type == BoundType::LB || type == BoundType::EQ;

  std::vector<SmallVector<int64_t, 8>> flatExprs;
  if (failed(flattenAlignedMapAndMergeLocals(
          boundMap, &flatExprs,
          addSemiAffineBounds == AddConservativeSemiAffineBounds::Yes)))
    return failure();
  assert(flatExprs.size() == boundMap.getNumResults());

  // Add one (in)equality for each result.
  for (const auto &flatExpr : flatExprs) {
    SmallVector<int64_t> ineq(getNumCols(), 0);
    // Dims and symbols.
    for (unsigned j = 0, e = boundMap.getNumInputs(); j < e; j++) {
      ineq[j] = lower ? -flatExpr[j] : flatExpr[j];
    }
    // Invalid bound: pos appears in `boundMap`.
    // TODO: This should be an assertion. Fix `addDomainFromSliceMaps` and/or
    // its callers to prevent invalid bounds from being added.
    if (ineq[pos] != 0)
      continue;
    ineq[pos] = lower ? 1 : -1;
    // Local columns of `ineq` are at the beginning.
    unsigned j = getNumDimVars() + getNumSymbolVars();
    unsigned end = flatExpr.size() - 1;
    for (unsigned i = boundMap.getNumInputs(); i < end; i++, j++) {
      ineq[j] = lower ? -flatExpr[i] : flatExpr[i];
    }
    // Make the bound closed in if flatExpr is open. The inequality is always
    // created in the upper bound form, so the adjustment is -1.
    int64_t boundAdjustment = (isClosedBound || type == BoundType::EQ) ? 0 : -1;
    // Constant term.
    ineq[getNumCols() - 1] = (lower ? -flatExpr[flatExpr.size() - 1]
                                    : flatExpr[flatExpr.size() - 1]) +
                             boundAdjustment;
    type == BoundType::EQ ? addEquality(ineq) : addInequality(ineq);
  }

  return success();
}

LogicalResult FlatLinearConstraints::addBound(
    BoundType type, unsigned pos, AffineMap boundMap,
    AddConservativeSemiAffineBounds addSemiAffineBounds) {
  return addBound(type, pos, boundMap,
                  /*isClosedBound=*/type != BoundType::UB, addSemiAffineBounds);
}

/// Compute an explicit representation for local vars. For all systems coming
/// from MLIR integer sets, maps, or expressions where local vars were
/// introduced to model floordivs and mods, this always succeeds.
LogicalResult
FlatLinearConstraints::computeLocalVars(SmallVectorImpl<AffineExpr> &memo,
                                        MLIRContext *context) const {
  unsigned numDims = getNumDimVars();
  unsigned numSyms = getNumSymbolVars();

  // Initialize dimensional and symbolic variables.
  for (unsigned i = 0; i < numDims; i++)
    memo[i] = getAffineDimExpr(i, context);
  for (unsigned i = numDims, e = numDims + numSyms; i < e; i++)
    memo[i] = getAffineSymbolExpr(i - numDims, context);

  bool changed;
  do {
    // Each time `changed` is true at the end of this iteration, one or more
    // local vars would have been detected as floordivs and set in memo; so the
    // number of null entries in memo[...] strictly reduces; so this converges.
    changed = false;
    for (unsigned i = 0, e = getNumLocalVars(); i < e; ++i)
      if (!memo[numDims + numSyms + i] &&
          detectAsFloorDiv(*this, /*pos=*/numDims + numSyms + i, context, memo))
        changed = true;
  } while (changed);

  ArrayRef<AffineExpr> localExprs =
      ArrayRef<AffineExpr>(memo).take_back(getNumLocalVars());
  return success(
      llvm::all_of(localExprs, [](AffineExpr expr) { return expr; }));
}

IntegerSet FlatLinearConstraints::getAsIntegerSet(MLIRContext *context) const {
  if (getNumConstraints() == 0)
    // Return universal set (always true): 0 == 0.
    return IntegerSet::get(getNumDimVars(), getNumSymbolVars(),
                           getAffineConstantExpr(/*constant=*/0, context),
                           /*eqFlags=*/true);

  // Construct local references.
  SmallVector<AffineExpr, 8> memo(getNumVars(), AffineExpr());

  if (failed(computeLocalVars(memo, context))) {
    // Check if the local variables without an explicit representation have
    // zero coefficients everywhere.
    SmallVector<unsigned> noLocalRepVars;
    unsigned numDimsSymbols = getNumDimAndSymbolVars();
    for (unsigned i = numDimsSymbols, e = getNumVars(); i < e; ++i) {
      if (!memo[i] && !isColZero(/*pos=*/i))
        noLocalRepVars.push_back(i - numDimsSymbols);
    }
    if (!noLocalRepVars.empty()) {
      LLVM_DEBUG({
        llvm::dbgs() << "local variables at position(s) ";
        llvm::interleaveComma(noLocalRepVars, llvm::dbgs());
        llvm::dbgs() << " do not have an explicit representation in:\n";
        this->dump();
      });
      return IntegerSet();
    }
  }

  ArrayRef<AffineExpr> localExprs =
      ArrayRef<AffineExpr>(memo).take_back(getNumLocalVars());

  // Construct the IntegerSet from the equalities/inequalities.
  unsigned numDims = getNumDimVars();
  unsigned numSyms = getNumSymbolVars();

  SmallVector<bool, 16> eqFlags(getNumConstraints());
  std::fill(eqFlags.begin(), eqFlags.begin() + getNumEqualities(), true);
  std::fill(eqFlags.begin() + getNumEqualities(), eqFlags.end(), false);

  SmallVector<AffineExpr, 8> exprs;
  exprs.reserve(getNumConstraints());

  for (unsigned i = 0, e = getNumEqualities(); i < e; ++i)
    exprs.push_back(getAffineExprFromFlatForm(getEquality64(i), numDims,
                                              numSyms, localExprs, context));
  for (unsigned i = 0, e = getNumInequalities(); i < e; ++i)
    exprs.push_back(getAffineExprFromFlatForm(getInequality64(i), numDims,
                                              numSyms, localExprs, context));
  return IntegerSet::get(numDims, numSyms, exprs, eqFlags);
}

//===----------------------------------------------------------------------===//
// FlatLinearValueConstraints
//===----------------------------------------------------------------------===//

// Construct from an IntegerSet.
FlatLinearValueConstraints::FlatLinearValueConstraints(IntegerSet set,
                                                       ValueRange operands)
    : FlatLinearConstraints(set.getNumInequalities(), set.getNumEqualities(),
                            set.getNumDims() + set.getNumSymbols() + 1,
                            set.getNumDims(), set.getNumSymbols(),
                            /*numLocals=*/0) {
  assert((operands.empty() || set.getNumInputs() == operands.size()) &&
         "operand count mismatch");
  // Set the values for the non-local variables.
  for (unsigned i = 0, e = operands.size(); i < e; ++i)
    setValue(i, operands[i]);

  // Flatten expressions and add them to the constraint system.
  std::vector<SmallVector<int64_t, 8>> flatExprs;
  FlatLinearConstraints localVarCst;
  if (failed(getFlattenedAffineExprs(set, &flatExprs, &localVarCst))) {
    assert(false && "flattening unimplemented for semi-affine integer sets");
    return;
  }
  assert(flatExprs.size() == set.getNumConstraints());
  insertVar(VarKind::Local, getNumVarKind(VarKind::Local),
            /*num=*/localVarCst.getNumLocalVars());

  for (unsigned i = 0, e = flatExprs.size(); i < e; ++i) {
    const auto &flatExpr = flatExprs[i];
    assert(flatExpr.size() == getNumCols());
    if (set.getEqFlags()[i]) {
      addEquality(flatExpr);
    } else {
      addInequality(flatExpr);
    }
  }
  // Add the other constraints involving local vars from flattening.
  append(localVarCst);
}

unsigned FlatLinearValueConstraints::appendDimVar(ValueRange vals) {
  unsigned pos = getNumDimVars();
  return insertVar(VarKind::SetDim, pos, vals);
}

unsigned FlatLinearValueConstraints::appendSymbolVar(ValueRange vals) {
  unsigned pos = getNumSymbolVars();
  return insertVar(VarKind::Symbol, pos, vals);
}

unsigned FlatLinearValueConstraints::insertDimVar(unsigned pos,
                                                  ValueRange vals) {
  return insertVar(VarKind::SetDim, pos, vals);
}

unsigned FlatLinearValueConstraints::insertSymbolVar(unsigned pos,
                                                     ValueRange vals) {
  return insertVar(VarKind::Symbol, pos, vals);
}

unsigned FlatLinearValueConstraints::insertVar(VarKind kind, unsigned pos,
                                               unsigned num) {
  unsigned absolutePos = IntegerPolyhedron::insertVar(kind, pos, num);

  return absolutePos;
}

unsigned FlatLinearValueConstraints::insertVar(VarKind kind, unsigned pos,
                                               ValueRange vals) {
  assert(!vals.empty() && "expected ValueRange with Values.");
  assert(kind != VarKind::Local &&
         "values cannot be attached to local variables.");
  unsigned num = vals.size();
  unsigned absolutePos = IntegerPolyhedron::insertVar(kind, pos, num);

  // If a Value is provided, insert it; otherwise use std::nullopt.
  for (unsigned i = 0, e = vals.size(); i < e; ++i)
    if (vals[i])
      setValue(absolutePos + i, vals[i]);

  return absolutePos;
}

/// Checks if two constraint systems are in the same space, i.e., if they are
/// associated with the same set of variables, appearing in the same order.
static bool areVarsAligned(const FlatLinearValueConstraints &a,
                           const FlatLinearValueConstraints &b) {
  if (a.getNumDomainVars() != b.getNumDomainVars() ||
      a.getNumRangeVars() != b.getNumRangeVars() ||
      a.getNumSymbolVars() != b.getNumSymbolVars())
    return false;
  SmallVector<std::optional<Value>> aMaybeValues = a.getMaybeValues(),
                                    bMaybeValues = b.getMaybeValues();
  return std::equal(aMaybeValues.begin(), aMaybeValues.end(),
                    bMaybeValues.begin(), bMaybeValues.end());
}

/// Calls areVarsAligned to check if two constraint systems have the same set
/// of variables in the same order.
bool FlatLinearValueConstraints::areVarsAlignedWithOther(
    const FlatLinearConstraints &other) {
  return areVarsAligned(*this, other);
}

/// Checks if the SSA values associated with `cst`'s variables in range
/// [start, end) are unique.
static bool LLVM_ATTRIBUTE_UNUSED areVarsUnique(
    const FlatLinearValueConstraints &cst, unsigned start, unsigned end) {

  assert(start <= cst.getNumDimAndSymbolVars() &&
         "Start position out of bounds");
  assert(end <= cst.getNumDimAndSymbolVars() && "End position out of bounds");

  if (start >= end)
    return true;

  SmallPtrSet<Value, 8> uniqueVars;
  SmallVector<std::optional<Value>, 8> maybeValuesAll = cst.getMaybeValues();
  ArrayRef<std::optional<Value>> maybeValues = {maybeValuesAll.data() + start,
                                                maybeValuesAll.data() + end};

  for (std::optional<Value> val : maybeValues)
    if (val && !uniqueVars.insert(*val).second)
      return false;

  return true;
}

/// Checks if the SSA values associated with `cst`'s variables are unique.
static bool LLVM_ATTRIBUTE_UNUSED
areVarsUnique(const FlatLinearValueConstraints &cst) {
  return areVarsUnique(cst, 0, cst.getNumDimAndSymbolVars());
}

/// Checks if the SSA values associated with `cst`'s variables of kind `kind`
/// are unique.
static bool LLVM_ATTRIBUTE_UNUSED
areVarsUnique(const FlatLinearValueConstraints &cst, VarKind kind) {

  if (kind == VarKind::SetDim)
    return areVarsUnique(cst, 0, cst.getNumDimVars());
  if (kind == VarKind::Symbol)
    return areVarsUnique(cst, cst.getNumDimVars(),
                         cst.getNumDimAndSymbolVars());
  llvm_unreachable("Unexpected VarKind");
}

/// Merge and align the variables of A and B starting at 'offset', so that
/// both constraint systems get the union of the contained variables that is
/// dimension-wise and symbol-wise unique; both constraint systems are updated
/// so that they have the union of all variables, with A's original
/// variables appearing first followed by any of B's variables that didn't
/// appear in A. Local variables in B that have the same division
/// representation as local variables in A are merged into one. We allow A
/// and B to have non-unique values for their variables; in such cases, they are
/// still aligned with the variables appearing first aligned with those
/// appearing first in the other system from left to right.
//  E.g.: Input: A has ((%i, %j) [%M, %N]) and B has (%k, %j) [%P, %N, %M])
//        Output: both A, B have (%i, %j, %k) [%M, %N, %P]
static void mergeAndAlignVars(unsigned offset, FlatLinearValueConstraints *a,
                              FlatLinearValueConstraints *b) {
  assert(offset <= a->getNumDimVars() && offset <= b->getNumDimVars());

  assert(llvm::all_of(
      llvm::drop_begin(a->getMaybeValues(), offset),
      [](const std::optional<Value> &var) { return var.has_value(); }));

  assert(llvm::all_of(
      llvm::drop_begin(b->getMaybeValues(), offset),
      [](const std::optional<Value> &var) { return var.has_value(); }));

  SmallVector<Value, 4> aDimValues;
  a->getValues(offset, a->getNumDimVars(), &aDimValues);

  {
    // Merge dims from A into B.
    unsigned d = offset;
    for (Value aDimValue : aDimValues) {
      unsigned loc;
      // Find from the position `d` since we'd like to also consider the
      // possibility of multiple variables with the same `Value`. We align with
      // the next appearing one.
      if (b->findVar(aDimValue, &loc, d)) {
        assert(loc >= offset && "A's dim appears in B's aligned range");
        assert(loc < b->getNumDimVars() &&
               "A's dim appears in B's non-dim position");
        b->swapVar(d, loc);
      } else {
        b->insertDimVar(d, aDimValue);
      }
      d++;
    }
    // Dimensions that are in B, but not in A, are added at the end.
    for (unsigned t = a->getNumDimVars(), e = b->getNumDimVars(); t < e; t++) {
      a->appendDimVar(b->getValue(t));
    }
    assert(a->getNumDimVars() == b->getNumDimVars() &&
           "expected same number of dims");
  }

  // Merge and align symbols of A and B
  a->mergeSymbolVars(*b);
  // Merge and align locals of A and B
  a->mergeLocalVars(*b);

  assert(areVarsAligned(*a, *b) && "IDs expected to be aligned");
}

// Call 'mergeAndAlignVars' to align constraint systems of 'this' and 'other'.
void FlatLinearValueConstraints::mergeAndAlignVarsWithOther(
    unsigned offset, FlatLinearValueConstraints *other) {
  mergeAndAlignVars(offset, this, other);
}

/// Merge and align symbols of `this` and `other` such that both get union of
/// of symbols. Existing symbols need not be unique; they will be aligned from
/// left to right with duplicates aligned in the same order. Symbols with Value
/// as `None` are considered to be inequal to all other symbols.
void FlatLinearValueConstraints::mergeSymbolVars(
    FlatLinearValueConstraints &other) {

  SmallVector<Value, 4> aSymValues;
  getValues(getNumDimVars(), getNumDimAndSymbolVars(), &aSymValues);

  // Merge symbols: merge symbols into `other` first from `this`.
  unsigned s = other.getNumDimVars();
  for (Value aSymValue : aSymValues) {
    unsigned loc;
    // If the var is a symbol in `other`, then align it, otherwise assume that
    // it is a new symbol. Search in `other` starting at position `s` since the
    // left of it is aligned.
    if (other.findVar(aSymValue, &loc, s) && loc >= other.getNumDimVars() &&
        loc < other.getNumDimAndSymbolVars())
      other.swapVar(s, loc);
    else
      other.insertSymbolVar(s - other.getNumDimVars(), aSymValue);
    s++;
  }

  // Symbols that are in other, but not in this, are added at the end.
  for (unsigned t = other.getNumDimVars() + getNumSymbolVars(),
                e = other.getNumDimAndSymbolVars();
       t < e; t++)
    insertSymbolVar(getNumSymbolVars(), other.getValue(t));

  assert(getNumSymbolVars() == other.getNumSymbolVars() &&
         "expected same number of symbols");
}

void FlatLinearValueConstraints::removeVarRange(VarKind kind, unsigned varStart,
                                                unsigned varLimit) {
  IntegerPolyhedron::removeVarRange(kind, varStart, varLimit);
}

AffineMap
FlatLinearValueConstraints::computeAlignedMap(AffineMap map,
                                              ValueRange operands) const {
  assert(map.getNumInputs() == operands.size() && "number of inputs mismatch");

  SmallVector<Value> dims, syms;
#ifndef NDEBUG
  SmallVector<Value> newSyms;
  SmallVector<Value> *newSymsPtr = &newSyms;
#else
  SmallVector<Value> *newSymsPtr = nullptr;
#endif // NDEBUG

  dims.reserve(getNumDimVars());
  syms.reserve(getNumSymbolVars());
  for (unsigned i = 0, e = getNumVarKind(VarKind::SetDim); i < e; ++i) {
    Identifier id = space.getId(VarKind::SetDim, i);
    dims.push_back(id.hasValue() ? Value(id.getValue<Value>()) : Value());
  }
  for (unsigned i = 0, e = getNumVarKind(VarKind::Symbol); i < e; ++i) {
    Identifier id = space.getId(VarKind::Symbol, i);
    syms.push_back(id.hasValue() ? Value(id.getValue<Value>()) : Value());
  }

  AffineMap alignedMap =
      alignAffineMapWithValues(map, operands, dims, syms, newSymsPtr);
  // All symbols are already part of this FlatAffineValueConstraints.
  assert(syms.size() == newSymsPtr->size() && "unexpected new/missing symbols");
  assert(std::equal(syms.begin(), syms.end(), newSymsPtr->begin()) &&
         "unexpected new/missing symbols");
  return alignedMap;
}

bool FlatLinearValueConstraints::findVar(Value val, unsigned *pos,
                                         unsigned offset) const {
  SmallVector<std::optional<Value>> maybeValues = getMaybeValues();
  for (unsigned i = offset, e = maybeValues.size(); i < e; ++i)
    if (maybeValues[i] && maybeValues[i].value() == val) {
      *pos = i;
      return true;
    }
  return false;
}

bool FlatLinearValueConstraints::containsVar(Value val) const {
  unsigned pos;
  return findVar(val, &pos, 0);
}

void FlatLinearValueConstraints::addBound(BoundType type, Value val,
                                          int64_t value) {
  unsigned pos;
  if (!findVar(val, &pos))
    // This is a pre-condition for this method.
    assert(0 && "var not found");
  addBound(type, pos, value);
}

void FlatLinearConstraints::printSpace(raw_ostream &os) const {
  IntegerPolyhedron::printSpace(os);
  os << "(";
  for (unsigned i = 0, e = getNumDimAndSymbolVars(); i < e; i++)
    os << "None\t";
  for (unsigned i = getVarKindOffset(VarKind::Local),
                e = getVarKindEnd(VarKind::Local);
       i < e; ++i)
    os << "Local\t";
  os << "const)\n";
}

void FlatLinearValueConstraints::printSpace(raw_ostream &os) const {
  IntegerPolyhedron::printSpace(os);
  os << "(";
  for (unsigned i = 0, e = getNumDimAndSymbolVars(); i < e; i++) {
    if (hasValue(i))
      os << "Value\t";
    else
      os << "None\t";
  }
  for (unsigned i = getVarKindOffset(VarKind::Local),
                e = getVarKindEnd(VarKind::Local);
       i < e; ++i)
    os << "Local\t";
  os << "const)\n";
}

void FlatLinearValueConstraints::projectOut(Value val) {
  unsigned pos;
  bool ret = findVar(val, &pos);
  assert(ret);
  (void)ret;
  fourierMotzkinEliminate(pos);
}

LogicalResult FlatLinearValueConstraints::unionBoundingBox(
    const FlatLinearValueConstraints &otherCst) {
  assert(otherCst.getNumDimVars() == getNumDimVars() && "dims mismatch");
  SmallVector<std::optional<Value>> maybeValues = getMaybeValues(),
                                    otherMaybeValues =
                                        otherCst.getMaybeValues();
  assert(std::equal(maybeValues.begin(), maybeValues.begin() + getNumDimVars(),
                    otherMaybeValues.begin(),
                    otherMaybeValues.begin() + getNumDimVars()) &&
         "dim values mismatch");
  assert(otherCst.getNumLocalVars() == 0 && "local vars not supported here");
  assert(getNumLocalVars() == 0 && "local vars not supported yet here");

  // Align `other` to this.
  if (!areVarsAligned(*this, otherCst)) {
    FlatLinearValueConstraints otherCopy(otherCst);
    mergeAndAlignVars(/*offset=*/getNumDimVars(), this, &otherCopy);
    return IntegerPolyhedron::unionBoundingBox(otherCopy);
  }

  return IntegerPolyhedron::unionBoundingBox(otherCst);
}

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//

AffineMap mlir::alignAffineMapWithValues(AffineMap map, ValueRange operands,
                                         ValueRange dims, ValueRange syms,
                                         SmallVector<Value> *newSyms) {
  assert(operands.size() == map.getNumInputs() &&
         "expected same number of operands and map inputs");
  MLIRContext *ctx = map.getContext();
  Builder builder(ctx);
  SmallVector<AffineExpr> dimReplacements(map.getNumDims(), {});
  unsigned numSymbols = syms.size();
  SmallVector<AffineExpr> symReplacements(map.getNumSymbols(), {});
  if (newSyms) {
    newSyms->clear();
    newSyms->append(syms.begin(), syms.end());
  }

  for (const auto &operand : llvm::enumerate(operands)) {
    // Compute replacement dim/sym of operand.
    AffineExpr replacement;
    auto dimIt = llvm::find(dims, operand.value());
    auto symIt = llvm::find(syms, operand.value());
    if (dimIt != dims.end()) {
      replacement =
          builder.getAffineDimExpr(std::distance(dims.begin(), dimIt));
    } else if (symIt != syms.end()) {
      replacement =
          builder.getAffineSymbolExpr(std::distance(syms.begin(), symIt));
    } else {
      // This operand is neither a dimension nor a symbol. Add it as a new
      // symbol.
      replacement = builder.getAffineSymbolExpr(numSymbols++);
      if (newSyms)
        newSyms->push_back(operand.value());
    }
    // Add to corresponding replacements vector.
    if (operand.index() < map.getNumDims()) {
      dimReplacements[operand.index()] = replacement;
    } else {
      symReplacements[operand.index() - map.getNumDims()] = replacement;
    }
  }

  return map.replaceDimsAndSymbols(dimReplacements, symReplacements,
                                   dims.size(), numSymbols);
}

LogicalResult
mlir::getMultiAffineFunctionFromMap(AffineMap map,
                                    MultiAffineFunction &multiAff) {
  FlatLinearConstraints cst;
  std::vector<SmallVector<int64_t, 8>> flattenedExprs;
  LogicalResult result = getFlattenedAffineExprs(map, &flattenedExprs, &cst);

  if (result.failed())
    return failure();

  DivisionRepr divs = cst.getLocalReprs();
  assert(divs.hasAllReprs() &&
         "AffineMap cannot produce divs without local representation");

  // TODO: We shouldn't have to do this conversion.
  Matrix<DynamicAPInt> mat(map.getNumResults(),
                           map.getNumInputs() + divs.getNumDivs() + 1);
  for (unsigned i = 0, e = flattenedExprs.size(); i < e; ++i)
    for (unsigned j = 0, f = flattenedExprs[i].size(); j < f; ++j)
      mat(i, j) = flattenedExprs[i][j];

  multiAff = MultiAffineFunction(
      PresburgerSpace::getRelationSpace(map.getNumDims(), map.getNumResults(),
                                        map.getNumSymbols(), divs.getNumDivs()),
      mat, divs);

  return success();
}
