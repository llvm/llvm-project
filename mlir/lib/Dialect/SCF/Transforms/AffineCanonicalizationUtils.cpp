//===- AffineCanonicalizationUtils.cpp - Affine Canonicalization in SCF ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utility functions to canonicalize affine ops within SCF op regions.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SCF/AffineCanonicalizationUtils.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "mlir-scf-affine-utils"

using namespace mlir;

static void unpackOptionalValues(ArrayRef<Optional<Value>> source,
                                 SmallVector<Value> &target) {
  target = llvm::to_vector<4>(llvm::map_range(source, [](Optional<Value> val) {
    return val.hasValue() ? *val : Value();
  }));
}

/// Bound an identifier `pos` in a given FlatAffineValueConstraints with
/// constraints drawn from an affine map. Before adding the constraint, the
/// dimensions/symbols of the affine map are aligned with `constraints`.
/// `operands` are the SSA Value operands used with the affine map.
/// Note: This function adds a new symbol column to the `constraints` for each
/// dimension/symbol that exists in the affine map but not in `constraints`.
static LogicalResult alignAndAddBound(FlatAffineValueConstraints &constraints,
                                      FlatAffineConstraints::BoundType type,
                                      unsigned pos, AffineMap map,
                                      ValueRange operands) {
  SmallVector<Value> dims, syms, newSyms;
  unpackOptionalValues(constraints.getMaybeDimValues(), dims);
  unpackOptionalValues(constraints.getMaybeSymbolValues(), syms);

  AffineMap alignedMap =
      alignAffineMapWithValues(map, operands, dims, syms, &newSyms);
  for (unsigned i = syms.size(); i < newSyms.size(); ++i)
    constraints.appendSymbolId(newSyms[i]);
  return constraints.addBound(type, pos, alignedMap);
}

/// Add `val` to each result of `map`.
static AffineMap addConstToResults(AffineMap map, int64_t val) {
  SmallVector<AffineExpr> newResults;
  for (AffineExpr r : map.getResults())
    newResults.push_back(r + val);
  return AffineMap::get(map.getNumDims(), map.getNumSymbols(), newResults,
                        map.getContext());
}

/// This function tries to canonicalize min/max operations by proving that their
/// value is bounded by the same lower and upper bound. In that case, the
/// operation can be folded away.
///
/// Bounds are computed by FlatAffineValueConstraints. Invariants required for
/// finding/proving bounds should be supplied via `constraints`.
///
/// 1. Add dimensions for `op` and `opBound` (lower or upper bound of `op`).
/// 2. Compute an upper bound of `op` (in case of `isMin`) or a lower bound (in
///    case of `!isMin`) and bind it to `opBound`. SSA values that are used in
///    `op` but are not part of `constraints`, are added as extra symbols.
/// 3. For each result of `op`: Add result as a dimension `r_i`. Prove that:
///    * If `isMin`: r_i >= opBound
///    * If `isMax`: r_i <= opBound
///    If this is the case, ub(op) == lb(op).
/// 4. Replace `op` with `opBound`.
///
/// In summary, the following constraints are added throughout this function.
/// Note: `invar` are dimensions added by the caller to express the invariants.
/// (Showing only the case where `isMin`.)
///
///  invar |    op | opBound | r_i | extra syms... | const |           eq/ineq
///  ------+-------+---------+-----+---------------+-------+-------------------
///   (various eq./ineq. constraining `invar`, added by the caller)
///    ... |     0 |       0 |   0 |             0 |   ... |               ...
///  ------+-------+---------+-----+---------------+-------+-------------------
///   (various ineq. constraining `op` in terms of `op` operands (`invar` and
///    extra `op` operands "extra syms" that are not in `invar`)).
///    ... |    -1 |       0 |   0 |           ... |   ... |              >= 0
///  ------+-------+---------+-----+---------------+-------+-------------------
///   (set `opBound` to `op` upper bound in terms of `invar` and "extra syms")
///    ... |     0 |      -1 |   0 |           ... |   ... |               = 0
///  ------+-------+---------+-----+---------------+-------+-------------------
///   (for each `op` map result r_i: set r_i to corresponding map result,
///    prove that r_i >= minOpUb via contradiction)
///    ... |     0 |       0 |  -1 |           ... |   ... |               = 0
///      0 |     0 |       1 |  -1 |             0 |    -1 |              >= 0
///
static LogicalResult
canonicalizeMinMaxOp(RewriterBase &rewriter, Operation *op, AffineMap map,
                     ValueRange operands, bool isMin,
                     FlatAffineValueConstraints constraints) {
  RewriterBase::InsertionGuard guard(rewriter);
  unsigned numResults = map.getNumResults();

  // Add a few extra dimensions.
  unsigned dimOp = constraints.appendDimId();      // `op`
  unsigned dimOpBound = constraints.appendDimId(); // `op` lower/upper bound
  unsigned resultDimStart = constraints.appendDimId(/*num=*/numResults);

  // Add an inequality for each result expr_i of map:
  // isMin: op <= expr_i, !isMin: op >= expr_i
  auto boundType =
      isMin ? FlatAffineConstraints::UB : FlatAffineConstraints::LB;
  // Upper bounds are exclusive, so add 1. (`affine.min` ops are inclusive.)
  AffineMap mapLbUb = isMin ? addConstToResults(map, 1) : map;
  if (failed(
          alignAndAddBound(constraints, boundType, dimOp, mapLbUb, operands)))
    return failure();

  // Try to compute a lower/upper bound for op, expressed in terms of the other
  // `dims` and extra symbols.
  SmallVector<AffineMap> opLb(1), opUb(1);
  constraints.getSliceBounds(dimOp, 1, rewriter.getContext(), &opLb, &opUb);
  AffineMap sliceBound = isMin ? opUb[0] : opLb[0];
  // TODO: `getSliceBounds` may return multiple bounds at the moment. This is
  // a TODO of `getSliceBounds` and not handled here.
  if (!sliceBound || sliceBound.getNumResults() != 1)
    return failure(); // No or multiple bounds found.
  // Recover the inclusive UB in the case of an `affine.min`.
  AffineMap boundMap = isMin ? addConstToResults(sliceBound, -1) : sliceBound;

  // Add an equality: Set dimOpBound to computed bound.
  // Add back dimension for op. (Was removed by `getSliceBounds`.)
  AffineMap alignedBoundMap = boundMap.shiftDims(/*shift=*/1, /*offset=*/dimOp);
  if (failed(constraints.addBound(FlatAffineConstraints::EQ, dimOpBound,
                                  alignedBoundMap)))
    return failure();

  // If the constraint system is empty, there is an inconsistency. (E.g., this
  // can happen if loop lb > ub.)
  if (constraints.isEmpty())
    return failure();

  // In the case of `isMin` (`!isMin` is inversed):
  // Prove that each result of `map` has a lower bound that is equal to (or
  // greater than) the upper bound of `op` (`dimOpBound`). In that case, `op`
  // can be replaced with the bound. I.e., prove that for each result
  // expr_i (represented by dimension r_i):
  //
  // r_i >= opBound
  //
  // To prove this inequality, add its negation to the constraint set and prove
  // that the constraint set is empty.
  for (unsigned i = resultDimStart; i < resultDimStart + numResults; ++i) {
    FlatAffineValueConstraints newConstr(constraints);

    // Add an equality: r_i = expr_i
    // Note: These equalities could have been added earlier and used to express
    // minOp <= expr_i. However, then we run the risk that `getSliceBounds`
    // computes minOpUb in terms of r_i dims, which is not desired.
    if (failed(alignAndAddBound(newConstr, FlatAffineConstraints::EQ, i,
                                map.getSubMap({i - resultDimStart}), operands)))
      return failure();

    // If `isMin`:  Add inequality: r_i < opBound
    //              equiv.: opBound - r_i - 1 >= 0
    // If `!isMin`: Add inequality: r_i > opBound
    //              equiv.: -opBound + r_i - 1 >= 0
    SmallVector<int64_t> ineq(newConstr.getNumCols(), 0);
    ineq[dimOpBound] = isMin ? 1 : -1;
    ineq[i] = isMin ? -1 : 1;
    ineq[newConstr.getNumCols() - 1] = -1;
    newConstr.addInequality(ineq);
    if (!newConstr.isEmpty())
      return failure();
  }

  // Lower and upper bound of `op` are equal. Replace `minOp` with its bound.
  AffineMap newMap = alignedBoundMap;
  SmallVector<Value> newOperands;
  unpackOptionalValues(constraints.getMaybeDimAndSymbolValues(), newOperands);
  mlir::canonicalizeMapAndOperands(&newMap, &newOperands);
  rewriter.setInsertionPoint(op);
  rewriter.replaceOpWithNewOp<AffineApplyOp>(op, newMap, newOperands);
  return success();
}

static LogicalResult
addLoopRangeConstraints(FlatAffineValueConstraints &constraints, Value iv,
                        Value lb, Value ub, Value step,
                        RewriterBase &rewriter) {
  // FlatAffineConstraints does not support semi-affine expressions.
  // Therefore, only constant step values are supported.
  auto stepInt = getConstantIntValue(step);
  if (!stepInt)
    return failure();

  unsigned dimIv = constraints.appendDimId(iv);
  unsigned dimLb = constraints.appendDimId(lb);
  unsigned dimUb = constraints.appendDimId(ub);

  // If loop lower/upper bounds are constant: Add EQ constraint.
  Optional<int64_t> lbInt = getConstantIntValue(lb);
  Optional<int64_t> ubInt = getConstantIntValue(ub);
  if (lbInt)
    constraints.addBound(FlatAffineConstraints::EQ, dimLb, *lbInt);
  if (ubInt)
    constraints.addBound(FlatAffineConstraints::EQ, dimUb, *ubInt);

  // iv >= lb (equiv.: iv - lb >= 0)
  SmallVector<int64_t> ineqLb(constraints.getNumCols(), 0);
  ineqLb[dimIv] = 1;
  ineqLb[dimLb] = -1;
  constraints.addInequality(ineqLb);

  // iv < lb + step * ((ub - lb - 1) floorDiv step) + 1
  AffineExpr exprLb = lbInt ? rewriter.getAffineConstantExpr(*lbInt)
                            : rewriter.getAffineDimExpr(dimLb);
  AffineExpr exprUb = ubInt ? rewriter.getAffineConstantExpr(*ubInt)
                            : rewriter.getAffineDimExpr(dimUb);
  AffineExpr ivUb =
      exprLb + 1 + (*stepInt * ((exprUb - exprLb - 1).floorDiv(*stepInt)));
  auto map = AffineMap::get(
      /*dimCount=*/constraints.getNumDimIds(),
      /*symbolCount=*/constraints.getNumSymbolIds(), /*result=*/ivUb);

  return constraints.addBound(FlatAffineConstraints::UB, dimIv, map);
}

/// Canonicalize min/max operations in the context of for loops with a known
/// range. Call `canonicalizeMinMaxOp` and add the following constraints to
/// the constraint system (along with the missing dimensions):
///
/// * iv >= lb
/// * iv < lb + step * ((ub - lb - 1) floorDiv step) + 1
///
/// Note: Due to limitations of FlatAffineConstraints, only constant step sizes
/// are currently supported.
LogicalResult scf::canonicalizeMinMaxOpInLoop(RewriterBase &rewriter,
                                              Operation *op, AffineMap map,
                                              ValueRange operands, bool isMin,
                                              LoopMatcherFn loopMatcher) {
  FlatAffineValueConstraints constraints;
  DenseSet<Value> allIvs;

  // Find all iteration variables among `minOp`'s operands add constrain them.
  for (Value operand : operands) {
    // Skip duplicate ivs.
    if (llvm::find(allIvs, operand) != allIvs.end())
      continue;

    // If `operand` is an iteration variable: Find corresponding loop
    // bounds and step.
    Value iv = operand;
    Value lb, ub, step;
    if (failed(loopMatcher(operand, lb, ub, step)))
      continue;
    allIvs.insert(iv);

    if (failed(
            addLoopRangeConstraints(constraints, iv, lb, ub, step, rewriter)))
      return failure();
  }

  return canonicalizeMinMaxOp(rewriter, op, map, operands, isMin, constraints);
}

/// Try to simplify a min/max operation `op` after loop peeling. This function
/// can simplify min/max operations such as (ub is the previous upper bound of
/// the unpeeled loop):
/// ```
/// #map = affine_map<(d0)[s0, s1] -> (s0, -d0 + s1)>
/// %r = affine.min #affine.min #map(%iv)[%step, %ub]
/// ```
/// and rewrites them into (in the case the peeled loop):
/// ```
/// %r = %step
/// ```
/// min/max operations inside the partial iteration are rewritten in a similar
/// way.
///
/// This function builds up a set of constraints, capable of proving that:
/// * Inside the peeled loop: min(step, ub - iv) == step
/// * Inside the partial iteration: min(step, ub - iv) == ub - iv
///
/// Returns `success` if the given operation was replaced by a new operation;
/// `failure` otherwise.
///
/// Note: `ub` is the previous upper bound of the loop (before peeling).
/// `insideLoop` must be true for min/max ops inside the loop and false for
/// affine.min ops inside the partial iteration. For an explanation of the other
/// parameters, see comment of `canonicalizeMinMaxOpInLoop`.
LogicalResult scf::rewritePeeledMinMaxOp(RewriterBase &rewriter, Operation *op,
                                         AffineMap map, ValueRange operands,
                                         bool isMin, Value iv, Value ub,
                                         Value step, bool insideLoop) {
  FlatAffineValueConstraints constraints;
  constraints.appendDimId({iv, ub, step});
  if (auto constUb = getConstantIntValue(ub))
    constraints.addBound(FlatAffineConstraints::EQ, 1, *constUb);
  if (auto constStep = getConstantIntValue(step))
    constraints.addBound(FlatAffineConstraints::EQ, 2, *constStep);

  // Add loop peeling invariant. This is the main piece of knowledge that
  // enables AffineMinOp simplification.
  if (insideLoop) {
    // ub - iv >= step (equiv.: -iv + ub - step + 0 >= 0)
    // Intuitively: Inside the peeled loop, every iteration is a "full"
    // iteration, i.e., step divides the iteration space `ub - lb` evenly.
    constraints.addInequality({-1, 1, -1, 0});
  } else {
    // ub - iv < step (equiv.: iv + -ub + step - 1 >= 0)
    // Intuitively: `iv` is the split bound here, i.e., the iteration variable
    // value of the very last iteration (in the unpeeled loop). At that point,
    // there are less than `step` elements remaining. (Otherwise, the peeled
    // loop would run for at least one more iteration.)
    constraints.addInequality({1, -1, 1, -1});
  }

  return canonicalizeMinMaxOp(rewriter, op, map, operands, isMin, constraints);
}
