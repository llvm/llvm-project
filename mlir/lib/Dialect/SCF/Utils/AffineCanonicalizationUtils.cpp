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

#include <utility>

#include "mlir/Dialect/SCF/Utils/AffineCanonicalizationUtils.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "mlir-scf-affine-utils"

using namespace mlir;
using namespace presburger;

static FailureOr<AffineApplyOp>
canonicalizeMinMaxOp(RewriterBase &rewriter, Operation *op,
                     FlatAffineValueConstraints constraints) {
  RewriterBase::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(op);
  FailureOr<AffineValueMap> simplified =
      mlir::simplifyConstrainedMinMaxOp(op, std::move(constraints));
  if (failed(simplified))
    return failure();
  return rewriter.replaceOpWithNewOp<AffineApplyOp>(
      op, simplified->getAffineMap(), simplified->getOperands());
}

static LogicalResult
addLoopRangeConstraints(FlatAffineValueConstraints &constraints, Value iv,
                        OpFoldResult lb, OpFoldResult ub, OpFoldResult step,
                        RewriterBase &rewriter) {
  // IntegerPolyhedron does not support semi-affine expressions.
  // Therefore, only constant step values are supported.
  auto stepInt = getConstantIntValue(step);
  if (!stepInt)
    return failure();

  unsigned dimIv = constraints.appendDimVar(iv);
  auto lbv = lb.dyn_cast<Value>();
  unsigned symLb = lbv ? constraints.appendSymbolVar(lbv)
                       : constraints.appendSymbolVar(/*num=*/1);
  auto ubv = ub.dyn_cast<Value>();
  unsigned symUb = ubv ? constraints.appendSymbolVar(ubv)
                       : constraints.appendSymbolVar(/*num=*/1);

  // If loop lower/upper bounds are constant: Add EQ constraint.
  std::optional<int64_t> lbInt = getConstantIntValue(lb);
  std::optional<int64_t> ubInt = getConstantIntValue(ub);
  if (lbInt)
    constraints.addBound(IntegerPolyhedron::EQ, symLb, *lbInt);
  if (ubInt)
    constraints.addBound(IntegerPolyhedron::EQ, symUb, *ubInt);

  // Lower bound: iv >= lb (equiv.: iv - lb >= 0)
  SmallVector<int64_t> ineqLb(constraints.getNumCols(), 0);
  ineqLb[dimIv] = 1;
  ineqLb[symLb] = -1;
  constraints.addInequality(ineqLb);

  // Upper bound
  AffineExpr ivUb;
  if (lbInt && ubInt && (*lbInt + *stepInt >= *ubInt)) {
    // The loop has at most one iteration.
    // iv < lb + 1
    // TODO: Try to derive this constraint by simplifying the expression in
    // the else-branch.
    ivUb =
        rewriter.getAffineSymbolExpr(symLb - constraints.getNumDimVars()) + 1;
  } else {
    // The loop may have more than one iteration.
    // iv < lb + step * ((ub - lb - 1) floorDiv step) + 1
    AffineExpr exprLb =
        lbInt
            ? rewriter.getAffineConstantExpr(*lbInt)
            : rewriter.getAffineSymbolExpr(symLb - constraints.getNumDimVars());
    AffineExpr exprUb =
        ubInt
            ? rewriter.getAffineConstantExpr(*ubInt)
            : rewriter.getAffineSymbolExpr(symUb - constraints.getNumDimVars());
    ivUb = exprLb + 1 + (*stepInt * ((exprUb - exprLb - 1).floorDiv(*stepInt)));
  }
  auto map = AffineMap::get(
      /*dimCount=*/constraints.getNumDimVars(),
      /*symbolCount=*/constraints.getNumSymbolVars(), /*result=*/ivUb);

  return constraints.addBound(IntegerPolyhedron::UB, dimIv, map);
}

/// Canonicalize min/max operations in the context of for loops with a known
/// range. Call `canonicalizeMinMaxOp` and add the following constraints to
/// the constraint system (along with the missing dimensions):
///
/// * iv >= lb
/// * iv < lb + step * ((ub - lb - 1) floorDiv step) + 1
///
/// Note: Due to limitations of IntegerPolyhedron, only constant step sizes
/// are currently supported.
LogicalResult scf::canonicalizeMinMaxOpInLoop(RewriterBase &rewriter,
                                              Operation *op,
                                              LoopMatcherFn loopMatcher) {
  FlatAffineValueConstraints constraints;
  DenseSet<Value> allIvs;

  // Find all iteration variables among `minOp`'s operands add constrain them.
  for (Value operand : op->getOperands()) {
    // Skip duplicate ivs.
    if (llvm::is_contained(allIvs, operand))
      continue;

    // If `operand` is an iteration variable: Find corresponding loop
    // bounds and step.
    Value iv = operand;
    OpFoldResult lb, ub, step;
    if (failed(loopMatcher(operand, lb, ub, step)))
      continue;
    allIvs.insert(iv);

    if (failed(
            addLoopRangeConstraints(constraints, iv, lb, ub, step, rewriter)))
      return failure();
  }

  return canonicalizeMinMaxOp(rewriter, op, constraints);
}

/// Try to simplify the given affine.min/max operation `op` after loop peeling.
/// This function can simplify min/max operations such as (ub is the previous
/// upper bound of the unpeeled loop):
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
                                         Value iv, Value ub, Value step,
                                         bool insideLoop) {
  FlatAffineValueConstraints constraints;
  constraints.appendDimVar({iv});
  constraints.appendSymbolVar({ub, step});
  if (auto constUb = getConstantIntValue(ub))
    constraints.addBound(IntegerPolyhedron::EQ, 1, *constUb);
  if (auto constStep = getConstantIntValue(step))
    constraints.addBound(IntegerPolyhedron::EQ, 2, *constStep);

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

  return canonicalizeMinMaxOp(rewriter, op, constraints);
}
