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

#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/AffineCanonicalizationUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "mlir-scf-affine-utils"

using namespace mlir;
using namespace affine;
using namespace presburger;

LogicalResult scf::matchForLikeLoop(Value iv, OpFoldResult &lb,
                                    OpFoldResult &ub, OpFoldResult &step) {
  if (scf::ForOp forOp = scf::getForInductionVarOwner(iv)) {
    lb = forOp.getLowerBound();
    ub = forOp.getUpperBound();
    step = forOp.getStep();
    return success();
  }
  if (scf::ParallelOp parOp = scf::getParallelForInductionVarOwner(iv)) {
    for (unsigned idx = 0; idx < parOp.getNumLoops(); ++idx) {
      if (parOp.getInductionVars()[idx] == iv) {
        lb = parOp.getLowerBound()[idx];
        ub = parOp.getUpperBound()[idx];
        step = parOp.getStep()[idx];
        return success();
      }
    }
    return failure();
  }
  if (scf::ForallOp forallOp = scf::getForallOpThreadIndexOwner(iv)) {
    for (int64_t idx = 0; idx < forallOp.getRank(); ++idx) {
      if (forallOp.getInductionVar(idx) == iv) {
        lb = forallOp.getMixedLowerBound()[idx];
        ub = forallOp.getMixedUpperBound()[idx];
        step = forallOp.getMixedStep()[idx];
        return success();
      }
    }
    return failure();
  }
  return failure();
}

static FailureOr<AffineApplyOp>
canonicalizeMinMaxOp(RewriterBase &rewriter, Operation *op,
                     FlatAffineValueConstraints constraints) {
  RewriterBase::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(op);
  FailureOr<AffineValueMap> simplified =
      affine::simplifyConstrainedMinMaxOp(op, std::move(constraints));
  if (failed(simplified))
    return failure();
  return rewriter.replaceOpWithNewOp<AffineApplyOp>(
      op, simplified->getAffineMap(), simplified->getOperands());
}

LogicalResult scf::addLoopRangeConstraints(FlatAffineValueConstraints &cstr,
                                           Value iv, OpFoldResult lb,
                                           OpFoldResult ub, OpFoldResult step) {
  Builder b(iv.getContext());

  // IntegerPolyhedron does not support semi-affine expressions.
  // Therefore, only constant step values are supported.
  auto stepInt = getConstantIntValue(step);
  if (!stepInt)
    return failure();

  unsigned dimIv = cstr.appendDimVar(iv);
  auto lbv = llvm::dyn_cast_if_present<Value>(lb);
  unsigned symLb =
      lbv ? cstr.appendSymbolVar(lbv) : cstr.appendSymbolVar(/*num=*/1);
  auto ubv = llvm::dyn_cast_if_present<Value>(ub);
  unsigned symUb =
      ubv ? cstr.appendSymbolVar(ubv) : cstr.appendSymbolVar(/*num=*/1);

  // If loop lower/upper bounds are constant: Add EQ constraint.
  std::optional<int64_t> lbInt = getConstantIntValue(lb);
  std::optional<int64_t> ubInt = getConstantIntValue(ub);
  if (lbInt)
    cstr.addBound(BoundType::EQ, symLb, *lbInt);
  if (ubInt)
    cstr.addBound(BoundType::EQ, symUb, *ubInt);

  // Lower bound: iv >= lb (equiv.: iv - lb >= 0)
  SmallVector<int64_t> ineqLb(cstr.getNumCols(), 0);
  ineqLb[dimIv] = 1;
  ineqLb[symLb] = -1;
  cstr.addInequality(ineqLb);

  // Upper bound
  AffineExpr ivUb;
  if (lbInt && ubInt && (*lbInt + *stepInt >= *ubInt)) {
    // The loop has at most one iteration.
    // iv < lb + 1
    // TODO: Try to derive this constraint by simplifying the expression in
    // the else-branch.
    ivUb = b.getAffineSymbolExpr(symLb - cstr.getNumDimVars()) + 1;
  } else {
    // The loop may have more than one iteration.
    // iv < lb + step * ((ub - lb - 1) floorDiv step) + 1
    AffineExpr exprLb =
        lbInt ? b.getAffineConstantExpr(*lbInt)
              : b.getAffineSymbolExpr(symLb - cstr.getNumDimVars());
    AffineExpr exprUb =
        ubInt ? b.getAffineConstantExpr(*ubInt)
              : b.getAffineSymbolExpr(symUb - cstr.getNumDimVars());
    ivUb = exprLb + 1 + (*stepInt * ((exprUb - exprLb - 1).floorDiv(*stepInt)));
  }
  auto map = AffineMap::get(
      /*dimCount=*/cstr.getNumDimVars(),
      /*symbolCount=*/cstr.getNumSymbolVars(), /*result=*/ivUb);

  return cstr.addBound(BoundType::UB, dimIv, map);
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
    if (allIvs.contains(operand))
      continue;

    // If `operand` is an iteration variable: Find corresponding loop
    // bounds and step.
    Value iv = operand;
    OpFoldResult lb, ub, step;
    if (failed(loopMatcher(operand, lb, ub, step)))
      continue;
    allIvs.insert(iv);

    if (failed(addLoopRangeConstraints(constraints, iv, lb, ub, step)))
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
    constraints.addBound(BoundType::EQ, 1, *constUb);
  if (auto constStep = getConstantIntValue(step))
    constraints.addBound(BoundType::EQ, 2, *constStep);

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
