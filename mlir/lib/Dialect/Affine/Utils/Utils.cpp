//===- Utils.cpp ---- Utilities for affine dialect transformation ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements miscellaneous transformation utilities for the Affine
// dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/Utils.h"

#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "affine-utils"

using namespace mlir;
using namespace presburger;

namespace {
/// Visit affine expressions recursively and build the sequence of operations
/// that correspond to it.  Visitation functions return an Value of the
/// expression subtree they visited or `nullptr` on error.
class AffineApplyExpander
    : public AffineExprVisitor<AffineApplyExpander, Value> {
public:
  /// This internal class expects arguments to be non-null, checks must be
  /// performed at the call site.
  AffineApplyExpander(OpBuilder &builder, ValueRange dimValues,
                      ValueRange symbolValues, Location loc)
      : builder(builder), dimValues(dimValues), symbolValues(symbolValues),
        loc(loc) {}

  template <typename OpTy>
  Value buildBinaryExpr(AffineBinaryOpExpr expr) {
    auto lhs = visit(expr.getLHS());
    auto rhs = visit(expr.getRHS());
    if (!lhs || !rhs)
      return nullptr;
    auto op = builder.create<OpTy>(loc, lhs, rhs);
    return op.getResult();
  }

  Value visitAddExpr(AffineBinaryOpExpr expr) {
    return buildBinaryExpr<arith::AddIOp>(expr);
  }

  Value visitMulExpr(AffineBinaryOpExpr expr) {
    return buildBinaryExpr<arith::MulIOp>(expr);
  }

  /// Euclidean modulo operation: negative RHS is not allowed.
  /// Remainder of the euclidean integer division is always non-negative.
  ///
  /// Implemented as
  ///
  ///     a mod b =
  ///         let remainder = srem a, b;
  ///             negative = a < 0 in
  ///         select negative, remainder + b, remainder.
  Value visitModExpr(AffineBinaryOpExpr expr) {
    auto rhsConst = expr.getRHS().dyn_cast<AffineConstantExpr>();
    if (!rhsConst) {
      emitError(
          loc,
          "semi-affine expressions (modulo by non-const) are not supported");
      return nullptr;
    }
    if (rhsConst.getValue() <= 0) {
      emitError(loc, "modulo by non-positive value is not supported");
      return nullptr;
    }

    auto lhs = visit(expr.getLHS());
    auto rhs = visit(expr.getRHS());
    assert(lhs && rhs && "unexpected affine expr lowering failure");

    Value remainder = builder.create<arith::RemSIOp>(loc, lhs, rhs);
    Value zeroCst = builder.create<arith::ConstantIndexOp>(loc, 0);
    Value isRemainderNegative = builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::slt, remainder, zeroCst);
    Value correctedRemainder =
        builder.create<arith::AddIOp>(loc, remainder, rhs);
    Value result = builder.create<arith::SelectOp>(
        loc, isRemainderNegative, correctedRemainder, remainder);
    return result;
  }

  /// Floor division operation (rounds towards negative infinity).
  ///
  /// For positive divisors, it can be implemented without branching and with a
  /// single division operation as
  ///
  ///        a floordiv b =
  ///            let negative = a < 0 in
  ///            let absolute = negative ? -a - 1 : a in
  ///            let quotient = absolute / b in
  ///                negative ? -quotient - 1 : quotient
  Value visitFloorDivExpr(AffineBinaryOpExpr expr) {
    auto rhsConst = expr.getRHS().dyn_cast<AffineConstantExpr>();
    if (!rhsConst) {
      emitError(
          loc,
          "semi-affine expressions (division by non-const) are not supported");
      return nullptr;
    }
    if (rhsConst.getValue() <= 0) {
      emitError(loc, "division by non-positive value is not supported");
      return nullptr;
    }

    auto lhs = visit(expr.getLHS());
    auto rhs = visit(expr.getRHS());
    assert(lhs && rhs && "unexpected affine expr lowering failure");

    Value zeroCst = builder.create<arith::ConstantIndexOp>(loc, 0);
    Value noneCst = builder.create<arith::ConstantIndexOp>(loc, -1);
    Value negative = builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::slt, lhs, zeroCst);
    Value negatedDecremented = builder.create<arith::SubIOp>(loc, noneCst, lhs);
    Value dividend =
        builder.create<arith::SelectOp>(loc, negative, negatedDecremented, lhs);
    Value quotient = builder.create<arith::DivSIOp>(loc, dividend, rhs);
    Value correctedQuotient =
        builder.create<arith::SubIOp>(loc, noneCst, quotient);
    Value result = builder.create<arith::SelectOp>(loc, negative,
                                                   correctedQuotient, quotient);
    return result;
  }

  /// Ceiling division operation (rounds towards positive infinity).
  ///
  /// For positive divisors, it can be implemented without branching and with a
  /// single division operation as
  ///
  ///     a ceildiv b =
  ///         let negative = a <= 0 in
  ///         let absolute = negative ? -a : a - 1 in
  ///         let quotient = absolute / b in
  ///             negative ? -quotient : quotient + 1
  Value visitCeilDivExpr(AffineBinaryOpExpr expr) {
    auto rhsConst = expr.getRHS().dyn_cast<AffineConstantExpr>();
    if (!rhsConst) {
      emitError(loc) << "semi-affine expressions (division by non-const) are "
                        "not supported";
      return nullptr;
    }
    if (rhsConst.getValue() <= 0) {
      emitError(loc, "division by non-positive value is not supported");
      return nullptr;
    }
    auto lhs = visit(expr.getLHS());
    auto rhs = visit(expr.getRHS());
    assert(lhs && rhs && "unexpected affine expr lowering failure");

    Value zeroCst = builder.create<arith::ConstantIndexOp>(loc, 0);
    Value oneCst = builder.create<arith::ConstantIndexOp>(loc, 1);
    Value nonPositive = builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::sle, lhs, zeroCst);
    Value negated = builder.create<arith::SubIOp>(loc, zeroCst, lhs);
    Value decremented = builder.create<arith::SubIOp>(loc, lhs, oneCst);
    Value dividend =
        builder.create<arith::SelectOp>(loc, nonPositive, negated, decremented);
    Value quotient = builder.create<arith::DivSIOp>(loc, dividend, rhs);
    Value negatedQuotient =
        builder.create<arith::SubIOp>(loc, zeroCst, quotient);
    Value incrementedQuotient =
        builder.create<arith::AddIOp>(loc, quotient, oneCst);
    Value result = builder.create<arith::SelectOp>(
        loc, nonPositive, negatedQuotient, incrementedQuotient);
    return result;
  }

  Value visitConstantExpr(AffineConstantExpr expr) {
    auto op = builder.create<arith::ConstantIndexOp>(loc, expr.getValue());
    return op.getResult();
  }

  Value visitDimExpr(AffineDimExpr expr) {
    assert(expr.getPosition() < dimValues.size() &&
           "affine dim position out of range");
    return dimValues[expr.getPosition()];
  }

  Value visitSymbolExpr(AffineSymbolExpr expr) {
    assert(expr.getPosition() < symbolValues.size() &&
           "symbol dim position out of range");
    return symbolValues[expr.getPosition()];
  }

private:
  OpBuilder &builder;
  ValueRange dimValues;
  ValueRange symbolValues;

  Location loc;
};
} // namespace

/// Create a sequence of operations that implement the `expr` applied to the
/// given dimension and symbol values.
mlir::Value mlir::expandAffineExpr(OpBuilder &builder, Location loc,
                                   AffineExpr expr, ValueRange dimValues,
                                   ValueRange symbolValues) {
  return AffineApplyExpander(builder, dimValues, symbolValues, loc).visit(expr);
}

/// Create a sequence of operations that implement the `affineMap` applied to
/// the given `operands` (as it it were an AffineApplyOp).
Optional<SmallVector<Value, 8>> mlir::expandAffineMap(OpBuilder &builder,
                                                      Location loc,
                                                      AffineMap affineMap,
                                                      ValueRange operands) {
  auto numDims = affineMap.getNumDims();
  auto expanded = llvm::to_vector<8>(
      llvm::map_range(affineMap.getResults(),
                      [numDims, &builder, loc, operands](AffineExpr expr) {
                        return expandAffineExpr(builder, loc, expr,
                                                operands.take_front(numDims),
                                                operands.drop_front(numDims));
                      }));
  if (llvm::all_of(expanded, [](Value v) { return v; }))
    return expanded;
  return None;
}

/// Promotes the `then` or the `else` block of `ifOp` (depending on whether
/// `elseBlock` is false or true) into `ifOp`'s containing block, and discards
/// the rest of the op.
static void promoteIfBlock(AffineIfOp ifOp, bool elseBlock) {
  if (elseBlock)
    assert(ifOp.hasElse() && "else block expected");

  Block *destBlock = ifOp->getBlock();
  Block *srcBlock = elseBlock ? ifOp.getElseBlock() : ifOp.getThenBlock();
  destBlock->getOperations().splice(
      Block::iterator(ifOp), srcBlock->getOperations(), srcBlock->begin(),
      std::prev(srcBlock->end()));
  ifOp.erase();
}

/// Returns the outermost affine.for/parallel op that the `ifOp` is invariant
/// on. The `ifOp` could be hoisted and placed right before such an operation.
/// This method assumes that the ifOp has been canonicalized (to be correct and
/// effective).
static Operation *getOutermostInvariantForOp(AffineIfOp ifOp) {
  // Walk up the parents past all for op that this conditional is invariant on.
  auto ifOperands = ifOp.getOperands();
  auto *res = ifOp.getOperation();
  while (!isa<func::FuncOp>(res->getParentOp())) {
    auto *parentOp = res->getParentOp();
    if (auto forOp = dyn_cast<AffineForOp>(parentOp)) {
      if (llvm::is_contained(ifOperands, forOp.getInductionVar()))
        break;
    } else if (auto parallelOp = dyn_cast<AffineParallelOp>(parentOp)) {
      for (auto iv : parallelOp.getIVs())
        if (llvm::is_contained(ifOperands, iv))
          break;
    } else if (!isa<AffineIfOp>(parentOp)) {
      // Won't walk up past anything other than affine.for/if ops.
      break;
    }
    // You can always hoist up past any affine.if ops.
    res = parentOp;
  }
  return res;
}

/// A helper for the mechanics of mlir::hoistAffineIfOp. Hoists `ifOp` just over
/// `hoistOverOp`. Returns the new hoisted op if any hoisting happened,
/// otherwise the same `ifOp`.
static AffineIfOp hoistAffineIfOp(AffineIfOp ifOp, Operation *hoistOverOp) {
  // No hoisting to do.
  if (hoistOverOp == ifOp)
    return ifOp;

  // Create the hoisted 'if' first. Then, clone the op we are hoisting over for
  // the else block. Then drop the else block of the original 'if' in the 'then'
  // branch while promoting its then block, and analogously drop the 'then'
  // block of the original 'if' from the 'else' branch while promoting its else
  // block.
  BlockAndValueMapping operandMap;
  OpBuilder b(hoistOverOp);
  auto hoistedIfOp = b.create<AffineIfOp>(ifOp.getLoc(), ifOp.getIntegerSet(),
                                          ifOp.getOperands(),
                                          /*elseBlock=*/true);

  // Create a clone of hoistOverOp to use for the else branch of the hoisted
  // conditional. The else block may get optimized away if empty.
  Operation *hoistOverOpClone = nullptr;
  // We use this unique name to identify/find  `ifOp`'s clone in the else
  // version.
  StringAttr idForIfOp = b.getStringAttr("__mlir_if_hoisting");
  operandMap.clear();
  b.setInsertionPointAfter(hoistOverOp);
  // We'll set an attribute to identify this op in a clone of this sub-tree.
  ifOp->setAttr(idForIfOp, b.getBoolAttr(true));
  hoistOverOpClone = b.clone(*hoistOverOp, operandMap);

  // Promote the 'then' block of the original affine.if in the then version.
  promoteIfBlock(ifOp, /*elseBlock=*/false);

  // Move the then version to the hoisted if op's 'then' block.
  auto *thenBlock = hoistedIfOp.getThenBlock();
  thenBlock->getOperations().splice(thenBlock->begin(),
                                    hoistOverOp->getBlock()->getOperations(),
                                    Block::iterator(hoistOverOp));

  // Find the clone of the original affine.if op in the else version.
  AffineIfOp ifCloneInElse;
  hoistOverOpClone->walk([&](AffineIfOp ifClone) {
    if (!ifClone->getAttr(idForIfOp))
      return WalkResult::advance();
    ifCloneInElse = ifClone;
    return WalkResult::interrupt();
  });
  assert(ifCloneInElse && "if op clone should exist");
  // For the else block, promote the else block of the original 'if' if it had
  // one; otherwise, the op itself is to be erased.
  if (!ifCloneInElse.hasElse())
    ifCloneInElse.erase();
  else
    promoteIfBlock(ifCloneInElse, /*elseBlock=*/true);

  // Move the else version into the else block of the hoisted if op.
  auto *elseBlock = hoistedIfOp.getElseBlock();
  elseBlock->getOperations().splice(
      elseBlock->begin(), hoistOverOpClone->getBlock()->getOperations(),
      Block::iterator(hoistOverOpClone));

  return hoistedIfOp;
}

LogicalResult
mlir::affineParallelize(AffineForOp forOp,
                        ArrayRef<LoopReduction> parallelReductions) {
  // Fail early if there are iter arguments that are not reductions.
  unsigned numReductions = parallelReductions.size();
  if (numReductions != forOp.getNumIterOperands())
    return failure();

  Location loc = forOp.getLoc();
  OpBuilder outsideBuilder(forOp);
  AffineMap lowerBoundMap = forOp.getLowerBoundMap();
  ValueRange lowerBoundOperands = forOp.getLowerBoundOperands();
  AffineMap upperBoundMap = forOp.getUpperBoundMap();
  ValueRange upperBoundOperands = forOp.getUpperBoundOperands();

  // Creating empty 1-D affine.parallel op.
  auto reducedValues = llvm::to_vector<4>(llvm::map_range(
      parallelReductions, [](const LoopReduction &red) { return red.value; }));
  auto reductionKinds = llvm::to_vector<4>(llvm::map_range(
      parallelReductions, [](const LoopReduction &red) { return red.kind; }));
  AffineParallelOp newPloop = outsideBuilder.create<AffineParallelOp>(
      loc, ValueRange(reducedValues).getTypes(), reductionKinds,
      llvm::makeArrayRef(lowerBoundMap), lowerBoundOperands,
      llvm::makeArrayRef(upperBoundMap), upperBoundOperands,
      llvm::makeArrayRef(forOp.getStep()));
  // Steal the body of the old affine for op.
  newPloop.getRegion().takeBody(forOp.getRegion());
  Operation *yieldOp = &newPloop.getBody()->back();

  // Handle the initial values of reductions because the parallel loop always
  // starts from the neutral value.
  SmallVector<Value> newResults;
  newResults.reserve(numReductions);
  for (unsigned i = 0; i < numReductions; ++i) {
    Value init = forOp.getIterOperands()[i];
    // This works because we are only handling single-op reductions at the
    // moment. A switch on reduction kind or a mechanism to collect operations
    // participating in the reduction will be necessary for multi-op reductions.
    Operation *reductionOp = yieldOp->getOperand(i).getDefiningOp();
    assert(reductionOp && "yielded value is expected to be produced by an op");
    outsideBuilder.getInsertionBlock()->getOperations().splice(
        outsideBuilder.getInsertionPoint(), newPloop.getBody()->getOperations(),
        reductionOp);
    reductionOp->setOperands({init, newPloop->getResult(i)});
    forOp->getResult(i).replaceAllUsesWith(reductionOp->getResult(0));
  }

  // Update the loop terminator to yield reduced values bypassing the reduction
  // operation itself (now moved outside of the loop) and erase the block
  // arguments that correspond to reductions. Note that the loop always has one
  // "main" induction variable whenc coming from a non-parallel for.
  unsigned numIVs = 1;
  yieldOp->setOperands(reducedValues);
  newPloop.getBody()->eraseArguments(
      llvm::to_vector<4>(llvm::seq<unsigned>(numIVs, numReductions + numIVs)));

  forOp.erase();
  return success();
}

// Returns success if any hoisting happened.
LogicalResult mlir::hoistAffineIfOp(AffineIfOp ifOp, bool *folded) {
  // Bail out early if the ifOp returns a result.  TODO: Consider how to
  // properly support this case.
  if (ifOp.getNumResults() != 0)
    return failure();

  // Apply canonicalization patterns and folding - this is necessary for the
  // hoisting check to be correct (operands should be composed), and to be more
  // effective (no unused operands). Since the pattern rewriter's folding is
  // entangled with application of patterns, we may fold/end up erasing the op,
  // in which case we return with `folded` being set.
  RewritePatternSet patterns(ifOp.getContext());
  AffineIfOp::getCanonicalizationPatterns(patterns, ifOp.getContext());
  bool erased;
  FrozenRewritePatternSet frozenPatterns(std::move(patterns));
  (void)applyOpPatternsAndFold(ifOp, frozenPatterns, &erased);
  if (erased) {
    if (folded)
      *folded = true;
    return failure();
  }
  if (folded)
    *folded = false;

  // The folding above should have ensured this, but the affine.if's
  // canonicalization is missing composition of affine.applys into it.
  assert(llvm::all_of(ifOp.getOperands(),
                      [](Value v) {
                        return isTopLevelValue(v) || isForInductionVar(v);
                      }) &&
         "operands not composed");

  // We are going hoist as high as possible.
  // TODO: this could be customized in the future.
  auto *hoistOverOp = getOutermostInvariantForOp(ifOp);

  AffineIfOp hoistedIfOp = ::hoistAffineIfOp(ifOp, hoistOverOp);
  // Nothing to hoist over.
  if (hoistedIfOp == ifOp)
    return failure();

  // Canonicalize to remove dead else blocks (happens whenever an 'if' moves up
  // a sequence of affine.fors that are all perfectly nested).
  (void)applyPatternsAndFoldGreedily(
      hoistedIfOp->getParentWithTrait<OpTrait::IsIsolatedFromAbove>(),
      frozenPatterns);

  return success();
}

// Return the min expr after replacing the given dim.
AffineExpr mlir::substWithMin(AffineExpr e, AffineExpr dim, AffineExpr min,
                              AffineExpr max, bool positivePath) {
  if (e == dim)
    return positivePath ? min : max;
  if (auto bin = e.dyn_cast<AffineBinaryOpExpr>()) {
    AffineExpr lhs = bin.getLHS();
    AffineExpr rhs = bin.getRHS();
    if (bin.getKind() == mlir::AffineExprKind::Add)
      return substWithMin(lhs, dim, min, max, positivePath) +
             substWithMin(rhs, dim, min, max, positivePath);

    auto c1 = bin.getLHS().dyn_cast<AffineConstantExpr>();
    auto c2 = bin.getRHS().dyn_cast<AffineConstantExpr>();
    if (c1 && c1.getValue() < 0)
      return getAffineBinaryOpExpr(
          bin.getKind(), c1, substWithMin(rhs, dim, min, max, !positivePath));
    if (c2 && c2.getValue() < 0)
      return getAffineBinaryOpExpr(
          bin.getKind(), substWithMin(lhs, dim, min, max, !positivePath), c2);
    return getAffineBinaryOpExpr(
        bin.getKind(), substWithMin(lhs, dim, min, max, positivePath),
        substWithMin(rhs, dim, min, max, positivePath));
  }
  return e;
}

void mlir::normalizeAffineParallel(AffineParallelOp op) {
  // Loops with min/max in bounds are not normalized at the moment.
  if (op.hasMinMaxBounds())
    return;

  AffineMap lbMap = op.getLowerBoundsMap();
  SmallVector<int64_t, 8> steps = op.getSteps();
  // No need to do any work if the parallel op is already normalized.
  bool isAlreadyNormalized =
      llvm::all_of(llvm::zip(steps, lbMap.getResults()), [](auto tuple) {
        int64_t step = std::get<0>(tuple);
        auto lbExpr =
            std::get<1>(tuple).template dyn_cast<AffineConstantExpr>();
        return lbExpr && lbExpr.getValue() == 0 && step == 1;
      });
  if (isAlreadyNormalized)
    return;

  AffineValueMap ranges;
  AffineValueMap::difference(op.getUpperBoundsValueMap(),
                             op.getLowerBoundsValueMap(), &ranges);
  auto builder = OpBuilder::atBlockBegin(op.getBody());
  auto zeroExpr = builder.getAffineConstantExpr(0);
  SmallVector<AffineExpr, 8> lbExprs;
  SmallVector<AffineExpr, 8> ubExprs;
  for (unsigned i = 0, e = steps.size(); i < e; ++i) {
    int64_t step = steps[i];

    // Adjust the lower bound to be 0.
    lbExprs.push_back(zeroExpr);

    // Adjust the upper bound expression: 'range / step'.
    AffineExpr ubExpr = ranges.getResult(i).ceilDiv(step);
    ubExprs.push_back(ubExpr);

    // Adjust the corresponding IV: 'lb + i * step'.
    BlockArgument iv = op.getBody()->getArgument(i);
    AffineExpr lbExpr = lbMap.getResult(i);
    unsigned nDims = lbMap.getNumDims();
    auto expr = lbExpr + builder.getAffineDimExpr(nDims) * step;
    auto map = AffineMap::get(/*dimCount=*/nDims + 1,
                              /*symbolCount=*/lbMap.getNumSymbols(), expr);

    // Use an 'affine.apply' op that will be simplified later in subsequent
    // canonicalizations.
    OperandRange lbOperands = op.getLowerBoundsOperands();
    OperandRange dimOperands = lbOperands.take_front(nDims);
    OperandRange symbolOperands = lbOperands.drop_front(nDims);
    SmallVector<Value, 8> applyOperands{dimOperands};
    applyOperands.push_back(iv);
    applyOperands.append(symbolOperands.begin(), symbolOperands.end());
    auto apply = builder.create<AffineApplyOp>(op.getLoc(), map, applyOperands);
    iv.replaceAllUsesExcept(apply, apply);
  }

  SmallVector<int64_t, 8> newSteps(op.getNumDims(), 1);
  op.setSteps(newSteps);
  auto newLowerMap = AffineMap::get(
      /*dimCount=*/0, /*symbolCount=*/0, lbExprs, op.getContext());
  op.setLowerBounds({}, newLowerMap);
  auto newUpperMap = AffineMap::get(ranges.getNumDims(), ranges.getNumSymbols(),
                                    ubExprs, op.getContext());
  op.setUpperBounds(ranges.getOperands(), newUpperMap);
}

/// Normalizes affine.for ops. If the affine.for op has only a single iteration
/// only then it is simply promoted, else it is normalized in the traditional
/// way, by converting the lower bound to zero and loop step to one. The upper
/// bound is set to the trip count of the loop. For now, original loops must
/// have lower bound with a single result only. There is no such restriction on
/// upper bounds.
LogicalResult mlir::normalizeAffineFor(AffineForOp op) {
  if (succeeded(promoteIfSingleIteration(op)))
    return success();

  // Check if the forop is already normalized.
  if (op.hasConstantLowerBound() && (op.getConstantLowerBound() == 0) &&
      (op.getStep() == 1))
    return success();

  // Check if the lower bound has a single result only. Loops with a max lower
  // bound can't be normalized without additional support like
  // affine.execute_region's. If the lower bound does not have a single result
  // then skip this op.
  if (op.getLowerBoundMap().getNumResults() != 1)
    return failure();

  Location loc = op.getLoc();
  OpBuilder opBuilder(op);
  int64_t origLoopStep = op.getStep();

  // Calculate upperBound for normalized loop.
  SmallVector<Value, 4> ubOperands;
  AffineBound lb = op.getLowerBound();
  AffineBound ub = op.getUpperBound();
  ubOperands.reserve(ub.getNumOperands() + lb.getNumOperands());
  AffineMap origLbMap = lb.getMap();
  AffineMap origUbMap = ub.getMap();

  // Add dimension operands from upper/lower bound.
  for (unsigned j = 0, e = origUbMap.getNumDims(); j < e; ++j)
    ubOperands.push_back(ub.getOperand(j));
  for (unsigned j = 0, e = origLbMap.getNumDims(); j < e; ++j)
    ubOperands.push_back(lb.getOperand(j));

  // Add symbol operands from upper/lower bound.
  for (unsigned j = 0, e = origUbMap.getNumSymbols(); j < e; ++j)
    ubOperands.push_back(ub.getOperand(origUbMap.getNumDims() + j));
  for (unsigned j = 0, e = origLbMap.getNumSymbols(); j < e; ++j)
    ubOperands.push_back(lb.getOperand(origLbMap.getNumDims() + j));

  // Add original result expressions from lower/upper bound map.
  SmallVector<AffineExpr, 1> origLbExprs(origLbMap.getResults().begin(),
                                         origLbMap.getResults().end());
  SmallVector<AffineExpr, 2> origUbExprs(origUbMap.getResults().begin(),
                                         origUbMap.getResults().end());
  SmallVector<AffineExpr, 4> newUbExprs;

  // The original upperBound can have more than one result. For the new
  // upperBound of this loop, take difference of all possible combinations of
  // the ub results and lb result and ceildiv with the loop step. For e.g.,
  //
  //  affine.for %i1 = 0 to min affine_map<(d0)[] -> (d0 + 32, 1024)>(%i0)
  //  will have an upperBound map as,
  //  affine_map<(d0)[] -> (((d0 + 32) - 0) ceildiv 1, (1024 - 0) ceildiv
  //  1)>(%i0)
  //
  // Insert all combinations of upper/lower bound results.
  for (unsigned i = 0, e = origUbExprs.size(); i < e; ++i) {
    newUbExprs.push_back(
        (origUbExprs[i] - origLbExprs[0]).ceilDiv(origLoopStep));
  }

  // Construct newUbMap.
  AffineMap newUbMap =
      AffineMap::get(origLbMap.getNumDims() + origUbMap.getNumDims(),
                     origLbMap.getNumSymbols() + origUbMap.getNumSymbols(),
                     newUbExprs, opBuilder.getContext());
  canonicalizeMapAndOperands(&newUbMap, &ubOperands);

  SmallVector<Value, 4> lbOperands(lb.getOperands().begin(),
                                   lb.getOperands().begin() +
                                       lb.getMap().getNumDims());

  // Normalize the loop.
  op.setUpperBound(ubOperands, newUbMap);
  op.setLowerBound({}, opBuilder.getConstantAffineMap(0));
  op.setStep(1);

  // Calculate the Value of new loopIV. Create affine.apply for the value of
  // the loopIV in normalized loop.
  opBuilder.setInsertionPointToStart(op.getBody());
  // Add an extra dim operand for loopIV.
  lbOperands.push_back(op.getInductionVar());
  // Add symbol operands from lower bound.
  for (unsigned j = 0, e = origLbMap.getNumSymbols(); j < e; ++j)
    lbOperands.push_back(lb.getOperand(origLbMap.getNumDims() + j));

  AffineExpr origIVExpr = opBuilder.getAffineDimExpr(lb.getMap().getNumDims());
  AffineExpr newIVExpr = origIVExpr * origLoopStep + origLbMap.getResult(0);
  AffineMap ivMap = AffineMap::get(origLbMap.getNumDims() + 1,
                                   origLbMap.getNumSymbols(), newIVExpr);
  canonicalizeMapAndOperands(&ivMap, &lbOperands);
  Operation *newIV = opBuilder.create<AffineApplyOp>(loc, ivMap, lbOperands);
  op.getInductionVar().replaceAllUsesExcept(newIV->getResult(0), newIV);
  return success();
}

/// Ensure that all operations that could be executed after `start`
/// (noninclusive) and prior to `memOp` (e.g. on a control flow/op path
/// between the operations) do not have the potential memory effect
/// `EffectType` on `memOp`. `memOp`  is an operation that reads or writes to
/// a memref. For example, if `EffectType` is MemoryEffects::Write, this method
/// will check if there is no write to the memory between `start` and `memOp`
/// that would change the read within `memOp`.
template <typename EffectType, typename T>
static bool hasNoInterveningEffect(Operation *start, T memOp) {
  Value memref = memOp.getMemRef();
  bool isOriginalAllocation = memref.getDefiningOp<memref::AllocaOp>() ||
                              memref.getDefiningOp<memref::AllocOp>();

  // A boolean representing whether an intervening operation could have impacted
  // memOp.
  bool hasSideEffect = false;

  // Check whether the effect on memOp can be caused by a given operation op.
  std::function<void(Operation *)> checkOperation = [&](Operation *op) {
    // If the effect has alreay been found, early exit,
    if (hasSideEffect)
      return;

    if (auto memEffect = dyn_cast<MemoryEffectOpInterface>(op)) {
      SmallVector<MemoryEffects::EffectInstance, 1> effects;
      memEffect.getEffects(effects);

      bool opMayHaveEffect = false;
      for (auto effect : effects) {
        // If op causes EffectType on a potentially aliasing location for
        // memOp, mark as having the effect.
        if (isa<EffectType>(effect.getEffect())) {
          if (isOriginalAllocation && effect.getValue() &&
              (effect.getValue().getDefiningOp<memref::AllocaOp>() ||
               effect.getValue().getDefiningOp<memref::AllocOp>())) {
            if (effect.getValue() != memref)
              continue;
          }
          opMayHaveEffect = true;
          break;
        }
      }

      if (!opMayHaveEffect)
        return;

      // If the side effect comes from an affine read or write, try to
      // prove the side effecting `op` cannot reach `memOp`.
      if (isa<AffineReadOpInterface, AffineWriteOpInterface>(op)) {
        MemRefAccess srcAccess(op);
        MemRefAccess destAccess(memOp);
        // Dependence analysis is only correct if both ops operate on the same
        // memref.
        if (srcAccess.memref == destAccess.memref) {
          FlatAffineValueConstraints dependenceConstraints;

          // Number of loops containing the start op and the ending operation.
          unsigned minSurroundingLoops =
              getNumCommonSurroundingLoops(*start, *memOp);

          // Number of loops containing the operation `op` which has the
          // potential memory side effect and can occur on a path between
          // `start` and `memOp`.
          unsigned nsLoops = getNumCommonSurroundingLoops(*op, *memOp);

          // For ease, let's consider the case that `op` is a store and we're
          // looking for other potential stores (e.g `op`) that overwrite memory
          // after `start`, and before being read in `memOp`. In this case, we
          // only need to consider other potential stores with depth >
          // minSurrounding loops since `start` would overwrite any store with a
          // smaller number of surrounding loops before.
          unsigned d;
          for (d = nsLoops + 1; d > minSurroundingLoops; d--) {
            DependenceResult result = checkMemrefAccessDependence(
                srcAccess, destAccess, d, &dependenceConstraints,
                /*dependenceComponents=*/nullptr);
            if (hasDependence(result)) {
              hasSideEffect = true;
              return;
            }
          }

          // No side effect was seen, simply return.
          return;
        }
      }
      hasSideEffect = true;
      return;
    }

    if (op->hasTrait<OpTrait::HasRecursiveSideEffects>()) {
      // Recurse into the regions for this op and check whether the internal
      // operations may have the side effect `EffectType` on memOp.
      for (Region &region : op->getRegions())
        for (Block &block : region)
          for (Operation &op : block)
            checkOperation(&op);
      return;
    }

    // Otherwise, conservatively assume generic operations have the effect
    // on the operation
    hasSideEffect = true;
  };

  // Check all paths from ancestor op `parent` to the operation `to` for the
  // effect. It is known that `to` must be contained within `parent`.
  auto until = [&](Operation *parent, Operation *to) {
    // TODO check only the paths from `parent` to `to`.
    // Currently we fallback and check the entire parent op, rather than
    // just the paths from the parent path, stopping after reaching `to`.
    // This is conservatively correct, but could be made more aggressive.
    assert(parent->isAncestor(to));
    checkOperation(parent);
  };

  // Check for all paths from operation `from` to operation `untilOp` for the
  // given memory effect.
  std::function<void(Operation *, Operation *)> recur =
      [&](Operation *from, Operation *untilOp) {
        assert(
            from->getParentRegion()->isAncestor(untilOp->getParentRegion()) &&
            "Checking for side effect between two operations without a common "
            "ancestor");

        // If the operations are in different regions, recursively consider all
        // path from `from` to the parent of `to` and all paths from the parent
        // of `to` to `to`.
        if (from->getParentRegion() != untilOp->getParentRegion()) {
          recur(from, untilOp->getParentOp());
          until(untilOp->getParentOp(), untilOp);
          return;
        }

        // Now, assuming that `from` and `to` exist in the same region, perform
        // a CFG traversal to check all the relevant operations.

        // Additional blocks to consider.
        SmallVector<Block *, 2> todoBlocks;
        {
          // First consider the parent block of `from` an check all operations
          // after `from`.
          for (auto iter = ++from->getIterator(), end = from->getBlock()->end();
               iter != end && &*iter != untilOp; ++iter) {
            checkOperation(&*iter);
          }

          // If the parent of `from` doesn't contain `to`, add the successors
          // to the list of blocks to check.
          if (untilOp->getBlock() != from->getBlock())
            for (Block *succ : from->getBlock()->getSuccessors())
              todoBlocks.push_back(succ);
        }

        SmallPtrSet<Block *, 4> done;
        // Traverse the CFG until hitting `to`.
        while (!todoBlocks.empty()) {
          Block *blk = todoBlocks.pop_back_val();
          if (done.count(blk))
            continue;
          done.insert(blk);
          for (auto &op : *blk) {
            if (&op == untilOp)
              break;
            checkOperation(&op);
            if (&op == blk->getTerminator())
              for (Block *succ : blk->getSuccessors())
                todoBlocks.push_back(succ);
          }
        }
      };
  recur(start, memOp);
  return !hasSideEffect;
}

/// Attempt to eliminate loadOp by replacing it with a value stored into memory
/// which the load is guaranteed to retrieve. This check involves three
/// components: 1) The store and load must be on the same location 2) The store
/// must dominate (and therefore must always occur prior to) the load 3) No
/// other operations will overwrite the memory loaded between the given load
/// and store.  If such a value exists, the replaced `loadOp` will be added to
/// `loadOpsToErase` and its memref will be added to `memrefsToErase`.
static LogicalResult forwardStoreToLoad(
    AffineReadOpInterface loadOp, SmallVectorImpl<Operation *> &loadOpsToErase,
    SmallPtrSetImpl<Value> &memrefsToErase, DominanceInfo &domInfo) {

  // The store op candidate for forwarding that satisfies all conditions
  // to replace the load, if any.
  Operation *lastWriteStoreOp = nullptr;

  for (auto *user : loadOp.getMemRef().getUsers()) {
    auto storeOp = dyn_cast<AffineWriteOpInterface>(user);
    if (!storeOp)
      continue;
    MemRefAccess srcAccess(storeOp);
    MemRefAccess destAccess(loadOp);

    // 1. Check if the store and the load have mathematically equivalent
    // affine access functions; this implies that they statically refer to the
    // same single memref element. As an example this filters out cases like:
    //     store %A[%i0 + 1]
    //     load %A[%i0]
    //     store %A[%M]
    //     load %A[%N]
    // Use the AffineValueMap difference based memref access equality checking.
    if (srcAccess != destAccess)
      continue;

    // 2. The store has to dominate the load op to be candidate.
    if (!domInfo.dominates(storeOp, loadOp))
      continue;

    // 3. Ensure there is no intermediate operation which could replace the
    // value in memory.
    if (!hasNoInterveningEffect<MemoryEffects::Write>(storeOp, loadOp))
      continue;

    // We now have a candidate for forwarding.
    assert(lastWriteStoreOp == nullptr &&
           "multiple simulataneous replacement stores");
    lastWriteStoreOp = storeOp;
  }

  if (!lastWriteStoreOp)
    return failure();

  // Perform the actual store to load forwarding.
  Value storeVal =
      cast<AffineWriteOpInterface>(lastWriteStoreOp).getValueToStore();
  // Check if 2 values have the same shape. This is needed for affine vector
  // loads and stores.
  if (storeVal.getType() != loadOp.getValue().getType())
    return failure();
  loadOp.getValue().replaceAllUsesWith(storeVal);
  // Record the memref for a later sweep to optimize away.
  memrefsToErase.insert(loadOp.getMemRef());
  // Record this to erase later.
  loadOpsToErase.push_back(loadOp);
  return success();
}

// This attempts to find stores which have no impact on the final result.
// A writing op writeA will be eliminated if there exists an op writeB if
// 1) writeA and writeB have mathematically equivalent affine access functions.
// 2) writeB postdominates writeA.
// 3) There is no potential read between writeA and writeB.
static void findUnusedStore(AffineWriteOpInterface writeA,
                            SmallVectorImpl<Operation *> &opsToErase,
                            PostDominanceInfo &postDominanceInfo) {

  for (Operation *user : writeA.getMemRef().getUsers()) {
    // Only consider writing operations.
    auto writeB = dyn_cast<AffineWriteOpInterface>(user);
    if (!writeB)
      continue;

    // The operations must be distinct.
    if (writeB == writeA)
      continue;

    // Both operations must lie in the same region.
    if (writeB->getParentRegion() != writeA->getParentRegion())
      continue;

    // Both operations must write to the same memory.
    MemRefAccess srcAccess(writeB);
    MemRefAccess destAccess(writeA);

    if (srcAccess != destAccess)
      continue;

    // writeB must postdominate writeA.
    if (!postDominanceInfo.postDominates(writeB, writeA))
      continue;

    // There cannot be an operation which reads from memory between
    // the two writes.
    if (!hasNoInterveningEffect<MemoryEffects::Read>(writeA, writeB))
      continue;

    opsToErase.push_back(writeA);
    break;
  }
}

// The load to load forwarding / redundant load elimination is similar to the
// store to load forwarding.
// loadA will be be replaced with loadB if:
// 1) loadA and loadB have mathematically equivalent affine access functions.
// 2) loadB dominates loadA.
// 3) There is no write between loadA and loadB.
static void loadCSE(AffineReadOpInterface loadA,
                    SmallVectorImpl<Operation *> &loadOpsToErase,
                    DominanceInfo &domInfo) {
  SmallVector<AffineReadOpInterface, 4> loadCandidates;
  for (auto *user : loadA.getMemRef().getUsers()) {
    auto loadB = dyn_cast<AffineReadOpInterface>(user);
    if (!loadB || loadB == loadA)
      continue;

    MemRefAccess srcAccess(loadB);
    MemRefAccess destAccess(loadA);

    // 1. The accesses have to be to the same location.
    if (srcAccess != destAccess) {
      continue;
    }

    // 2. The store has to dominate the load op to be candidate.
    if (!domInfo.dominates(loadB, loadA))
      continue;

    // 3. There is no write between loadA and loadB.
    if (!hasNoInterveningEffect<MemoryEffects::Write>(loadB.getOperation(),
                                                      loadA))
      continue;

    // Check if two values have the same shape. This is needed for affine vector
    // loads.
    if (loadB.getValue().getType() != loadA.getValue().getType())
      continue;

    loadCandidates.push_back(loadB);
  }

  // Of the legal load candidates, use the one that dominates all others
  // to minimize the subsequent need to loadCSE
  Value loadB;
  for (AffineReadOpInterface option : loadCandidates) {
    if (llvm::all_of(loadCandidates, [&](AffineReadOpInterface depStore) {
          return depStore == option ||
                 domInfo.dominates(option.getOperation(),
                                   depStore.getOperation());
        })) {
      loadB = option.getValue();
      break;
    }
  }

  if (loadB) {
    loadA.getValue().replaceAllUsesWith(loadB);
    // Record this to erase later.
    loadOpsToErase.push_back(loadA);
  }
}

// The store to load forwarding and load CSE rely on three conditions:
//
// 1) store/load providing a replacement value and load being replaced need to
// have mathematically equivalent affine access functions (checked after full
// composition of load/store operands); this implies that they access the same
// single memref element for all iterations of the common surrounding loop,
//
// 2) the store/load op should dominate the load op,
//
// 3) no operation that may write to memory read by the load being replaced can
// occur after executing the instruction (load or store) providing the
// replacement value and before the load being replaced (thus potentially
// allowing overwriting the memory read by the load).
//
// The above conditions are simple to check, sufficient, and powerful for most
// cases in practice - they are sufficient, but not necessary --- since they
// don't reason about loops that are guaranteed to execute at least once or
// multiple sources to forward from.
//
// TODO: more forwarding can be done when support for
// loop/conditional live-out SSA values is available.
// TODO: do general dead store elimination for memref's. This pass
// currently only eliminates the stores only if no other loads/uses (other
// than dealloc) remain.
//
void mlir::affineScalarReplace(func::FuncOp f, DominanceInfo &domInfo,
                               PostDominanceInfo &postDomInfo) {
  // Load op's whose results were replaced by those forwarded from stores.
  SmallVector<Operation *, 8> opsToErase;

  // A list of memref's that are potentially dead / could be eliminated.
  SmallPtrSet<Value, 4> memrefsToErase;

  // Walk all load's and perform store to load forwarding.
  f.walk([&](AffineReadOpInterface loadOp) {
    if (failed(
            forwardStoreToLoad(loadOp, opsToErase, memrefsToErase, domInfo))) {
      loadCSE(loadOp, opsToErase, domInfo);
    }
  });

  // Erase all load op's whose results were replaced with store fwd'ed ones.
  for (auto *op : opsToErase)
    op->erase();
  opsToErase.clear();

  // Walk all store's and perform unused store elimination
  f.walk([&](AffineWriteOpInterface storeOp) {
    findUnusedStore(storeOp, opsToErase, postDomInfo);
  });
  // Erase all store op's which don't impact the program
  for (auto *op : opsToErase)
    op->erase();

  // Check if the store fwd'ed memrefs are now left with only stores and can
  // thus be completely deleted. Note: the canonicalize pass should be able
  // to do this as well, but we'll do it here since we collected these anyway.
  for (auto memref : memrefsToErase) {
    // If the memref hasn't been alloc'ed in this function, skip.
    Operation *defOp = memref.getDefiningOp();
    if (!defOp || !isa<memref::AllocOp>(defOp))
      // TODO: if the memref was returned by a 'call' operation, we
      // could still erase it if the call had no side-effects.
      continue;
    if (llvm::any_of(memref.getUsers(), [&](Operation *ownerOp) {
          return !isa<AffineWriteOpInterface, memref::DeallocOp>(ownerOp);
        }))
      continue;

    // Erase all stores, the dealloc, and the alloc on the memref.
    for (auto *user : llvm::make_early_inc_range(memref.getUsers()))
      user->erase();
    defOp->erase();
  }
}

// Perform the replacement in `op`.
LogicalResult mlir::replaceAllMemRefUsesWith(Value oldMemRef, Value newMemRef,
                                             Operation *op,
                                             ArrayRef<Value> extraIndices,
                                             AffineMap indexRemap,
                                             ArrayRef<Value> extraOperands,
                                             ArrayRef<Value> symbolOperands,
                                             bool allowNonDereferencingOps) {
  unsigned newMemRefRank = newMemRef.getType().cast<MemRefType>().getRank();
  (void)newMemRefRank; // unused in opt mode
  unsigned oldMemRefRank = oldMemRef.getType().cast<MemRefType>().getRank();
  (void)oldMemRefRank; // unused in opt mode
  if (indexRemap) {
    assert(indexRemap.getNumSymbols() == symbolOperands.size() &&
           "symbolic operand count mismatch");
    assert(indexRemap.getNumInputs() ==
           extraOperands.size() + oldMemRefRank + symbolOperands.size());
    assert(indexRemap.getNumResults() + extraIndices.size() == newMemRefRank);
  } else {
    assert(oldMemRefRank + extraIndices.size() == newMemRefRank);
  }

  // Assert same elemental type.
  assert(oldMemRef.getType().cast<MemRefType>().getElementType() ==
         newMemRef.getType().cast<MemRefType>().getElementType());

  SmallVector<unsigned, 2> usePositions;
  for (const auto &opEntry : llvm::enumerate(op->getOperands())) {
    if (opEntry.value() == oldMemRef)
      usePositions.push_back(opEntry.index());
  }

  // If memref doesn't appear, nothing to do.
  if (usePositions.empty())
    return success();

  if (usePositions.size() > 1) {
    // TODO: extend it for this case when needed (rare).
    assert(false && "multiple dereferencing uses in a single op not supported");
    return failure();
  }

  unsigned memRefOperandPos = usePositions.front();

  OpBuilder builder(op);
  // The following checks if op is dereferencing memref and performs the access
  // index rewrites.
  auto affMapAccInterface = dyn_cast<AffineMapAccessInterface>(op);
  if (!affMapAccInterface) {
    if (!allowNonDereferencingOps) {
      // Failure: memref used in a non-dereferencing context (potentially
      // escapes); no replacement in these cases unless allowNonDereferencingOps
      // is set.
      return failure();
    }
    op->setOperand(memRefOperandPos, newMemRef);
    return success();
  }
  // Perform index rewrites for the dereferencing op and then replace the op
  NamedAttribute oldMapAttrPair =
      affMapAccInterface.getAffineMapAttrForMemRef(oldMemRef);
  AffineMap oldMap = oldMapAttrPair.getValue().cast<AffineMapAttr>().getValue();
  unsigned oldMapNumInputs = oldMap.getNumInputs();
  SmallVector<Value, 4> oldMapOperands(
      op->operand_begin() + memRefOperandPos + 1,
      op->operand_begin() + memRefOperandPos + 1 + oldMapNumInputs);

  // Apply 'oldMemRefOperands = oldMap(oldMapOperands)'.
  SmallVector<Value, 4> oldMemRefOperands;
  SmallVector<Value, 4> affineApplyOps;
  oldMemRefOperands.reserve(oldMemRefRank);
  if (oldMap != builder.getMultiDimIdentityMap(oldMap.getNumDims())) {
    for (auto resultExpr : oldMap.getResults()) {
      auto singleResMap = AffineMap::get(oldMap.getNumDims(),
                                         oldMap.getNumSymbols(), resultExpr);
      auto afOp = builder.create<AffineApplyOp>(op->getLoc(), singleResMap,
                                                oldMapOperands);
      oldMemRefOperands.push_back(afOp);
      affineApplyOps.push_back(afOp);
    }
  } else {
    oldMemRefOperands.assign(oldMapOperands.begin(), oldMapOperands.end());
  }

  // Construct new indices as a remap of the old ones if a remapping has been
  // provided. The indices of a memref come right after it, i.e.,
  // at position memRefOperandPos + 1.
  SmallVector<Value, 4> remapOperands;
  remapOperands.reserve(extraOperands.size() + oldMemRefRank +
                        symbolOperands.size());
  remapOperands.append(extraOperands.begin(), extraOperands.end());
  remapOperands.append(oldMemRefOperands.begin(), oldMemRefOperands.end());
  remapOperands.append(symbolOperands.begin(), symbolOperands.end());

  SmallVector<Value, 4> remapOutputs;
  remapOutputs.reserve(oldMemRefRank);

  if (indexRemap &&
      indexRemap != builder.getMultiDimIdentityMap(indexRemap.getNumDims())) {
    // Remapped indices.
    for (auto resultExpr : indexRemap.getResults()) {
      auto singleResMap = AffineMap::get(
          indexRemap.getNumDims(), indexRemap.getNumSymbols(), resultExpr);
      auto afOp = builder.create<AffineApplyOp>(op->getLoc(), singleResMap,
                                                remapOperands);
      remapOutputs.push_back(afOp);
      affineApplyOps.push_back(afOp);
    }
  } else {
    // No remapping specified.
    remapOutputs.assign(remapOperands.begin(), remapOperands.end());
  }

  SmallVector<Value, 4> newMapOperands;
  newMapOperands.reserve(newMemRefRank);

  // Prepend 'extraIndices' in 'newMapOperands'.
  for (Value extraIndex : extraIndices) {
    assert(extraIndex.getDefiningOp()->getNumResults() == 1 &&
           "single result op's expected to generate these indices");
    assert((isValidDim(extraIndex) || isValidSymbol(extraIndex)) &&
           "invalid memory op index");
    newMapOperands.push_back(extraIndex);
  }

  // Append 'remapOutputs' to 'newMapOperands'.
  newMapOperands.append(remapOutputs.begin(), remapOutputs.end());

  // Create new fully composed AffineMap for new op to be created.
  assert(newMapOperands.size() == newMemRefRank);
  auto newMap = builder.getMultiDimIdentityMap(newMemRefRank);
  // TODO: Avoid creating/deleting temporary AffineApplyOps here.
  fullyComposeAffineMapAndOperands(&newMap, &newMapOperands);
  newMap = simplifyAffineMap(newMap);
  canonicalizeMapAndOperands(&newMap, &newMapOperands);
  // Remove any affine.apply's that became dead as a result of composition.
  for (Value value : affineApplyOps)
    if (value.use_empty())
      value.getDefiningOp()->erase();

  OperationState state(op->getLoc(), op->getName());
  // Construct the new operation using this memref.
  state.operands.reserve(op->getNumOperands() + extraIndices.size());
  // Insert the non-memref operands.
  state.operands.append(op->operand_begin(),
                        op->operand_begin() + memRefOperandPos);
  // Insert the new memref value.
  state.operands.push_back(newMemRef);

  // Insert the new memref map operands.
  state.operands.append(newMapOperands.begin(), newMapOperands.end());

  // Insert the remaining operands unmodified.
  state.operands.append(op->operand_begin() + memRefOperandPos + 1 +
                            oldMapNumInputs,
                        op->operand_end());

  // Result types don't change. Both memref's are of the same elemental type.
  state.types.reserve(op->getNumResults());
  for (auto result : op->getResults())
    state.types.push_back(result.getType());

  // Add attribute for 'newMap', other Attributes do not change.
  auto newMapAttr = AffineMapAttr::get(newMap);
  for (auto namedAttr : op->getAttrs()) {
    if (namedAttr.getName() == oldMapAttrPair.getName())
      state.attributes.push_back({namedAttr.getName(), newMapAttr});
    else
      state.attributes.push_back(namedAttr);
  }

  // Create the new operation.
  auto *repOp = builder.create(state);
  op->replaceAllUsesWith(repOp);
  op->erase();

  return success();
}

LogicalResult mlir::replaceAllMemRefUsesWith(
    Value oldMemRef, Value newMemRef, ArrayRef<Value> extraIndices,
    AffineMap indexRemap, ArrayRef<Value> extraOperands,
    ArrayRef<Value> symbolOperands, Operation *domOpFilter,
    Operation *postDomOpFilter, bool allowNonDereferencingOps,
    bool replaceInDeallocOp) {
  unsigned newMemRefRank = newMemRef.getType().cast<MemRefType>().getRank();
  (void)newMemRefRank; // unused in opt mode
  unsigned oldMemRefRank = oldMemRef.getType().cast<MemRefType>().getRank();
  (void)oldMemRefRank;
  if (indexRemap) {
    assert(indexRemap.getNumSymbols() == symbolOperands.size() &&
           "symbol operand count mismatch");
    assert(indexRemap.getNumInputs() ==
           extraOperands.size() + oldMemRefRank + symbolOperands.size());
    assert(indexRemap.getNumResults() + extraIndices.size() == newMemRefRank);
  } else {
    assert(oldMemRefRank + extraIndices.size() == newMemRefRank);
  }

  // Assert same elemental type.
  assert(oldMemRef.getType().cast<MemRefType>().getElementType() ==
         newMemRef.getType().cast<MemRefType>().getElementType());

  std::unique_ptr<DominanceInfo> domInfo;
  std::unique_ptr<PostDominanceInfo> postDomInfo;
  if (domOpFilter)
    domInfo = std::make_unique<DominanceInfo>(
        domOpFilter->getParentOfType<func::FuncOp>());

  if (postDomOpFilter)
    postDomInfo = std::make_unique<PostDominanceInfo>(
        postDomOpFilter->getParentOfType<func::FuncOp>());

  // Walk all uses of old memref; collect ops to perform replacement. We use a
  // DenseSet since an operation could potentially have multiple uses of a
  // memref (although rare), and the replacement later is going to erase ops.
  DenseSet<Operation *> opsToReplace;
  for (auto *op : oldMemRef.getUsers()) {
    // Skip this use if it's not dominated by domOpFilter.
    if (domOpFilter && !domInfo->dominates(domOpFilter, op))
      continue;

    // Skip this use if it's not post-dominated by postDomOpFilter.
    if (postDomOpFilter && !postDomInfo->postDominates(postDomOpFilter, op))
      continue;

    // Skip dealloc's - no replacement is necessary, and a memref replacement
    // at other uses doesn't hurt these dealloc's.
    if (isa<memref::DeallocOp>(op) && !replaceInDeallocOp)
      continue;

    // Check if the memref was used in a non-dereferencing context. It is fine
    // for the memref to be used in a non-dereferencing way outside of the
    // region where this replacement is happening.
    if (!isa<AffineMapAccessInterface>(*op)) {
      if (!allowNonDereferencingOps) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Memref replacement failed: non-deferencing memref op: \n"
                   << *op << '\n');
        return failure();
      }
      // Non-dereferencing ops with the MemRefsNormalizable trait are
      // supported for replacement.
      if (!op->hasTrait<OpTrait::MemRefsNormalizable>()) {
        LLVM_DEBUG(llvm::dbgs() << "Memref replacement failed: use without a "
                                   "memrefs normalizable trait: \n"
                                << *op << '\n');
        return failure();
      }
    }

    // We'll first collect and then replace --- since replacement erases the op
    // that has the use, and that op could be postDomFilter or domFilter itself!
    opsToReplace.insert(op);
  }

  for (auto *op : opsToReplace) {
    if (failed(replaceAllMemRefUsesWith(
            oldMemRef, newMemRef, op, extraIndices, indexRemap, extraOperands,
            symbolOperands, allowNonDereferencingOps)))
      llvm_unreachable("memref replacement guaranteed to succeed here");
  }

  return success();
}

/// Given an operation, inserts one or more single result affine
/// apply operations, results of which are exclusively used by this operation
/// operation. The operands of these newly created affine apply ops are
/// guaranteed to be loop iterators or terminal symbols of a function.
///
/// Before
///
/// affine.for %i = 0 to #map(%N)
///   %idx = affine.apply (d0) -> (d0 mod 2) (%i)
///   "send"(%idx, %A, ...)
///   "compute"(%idx)
///
/// After
///
/// affine.for %i = 0 to #map(%N)
///   %idx = affine.apply (d0) -> (d0 mod 2) (%i)
///   "send"(%idx, %A, ...)
///   %idx_ = affine.apply (d0) -> (d0 mod 2) (%i)
///   "compute"(%idx_)
///
/// This allows applying different transformations on send and compute (for eg.
/// different shifts/delays).
///
/// Returns nullptr either if none of opInst's operands were the result of an
/// affine.apply and thus there was no affine computation slice to create, or if
/// all the affine.apply op's supplying operands to this opInst did not have any
/// uses besides this opInst; otherwise returns the list of affine.apply
/// operations created in output argument `sliceOps`.
void mlir::createAffineComputationSlice(
    Operation *opInst, SmallVectorImpl<AffineApplyOp> *sliceOps) {
  // Collect all operands that are results of affine apply ops.
  SmallVector<Value, 4> subOperands;
  subOperands.reserve(opInst->getNumOperands());
  for (auto operand : opInst->getOperands())
    if (isa_and_nonnull<AffineApplyOp>(operand.getDefiningOp()))
      subOperands.push_back(operand);

  // Gather sequence of AffineApplyOps reachable from 'subOperands'.
  SmallVector<Operation *, 4> affineApplyOps;
  getReachableAffineApplyOps(subOperands, affineApplyOps);
  // Skip transforming if there are no affine maps to compose.
  if (affineApplyOps.empty())
    return;

  // Check if all uses of the affine apply op's lie only in this op op, in
  // which case there would be nothing to do.
  bool localized = true;
  for (auto *op : affineApplyOps) {
    for (auto result : op->getResults()) {
      for (auto *user : result.getUsers()) {
        if (user != opInst) {
          localized = false;
          break;
        }
      }
    }
  }
  if (localized)
    return;

  OpBuilder builder(opInst);
  SmallVector<Value, 4> composedOpOperands(subOperands);
  auto composedMap = builder.getMultiDimIdentityMap(composedOpOperands.size());
  fullyComposeAffineMapAndOperands(&composedMap, &composedOpOperands);

  // Create an affine.apply for each of the map results.
  sliceOps->reserve(composedMap.getNumResults());
  for (auto resultExpr : composedMap.getResults()) {
    auto singleResMap = AffineMap::get(composedMap.getNumDims(),
                                       composedMap.getNumSymbols(), resultExpr);
    sliceOps->push_back(builder.create<AffineApplyOp>(
        opInst->getLoc(), singleResMap, composedOpOperands));
  }

  // Construct the new operands that include the results from the composed
  // affine apply op above instead of existing ones (subOperands). So, they
  // differ from opInst's operands only for those operands in 'subOperands', for
  // which they will be replaced by the corresponding one from 'sliceOps'.
  SmallVector<Value, 4> newOperands(opInst->getOperands());
  for (unsigned i = 0, e = newOperands.size(); i < e; i++) {
    // Replace the subOperands from among the new operands.
    unsigned j, f;
    for (j = 0, f = subOperands.size(); j < f; j++) {
      if (newOperands[i] == subOperands[j])
        break;
    }
    if (j < subOperands.size()) {
      newOperands[i] = (*sliceOps)[j];
    }
  }
  for (unsigned idx = 0, e = newOperands.size(); idx < e; idx++) {
    opInst->setOperand(idx, newOperands[idx]);
  }
}

/// Enum to set patterns of affine expr in tiled-layout map.
/// TileFloorDiv: <dim expr> div <tile size>
/// TileMod: <dim expr> mod <tile size>
/// TileNone: None of the above
/// Example:
/// #tiled_2d_128x256 = affine_map<(d0, d1)
///            -> (d0 div 128, d1 div 256, d0 mod 128, d1 mod 256)>
/// "d0 div 128" and "d1 div 256" ==> TileFloorDiv
/// "d0 mod 128" and "d1 mod 256" ==> TileMod
enum TileExprPattern { TileFloorDiv, TileMod, TileNone };

/// Check if `map` is a tiled layout. In the tiled layout, specific k dimensions
/// being floordiv'ed by respective tile sizes appeare in a mod with the same
/// tile sizes, and no other expression involves those k dimensions. This
/// function stores a vector of tuples (`tileSizePos`) including AffineExpr for
/// tile size, positions of corresponding `floordiv` and `mod`. If it is not a
/// tiled layout, an empty vector is returned.
static LogicalResult getTileSizePos(
    AffineMap map,
    SmallVectorImpl<std::tuple<AffineExpr, unsigned, unsigned>> &tileSizePos) {
  // Create `floordivExprs` which is a vector of tuples including LHS and RHS of
  // `floordiv` and its position in `map` output.
  // Example: #tiled_2d_128x256 = affine_map<(d0, d1)
  //                -> (d0 div 128, d1 div 256, d0 mod 128, d1 mod 256)>
  // In this example, `floordivExprs` includes {d0, 128, 0} and {d1, 256, 1}.
  SmallVector<std::tuple<AffineExpr, AffineExpr, unsigned>, 4> floordivExprs;
  unsigned pos = 0;
  for (AffineExpr expr : map.getResults()) {
    if (expr.getKind() == AffineExprKind::FloorDiv) {
      AffineBinaryOpExpr binaryExpr = expr.cast<AffineBinaryOpExpr>();
      if (binaryExpr.getRHS().isa<AffineConstantExpr>())
        floordivExprs.emplace_back(
            std::make_tuple(binaryExpr.getLHS(), binaryExpr.getRHS(), pos));
    }
    pos++;
  }
  // Not tiled layout if `floordivExprs` is empty.
  if (floordivExprs.empty()) {
    tileSizePos = SmallVector<std::tuple<AffineExpr, unsigned, unsigned>>{};
    return success();
  }

  // Check if LHS of `floordiv` is used in LHS of `mod`. If not used, `map` is
  // not tiled layout.
  for (std::tuple<AffineExpr, AffineExpr, unsigned> fexpr : floordivExprs) {
    AffineExpr floordivExprLHS = std::get<0>(fexpr);
    AffineExpr floordivExprRHS = std::get<1>(fexpr);
    unsigned floordivPos = std::get<2>(fexpr);

    // Walk affinexpr of `map` output except `fexpr`, and check if LHS and RHS
    // of `fexpr` are used in LHS and RHS of `mod`. If LHS of `fexpr` is used
    // other expr, the map is not tiled layout. Example of non tiled layout:
    //   affine_map<(d0, d1, d2) -> (d0, d1, d2 floordiv 256, d2 floordiv 256)>
    //   affine_map<(d0, d1, d2) -> (d0, d1, d2 floordiv 256, d2 mod 128)>
    //   affine_map<(d0, d1, d2) -> (d0, d1, d2 floordiv 256, d2 mod 256, d2 mod
    //   256)>
    bool found = false;
    pos = 0;
    for (AffineExpr expr : map.getResults()) {
      bool notTiled = false;
      if (pos != floordivPos) {
        expr.walk([&](AffineExpr e) {
          if (e == floordivExprLHS) {
            if (expr.getKind() == AffineExprKind::Mod) {
              AffineBinaryOpExpr binaryExpr = expr.cast<AffineBinaryOpExpr>();
              // If LHS and RHS of `mod` are the same with those of floordiv.
              if (floordivExprLHS == binaryExpr.getLHS() &&
                  floordivExprRHS == binaryExpr.getRHS()) {
                // Save tile size (RHS of `mod`), and position of `floordiv` and
                // `mod` if same expr with `mod` is not found yet.
                if (!found) {
                  tileSizePos.emplace_back(
                      std::make_tuple(binaryExpr.getRHS(), floordivPos, pos));
                  found = true;
                } else {
                  // Non tiled layout: Have multilpe `mod` with the same LHS.
                  // eg. affine_map<(d0, d1, d2) -> (d0, d1, d2 floordiv 256, d2
                  // mod 256, d2 mod 256)>
                  notTiled = true;
                }
              } else {
                // Non tiled layout: RHS of `mod` is different from `floordiv`.
                // eg. affine_map<(d0, d1, d2) -> (d0, d1, d2 floordiv 256, d2
                // mod 128)>
                notTiled = true;
              }
            } else {
              // Non tiled layout: LHS is the same, but not `mod`.
              // eg. affine_map<(d0, d1, d2) -> (d0, d1, d2 floordiv 256, d2
              // floordiv 256)>
              notTiled = true;
            }
          }
        });
      }
      if (notTiled) {
        tileSizePos = SmallVector<std::tuple<AffineExpr, unsigned, unsigned>>{};
        return success();
      }
      pos++;
    }
  }
  return success();
}

/// Check if `dim` dimension of memrefType with `layoutMap` becomes dynamic
/// after normalization. Dimensions that include dynamic dimensions in the map
/// output will become dynamic dimensions. Return true if `dim` is dynamic
/// dimension.
///
/// Example:
/// #map0 = affine_map<(d0, d1) -> (d0, d1 floordiv 32, d1 mod 32)>
///
/// If d1 is dynamic dimension, 2nd and 3rd dimension of map output are dynamic.
/// memref<4x?xf32, #map0>  ==>  memref<4x?x?xf32>
static bool
isNormalizedMemRefDynamicDim(unsigned dim, AffineMap layoutMap,
                             SmallVectorImpl<unsigned> &inMemrefTypeDynDims,
                             MLIRContext *context) {
  bool isDynamicDim = false;
  AffineExpr expr = layoutMap.getResults()[dim];
  // Check if affine expr of the dimension includes dynamic dimension of input
  // memrefType.
  expr.walk([&inMemrefTypeDynDims, &isDynamicDim, &context](AffineExpr e) {
    if (e.isa<AffineDimExpr>()) {
      for (unsigned dm : inMemrefTypeDynDims) {
        if (e == getAffineDimExpr(dm, context)) {
          isDynamicDim = true;
        }
      }
    }
  });
  return isDynamicDim;
}

/// Create affine expr to calculate dimension size for a tiled-layout map.
static AffineExpr createDimSizeExprForTiledLayout(AffineExpr oldMapOutput,
                                                  TileExprPattern pat) {
  // Create map output for the patterns.
  // "floordiv <tile size>" ==> "ceildiv <tile size>"
  // "mod <tile size>" ==> "<tile size>"
  AffineExpr newMapOutput;
  AffineBinaryOpExpr binaryExpr = nullptr;
  switch (pat) {
  case TileExprPattern::TileMod:
    binaryExpr = oldMapOutput.cast<AffineBinaryOpExpr>();
    newMapOutput = binaryExpr.getRHS();
    break;
  case TileExprPattern::TileFloorDiv:
    binaryExpr = oldMapOutput.cast<AffineBinaryOpExpr>();
    newMapOutput = getAffineBinaryOpExpr(
        AffineExprKind::CeilDiv, binaryExpr.getLHS(), binaryExpr.getRHS());
    break;
  default:
    newMapOutput = oldMapOutput;
  }
  return newMapOutput;
}

/// Create new maps to calculate each dimension size of `newMemRefType`, and
/// create `newDynamicSizes` from them by using AffineApplyOp.
///
/// Steps for normalizing dynamic memrefs for a tiled layout map
/// Example:
///    #map0 = affine_map<(d0, d1) -> (d0, d1 floordiv 32, d1 mod 32)>
///    %0 = dim %arg0, %c1 :memref<4x?xf32>
///    %1 = alloc(%0) : memref<4x?xf32, #map0>
///
/// (Before this function)
/// 1. Check if `map`(#map0) is a tiled layout using `getTileSizePos()`. Only
/// single layout map is supported.
///
/// 2. Create normalized memrefType using `isNormalizedMemRefDynamicDim()`. It
/// is memref<4x?x?xf32> in the above example.
///
/// (In this function)
/// 3. Create new maps to calculate each dimension of the normalized memrefType
/// using `createDimSizeExprForTiledLayout()`. In the tiled layout, the
/// dimension size can be calculated by replacing "floordiv <tile size>" with
/// "ceildiv <tile size>" and "mod <tile size>" with "<tile size>".
/// - New map in the above example
///   #map0 = affine_map<(d0, d1) -> (d0)>
///   #map1 = affine_map<(d0, d1) -> (d1 ceildiv 32)>
///   #map2 = affine_map<(d0, d1) -> (32)>
///
/// 4. Create AffineApplyOp to apply the new maps. The output of AffineApplyOp
/// is used in dynamicSizes of new AllocOp.
///   %0 = dim %arg0, %c1 : memref<4x?xf32>
///   %c4 = arith.constant 4 : index
///   %1 = affine.apply #map1(%c4, %0)
///   %2 = affine.apply #map2(%c4, %0)
static void createNewDynamicSizes(MemRefType oldMemRefType,
                                  MemRefType newMemRefType, AffineMap map,
                                  memref::AllocOp *allocOp, OpBuilder b,
                                  SmallVectorImpl<Value> &newDynamicSizes) {
  // Create new input for AffineApplyOp.
  SmallVector<Value, 4> inAffineApply;
  ArrayRef<int64_t> oldMemRefShape = oldMemRefType.getShape();
  unsigned dynIdx = 0;
  for (unsigned d = 0; d < oldMemRefType.getRank(); ++d) {
    if (oldMemRefShape[d] < 0) {
      // Use dynamicSizes of allocOp for dynamic dimension.
      inAffineApply.emplace_back(allocOp->getDynamicSizes()[dynIdx]);
      dynIdx++;
    } else {
      // Create ConstantOp for static dimension.
      Attribute constantAttr =
          b.getIntegerAttr(b.getIndexType(), oldMemRefShape[d]);
      inAffineApply.emplace_back(
          b.create<arith::ConstantOp>(allocOp->getLoc(), constantAttr));
    }
  }

  // Create new map to calculate each dimension size of new memref for each
  // original map output. Only for dynamic dimesion of `newMemRefType`.
  unsigned newDimIdx = 0;
  ArrayRef<int64_t> newMemRefShape = newMemRefType.getShape();
  SmallVector<std::tuple<AffineExpr, unsigned, unsigned>> tileSizePos;
  (void)getTileSizePos(map, tileSizePos);
  for (AffineExpr expr : map.getResults()) {
    if (newMemRefShape[newDimIdx] < 0) {
      // Create new maps to calculate each dimension size of new memref.
      enum TileExprPattern pat = TileExprPattern::TileNone;
      for (auto pos : tileSizePos) {
        if (newDimIdx == std::get<1>(pos))
          pat = TileExprPattern::TileFloorDiv;
        else if (newDimIdx == std::get<2>(pos))
          pat = TileExprPattern::TileMod;
      }
      AffineExpr newMapOutput = createDimSizeExprForTiledLayout(expr, pat);
      AffineMap newMap =
          AffineMap::get(map.getNumInputs(), map.getNumSymbols(), newMapOutput);
      Value affineApp =
          b.create<AffineApplyOp>(allocOp->getLoc(), newMap, inAffineApply);
      newDynamicSizes.emplace_back(affineApp);
    }
    newDimIdx++;
  }
}

// TODO: Currently works for static memrefs with a single layout map.
LogicalResult mlir::normalizeMemRef(memref::AllocOp *allocOp) {
  MemRefType memrefType = allocOp->getType();
  OpBuilder b(*allocOp);

  // Fetch a new memref type after normalizing the old memref to have an
  // identity map layout.
  MemRefType newMemRefType =
      normalizeMemRefType(memrefType, b, allocOp->getSymbolOperands().size());
  if (newMemRefType == memrefType)
    // Either memrefType already had an identity map or the map couldn't be
    // transformed to an identity map.
    return failure();

  Value oldMemRef = allocOp->getResult();

  SmallVector<Value, 4> symbolOperands(allocOp->getSymbolOperands());
  AffineMap layoutMap = memrefType.getLayout().getAffineMap();
  memref::AllocOp newAlloc;
  // Check if `layoutMap` is a tiled layout. Only single layout map is
  // supported for normalizing dynamic memrefs.
  SmallVector<std::tuple<AffineExpr, unsigned, unsigned>> tileSizePos;
  (void)getTileSizePos(layoutMap, tileSizePos);
  if (newMemRefType.getNumDynamicDims() > 0 && !tileSizePos.empty()) {
    MemRefType oldMemRefType = oldMemRef.getType().cast<MemRefType>();
    SmallVector<Value, 4> newDynamicSizes;
    createNewDynamicSizes(oldMemRefType, newMemRefType, layoutMap, allocOp, b,
                          newDynamicSizes);
    // Add the new dynamic sizes in new AllocOp.
    newAlloc =
        b.create<memref::AllocOp>(allocOp->getLoc(), newMemRefType,
                                  newDynamicSizes, allocOp->getAlignmentAttr());
  } else {
    newAlloc = b.create<memref::AllocOp>(allocOp->getLoc(), newMemRefType,
                                         allocOp->getAlignmentAttr());
  }
  // Replace all uses of the old memref.
  if (failed(replaceAllMemRefUsesWith(oldMemRef, /*newMemRef=*/newAlloc,
                                      /*extraIndices=*/{},
                                      /*indexRemap=*/layoutMap,
                                      /*extraOperands=*/{},
                                      /*symbolOperands=*/symbolOperands,
                                      /*domOpFilter=*/nullptr,
                                      /*postDomOpFilter=*/nullptr,
                                      /*allowNonDereferencingOps=*/true))) {
    // If it failed (due to escapes for example), bail out.
    newAlloc.erase();
    return failure();
  }
  // Replace any uses of the original alloc op and erase it. All remaining uses
  // have to be dealloc's; RAMUW above would've failed otherwise.
  assert(llvm::all_of(oldMemRef.getUsers(), [](Operation *op) {
    return isa<memref::DeallocOp>(op);
  }));
  oldMemRef.replaceAllUsesWith(newAlloc);
  allocOp->erase();
  return success();
}

MemRefType mlir::normalizeMemRefType(MemRefType memrefType, OpBuilder b,
                                     unsigned numSymbolicOperands) {
  unsigned rank = memrefType.getRank();
  if (rank == 0)
    return memrefType;

  if (memrefType.getLayout().isIdentity()) {
    // Either no maps is associated with this memref or this memref has
    // a trivial (identity) map.
    return memrefType;
  }
  AffineMap layoutMap = memrefType.getLayout().getAffineMap();

  // We don't do any checks for one-to-one'ness; we assume that it is
  // one-to-one.

  // Normalize only static memrefs and dynamic memrefs with a tiled-layout map
  // for now.
  // TODO: Normalize the other types of dynamic memrefs.
  SmallVector<std::tuple<AffineExpr, unsigned, unsigned>> tileSizePos;
  (void)getTileSizePos(layoutMap, tileSizePos);
  if (memrefType.getNumDynamicDims() > 0 && tileSizePos.empty())
    return memrefType;

  // We have a single map that is not an identity map. Create a new memref
  // with the right shape and an identity layout map.
  ArrayRef<int64_t> shape = memrefType.getShape();
  // FlatAffineValueConstraint may later on use symbolicOperands.
  FlatAffineValueConstraints fac(rank, numSymbolicOperands);
  SmallVector<unsigned, 4> memrefTypeDynDims;
  for (unsigned d = 0; d < rank; ++d) {
    // Use constraint system only in static dimensions.
    if (shape[d] > 0) {
      fac.addBound(IntegerPolyhedron::LB, d, 0);
      fac.addBound(IntegerPolyhedron::UB, d, shape[d] - 1);
    } else {
      memrefTypeDynDims.emplace_back(d);
    }
  }
  // We compose this map with the original index (logical) space to derive
  // the upper bounds for the new index space.
  unsigned newRank = layoutMap.getNumResults();
  if (failed(fac.composeMatchingMap(layoutMap)))
    return memrefType;
  // TODO: Handle semi-affine maps.
  // Project out the old data dimensions.
  fac.projectOut(newRank, fac.getNumVars() - newRank - fac.getNumLocalVars());
  SmallVector<int64_t, 4> newShape(newRank);
  for (unsigned d = 0; d < newRank; ++d) {
    // Check if each dimension of normalized memrefType is dynamic.
    bool isDynDim = isNormalizedMemRefDynamicDim(
        d, layoutMap, memrefTypeDynDims, b.getContext());
    if (isDynDim) {
      newShape[d] = -1;
    } else {
      // The lower bound for the shape is always zero.
      auto ubConst = fac.getConstantBound(IntegerPolyhedron::UB, d);
      // For a static memref and an affine map with no symbols, this is
      // always bounded.
      assert(ubConst && "should always have an upper bound");
      if (ubConst.getValue() < 0)
        // This is due to an invalid map that maps to a negative space.
        return memrefType;
      // If dimension of new memrefType is dynamic, the value is -1.
      newShape[d] = ubConst.getValue() + 1;
    }
  }

  // Create the new memref type after trivializing the old layout map.
  MemRefType newMemRefType =
      MemRefType::Builder(memrefType)
          .setShape(newShape)
          .setLayout(AffineMapAttr::get(b.getMultiDimIdentityMap(newRank)));

  return newMemRefType;
}
