//===- AffineToStandard.cpp - Lower affine constructs to primitives -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file lowers affine constructs (If and For statements, AffineApply
// operations) within a function into their standard If and For equivalent ops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"

#include "../PassDetail.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::vector;

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
    return buildBinaryExpr<AddIOp>(expr);
  }

  Value visitMulExpr(AffineBinaryOpExpr expr) {
    return buildBinaryExpr<MulIOp>(expr);
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

    Value remainder = builder.create<SignedRemIOp>(loc, lhs, rhs);
    Value zeroCst = builder.create<ConstantIndexOp>(loc, 0);
    Value isRemainderNegative =
        builder.create<CmpIOp>(loc, CmpIPredicate::slt, remainder, zeroCst);
    Value correctedRemainder = builder.create<AddIOp>(loc, remainder, rhs);
    Value result = builder.create<SelectOp>(loc, isRemainderNegative,
                                            correctedRemainder, remainder);
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

    Value zeroCst = builder.create<ConstantIndexOp>(loc, 0);
    Value noneCst = builder.create<ConstantIndexOp>(loc, -1);
    Value negative =
        builder.create<CmpIOp>(loc, CmpIPredicate::slt, lhs, zeroCst);
    Value negatedDecremented = builder.create<SubIOp>(loc, noneCst, lhs);
    Value dividend =
        builder.create<SelectOp>(loc, negative, negatedDecremented, lhs);
    Value quotient = builder.create<SignedDivIOp>(loc, dividend, rhs);
    Value correctedQuotient = builder.create<SubIOp>(loc, noneCst, quotient);
    Value result =
        builder.create<SelectOp>(loc, negative, correctedQuotient, quotient);
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

    Value zeroCst = builder.create<ConstantIndexOp>(loc, 0);
    Value oneCst = builder.create<ConstantIndexOp>(loc, 1);
    Value nonPositive =
        builder.create<CmpIOp>(loc, CmpIPredicate::sle, lhs, zeroCst);
    Value negated = builder.create<SubIOp>(loc, zeroCst, lhs);
    Value decremented = builder.create<SubIOp>(loc, lhs, oneCst);
    Value dividend =
        builder.create<SelectOp>(loc, nonPositive, negated, decremented);
    Value quotient = builder.create<SignedDivIOp>(loc, dividend, rhs);
    Value negatedQuotient = builder.create<SubIOp>(loc, zeroCst, quotient);
    Value incrementedQuotient = builder.create<AddIOp>(loc, quotient, oneCst);
    Value result = builder.create<SelectOp>(loc, nonPositive, negatedQuotient,
                                            incrementedQuotient);
    return result;
  }

  Value visitConstantExpr(AffineConstantExpr expr) {
    auto valueAttr =
        builder.getIntegerAttr(builder.getIndexType(), expr.getValue());
    auto op =
        builder.create<ConstantOp>(loc, builder.getIndexType(), valueAttr);
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

/// Given a range of values, emit the code that reduces them with "min" or "max"
/// depending on the provided comparison predicate.  The predicate defines which
/// comparison to perform, "lt" for "min", "gt" for "max" and is used for the
/// `cmpi` operation followed by the `select` operation:
///
///   %cond   = cmpi "predicate" %v0, %v1
///   %result = select %cond, %v0, %v1
///
/// Multiple values are scanned in a linear sequence.  This creates a data
/// dependences that wouldn't exist in a tree reduction, but is easier to
/// recognize as a reduction by the subsequent passes.
static Value buildMinMaxReductionSeq(Location loc, CmpIPredicate predicate,
                                     ValueRange values, OpBuilder &builder) {
  assert(!llvm::empty(values) && "empty min/max chain");

  auto valueIt = values.begin();
  Value value = *valueIt++;
  for (; valueIt != values.end(); ++valueIt) {
    auto cmpOp = builder.create<CmpIOp>(loc, predicate, value, *valueIt);
    value = builder.create<SelectOp>(loc, cmpOp.getResult(), value, *valueIt);
  }

  return value;
}

/// Emit instructions that correspond to computing the maximum value among the
/// values of a (potentially) multi-output affine map applied to `operands`.
static Value lowerAffineMapMax(OpBuilder &builder, Location loc, AffineMap map,
                               ValueRange operands) {
  if (auto values = expandAffineMap(builder, loc, map, operands))
    return buildMinMaxReductionSeq(loc, CmpIPredicate::sgt, *values, builder);
  return nullptr;
}

/// Emit instructions that correspond to computing the minimum value among the
/// values of a (potentially) multi-output affine map applied to `operands`.
static Value lowerAffineMapMin(OpBuilder &builder, Location loc, AffineMap map,
                               ValueRange operands) {
  if (auto values = expandAffineMap(builder, loc, map, operands))
    return buildMinMaxReductionSeq(loc, CmpIPredicate::slt, *values, builder);
  return nullptr;
}

/// Emit instructions that correspond to the affine map in the upper bound
/// applied to the respective operands, and compute the minimum value across
/// the results.
Value mlir::lowerAffineUpperBound(AffineForOp op, OpBuilder &builder) {
  return lowerAffineMapMin(builder, op.getLoc(), op.getUpperBoundMap(),
                           op.getUpperBoundOperands());
}

/// Emit instructions that correspond to the affine map in the lower bound
/// applied to the respective operands, and compute the maximum value across
/// the results.
Value mlir::lowerAffineLowerBound(AffineForOp op, OpBuilder &builder) {
  return lowerAffineMapMax(builder, op.getLoc(), op.getLowerBoundMap(),
                           op.getLowerBoundOperands());
}

namespace {
class AffineMinLowering : public OpRewritePattern<AffineMinOp> {
public:
  using OpRewritePattern<AffineMinOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineMinOp op,
                                PatternRewriter &rewriter) const override {
    Value reduced =
        lowerAffineMapMin(rewriter, op.getLoc(), op.map(), op.operands());
    if (!reduced)
      return failure();

    rewriter.replaceOp(op, reduced);
    return success();
  }
};

class AffineMaxLowering : public OpRewritePattern<AffineMaxOp> {
public:
  using OpRewritePattern<AffineMaxOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineMaxOp op,
                                PatternRewriter &rewriter) const override {
    Value reduced =
        lowerAffineMapMax(rewriter, op.getLoc(), op.map(), op.operands());
    if (!reduced)
      return failure();

    rewriter.replaceOp(op, reduced);
    return success();
  }
};

/// Affine yields ops are removed.
class AffineYieldOpLowering : public OpRewritePattern<AffineYieldOp> {
public:
  using OpRewritePattern<AffineYieldOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineYieldOp op,
                                PatternRewriter &rewriter) const override {
    if (isa<scf::ParallelOp>(op->getParentOp())) {
      // scf.parallel does not yield any values via its terminator scf.yield but
      // models reductions differently using additional ops in its region.
      rewriter.replaceOpWithNewOp<scf::YieldOp>(op);
      return success();
    }
    rewriter.replaceOpWithNewOp<scf::YieldOp>(op, op.operands());
    return success();
  }
};

class AffineForLowering : public OpRewritePattern<AffineForOp> {
public:
  using OpRewritePattern<AffineForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineForOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value lowerBound = lowerAffineLowerBound(op, rewriter);
    Value upperBound = lowerAffineUpperBound(op, rewriter);
    Value step = rewriter.create<ConstantIndexOp>(loc, op.getStep());
    auto scfForOp = rewriter.create<scf::ForOp>(loc, lowerBound, upperBound,
                                                step, op.getIterOperands());
    rewriter.eraseBlock(scfForOp.getBody());
    rewriter.inlineRegionBefore(op.region(), scfForOp.region(),
                                scfForOp.region().end());
    rewriter.replaceOp(op, scfForOp.results());
    return success();
  }
};

/// Returns the identity value associated with an AtomicRMWKind op.
static Value getIdentityValue(AtomicRMWKind op, OpBuilder &builder,
                              Location loc) {
  switch (op) {
  case AtomicRMWKind::addf:
    return builder.create<ConstantOp>(loc, builder.getF32FloatAttr(0));
  case AtomicRMWKind::addi:
    return builder.create<ConstantOp>(loc, builder.getI32IntegerAttr(0));
  case AtomicRMWKind::mulf:
    return builder.create<ConstantOp>(loc, builder.getF32FloatAttr(1));
  case AtomicRMWKind::muli:
    return builder.create<ConstantOp>(loc, builder.getI32IntegerAttr(1));
  // TODO: Add remaining reduction operations.
  default:
    (void)emitOptionalError(loc, "Reduction operation type not supported");
    break;
  }
  return nullptr;
}

/// Return the value obtained by applying the reduction operation kind
/// associated with a binary AtomicRMWKind op to `lhs` and `rhs`.
static Value getReductionOp(AtomicRMWKind op, OpBuilder &builder, Location loc,
                            Value lhs, Value rhs) {
  switch (op) {
  case AtomicRMWKind::addf:
    return builder.create<AddFOp>(loc, lhs, rhs);
  case AtomicRMWKind::addi:
    return builder.create<AddIOp>(loc, lhs, rhs);
  case AtomicRMWKind::mulf:
    return builder.create<MulFOp>(loc, lhs, rhs);
  case AtomicRMWKind::muli:
    return builder.create<MulIOp>(loc, lhs, rhs);
  // TODO: Add remaining reduction operations.
  default:
    (void)emitOptionalError(loc, "Reduction operation type not supported");
    break;
  }
  return nullptr;
}

/// Convert an `affine.parallel` (loop nest) operation into a `scf.parallel`
/// operation.
class AffineParallelLowering : public OpRewritePattern<AffineParallelOp> {
public:
  using OpRewritePattern<AffineParallelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineParallelOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    SmallVector<Value, 8> steps;
    SmallVector<Value, 8> upperBoundTuple;
    SmallVector<Value, 8> lowerBoundTuple;
    SmallVector<Value, 8> identityVals;
    // Finding lower and upper bound by expanding the map expression.
    // Checking if expandAffineMap is not giving NULL.
    Optional<SmallVector<Value, 8>> lowerBound = expandAffineMap(
        rewriter, loc, op.lowerBoundsMap(), op.getLowerBoundsOperands());
    Optional<SmallVector<Value, 8>> upperBound = expandAffineMap(
        rewriter, loc, op.upperBoundsMap(), op.getUpperBoundsOperands());
    if (!lowerBound || !upperBound)
      return failure();
    upperBoundTuple = *upperBound;
    lowerBoundTuple = *lowerBound;
    steps.reserve(op.steps().size());
    for (Attribute step : op.steps())
      steps.push_back(rewriter.create<ConstantIndexOp>(
          loc, step.cast<IntegerAttr>().getInt()));
    // Get the terminator op.
    Operation *affineParOpTerminator = op.getBody()->getTerminator();
    scf::ParallelOp parOp;
    if (op.results().empty()) {
      // Case with no reduction operations/return values.
      parOp = rewriter.create<scf::ParallelOp>(loc, lowerBoundTuple,
                                               upperBoundTuple, steps,
                                               /*bodyBuilderFn=*/nullptr);
      rewriter.eraseBlock(parOp.getBody());
      rewriter.inlineRegionBefore(op.region(), parOp.region(),
                                  parOp.region().end());
      rewriter.replaceOp(op, parOp.results());
      return success();
    }
    // Case with affine.parallel with reduction operations/return values.
    // scf.parallel handles the reduction operation differently unlike
    // affine.parallel.
    ArrayRef<Attribute> reductions = op.reductions().getValue();
    for (Attribute reduction : reductions) {
      // For each of the reduction operations get the identity values for
      // initialization of the result values.
      Optional<AtomicRMWKind> reductionOp = symbolizeAtomicRMWKind(
          static_cast<uint64_t>(reduction.cast<IntegerAttr>().getInt()));
      assert(reductionOp.hasValue() &&
             "Reduction operation cannot be of None Type");
      AtomicRMWKind reductionOpValue = reductionOp.getValue();
      identityVals.push_back(getIdentityValue(reductionOpValue, rewriter, loc));
    }
    parOp = rewriter.create<scf::ParallelOp>(
        loc, lowerBoundTuple, upperBoundTuple, steps, identityVals,
        /*bodyBuilderFn=*/nullptr);

    //  Copy the body of the affine.parallel op.
    rewriter.eraseBlock(parOp.getBody());
    rewriter.inlineRegionBefore(op.region(), parOp.region(),
                                parOp.region().end());
    assert(reductions.size() == affineParOpTerminator->getNumOperands() &&
           "Unequal number of reductions and operands.");
    for (unsigned i = 0, end = reductions.size(); i < end; i++) {
      // For each of the reduction operations get the respective mlir::Value.
      Optional<AtomicRMWKind> reductionOp =
          symbolizeAtomicRMWKind(reductions[i].cast<IntegerAttr>().getInt());
      assert(reductionOp.hasValue() &&
             "Reduction Operation cannot be of None Type");
      AtomicRMWKind reductionOpValue = reductionOp.getValue();
      rewriter.setInsertionPoint(&parOp.getBody()->back());
      auto reduceOp = rewriter.create<scf::ReduceOp>(
          loc, affineParOpTerminator->getOperand(i));
      rewriter.setInsertionPointToEnd(&reduceOp.reductionOperator().front());
      Value reductionResult =
          getReductionOp(reductionOpValue, rewriter, loc,
                         reduceOp.reductionOperator().front().getArgument(0),
                         reduceOp.reductionOperator().front().getArgument(1));
      rewriter.create<scf::ReduceReturnOp>(loc, reductionResult);
    }
    rewriter.replaceOp(op, parOp.results());
    return success();
  }
};

class AffineIfLowering : public OpRewritePattern<AffineIfOp> {
public:
  using OpRewritePattern<AffineIfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineIfOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    // Now we just have to handle the condition logic.
    auto integerSet = op.getIntegerSet();
    Value zeroConstant = rewriter.create<ConstantIndexOp>(loc, 0);
    SmallVector<Value, 8> operands(op.getOperands());
    auto operandsRef = llvm::makeArrayRef(operands);

    // Calculate cond as a conjunction without short-circuiting.
    Value cond = nullptr;
    for (unsigned i = 0, e = integerSet.getNumConstraints(); i < e; ++i) {
      AffineExpr constraintExpr = integerSet.getConstraint(i);
      bool isEquality = integerSet.isEq(i);

      // Build and apply an affine expression
      auto numDims = integerSet.getNumDims();
      Value affResult = expandAffineExpr(rewriter, loc, constraintExpr,
                                         operandsRef.take_front(numDims),
                                         operandsRef.drop_front(numDims));
      if (!affResult)
        return failure();
      auto pred = isEquality ? CmpIPredicate::eq : CmpIPredicate::sge;
      Value cmpVal =
          rewriter.create<CmpIOp>(loc, pred, affResult, zeroConstant);
      cond =
          cond ? rewriter.create<AndOp>(loc, cond, cmpVal).getResult() : cmpVal;
    }
    cond = cond ? cond
                : rewriter.create<ConstantIntOp>(loc, /*value=*/1, /*width=*/1);

    bool hasElseRegion = !op.elseRegion().empty();
    auto ifOp = rewriter.create<scf::IfOp>(loc, cond, hasElseRegion);
    rewriter.inlineRegionBefore(op.thenRegion(), &ifOp.thenRegion().back());
    rewriter.eraseBlock(&ifOp.thenRegion().back());
    if (hasElseRegion) {
      rewriter.inlineRegionBefore(op.elseRegion(), &ifOp.elseRegion().back());
      rewriter.eraseBlock(&ifOp.elseRegion().back());
    }

    // Ok, we're done!
    rewriter.eraseOp(op);
    return success();
  }
};

/// Convert an "affine.apply" operation into a sequence of arithmetic
/// operations using the StandardOps dialect.
class AffineApplyLowering : public OpRewritePattern<AffineApplyOp> {
public:
  using OpRewritePattern<AffineApplyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineApplyOp op,
                                PatternRewriter &rewriter) const override {
    auto maybeExpandedMap =
        expandAffineMap(rewriter, op.getLoc(), op.getAffineMap(),
                        llvm::to_vector<8>(op.getOperands()));
    if (!maybeExpandedMap)
      return failure();
    rewriter.replaceOp(op, *maybeExpandedMap);
    return success();
  }
};

/// Apply the affine map from an 'affine.load' operation to its operands, and
/// feed the results to a newly created 'memref.load' operation (which replaces
/// the original 'affine.load').
class AffineLoadLowering : public OpRewritePattern<AffineLoadOp> {
public:
  using OpRewritePattern<AffineLoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineLoadOp op,
                                PatternRewriter &rewriter) const override {
    // Expand affine map from 'affineLoadOp'.
    SmallVector<Value, 8> indices(op.getMapOperands());
    auto resultOperands =
        expandAffineMap(rewriter, op.getLoc(), op.getAffineMap(), indices);
    if (!resultOperands)
      return failure();

    // Build vector.load memref[expandedMap.results].
    rewriter.replaceOpWithNewOp<memref::LoadOp>(op, op.getMemRef(),
                                                *resultOperands);
    return success();
  }
};

/// Apply the affine map from an 'affine.prefetch' operation to its operands,
/// and feed the results to a newly created 'memref.prefetch' operation (which
/// replaces the original 'affine.prefetch').
class AffinePrefetchLowering : public OpRewritePattern<AffinePrefetchOp> {
public:
  using OpRewritePattern<AffinePrefetchOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffinePrefetchOp op,
                                PatternRewriter &rewriter) const override {
    // Expand affine map from 'affinePrefetchOp'.
    SmallVector<Value, 8> indices(op.getMapOperands());
    auto resultOperands =
        expandAffineMap(rewriter, op.getLoc(), op.getAffineMap(), indices);
    if (!resultOperands)
      return failure();

    // Build memref.prefetch memref[expandedMap.results].
    rewriter.replaceOpWithNewOp<memref::PrefetchOp>(
        op, op.memref(), *resultOperands, op.isWrite(), op.localityHint(),
        op.isDataCache());
    return success();
  }
};

/// Apply the affine map from an 'affine.store' operation to its operands, and
/// feed the results to a newly created 'memref.store' operation (which replaces
/// the original 'affine.store').
class AffineStoreLowering : public OpRewritePattern<AffineStoreOp> {
public:
  using OpRewritePattern<AffineStoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineStoreOp op,
                                PatternRewriter &rewriter) const override {
    // Expand affine map from 'affineStoreOp'.
    SmallVector<Value, 8> indices(op.getMapOperands());
    auto maybeExpandedMap =
        expandAffineMap(rewriter, op.getLoc(), op.getAffineMap(), indices);
    if (!maybeExpandedMap)
      return failure();

    // Build memref.store valueToStore, memref[expandedMap.results].
    rewriter.replaceOpWithNewOp<memref::StoreOp>(
        op, op.getValueToStore(), op.getMemRef(), *maybeExpandedMap);
    return success();
  }
};

/// Apply the affine maps from an 'affine.dma_start' operation to each of their
/// respective map operands, and feed the results to a newly created
/// 'memref.dma_start' operation (which replaces the original
/// 'affine.dma_start').
class AffineDmaStartLowering : public OpRewritePattern<AffineDmaStartOp> {
public:
  using OpRewritePattern<AffineDmaStartOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineDmaStartOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value, 8> operands(op.getOperands());
    auto operandsRef = llvm::makeArrayRef(operands);

    // Expand affine map for DMA source memref.
    auto maybeExpandedSrcMap = expandAffineMap(
        rewriter, op.getLoc(), op.getSrcMap(),
        operandsRef.drop_front(op.getSrcMemRefOperandIndex() + 1));
    if (!maybeExpandedSrcMap)
      return failure();
    // Expand affine map for DMA destination memref.
    auto maybeExpandedDstMap = expandAffineMap(
        rewriter, op.getLoc(), op.getDstMap(),
        operandsRef.drop_front(op.getDstMemRefOperandIndex() + 1));
    if (!maybeExpandedDstMap)
      return failure();
    // Expand affine map for DMA tag memref.
    auto maybeExpandedTagMap = expandAffineMap(
        rewriter, op.getLoc(), op.getTagMap(),
        operandsRef.drop_front(op.getTagMemRefOperandIndex() + 1));
    if (!maybeExpandedTagMap)
      return failure();

    // Build memref.dma_start operation with affine map results.
    rewriter.replaceOpWithNewOp<memref::DmaStartOp>(
        op, op.getSrcMemRef(), *maybeExpandedSrcMap, op.getDstMemRef(),
        *maybeExpandedDstMap, op.getNumElements(), op.getTagMemRef(),
        *maybeExpandedTagMap, op.getStride(), op.getNumElementsPerStride());
    return success();
  }
};

/// Apply the affine map from an 'affine.dma_wait' operation tag memref,
/// and feed the results to a newly created 'memref.dma_wait' operation (which
/// replaces the original 'affine.dma_wait').
class AffineDmaWaitLowering : public OpRewritePattern<AffineDmaWaitOp> {
public:
  using OpRewritePattern<AffineDmaWaitOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineDmaWaitOp op,
                                PatternRewriter &rewriter) const override {
    // Expand affine map for DMA tag memref.
    SmallVector<Value, 8> indices(op.getTagIndices());
    auto maybeExpandedTagMap =
        expandAffineMap(rewriter, op.getLoc(), op.getTagMap(), indices);
    if (!maybeExpandedTagMap)
      return failure();

    // Build memref.dma_wait operation with affine map results.
    rewriter.replaceOpWithNewOp<memref::DmaWaitOp>(
        op, op.getTagMemRef(), *maybeExpandedTagMap, op.getNumElements());
    return success();
  }
};

/// Apply the affine map from an 'affine.vector_load' operation to its operands,
/// and feed the results to a newly created 'vector.load' operation (which
/// replaces the original 'affine.vector_load').
class AffineVectorLoadLowering : public OpRewritePattern<AffineVectorLoadOp> {
public:
  using OpRewritePattern<AffineVectorLoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineVectorLoadOp op,
                                PatternRewriter &rewriter) const override {
    // Expand affine map from 'affineVectorLoadOp'.
    SmallVector<Value, 8> indices(op.getMapOperands());
    auto resultOperands =
        expandAffineMap(rewriter, op.getLoc(), op.getAffineMap(), indices);
    if (!resultOperands)
      return failure();

    // Build vector.load memref[expandedMap.results].
    rewriter.replaceOpWithNewOp<vector::LoadOp>(
        op, op.getVectorType(), op.getMemRef(), *resultOperands);
    return success();
  }
};

/// Apply the affine map from an 'affine.vector_store' operation to its
/// operands, and feed the results to a newly created 'vector.store' operation
/// (which replaces the original 'affine.vector_store').
class AffineVectorStoreLowering : public OpRewritePattern<AffineVectorStoreOp> {
public:
  using OpRewritePattern<AffineVectorStoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineVectorStoreOp op,
                                PatternRewriter &rewriter) const override {
    // Expand affine map from 'affineVectorStoreOp'.
    SmallVector<Value, 8> indices(op.getMapOperands());
    auto maybeExpandedMap =
        expandAffineMap(rewriter, op.getLoc(), op.getAffineMap(), indices);
    if (!maybeExpandedMap)
      return failure();

    rewriter.replaceOpWithNewOp<vector::StoreOp>(
        op, op.getValueToStore(), op.getMemRef(), *maybeExpandedMap);
    return success();
  }
};

} // end namespace

void mlir::populateAffineToStdConversionPatterns(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  // clang-format off
  patterns.insert<
      AffineApplyLowering,
      AffineDmaStartLowering,
      AffineDmaWaitLowering,
      AffineLoadLowering,
      AffineMinLowering,
      AffineMaxLowering,
      AffineParallelLowering,
      AffinePrefetchLowering,
      AffineStoreLowering,
      AffineForLowering,
      AffineIfLowering,
      AffineYieldOpLowering>(ctx);
  // clang-format on
}

void mlir::populateAffineToVectorConversionPatterns(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  // clang-format off
  patterns.insert<
      AffineVectorLoadLowering,
      AffineVectorStoreLowering>(ctx);
  // clang-format on
}

namespace {
class LowerAffinePass : public ConvertAffineToStandardBase<LowerAffinePass> {
  void runOnOperation() override {
    OwningRewritePatternList patterns;
    populateAffineToStdConversionPatterns(patterns, &getContext());
    populateAffineToVectorConversionPatterns(patterns, &getContext());
    ConversionTarget target(getContext());
    target.addLegalDialect<memref::MemRefDialect, scf::SCFDialect,
                           StandardOpsDialect, VectorDialect>();
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

/// Lowers If and For operations within a function into their lower level CFG
/// equivalent blocks.
std::unique_ptr<Pass> mlir::createLowerAffinePass() {
  return std::make_unique<LowerAffinePass>();
}
