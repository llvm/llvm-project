//===- IntRangeOptimizations.cpp - Optimizations based on integer ranges --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <utility>

#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"

#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::arith {
#define GEN_PASS_DEF_ARITHINTRANGEOPTS
#include "mlir/Dialect/Arith/Transforms/Passes.h.inc"

#define GEN_PASS_DEF_ARITHINTRANGENARROWING
#include "mlir/Dialect/Arith/Transforms/Passes.h.inc"
} // namespace mlir::arith

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::dataflow;

static std::optional<APInt> getMaybeConstantValue(DataFlowSolver &solver,
                                                  Value value) {
  auto *maybeInferredRange =
      solver.lookupState<IntegerValueRangeLattice>(value);
  if (!maybeInferredRange || maybeInferredRange->getValue().isUninitialized())
    return std::nullopt;
  const ConstantIntRanges &inferredRange =
      maybeInferredRange->getValue().getValue();
  return inferredRange.getConstantValue();
}

/// Patterned after SCCP
static LogicalResult maybeReplaceWithConstant(DataFlowSolver &solver,
                                              PatternRewriter &rewriter,
                                              Value value) {
  if (value.use_empty())
    return failure();
  std::optional<APInt> maybeConstValue = getMaybeConstantValue(solver, value);
  if (!maybeConstValue.has_value())
    return failure();

  Type type = value.getType();
  Location loc = value.getLoc();
  Operation *maybeDefiningOp = value.getDefiningOp();
  Dialect *valueDialect =
      maybeDefiningOp ? maybeDefiningOp->getDialect()
                      : value.getParentRegion()->getParentOp()->getDialect();

  Attribute constAttr;
  if (auto shaped = dyn_cast<ShapedType>(type)) {
    constAttr = mlir::DenseIntElementsAttr::get(shaped, *maybeConstValue);
  } else {
    constAttr = rewriter.getIntegerAttr(type, *maybeConstValue);
  }
  Operation *constOp =
      valueDialect->materializeConstant(rewriter, constAttr, type, loc);
  // Fall back to arith.constant if the dialect materializer doesn't know what
  // to do with an integer constant.
  if (!constOp)
    constOp = rewriter.getContext()
                  ->getLoadedDialect<ArithDialect>()
                  ->materializeConstant(rewriter, constAttr, type, loc);
  if (!constOp)
    return failure();

  rewriter.replaceAllUsesWith(value, constOp->getResult(0));
  return success();
}

namespace {
class DataFlowListener : public RewriterBase::Listener {
public:
  DataFlowListener(DataFlowSolver &s) : s(s) {}

protected:
  void notifyOperationErased(Operation *op) override {
    s.eraseState(s.getProgramPointAfter(op));
    for (Value res : op->getResults())
      s.eraseState(res);
  }

  DataFlowSolver &s;
};

/// Rewrite any results of `op` that were inferred to be constant integers to
/// and replace their uses with that constant. Return success() if all results
/// where thus replaced and the operation is erased. Also replace any block
/// arguments with their constant values.
struct MaterializeKnownConstantValues : public RewritePattern {
  MaterializeKnownConstantValues(MLIRContext *context, DataFlowSolver &s)
      : RewritePattern(Pattern::MatchAnyOpTypeTag(), /*benefit=*/1, context),
        solver(s) {}

  LogicalResult match(Operation *op) const override {
    if (matchPattern(op, m_Constant()))
      return failure();

    auto needsReplacing = [&](Value v) {
      return getMaybeConstantValue(solver, v).has_value() && !v.use_empty();
    };
    bool hasConstantResults = llvm::any_of(op->getResults(), needsReplacing);
    if (op->getNumRegions() == 0)
      return success(hasConstantResults);
    bool hasConstantRegionArgs = false;
    for (Region &region : op->getRegions()) {
      for (Block &block : region.getBlocks()) {
        hasConstantRegionArgs |=
            llvm::any_of(block.getArguments(), needsReplacing);
      }
    }
    return success(hasConstantResults || hasConstantRegionArgs);
  }

  void rewrite(Operation *op, PatternRewriter &rewriter) const override {
    bool replacedAll = (op->getNumResults() != 0);
    for (Value v : op->getResults())
      replacedAll &=
          (succeeded(maybeReplaceWithConstant(solver, rewriter, v)) ||
           v.use_empty());
    if (replacedAll && isOpTriviallyDead(op)) {
      rewriter.eraseOp(op);
      return;
    }

    PatternRewriter::InsertionGuard guard(rewriter);
    for (Region &region : op->getRegions()) {
      for (Block &block : region.getBlocks()) {
        rewriter.setInsertionPointToStart(&block);
        for (BlockArgument &arg : block.getArguments()) {
          (void)maybeReplaceWithConstant(solver, rewriter, arg);
        }
      }
    }
  }

private:
  DataFlowSolver &solver;
};

template <typename RemOp>
struct DeleteTrivialRem : public OpRewritePattern<RemOp> {
  DeleteTrivialRem(MLIRContext *context, DataFlowSolver &s)
      : OpRewritePattern<RemOp>(context), solver(s) {}

  LogicalResult matchAndRewrite(RemOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op.getOperand(0);
    Value rhs = op.getOperand(1);
    auto maybeModulus = getConstantIntValue(rhs);
    if (!maybeModulus.has_value())
      return failure();
    int64_t modulus = *maybeModulus;
    if (modulus <= 0)
      return failure();
    auto *maybeLhsRange = solver.lookupState<IntegerValueRangeLattice>(lhs);
    if (!maybeLhsRange || maybeLhsRange->getValue().isUninitialized())
      return failure();
    const ConstantIntRanges &lhsRange = maybeLhsRange->getValue().getValue();
    const APInt &min = isa<RemUIOp>(op) ? lhsRange.umin() : lhsRange.smin();
    const APInt &max = isa<RemUIOp>(op) ? lhsRange.umax() : lhsRange.smax();
    // The minima and maxima here are given as closed ranges, we must be
    // strictly less than the modulus.
    if (min.isNegative() || min.uge(modulus))
      return failure();
    if (max.isNegative() || max.uge(modulus))
      return failure();
    if (!min.ule(max))
      return failure();

    // With all those conditions out of the way, we know thas this invocation of
    // a remainder is a noop because the input is strictly within the range
    // [0, modulus), so get rid of it.
    rewriter.replaceOp(op, ValueRange{lhs});
    return success();
  }

private:
  DataFlowSolver &solver;
};

/// Check if `type` is index or integer type with `getWidth() > targetBitwidth`.
static Type checkIntType(Type type, unsigned targetBitwidth) {
  Type elemType = getElementTypeOrSelf(type);
  if (isa<IndexType>(elemType))
    return type;

  if (auto intType = dyn_cast<IntegerType>(elemType))
    if (intType.getWidth() > targetBitwidth)
      return type;

  return nullptr;
}

/// Check if op have same type for all operands and results and this type
/// is suitable for truncation.
/// Retuns args type or empty.
static Type checkElementwiseOpType(Operation *op, unsigned targetBitwidth) {
  if (op->getNumOperands() == 0 || op->getNumResults() == 0)
    return nullptr;

  Type type;
  for (auto range :
       {ValueRange(op->getOperands()), ValueRange(op->getResults())}) {
    for (Value val : range) {
      if (!type) {
        type = val.getType();
        continue;
      }

      if (type != val.getType())
        return nullptr;
    }
  }

  return checkIntType(type, targetBitwidth);
}

/// Return union of all operands values ranges.
static std::optional<ConstantIntRanges> getOperandsRange(DataFlowSolver &solver,
                                                         ValueRange operands) {
  std::optional<ConstantIntRanges> ret;
  for (Value value : operands) {
    auto *maybeInferredRange =
        solver.lookupState<IntegerValueRangeLattice>(value);
    if (!maybeInferredRange || maybeInferredRange->getValue().isUninitialized())
      return std::nullopt;

    const ConstantIntRanges &inferredRange =
        maybeInferredRange->getValue().getValue();

    ret = (ret ? ret->rangeUnion(inferredRange) : inferredRange);
  }
  return ret;
}

/// Return int type truncated to `targetBitwidth`. If `srcType` is shaped,
/// return shaped type as well.
static Type getTargetType(Type srcType, unsigned targetBitwidth) {
  auto dstType = IntegerType::get(srcType.getContext(), targetBitwidth);
  if (auto shaped = dyn_cast<ShapedType>(srcType))
    return shaped.clone(dstType);

  assert(srcType.isIntOrIndex() && "Invalid src type");
  return dstType;
}

/// Check provided `range` is inside `smin, smax, umin, umax` bounds.
static bool checkRange(const ConstantIntRanges &range, APInt smin, APInt smax,
                       APInt umin, APInt umax) {
  auto sge = [](APInt val1, APInt val2) -> bool {
    unsigned width = std::max(val1.getBitWidth(), val2.getBitWidth());
    val1 = val1.sext(width);
    val2 = val2.sext(width);
    return val1.sge(val2);
  };
  auto sle = [](APInt val1, APInt val2) -> bool {
    unsigned width = std::max(val1.getBitWidth(), val2.getBitWidth());
    val1 = val1.sext(width);
    val2 = val2.sext(width);
    return val1.sle(val2);
  };
  auto uge = [](APInt val1, APInt val2) -> bool {
    unsigned width = std::max(val1.getBitWidth(), val2.getBitWidth());
    val1 = val1.zext(width);
    val2 = val2.zext(width);
    return val1.uge(val2);
  };
  auto ule = [](APInt val1, APInt val2) -> bool {
    unsigned width = std::max(val1.getBitWidth(), val2.getBitWidth());
    val1 = val1.zext(width);
    val2 = val2.zext(width);
    return val1.ule(val2);
  };
  return sge(range.smin(), smin) && sle(range.smax(), smax) &&
         uge(range.umin(), umin) && ule(range.umax(), umax);
}

static Value doCast(OpBuilder &builder, Location loc, Value src, Type dstType) {
  Type srcType = src.getType();
  assert(isa<VectorType>(srcType) == isa<VectorType>(dstType) &&
         "Mixing vector and non-vector types");
  Type srcElemType = getElementTypeOrSelf(srcType);
  Type dstElemType = getElementTypeOrSelf(dstType);
  assert(srcElemType.isIntOrIndex() && "Invalid src type");
  assert(dstElemType.isIntOrIndex() && "Invalid dst type");
  if (srcType == dstType)
    return src;

  if (isa<IndexType>(srcElemType) || isa<IndexType>(dstElemType))
    return builder.create<arith::IndexCastUIOp>(loc, dstType, src);

  auto srcInt = cast<IntegerType>(srcElemType);
  auto dstInt = cast<IntegerType>(dstElemType);
  if (dstInt.getWidth() < srcInt.getWidth())
    return builder.create<arith::TruncIOp>(loc, dstType, src);

  return builder.create<arith::ExtUIOp>(loc, dstType, src);
}

struct NarrowElementwise final : OpTraitRewritePattern<OpTrait::Elementwise> {
  NarrowElementwise(MLIRContext *context, DataFlowSolver &s,
                    ArrayRef<unsigned> target)
      : OpTraitRewritePattern(context), solver(s), targetBitwidths(target) {}

  using OpTraitRewritePattern::OpTraitRewritePattern;
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    std::optional<ConstantIntRanges> range =
        getOperandsRange(solver, op->getResults());
    if (!range)
      return failure();

    for (unsigned targetBitwidth : targetBitwidths) {
      Type srcType = checkElementwiseOpType(op, targetBitwidth);
      if (!srcType)
        continue;

      // We are truncating op args to the desired bitwidth before the op and
      // then extending op results back to the original width after. extui and
      // exti will produce different results for negative values, so limit
      // signed range to non-negative values.
      auto smin = APInt::getZero(targetBitwidth);
      auto smax = APInt::getSignedMaxValue(targetBitwidth);
      auto umin = APInt::getMinValue(targetBitwidth);
      auto umax = APInt::getMaxValue(targetBitwidth);
      if (!checkRange(*range, smin, smax, umin, umax))
        continue;

      Type targetType = getTargetType(srcType, targetBitwidth);
      if (targetType == srcType)
        continue;

      Location loc = op->getLoc();
      IRMapping mapping;
      for (Value arg : op->getOperands()) {
        Value newArg = doCast(rewriter, loc, arg, targetType);
        mapping.map(arg, newArg);
      }

      Operation *newOp = rewriter.clone(*op, mapping);
      rewriter.modifyOpInPlace(newOp, [&]() {
        for (OpResult res : newOp->getResults()) {
          res.setType(targetType);
        }
      });
      SmallVector<Value> newResults;
      for (Value res : newOp->getResults())
        newResults.emplace_back(doCast(rewriter, loc, res, srcType));

      rewriter.replaceOp(op, newResults);
      return success();
    }
    return failure();
  }

private:
  DataFlowSolver &solver;
  SmallVector<unsigned, 4> targetBitwidths;
};

struct NarrowCmpI final : OpRewritePattern<arith::CmpIOp> {
  NarrowCmpI(MLIRContext *context, DataFlowSolver &s, ArrayRef<unsigned> target)
      : OpRewritePattern(context), solver(s), targetBitwidths(target) {}

  LogicalResult matchAndRewrite(arith::CmpIOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();

    std::optional<ConstantIntRanges> range =
        getOperandsRange(solver, {lhs, rhs});
    if (!range)
      return failure();

    for (unsigned targetBitwidth : targetBitwidths) {
      Type srcType = checkIntType(lhs.getType(), targetBitwidth);
      if (!srcType)
        continue;

      auto smin = APInt::getSignedMinValue(targetBitwidth);
      auto smax = APInt::getSignedMaxValue(targetBitwidth);
      auto umin = APInt::getMinValue(targetBitwidth);
      auto umax = APInt::getMaxValue(targetBitwidth);
      if (!checkRange(*range, smin, smax, umin, umax))
        continue;

      Type targetType = getTargetType(srcType, targetBitwidth);
      if (targetType == srcType)
        continue;

      Location loc = op->getLoc();
      IRMapping mapping;
      for (Value arg : op->getOperands()) {
        Value newArg = doCast(rewriter, loc, arg, targetType);
        mapping.map(arg, newArg);
      }

      Operation *newOp = rewriter.clone(*op, mapping);
      rewriter.replaceOp(op, newOp->getResults());
      return success();
    }
    return failure();
  }

private:
  DataFlowSolver &solver;
  SmallVector<unsigned, 4> targetBitwidths;
};

struct IntRangeOptimizationsPass final
    : arith::impl::ArithIntRangeOptsBase<IntRangeOptimizationsPass> {

  void runOnOperation() override {
    Operation *op = getOperation();
    MLIRContext *ctx = op->getContext();
    DataFlowSolver solver;
    solver.load<DeadCodeAnalysis>();
    solver.load<IntegerRangeAnalysis>();
    if (failed(solver.initializeAndRun(op)))
      return signalPassFailure();

    DataFlowListener listener(solver);

    RewritePatternSet patterns(ctx);
    populateIntRangeOptimizationsPatterns(patterns, solver);

    GreedyRewriteConfig config;
    config.listener = &listener;

    if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns), config)))
      signalPassFailure();
  }
};

struct IntRangeNarrowingPass final
    : arith::impl::ArithIntRangeNarrowingBase<IntRangeNarrowingPass> {
  using ArithIntRangeNarrowingBase::ArithIntRangeNarrowingBase;

  void runOnOperation() override {
    Operation *op = getOperation();
    MLIRContext *ctx = op->getContext();
    DataFlowSolver solver;
    solver.load<DeadCodeAnalysis>();
    solver.load<IntegerRangeAnalysis>();
    if (failed(solver.initializeAndRun(op)))
      return signalPassFailure();

    DataFlowListener listener(solver);

    RewritePatternSet patterns(ctx);
    populateIntRangeNarrowingPatterns(patterns, solver, bitwidthsSupported);

    GreedyRewriteConfig config;
    config.listener = &listener;

    if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns), config)))
      signalPassFailure();
  }
};
} // namespace

void mlir::arith::populateIntRangeOptimizationsPatterns(
    RewritePatternSet &patterns, DataFlowSolver &solver) {
  patterns.add<MaterializeKnownConstantValues, DeleteTrivialRem<RemSIOp>,
               DeleteTrivialRem<RemUIOp>>(patterns.getContext(), solver);
}

void mlir::arith::populateIntRangeNarrowingPatterns(
    RewritePatternSet &patterns, DataFlowSolver &solver,
    ArrayRef<unsigned> bitwidthsSupported) {
  patterns.add<NarrowElementwise, NarrowCmpI>(patterns.getContext(), solver,
                                              bitwidthsSupported);
}

std::unique_ptr<Pass> mlir::arith::createIntRangeOptimizationsPass() {
  return std::make_unique<IntRangeOptimizationsPass>();
}
