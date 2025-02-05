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

static void copyIntegerRange(DataFlowSolver &solver, Value oldVal,
                             Value newVal) {
  assert(oldVal.getType() == newVal.getType() &&
         "Can't copy integer ranges between different types");
  auto *oldState = solver.lookupState<IntegerValueRangeLattice>(oldVal);
  if (!oldState)
    return;
  (void)solver.getOrCreateState<IntegerValueRangeLattice>(newVal)->join(
      *oldState);
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

  copyIntegerRange(solver, value, constOp->getResult(0));
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

/// Gather ranges for all the values in `values`. Appends to the existing
/// vector.
static LogicalResult collectRanges(DataFlowSolver &solver, ValueRange values,
                                   SmallVectorImpl<ConstantIntRanges> &ranges) {
  for (Value val : values) {
    auto *maybeInferredRange =
        solver.lookupState<IntegerValueRangeLattice>(val);
    if (!maybeInferredRange || maybeInferredRange->getValue().isUninitialized())
      return failure();

    const ConstantIntRanges &inferredRange =
        maybeInferredRange->getValue().getValue();
    ranges.push_back(inferredRange);
  }
  return success();
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

namespace {
// Enum for tracking which type of truncation should be performed
// to narrow an operation, if any.
enum class CastKind : uint8_t { None, Signed, Unsigned, Both };
} // namespace

/// If the values within `range` can be represented using only `width` bits,
/// return the kind of truncation needed to preserve that property.
///
/// This check relies on the fact that the signed and unsigned ranges are both
/// always correct, but that one might be an approximation of the other,
/// so we want to use the correct truncation operation.
static CastKind checkTruncatability(const ConstantIntRanges &range,
                                    unsigned targetWidth) {
  unsigned srcWidth = range.smin().getBitWidth();
  if (srcWidth <= targetWidth)
    return CastKind::None;
  unsigned removedWidth = srcWidth - targetWidth;
  // The sign bits need to extend into the sign bit of the target width. For
  // example, if we're truncating 64 bits to 32, we need 64 - 32 + 1 = 33 sign
  // bits.
  bool canTruncateSigned =
      range.smin().getNumSignBits() >= (removedWidth + 1) &&
      range.smax().getNumSignBits() >= (removedWidth + 1);
  bool canTruncateUnsigned = range.umin().countLeadingZeros() >= removedWidth &&
                             range.umax().countLeadingZeros() >= removedWidth;
  if (canTruncateSigned && canTruncateUnsigned)
    return CastKind::Both;
  if (canTruncateSigned)
    return CastKind::Signed;
  if (canTruncateUnsigned)
    return CastKind::Unsigned;
  return CastKind::None;
}

static CastKind mergeCastKinds(CastKind lhs, CastKind rhs) {
  if (lhs == CastKind::None || rhs == CastKind::None)
    return CastKind::None;
  if (lhs == CastKind::Both)
    return rhs;
  if (rhs == CastKind::Both)
    return lhs;
  if (lhs == rhs)
    return lhs;
  return CastKind::None;
}

static Value doCast(OpBuilder &builder, Location loc, Value src, Type dstType,
                    CastKind castKind) {
  Type srcType = src.getType();
  assert(isa<VectorType>(srcType) == isa<VectorType>(dstType) &&
         "Mixing vector and non-vector types");
  assert(castKind != CastKind::None && "Can't cast when casting isn't allowed");
  Type srcElemType = getElementTypeOrSelf(srcType);
  Type dstElemType = getElementTypeOrSelf(dstType);
  assert(srcElemType.isIntOrIndex() && "Invalid src type");
  assert(dstElemType.isIntOrIndex() && "Invalid dst type");
  if (srcType == dstType)
    return src;

  if (isa<IndexType>(srcElemType) || isa<IndexType>(dstElemType)) {
    if (castKind == CastKind::Signed)
      return builder.create<arith::IndexCastOp>(loc, dstType, src);
    return builder.create<arith::IndexCastUIOp>(loc, dstType, src);
  }

  auto srcInt = cast<IntegerType>(srcElemType);
  auto dstInt = cast<IntegerType>(dstElemType);
  if (dstInt.getWidth() < srcInt.getWidth())
    return builder.create<arith::TruncIOp>(loc, dstType, src);

  if (castKind == CastKind::Signed)
    return builder.create<arith::ExtSIOp>(loc, dstType, src);
  return builder.create<arith::ExtUIOp>(loc, dstType, src);
}

struct NarrowElementwise final : OpTraitRewritePattern<OpTrait::Elementwise> {
  NarrowElementwise(MLIRContext *context, DataFlowSolver &s,
                    ArrayRef<unsigned> target)
      : OpTraitRewritePattern(context), solver(s), targetBitwidths(target) {}

  using OpTraitRewritePattern::OpTraitRewritePattern;
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumResults() == 0)
      return rewriter.notifyMatchFailure(op, "can't narrow resultless op");

    SmallVector<ConstantIntRanges> ranges;
    if (failed(collectRanges(solver, op->getOperands(), ranges)))
      return rewriter.notifyMatchFailure(op, "input without specified range");
    if (failed(collectRanges(solver, op->getResults(), ranges)))
      return rewriter.notifyMatchFailure(op, "output without specified range");

    Type srcType = op->getResult(0).getType();
    if (!llvm::all_equal(op->getResultTypes()))
      return rewriter.notifyMatchFailure(op, "mismatched result types");
    if (op->getNumOperands() == 0 ||
        !llvm::all_of(op->getOperandTypes(),
                      [=](Type t) { return t == srcType; }))
      return rewriter.notifyMatchFailure(
          op, "no operands or operand types don't match result type");

    for (unsigned targetBitwidth : targetBitwidths) {
      CastKind castKind = CastKind::Both;
      for (const ConstantIntRanges &range : ranges) {
        castKind = mergeCastKinds(castKind,
                                  checkTruncatability(range, targetBitwidth));
        if (castKind == CastKind::None)
          break;
      }
      if (castKind == CastKind::None)
        continue;
      Type targetType = getTargetType(srcType, targetBitwidth);
      if (targetType == srcType)
        continue;

      Location loc = op->getLoc();
      IRMapping mapping;
      for (auto [arg, argRange] : llvm::zip_first(op->getOperands(), ranges)) {
        CastKind argCastKind = castKind;
        // When dealing with `index` values, preserve non-negativity in the
        // index_casts since we can't recover this in unsigned when equivalent.
        if (argCastKind == CastKind::Signed && argRange.smin().isNonNegative())
          argCastKind = CastKind::Both;
        Value newArg = doCast(rewriter, loc, arg, targetType, argCastKind);
        mapping.map(arg, newArg);
      }

      Operation *newOp = rewriter.clone(*op, mapping);
      rewriter.modifyOpInPlace(newOp, [&]() {
        for (OpResult res : newOp->getResults()) {
          res.setType(targetType);
        }
      });
      SmallVector<Value> newResults;
      for (auto [newRes, oldRes] :
           llvm::zip_equal(newOp->getResults(), op->getResults())) {
        Value castBack = doCast(rewriter, loc, newRes, srcType, castKind);
        copyIntegerRange(solver, oldRes, castBack);
        newResults.push_back(castBack);
      }

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

    SmallVector<ConstantIntRanges> ranges;
    if (failed(collectRanges(solver, op.getOperands(), ranges)))
      return failure();
    const ConstantIntRanges &lhsRange = ranges[0];
    const ConstantIntRanges &rhsRange = ranges[1];

    Type srcType = lhs.getType();
    for (unsigned targetBitwidth : targetBitwidths) {
      CastKind lhsCastKind = checkTruncatability(lhsRange, targetBitwidth);
      CastKind rhsCastKind = checkTruncatability(rhsRange, targetBitwidth);
      CastKind castKind = mergeCastKinds(lhsCastKind, rhsCastKind);
      // Note: this includes target width > src width.
      if (castKind == CastKind::None)
        continue;

      Type targetType = getTargetType(srcType, targetBitwidth);
      if (targetType == srcType)
        continue;

      Location loc = op->getLoc();
      IRMapping mapping;
      Value lhsCast = doCast(rewriter, loc, lhs, targetType, lhsCastKind);
      Value rhsCast = doCast(rewriter, loc, rhs, targetType, rhsCastKind);
      mapping.map(lhs, lhsCast);
      mapping.map(rhs, rhsCast);

      Operation *newOp = rewriter.clone(*op, mapping);
      copyIntegerRange(solver, op.getResult(), newOp->getResult(0));
      rewriter.replaceOp(op, newOp->getResults());
      return success();
    }
    return failure();
  }

private:
  DataFlowSolver &solver;
  SmallVector<unsigned, 4> targetBitwidths;
};

/// Fold index_cast(index_cast(%arg: i8, index), i8) -> %arg
/// This pattern assumes all passed `targetBitwidths` are not wider than index
/// type.
template <typename CastOp>
struct FoldIndexCastChain final : OpRewritePattern<CastOp> {
  FoldIndexCastChain(MLIRContext *context, ArrayRef<unsigned> target)
      : OpRewritePattern<CastOp>(context), targetBitwidths(target) {}

  LogicalResult matchAndRewrite(CastOp op,
                                PatternRewriter &rewriter) const override {
    auto srcOp = op.getIn().template getDefiningOp<CastOp>();
    if (!srcOp)
      return rewriter.notifyMatchFailure(op, "doesn't come from an index cast");

    Value src = srcOp.getIn();
    if (src.getType() != op.getType())
      return rewriter.notifyMatchFailure(op, "outer types don't match");

    if (!srcOp.getType().isIndex())
      return rewriter.notifyMatchFailure(op, "intermediate type isn't index");

    auto intType = dyn_cast<IntegerType>(op.getType());
    if (!intType || !llvm::is_contained(targetBitwidths, intType.getWidth()))
      return failure();

    rewriter.replaceOp(op, src);
    return success();
  }

private:
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

    if (failed(applyPatternsGreedily(op, std::move(patterns), config)))
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
    // We specifically need bottom-up traversal as cmpi pattern needs range
    // data, attached to its original argument values.
    config.useTopDownTraversal = false;
    config.listener = &listener;

    if (failed(applyPatternsGreedily(op, std::move(patterns), config)))
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
  patterns.add<FoldIndexCastChain<arith::IndexCastUIOp>,
               FoldIndexCastChain<arith::IndexCastOp>>(patterns.getContext(),
                                                       bitwidthsSupported);
}

std::unique_ptr<Pass> mlir::arith::createIntRangeOptimizationsPass() {
  return std::make_unique<IntRangeOptimizationsPass>();
}
