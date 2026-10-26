//===- LoopSpecialization.cpp - scf.parallel/SCR.for specialization -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Specializes parallel loops and for loops for easier unrolling and
// vectorization.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SCF/Transforms/Passes.h"

#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Utils/AffineCanonicalizationUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_SCFFORLOOPPEELING
#define GEN_PASS_DEF_SCFFORLOOPSPECIALIZATION
#define GEN_PASS_DEF_SCFPARALLELLOOPSPECIALIZATION
#include "mlir/Dialect/SCF/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::affine;
using scf::ForOp;
using scf::ParallelOp;

/// Rewrite a parallel loop with bounds defined by an affine.min with a constant
/// into 2 loops after checking if the bounds are equal to that constant. This
/// is beneficial if the loop will almost always have the constant bound and
/// that version can be fully unrolled and vectorized.
static void specializeParallelLoopForUnrolling(ParallelOp op) {
  SmallVector<int64_t, 2> constantIndices;
  constantIndices.reserve(op.getUpperBound().size());
  for (auto bound : op.getUpperBound()) {
    auto minOp = bound.getDefiningOp<AffineMinOp>();
    if (!minOp)
      return;
    int64_t minConstant = std::numeric_limits<int64_t>::max();
    for (AffineExpr expr : minOp.getMap().getResults()) {
      if (auto constantIndex = dyn_cast<AffineConstantExpr>(expr))
        minConstant = std::min(minConstant, constantIndex.getValue());
    }
    if (minConstant == std::numeric_limits<int64_t>::max())
      return;
    constantIndices.push_back(minConstant);
  }

  OpBuilder b(op);
  IRMapping map;
  Value cond;
  for (auto bound : llvm::zip(op.getUpperBound(), constantIndices)) {
    Value constant =
        arith::ConstantIndexOp::create(b, op.getLoc(), std::get<1>(bound));
    Value cmp = arith::CmpIOp::create(b, op.getLoc(), arith::CmpIPredicate::eq,
                                      std::get<0>(bound), constant);
    cond = cond ? arith::AndIOp::create(b, op.getLoc(), cond, cmp) : cmp;
    map.map(std::get<0>(bound), constant);
  }
  auto ifOp = scf::IfOp::create(b, op.getLoc(), cond, /*withElseRegion=*/true);
  ifOp.getThenBodyBuilder().clone(*op.getOperation(), map);
  ifOp.getElseBodyBuilder().clone(*op.getOperation());
  op.erase();
}

/// Rewrite a for loop with bounds defined by an affine.min with a constant into
/// 2 loops after checking if the bounds are equal to that constant. This is
/// beneficial if the loop will almost always have the constant bound and that
/// version can be fully unrolled and vectorized.
static void specializeForLoopForUnrolling(ForOp op) {
  auto bound = op.getUpperBound();
  auto minOp = bound.getDefiningOp<AffineMinOp>();
  if (!minOp)
    return;
  AffineMap canonicalizedMap = minOp.getMap();
  SmallVector<Value> operands(minOp.getMapOperands());
  affine::canonicalizeMapAndOperands(&canonicalizedMap, &operands);
  int64_t minConstant = std::numeric_limits<int64_t>::max();
  for (AffineExpr expr : canonicalizedMap.getResults()) {
    if (auto constantIndex = dyn_cast<AffineConstantExpr>(expr))
      minConstant = std::min(minConstant, constantIndex.getValue());
  }
  if (minConstant == std::numeric_limits<int64_t>::max())
    return;

  OpBuilder b(op);
  IRMapping map;
  Value constant = arith::ConstantOp::create(
      b, op.getLoc(),
      IntegerAttr::get(op.getUpperBound().getType(), minConstant));
  Value cond = arith::CmpIOp::create(b, op.getLoc(), arith::CmpIPredicate::eq,
                                     bound, constant);
  map.map(bound, constant);
  auto ifOp = scf::IfOp::create(b, op.getLoc(), cond, /*withElseRegion=*/true);
  ifOp.getThenBodyBuilder().clone(*op.getOperation(), map);
  ifOp.getElseBodyBuilder().clone(*op.getOperation());
  op.erase();
}

/// Rewrite a for loop with bounds/step that potentially do not divide evenly
/// into a for loop where the step divides the iteration space evenly, followed
/// by an scf.if for the last (partial) iteration (if any).
///
/// This function rewrites the given scf.for loop in-place and creates a new
/// scf.if operation for the last iteration. It replaces all uses of the
/// unpeeled loop with the results of the newly generated scf.if.
///
/// The newly generated scf.if operation is returned via `ifOp`. The boundary
/// at which the loop is split (new upper bound) is returned via `splitBound`.
/// The return value indicates whether the loop was rewritten or not.
///
/// Note: Loops with a step size of 0 cannot be peeled. Applying this function
/// to such a loop may result in IR with undefined behavior.
static LogicalResult peelForLoop(RewriterBase &b, ForOp forOp,
                                 ForOp &partialIteration, Value &splitBound) {
  RewriterBase::InsertionGuard guard(b);
  Value lbVal = forOp.getLowerBound();
  Value ubVal = forOp.getUpperBound();
  Value stepVal = forOp.getStep();
  b.setInsertionPoint(forOp);
  auto loc = forOp.getLoc();

  // Check whether we have a unsigned loop
  if (forOp.getUnsignedCmp()) {
    IntegerAttr lbAttr, ubAttr, stepAttr;
    if (matchPattern(lbVal, m_Constant(&lbAttr)) &&
        matchPattern(ubVal, m_Constant(&ubAttr)) &&
        matchPattern(stepVal, m_Constant(&stepAttr))) {

      // Get the constants as APInt at the loop’s own bit width
      APInt lb = lbAttr.getValue();
      APInt ub = ubAttr.getValue();
      APInt step = stepAttr.getValue();

      if (step.ule(1) || ub.ule(lb) || (ub - lb).urem(step).isZero())
        return failure();
    }

    // split = ub - ((ub - min(lb, ub)) remui step). The min makes an empty loop
    // (ub <= lb) give a zero remainder, so split = ub and both loops stay
    // empty. The wrapping subtraction is exact for ub > lb, even above the
    // signed range.
    Value lbEff = b.createOrFold<arith::MinUIOp>(loc, lbVal, ubVal);
    Value diff = b.createOrFold<arith::SubIOp>(loc, ubVal, lbEff);
    Value rem = b.createOrFold<arith::RemUIOp>(loc, diff, stepVal);
    splitBound = b.createOrFold<arith::SubIOp>(loc, ubVal, rem);

  } else {
    auto lbInt = getConstantIntValue(lbVal);
    auto ubInt = getConstantIntValue(ubVal);
    auto stepInt = getConstantIntValue(stepVal);
    // No specialization necessary if step size is 1. Also bail out in case of
    // an invalid zero or negative step which might have happened during
    // folding.
    if (stepInt && *stepInt <= 1)
      return failure();

    // No specialization necessary if step already divides upper bound evenly.
    // Fast path: lb, ub and step are constants.
    if (lbInt && ubInt && stepInt) {
      int64_t diff;
      // bail out if the subtraction overflows
      if (llvm::SubOverflow(*ubInt, *lbInt, diff) || diff % *stepInt == 0)
        return failure();
    }

    // Only the dynamic path computes the peeling bound with affine.apply, which
    // accepts only index operands.
    if ((!lbInt || !ubInt || !stepInt) &&
        !forOp.getInductionVar().getType().isIndex())
      return failure();

    // Slow path: Examine the ops that define lb, ub and step.
    AffineExpr sym0, sym1, sym2;
    bindSymbols(b.getContext(), sym0, sym1, sym2);
    SmallVector<Value> operands{forOp.getLowerBound(), forOp.getUpperBound(),
                                forOp.getStep()};
    AffineMap map = AffineMap::get(0, 3, {(sym1 - sym0) % sym2});
    affine::fullyComposeAffineMapAndOperands(&map, &operands);
    if (auto constExpr = dyn_cast<AffineConstantExpr>(map.getResult(0)))
      if (constExpr.getValue() == 0)
        return failure();

    // New upper bound: max(%lb, %ub - (%ub - %lb) mod %step). Affine `mod` is
    // non-negative, so without the clamp an empty loop (%lb >= %ub) gets a
    // split bound below %lb and its partial iteration executes.
    auto modMap = AffineMap::get(0, 3, {sym1 - ((sym1 - sym0) % sym2), sym0},
                                 b.getContext());
    splitBound = b.createOrFold<AffineMaxOp>(loc, modMap,
                                             ValueRange{forOp.getLowerBound(),
                                                        forOp.getUpperBound(),
                                                        forOp.getStep()});
    if (splitBound.getType() != forOp.getLowerBound().getType())
      splitBound = b.createOrFold<arith::IndexCastOp>(
          loc, forOp.getLowerBound().getType(), splitBound);
  }
  // Create ForOp for partial iteration.
  b.setInsertionPointAfter(forOp);
  partialIteration = cast<ForOp>(b.clone(*forOp.getOperation()));
  partialIteration.getLowerBoundMutable().assign(splitBound);
  b.replaceAllUsesWith(forOp.getResults(), partialIteration->getResults());
  partialIteration.getInitArgsMutable().assign(forOp->getResults());

  // Set new upper loop bound.
  b.modifyOpInPlace(forOp,
                    [&]() { forOp.getUpperBoundMutable().assign(splitBound); });

  return success();
}

static void rewriteAffineOpAfterPeeling(RewriterBase &rewriter, ForOp forOp,
                                        ForOp partialIteration,
                                        Value previousUb) {
  Value mainIv = forOp.getInductionVar();
  Value partialIv = partialIteration.getInductionVar();
  assert(forOp.getStep() == partialIteration.getStep() &&
         "expected same step in main and partial loop");
  Value step = forOp.getStep();

  forOp.walk([&](Operation *affineOp) {
    if (!isa<AffineMinOp, AffineMaxOp>(affineOp))
      return WalkResult::advance();
    (void)scf::rewritePeeledMinMaxOp(rewriter, affineOp, mainIv, previousUb,
                                     step,
                                     /*insideLoop=*/true);
    return WalkResult::advance();
  });
  partialIteration.walk([&](Operation *affineOp) {
    if (!isa<AffineMinOp, AffineMaxOp>(affineOp))
      return WalkResult::advance();
    (void)scf::rewritePeeledMinMaxOp(rewriter, affineOp, partialIv, previousUb,
                                     step, /*insideLoop=*/false);
    return WalkResult::advance();
  });
}

LogicalResult mlir::scf::peelForLoopAndSimplifyBounds(RewriterBase &rewriter,
                                                      ForOp forOp,
                                                      ForOp &partialIteration) {
  Value previousUb = forOp.getUpperBound();
  Value splitBound;
  if (failed(peelForLoop(rewriter, forOp, partialIteration, splitBound)))
    return failure();

  // The affine.min/max simplification relies on signed reasoning, which does
  // not hold for unsigned loops, so leave those ops untouched.
  if (forOp.getUnsignedCmp())
    return success();

  // Rewrite affine.min and affine.max ops.
  rewriteAffineOpAfterPeeling(rewriter, forOp, partialIteration, previousUb);

  return success();
}

/// Rewrites the original scf::ForOp as two scf::ForOp Ops, the first
/// scf::ForOp corresponds to the first iteration of the loop which can be
/// canonicalized away in the following optimizations. The second loop Op
/// contains the remaining iterations, with a lower bound updated as the
/// original lower bound plus the step (i.e. skips the first iteration).
LogicalResult mlir::scf::peelForLoopFirstIteration(RewriterBase &b, ForOp forOp,
                                                   ForOp &firstIteration) {
  RewriterBase::InsertionGuard guard(b);
  Value lbVal = forOp.getLowerBound();
  Value ubVal = forOp.getUpperBound();
  Value stepVal = forOp.getStep();
  auto loc = forOp.getLoc();

  Value splitBound;
  b.setInsertionPoint(forOp);

  // Check whether we have a unsigned loop
  if (forOp.getUnsignedCmp()) {
    IntegerAttr lbAttr, ubAttr, stepAttr;
    if (matchPattern(forOp.getLowerBound(), m_Constant(&lbAttr)) &&
        matchPattern(forOp.getUpperBound(), m_Constant(&ubAttr)) &&
        matchPattern(forOp.getStep(), m_Constant(&stepAttr))) {

      // Get the constants as APInt at the loop’s own bit width
      APInt lb = lbAttr.getValue();
      APInt ub = ubAttr.getValue();
      APInt step = stepAttr.getValue();

      // bail out if ub <= lb or ub - lb <= step (unsigned constants)
      if (ub.ule(lb) || (ub - lb).ule(step))
        return failure();
    }

    // split = max(lb, min(lb + step, ub)): the first loop runs one iteration,
    // or none if the loop is empty. If lb + step wraps, the loop has at most
    // one iteration and split falls back to lb, so the main loop does the work.
    // use unsigned ops (MinUIOp, MaxUIOp)
    Value sum = b.createOrFold<arith::AddIOp>(loc, lbVal, stepVal);
    Value capped = b.createOrFold<arith::MinUIOp>(loc, sum, ubVal);
    splitBound = b.createOrFold<arith::MaxUIOp>(loc, lbVal, capped);
  } else {
    auto lbInt = getConstantIntValue(lbVal);
    auto ubInt = getConstantIntValue(ubVal);
    auto stepInt = getConstantIntValue(stepVal);

    // Peeling is not needed if there is at most one iteration. scf.for requires
    // a positive step, so the trip count is ceil((UB - LB) / Step) when UB > LB
    // and 0 otherwise. With a positive step, "UB - LB <= Step" therefore covers
    // both the empty loop and the single-iteration loop. Bail out if the
    // subtraction overflows, since we can't reason about the trip count then.
    if (lbInt && ubInt && stepInt) {
      int64_t diff;
      if (llvm::SubOverflow(*ubInt, *lbInt, diff) || diff <= *stepInt)
        return failure();
    }

    // split = max(lb, min(lb + step, ub)): the first loop runs one iteration,
    // or none if the loop is empty. If lb + step wraps, the loop has at most
    // one iteration and split falls back to lb, so the main loop does the work.
    Value sum = b.createOrFold<arith::AddIOp>(loc, lbVal, stepVal);
    Value capped = b.createOrFold<arith::MinSIOp>(loc, sum, ubVal);
    splitBound = b.createOrFold<arith::MaxSIOp>(loc, lbVal, capped);
  }
  // Peel the first iteration.
  firstIteration = cast<ForOp>(b.clone(*forOp.getOperation()));
  b.modifyOpInPlace(firstIteration, [&]() {
    firstIteration.getUpperBoundMutable().assign(splitBound);
  });
  // Update main loop with new lower bound.
  b.modifyOpInPlace(forOp, [&]() {
    forOp.getInitArgsMutable().assign(firstIteration->getResults());
    forOp.getLowerBoundMutable().assign(splitBound);
  });

  return success();
}

static constexpr char kPeeledLoopLabel[] = "__peeled_loop__";
static constexpr char kPartialIterationLabel[] = "__partial_iteration__";

namespace {
struct ForLoopPeelingPattern : public OpRewritePattern<ForOp> {
  ForLoopPeelingPattern(MLIRContext *ctx, bool peelFront, bool skipPartial)
      : OpRewritePattern<ForOp>(ctx), peelFront(peelFront),
        skipPartial(skipPartial) {}

  LogicalResult matchAndRewrite(ForOp forOp,
                                PatternRewriter &rewriter) const override {

    // Do not peel already peeled loops.
    if (forOp->hasDiscardableAttr(kPeeledLoopLabel))
      return failure();

    scf::ForOp partialIteration;
    // The case for peeling the first iteration of the loop.
    if (peelFront) {
      if (failed(
              peelForLoopFirstIteration(rewriter, forOp, partialIteration))) {
        return failure();
      }
    } else {
      if (skipPartial) {
        // No peeling of loops inside the partial iteration of another peeled
        // loop.
        Operation *op = forOp.getOperation();
        while ((op = op->getParentOfType<scf::ForOp>())) {
          if (op->hasDiscardableAttr(kPartialIterationLabel))
            return failure();
        }
      }
      // Apply loop peeling.
      if (failed(
              peelForLoopAndSimplifyBounds(rewriter, forOp, partialIteration)))
        return failure();
    }

    // Apply label, so that the same loop is not rewritten a second time.
    rewriter.modifyOpInPlace(partialIteration, [&]() {
      partialIteration->setDiscardableAttr(kPeeledLoopLabel,
                                           rewriter.getUnitAttr());
      partialIteration->setDiscardableAttr(kPartialIterationLabel,
                                           rewriter.getUnitAttr());
    });
    rewriter.modifyOpInPlace(forOp, [&]() {
      forOp->setDiscardableAttr(kPeeledLoopLabel, rewriter.getUnitAttr());
    });
    return success();
  }

  // If set to true, the first iteration of the loop will be peeled. Otherwise,
  // the unevenly divisible loop will be peeled at the end.
  bool peelFront;

  /// If set to true, loops inside partial iterations of another peeled loop
  /// are not peeled. This reduces the size of the generated code. Partial
  /// iterations are not usually performance critical.
  /// Note: Takes into account the entire chain of parent operations, not just
  /// the direct parent.
  bool skipPartial;
};
} // namespace

namespace {
struct ParallelLoopSpecialization
    : public impl::SCFParallelLoopSpecializationBase<
          ParallelLoopSpecialization> {
  void runOnOperation() override {
    getOperation()->walk(
        [](ParallelOp op) { specializeParallelLoopForUnrolling(op); });
  }
};

struct ForLoopSpecialization
    : public impl::SCFForLoopSpecializationBase<ForLoopSpecialization> {
  void runOnOperation() override {
    getOperation()->walk([](ForOp op) { specializeForLoopForUnrolling(op); });
  }
};

struct ForLoopPeeling : public impl::SCFForLoopPeelingBase<ForLoopPeeling> {
  using impl::SCFForLoopPeelingBase<ForLoopPeeling>::SCFForLoopPeelingBase;

  void runOnOperation() override {
    auto *parentOp = getOperation();
    MLIRContext *ctx = parentOp->getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<ForLoopPeelingPattern>(ctx, peelFront, skipPartial);
    (void)applyPatternsGreedily(parentOp, std::move(patterns));

    // Drop the markers.
    parentOp->walk([](Operation *op) {
      op->removeDiscardableAttr(kPeeledLoopLabel);
      op->removeDiscardableAttr(kPartialIterationLabel);
    });
  }
};
} // namespace

std::unique_ptr<Pass> mlir::createParallelLoopSpecializationPass() {
  return std::make_unique<ParallelLoopSpecialization>();
}

std::unique_ptr<Pass> mlir::createForLoopSpecializationPass() {
  return std::make_unique<ForLoopSpecialization>();
}
