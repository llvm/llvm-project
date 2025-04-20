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
#include "llvm/ADT/DenseMap.h"

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
        b.create<arith::ConstantIndexOp>(op.getLoc(), std::get<1>(bound));
    Value cmp = b.create<arith::CmpIOp>(op.getLoc(), arith::CmpIPredicate::eq,
                                        std::get<0>(bound), constant);
    cond = cond ? b.create<arith::AndIOp>(op.getLoc(), cond, cmp) : cmp;
    map.map(std::get<0>(bound), constant);
  }
  auto ifOp = b.create<scf::IfOp>(op.getLoc(), cond, /*withElseRegion=*/true);
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
  int64_t minConstant = std::numeric_limits<int64_t>::max();
  for (AffineExpr expr : minOp.getMap().getResults()) {
    if (auto constantIndex = dyn_cast<AffineConstantExpr>(expr))
      minConstant = std::min(minConstant, constantIndex.getValue());
  }
  if (minConstant == std::numeric_limits<int64_t>::max())
    return;

  OpBuilder b(op);
  IRMapping map;
  Value constant = b.create<arith::ConstantIndexOp>(op.getLoc(), minConstant);
  Value cond = b.create<arith::CmpIOp>(op.getLoc(), arith::CmpIPredicate::eq,
                                       bound, constant);
  map.map(bound, constant);
  auto ifOp = b.create<scf::IfOp>(op.getLoc(), cond, /*withElseRegion=*/true);
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
static LogicalResult peelForLoop(RewriterBase &b, ForOp forOp,
                                 ForOp &partialIteration, Value &splitBound) {
  RewriterBase::InsertionGuard guard(b);
  auto lbInt = getConstantIntValue(forOp.getLowerBound());
  auto ubInt = getConstantIntValue(forOp.getUpperBound());
  auto stepInt = getConstantIntValue(forOp.getStep());

  // No specialization necessary if step size is 1. Also bail out in case of an
  // invalid zero or negative step which might have happened during folding.
  if (stepInt && *stepInt <= 1)
    return failure();

  // No specialization necessary if step already divides upper bound evenly.
  // Fast path: lb, ub and step are constants.
  if (lbInt && ubInt && stepInt && (*ubInt - *lbInt) % *stepInt == 0)
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

  // New upper bound: %ub - (%ub - %lb) mod %step
  auto modMap = AffineMap::get(0, 3, {sym1 - ((sym1 - sym0) % sym2)});
  b.setInsertionPoint(forOp);
  auto loc = forOp.getLoc();
  splitBound = b.createOrFold<AffineApplyOp>(loc, modMap,
                                             ValueRange{forOp.getLowerBound(),
                                                        forOp.getUpperBound(),
                                                        forOp.getStep()});

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
  auto lbInt = getConstantIntValue(forOp.getLowerBound());
  auto ubInt = getConstantIntValue(forOp.getUpperBound());
  auto stepInt = getConstantIntValue(forOp.getStep());

  // Peeling is not needed if there is one or less iteration.
  if (lbInt && ubInt && stepInt && ceil(float(*ubInt - *lbInt) / *stepInt) <= 1)
    return failure();

  AffineExpr lbSymbol, stepSymbol;
  bindSymbols(b.getContext(), lbSymbol, stepSymbol);

  // New lower bound for main loop: %lb + %step
  auto ubMap = AffineMap::get(0, 2, {lbSymbol + stepSymbol});
  b.setInsertionPoint(forOp);
  auto loc = forOp.getLoc();
  Value splitBound = b.createOrFold<AffineApplyOp>(
      loc, ubMap, ValueRange{forOp.getLowerBound(), forOp.getStep()});

  // Peel the first iteration.
  IRMapping map;
  map.map(forOp.getUpperBound(), splitBound);
  firstIteration = cast<ForOp>(b.clone(*forOp.getOperation(), map));

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
    if (forOp->hasAttr(kPeeledLoopLabel))
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
          if (op->hasAttr(kPartialIterationLabel))
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
      partialIteration->setAttr(kPeeledLoopLabel, rewriter.getUnitAttr());
      partialIteration->setAttr(kPartialIterationLabel, rewriter.getUnitAttr());
    });
    rewriter.modifyOpInPlace(forOp, [&]() {
      forOp->setAttr(kPeeledLoopLabel, rewriter.getUnitAttr());
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
  void runOnOperation() override {
    auto *parentOp = getOperation();
    MLIRContext *ctx = parentOp->getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<ForLoopPeelingPattern>(ctx, peelFront, skipPartial);
    (void)applyPatternsGreedily(parentOp, std::move(patterns));

    // Drop the markers.
    parentOp->walk([](Operation *op) {
      op->removeAttr(kPeeledLoopLabel);
      op->removeAttr(kPartialIterationLabel);
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

std::unique_ptr<Pass> mlir::createForLoopPeelingPass() {
  return std::make_unique<ForLoopPeeling>();
}
