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

#include "PassDetail.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Utils/AffineCanonicalizationUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseMap.h"

using namespace mlir;
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
    for (AffineExpr expr : minOp.map().getResults()) {
      if (auto constantIndex = expr.dyn_cast<AffineConstantExpr>())
        minConstant = std::min(minConstant, constantIndex.getValue());
    }
    if (minConstant == std::numeric_limits<int64_t>::max())
      return;
    constantIndices.push_back(minConstant);
  }

  OpBuilder b(op);
  BlockAndValueMapping map;
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
  for (AffineExpr expr : minOp.map().getResults()) {
    if (auto constantIndex = expr.dyn_cast<AffineConstantExpr>())
      minConstant = std::min(minConstant, constantIndex.getValue());
  }
  if (minConstant == std::numeric_limits<int64_t>::max())
    return;

  OpBuilder b(op);
  BlockAndValueMapping map;
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

  // No specialization necessary if step already divides upper bound evenly.
  if (lbInt && ubInt && stepInt && (*ubInt - *lbInt) % *stepInt == 0)
    return failure();
  // No specialization necessary if step size is 1.
  if (stepInt == static_cast<int64_t>(1))
    return failure();

  auto loc = forOp.getLoc();
  AffineExpr sym0, sym1, sym2;
  bindSymbols(b.getContext(), sym0, sym1, sym2);
  // New upper bound: %ub - (%ub - %lb) mod %step
  auto modMap = AffineMap::get(0, 3, {sym1 - ((sym1 - sym0) % sym2)});
  b.setInsertionPoint(forOp);
  splitBound = b.createOrFold<AffineApplyOp>(loc, modMap,
                                             ValueRange{forOp.getLowerBound(),
                                                        forOp.getUpperBound(),
                                                        forOp.getStep()});

  // Create ForOp for partial iteration.
  b.setInsertionPointAfter(forOp);
  partialIteration = cast<ForOp>(b.clone(*forOp.getOperation()));
  partialIteration.getLowerBoundMutable().assign(splitBound);
  forOp.replaceAllUsesWith(partialIteration->getResults());
  partialIteration.getInitArgsMutable().assign(forOp->getResults());

  // Set new upper loop bound.
  b.updateRootInPlace(
      forOp, [&]() { forOp.getUpperBoundMutable().assign(splitBound); });

  return success();
}

template <typename OpTy, bool IsMin>
static void rewriteAffineOpAfterPeeling(RewriterBase &rewriter, ForOp forOp,
                                        ForOp partialIteration,
                                        Value previousUb) {
  Value mainIv = forOp.getInductionVar();
  Value partialIv = partialIteration.getInductionVar();
  assert(forOp.getStep() == partialIteration.getStep() &&
         "expected same step in main and partial loop");
  Value step = forOp.getStep();

  forOp.walk([&](OpTy affineOp) {
    AffineMap map = affineOp.getAffineMap();
    (void)scf::rewritePeeledMinMaxOp(rewriter, affineOp, map,
                                     affineOp.operands(), IsMin, mainIv,
                                     previousUb, step,
                                     /*insideLoop=*/true);
  });
  partialIteration.walk([&](OpTy affineOp) {
    AffineMap map = affineOp.getAffineMap();
    (void)scf::rewritePeeledMinMaxOp(rewriter, affineOp, map,
                                     affineOp.operands(), IsMin, partialIv,
                                     previousUb, step, /*insideLoop=*/false);
  });
}

LogicalResult mlir::scf::peelAndCanonicalizeForLoop(RewriterBase &rewriter,
                                                    ForOp forOp,
                                                    ForOp &partialIteration) {
  Value previousUb = forOp.getUpperBound();
  Value splitBound;
  if (failed(peelForLoop(rewriter, forOp, partialIteration, splitBound)))
    return failure();

  // Rewrite affine.min and affine.max ops.
  rewriteAffineOpAfterPeeling<AffineMinOp, /*IsMin=*/true>(
      rewriter, forOp, partialIteration, previousUb);
  rewriteAffineOpAfterPeeling<AffineMaxOp, /*IsMin=*/false>(
      rewriter, forOp, partialIteration, previousUb);

  return success();
}

static constexpr char kPeeledLoopLabel[] = "__peeled_loop__";
static constexpr char kPartialIterationLabel[] = "__partial_iteration__";

namespace {
struct ForLoopPeelingPattern : public OpRewritePattern<ForOp> {
  ForLoopPeelingPattern(MLIRContext *ctx, bool skipPartial)
      : OpRewritePattern<ForOp>(ctx), skipPartial(skipPartial) {}

  LogicalResult matchAndRewrite(ForOp forOp,
                                PatternRewriter &rewriter) const override {
    // Do not peel already peeled loops.
    if (forOp->hasAttr(kPeeledLoopLabel))
      return failure();
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
    scf::ForOp partialIteration;
    if (failed(peelAndCanonicalizeForLoop(rewriter, forOp, partialIteration)))
      return failure();
    // Apply label, so that the same loop is not rewritten a second time.
    partialIteration->setAttr(kPeeledLoopLabel, rewriter.getUnitAttr());
    rewriter.updateRootInPlace(forOp, [&]() {
      forOp->setAttr(kPeeledLoopLabel, rewriter.getUnitAttr());
    });
    partialIteration->setAttr(kPartialIterationLabel, rewriter.getUnitAttr());
    return success();
  }

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
    : public SCFParallelLoopSpecializationBase<ParallelLoopSpecialization> {
  void runOnOperation() override {
    getOperation().walk(
        [](ParallelOp op) { specializeParallelLoopForUnrolling(op); });
  }
};

struct ForLoopSpecialization
    : public SCFForLoopSpecializationBase<ForLoopSpecialization> {
  void runOnOperation() override {
    getOperation().walk([](ForOp op) { specializeForLoopForUnrolling(op); });
  }
};

struct ForLoopPeeling : public SCFForLoopPeelingBase<ForLoopPeeling> {
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *ctx = funcOp.getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<ForLoopPeelingPattern>(ctx, skipPartial);
    (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));

    // Drop the markers.
    funcOp.walk([](Operation *op) {
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
