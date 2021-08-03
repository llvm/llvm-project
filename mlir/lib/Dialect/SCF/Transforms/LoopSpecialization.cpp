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
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/Passes.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SCF/Transforms.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
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
  constantIndices.reserve(op.upperBound().size());
  for (auto bound : op.upperBound()) {
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
  for (auto bound : llvm::zip(op.upperBound(), constantIndices)) {
    Value constant = b.create<ConstantIndexOp>(op.getLoc(), std::get<1>(bound));
    Value cmp = b.create<CmpIOp>(op.getLoc(), CmpIPredicate::eq,
                                 std::get<0>(bound), constant);
    cond = cond ? b.create<AndOp>(op.getLoc(), cond, cmp) : cmp;
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
  auto bound = op.upperBound();
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
  Value constant = b.create<ConstantIndexOp>(op.getLoc(), minConstant);
  Value cond =
      b.create<CmpIOp>(op.getLoc(), CmpIPredicate::eq, bound, constant);
  map.map(bound, constant);
  auto ifOp = b.create<scf::IfOp>(op.getLoc(), cond, /*withElseRegion=*/true);
  ifOp.getThenBodyBuilder().clone(*op.getOperation(), map);
  ifOp.getElseBodyBuilder().clone(*op.getOperation());
  op.erase();
}

/// Rewrite a for loop with bounds/step that potentially do not divide evenly
/// into a for loop where the step divides the iteration space evenly, followed
/// by an scf.if for the last (partial) iteration (if any).
LogicalResult mlir::scf::peelForLoop(RewriterBase &b, ForOp forOp,
                                     scf::IfOp &ifOp) {
  RewriterBase::InsertionGuard guard(b);
  auto lbInt = getConstantIntValue(forOp.lowerBound());
  auto ubInt = getConstantIntValue(forOp.upperBound());
  auto stepInt = getConstantIntValue(forOp.step());

  // No specialization necessary if step already divides upper bound evenly.
  if (lbInt && ubInt && stepInt && (*ubInt - *lbInt) % *stepInt == 0)
    return failure();
  // No specialization necessary if step size is 1.
  if (stepInt == static_cast<int64_t>(1))
    return failure();

  auto loc = forOp.getLoc();
  AffineExpr dim0, dim1, dim2;
  bindDims(b.getContext(), dim0, dim1, dim2);
  // New upper bound: %ub - (%ub - %lb) mod %step
  auto modMap = AffineMap::get(3, 0, {dim1 - ((dim1 - dim0) % dim2)});
  Value splitBound = b.createOrFold<AffineApplyOp>(
      loc, modMap,
      ValueRange{forOp.lowerBound(), forOp.upperBound(), forOp.step()});

  // Set new upper loop bound.
  Value previousUb = forOp.upperBound();
  b.updateRootInPlace(forOp,
                      [&]() { forOp.upperBoundMutable().assign(splitBound); });
  b.setInsertionPointAfter(forOp);

  // Do we need one more iteration?
  Value hasMoreIter =
      b.create<CmpIOp>(loc, CmpIPredicate::slt, splitBound, previousUb);

  // Create IfOp for last iteration.
  auto resultTypes = llvm::to_vector<4>(
      llvm::map_range(forOp.initArgs(), [](Value v) { return v.getType(); }));
  ifOp = b.create<scf::IfOp>(loc, resultTypes, hasMoreIter,
                             /*withElseRegion=*/!resultTypes.empty());
  forOp.replaceAllUsesWith(ifOp->getResults());

  // Build then case.
  BlockAndValueMapping bvm;
  bvm.map(forOp.region().getArgument(0), splitBound);
  for (auto it : llvm::zip(forOp.region().getArguments().drop_front(),
                           forOp->getResults())) {
    bvm.map(std::get<0>(it), std::get<1>(it));
  }
  b.cloneRegionBefore(forOp.region(), ifOp.thenRegion(),
                      ifOp.thenRegion().begin(), bvm);
  // Build else case.
  if (!resultTypes.empty())
    ifOp.getElseBodyBuilder().create<scf::YieldOp>(loc, forOp->getResults());

  return success();
}

static constexpr char kPeeledLoopLabel[] = "__peeled_loop__";

namespace {
struct ForLoopPeelingPattern : public OpRewritePattern<ForOp> {
  using OpRewritePattern<ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ForOp forOp,
                                PatternRewriter &rewriter) const override {
    if (forOp->hasAttr(kPeeledLoopLabel))
      return failure();

    scf::IfOp ifOp;
    if (failed(peelForLoop(rewriter, forOp, ifOp)))
      return failure();
    // Apply label, so that the same loop is not rewritten a second time.
    rewriter.updateRootInPlace(forOp, [&]() {
      forOp->setAttr(kPeeledLoopLabel, rewriter.getUnitAttr());
    });

    return success();
  }
};
} // namespace

namespace {
struct ParallelLoopSpecialization
    : public SCFParallelLoopSpecializationBase<ParallelLoopSpecialization> {
  void runOnFunction() override {
    getFunction().walk(
        [](ParallelOp op) { specializeParallelLoopForUnrolling(op); });
  }
};

struct ForLoopSpecialization
    : public SCFForLoopSpecializationBase<ForLoopSpecialization> {
  void runOnFunction() override {
    getFunction().walk([](ForOp op) { specializeForLoopForUnrolling(op); });
  }
};

struct ForLoopPeeling : public SCFForLoopPeelingBase<ForLoopPeeling> {
  void runOnFunction() override {
    FuncOp funcOp = getFunction();
    MLIRContext *ctx = funcOp.getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<ForLoopPeelingPattern>(ctx);
    (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));

    // Drop the marker.
    funcOp.walk([](ForOp op) { op->removeAttr(kPeeledLoopLabel); });
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
