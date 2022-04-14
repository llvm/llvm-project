//===- MergeCleanups.cpp - merge simple return/yield blocks ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/CIR/Passes.h"

#include "PassDetail.h"
#include "mlir/Dialect/CIR/IR/CIRDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace cir;

namespace {

template <typename ScopeLikeOpTy>
struct SimplifyRetYieldBlocks : public mlir::OpRewritePattern<ScopeLikeOpTy> {
  using OpRewritePattern<ScopeLikeOpTy>::OpRewritePattern;
  mlir::LogicalResult replaceScopeLikeOp(PatternRewriter &rewriter,
                                         ScopeLikeOpTy scopeLikeOp) const;

  SimplifyRetYieldBlocks(mlir::MLIRContext *context)
      : OpRewritePattern<ScopeLikeOpTy>(context, /*benefit=*/1) {}

  mlir::LogicalResult
  checkAndRewriteRegion(mlir::Region &r,
                        mlir::PatternRewriter &rewriter) const {
    auto &blocks = r.getBlocks();

    if (blocks.size() <= 1)
      return failure();

    // Rewrite something like this:
    //
    // cir.if %2 {
    //    %3 = cir.cst(3 : i32) : i32
    //    cir.br ^bb1
    //  ^bb1:  // pred: ^bb0
    //    cir.return %3 : i32
    //  }
    //
    // to this:
    //
    // cir.if %2 {
    //    %3 = cir.cst(3 : i32) : i32
    //    cir.return %3 : i32
    // }
    //
    SmallPtrSet<mlir::Block *, 4> candidateBlocks;
    for (Block &block : blocks) {
      if (block.isEntryBlock())
        continue;

      auto yieldVars = block.getOps<cir::YieldOp>();
      for (cir::YieldOp yield : yieldVars)
        candidateBlocks.insert(yield.getOperation()->getBlock());

      auto retVars = block.getOps<cir::ReturnOp>();
      for (cir::ReturnOp ret : retVars)
        candidateBlocks.insert(ret.getOperation()->getBlock());
    }

    bool Changed = false;
    for (auto *mergeSource : candidateBlocks) {
      if (!(mergeSource->hasNoSuccessors() && mergeSource->hasOneUse()))
        continue;
      auto *mergeDest = mergeSource->getSinglePredecessor();
      if (!mergeDest || mergeDest->getNumSuccessors() != 1)
        continue;
      rewriter.eraseOp(mergeDest->getTerminator());
      rewriter.mergeBlocks(mergeSource, mergeDest);
      Changed = true;
    }

    return Changed ? success() : failure();
  }

  mlir::LogicalResult
  matchAndRewrite(ScopeLikeOpTy op,
                  mlir::PatternRewriter &rewriter) const override {
    return replaceScopeLikeOp(rewriter, op);
  }
};

// Specialize the template to account for the different build signatures for
// IfOp, ScopeOp, FuncOp, SwitchOp, LoopOp.
template <>
mlir::LogicalResult
SimplifyRetYieldBlocks<IfOp>::replaceScopeLikeOp(PatternRewriter &rewriter,
                                                 IfOp ifOp) const {
  bool regionChanged = false;
  if (checkAndRewriteRegion(ifOp.getThenRegion(), rewriter).succeeded())
    regionChanged = true;
  if (checkAndRewriteRegion(ifOp.getElseRegion(), rewriter).succeeded())
    regionChanged = true;
  return regionChanged ? success() : failure();
}

template <>
mlir::LogicalResult
SimplifyRetYieldBlocks<ScopeOp>::replaceScopeLikeOp(PatternRewriter &rewriter,
                                                    ScopeOp scopeOp) const {
  bool regionChanged = false;
  if (checkAndRewriteRegion(scopeOp.getRegion(), rewriter).succeeded())
    regionChanged = true;
  return regionChanged ? success() : failure();
}

template <>
mlir::LogicalResult SimplifyRetYieldBlocks<mlir::FuncOp>::replaceScopeLikeOp(
    PatternRewriter &rewriter, mlir::FuncOp funcOp) const {
  bool regionChanged = false;
  if (checkAndRewriteRegion(funcOp.getRegion(), rewriter).succeeded())
    regionChanged = true;
  return regionChanged ? success() : failure();
}

template <>
mlir::LogicalResult SimplifyRetYieldBlocks<cir::SwitchOp>::replaceScopeLikeOp(
    PatternRewriter &rewriter, cir::SwitchOp switchOp) const {
  bool regionChanged = false;
  for (auto &r : switchOp.getRegions()) {
    if (checkAndRewriteRegion(r, rewriter).succeeded())
      regionChanged = true;
  }

  return regionChanged ? success() : failure();
}

template <>
mlir::LogicalResult SimplifyRetYieldBlocks<cir::LoopOp>::replaceScopeLikeOp(
    PatternRewriter &rewriter, cir::LoopOp loopOp) const {
  return checkAndRewriteRegion(loopOp.getBody(), rewriter);
}

void getMergeCleanupsPatterns(RewritePatternSet &results,
                              MLIRContext *context) {
  results.add<SimplifyRetYieldBlocks<IfOp>, SimplifyRetYieldBlocks<ScopeOp>,
              SimplifyRetYieldBlocks<mlir::FuncOp>,
              SimplifyRetYieldBlocks<cir::SwitchOp>,
              SimplifyRetYieldBlocks<cir::LoopOp>>(context);
}

struct MergeCleanupsPass : public MergeCleanupsBase<MergeCleanupsPass> {
  MergeCleanupsPass() = default;
  void runOnOperation() override;
};

// The same operation rewriting done here could have been performed
// by CanonicalizerPass (adding hasCanonicalizer for target Ops and implementing
// the same from above in CIRDialects.cpp). However, it's currently too
// aggressive for static analysis purposes, since it might remove things where
// a diagnostic can be generated.
//
// FIXME: perhaps we can add one more mode to GreedyRewriteConfig to
// disable this behavior.
void MergeCleanupsPass::runOnOperation() {
  auto op = getOperation();
  mlir::RewritePatternSet patterns(&getContext());
  getMergeCleanupsPatterns(patterns, &getContext());
  FrozenRewritePatternSet frozenPatterns(std::move(patterns));

  SmallVector<Operation *> opsToSimplify;
  op->walk([&](Operation *op) {
    if (isa<cir::IfOp, cir::ScopeOp, mlir::FuncOp, cir::SwitchOp, cir::LoopOp>(
            op))
      opsToSimplify.push_back(op);
  });

  for (auto *o : opsToSimplify) {
    bool erase = false;
    (void)applyOpPatternsAndFold(o, frozenPatterns, GreedyRewriteConfig(),
                                 &erase);
  }
}
} // namespace

std::unique_ptr<Pass> mlir::createMergeCleanupsPass() {
  return std::make_unique<MergeCleanupsPass>();
}
