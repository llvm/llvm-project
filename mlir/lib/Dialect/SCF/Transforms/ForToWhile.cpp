//===- ForToWhile.cpp - scf.for to scf.while loop conversion --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Transforms SCF.ForOp's into SCF.WhileOp's.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SCF/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_SCFFORTOWHILELOOP
#include "mlir/Dialect/SCF/Transforms/Passes.h.inc"
} // namespace mlir

using namespace llvm;
using namespace mlir;
using scf::ForOp;
using scf::WhileOp;

namespace {

struct ForLoopLoweringPattern : public OpRewritePattern<ForOp> {
  using OpRewritePattern<ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ForOp forOp,
                                PatternRewriter &rewriter) const override {
    // Generate type signature for the loop-carried values. The induction
    // variable is placed first, followed by the forOp.iterArgs.
    SmallVector<Type> lcvTypes;
    SmallVector<Location> lcvLocs;
    lcvTypes.push_back(forOp.getInductionVar().getType());
    lcvLocs.push_back(forOp.getInductionVar().getLoc());
    for (Value value : forOp.getInitArgs()) {
      lcvTypes.push_back(value.getType());
      lcvLocs.push_back(value.getLoc());
    }

    // Build scf.WhileOp
    SmallVector<Value> initArgs;
    initArgs.push_back(forOp.getLowerBound());
    llvm::append_range(initArgs, forOp.getInitArgs());
    auto whileOp = rewriter.create<WhileOp>(forOp.getLoc(), lcvTypes, initArgs,
                                            forOp->getAttrs());

    // 'before' region contains the loop condition and forwarding of iteration
    // arguments to the 'after' region.
    auto *beforeBlock = rewriter.createBlock(
        &whileOp.getBefore(), whileOp.getBefore().begin(), lcvTypes, lcvLocs);
    rewriter.setInsertionPointToStart(&whileOp.getBefore().front());
    auto cmpOp = rewriter.create<arith::CmpIOp>(
        whileOp.getLoc(), arith::CmpIPredicate::slt,
        beforeBlock->getArgument(0), forOp.getUpperBound());
    rewriter.create<scf::ConditionOp>(whileOp.getLoc(), cmpOp.getResult(),
                                      beforeBlock->getArguments());

    // Inline for-loop body into an executeRegion operation in the "after"
    // region. The return type of the execRegionOp does not contain the
    // iv - yields in the source for-loop contain only iterArgs.
    auto *afterBlock = rewriter.createBlock(
        &whileOp.getAfter(), whileOp.getAfter().begin(), lcvTypes, lcvLocs);

    // Add induction variable incrementation
    rewriter.setInsertionPointToEnd(afterBlock);
    auto ivIncOp = rewriter.create<arith::AddIOp>(
        whileOp.getLoc(), afterBlock->getArgument(0), forOp.getStep());

    // Rewrite uses of the for-loop block arguments to the new while-loop
    // "after" arguments
    for (const auto &barg : enumerate(forOp.getBody(0)->getArguments()))
      barg.value().replaceAllUsesWith(afterBlock->getArgument(barg.index()));

    // Inline for-loop body operations into 'after' region.
    for (auto &arg : llvm::make_early_inc_range(*forOp.getBody()))
      arg.moveBefore(afterBlock, afterBlock->end());

    // Add incremented IV to yield operations
    for (auto yieldOp : afterBlock->getOps<scf::YieldOp>()) {
      SmallVector<Value> yieldOperands = yieldOp.getOperands();
      yieldOperands.insert(yieldOperands.begin(), ivIncOp.getResult());
      yieldOp->setOperands(yieldOperands);
    }

    // We cannot do a direct replacement of the forOp since the while op returns
    // an extra value (the induction variable escapes the loop through being
    // carried in the set of iterargs). Instead, rewrite uses of the forOp
    // results.
    for (const auto &arg : llvm::enumerate(forOp.getResults()))
      arg.value().replaceAllUsesWith(whileOp.getResult(arg.index() + 1));

    rewriter.eraseOp(forOp);
    return success();
  }
};

struct ForToWhileLoop : public impl::SCFForToWhileLoopBase<ForToWhileLoop> {
  void runOnOperation() override {
    auto *parentOp = getOperation();
    MLIRContext *ctx = parentOp->getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<ForLoopLoweringPattern>(ctx);
    (void)applyPatternsAndFoldGreedily(parentOp, std::move(patterns));
  }
};
} // namespace

std::unique_ptr<Pass> mlir::createForToWhileLoopPass() {
  return std::make_unique<ForToWhileLoop>();
}
