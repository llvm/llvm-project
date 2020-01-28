//===- ConvertLoopToStandard.cpp - ControlFlow to CFG conversion ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to convert loop.for, loop.if and loop.terminator
// ops into standard CFG ops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LoopToStandard/ConvertLoopToStandard.h"
#include "mlir/Dialect/LoopOps/LoopOps.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/Functional.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/Utils.h"

using namespace mlir;
using namespace mlir::loop;

namespace {

struct LoopToStandardPass : public OperationPass<LoopToStandardPass> {
  void runOnOperation() override;
};

// Create a CFG subgraph for the loop around its body blocks (if the body
// contained other loops, they have been already lowered to a flow of blocks).
// Maintain the invariants that a CFG subgraph created for any loop has a single
// entry and a single exit, and that the entry/exit blocks are respectively
// first/last blocks in the parent region.  The original loop operation is
// replaced by the initialization operations that set up the initial value of
// the loop induction variable (%iv) and computes the loop bounds that are loop-
// invariant for affine loops.  The operations following the original loop.for
// are split out into a separate continuation (exit) block. A condition block is
// created before the continuation block. It checks the exit condition of the
// loop and branches either to the continuation block, or to the first block of
// the body. Induction variable modification is appended to the last block of
// the body (which is the exit block from the body subgraph thanks to the
// invariant we maintain) along with a branch that loops back to the condition
// block.
//
//      +---------------------------------+
//      |   <code before the ForOp>       |
//      |   <compute initial %iv value>   |
//      |   br cond(%iv)                  |
//      +---------------------------------+
//             |
//  -------|   |
//  |      v   v
//  |   +--------------------------------+
//  |   | cond(%iv):                     |
//  |   |   <compare %iv to upper bound> |
//  |   |   cond_br %r, body, end        |
//  |   +--------------------------------+
//  |          |               |
//  |          |               -------------|
//  |          v                            |
//  |   +--------------------------------+  |
//  |   | body-first:                    |  |
//  |   |   <body contents>              |  |
//  |   +--------------------------------+  |
//  |                   |                   |
//  |                  ...                  |
//  |                   |                   |
//  |   +--------------------------------+  |
//  |   | body-last:                     |  |
//  |   |   <body contents>              |  |
//  |   |   %new_iv =<add step to %iv>   |  |
//  |   |   br cond(%new_iv)             |  |
//  |   +--------------------------------+  |
//  |          |                            |
//  |-----------        |--------------------
//                      v
//      +--------------------------------+
//      | end:                           |
//      |   <code after the ForOp> |
//      +--------------------------------+
//
struct ForLowering : public OpRewritePattern<ForOp> {
  using OpRewritePattern<ForOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(ForOp forOp,
                                     PatternRewriter &rewriter) const override;
};

// Create a CFG subgraph for the loop.if operation (including its "then" and
// optional "else" operation blocks).  We maintain the invariants that the
// subgraph has a single entry and a single exit point, and that the entry/exit
// blocks are respectively the first/last block of the enclosing region. The
// operations following the loop.if are split into a continuation (subgraph
// exit) block. The condition is lowered to a chain of blocks that implement the
// short-circuit scheme.  Condition blocks are created by splitting out an empty
// block from the block that contains the loop.if operation.  They
// conditionally branch to either the first block of the "then" region, or to
// the first block of the "else" region.  If the latter is absent, they branch
// to the continuation block instead.  The last blocks of "then" and "else"
// regions (which are known to be exit blocks thanks to the invariant we
// maintain).
//
//      +--------------------------------+
//      | <code before the IfOp>         |
//      | cond_br %cond, %then, %else    |
//      +--------------------------------+
//             |              |
//             |              --------------|
//             v                            |
//      +--------------------------------+  |
//      | then:                          |  |
//      |   <then contents>              |  |
//      |   br continue                  |  |
//      +--------------------------------+  |
//             |                            |
//   |----------               |-------------
//   |                         V
//   |  +--------------------------------+
//   |  | else:                          |
//   |  |   <else contents>              |
//   |  |   br continue                  |
//   |  +--------------------------------+
//   |         |
//   ------|   |
//         v   v
//      +--------------------------------+
//      | continue:                      |
//      |   <code after the IfOp>  |
//      +--------------------------------+
//
struct IfLowering : public OpRewritePattern<IfOp> {
  using OpRewritePattern<IfOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(IfOp ifOp,
                                     PatternRewriter &rewriter) const override;
};

struct ParallelLowering : public OpRewritePattern<mlir::loop::ParallelOp> {
  using OpRewritePattern<mlir::loop::ParallelOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(mlir::loop::ParallelOp parallelOp,
                                     PatternRewriter &rewriter) const override;
};
} // namespace

PatternMatchResult
ForLowering::matchAndRewrite(ForOp forOp, PatternRewriter &rewriter) const {
  Location loc = forOp.getLoc();

  // Start by splitting the block containing the 'loop.for' into two parts.
  // The part before will get the init code, the part after will be the end
  // point.
  auto *initBlock = rewriter.getInsertionBlock();
  auto initPosition = rewriter.getInsertionPoint();
  auto *endBlock = rewriter.splitBlock(initBlock, initPosition);

  // Use the first block of the loop body as the condition block since it is
  // the block that has the induction variable as its argument.  Split out
  // all operations from the first block into a new block.  Move all body
  // blocks from the loop body region to the region containing the loop.
  auto *conditionBlock = &forOp.region().front();
  auto *firstBodyBlock =
      rewriter.splitBlock(conditionBlock, conditionBlock->begin());
  auto *lastBodyBlock = &forOp.region().back();
  rewriter.inlineRegionBefore(forOp.region(), endBlock);
  auto iv = conditionBlock->getArgument(0);

  // Append the induction variable stepping logic to the last body block and
  // branch back to the condition block.  Construct an expression f :
  // (x -> x+step) and apply this expression to the induction variable.
  rewriter.eraseOp(lastBodyBlock->getTerminator());
  rewriter.setInsertionPointToEnd(lastBodyBlock);
  auto step = forOp.step();
  auto stepped = rewriter.create<AddIOp>(loc, iv, step).getResult();
  if (!stepped)
    return matchFailure();
  rewriter.create<BranchOp>(loc, conditionBlock, stepped);

  // Compute loop bounds before branching to the condition.
  rewriter.setInsertionPointToEnd(initBlock);
  Value lowerBound = forOp.lowerBound();
  Value upperBound = forOp.upperBound();
  if (!lowerBound || !upperBound)
    return matchFailure();
  rewriter.create<BranchOp>(loc, conditionBlock, lowerBound);

  // With the body block done, we can fill in the condition block.
  rewriter.setInsertionPointToEnd(conditionBlock);
  auto comparison =
      rewriter.create<CmpIOp>(loc, CmpIPredicate::slt, iv, upperBound);

  rewriter.create<CondBranchOp>(loc, comparison, firstBodyBlock,
                                ArrayRef<Value>(), endBlock, ArrayRef<Value>());
  // Ok, we're done!
  rewriter.eraseOp(forOp);
  return matchSuccess();
}

PatternMatchResult
IfLowering::matchAndRewrite(IfOp ifOp, PatternRewriter &rewriter) const {
  auto loc = ifOp.getLoc();

  // Start by splitting the block containing the 'loop.if' into two parts.
  // The part before will contain the condition, the part after will be the
  // continuation point.
  auto *condBlock = rewriter.getInsertionBlock();
  auto opPosition = rewriter.getInsertionPoint();
  auto *continueBlock = rewriter.splitBlock(condBlock, opPosition);

  // Move blocks from the "then" region to the region containing 'loop.if',
  // place it before the continuation block, and branch to it.
  auto &thenRegion = ifOp.thenRegion();
  auto *thenBlock = &thenRegion.front();
  rewriter.eraseOp(thenRegion.back().getTerminator());
  rewriter.setInsertionPointToEnd(&thenRegion.back());
  rewriter.create<BranchOp>(loc, continueBlock);
  rewriter.inlineRegionBefore(thenRegion, continueBlock);

  // Move blocks from the "else" region (if present) to the region containing
  // 'loop.if', place it before the continuation block and branch to it.  It
  // will be placed after the "then" regions.
  auto *elseBlock = continueBlock;
  auto &elseRegion = ifOp.elseRegion();
  if (!elseRegion.empty()) {
    elseBlock = &elseRegion.front();
    rewriter.eraseOp(elseRegion.back().getTerminator());
    rewriter.setInsertionPointToEnd(&elseRegion.back());
    rewriter.create<BranchOp>(loc, continueBlock);
    rewriter.inlineRegionBefore(elseRegion, continueBlock);
  }

  rewriter.setInsertionPointToEnd(condBlock);
  rewriter.create<CondBranchOp>(loc, ifOp.condition(), thenBlock,
                                /*trueArgs=*/ArrayRef<Value>(), elseBlock,
                                /*falseArgs=*/ArrayRef<Value>());

  // Ok, we're done!
  rewriter.eraseOp(ifOp);
  return matchSuccess();
}

PatternMatchResult
ParallelLowering::matchAndRewrite(ParallelOp parallelOp,
                                  PatternRewriter &rewriter) const {
  Location loc = parallelOp.getLoc();
  BlockAndValueMapping mapping;

  if (parallelOp.getNumResults() != 0) {
    // TODO: Implement lowering of parallelOp with reductions.
    return matchFailure();
  }

  // For a parallel loop, we essentially need to create an n-dimensional loop
  // nest. We do this by translating to loop.for ops and have those lowered in
  // a further rewrite.
  for (auto loop_operands :
       llvm::zip(parallelOp.getInductionVars(), parallelOp.lowerBound(),
                 parallelOp.upperBound(), parallelOp.step())) {
    Value iv, lower, upper, step;
    std::tie(iv, lower, upper, step) = loop_operands;
    ForOp forOp = rewriter.create<ForOp>(loc, lower, upper, step);
    mapping.map(iv, forOp.getInductionVar());
    rewriter.setInsertionPointToStart(forOp.getBody());
  }

  // Now copy over the contents of the body.
  for (auto &op : parallelOp.body().front().without_terminator())
    rewriter.clone(op, mapping);

  rewriter.eraseOp(parallelOp);

  return matchSuccess();
}

void mlir::populateLoopToStdConversionPatterns(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns.insert<ForLowering, IfLowering, ParallelLowering>(ctx);
}

void LoopToStandardPass::runOnOperation() {
  OwningRewritePatternList patterns;
  populateLoopToStdConversionPatterns(patterns, &getContext());
  ConversionTarget target(getContext());
  target.addLegalDialect<StandardOpsDialect>();
  if (failed(applyPartialConversion(getOperation(), target, patterns)))
    signalPassFailure();
}

std::unique_ptr<Pass> mlir::createLowerToCFGPass() {
  return std::make_unique<LoopToStandardPass>();
}

static PassRegistration<LoopToStandardPass>
    pass("convert-loop-to-std", "Convert Loop dialect to Standard dialect, "
                                "replacing structured control flow with a CFG");
