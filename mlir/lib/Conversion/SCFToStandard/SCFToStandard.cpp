//===- SCFToStandard.cpp - ControlFlow to CFG conversion ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to convert scf.for, scf.if and loop.terminator
// ops into standard CFG ops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "../PassDetail.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/Utils.h"

using namespace mlir;
using namespace mlir::scf;

namespace {

struct SCFToStandardPass : public SCFToStandardBase<SCFToStandardPass> {
  void runOnOperation() override;
};

// Create a CFG subgraph for the loop around its body blocks (if the body
// contained other loops, they have been already lowered to a flow of blocks).
// Maintain the invariants that a CFG subgraph created for any loop has a single
// entry and a single exit, and that the entry/exit blocks are respectively
// first/last blocks in the parent region.  The original loop operation is
// replaced by the initialization operations that set up the initial value of
// the loop induction variable (%iv) and computes the loop bounds that are loop-
// invariant for affine loops.  The operations following the original scf.for
// are split out into a separate continuation (exit) block. A condition block is
// created before the continuation block. It checks the exit condition of the
// loop and branches either to the continuation block, or to the first block of
// the body. The condition block takes as arguments the values of the induction
// variable followed by loop-carried values. Since it dominates both the body
// blocks and the continuation block, loop-carried values are visible in all of
// those blocks. Induction variable modification is appended to the last block
// of the body (which is the exit block from the body subgraph thanks to the
// invariant we maintain) along with a branch that loops back to the condition
// block. Loop-carried values are the loop terminator operands, which are
// forwarded to the branch.
//
//      +---------------------------------+
//      |   <code before the ForOp>       |
//      |   <definitions of %init...>     |
//      |   <compute initial %iv value>   |
//      |   br cond(%iv, %init...)        |
//      +---------------------------------+
//             |
//  -------|   |
//  |      v   v
//  |   +--------------------------------+
//  |   | cond(%iv, %init...):           |
//  |   |   <compare %iv to upper bound> |
//  |   |   cond_br %r, body, end        |
//  |   +--------------------------------+
//  |          |               |
//  |          |               -------------|
//  |          v                            |
//  |   +--------------------------------+  |
//  |   | body-first:                    |  |
//  |   |   <%init visible by dominance> |  |
//  |   |   <body contents>              |  |
//  |   +--------------------------------+  |
//  |                   |                   |
//  |                  ...                  |
//  |                   |                   |
//  |   +--------------------------------+  |
//  |   | body-last:                     |  |
//  |   |   <body contents>              |  |
//  |   |   <operands of yield = %yields>|  |
//  |   |   %new_iv =<add step to %iv>   |  |
//  |   |   br cond(%new_iv, %yields)    |  |
//  |   +--------------------------------+  |
//  |          |                            |
//  |-----------        |--------------------
//                      v
//      +--------------------------------+
//      | end:                           |
//      |   <code after the ForOp>       |
//      |   <%init visible by dominance> |
//      +--------------------------------+
//
struct ForLowering : public OpRewritePattern<ForOp> {
  using OpRewritePattern<ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ForOp forOp,
                                PatternRewriter &rewriter) const override;
};

// Create a CFG subgraph for the scf.if operation (including its "then" and
// optional "else" operation blocks).  We maintain the invariants that the
// subgraph has a single entry and a single exit point, and that the entry/exit
// blocks are respectively the first/last block of the enclosing region. The
// operations following the scf.if are split into a continuation (subgraph
// exit) block. The condition is lowered to a chain of blocks that implement the
// short-circuit scheme. The "scf.if" operation is replaced with a conditional
// branch to either the first block of the "then" region, or to the first block
// of the "else" region. In these blocks, "scf.yield" is unconditional branches
// to the post-dominating block. When the "scf.if" does not return values, the
// post-dominating block is the same as the continuation block. When it returns
// values, the post-dominating block is a new block with arguments that
// correspond to the values returned by the "scf.if" that unconditionally
// branches to the continuation block. This allows block arguments to dominate
// any uses of the hitherto "scf.if" results that they replaced. (Inserting a
// new block allows us to avoid modifying the argument list of an existing
// block, which is illegal in a conversion pattern). When the "else" region is
// empty, which is only allowed for "scf.if"s that don't return values, the
// condition branches directly to the continuation block.
//
// CFG for a scf.if with else and without results.
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
//      |   <code after the IfOp>        |
//      +--------------------------------+
//
// CFG for a scf.if with results.
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
//      |   br dom(%args...)             |  |
//      +--------------------------------+  |
//             |                            |
//   |----------               |-------------
//   |                         V
//   |  +--------------------------------+
//   |  | else:                          |
//   |  |   <else contents>              |
//   |  |   br dom(%args...)             |
//   |  +--------------------------------+
//   |         |
//   ------|   |
//         v   v
//      +--------------------------------+
//      | dom(%args...):                 |
//      |   br continue                  |
//      +--------------------------------+
//             |
//             v
//      +--------------------------------+
//      | continue:                      |
//      | <code after the IfOp>          |
//      +--------------------------------+
//
struct IfLowering : public OpRewritePattern<IfOp> {
  using OpRewritePattern<IfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IfOp ifOp,
                                PatternRewriter &rewriter) const override;
};

struct ExecuteRegionLowering : public OpRewritePattern<ExecuteRegionOp> {
  using OpRewritePattern<ExecuteRegionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ExecuteRegionOp op,
                                PatternRewriter &rewriter) const override;
};

struct ParallelLowering : public OpRewritePattern<mlir::scf::ParallelOp> {
  using OpRewritePattern<mlir::scf::ParallelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::scf::ParallelOp parallelOp,
                                PatternRewriter &rewriter) const override;
};

/// Create a CFG subgraph for this loop construct. The regions of the loop need
/// not be a single block anymore (for example, if other SCF constructs that
/// they contain have been already converted to CFG), but need to be single-exit
/// from the last block of each region. The operations following the original
/// WhileOp are split into a new continuation block. Both regions of the WhileOp
/// are inlined, and their terminators are rewritten to organize the control
/// flow implementing the loop as follows.
///
///      +---------------------------------+
///      |   <code before the WhileOp>     |
///      |   br ^before(%operands...)      |
///      +---------------------------------+
///             |
///  -------|   |
///  |      v   v
///  |   +--------------------------------+
///  |   | ^before(%bargs...):            |
///  |   |   %vals... = <some payload>    |
///  |   +--------------------------------+
///  |                   |
///  |                  ...
///  |                   |
///  |   +--------------------------------+
///  |   | ^before-last:
///  |   |   %cond = <compute condition>  |
///  |   |   cond_br %cond,               |
///  |   |        ^after(%vals...), ^cont |
///  |   +--------------------------------+
///  |          |               |
///  |          |               -------------|
///  |          v                            |
///  |   +--------------------------------+  |
///  |   | ^after(%aargs...):             |  |
///  |   |   <body contents>              |  |
///  |   +--------------------------------+  |
///  |                   |                   |
///  |                  ...                  |
///  |                   |                   |
///  |   +--------------------------------+  |
///  |   | ^after-last:                   |  |
///  |   |   %yields... = <some payload>  |  |
///  |   |   br ^before(%yields...)       |  |
///  |   +--------------------------------+  |
///  |          |                            |
///  |-----------        |--------------------
///                      v
///      +--------------------------------+
///      | ^cont:                         |
///      |   <code after the WhileOp>     |
///      |   <%vals from 'before' region  |
///      |          visible by dominance> |
///      +--------------------------------+
///
/// Values are communicated between ex-regions (the groups of blocks that used
/// to form a region before inlining) through block arguments of their
/// entry blocks, which are visible in all other dominated blocks. Similarly,
/// the results of the WhileOp are defined in the 'before' region, which is
/// required to have a single existing block, and are therefore accessible in
/// the continuation block due to dominance.
struct WhileLowering : public OpRewritePattern<WhileOp> {
  using OpRewritePattern<WhileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(WhileOp whileOp,
                                PatternRewriter &rewriter) const override;
};

/// Optimized version of the above for the case of the "after" region merely
/// forwarding its arguments back to the "before" region (i.e., a "do-while"
/// loop). This avoid inlining the "after" region completely and branches back
/// to the "before" entry instead.
struct DoWhileLowering : public OpRewritePattern<WhileOp> {
  using OpRewritePattern<WhileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(WhileOp whileOp,
                                PatternRewriter &rewriter) const override;
};
} // namespace

LogicalResult ForLowering::matchAndRewrite(ForOp forOp,
                                           PatternRewriter &rewriter) const {
  Location loc = forOp.getLoc();

  // Start by splitting the block containing the 'scf.for' into two parts.
  // The part before will get the init code, the part after will be the end
  // point.
  auto *initBlock = rewriter.getInsertionBlock();
  auto initPosition = rewriter.getInsertionPoint();
  auto *endBlock = rewriter.splitBlock(initBlock, initPosition);

  // Use the first block of the loop body as the condition block since it is the
  // block that has the induction variable and loop-carried values as arguments.
  // Split out all operations from the first block into a new block. Move all
  // body blocks from the loop body region to the region containing the loop.
  auto *conditionBlock = &forOp.region().front();
  auto *firstBodyBlock =
      rewriter.splitBlock(conditionBlock, conditionBlock->begin());
  auto *lastBodyBlock = &forOp.region().back();
  rewriter.inlineRegionBefore(forOp.region(), endBlock);
  auto iv = conditionBlock->getArgument(0);

  // Append the induction variable stepping logic to the last body block and
  // branch back to the condition block. Loop-carried values are taken from
  // operands of the loop terminator.
  Operation *terminator = lastBodyBlock->getTerminator();
  rewriter.setInsertionPointToEnd(lastBodyBlock);
  auto step = forOp.step();
  auto stepped = rewriter.create<AddIOp>(loc, iv, step).getResult();
  if (!stepped)
    return failure();

  SmallVector<Value, 8> loopCarried;
  loopCarried.push_back(stepped);
  loopCarried.append(terminator->operand_begin(), terminator->operand_end());
  rewriter.create<BranchOp>(loc, conditionBlock, loopCarried);
  rewriter.eraseOp(terminator);

  // Compute loop bounds before branching to the condition.
  rewriter.setInsertionPointToEnd(initBlock);
  Value lowerBound = forOp.lowerBound();
  Value upperBound = forOp.upperBound();
  if (!lowerBound || !upperBound)
    return failure();

  // The initial values of loop-carried values is obtained from the operands
  // of the loop operation.
  SmallVector<Value, 8> destOperands;
  destOperands.push_back(lowerBound);
  auto iterOperands = forOp.getIterOperands();
  destOperands.append(iterOperands.begin(), iterOperands.end());
  rewriter.create<BranchOp>(loc, conditionBlock, destOperands);

  // With the body block done, we can fill in the condition block.
  rewriter.setInsertionPointToEnd(conditionBlock);
  auto comparison =
      rewriter.create<CmpIOp>(loc, CmpIPredicate::slt, iv, upperBound);

  rewriter.create<CondBranchOp>(loc, comparison, firstBodyBlock,
                                ArrayRef<Value>(), endBlock, ArrayRef<Value>());
  // The result of the loop operation is the values of the condition block
  // arguments except the induction variable on the last iteration.
  rewriter.replaceOp(forOp, conditionBlock->getArguments().drop_front());
  return success();
}

LogicalResult IfLowering::matchAndRewrite(IfOp ifOp,
                                          PatternRewriter &rewriter) const {
  auto loc = ifOp.getLoc();

  // Start by splitting the block containing the 'scf.if' into two parts.
  // The part before will contain the condition, the part after will be the
  // continuation point.
  auto *condBlock = rewriter.getInsertionBlock();
  auto opPosition = rewriter.getInsertionPoint();
  auto *remainingOpsBlock = rewriter.splitBlock(condBlock, opPosition);
  Block *continueBlock;
  if (ifOp.getNumResults() == 0) {
    continueBlock = remainingOpsBlock;
  } else {
    continueBlock =
        rewriter.createBlock(remainingOpsBlock, ifOp.getResultTypes());
    rewriter.create<BranchOp>(loc, remainingOpsBlock);
  }

  // Move blocks from the "then" region to the region containing 'scf.if',
  // place it before the continuation block, and branch to it.
  auto &thenRegion = ifOp.thenRegion();
  auto *thenBlock = &thenRegion.front();
  Operation *thenTerminator = thenRegion.back().getTerminator();
  ValueRange thenTerminatorOperands = thenTerminator->getOperands();
  rewriter.setInsertionPointToEnd(&thenRegion.back());
  rewriter.create<BranchOp>(loc, continueBlock, thenTerminatorOperands);
  rewriter.eraseOp(thenTerminator);
  rewriter.inlineRegionBefore(thenRegion, continueBlock);

  // Move blocks from the "else" region (if present) to the region containing
  // 'scf.if', place it before the continuation block and branch to it.  It
  // will be placed after the "then" regions.
  auto *elseBlock = continueBlock;
  auto &elseRegion = ifOp.elseRegion();
  if (!elseRegion.empty()) {
    elseBlock = &elseRegion.front();
    Operation *elseTerminator = elseRegion.back().getTerminator();
    ValueRange elseTerminatorOperands = elseTerminator->getOperands();
    rewriter.setInsertionPointToEnd(&elseRegion.back());
    rewriter.create<BranchOp>(loc, continueBlock, elseTerminatorOperands);
    rewriter.eraseOp(elseTerminator);
    rewriter.inlineRegionBefore(elseRegion, continueBlock);
  }

  rewriter.setInsertionPointToEnd(condBlock);
  rewriter.create<CondBranchOp>(loc, ifOp.condition(), thenBlock,
                                /*trueArgs=*/ArrayRef<Value>(), elseBlock,
                                /*falseArgs=*/ArrayRef<Value>());

  // Ok, we're done!
  rewriter.replaceOp(ifOp, continueBlock->getArguments());
  return success();
}

LogicalResult
ExecuteRegionLowering::matchAndRewrite(ExecuteRegionOp op,
                                       PatternRewriter &rewriter) const {
  auto loc = op.getLoc();

  auto *condBlock = rewriter.getInsertionBlock();
  auto opPosition = rewriter.getInsertionPoint();
  auto *remainingOpsBlock = rewriter.splitBlock(condBlock, opPosition);

  auto &region = op.region();
  rewriter.setInsertionPointToEnd(condBlock);
  rewriter.create<BranchOp>(loc, &region.front());

  for (Block &block : region) {
    if (auto terminator = dyn_cast<scf::YieldOp>(block.getTerminator())) {
      ValueRange terminatorOperands = terminator->getOperands();
      rewriter.setInsertionPointToEnd(&block);
      rewriter.create<BranchOp>(loc, remainingOpsBlock, terminatorOperands);
      rewriter.eraseOp(terminator);
    }
  }

  rewriter.inlineRegionBefore(region, remainingOpsBlock);

  SmallVector<Value> vals;
  for (auto arg : remainingOpsBlock->addArguments(op->getResultTypes())) {
    vals.push_back(arg);
  }
  rewriter.replaceOp(op, vals);
  return success();
}

LogicalResult
ParallelLowering::matchAndRewrite(ParallelOp parallelOp,
                                  PatternRewriter &rewriter) const {
  Location loc = parallelOp.getLoc();

  // For a parallel loop, we essentially need to create an n-dimensional loop
  // nest. We do this by translating to scf.for ops and have those lowered in
  // a further rewrite. If a parallel loop contains reductions (and thus returns
  // values), forward the initial values for the reductions down the loop
  // hierarchy and bubble up the results by modifying the "yield" terminator.
  SmallVector<Value, 4> iterArgs = llvm::to_vector<4>(parallelOp.initVals());
  SmallVector<Value, 4> ivs;
  ivs.reserve(parallelOp.getNumLoops());
  bool first = true;
  SmallVector<Value, 4> loopResults(iterArgs);
  for (auto loop_operands :
       llvm::zip(parallelOp.getInductionVars(), parallelOp.lowerBound(),
                 parallelOp.upperBound(), parallelOp.step())) {
    Value iv, lower, upper, step;
    std::tie(iv, lower, upper, step) = loop_operands;
    ForOp forOp = rewriter.create<ForOp>(loc, lower, upper, step, iterArgs);
    ivs.push_back(forOp.getInductionVar());
    auto iterRange = forOp.getRegionIterArgs();
    iterArgs.assign(iterRange.begin(), iterRange.end());

    if (first) {
      // Store the results of the outermost loop that will be used to replace
      // the results of the parallel loop when it is fully rewritten.
      loopResults.assign(forOp.result_begin(), forOp.result_end());
      first = false;
    } else if (!forOp.getResults().empty()) {
      // A loop is constructed with an empty "yield" terminator if there are
      // no results.
      rewriter.setInsertionPointToEnd(rewriter.getInsertionBlock());
      rewriter.create<scf::YieldOp>(loc, forOp.getResults());
    }

    rewriter.setInsertionPointToStart(forOp.getBody());
  }

  // First, merge reduction blocks into the main region.
  SmallVector<Value, 4> yieldOperands;
  yieldOperands.reserve(parallelOp.getNumResults());
  for (auto &op : *parallelOp.getBody()) {
    auto reduce = dyn_cast<ReduceOp>(op);
    if (!reduce)
      continue;

    Block &reduceBlock = reduce.reductionOperator().front();
    Value arg = iterArgs[yieldOperands.size()];
    yieldOperands.push_back(reduceBlock.getTerminator()->getOperand(0));
    rewriter.eraseOp(reduceBlock.getTerminator());
    rewriter.mergeBlockBefore(&reduceBlock, &op, {arg, reduce.operand()});
    rewriter.eraseOp(reduce);
  }

  // Then merge the loop body without the terminator.
  rewriter.eraseOp(parallelOp.getBody()->getTerminator());
  Block *newBody = rewriter.getInsertionBlock();
  if (newBody->empty())
    rewriter.mergeBlocks(parallelOp.getBody(), newBody, ivs);
  else
    rewriter.mergeBlockBefore(parallelOp.getBody(), newBody->getTerminator(),
                              ivs);

  // Finally, create the terminator if required (for loops with no results, it
  // has been already created in loop construction).
  if (!yieldOperands.empty()) {
    rewriter.setInsertionPointToEnd(rewriter.getInsertionBlock());
    rewriter.create<scf::YieldOp>(loc, yieldOperands);
  }

  rewriter.replaceOp(parallelOp, loopResults);

  return success();
}

LogicalResult WhileLowering::matchAndRewrite(WhileOp whileOp,
                                             PatternRewriter &rewriter) const {
  OpBuilder::InsertionGuard guard(rewriter);
  Location loc = whileOp.getLoc();

  // Split the current block before the WhileOp to create the inlining point.
  Block *currentBlock = rewriter.getInsertionBlock();
  Block *continuation =
      rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());

  // Inline both regions.
  Block *after = &whileOp.after().front();
  Block *afterLast = &whileOp.after().back();
  Block *before = &whileOp.before().front();
  Block *beforeLast = &whileOp.before().back();
  rewriter.inlineRegionBefore(whileOp.after(), continuation);
  rewriter.inlineRegionBefore(whileOp.before(), after);

  // Branch to the "before" region.
  rewriter.setInsertionPointToEnd(currentBlock);
  rewriter.create<BranchOp>(loc, before, whileOp.inits());

  // Replace terminators with branches. Assuming bodies are SESE, which holds
  // given only the patterns from this file, we only need to look at the last
  // block. This should be reconsidered if we allow break/continue in SCF.
  rewriter.setInsertionPointToEnd(beforeLast);
  auto condOp = cast<ConditionOp>(beforeLast->getTerminator());
  rewriter.replaceOpWithNewOp<CondBranchOp>(condOp, condOp.condition(), after,
                                            condOp.args(), continuation,
                                            ValueRange());

  rewriter.setInsertionPointToEnd(afterLast);
  auto yieldOp = cast<scf::YieldOp>(afterLast->getTerminator());
  rewriter.replaceOpWithNewOp<BranchOp>(yieldOp, before, yieldOp.results());

  // Replace the op with values "yielded" from the "before" region, which are
  // visible by dominance.
  rewriter.replaceOp(whileOp, condOp.args());

  return success();
}

LogicalResult
DoWhileLowering::matchAndRewrite(WhileOp whileOp,
                                 PatternRewriter &rewriter) const {
  if (!llvm::hasSingleElement(whileOp.after()))
    return rewriter.notifyMatchFailure(whileOp,
                                       "do-while simplification applicable to "
                                       "single-block 'after' region only");

  Block &afterBlock = whileOp.after().front();
  if (!llvm::hasSingleElement(afterBlock))
    return rewriter.notifyMatchFailure(whileOp,
                                       "do-while simplification applicable "
                                       "only if 'after' region has no payload");

  auto yield = dyn_cast<scf::YieldOp>(&afterBlock.front());
  if (!yield || yield.results() != afterBlock.getArguments())
    return rewriter.notifyMatchFailure(whileOp,
                                       "do-while simplification applicable "
                                       "only to forwarding 'after' regions");

  // Split the current block before the WhileOp to create the inlining point.
  OpBuilder::InsertionGuard guard(rewriter);
  Block *currentBlock = rewriter.getInsertionBlock();
  Block *continuation =
      rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());

  // Only the "before" region should be inlined.
  Block *before = &whileOp.before().front();
  Block *beforeLast = &whileOp.before().back();
  rewriter.inlineRegionBefore(whileOp.before(), continuation);

  // Branch to the "before" region.
  rewriter.setInsertionPointToEnd(currentBlock);
  rewriter.create<BranchOp>(whileOp.getLoc(), before, whileOp.inits());

  // Loop around the "before" region based on condition.
  rewriter.setInsertionPointToEnd(beforeLast);
  auto condOp = cast<ConditionOp>(beforeLast->getTerminator());
  rewriter.replaceOpWithNewOp<CondBranchOp>(condOp, condOp.condition(), before,
                                            condOp.args(), continuation,
                                            ValueRange());

  // Replace the op with values "yielded" from the "before" region, which are
  // visible by dominance.
  rewriter.replaceOp(whileOp, condOp.args());

  return success();
}

void mlir::populateLoopToStdConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<ForLowering, IfLowering, ParallelLowering, WhileLowering,
               ExecuteRegionLowering>(patterns.getContext());
  patterns.add<DoWhileLowering>(patterns.getContext(), /*benefit=*/2);
}

void SCFToStandardPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  populateLoopToStdConversionPatterns(patterns);
  // Configure conversion to lower out scf.for, scf.if, scf.parallel and
  // scf.while. Anything else is fine.
  ConversionTarget target(getContext());
  target.addIllegalOp<scf::ForOp, scf::IfOp, scf::ParallelOp, scf::WhileOp,
                      scf::ExecuteRegionOp>();
  target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<Pass> mlir::createLowerToCFGPass() {
  return std::make_unique<SCFToStandardPass>();
}
