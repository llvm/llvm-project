//===- LoopToStandard.cpp - ControlFlow to CFG conversion -----------------===//
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
#include "mlir/Dialect/StandardOps/IR/Ops.h"
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
//      |   <code after the IfOp>        |
//      +--------------------------------+
//
struct IfLowering : public OpRewritePattern<IfOp> {
  using OpRewritePattern<IfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IfOp ifOp,
                                PatternRewriter &rewriter) const override;
};

struct ParallelLowering : public OpRewritePattern<mlir::loop::ParallelOp> {
  using OpRewritePattern<mlir::loop::ParallelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::loop::ParallelOp parallelOp,
                                PatternRewriter &rewriter) const override;
};
} // namespace

LogicalResult ForLowering::matchAndRewrite(ForOp forOp,
                                           PatternRewriter &rewriter) const {
  Location loc = forOp.getLoc();

  // Start by splitting the block containing the 'loop.for' into two parts.
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
  return success();
}

LogicalResult
ParallelLowering::matchAndRewrite(ParallelOp parallelOp,
                                  PatternRewriter &rewriter) const {
  Location loc = parallelOp.getLoc();
  BlockAndValueMapping mapping;

  // For a parallel loop, we essentially need to create an n-dimensional loop
  // nest. We do this by translating to loop.for ops and have those lowered in
  // a further rewrite. If a parallel loop contains reductions (and thus returns
  // values), forward the initial values for the reductions down the loop
  // hierarchy and bubble up the results by modifying the "yield" terminator.
  SmallVector<Value, 4> iterArgs = llvm::to_vector<4>(parallelOp.initVals());
  bool first = true;
  SmallVector<Value, 4> loopResults(iterArgs);
  for (auto loop_operands :
       llvm::zip(parallelOp.getInductionVars(), parallelOp.lowerBound(),
                 parallelOp.upperBound(), parallelOp.step())) {
    Value iv, lower, upper, step;
    std::tie(iv, lower, upper, step) = loop_operands;
    ForOp forOp = rewriter.create<ForOp>(loc, lower, upper, step, iterArgs);
    mapping.map(iv, forOp.getInductionVar());
    auto iterRange = forOp.getRegionIterArgs();
    iterArgs.assign(iterRange.begin(), iterRange.end());

    if (first) {
      // Store the results of the outermost loop that will be used to replace
      // the results of the parallel loop when it is fully rewritten.
      loopResults.assign(forOp.result_begin(), forOp.result_end());
      first = false;
    } else {
      // A loop is constructed with an empty "yield" terminator by default.
      // Replace it with another "yield" that forwards the results of the nested
      // loop to the parent loop. We need to explicitly make sure the new
      // terminator is the last operation in the block because further
      // transforms rely on this.
      rewriter.setInsertionPointToEnd(rewriter.getInsertionBlock());
      rewriter.replaceOpWithNewOp<YieldOp>(
          rewriter.getInsertionBlock()->getTerminator(), forOp.getResults());
    }

    rewriter.setInsertionPointToStart(forOp.getBody());
  }

  // Now copy over the contents of the body.
  SmallVector<Value, 4> yieldOperands;
  yieldOperands.reserve(parallelOp.getNumResults());
  for (auto &op : parallelOp.getBody()->without_terminator()) {
    // Reduction blocks are handled differently.
    auto reduce = dyn_cast<ReduceOp>(op);
    if (!reduce) {
      rewriter.clone(op, mapping);
      continue;
    }

    // Clone the body of the reduction operation into the body of the loop,
    // using operands of "loop.reduce" and iteration arguments corresponding
    // to the reduction value to replace arguments of the reduction block.
    // Collect operands of "loop.reduce.return" to be returned by a final
    // "loop.yield" instead.
    Value arg = iterArgs[yieldOperands.size()];
    Block &reduceBlock = reduce.reductionOperator().front();
    mapping.map(reduceBlock.getArgument(0), mapping.lookupOrDefault(arg));
    mapping.map(reduceBlock.getArgument(1),
                mapping.lookupOrDefault(reduce.operand()));
    for (auto &nested : reduceBlock.without_terminator())
      rewriter.clone(nested, mapping);
    yieldOperands.push_back(
        mapping.lookup(reduceBlock.getTerminator()->getOperand(0)));
  }

  rewriter.setInsertionPointToEnd(rewriter.getInsertionBlock());
  rewriter.replaceOpWithNewOp<YieldOp>(
      rewriter.getInsertionBlock()->getTerminator(), yieldOperands);

  rewriter.replaceOp(parallelOp, loopResults);

  return success();
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
