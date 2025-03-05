//===- UpliftWhileToFor.cpp - scf.while to scf.for loop uplifting ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Transforms SCF.WhileOp's into SCF.ForOp's.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SCF/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;

namespace {
struct UpliftWhileOp : public OpRewritePattern<scf::WhileOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::WhileOp loop,
                                PatternRewriter &rewriter) const override {
    return upliftWhileToForLoop(rewriter, loop);
  }
};
} // namespace

FailureOr<scf::ForOp> mlir::scf::upliftWhileToForLoop(RewriterBase &rewriter,
                                                      scf::WhileOp loop) {
  Block *beforeBody = loop.getBeforeBody();
  if (!llvm::hasSingleElement(beforeBody->without_terminator()))
    return rewriter.notifyMatchFailure(loop, "Loop body must have single op");

  auto cmp = dyn_cast<arith::CmpIOp>(beforeBody->front());
  if (!cmp)
    return rewriter.notifyMatchFailure(loop,
                                       "Loop body must have single cmp op");

  scf::ConditionOp beforeTerm = loop.getConditionOp();
  if (!cmp->hasOneUse() || beforeTerm.getCondition() != cmp.getResult())
    return rewriter.notifyMatchFailure(loop, [&](Diagnostic &diag) {
      diag << "Expected single condition use: " << *cmp;
    });

  // All `before` block args must be directly forwarded to ConditionOp.
  // They will be converted to `scf.for` `iter_vars` except induction var.
  if (ValueRange(beforeBody->getArguments()) != beforeTerm.getArgs())
    return rewriter.notifyMatchFailure(loop, "Invalid args order");

  using Pred = arith::CmpIPredicate;
  Pred predicate = cmp.getPredicate();
  if (predicate != Pred::slt && predicate != Pred::sgt)
    return rewriter.notifyMatchFailure(loop, [&](Diagnostic &diag) {
      diag << "Expected 'slt' or 'sgt' predicate: " << *cmp;
    });

  BlockArgument inductionVar;
  Value ub;
  DominanceInfo dom;

  // Check if cmp has a suitable form. One of the arguments must be a `before`
  // block arg, other must be defined outside `scf.while` and will be treated
  // as upper bound.
  for (bool reverse : {false, true}) {
    auto expectedPred = reverse ? Pred::sgt : Pred::slt;
    if (cmp.getPredicate() != expectedPred)
      continue;

    auto arg1 = reverse ? cmp.getRhs() : cmp.getLhs();
    auto arg2 = reverse ? cmp.getLhs() : cmp.getRhs();

    auto blockArg = dyn_cast<BlockArgument>(arg1);
    if (!blockArg || blockArg.getOwner() != beforeBody)
      continue;

    if (!dom.properlyDominates(arg2, loop))
      continue;

    inductionVar = blockArg;
    ub = arg2;
    break;
  }

  if (!inductionVar)
    return rewriter.notifyMatchFailure(loop, [&](Diagnostic &diag) {
      diag << "Unrecognized cmp form: " << *cmp;
    });

  // inductionVar must have 2 uses: one is in `cmp` and other is `condition`
  // arg.
  if (!llvm::hasNItems(inductionVar.getUses(), 2))
    return rewriter.notifyMatchFailure(loop, [&](Diagnostic &diag) {
      diag << "Unrecognized induction var: " << inductionVar;
    });

  Block *afterBody = loop.getAfterBody();
  scf::YieldOp afterTerm = loop.getYieldOp();
  unsigned argNumber = inductionVar.getArgNumber();
  Value afterTermIndArg = afterTerm.getResults()[argNumber];

  Value inductionVarAfter = afterBody->getArgument(argNumber);

  // Find suitable `addi` op inside `after` block, one of the args must be an
  // Induction var passed from `before` block and second arg must be defined
  // outside of the loop and will be considered step value.
  // TODO: Add `subi` support?
  auto addOp = afterTermIndArg.getDefiningOp<arith::AddIOp>();
  if (!addOp)
    return rewriter.notifyMatchFailure(loop, "Didn't found suitable 'addi' op");

  Value step;
  if (addOp.getLhs() == inductionVarAfter) {
    step = addOp.getRhs();
  } else if (addOp.getRhs() == inductionVarAfter) {
    step = addOp.getLhs();
  }

  if (!step || !dom.properlyDominates(step, loop))
    return rewriter.notifyMatchFailure(loop, "Invalid 'addi' form");

  Value lb = loop.getInits()[argNumber];

  assert(lb.getType().isIntOrIndex());
  assert(lb.getType() == ub.getType());
  assert(lb.getType() == step.getType());

  llvm::SmallVector<Value> newArgs;

  // Populate inits for new `scf.for`, skip induction var.
  newArgs.reserve(loop.getInits().size());
  for (auto &&[i, init] : llvm::enumerate(loop.getInits())) {
    if (i == argNumber)
      continue;

    newArgs.emplace_back(init);
  }

  Location loc = loop.getLoc();

  // With `builder == nullptr`, ForOp::build will try to insert terminator at
  // the end of newly created block and we don't want it. Provide empty
  // dummy builder instead.
  auto emptyBuilder = [](OpBuilder &, Location, Value, ValueRange) {};
  auto newLoop =
      rewriter.create<scf::ForOp>(loc, lb, ub, step, newArgs, emptyBuilder);

  Block *newBody = newLoop.getBody();

  // Populate block args for `scf.for` body, move induction var to the front.
  newArgs.clear();
  ValueRange newBodyArgs = newBody->getArguments();
  for (auto i : llvm::seq<size_t>(0, newBodyArgs.size())) {
    if (i < argNumber) {
      newArgs.emplace_back(newBodyArgs[i + 1]);
    } else if (i == argNumber) {
      newArgs.emplace_back(newBodyArgs.front());
    } else {
      newArgs.emplace_back(newBodyArgs[i]);
    }
  }

  rewriter.inlineBlockBefore(loop.getAfterBody(), newBody, newBody->end(),
                             newArgs);

  auto term = cast<scf::YieldOp>(newBody->getTerminator());

  // Populate new yield args, skipping the induction var.
  newArgs.clear();
  for (auto &&[i, arg] : llvm::enumerate(term.getResults())) {
    if (i == argNumber)
      continue;

    newArgs.emplace_back(arg);
  }

  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(term);
  rewriter.replaceOpWithNewOp<scf::YieldOp>(term, newArgs);

  // Compute induction var value after loop execution.
  rewriter.setInsertionPointAfter(newLoop);
  Value one;
  if (isa<IndexType>(step.getType())) {
    one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  } else {
    one = rewriter.create<arith::ConstantIntOp>(loc, 1, step.getType());
  }

  Value stepDec = rewriter.create<arith::SubIOp>(loc, step, one);
  Value len = rewriter.create<arith::SubIOp>(loc, ub, lb);
  len = rewriter.create<arith::AddIOp>(loc, len, stepDec);
  len = rewriter.create<arith::DivSIOp>(loc, len, step);
  len = rewriter.create<arith::SubIOp>(loc, len, one);
  Value res = rewriter.create<arith::MulIOp>(loc, len, step);
  res = rewriter.create<arith::AddIOp>(loc, lb, res);

  // Reconstruct `scf.while` results, inserting final induction var value
  // into proper place.
  newArgs.clear();
  llvm::append_range(newArgs, newLoop.getResults());
  newArgs.insert(newArgs.begin() + argNumber, res);
  rewriter.replaceOp(loop, newArgs);
  return newLoop;
}

void mlir::scf::populateUpliftWhileToForPatterns(RewritePatternSet &patterns) {
  patterns.add<UpliftWhileOp>(patterns.getContext());
}
