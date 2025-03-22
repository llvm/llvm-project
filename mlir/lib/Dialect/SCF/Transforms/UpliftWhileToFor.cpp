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

static Operation *findOpToMoveFromBefore(scf::WhileOp loop) {
  Block *body = loop.getBeforeBody();
  if (body->without_terminator().empty())
    return nullptr;

  // Check last op first.
  // TODO: It's usually safe to move and duplicate last op even if it has side
  // effects, as long as the sequence of the ops executed on each path will stay
  // the same. Exceptions are GPU barrier/group ops, LLVM proper has
  // convergent attribute/semantics to check this, but we doesn't model it yet.
  Operation *lastOp = &(*std::prev(body->without_terminator().end()));

  auto term = loop.getConditionOp();
  Operation *termCondOp = term.getCondition().getDefiningOp();
  if (lastOp != termCondOp)
    return lastOp;

  // Try to move terminator args producers.
  for (Value termArg : term.getArgs()) {
    Operation *op = termArg.getDefiningOp();
    if (!op || op->getParentOp() != loop || op == termCondOp || !isPure(op))
      continue;

    // Each result must be only used as terminator arg, meaning it can have one
    // use at max, duplicated terminator args must be already cleaned up
    // by canonicalizations at this point.
    if (!llvm::all_of(op->getResults(), [&](Value val) {
          return val.hasOneUse() || val.use_empty();
        }))
      continue;

    return op;
  }
  return nullptr;
}

namespace {
/// `scf.while` uplifting expects before block consisting of single cmp op,
/// try to move ops from before block to after block and to after loop.
///
/// ```
/// scf.while(...) {
/// before:
///   ...
///   some_op()
///   scf.condition ..
/// after:
///   ...
/// }
/// ```
/// to
/// ```
/// scf.while(...) {
/// before:
///   ...
///   scf.condition ..
/// after:
///   some_op()
///   ...
/// }
/// some_op()
/// ```
struct MoveOpsFromBefore : public OpRewritePattern<scf::WhileOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::WhileOp loop,
                                PatternRewriter &rewriter) const override {
    Operation *opToMove = findOpToMoveFromBefore(loop);
    if (!opToMove)
      return rewriter.notifyMatchFailure(loop, "No suitable ops found");

    auto condOp = loop.getConditionOp();
    SmallVector<Value> newCondArgs;

    // Populate new terminator args.

    // Add original terminator args, except args produced by the op we decided
    // to move.
    for (Value arg : condOp.getArgs()) {
      if (arg.getDefiningOp() == opToMove)
        continue;

      newCondArgs.emplace_back(arg);
    }
    auto originalArgsOffset = newCondArgs.size();

    // Add moved op operands to terminator args, if they are defined in loop
    // block.
    DominanceInfo dom;
    for (Value arg : opToMove->getOperands()) {
      if (dom.properlyDominates(arg, loop))
        continue;

      newCondArgs.emplace_back(arg);
    }

    // Create new loop.
    ValueRange tempRange(newCondArgs);
    auto newLoop = rewriter.create<mlir::scf::WhileOp>(
        loop.getLoc(), TypeRange(tempRange), loop.getInits(), nullptr, nullptr);

    OpBuilder::InsertionGuard g(rewriter);

    // Create new terminator, old terminator will be deleted later.
    rewriter.setInsertionPoint(condOp);
    rewriter.create<scf::ConditionOp>(condOp.getLoc(), condOp.getCondition(),
                                      newCondArgs);

    Block *oldBefore = loop.getBeforeBody();
    Block *newBefore = newLoop.getBeforeBody();

    // Inline before block as is.
    rewriter.inlineBlockBefore(oldBefore, newBefore, newBefore->begin(),
                               newBefore->getArguments());

    Block *oldAfter = loop.getAfterBody();
    Block *newAfter = newLoop.getAfterBody();

    // Build mapping between original op args and new after block args/new loop
    // results.
    IRMapping afterBodyMapping;
    IRMapping afterLoopMapping;
    {
      ValueRange blockArgs =
          newAfter->getArguments().drop_front(originalArgsOffset);
      ValueRange newLoopArgs =
          newLoop.getResults().drop_front(originalArgsOffset);
      for (Value arg : opToMove->getOperands()) {
        if (dom.properlyDominates(arg, loop))
          continue;

        assert(!blockArgs.empty());
        assert(!newLoopArgs.empty());
        afterBodyMapping.map(arg, blockArgs.front());
        afterLoopMapping.map(arg, newLoopArgs.front());
        blockArgs = blockArgs.drop_front();
        newLoopArgs = newLoopArgs.drop_front();
      }
    }

    {
      // Clone op into after body.
      rewriter.setInsertionPointToStart(oldAfter);
      Operation *newAfterBodyOp = rewriter.clone(*opToMove, afterBodyMapping);

      // Clone op after loop.
      rewriter.setInsertionPointAfter(newLoop);
      Operation *newAfterLoopOp = rewriter.clone(*opToMove, afterLoopMapping);

      // Build mapping between old and new after block args and between old and
      // new loop results.
      ValueRange blockArgs =
          newAfter->getArguments().take_front(originalArgsOffset);
      ValueRange newLoopArgs =
          newLoop.getResults().take_front(originalArgsOffset);
      SmallVector<Value> argsMapping;
      SmallVector<Value> newLoopResults;
      for (Value arg : condOp.getArgs()) {
        if (arg.getDefiningOp() == opToMove) {
          auto resNumber = cast<OpResult>(arg).getResultNumber();
          argsMapping.emplace_back(newAfterBodyOp->getResult(resNumber));
          newLoopResults.emplace_back(newAfterLoopOp->getResult(resNumber));
          continue;
        }

        assert(!blockArgs.empty());
        assert(!newLoopArgs.empty());
        argsMapping.emplace_back(blockArgs.front());
        newLoopResults.emplace_back(newLoopArgs.front());
        blockArgs = blockArgs.drop_front();
        newLoopArgs = newLoopArgs.drop_front();
      }

      // Inline after block.
      rewriter.inlineBlockBefore(oldAfter, newAfter, newAfter->begin(),
                                 argsMapping);

      // Replace loop.
      rewriter.replaceOp(loop, newLoopResults);
    }

    // Finally, we can remove old terminator and the original op.
    rewriter.eraseOp(condOp);
    rewriter.eraseOp(opToMove);
    return success();
  }
};

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

void mlir::scf::populatePrepareUpliftWhileToForPatterns(
    RewritePatternSet &patterns) {
  patterns.add<MoveOpsFromBefore>(patterns.getContext());
}

void mlir::scf::populateUpliftWhileToForPatterns(RewritePatternSet &patterns) {
  patterns.add<UpliftWhileOp>(patterns.getContext());
}
