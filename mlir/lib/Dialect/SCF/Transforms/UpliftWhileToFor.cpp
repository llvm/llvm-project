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
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_SCFUPLIFTWHILETOFOR
#include "mlir/Dialect/SCF/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {
struct UpliftWhileOp : public OpRewritePattern<scf::WhileOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::WhileOp loop,
                                PatternRewriter &rewriter) const override {
    Block *beforeBody = loop.getBeforeBody();
    if (!llvm::hasSingleElement(beforeBody->without_terminator()))
      return rewriter.notifyMatchFailure(loop, "Loop body must have single op");

    auto cmp = dyn_cast<arith::CmpIOp>(beforeBody->front());
    if (!cmp)
      return rewriter.notifyMatchFailure(loop,
                                         "Loop body must have single cmp op");

    auto beforeTerm = cast<scf::ConditionOp>(beforeBody->getTerminator());
    if (!llvm::hasSingleElement(cmp->getUses()) &&
        beforeTerm.getCondition() == cmp.getResult())
      return rewriter.notifyMatchFailure(loop, [&](Diagnostic &diag) {
        diag << "Expected single condiditon use: " << *cmp;
      });

    if (ValueRange(beforeBody->getArguments()) != beforeTerm.getArgs())
      return rewriter.notifyMatchFailure(loop, "Invalid args order");

    using Pred = arith::CmpIPredicate;
    auto predicate = cmp.getPredicate();
    if (predicate != Pred::slt && predicate != Pred::sgt)
      return rewriter.notifyMatchFailure(loop, [&](Diagnostic &diag) {
        diag << "Expected 'slt' or 'sgt' predicate: " << *cmp;
      });

    BlockArgument iterVar;
    Value end;
    DominanceInfo dom;
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

      iterVar = blockArg;
      end = arg2;
      break;
    }

    if (!iterVar)
      return rewriter.notifyMatchFailure(loop, [&](Diagnostic &diag) {
        diag << "Unrecognized cmp form: " << *cmp;
      });

    if (!llvm::hasNItems(iterVar.getUses(), 2))
      return rewriter.notifyMatchFailure(loop, [&](Diagnostic &diag) {
        diag << "Unrecognized iter var: " << iterVar;
      });

    Block *afterBody = loop.getAfterBody();
    auto afterTerm = cast<scf::YieldOp>(afterBody->getTerminator());
    auto argNumber = iterVar.getArgNumber();
    auto afterTermIterArg = afterTerm.getResults()[argNumber];

    auto iterVarAfter = afterBody->getArgument(argNumber);

    Value step;
    for (auto &use : iterVarAfter.getUses()) {
      auto owner = dyn_cast<arith::AddIOp>(use.getOwner());
      if (!owner)
        continue;

      auto other =
          (iterVarAfter == owner.getLhs() ? owner.getRhs() : owner.getLhs());
      if (!dom.properlyDominates(other, loop))
        continue;

      if (afterTermIterArg != owner.getResult())
        continue;

      step = other;
      break;
    }

    if (!step)
      return rewriter.notifyMatchFailure(loop,
                                         "Didn't found suitable 'add' op");

    auto begin = loop.getInits()[argNumber];

    assert(begin.getType().isIntOrIndex());
    assert(begin.getType() == end.getType());
    assert(begin.getType() == step.getType());

    llvm::SmallVector<Value> mapping;
    mapping.reserve(loop.getInits().size());
    for (auto &&[i, init] : llvm::enumerate(loop.getInits())) {
      if (i == argNumber)
        continue;

      mapping.emplace_back(init);
    }

    auto loc = loop.getLoc();
    auto emptyBuidler = [](OpBuilder &, Location, Value, ValueRange) {};
    auto newLoop = rewriter.create<scf::ForOp>(loc, begin, end, step, mapping,
                                               emptyBuidler);

    Block *newBody = newLoop.getBody();

    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointToStart(newBody);
    Value newIterVar = newBody->getArgument(0);

    mapping.clear();
    auto newArgs = newBody->getArguments();
    for (auto i : llvm::seq<size_t>(0, newArgs.size())) {
      if (i < argNumber) {
        mapping.emplace_back(newArgs[i + 1]);
      } else if (i == argNumber) {
        mapping.emplace_back(newArgs.front());
      } else {
        mapping.emplace_back(newArgs[i]);
      }
    }

    rewriter.inlineBlockBefore(loop.getAfterBody(), newBody, newBody->end(),
                               mapping);

    auto term = cast<scf::YieldOp>(newBody->getTerminator());

    mapping.clear();
    for (auto &&[i, arg] : llvm::enumerate(term.getResults())) {
      if (i == argNumber)
        continue;

      mapping.emplace_back(arg);
    }

    rewriter.setInsertionPoint(term);
    rewriter.replaceOpWithNewOp<scf::YieldOp>(term, mapping);

    rewriter.setInsertionPointAfter(newLoop);
    Value one;
    if (isa<IndexType>(step.getType())) {
      one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    } else {
      one = rewriter.create<arith::ConstantIntOp>(loc, 1, step.getType());
    }

    Value stepDec = rewriter.create<arith::SubIOp>(loc, step, one);
    Value len = rewriter.create<arith::SubIOp>(loc, end, begin);
    len = rewriter.create<arith::AddIOp>(loc, len, stepDec);
    len = rewriter.create<arith::DivSIOp>(loc, len, step);
    len = rewriter.create<arith::SubIOp>(loc, len, one);
    Value res = rewriter.create<arith::MulIOp>(loc, len, step);
    res = rewriter.create<arith::AddIOp>(loc, begin, res);

    mapping.clear();
    llvm::append_range(mapping, newLoop.getResults());
    mapping.insert(mapping.begin() + argNumber, res);
    rewriter.replaceOp(loop, mapping);
    return success();
  }
};

struct SCFUpliftWhileToFor final
    : impl::SCFUpliftWhileToForBase<SCFUpliftWhileToFor> {
  using SCFUpliftWhileToForBase::SCFUpliftWhileToForBase;

  void runOnOperation() override {
    Operation *op = getOperation();
    MLIRContext *ctx = op->getContext();
    RewritePatternSet patterns(ctx);
    mlir::scf::populateUpliftWhileToForPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

void mlir::scf::populateUpliftWhileToForPatterns(RewritePatternSet &patterns) {
  patterns.add<UpliftWhileOp>(patterns.getContext());
}
