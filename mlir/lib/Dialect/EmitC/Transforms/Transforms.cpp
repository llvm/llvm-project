//===- Transforms.cpp - Patterns and transforms for the EmitC dialect -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/EmitC/Transforms/Transforms.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace emitc {

ExpressionOp createExpression(Operation *op, OpBuilder &builder) {
  assert(isa<emitc::CExpressionInterface>(op) && "Expected a C expression");

  // Create an expression yielding the value returned by op.
  assert(op->getNumResults() == 1 && "Expected exactly one result");
  Value result = op->getResult(0);
  Type resultType = result.getType();
  Location loc = op->getLoc();

  builder.setInsertionPointAfter(op);
  auto expressionOp = emitc::ExpressionOp::create(builder, loc, resultType);

  // Replace all op's uses with the new expression's result.
  result.replaceAllUsesWith(expressionOp.getResult());

  // Create an op to yield op's value.
  Region &region = expressionOp.getRegion();
  Block &block = region.emplaceBlock();
  builder.setInsertionPointToEnd(&block);
  auto yieldOp = emitc::YieldOp::create(builder, loc, result);

  // Move op into the new expression.
  op->moveBefore(yieldOp);

  return expressionOp;
}

} // namespace emitc
} // namespace mlir

using namespace mlir;
using namespace mlir::emitc;

namespace {

struct FoldExpressionOp : public OpRewritePattern<ExpressionOp> {
  using OpRewritePattern<ExpressionOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(ExpressionOp expressionOp,
                                PatternRewriter &rewriter) const override {
    bool anythingFolded = false;
    for (Operation &op : llvm::make_early_inc_range(
             expressionOp.getBody()->without_terminator())) {
      // Don't fold expressions whose result value has its address taken.
      auto applyOp = dyn_cast<emitc::ApplyOp>(op);
      if (applyOp && applyOp.getApplicableOperator() == "&")
        continue;

      for (Value operand : op.getOperands()) {
        auto usedExpression =
            dyn_cast_if_present<ExpressionOp>(operand.getDefiningOp());

        if (!usedExpression)
          continue;

        // Don't fold expressions with multiple users: assume any
        // re-materialization was done separately.
        if (!usedExpression.getResult().hasOneUse())
          continue;

        // Don't fold expressions with side effects.
        if (usedExpression.hasSideEffects())
          continue;

        // Fold the used expression into this expression by cloning all
        // instructions in the used expression just before the operation using
        // its value.
        rewriter.setInsertionPoint(&op);
        IRMapping mapper;
        for (Operation &opToClone :
             usedExpression.getBody()->without_terminator()) {
          Operation *clone = rewriter.clone(opToClone, mapper);
          mapper.map(&opToClone, clone);
        }

        Operation *expressionRoot = usedExpression.getRootOp();
        Operation *clonedExpressionRootOp = mapper.lookup(expressionRoot);
        assert(clonedExpressionRootOp &&
               "Expected cloned expression root to be in mapper");
        assert(clonedExpressionRootOp->getNumResults() == 1 &&
               "Expected cloned root to have a single result");

        rewriter.replaceOp(usedExpression, clonedExpressionRootOp);
        anythingFolded = true;
      }
    }
    return anythingFolded ? success() : failure();
  }
};

} // namespace

void mlir::emitc::populateExpressionPatterns(RewritePatternSet &patterns) {
  patterns.add<FoldExpressionOp>(patterns.getContext());
}
