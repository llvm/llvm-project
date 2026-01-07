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
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/STLExtras.h"

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
  auto expressionOp =
      emitc::ExpressionOp::create(builder, loc, resultType, op->getOperands());

  // Replace all op's uses with the new expression's result.
  result.replaceAllUsesWith(expressionOp.getResult());

  Block &block = expressionOp.createBody();
  IRMapping mapper;
  for (auto [operand, arg] :
       llvm::zip(expressionOp.getOperands(), block.getArguments()))
    mapper.map(operand, arg);
  builder.setInsertionPointToEnd(&block);

  Operation *rootOp = builder.clone(*op, mapper);
  op->erase();

  // Create an op to yield op's value.
  emitc::YieldOp::create(builder, loc, rootOp->getResults()[0]);
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
    Block *expressionBody = expressionOp.getBody();
    ExpressionOp usedExpression;
    SetVector<Value> foldedOperands;

    auto takesItsOperandsAddress = [](Operation *user) {
      auto applyOp = dyn_cast<emitc::ApplyOp>(user);
      return applyOp && applyOp.getApplicableOperator() == "&";
    };

    // Select as expression to fold the first operand expression that
    // - doesn't have its result value's address taken,
    // - has a single user: assume any re-materialization was done separately,
    // - has no side effects,
    // and save all other operands to be used later as operands in the folded
    // expression.
    for (auto [operand, arg] : llvm::zip(expressionOp.getOperands(),
                                         expressionBody->getArguments())) {
      ExpressionOp operandExpression = operand.getDefiningOp<ExpressionOp>();
      if (usedExpression || !operandExpression ||
          llvm::any_of(arg.getUsers(), takesItsOperandsAddress) ||
          !operandExpression.getResult().hasOneUse() ||
          operandExpression.hasSideEffects())
        foldedOperands.insert(operand);
      else
        usedExpression = operandExpression;
    }

    // If no operand expression was selected, bail out.
    if (!usedExpression)
      return failure();

    // Collect additional operands from the folded expression.
    for (Value operand : usedExpression.getOperands())
      foldedOperands.insert(operand);

    // Create a new expression to hold the folding result.
    rewriter.setInsertionPointAfter(expressionOp);
    auto foldedExpression = emitc::ExpressionOp::create(
        rewriter, expressionOp.getLoc(), expressionOp.getResult().getType(),
        foldedOperands.getArrayRef(), expressionOp.getDoNotInline());
    Block &foldedExpressionBody = foldedExpression.createBody();

    // Map each operand of the new expression to its matching block argument.
    IRMapping mapper;
    for (auto [operand, arg] : llvm::zip(foldedExpression.getOperands(),
                                         foldedExpressionBody.getArguments()))
      mapper.map(operand, arg);

    // Prepare to fold the used expression and the matched expression into the
    // newly created folded expression.
    auto foldExpression = [&rewriter, &mapper](ExpressionOp expressionToFold,
                                               bool withTerminator) {
      Block *expressionToFoldBody = expressionToFold.getBody();
      for (auto [operand, arg] :
           llvm::zip(expressionToFold.getOperands(),
                     expressionToFoldBody->getArguments())) {
        mapper.map(arg, mapper.lookup(operand));
      }

      for (Operation &opToClone : expressionToFoldBody->without_terminator())
        rewriter.clone(opToClone, mapper);

      if (withTerminator)
        rewriter.clone(*expressionToFoldBody->getTerminator(), mapper);
    };
    rewriter.setInsertionPointToStart(&foldedExpressionBody);

    // First, fold the used expression into the new expression and map its
    // result to the clone of its root operation within the new expression.
    foldExpression(usedExpression, /*withTerminator=*/false);
    Operation *expressionRoot = usedExpression.getRootOp();
    Operation *clonedExpressionRootOp = mapper.lookup(expressionRoot);
    assert(clonedExpressionRootOp &&
           "Expected cloned expression root to be in mapper");
    assert(clonedExpressionRootOp->getNumResults() == 1 &&
           "Expected cloned root to have a single result");
    mapper.map(usedExpression.getResult(),
               clonedExpressionRootOp->getResults()[0]);

    // Now fold the matched expression into the new expression.
    foldExpression(expressionOp, /*withTerminator=*/true);

    // Complete the rewrite.
    rewriter.replaceOp(expressionOp, foldedExpression);
    rewriter.eraseOp(usedExpression);

    return success();
  }
};

} // namespace

void mlir::emitc::populateExpressionPatterns(RewritePatternSet &patterns) {
  patterns.add<FoldExpressionOp>(patterns.getContext());
}
