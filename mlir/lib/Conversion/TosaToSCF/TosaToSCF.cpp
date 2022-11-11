//===- TosaToSCF.cpp - Lowering Tosa to SCF Dialect -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// These rewriters lower from the Tosa to the SCF dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/TosaToSCF/TosaToSCF.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace tosa;

static void inlineIfCase(Region &srcRegion, Region &dstRegion,
                         OperandRange operands, PatternRewriter &rewriter) {
  rewriter.cloneRegionBefore(srcRegion, &dstRegion.front());
  rewriter.eraseBlock(&dstRegion.back());

  Block *headBlock = &dstRegion.front();
  for (auto it : llvm::zip(headBlock->getArguments(), operands))
    std::get<0>(it).replaceAllUsesWith(std::get<1>(it));

  auto yield = cast<YieldOp>(headBlock->getTerminator());
  rewriter.setInsertionPoint(yield);
  rewriter.create<scf::YieldOp>(yield.getLoc(), yield.getInputs());
  rewriter.eraseOp(yield);

  headBlock->eraseArguments(0, headBlock->getNumArguments());
}

static void inlineWhileCase(Region &srcRegion, Region &dstRegion,
                            PatternRewriter &rewriter, bool isCond) {
  rewriter.cloneRegionBefore(srcRegion, &dstRegion.back());
  rewriter.eraseBlock(&dstRegion.back());

  Block *headBlock = &dstRegion.front();

  auto yield = cast<YieldOp>(headBlock->getTerminator());
  rewriter.setInsertionPoint(yield);
  if (isCond) {
    auto condition =
        rewriter.create<tensor::ExtractOp>(yield.getLoc(), yield.getOperand(0));
    rewriter.create<scf::ConditionOp>(yield.getLoc(), condition,
                                      headBlock->getArguments());
  } else {
    rewriter.setInsertionPoint(yield);
    rewriter.create<scf::YieldOp>(yield.getLoc(), yield.getInputs());
  }
  rewriter.eraseOp(yield);
}

namespace {

class IfOpConverter : public OpRewritePattern<tosa::IfOp> {
public:
  using OpRewritePattern<tosa::IfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::IfOp op,
                                PatternRewriter &rewriter) const final {
    auto condition =
        rewriter.create<tensor::ExtractOp>(op.getLoc(), op.getCond());
    auto newIf = rewriter.create<scf::IfOp>(op.getLoc(), op.getResultTypes(),
                                            condition, true);

    inlineIfCase(op.getThenBranch(), newIf.getThenRegion(), op.getInputs(),
                 rewriter);
    inlineIfCase(op.getElseBranch(), newIf.getElseRegion(), op.getInputs(),
                 rewriter);

    rewriter.replaceOp(op, newIf.getResults());
    return success();
  }
};

class WhileOpConverter : public OpRewritePattern<tosa::WhileOp> {
public:
  using OpRewritePattern<tosa::WhileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::WhileOp op,
                                PatternRewriter &rewriter) const final {
    auto newWhile = rewriter.create<scf::WhileOp>(
        op.getLoc(), op.getResultTypes(), op.getInputs());
    rewriter.createBlock(&newWhile.getBefore());
    rewriter.createBlock(&newWhile.getAfter());

    inlineWhileCase(op.getCond(), newWhile.getBefore(), rewriter, true);
    inlineWhileCase(op.getBody(), newWhile.getAfter(), rewriter, false);

    rewriter.replaceOp(op, newWhile.getResults());

    return success();
  }
};

} // namespace

void mlir::tosa::populateTosaToSCFConversionPatterns(
    RewritePatternSet *patterns) {
  patterns->add<IfOpConverter>(patterns->getContext());
  patterns->add<WhileOpConverter>(patterns->getContext());
}
