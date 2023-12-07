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
#include "mlir/IR/IRMapping.h"
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

class ScatterOpConverter : public OpRewritePattern<tosa::ScatterOp> {
  static Value createTensorDim(OpBuilder &builder, Location loc, Value tensor,
                               int64_t dim) {
    return builder.createOrFold<tensor::DimOp>(loc, tensor, dim);
  }

  static Value createIndexConst(OpBuilder &builder, Location loc,
                                int64_t value) {
    return builder.create<arith::ConstantIndexOp>(loc, value);
  }

public:
  using OpRewritePattern<tosa::ScatterOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::ScatterOp scatter,
                                PatternRewriter &rewriter) const final {
    auto valuesIn = scatter.getValuesIn();
    auto indices = scatter.getIndices();
    auto input = scatter.getInput();
    auto loc = scatter.getLoc();

    // N, W, C are chosen to match the TOSA spec
    auto dimN = createTensorDim(rewriter, loc, input, 0);
    auto dimW = createTensorDim(rewriter, loc, input, 1);
    auto dimC = createTensorDim(rewriter, loc, input, 2);

    auto zero = createIndexConst(rewriter, loc, 0);
    auto one = createIndexConst(rewriter, loc, 1);

    // Loop bounds
    auto lbs = llvm::SmallVector<Value>(2, zero);
    auto steps = llvm::SmallVector<Value>(2, one);
    auto ubs = llvm::SmallVector<Value>{{dimN, dimW}};

    auto buildBody = [&](OpBuilder &builder, Location loc, ValueRange ivs,
                         ValueRange args) -> scf::ValueVector {
      auto n = ivs[0];

      // Read the index and cast it to index type
      auto index = builder.create<tensor::ExtractOp>(loc, indices, ivs);
      auto castIndex = builder.create<arith::IndexCastOp>(
          loc, builder.getIndexType(), index);

      // Offset, sizes, and strides for the input tensor
      auto inputOffset = llvm::to_vector(ivs);
      inputOffset.push_back(zero);

      llvm::SmallVector<Value> sizes = {one, one, dimC};
      llvm::SmallVector<Value> strides = {one, one, one};

      auto slice = builder.create<tensor::ExtractSliceOp>(
          loc, input, inputOffset, sizes, strides);

      // Insert the slice into the output accumulator tensor.
      llvm::SmallVector<Value> outputOffset = {n, castIndex, zero};
      auto updated = builder.create<tensor::InsertSliceOp>(
          loc, slice, args[0], outputOffset, sizes, strides);

      return {updated};
    };

    auto loops = scf::buildLoopNest(rewriter, loc, lbs, ubs, steps,
                                    ValueRange{valuesIn}, buildBody);
    rewriter.replaceOp(scatter, loops.results);

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
  patterns->add<IfOpConverter, ScatterOpConverter, WhileOpConverter>(
      patterns->getContext());
}
