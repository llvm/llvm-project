//===- ConvertShapeConstraints.cpp - Conversion of shape constraints ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ShapeToStandard/ShapeToStandard.h"

#include "../PassDetail.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {
class ConvertCstrBroadcastableOp
    : public OpRewritePattern<shape::CstrBroadcastableOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(shape::CstrBroadcastableOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getType().isa<shape::ShapeType>() ||
        op.lhs().getType().isa<shape::ShapeType>() ||
        op.rhs().getType().isa<shape::ShapeType>()) {
      return rewriter.notifyMatchFailure(
          op, "cannot convert error-propagating shapes");
    }

    auto loc = op.getLoc();
    Value zero = rewriter.create<ConstantIndexOp>(loc, 0);
    Value one = rewriter.create<ConstantIndexOp>(loc, 1);

    // Find smaller and greater rank and extent tensor.
    Value lhsRank = rewriter.create<DimOp>(loc, op.lhs(), zero);
    Value rhsRank = rewriter.create<DimOp>(loc, op.rhs(), zero);
    Value lhsRankULE =
        rewriter.create<CmpIOp>(loc, CmpIPredicate::ule, lhsRank, rhsRank);
    Type indexTy = rewriter.getIndexType();
    Value lesserRank =
        rewriter.create<SelectOp>(loc, lhsRankULE, lhsRank, rhsRank);
    Value greaterRank =
        rewriter.create<SelectOp>(loc, lhsRankULE, rhsRank, lhsRank);
    Value lesserRankOperand =
        rewriter.create<SelectOp>(loc, lhsRankULE, op.lhs(), op.rhs());
    Value greaterRankOperand =
        rewriter.create<SelectOp>(loc, lhsRankULE, op.rhs(), op.lhs());

    Value rankDiff =
        rewriter.create<SubIOp>(loc, indexTy, greaterRank, lesserRank);

    // Generate code to compare the shapes extent by extent, and emit errors for
    // non-broadcast-compatible shapes.
    // Two extents are broadcast-compatible if
    // 1. they are both equal, or
    // 2. at least one of them is 1.

    rewriter.create<scf::ForOp>(
        loc, rankDiff, greaterRank, one, llvm::None,
        [&](OpBuilder &b, Location loc, Value iv, ValueRange) {
          Value greaterRankOperandExtent = b.create<tensor::ExtractOp>(
              loc, greaterRankOperand, ValueRange{iv});
          Value ivShifted = b.create<SubIOp>(loc, indexTy, iv, rankDiff);
          Value lesserRankOperandExtent = b.create<tensor::ExtractOp>(
              loc, lesserRankOperand, ValueRange{ivShifted});

          Value greaterRankOperandExtentIsOne = b.create<CmpIOp>(
              loc, CmpIPredicate::eq, greaterRankOperandExtent, one);
          Value lesserRankOperandExtentIsOne = b.create<CmpIOp>(
              loc, CmpIPredicate::eq, lesserRankOperandExtent, one);
          Value extentsAgree =
              b.create<CmpIOp>(loc, CmpIPredicate::eq, greaterRankOperandExtent,
                               lesserRankOperandExtent);
          auto broadcastIsValid =
              b.create<OrOp>(loc, b.getI1Type(), extentsAgree,
                             b.create<OrOp>(loc, greaterRankOperandExtentIsOne,
                                            lesserRankOperandExtentIsOne));
          b.create<AssertOp>(loc, broadcastIsValid, "invalid broadcast");
          b.create<scf::YieldOp>(loc);
        });

    rewriter.replaceOpWithNewOp<shape::ConstWitnessOp>(op, true);
    return success();
  }
};
} // namespace

namespace {
class ConvertCstrRequireOp : public OpRewritePattern<shape::CstrRequireOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(shape::CstrRequireOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.create<AssertOp>(op.getLoc(), op.pred(), op.msgAttr());
    rewriter.replaceOpWithNewOp<shape::ConstWitnessOp>(op, true);
    return success();
  }
};
} // namespace

void mlir::populateConvertShapeConstraintsConversionPatterns(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns.insert<ConvertCstrBroadcastableOp>(ctx);
  patterns.insert<ConvertCstrRequireOp>(ctx);
}

namespace {
// This pass eliminates shape constraints from the program, converting them to
// eager (side-effecting) error handling code. After eager error handling code
// is emitted, witnesses are satisfied, so they are replace with
// `shape.const_witness true`.
class ConvertShapeConstraints
    : public ConvertShapeConstraintsBase<ConvertShapeConstraints> {
  void runOnOperation() override {
    auto func = getOperation();
    auto *context = &getContext();

    OwningRewritePatternList patterns;
    populateConvertShapeConstraintsConversionPatterns(patterns, context);

    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::createConvertShapeConstraintsPass() {
  return std::make_unique<ConvertShapeConstraints>();
}
