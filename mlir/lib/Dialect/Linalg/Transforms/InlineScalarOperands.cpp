//===- InlineScalarOperands.cpp - Pass to inline scalar operands =============//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements patterns/pass to inline scalar operands into a generic
// operation. A scalar operand is an operand whose indexing map has a constant
// rhs.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::linalg;

namespace {
struct InlineScalarOperands : public OpRewritePattern<GenericOp> {
  using OpRewritePattern<GenericOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    if (!genericOp.hasTensorSemantics())
      return failure();

    SmallVector<size_t> scalarOperands;
    SmallVector<AffineMap> newIndexingMaps;
    SmallVector<Value> newOperands;
    for (OpOperand *opOperand : genericOp.getInputOperands()) {
      AffineMap map = genericOp.getTiedIndexingMap(opOperand);
      if (genericOp.isInputTensor(opOperand) && map.isConstant()) {
        scalarOperands.emplace_back(opOperand->getOperandNumber());
      } else {
        newIndexingMaps.emplace_back(map);
        newOperands.emplace_back(opOperand->get());
      }
    }

    if (scalarOperands.empty())
      return failure();

    for (OpOperand *opOperand : genericOp.getOutputOperands())
      newIndexingMaps.emplace_back(genericOp.getTiedIndexingMap(opOperand));

    Location loc = genericOp->getLoc();
    SmallVector<Value> outputOperands = genericOp.getOutputOperands();
    auto newOp = rewriter.create<GenericOp>(
        loc, genericOp->getResultTypes(), newOperands, outputOperands,
        newIndexingMaps,
        llvm::to_vector<4>(
            genericOp.iterator_types().template getAsValueRange<StringAttr>()));
    rewriter.cloneRegionBefore(genericOp.region(), newOp.region(),
                               newOp.region().begin());

    Block *body = newOp.getBody();
    PatternRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(body);

    for (auto idx : llvm::reverse(scalarOperands)) {
      OpOperand *opOperand = genericOp.getInputOperand(idx);
      AffineMap map = genericOp.getTiedIndexingMap(opOperand);
      SmallVector<int64_t> indices = map.getConstantResults();
      SmallVector<Value> indicesValues;
      for (auto idx : indices)
        indicesValues.emplace_back(rewriter.create<ConstantIndexOp>(loc, idx));
      Value extractedValue = rewriter.create<tensor::ExtractOp>(
          loc, opOperand->get(), indicesValues);
      body->getArgument(idx).replaceAllUsesWith(extractedValue);
      body->eraseArgument(idx);
    }

    rewriter.replaceOp(genericOp, newOp->getResults());
    return success();
  }
};
} // namespace

/// Patterns that are used to inline constant operands into linalg generic
/// ops.
void mlir::linalg::populateInlineConstantOperandsPatterns(
    RewritePatternSet &patterns) {
  auto *context = patterns.getContext();
  patterns.add<InlineScalarOperands>(context);
}

namespace {
/// Pass that removes unit-extent dims within generic ops.
struct LinalgInlineScalarOperandsPass
    : public LinalgInlineScalarOperandsBase<LinalgInlineScalarOperandsPass> {
  void runOnFunction() override {
    FuncOp funcOp = getFunction();
    MLIRContext *context = funcOp.getContext();
    RewritePatternSet patterns(context);

    populateInlineConstantOperandsPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(funcOp.getBody(), std::move(patterns));
  }
};
} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::createLinalgInlineScalarOperandsPass() {
  return std::make_unique<LinalgInlineScalarOperandsPass>();
}
