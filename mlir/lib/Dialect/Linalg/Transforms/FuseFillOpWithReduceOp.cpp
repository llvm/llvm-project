//===- FuseFillOpWithReduceOp.cpp - Fuse linalg fill with reduce producer -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements patterns that fuses a linalg.generic -> tensor.pad op
// chain into a tensor.extract_slice -> linalg.generic -> tensor.insert_slice
// op chain.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/Transforms/Transforms.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

/// Fold linalg.fill into linalg.reduce by creating a fused linalg.generic
/// operation. The fill operation is expected to happen only on the first index
/// of the reduction dimension. Currently only one reduction dimension is
/// supported. Given the pattern:
///   %empty = tensor.empty() : tensor<i8>
///   %filled = linalg.fill ins(%c0 : i8) outs(%empty : tensor<i8>) ->
///   tensor<i8> %reduced = linalg.reduce ins(%0 : tensor<147456xi8>)
///   outs(%filled : tensor<i8>) dimensions = [0]
///     (%in: i8, %init: i8) {
///       %3 = arith.addi %in, %init : i8
///       linalg.yield %3 : i8
///   }
/// The pattern is rewritten into:
///   %empty = tensor.empty() : tensor<i8>
///   %reduced = linalg.generic ins(%0 : tensor<147456xi8>) outs(%empty :
///   tensor<i8>) {
///     ^bb0(%in: i8, %init: i8):
///       %cst = arith.constant 0 : index
///       %index = linalg.index %c0 : index
///       %cmp = arith.cmpi eq, %cst, %index : i1
///       %sum = arith.select %cmp, %c0, %init : i8
///       %res = arith.addi %in, %sum : i8
///       linalg.yield %res : i8
///   }
struct FoldFillWithReduceOp : public OpRewritePattern<linalg::ReduceOp> {
  using OpRewritePattern<linalg::ReduceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::ReduceOp reduceOp,
                                PatternRewriter &rewriter) const override {
    if (!reduceOp.hasPureTensorSemantics())
      return rewriter.notifyMatchFailure(
          reduceOp, "skip reduce op with non-pure tensor semantics");
    if (reduceOp.getDimensions().size() != 1)
      return rewriter.notifyMatchFailure(
          reduceOp, "skip reduce op with non-single dimension");
    if (reduceOp.getNumDpsInputs() != 1 || reduceOp.getNumDpsInits() != 1)
      return rewriter.notifyMatchFailure(
          reduceOp, "skip reduce op with multiple number of inputs/results");
    auto fillOp = reduceOp.getInits()[0].getDefiningOp<linalg::FillOp>();
    if (!fillOp)
      return rewriter.notifyMatchFailure(
          reduceOp,
          "skip reduce op with inits not directly based on fill operation");

    long dim = reduceOp.getDimensions()[0];
    // Note: on success, the `reduceOp` is replaced with a genericOp and no
    // longer valid.
    auto failureOrGenericOp = linalg::generalizeNamedOp(rewriter, reduceOp);
    if (failed(failureOrGenericOp))
      return rewriter.notifyMatchFailure(reduceOp,
                                         "failed to generalize reduce op");

    linalg::GenericOp genericReduceOp = *failureOrGenericOp;
    auto operandIdx = -1;
    for (auto &use : genericReduceOp->getOpOperands()) {
      if (use.get().getDefiningOp() == fillOp)
        operandIdx = use.getOperandNumber();
    }
    assert(operandIdx != -1 && "fill op not found in reduce op uses");

    Location loc = genericReduceOp.getLoc();
    auto blockArg = genericReduceOp.getMatchingBlockArgument(
        &genericReduceOp->getOpOperand(operandIdx));
    rewriter.setInsertionPointToStart(genericReduceOp.getBody());
    auto constZeroIndexOp = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto linalgIndexOp = rewriter.create<linalg::IndexOp>(loc, dim);
    auto cmpIOp = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                 constZeroIndexOp.getResult(),
                                                 linalgIndexOp.getResult());
    auto selectOp =
        rewriter.create<arith::SelectOp>(loc, cmpIOp, fillOp.value(), blockArg);
    rewriter.replaceAllUsesExcept(blockArg, selectOp.getResult(), selectOp);
    genericReduceOp->setOperand(operandIdx, fillOp.getDpsInitOperand(0)->get());

    return success();
  }
};

} // namespace

void mlir::linalg::populateFuseFillOpWithReduceOpPatterns(
    RewritePatternSet &patterns) {
  patterns.add<FoldFillWithReduceOp>(patterns.getContext());
}
