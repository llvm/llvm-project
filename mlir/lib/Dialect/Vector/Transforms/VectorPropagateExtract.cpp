//===- VectorPropagateExtract.cpp - vector.extract propagation - ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements patterns for vector.extract propagation.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"

using namespace mlir;

namespace {

/// Pattern to rewrite a ExtractOp(Elementwise) -> Elementwise(ExtractOp).
class ExtractOpFromElementwise final
    : public OpRewritePattern<vector::ExtractOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ExtractOp op,
                                PatternRewriter &rewriter) const override {
    Operation *eltwise = op.getVector().getDefiningOp();

    // Elementwise op with single result and `extract` is single user.
    if (!eltwise || !OpTrait::hasElementwiseMappableTraits(eltwise) ||
        eltwise->getNumResults() != 1 || !eltwise->hasOneUse())
      return failure();

    // Arguments and result types must match.
    if (!llvm::all_equal(llvm::concat<Type>(eltwise->getOperandTypes(),
                                            eltwise->getResultTypes())))
      return failure();

    Type dstType = op.getType();

    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(eltwise);

    IRMapping mapping;
    Location loc = eltwise->getLoc();
    for (auto &&[i, arg] : llvm::enumerate(eltwise->getOperands())) {
      Value newArg =
          rewriter.create<vector::ExtractOp>(loc, arg, op.getMixedPosition());
      mapping.map(arg, newArg);
    }

    Operation *newEltwise = rewriter.clone(*eltwise, mapping);
    newEltwise->getResult(0).setType(dstType);

    rewriter.replaceOp(op, newEltwise);
    rewriter.eraseOp(eltwise);
    return success();
  }
};

} // namespace

void mlir::vector::populateVectorPropagateExtractsPatterns(
    RewritePatternSet &patterns) {
  patterns.add<ExtractOpFromElementwise>(patterns.getContext());
}
