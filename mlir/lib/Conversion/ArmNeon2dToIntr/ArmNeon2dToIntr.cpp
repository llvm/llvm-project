//===- ArmNeon2dToIntr.cpp - convert Arm Neon 2d ops to intrinsics --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ArmNeon2dToIntr/ArmNeon2dToIntr.h"
#include "../PassDetail.h"
#include "mlir/Dialect/ArmNeon/ArmNeonDialect.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::arm_neon;

namespace {

class Sdot2dLoweringPattern : public OpRewritePattern<Sdot2dOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  /// Convert to 1-dimensional vector type to match the requirements of
  /// arm.neon.intr.sdot
  LogicalResult matchAndRewrite(Sdot2dOp op,
                                PatternRewriter &rewriter) const override {
    Type elemType = op.b().getType().cast<VectorType>().getElementType();
    int length = op.b().getType().cast<VectorType>().getShape()[0] *
                 Sdot2dOp::kReductionSize;
    VectorType flattenedVectorType = VectorType::get({length}, elemType);
    Value b2d = op.b();
    Value c2d = op.c();
    Location loc = op.getLoc();
    Value b1d =
        rewriter.create<vector::ShapeCastOp>(loc, flattenedVectorType, b2d);
    Value c1d =
        rewriter.create<vector::ShapeCastOp>(loc, flattenedVectorType, c2d);
    Value newOp =
        rewriter.create<SdotOp>(loc, op.res().getType(), op.a(), b1d, c1d);
    rewriter.replaceOp(op, {newOp});
    return success();
  }
};

class ConvertArmNeon2dToIntr
    : public ConvertArmNeon2dToIntrBase<ConvertArmNeon2dToIntr> {
  void runOnOperation() override {
    auto func = getOperation();
    auto *context = &getContext();

    RewritePatternSet patterns(context);
    populateConvertArmNeon2dToIntrPatterns(patterns);

    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

namespace mlir {

void populateConvertArmNeon2dToIntrPatterns(RewritePatternSet &patterns) {
  patterns.add<Sdot2dLoweringPattern>(patterns.getContext());
}

std::unique_ptr<OperationPass<FuncOp>> createConvertArmNeon2dToIntrPass() {
  return std::make_unique<ConvertArmNeon2dToIntr>();
}

} // namespace mlir
