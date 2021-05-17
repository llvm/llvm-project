//===- ComplexToStandard.cpp - conversion from Complex to Standard dialect ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ComplexToStandard/ComplexToStandard.h"

#include <memory>

#include "../PassDetail.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {
struct AbsOpConversion : public OpConversionPattern<complex::AbsOp> {
  using OpConversionPattern<complex::AbsOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(complex::AbsOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    complex::AbsOp::Adaptor transformed(operands);
    auto loc = op.getLoc();
    auto type = op.getType();

    Value real =
        rewriter.create<complex::ReOp>(loc, type, transformed.complex());
    Value imag =
        rewriter.create<complex::ImOp>(loc, type, transformed.complex());
    Value realSqr = rewriter.create<MulFOp>(loc, real, real);
    Value imagSqr = rewriter.create<MulFOp>(loc, imag, imag);
    Value sqNorm = rewriter.create<AddFOp>(loc, realSqr, imagSqr);

    rewriter.replaceOpWithNewOp<math::SqrtOp>(op, sqNorm);
    return success();
  }
};
} // namespace

void mlir::populateComplexToStandardConversionPatterns(
    RewritePatternSet &patterns) {
  patterns.add<AbsOpConversion>(patterns.getContext());
}

namespace {
struct ConvertComplexToStandardPass
    : public ConvertComplexToStandardBase<ConvertComplexToStandardPass> {
  void runOnFunction() override;
};

void ConvertComplexToStandardPass::runOnFunction() {
  auto function = getFunction();

  // Convert to the Standard dialect using the converter defined above.
  RewritePatternSet patterns(&getContext());
  populateComplexToStandardConversionPatterns(patterns);

  ConversionTarget target(getContext());
  target.addLegalDialect<StandardOpsDialect, math::MathDialect,
                         complex::ComplexDialect>();
  target.addIllegalOp<complex::AbsOp>();
  if (failed(applyPartialConversion(function, target, std::move(patterns))))
    signalPassFailure();
}
} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::createConvertComplexToStandardPass() {
  return std::make_unique<ConvertComplexToStandardPass>();
}
