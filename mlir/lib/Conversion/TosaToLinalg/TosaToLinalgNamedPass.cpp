//===- TosaToLinalgPass.cpp - Lowering Tosa to Linalg Dialect -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transformation pass legalizes Tosa operations to the Linalg dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/TosaToLinalg/TosaToLinalg.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCFDialect.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
#define GEN_PASS_DEF_TOSATOLINALGNAMED
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {
struct TosaToLinalgNamed
    : public impl::TosaToLinalgNamedBase<TosaToLinalgNamed> {
public:
  TosaToLinalgNamed(const TosaToLinalgNamedOptions &options)
      : impl::TosaToLinalgNamedBase<TosaToLinalgNamed>(options) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<arith::ArithDialect, linalg::LinalgDialect, math::MathDialect,
                tensor::TensorDialect, scf::SCFDialect>();
  }

  void runOnOperation() override {
    TypeConverter converter;
    tosa::populateTosaTypeConversion(converter);

    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());
    target.addLegalDialect<linalg::LinalgDialect, tosa::TosaDialect,
                           tensor::TensorDialect, scf::SCFDialect>();

    // Not every TOSA op can be legalized to linalg.
    target.addIllegalOp<tosa::Conv2DOp>();
    target.addIllegalOp<tosa::Conv3DOp>();
    target.addIllegalOp<tosa::DepthwiseConv2DOp>();
    target.addIllegalOp<tosa::MaxPool2dOp>();
    target.addIllegalOp<tosa::AvgPool2dOp>();
    target.addIllegalOp<tosa::MatMulOp>();
    target.addIllegalOp<tosa::TransposeOp>();

    // TOSA elementwise ops are legal unless they can be lowered to a
    // linalg.elementwise named op.
    target.addDynamicallyLegalOp<
        tosa::AbsOp, tosa::CeilOp, tosa::FloorOp, tosa::ExpOp, tosa::LogOp,
        tosa::RsqrtOp, tosa::SinOp, tosa::CosOp, tosa::TanhOp, tosa::ErfOp,
        tosa::ReciprocalOp, tosa::PowOp, tosa::AddOp, tosa::SubOp,
        tosa::IntDivOp, tosa::MulOp, tosa::NegateOp, tosa::MaximumOp,
        tosa::MinimumOp, tosa::SelectOp>([](Operation *op) {
      return !tosa::isConvertibleToLinalgElementwise(op);
    });

    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    FunctionOpInterface func = getOperation();
    TosaToLinalgNamedOptions options;
    options.preferConv2DKernelLayoutHWCF = preferConv2DKernelLayoutHWCF;
    tosa::populateTosaToLinalgNamedConversionPatterns(converter, &patterns,
                                                      options);
    if (failed(applyFullConversion(func, target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass>
mlir::tosa::createTosaToLinalgNamed(const TosaToLinalgNamedOptions &options) {
  return std::make_unique<TosaToLinalgNamed>(options);
}
