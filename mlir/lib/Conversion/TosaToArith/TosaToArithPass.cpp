//===- TosaToArithPass.cpp - Lowering Tosa to Linalg Dialect -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transformation pass legalizes Tosa operations to the Arith dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/TosaToArith/TosaToArith.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
#define GEN_PASS_DEF_TOSATOARITHPASS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace tosa;

namespace {
struct TosaToArith : public impl::TosaToArithPassBase<TosaToArith> {
  using Base::Base;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());
    target.addIllegalOp<tosa::ConstOp>();
    target.addLegalDialect<arith::ArithDialect>();

    mlir::tosa::populateTosaToArithConversionPatterns(&patterns);

    if (this->includeApplyRescale) {
      mlir::tosa::populateTosaRescaleToArithConversionPatterns(&patterns,
                                                               this->use32Bit);
      target.addIllegalOp<tosa::ApplyScaleOp>();
    }

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace
