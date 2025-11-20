//===- MathToEmitCPass.cpp - Math to EmitC Pass -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to convert the Math dialect to the EmitC dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/MathToEmitC/MathToEmitCPass.h"
#include "mlir/Conversion/MathToEmitC/MathToEmitC.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTMATHTOEMITC
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
namespace {

//  Replaces Math operations with `emitc.call_opaque` operations.
struct ConvertMathToEmitC
    : public impl::ConvertMathToEmitCBase<ConvertMathToEmitC> {
  using ConvertMathToEmitCBase::ConvertMathToEmitCBase;

public:
  void runOnOperation() final;
};

} // namespace

void ConvertMathToEmitC::runOnOperation() {
  ConversionTarget target(getContext());
  target.addLegalOp<emitc::CallOpaqueOp>();

  target.addIllegalOp<math::FloorOp, math::ExpOp, math::RoundOp, math::CosOp,
                      math::SinOp, math::Atan2Op, math::CeilOp, math::AcosOp,
                      math::AsinOp, math::AbsFOp, math::PowFOp>();

  RewritePatternSet patterns(&getContext());
  populateConvertMathToEmitCPatterns(patterns, languageTarget);

  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}
