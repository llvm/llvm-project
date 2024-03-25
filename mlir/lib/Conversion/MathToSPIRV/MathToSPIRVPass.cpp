//===- MathToSPIRVPass.cpp - Math to SPIR-V Passes ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to convert standard dialect to SPIR-V dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/MathToSPIRV/MathToSPIRVPass.h"

#include "mlir/Conversion/MathToSPIRV/MathToSPIRV.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTMATHTOSPIRV
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {
/// A pass converting MLIR Math operations into the SPIR-V dialect.
class ConvertMathToSPIRVPass
    : public impl::ConvertMathToSPIRVBase<ConvertMathToSPIRVPass> {
  void runOnOperation() override;
};
} // namespace

void ConvertMathToSPIRVPass::runOnOperation() {
  MLIRContext *context = &getContext();
  Operation *op = getOperation();

  auto targetAttr = spirv::lookupTargetEnvOrDefault(op);
  std::unique_ptr<ConversionTarget> target =
      SPIRVConversionTarget::get(targetAttr);

  SPIRVTypeConverter typeConverter(targetAttr);

  // Use UnrealizedConversionCast as the bridge so that we don't need to pull
  // in patterns for other dialects.
  target->addLegalOp<UnrealizedConversionCastOp>();

  RewritePatternSet patterns(context);
  populateMathToSPIRVPatterns(typeConverter, patterns);

  if (failed(applyPartialConversion(op, *target, std::move(patterns))))
    return signalPassFailure();
}

std::unique_ptr<OperationPass<>> mlir::createConvertMathToSPIRVPass() {
  return std::make_unique<ConvertMathToSPIRVPass>();
}
