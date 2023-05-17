//===- ComplexToSPIRVPass.cpp - Complex to SPIR-V Passes ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to convert Complex dialect to SPIR-V dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ComplexToSPIRV/ComplexToSPIRVPass.h"

#include "mlir/Conversion/ComplexToSPIRV/ComplexToSPIRV.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTCOMPLEXTOSPIRVPASS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {
/// A pass converting MLIR Complex operations into the SPIR-V dialect.
class ConvertComplexToSPIRVPass
    : public impl::ConvertComplexToSPIRVPassBase<ConvertComplexToSPIRVPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    Operation *op = getOperation();

    auto targetAttr = spirv::lookupTargetEnvOrDefault(op);
    std::unique_ptr<ConversionTarget> target =
        SPIRVConversionTarget::get(targetAttr);

    SPIRVConversionOptions options;
    SPIRVTypeConverter typeConverter(targetAttr, options);

    // Use UnrealizedConversionCast as the bridge so that we don't need to pull
    // in patterns for other dialects.
    auto addUnrealizedCast = [](OpBuilder &builder, Type type,
                                ValueRange inputs, Location loc) {
      auto cast = builder.create<UnrealizedConversionCastOp>(loc, type, inputs);
      return std::optional<Value>(cast.getResult(0));
    };
    typeConverter.addSourceMaterialization(addUnrealizedCast);
    typeConverter.addTargetMaterialization(addUnrealizedCast);
    target->addLegalOp<UnrealizedConversionCastOp>();

    RewritePatternSet patterns(context);
    populateComplexToSPIRVPatterns(typeConverter, patterns);

    if (failed(applyPartialConversion(op, *target, std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace
