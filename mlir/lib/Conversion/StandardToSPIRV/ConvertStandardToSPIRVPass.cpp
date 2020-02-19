//===- ConvertStandardToSPIRVPass.cpp - Convert Std Ops to SPIR-V Ops -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to convert MLIR standard ops into the SPIR-V
// ops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/StandardToSPIRV/ConvertStandardToSPIRVPass.h"
#include "mlir/Conversion/StandardToSPIRV/ConvertStandardToSPIRV.h"
#include "mlir/Dialect/SPIRV/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/SPIRVLowering.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {
/// A pass converting MLIR Standard operations into the SPIR-V dialect.
class ConvertStandardToSPIRVPass
    : public ModulePass<ConvertStandardToSPIRVPass> {
  void runOnModule() override;
};
} // namespace

void ConvertStandardToSPIRVPass::runOnModule() {
  MLIRContext *context = &getContext();
  ModuleOp module = getModule();

  SPIRVTypeConverter typeConverter;
  OwningRewritePatternList patterns;
  populateStandardToSPIRVPatterns(context, typeConverter, patterns);
  populateBuiltinFuncToSPIRVPatterns(context, typeConverter, patterns);

  std::unique_ptr<ConversionTarget> target = spirv::SPIRVConversionTarget::get(
      spirv::lookupTargetEnvOrDefault(module), context);

  if (failed(applyPartialConversion(module, *target, patterns))) {
    return signalPassFailure();
  }
}

std::unique_ptr<OpPassBase<ModuleOp>> mlir::createConvertStandardToSPIRVPass() {
  return std::make_unique<ConvertStandardToSPIRVPass>();
}

static PassRegistration<ConvertStandardToSPIRVPass>
    pass("convert-std-to-spirv", "Convert Standard Ops to SPIR-V dialect");
