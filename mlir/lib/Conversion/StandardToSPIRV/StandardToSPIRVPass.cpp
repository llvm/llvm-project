//===- StandardToSPIRVPass.cpp - Standard to SPIR-V Passes ----------------===//
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

#include "mlir/Conversion/StandardToSPIRV/StandardToSPIRVPass.h"
#include "../PassDetail.h"
#include "mlir/Conversion/StandardToSPIRV/StandardToSPIRV.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"

using namespace mlir;

namespace {
/// A pass converting MLIR Standard operations into the SPIR-V dialect.
class ConvertStandardToSPIRVPass
    : public ConvertStandardToSPIRVBase<ConvertStandardToSPIRVPass> {
  void runOnOperation() override;
};
} // namespace

void ConvertStandardToSPIRVPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  auto targetAttr = spirv::lookupTargetEnvOrDefault(module);
  std::unique_ptr<ConversionTarget> target =
      spirv::SPIRVConversionTarget::get(targetAttr);

  SPIRVTypeConverter typeConverter(targetAttr);
  RewritePatternSet patterns(context);
  populateStandardToSPIRVPatterns(typeConverter, patterns);
  populateTensorToSPIRVPatterns(typeConverter,
                                /*byteCountThreshold=*/64, patterns);
  populateBuiltinFuncToSPIRVPatterns(typeConverter, patterns);

  if (failed(applyPartialConversion(module, *target, std::move(patterns))))
    return signalPassFailure();
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createConvertStandardToSPIRVPass() {
  return std::make_unique<ConvertStandardToSPIRVPass>();
}
