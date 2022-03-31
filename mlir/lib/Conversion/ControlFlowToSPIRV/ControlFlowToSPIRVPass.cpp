//===- ControlFlowToSPIRVPass.cpp - ControlFlow to SPIR-V Pass ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to convert ControlFlow dialect to SPIR-V dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ControlFlowToSPIRV/ControlFlowToSPIRVPass.h"
#include "../PassDetail.h"
#include "mlir/Conversion/ControlFlowToSPIRV/ControlFlowToSPIRV.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"

using namespace mlir;

namespace {
/// A pass converting MLIR ControlFlow operations into the SPIR-V dialect.
class ConvertControlFlowToSPIRVPass
    : public ConvertControlFlowToSPIRVBase<ConvertControlFlowToSPIRVPass> {
  void runOnOperation() override;
};
} // namespace

void ConvertControlFlowToSPIRVPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  auto targetAttr = spirv::lookupTargetEnvOrDefault(module);
  std::unique_ptr<ConversionTarget> target =
      SPIRVConversionTarget::get(targetAttr);

  SPIRVTypeConverter::Options options;
  options.emulateNon32BitScalarTypes = this->emulateNon32BitScalarTypes;
  SPIRVTypeConverter typeConverter(targetAttr, options);

  RewritePatternSet patterns(context);
  cf::populateControlFlowToSPIRVPatterns(typeConverter, patterns);

  if (failed(applyPartialConversion(module, *target, std::move(patterns))))
    return signalPassFailure();
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createConvertControlFlowToSPIRVPass() {
  return std::make_unique<ConvertControlFlowToSPIRVPass>();
}
