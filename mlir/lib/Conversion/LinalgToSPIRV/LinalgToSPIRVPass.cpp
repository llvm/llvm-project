//===- LinalgToSPIRVPass.cpp - Linalg to SPIR-V conversion pass -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LinalgToSPIRV/LinalgToSPIRVPass.h"
#include "mlir/Conversion/LinalgToSPIRV/LinalgToSPIRV.h"
#include "mlir/Dialect/SPIRV/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/SPIRVLowering.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {
/// A pass converting MLIR Linalg ops into SPIR-V ops.
class LinalgToSPIRVPass : public ModulePass<LinalgToSPIRVPass> {
  void runOnModule() override;
};
} // namespace

void LinalgToSPIRVPass::runOnModule() {
  MLIRContext *context = &getContext();
  ModuleOp module = getModule();

  SPIRVTypeConverter typeConverter;
  OwningRewritePatternList patterns;
  populateLinalgToSPIRVPatterns(context, typeConverter, patterns);
  populateBuiltinFuncToSPIRVPatterns(context, typeConverter, patterns);

  auto targetEnv = spirv::lookupTargetEnvOrDefault(module);
  std::unique_ptr<ConversionTarget> target =
      spirv::SPIRVConversionTarget::get(targetEnv, context);

  // Allow builtin ops.
  target->addLegalOp<ModuleOp, ModuleTerminatorOp>();
  target->addDynamicallyLegalOp<FuncOp>(
      [&](FuncOp op) { return typeConverter.isSignatureLegal(op.getType()); });

  if (failed(applyFullConversion(module, *target, patterns)))
    return signalPassFailure();
}

std::unique_ptr<OpPassBase<ModuleOp>> mlir::createLinalgToSPIRVPass() {
  return std::make_unique<LinalgToSPIRVPass>();
}

static PassRegistration<LinalgToSPIRVPass>
    pass("convert-linalg-to-spirv", "Convert Linalg ops to SPIR-V ops");
