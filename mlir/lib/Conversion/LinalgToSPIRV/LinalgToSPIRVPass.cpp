//===- LinalgToSPIRVPass.cpp - Linalg to SPIR-V Passes --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LinalgToSPIRV/LinalgToSPIRVPass.h"
#include "../PassDetail.h"
#include "mlir/Conversion/LinalgToSPIRV/LinalgToSPIRV.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"

using namespace mlir;

namespace {
/// A pass converting MLIR Linalg ops into SPIR-V ops.
class LinalgToSPIRVPass : public ConvertLinalgToSPIRVBase<LinalgToSPIRVPass> {
  void runOnOperation() override;
};
} // namespace

void LinalgToSPIRVPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  auto targetAttr = spirv::lookupTargetEnvOrDefault(module);
  std::unique_ptr<ConversionTarget> target =
      spirv::SPIRVConversionTarget::get(targetAttr);

  SPIRVTypeConverter typeConverter(targetAttr);
  OwningRewritePatternList patterns;
  populateLinalgToSPIRVPatterns(context, typeConverter, patterns);
  populateBuiltinFuncToSPIRVPatterns(context, typeConverter, patterns);

  // Allow builtin ops.
  target->addLegalOp<ModuleOp, ModuleTerminatorOp>();
  target->addDynamicallyLegalOp<FuncOp>([&](FuncOp op) {
    return typeConverter.isSignatureLegal(op.getType()) &&
           typeConverter.isLegal(&op.getBody());
  });

  if (failed(applyFullConversion(module, *target, std::move(patterns))))
    return signalPassFailure();
}

std::unique_ptr<OperationPass<ModuleOp>> mlir::createLinalgToSPIRVPass() {
  return std::make_unique<LinalgToSPIRVPass>();
}
