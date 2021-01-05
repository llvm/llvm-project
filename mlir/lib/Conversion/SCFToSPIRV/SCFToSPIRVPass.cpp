//===- SCFToSPIRVPass.cpp - SCF to SPIR-V Passes --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to convert SCF dialect into SPIR-V dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/SCFToSPIRV/SCFToSPIRVPass.h"

#include "../PassDetail.h"
#include "mlir/Conversion/SCFToSPIRV/SCFToSPIRV.h"
#include "mlir/Conversion/StandardToSPIRV/StandardToSPIRV.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"

using namespace mlir;

namespace {
struct SCFToSPIRVPass : public SCFToSPIRVBase<SCFToSPIRVPass> {
  void runOnOperation() override;
};
} // namespace

void SCFToSPIRVPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  auto targetAttr = spirv::lookupTargetEnvOrDefault(module);
  std::unique_ptr<ConversionTarget> target =
      spirv::SPIRVConversionTarget::get(targetAttr);

  SPIRVTypeConverter typeConverter(targetAttr);
  ScfToSPIRVContext scfContext;
  OwningRewritePatternList patterns;
  populateSCFToSPIRVPatterns(context, typeConverter, scfContext, patterns);
  populateStandardToSPIRVPatterns(context, typeConverter, patterns);
  populateBuiltinFuncToSPIRVPatterns(context, typeConverter, patterns);

  if (failed(applyPartialConversion(module, *target, std::move(patterns))))
    return signalPassFailure();
}

std::unique_ptr<OperationPass<ModuleOp>> mlir::createConvertSCFToSPIRVPass() {
  return std::make_unique<SCFToSPIRVPass>();
}
