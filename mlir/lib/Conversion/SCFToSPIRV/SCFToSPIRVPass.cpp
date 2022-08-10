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
#include "mlir/Conversion/ArithmeticToSPIRV/ArithmeticToSPIRV.h"
#include "mlir/Conversion/FuncToSPIRV/FuncToSPIRV.h"
#include "mlir/Conversion/MemRefToSPIRV/MemRefToSPIRV.h"
#include "mlir/Conversion/SCFToSPIRV/SCFToSPIRV.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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
  Operation *op = getOperation();

  auto targetAttr = spirv::lookupTargetEnvOrDefault(op);
  std::unique_ptr<ConversionTarget> target =
      SPIRVConversionTarget::get(targetAttr);

  SPIRVTypeConverter typeConverter(targetAttr);
  ScfToSPIRVContext scfContext;
  RewritePatternSet patterns(context);
  populateSCFToSPIRVPatterns(typeConverter, scfContext, patterns);

  // TODO: Change SPIR-V conversion to be progressive and remove the following
  // patterns.
  mlir::arith::populateArithmeticToSPIRVPatterns(typeConverter, patterns);
  populateFuncToSPIRVPatterns(typeConverter, patterns);
  populateMemRefToSPIRVPatterns(typeConverter, patterns);
  populateBuiltinFuncToSPIRVPatterns(typeConverter, patterns);

  if (failed(applyPartialConversion(op, *target, std::move(patterns))))
    return signalPassFailure();
}

std::unique_ptr<OperationPass<>> mlir::createConvertSCFToSPIRVPass() {
  return std::make_unique<SCFToSPIRVPass>();
}
