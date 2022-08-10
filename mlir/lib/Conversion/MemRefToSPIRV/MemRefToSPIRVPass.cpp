//===- MemRefToSPIRVPass.cpp - MemRef to SPIR-V Passes ----------------===//
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

#include "mlir/Conversion/MemRefToSPIRV/MemRefToSPIRVPass.h"
#include "../PassDetail.h"
#include "mlir/Conversion/MemRefToSPIRV/MemRefToSPIRV.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"

using namespace mlir;

namespace {
/// A pass converting MLIR MemRef operations into the SPIR-V dialect.
class ConvertMemRefToSPIRVPass
    : public ConvertMemRefToSPIRVBase<ConvertMemRefToSPIRVPass> {
  void runOnOperation() override;
};
} // namespace

void ConvertMemRefToSPIRVPass::runOnOperation() {
  MLIRContext *context = &getContext();
  Operation *op = getOperation();

  auto targetAttr = spirv::lookupTargetEnvOrDefault(op);
  std::unique_ptr<ConversionTarget> target =
      SPIRVConversionTarget::get(targetAttr);

  SPIRVTypeConverter::Options options;
  options.boolNumBits = this->boolNumBits;
  SPIRVTypeConverter typeConverter(targetAttr, options);

  // Use UnrealizedConversionCast as the bridge so that we don't need to pull in
  // patterns for other dialects.
  auto addUnrealizedCast = [](OpBuilder &builder, Type type, ValueRange inputs,
                              Location loc) {
    auto cast = builder.create<UnrealizedConversionCastOp>(loc, type, inputs);
    return Optional<Value>(cast.getResult(0));
  };
  typeConverter.addSourceMaterialization(addUnrealizedCast);
  typeConverter.addTargetMaterialization(addUnrealizedCast);
  target->addLegalOp<UnrealizedConversionCastOp>();

  RewritePatternSet patterns(context);
  populateMemRefToSPIRVPatterns(typeConverter, patterns);

  if (failed(applyPartialConversion(op, *target, std::move(patterns))))
    return signalPassFailure();
}

std::unique_ptr<OperationPass<>> mlir::createConvertMemRefToSPIRVPass() {
  return std::make_unique<ConvertMemRefToSPIRVPass>();
}
