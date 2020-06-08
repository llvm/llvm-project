//===- ConvertSPIRVToLLVMPass.cpp - Convert SPIR-V ops to LLVM ops --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to convert MLIR SPIR-V ops into LLVM ops
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/SPIRVToLLVM/ConvertSPIRVToLLVMPass.h"
#include "../PassDetail.h"
#include "mlir/Conversion/SPIRVToLLVM/ConvertSPIRVToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SPIRV/SPIRVDialect.h"

using namespace mlir;

namespace {
/// A pass converting MLIR SPIR-V operations into LLVM dialect.
class ConvertSPIRVToLLVMPass
    : public ConvertSPIRVToLLVMBase<ConvertSPIRVToLLVMPass> {
  void runOnOperation() override;
};
} // namespace

void ConvertSPIRVToLLVMPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();
  LLVMTypeConverter converter(&getContext());

  OwningRewritePatternList patterns;
  populateSPIRVToLLVMConversionPatterns(context, converter, patterns);

  // Currently pulls in Std to LLVM conversion patterns
  // that help with testing. This allows to convert
  // function arguments to LLVM.
  populateStdToLLVMConversionPatterns(converter, patterns);

  ConversionTarget target(getContext());
  target.addIllegalDialect<spirv::SPIRVDialect>();
  target.addLegalDialect<LLVM::LLVMDialect>();

  if (failed(applyPartialConversion(module, target, patterns, &converter)))
    signalPassFailure();
}

std::unique_ptr<OperationPass<ModuleOp>> mlir::createConvertSPIRVToLLVMPass() {
  return std::make_unique<ConvertSPIRVToLLVMPass>();
}
