//===- SPIRVToLLVMPass.cpp - SPIR-V to LLVM Passes ------------------------===//
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

#include "mlir/Conversion/SPIRVToLLVM/SPIRVToLLVMPass.h"

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/SPIRVToLLVM/SPIRVToLLVM.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTSPIRVTOLLVMPASS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {
/// A pass converting MLIR SPIR-V operations into LLVM dialect.
class ConvertSPIRVToLLVMPass
    : public impl::ConvertSPIRVToLLVMPassBase<ConvertSPIRVToLLVMPass> {
  void runOnOperation() override;

public:
  using Base::Base;
};
} // namespace

void ConvertSPIRVToLLVMPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();
  LLVMTypeConverter converter(&getContext());

  // Encode global variable's descriptor set and binding if they exist.
  encodeBindAttribute(module);

  RewritePatternSet patterns(context);

  populateSPIRVToLLVMTypeConversion(converter);

  populateSPIRVToLLVMModuleConversionPatterns(converter, patterns);
  populateSPIRVToLLVMConversionPatterns(converter, patterns);
  populateSPIRVToLLVMFunctionConversionPatterns(converter, patterns);

  ConversionTarget target(*context);
  target.addIllegalDialect<spirv::SPIRVDialect>();
  target.addLegalDialect<LLVM::LLVMDialect>();

  // Set `ModuleOp` as legal for `spirv.module` conversion.
  target.addLegalOp<ModuleOp>();
  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}
