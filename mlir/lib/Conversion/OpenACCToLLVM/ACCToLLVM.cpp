//===- ACCToLLVM.cpp - Convert OpenACC to LLVM dialect ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/OpenACCToLLVM/ACCToLLVM.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTACCTOLLVMPASS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {
struct ConvertACCToLLVMPass
    : public impl::ConvertACCToLLVMPassBase<ConvertACCToLLVMPass> {
  using Base::Base;

  void runOnOperation() override;
};
} // namespace

void ConvertACCToLLVMPass::runOnOperation() {
  ModuleOp module = getOperation();

  LLVMTypeConverter converter(&getContext());
  RewritePatternSet patterns(&getContext());
  arith::populateArithToLLVMConversionPatterns(converter, patterns);
  cf::populateControlFlowToLLVMConversionPatterns(converter, patterns);
  populateFuncToLLVMConversionPatterns(converter, patterns);

  // The device_type numbering is implementation-defined by the target
  // runtime. For now assume the same numbering as the OpenACC dialect.
  acc::ACCRuntimeCallConfig runtimeConfig;
  acc::populateDialectIdentityDeviceTypeMapping(runtimeConfig);
  populateACCExecutableDirectivePatterns(converter, patterns, runtimeConfig);

  LLVMConversionTarget target(getContext());
  configureACCExecutableDirectiveConversionLegality(target);
  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}
