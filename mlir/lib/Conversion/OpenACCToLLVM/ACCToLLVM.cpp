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
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Dialect/OpenACC/Analysis/OpenACCSupport.h"
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

  SymbolTable symbolTable(module);
  LLVMTypeConverter converter(&getContext());
  RewritePatternSet patterns(&getContext());
  arith::populateArithToLLVMConversionPatterns(converter, patterns);
  cf::populateControlFlowToLLVMConversionPatterns(converter, patterns);
  populateFuncToLLVMConversionPatterns(converter, patterns);
  populateFinalizeMemRefToLLVMConversionPatterns(converter, patterns);

  acc::ACCRuntimeCallConfig runtimeConfig;

  populateACCExecutableDirectivePatterns(
      converter, patterns, module.getBodyRegion(), symbolTable, runtimeConfig);

  acc::OpenACCSupport &accSupport = getAnalysis<acc::OpenACCSupport>();
  populateACCDataDirectivePatterns(converter, patterns, accSupport,
                                   module.getBodyRegion(), symbolTable,
                                   runtimeConfig);
  populateACCAtomicPatterns(converter, patterns, accSupport);
  populateACCDataClauseOpPatterns(converter, patterns);

  LLVMConversionTarget target(getContext());
  configureACCExecutableDirectiveConversionLegality(target);
  configureACCDataDirectiveConversionLegality(target);
  configureACCAtomicConversionLegality(target);
  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}
