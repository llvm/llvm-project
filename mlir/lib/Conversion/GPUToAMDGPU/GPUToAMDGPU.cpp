//===- GPUToAMDGPU.cpp - GPU to AMDGPU dialect conversion -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/GPUToAMDGPU/GPUToAMDGPU.h"
#include "../PassDetail.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/AMDGPU/AMDGPUDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"

using namespace mlir;

namespace {
struct ConvertGPUToAMDGPUPass
    : public ConvertGPUToAMDGPUBase<ConvertGPUToAMDGPUPass> {
  ConvertGPUToAMDGPUPass() = default;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    LLVMTypeConverter converter(&getContext());
    populateGPUToAMDGPUConversionPatterns(converter, patterns);
    LLVMConversionTarget target(getContext());
    target.addLegalDialect<::mlir::LLVM::LLVMDialect>();
    target.addLegalDialect<::mlir::AMDGPU::AMDGPUDialect>();
    target.addLegalDialect<::mlir::ROCDL::ROCDLDialect>();
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

void mlir::populateGPUToAMDGPUConversionPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns) {
}

std::unique_ptr<Pass> mlir::createConvertGPUToAMDGPUPass() {
  return std::make_unique<ConvertGPUToAMDGPUPass>();
}