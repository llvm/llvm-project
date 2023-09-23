//===- LowerGpuOpsToAMDGPUOps.cpp - MLIR GPU to AMD GPU lowering passes ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to generate AMDGPU operations for higher-level
// GPU operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/GPUToAMDGPU/GPUToAMDGPUPass.h"
#include "mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/AMDGPU/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTGPUOPSTOAMDGPUOPS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {
struct LowerGpuOpsToAMDGPUOpsPass
    : public impl::ConvertGpuOpsToAMDGPUOpsBase<LowerGpuOpsToAMDGPUOpsPass> {
  LowerGpuOpsToAMDGPUOpsPass() = default;
  LowerGpuOpsToAMDGPUOpsPass(const std::string &chipset, unsigned warpSize) {
    if (this->chipset.getNumOccurrences() == 0)
      this->chipset = chipset;
    if (this->warpSize.getNumOccurrences() == 0)
      this->warpSize = warpSize;
  }

  void runOnOperation() override {
    gpu::GPUModuleOp m = getOperation();
    MLIRContext *ctx = m.getContext();

    // Request C wrapper emission.
    for (auto func : m.getOps<func::FuncOp>()) {
      func->setAttr(LLVM::LLVMDialect::getEmitCWrapperAttrName(),
                    UnitAttr::get(ctx));
    }

    FailureOr<amdgpu::Chipset> maybeChipset = amdgpu::Chipset::parse(chipset);
    if (failed(maybeChipset)) {
      emitError(UnknownLoc::get(ctx), "Invalid chipset name: " + chipset);
      return signalPassFailure();
    }

    TypeConverter converter;

    RewritePatternSet amdgpuPatterns(ctx);

    populateGpuToAMDGPUConversionPatterns(converter, amdgpuPatterns,
                                          this->chipset, this->warpSize);
    ConversionTarget target(*ctx);
    // We do not mark GPU dialect illegal as other GPU ops and WMMA ops
    // unsupported by pattersn defined here are still allowed.
    target.addLegalDialect<amdgpu::AMDGPUDialect>();

    if (failed(applyPartialConversion(m, target, std::move(amdgpuPatterns))))
      signalPassFailure();
  }
};

} // namespace

void mlir::populateGpuToAMDGPUConversionPatterns(TypeConverter &converter,
                                                 RewritePatternSet &patterns,
                                                 StringRef chipset,
                                                 unsigned warpSize) {
  // Lowering for MMAMatrixType.
  converter.addConversion([&](gpu::MMAMatrixType type) -> Type {
    return amd::convertWMMAToROCDLLLVMType(type);
  });

  // We need to add target and source materializations so that the IR still
  // remains valid after the `gpu.mma_matrix` type conversion is done.
  auto buildUnrealizedCast = [](OpBuilder &builder, Type type,
                                ValueRange inputs, Location loc) {
    auto cast = builder.create<UnrealizedConversionCastOp>(loc, type, inputs);
    return std::optional<Value>(cast.getResult(0));
  };
  converter.addSourceMaterialization(buildUnrealizedCast);
  converter.addTargetMaterialization(buildUnrealizedCast);

  /// Collect a set of patterns to convert WMMA ops from GPU dialect to NVVM.
  populateGpuWMMAToAMDGPUConversionPatterns(converter, patterns, chipset,
                                            warpSize);
}

std::unique_ptr<OperationPass<gpu::GPUModuleOp>>
mlir::createLowerGpuOpsToAMDGPUOpsPass(const std::string &chipset,
                                       unsigned warpSize) {
  return std::make_unique<LowerGpuOpsToAMDGPUOpsPass>(chipset, warpSize);
}
