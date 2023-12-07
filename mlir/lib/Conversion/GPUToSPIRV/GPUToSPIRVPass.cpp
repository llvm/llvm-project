//===- GPUToSPIRVPass.cpp - GPU to SPIR-V Passes --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to convert a kernel function in the GPU Dialect
// into a spirv.module operation.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/GPUToSPIRV/GPUToSPIRVPass.h"

#include "mlir/Conversion/ArithToSPIRV/ArithToSPIRV.h"
#include "mlir/Conversion/FuncToSPIRV/FuncToSPIRV.h"
#include "mlir/Conversion/GPUToSPIRV/GPUToSPIRV.h"
#include "mlir/Conversion/MemRefToSPIRV/MemRefToSPIRV.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTGPUTOSPIRV
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {
/// Pass to lower GPU Dialect to SPIR-V. The pass only converts the gpu.func ops
/// inside gpu.module ops. i.e., the function that are referenced in
/// gpu.launch_func ops. For each such function
///
/// 1) Create a spirv::ModuleOp, and clone the function into spirv::ModuleOp
/// (the original function is still needed by the gpu::LaunchKernelOp, so cannot
/// replace it).
///
/// 2) Lower the body of the spirv::ModuleOp.
class GPUToSPIRVPass : public impl::ConvertGPUToSPIRVBase<GPUToSPIRVPass> {
public:
  explicit GPUToSPIRVPass(bool mapMemorySpace)
      : mapMemorySpace(mapMemorySpace) {}
  void runOnOperation() override;

private:
  bool mapMemorySpace;
};
} // namespace

void GPUToSPIRVPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  SmallVector<Operation *, 1> gpuModules;
  OpBuilder builder(context);
  module.walk([&](gpu::GPUModuleOp moduleOp) {
    // Clone each GPU kernel module for conversion, given that the GPU
    // launch op still needs the original GPU kernel module.
    builder.setInsertionPoint(moduleOp.getOperation());
    gpuModules.push_back(builder.clone(*moduleOp.getOperation()));
  });

  // Run conversion for each module independently as they can have different
  // TargetEnv attributes.
  for (Operation *gpuModule : gpuModules) {
    // Map MemRef memory space to SPIR-V storage class first if requested.
    if (mapMemorySpace) {
      std::unique_ptr<ConversionTarget> target =
          spirv::getMemorySpaceToStorageClassTarget(*context);
      spirv::MemorySpaceToStorageClassMap memorySpaceMap =
          spirv::mapMemorySpaceToVulkanStorageClass;
      spirv::MemorySpaceToStorageClassConverter converter(memorySpaceMap);

      RewritePatternSet patterns(context);
      spirv::populateMemorySpaceToStorageClassPatterns(converter, patterns);

      if (failed(applyFullConversion(gpuModule, *target, std::move(patterns))))
        return signalPassFailure();
    }

    auto targetAttr = spirv::lookupTargetEnvOrDefault(gpuModule);
    std::unique_ptr<ConversionTarget> target =
        SPIRVConversionTarget::get(targetAttr);

    SPIRVConversionOptions options;
    options.use64bitIndex = this->use64bitIndex;
    SPIRVTypeConverter typeConverter(targetAttr, options);
    typeConverter.addConversion([&](gpu::MMAMatrixType type) -> Type {
      return convertMMAToSPIRVType(type);
    });
    RewritePatternSet patterns(context);
    populateGPUToSPIRVPatterns(typeConverter, patterns);
    populateGpuWMMAToSPIRVConversionPatterns(typeConverter, patterns);
    // TODO: Change SPIR-V conversion to be progressive and remove the following
    // patterns.
    mlir::arith::populateArithToSPIRVPatterns(typeConverter, patterns);
    populateMemRefToSPIRVPatterns(typeConverter, patterns);
    populateFuncToSPIRVPatterns(typeConverter, patterns);

    if (failed(applyFullConversion(gpuModule, *target, std::move(patterns))))
      return signalPassFailure();
  }
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createConvertGPUToSPIRVPass(bool mapMemorySpace) {
  return std::make_unique<GPUToSPIRVPass>(mapMemorySpace);
}
