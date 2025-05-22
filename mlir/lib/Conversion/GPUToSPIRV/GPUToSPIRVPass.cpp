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
#include "mlir/Conversion/SCFToSPIRV/SCFToSPIRV.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"
#include "mlir/IR/PatternMatch.h"

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
struct GPUToSPIRVPass final : impl::ConvertGPUToSPIRVBase<GPUToSPIRVPass> {
  explicit GPUToSPIRVPass(bool mapMemorySpace)
      : mapMemorySpace(mapMemorySpace) {}
  void runOnOperation() override;

private:
  bool mapMemorySpace;
};

void GPUToSPIRVPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  SmallVector<Operation *, 1> gpuModules;
  OpBuilder builder(context);

  auto targetEnvSupportsKernelCapability = [](gpu::GPUModuleOp moduleOp) {
    Operation *gpuModule = moduleOp.getOperation();
    auto targetAttr = spirv::lookupTargetEnvOrDefault(gpuModule);
    spirv::TargetEnv targetEnv(targetAttr);
    return targetEnv.allows(spirv::Capability::Kernel);
  };

  module.walk([&](gpu::GPUModuleOp moduleOp) {
    // Clone each GPU kernel module for conversion, given that the GPU
    // launch op still needs the original GPU kernel module.
    // For Vulkan Shader capabilities, we insert the newly converted SPIR-V
    // module right after the original GPU module, as that's the expectation of
    // the in-tree Vulkan runner.
    // For OpenCL Kernel capabilities, we insert the newly converted SPIR-V
    // module inside the original GPU module, as that's the expectaion of the
    // normal GPU compilation pipeline.
    if (targetEnvSupportsKernelCapability(moduleOp)) {
      builder.setInsertionPoint(moduleOp.getBody(),
                                moduleOp.getBody()->begin());
    } else {
      builder.setInsertionPoint(moduleOp.getOperation());
    }
    gpuModules.push_back(builder.clone(*moduleOp.getOperation()));
  });

  // Run conversion for each module independently as they can have different
  // TargetEnv attributes.
  for (Operation *gpuModule : gpuModules) {
    spirv::TargetEnvAttr targetAttr =
        spirv::lookupTargetEnvOrDefault(gpuModule);

    // Map MemRef memory space to SPIR-V storage class first if requested.
    if (mapMemorySpace) {
      spirv::MemorySpaceToStorageClassMap memorySpaceMap =
          targetEnvSupportsKernelCapability(
              dyn_cast<gpu::GPUModuleOp>(gpuModule))
              ? spirv::mapMemorySpaceToOpenCLStorageClass
              : spirv::mapMemorySpaceToVulkanStorageClass;
      spirv::MemorySpaceToStorageClassConverter converter(memorySpaceMap);
      spirv::convertMemRefTypesAndAttrs(gpuModule, converter);

      // Check if there are any illegal ops remaining.
      std::unique_ptr<ConversionTarget> target =
          spirv::getMemorySpaceToStorageClassTarget(*context);
      gpuModule->walk([&target, this](Operation *childOp) {
        if (target->isIllegal(childOp)) {
          childOp->emitOpError("failed to legalize memory space");
          signalPassFailure();
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
    }

    std::unique_ptr<ConversionTarget> target =
        SPIRVConversionTarget::get(targetAttr);

    SPIRVConversionOptions options;
    options.use64bitIndex = this->use64bitIndex;
    SPIRVTypeConverter typeConverter(targetAttr, options);
    populateMMAToSPIRVCoopMatrixTypeConversion(typeConverter);

    RewritePatternSet patterns(context);
    populateGPUToSPIRVPatterns(typeConverter, patterns);
    populateGpuWMMAToSPIRVCoopMatrixKHRConversionPatterns(typeConverter,
                                                          patterns);

    // TODO: Change SPIR-V conversion to be progressive and remove the following
    // patterns.
    ScfToSPIRVContext scfContext;
    populateSCFToSPIRVPatterns(typeConverter, scfContext, patterns);
    mlir::arith::populateArithToSPIRVPatterns(typeConverter, patterns);
    populateMemRefToSPIRVPatterns(typeConverter, patterns);
    populateFuncToSPIRVPatterns(typeConverter, patterns);

    if (failed(applyFullConversion(gpuModule, *target, std::move(patterns))))
      return signalPassFailure();
  }

  // For OpenCL, the gpu.func op in the original gpu.module op needs to be
  // replaced with an empty func.func op with the same arguments as the gpu.func
  // op. The func.func op needs gpu.kernel attribute set.
  module.walk([&](gpu::GPUModuleOp moduleOp) {
    if (targetEnvSupportsKernelCapability(moduleOp)) {
      moduleOp.walk([&](gpu::GPUFuncOp funcOp) {
        builder.setInsertionPoint(funcOp);
        auto newFuncOp = builder.create<func::FuncOp>(
            funcOp.getLoc(), funcOp.getName(), funcOp.getFunctionType());
        auto entryBlock = newFuncOp.addEntryBlock();
        builder.setInsertionPointToEnd(entryBlock);
        builder.create<func::ReturnOp>(funcOp.getLoc());
        newFuncOp->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                           builder.getUnitAttr());
        funcOp.erase();
      });
    }
  });
}

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createConvertGPUToSPIRVPass(bool mapMemorySpace) {
  return std::make_unique<GPUToSPIRVPass>(mapMemorySpace);
}
