//===------------------ TestVulkanRunnerPipeline.cpp --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements a pipeline for use by mlir-vulkan-runner tests.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ConvertToSPIRV/ConvertToSPIRVPass.h"
#include "mlir/Conversion/GPUToSPIRV/GPUToSPIRVPass.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"

using namespace mlir;

namespace {

struct VulkanRunnerPipelineOptions
    : PassPipelineOptions<VulkanRunnerPipelineOptions> {
  Option<bool> spirvWebGPUPrepare{
      *this, "spirv-webgpu-prepare",
      llvm::cl::desc("Run MLIR transforms used when targetting WebGPU")};
};

void buildTestVulkanRunnerPipeline(OpPassManager &passManager,
                                   const VulkanRunnerPipelineOptions &options) {
  passManager.addPass(createGpuKernelOutliningPass());
  passManager.addPass(memref::createFoldMemRefAliasOpsPass());

  GpuSPIRVAttachTargetOptions attachTargetOptions{};
  attachTargetOptions.spirvVersion = "v1.0";
  attachTargetOptions.spirvCapabilities.push_back("Shader");
  attachTargetOptions.spirvExtensions.push_back(
      "SPV_KHR_storage_buffer_storage_class");
  passManager.addPass(createGpuSPIRVAttachTarget(attachTargetOptions));

  ConvertToSPIRVPassOptions convertToSPIRVOptions{};
  convertToSPIRVOptions.convertGPUModules = true;
  convertToSPIRVOptions.nestInGPUModule = true;
  passManager.addPass(createConvertToSPIRVPass(convertToSPIRVOptions));

  OpPassManager &spirvModulePM =
      passManager.nest<gpu::GPUModuleOp>().nest<spirv::ModuleOp>();
  spirvModulePM.addPass(spirv::createSPIRVLowerABIAttributesPass());
  spirvModulePM.addPass(spirv::createSPIRVUpdateVCEPass());
  if (options.spirvWebGPUPrepare)
    spirvModulePM.addPass(spirv::createSPIRVWebGPUPreparePass());

  passManager.addPass(createGpuModuleToBinaryPass());
}

} // namespace

namespace mlir::test {
void registerTestVulkanRunnerPipeline() {
  PassPipelineRegistration<VulkanRunnerPipelineOptions>(
      "test-vulkan-runner-pipeline",
      "Runs a series of passes for lowering GPU-dialect MLIR to "
      "SPIR-V-dialect MLIR intended for mlir-vulkan-runner.",
      buildTestVulkanRunnerPipeline);
}
} // namespace mlir::test
