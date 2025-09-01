//===------------------ TestVulkanRunnerPipeline.cpp --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements a pipeline for use by Vulkan runner tests.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Conversion/GPUToSPIRV/GPUToSPIRVPass.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/Transforms/RequestCWrappers.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"

using namespace mlir;

// Defined in the test directory, no public header.
namespace mlir::test {
std::unique_ptr<Pass> createTestConvertToSPIRVPass(bool convertGPUModules,
                                                   bool nestInGPUModule);
} // namespace mlir::test

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

  passManager.addPass(test::createTestConvertToSPIRVPass(
      /*convertGPUModules=*/true, /*nestInGPUModule=*/true));

  OpPassManager &spirvModulePM =
      passManager.nest<gpu::GPUModuleOp>().nest<spirv::ModuleOp>();
  spirvModulePM.addPass(spirv::createSPIRVLowerABIAttributesPass());
  spirvModulePM.addPass(spirv::createSPIRVUpdateVCEPass());
  if (options.spirvWebGPUPrepare)
    spirvModulePM.addPass(spirv::createSPIRVWebGPUPreparePass());

  passManager.addPass(createGpuModuleToBinaryPass());

  passManager.addPass(createFinalizeMemRefToLLVMConversionPass());
  passManager.nest<func::FuncOp>().addPass(
      LLVM::createLLVMRequestCWrappersPass());
  // VulkanRuntimeWrappers.cpp requires these calling convention options.
  GpuToLLVMConversionPassOptions opt;
  opt.hostBarePtrCallConv = false;
  opt.kernelBarePtrCallConv = true;
  opt.kernelIntersperseSizeCallConv = true;
  passManager.addPass(createGpuToLLVMConversionPass(opt));
}

} // namespace

namespace mlir::test {
void registerTestVulkanRunnerPipeline() {
  PassPipelineRegistration<VulkanRunnerPipelineOptions>(
      "test-vulkan-runner-pipeline",
      "Runs a series of passes intended for Vulkan runner tests. Lowers GPU "
      "dialect to LLVM dialect for the host and to serialized Vulkan SPIR-V "
      "for the device.",
      buildTestVulkanRunnerPipeline);
}
} // namespace mlir::test
