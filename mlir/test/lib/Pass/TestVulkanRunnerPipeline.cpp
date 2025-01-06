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
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"

using namespace mlir;

namespace {

void buildTestVulkanRunnerPipeline(OpPassManager &passManager) {
  passManager.addPass(createGpuKernelOutliningPass());
  passManager.addPass(memref::createFoldMemRefAliasOpsPass());

  ConvertToSPIRVPassOptions convertToSPIRVOptions{};
  convertToSPIRVOptions.convertGPUModules = true;
  passManager.addPass(createConvertToSPIRVPass(convertToSPIRVOptions));
  OpPassManager &modulePM = passManager.nest<spirv::ModuleOp>();
  modulePM.addPass(spirv::createSPIRVLowerABIAttributesPass());
  modulePM.addPass(spirv::createSPIRVUpdateVCEPass());
}

} // namespace

namespace mlir::test {
void registerTestVulkanRunnerPipeline() {
  PassPipelineRegistration<>(
      "test-vulkan-runner-pipeline",
      "Runs a series of passes for lowering GPU-dialect MLIR to "
      "SPIR-V-dialect MLIR intended for mlir-vulkan-runner.",
      buildTestVulkanRunnerPipeline);
}
} // namespace mlir::test
