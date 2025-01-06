//===------------------ TestSPIRVCPURunnerPipeline.cpp --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements a pipeline for use by SPIR-V CPU Runner tests.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/GPUToSPIRV/GPUToSPIRVPass.h"
#include "mlir/Conversion/SPIRVToLLVM/SPIRVToLLVMPass.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"

using namespace mlir;

namespace {

void buildTestSPIRVCPURunnerPipeline(OpPassManager &passManager) {
  passManager.addPass(createGpuKernelOutliningPass());
  passManager.addPass(createConvertGPUToSPIRVPass(/*mapMemorySpace=*/true));

  OpPassManager &nestedPM = passManager.nest<spirv::ModuleOp>();
  nestedPM.addPass(spirv::createSPIRVLowerABIAttributesPass());
  nestedPM.addPass(spirv::createSPIRVUpdateVCEPass());
  passManager.addPass(createLowerHostCodeToLLVMPass());
  passManager.addPass(createConvertSPIRVToLLVMPass());
}

} // namespace

namespace mlir {
namespace test {
void registerTestSPIRVCPURunnerPipeline() {
  PassPipelineRegistration<>(
      "test-spirv-cpu-runner-pipeline",
      "Runs a series of passes for lowering SPIR-V-dialect MLIR to "
      "LLVM-dialect MLIR intended for SPIR-V CPU Runner tests.",
      buildTestSPIRVCPURunnerPipeline);
}
} // namespace test
} // namespace mlir
