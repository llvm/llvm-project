//===- mlir-vulkan-runner.cpp - MLIR Vulkan Execution Driver --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a command line utility that executes an MLIR file on the Vulkan by
// translating MLIR GPU module to SPIR-V and host part to LLVM IR before
// JIT-compiling and executing the latter.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/GPUToSPIRV/GPUToSPIRVPass.h"
#include "mlir/Conversion/GPUToVulkan/ConvertGPUToVulkanPass.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/RequestCWrappers.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/Transforms/Passes.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/ExecutionEngine/JitRunner.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/TargetSelect.h"

using namespace mlir;

namespace {
struct VulkanRunnerOptions {
  llvm::cl::OptionCategory category{"mlir-vulkan-runner options"};
  llvm::cl::opt<bool> spirvWebGPUPrepare{
      "vulkan-runner-spirv-webgpu-prepare",
      llvm::cl::desc("Run MLIR transforms used when targetting WebGPU"),
      llvm::cl::cat(category)};
};
} // namespace

static LogicalResult runMLIRPasses(Operation *op,
                                   VulkanRunnerOptions &options) {
  auto module = dyn_cast<ModuleOp>(op);
  if (!module)
    return op->emitOpError("expected a 'builtin.module' op");
  PassManager passManager(module.getContext());
  applyPassManagerCLOptions(passManager);

  passManager.addPass(createGpuKernelOutliningPass());
  passManager.addPass(memref::createFoldMemRefAliasOpsPass());

  passManager.addPass(createConvertGPUToSPIRVPass(/*mapMemorySpace=*/true));
  OpPassManager &modulePM = passManager.nest<spirv::ModuleOp>();
  modulePM.addPass(spirv::createSPIRVLowerABIAttributesPass());
  modulePM.addPass(spirv::createSPIRVUpdateVCEPass());
  if (options.spirvWebGPUPrepare)
    modulePM.addPass(spirv::createSPIRVWebGPUPreparePass());

  passManager.addPass(createConvertGpuLaunchFuncToVulkanLaunchFuncPass());
  LowerToLLVMOptions llvmOptions(module.getContext(), DataLayout(module));
  passManager.addPass(createFinalizeMemRefToLLVMConversionPass());
  passManager.addPass(createConvertVectorToLLVMPass());
  passManager.nest<func::FuncOp>().addPass(LLVM::createRequestCWrappersPass());
  passManager.addPass(createConvertFuncToLLVMPass(llvmOptions));
  passManager.addPass(createReconcileUnrealizedCastsPass());
  passManager.addPass(createConvertVulkanLaunchFuncToVulkanCallsPass());

  return passManager.run(module);
}

int main(int argc, char **argv) {
  llvm::llvm_shutdown_obj x;
  registerPassManagerCLOptions();

  llvm::InitLLVM y(argc, argv);
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  // Initialize runner-specific CLI options. These will be parsed and
  // initialzied in `JitRunnerMain`.
  VulkanRunnerOptions options;
  auto runPassesWithOptions = [&options](Operation *op, JitRunnerOptions &) {
    return runMLIRPasses(op, options);
  };

  mlir::JitRunnerConfig jitRunnerConfig;
  jitRunnerConfig.mlirTransformer = runPassesWithOptions;

  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::LLVM::LLVMDialect,
                  mlir::gpu::GPUDialect, mlir::spirv::SPIRVDialect,
                  mlir::func::FuncDialect, mlir::memref::MemRefDialect,
                  mlir::vector::VectorDialect>();
  mlir::registerLLVMDialectTranslation(registry);

  return mlir::JitRunnerMain(argc, argv, registry, jitRunnerConfig);
}
