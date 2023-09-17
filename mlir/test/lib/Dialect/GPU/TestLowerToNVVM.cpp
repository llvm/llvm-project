//===- TestLowerToNVVM.cpp - Test lowering to NVVM as a sink pass ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass for testing the lowering to NVVM as a generally
// usable sink pass.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/NVGPUToNVVM/NVGPUToNVVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVMPass.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

#if MLIR_CUDA_CONVERSIONS_ENABLED
namespace {
struct TestLowerToNVVMOptions
    : public PassPipelineOptions<TestLowerToNVVMOptions> {
  PassOptions::Option<int64_t> hostIndexBitWidth{
      *this, "host-index-bitwidth",
      llvm::cl::desc("Bitwidth of the index type for the host (warning this "
                     "should be 64 until the GPU layering is fixed)"),
      llvm::cl::init(64)};
  PassOptions::Option<bool> hostUseBarePtrCallConv{
      *this, "host-bare-ptr-calling-convention",
      llvm::cl::desc(
          "Whether to use the bareptr calling convention on the host (warning "
          "this should be false until the GPU layering is fixed)"),
      llvm::cl::init(false)};
  PassOptions::Option<int64_t> kernelIndexBitWidth{
      *this, "kernel-index-bitwidth",
      llvm::cl::desc("Bitwidth of the index type for the GPU kernels"),
      llvm::cl::init(64)};
  PassOptions::Option<bool> kernelUseBarePtrCallConv{
      *this, "kernel-bare-ptr-calling-convention",
      llvm::cl::desc(
          "Whether to use the bareptr calling convention on the kernel "
          "(warning this should be false until the GPU layering is fixed)"),
      llvm::cl::init(false)};
  PassOptions::Option<std::string> cubinTriple{
      *this, "cubin-triple",
      llvm::cl::desc("Triple to use to serialize to cubin."),
      llvm::cl::init("nvptx64-nvidia-cuda")};
  PassOptions::Option<std::string> cubinChip{
      *this, "cubin-chip", llvm::cl::desc("Chip to use to serialize to cubin."),
      llvm::cl::init("sm_50")};
  PassOptions::Option<std::string> cubinFeatures{
      *this, "cubin-features",
      llvm::cl::desc("Features to use to serialize to cubin."),
      llvm::cl::init("+ptx60")};
  PassOptions::Option<std::string> cubinFormat{
      *this, "cubin-format",
      llvm::cl::desc("Compilation format to use to serialize to cubin."),
      llvm::cl::init("isa")};
};

//===----------------------------------------------------------------------===//
// GPUModule-specific stuff.
//===----------------------------------------------------------------------===//
void buildGpuPassPipeline(OpPassManager &pm,
                          const TestLowerToNVVMOptions &options) {
  pm.addNestedPass<gpu::GPUModuleOp>(createStripDebugInfoPass());

  pm.addNestedPass<gpu::GPUModuleOp>(createConvertVectorToSCFPass());
  // Convert SCF to CF (always needed).
  pm.addNestedPass<gpu::GPUModuleOp>(createConvertSCFToCFPass());
  // Convert Math to LLVM (always needed).
  pm.addNestedPass<gpu::GPUModuleOp>(createConvertMathToLLVMPass());
  // Expand complicated MemRef operations before lowering them.
  pm.addNestedPass<gpu::GPUModuleOp>(memref::createExpandStridedMetadataPass());
  // The expansion may create affine expressions. Get rid of them.
  pm.addNestedPass<gpu::GPUModuleOp>(createLowerAffinePass());

  // Convert MemRef to LLVM (always needed).
  // TODO: C++20 designated initializers.
  FinalizeMemRefToLLVMConversionPassOptions
      finalizeMemRefToLLVMConversionPassOptions;
  // Must be 64b on the host, things don't compose properly around
  // gpu::LaunchOp and gpu::HostRegisterOp.
  // TODO: fix GPU layering.
  finalizeMemRefToLLVMConversionPassOptions.indexBitwidth =
      options.kernelIndexBitWidth;
  finalizeMemRefToLLVMConversionPassOptions.useOpaquePointers = true;
  pm.addNestedPass<gpu::GPUModuleOp>(createFinalizeMemRefToLLVMConversionPass(
      finalizeMemRefToLLVMConversionPassOptions));

  // Convert Func to LLVM (always needed).
  // TODO: C++20 designated initializers.
  ConvertFuncToLLVMPassOptions convertFuncToLLVMPassOptions;
  // Must be 64b on the host, things don't compose properly around
  // gpu::LaunchOp and gpu::HostRegisterOp.
  // TODO: fix GPU layering.
  convertFuncToLLVMPassOptions.indexBitwidth = options.kernelIndexBitWidth;
  convertFuncToLLVMPassOptions.useBarePtrCallConv =
      options.kernelUseBarePtrCallConv;
  convertFuncToLLVMPassOptions.useOpaquePointers = true;
  pm.addNestedPass<gpu::GPUModuleOp>(
      createConvertFuncToLLVMPass(convertFuncToLLVMPassOptions));

  // TODO: C++20 designated initializers.
  ConvertIndexToLLVMPassOptions convertIndexToLLVMPassOpt;
  // Must be 64b on the host, things don't compose properly around
  // gpu::LaunchOp and gpu::HostRegisterOp.
  // TODO: fix GPU layering.
  convertIndexToLLVMPassOpt.indexBitwidth = options.kernelIndexBitWidth;
  pm.addNestedPass<gpu::GPUModuleOp>(
      createConvertIndexToLLVMPass(convertIndexToLLVMPassOpt));

  // TODO: C++20 designated initializers.
  // The following pass is inconsistent.
  // TODO: fix inconsistence.
  ConvertGpuOpsToNVVMOpsOptions convertGpuOpsToNVVMOpsOptions;
  convertGpuOpsToNVVMOpsOptions.useBarePtrCallConv =
      options.kernelUseBarePtrCallConv;
  convertGpuOpsToNVVMOpsOptions.indexBitwidth = options.kernelIndexBitWidth;
  convertGpuOpsToNVVMOpsOptions.useOpaquePointers = true;
  pm.addNestedPass<gpu::GPUModuleOp>(
      createConvertGpuOpsToNVVMOps(convertGpuOpsToNVVMOpsOptions));

  // TODO: C++20 designated initializers.
  ConvertNVGPUToNVVMPassOptions convertNVGPUToNVVMPassOptions;
  convertNVGPUToNVVMPassOptions.useOpaquePointers = true;
  pm.addNestedPass<gpu::GPUModuleOp>(
      createConvertNVGPUToNVVMPass(convertNVGPUToNVVMPassOptions));
  pm.addNestedPass<gpu::GPUModuleOp>(createConvertSCFToCFPass());

  // Convert vector to LLVM (always needed).
  // TODO: C++20 designated initializers.
  ConvertVectorToLLVMPassOptions convertVectorToLLVMPassOptions;
  convertVectorToLLVMPassOptions.reassociateFPReductions = true;
  pm.addNestedPass<gpu::GPUModuleOp>(
      createConvertVectorToLLVMPass(convertVectorToLLVMPassOptions));

  // Sprinkle some cleanups.
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // Finally we can reconcile unrealized casts.
  pm.addNestedPass<gpu::GPUModuleOp>(createReconcileUnrealizedCastsPass());
}

void buildLowerToNVVMPassPipeline(OpPassManager &pm,
                                  const TestLowerToNVVMOptions &options) {
  //===----------------------------------------------------------------------===//
  // Host-specific stuff.
  //===----------------------------------------------------------------------===//
  // Important, must be run at the top-level.
  pm.addPass(createGpuKernelOutliningPass());

  // Important, all host passes must be run at the func level so that host
  // conversions can remain with 64 bit indices without polluting the GPU
  // kernel that may have 32 bit indices.
  // Must be 64b on the host, things don't compose properly around
  // gpu::LaunchOp and gpu::HostRegisterOp.
  // TODO: fix GPU layering.
  pm.addNestedPass<func::FuncOp>(createConvertVectorToSCFPass());
  // Convert SCF to CF (always needed).
  pm.addNestedPass<func::FuncOp>(createConvertSCFToCFPass());
  // Convert Math to LLVM (always needed).
  pm.addNestedPass<func::FuncOp>(createConvertMathToLLVMPass());
  // Expand complicated MemRef operations before lowering them.
  pm.addNestedPass<func::FuncOp>(memref::createExpandStridedMetadataPass());
  // The expansion may create affine expressions. Get rid of them.
  pm.addNestedPass<func::FuncOp>(createLowerAffinePass());

  // Convert MemRef to LLVM (always needed).
  // TODO: C++20 designated initializers.
  FinalizeMemRefToLLVMConversionPassOptions
      finalizeMemRefToLLVMConversionPassOptions;
  finalizeMemRefToLLVMConversionPassOptions.useAlignedAlloc = true;
  // Must be 64b on the host, things don't compose properly around
  // gpu::LaunchOp and gpu::HostRegisterOp.
  // TODO: fix GPU layering.
  finalizeMemRefToLLVMConversionPassOptions.indexBitwidth =
      options.hostIndexBitWidth;
  finalizeMemRefToLLVMConversionPassOptions.useOpaquePointers = true;
  pm.addNestedPass<func::FuncOp>(createFinalizeMemRefToLLVMConversionPass(
      finalizeMemRefToLLVMConversionPassOptions));

  // Convert Func to LLVM (always needed).
  // TODO: C++20 designated initializers.
  ConvertFuncToLLVMPassOptions convertFuncToLLVMPassOptions;
  // Must be 64b on the host, things don't compose properly around
  // gpu::LaunchOp and gpu::HostRegisterOp.
  // TODO: fix GPU layering.
  convertFuncToLLVMPassOptions.indexBitwidth = options.hostIndexBitWidth;
  convertFuncToLLVMPassOptions.useBarePtrCallConv =
      options.hostUseBarePtrCallConv;
  convertFuncToLLVMPassOptions.useOpaquePointers = true;
  pm.addNestedPass<func::FuncOp>(
      createConvertFuncToLLVMPass(convertFuncToLLVMPassOptions));

  // TODO: C++20 designated initializers.
  ConvertIndexToLLVMPassOptions convertIndexToLLVMPassOpt;
  // Must be 64b on the host, things don't compose properly around
  // gpu::LaunchOp and gpu::HostRegisterOp.
  // TODO: fix GPU layering.
  convertIndexToLLVMPassOpt.indexBitwidth = options.hostIndexBitWidth;
  pm.addNestedPass<func::FuncOp>(
      createConvertIndexToLLVMPass(convertIndexToLLVMPassOpt));

  pm.addNestedPass<func::FuncOp>(createArithToLLVMConversionPass());

  // Sprinkle some cleanups.
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(createCSEPass());

  //===----------------------------------------------------------------------===//
  // GPUModule-specific stuff.
  //===----------------------------------------------------------------------===//
  buildGpuPassPipeline(pm, options);

  //===----------------------------------------------------------------------===//
  // Host post-GPUModule-specific stuff.
  //===----------------------------------------------------------------------===//
  // Attach an NVVM target to all the GPU modules with the provided target
  // options.
  // TODO: C++20 designated initializers.
  GpuNVVMAttachTargetOptions nvvmTargetOptions;
  nvvmTargetOptions.triple = options.cubinTriple;
  nvvmTargetOptions.chip = options.cubinChip;
  nvvmTargetOptions.features = options.cubinFeatures;
  pm.addPass(createGpuNVVMAttachTarget(nvvmTargetOptions));

  // Convert GPU to LLVM.
  // TODO: C++20 designated initializers.
  GpuToLLVMConversionPassOptions gpuToLLVMConversionOptions;
  // Note: hostBarePtrCallConv must be false for now otherwise
  // gpu::HostRegister is ill-defined: it wants unranked memrefs but can't
  // lower the to bare ptr.
  gpuToLLVMConversionOptions.hostBarePtrCallConv =
      options.hostUseBarePtrCallConv;
  gpuToLLVMConversionOptions.kernelBarePtrCallConv =
      options.kernelUseBarePtrCallConv;
  gpuToLLVMConversionOptions.useOpaquePointers = true;

  // TODO: something useful here.
  // gpuToLLVMConversionOptions.gpuBinaryAnnotation = "";
  pm.addPass(createGpuToLLVMConversionPass(gpuToLLVMConversionOptions));

  // Serialize all GPU modules to binaries.
  GpuModuleToBinaryPassOptions gpuModuleToBinaryPassOptions;
  gpuModuleToBinaryPassOptions.compilationTarget = options.cubinFormat;
  pm.addPass(createGpuModuleToBinaryPass(gpuModuleToBinaryPassOptions));

  // Convert vector to LLVM (always needed).
  // TODO: C++20 designated initializers.
  ConvertVectorToLLVMPassOptions convertVectorToLLVMPassOptions;
  convertVectorToLLVMPassOptions.reassociateFPReductions = true;
  pm.addNestedPass<func::FuncOp>(
      createConvertVectorToLLVMPass(convertVectorToLLVMPassOptions));

  ConvertIndexToLLVMPassOptions convertIndexToLLVMPassOpt3;
  // Must be 64b on the host, things don't compose properly around
  // gpu::LaunchOp and gpu::HostRegisterOp.
  // TODO: fix GPU layering.
  convertIndexToLLVMPassOpt3.indexBitwidth = options.hostIndexBitWidth;
  pm.addPass(createConvertIndexToLLVMPass(convertIndexToLLVMPassOpt3));

  // Convert Func to LLVM (always needed).
  // TODO: C++20 designated initializers.
  ConvertFuncToLLVMPassOptions convertFuncToLLVMPassOptions2;
  // Must be 64b on the host, things don't compose properly around
  // gpu::LaunchOp and gpu::HostRegisterOp.
  convertFuncToLLVMPassOptions2.indexBitwidth = options.hostIndexBitWidth;
  convertFuncToLLVMPassOptions2.useBarePtrCallConv =
      options.hostUseBarePtrCallConv;
  convertFuncToLLVMPassOptions2.useOpaquePointers = true;
  pm.addPass(createConvertFuncToLLVMPass(convertFuncToLLVMPassOptions2));

  // Sprinkle some cleanups.
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // Finally we can reconcile unrealized casts.
  pm.addPass(createReconcileUnrealizedCastsPass());
}
} // namespace

namespace mlir {
namespace test {
void registerTestLowerToNVVM() {
  PassPipelineRegistration<TestLowerToNVVMOptions>(
      "test-lower-to-nvvm",
      "An example of pipeline to lower the main dialects (arith, linalg, "
      "memref, scf, vector) down to NVVM.",
      buildLowerToNVVMPassPipeline);
}
} // namespace test
} // namespace mlir
#endif // MLIR_CUDA_CONVERSIONS_ENABLED
