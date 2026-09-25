//===- GPUToROCDLPipeline.cpp - Lowering pipeline to ROCDL/AMDGPU --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a sink pipeline that lowers a payload containing
// `gpu.launch` / `gpu.module` ops to AMDGPU/ROCDL and emits an AMDGCN binary
// blob via `gpu-module-to-binary`. It is the AMD counterpart of
// `gpu-lower-to-nvvm-pipeline` and `gpu-lower-to-xevm-pipeline`.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/AMDGPUToROCDL/AMDGPUToROCDL.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h"
#include "mlir/Conversion/GPUToROCDL/Runtimes.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Pipelines/Passes.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Common pipeline
//===----------------------------------------------------------------------===//
void buildCommonPassPipeline(
    OpPassManager &pm, const mlir::gpu::GPUToROCDLPipelineOptions &options) {
  // Lower AMDGPU dialect ops (e.g. amdgpu.lds_barrier, amdgpu.dpp,
  // amdgpu.mfma, amdgpu.dot, ...) to ROCDL intrinsics first, while they may
  // still live in unout-lined `gpu.launch` bodies. Mirrors the way NVVM's
  // pipeline runs `convert-nvgpu-to-nvvm` before kernel outlining.
  ConvertAMDGPUToROCDLPassOptions amdgpuToROCDLOpt;
  amdgpuToROCDLOpt.chipset = options.chip;
  pm.addPass(createConvertAMDGPUToROCDLPass(amdgpuToROCDLOpt));

  pm.addPass(createGpuKernelOutliningPass());
  pm.addPass(createConvertVectorToSCFPass());
  pm.addPass(createSCFToControlFlowPass());
  pm.addPass(createConvertFuncToLLVMPass());
  pm.addPass(memref::createExpandStridedMetadataPass());

  GpuROCDLAttachTargetOptions rocdlTargetOptions;
  rocdlTargetOptions.triple = options.triple;
  rocdlTargetOptions.chip = options.chip;
  rocdlTargetOptions.features = options.features;
  rocdlTargetOptions.abiVersion = options.abiVersion;
  rocdlTargetOptions.optLevel = options.optLevel;
  rocdlTargetOptions.wave64Flag = options.wave64;
  pm.addPass(createGpuROCDLAttachTarget(rocdlTargetOptions));

  pm.addPass(createLowerAffinePass());
  pm.addPass(createArithToLLVMConversionPass());
  ConvertIndexToLLVMPassOptions convertIndexToLLVMPassOpt;
  convertIndexToLLVMPassOpt.indexBitwidth = options.indexBitWidth;
  pm.addPass(createConvertIndexToLLVMPass(convertIndexToLLVMPassOpt));
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
}

//===----------------------------------------------------------------------===//
// GPUModule-specific stuff.
//===----------------------------------------------------------------------===//
void buildGpuPassPipeline(OpPassManager &pm,
                          const mlir::gpu::GPUToROCDLPipelineOptions &options) {
  ConvertGpuOpsToROCDLOpsOptions opt;
  opt.chipset = options.chip;
  opt.useBarePtrCallConv = options.kernelUseBarePtrCallConv;
  opt.indexBitwidth = options.indexBitWidth;
  // Always declare HIP as the runtime so that gpu.printf etc. lower to the
  // matching runtime entry points exposed by `libmlir_rocm_runtime.so`.
  opt.runtime = mlir::gpu::amd::Runtime::HIP;
  pm.addNestedPass<gpu::GPUModuleOp>(createConvertGpuOpsToROCDLOps(opt));
  pm.addNestedPass<gpu::GPUModuleOp>(createCanonicalizerPass());
  pm.addNestedPass<gpu::GPUModuleOp>(createCSEPass());
  pm.addNestedPass<gpu::GPUModuleOp>(createReconcileUnrealizedCastsPass());
}

//===----------------------------------------------------------------------===//
// Host Post-GPU pipeline
//===----------------------------------------------------------------------===//
void buildHostPostPipeline(
    OpPassManager &pm, const mlir::gpu::GPUToROCDLPipelineOptions &options) {
  GpuToLLVMConversionPassOptions opt;
  opt.hostBarePtrCallConv = options.hostUseBarePtrCallConv;
  opt.kernelBarePtrCallConv = options.kernelUseBarePtrCallConv;
  pm.addPass(createGpuToLLVMConversionPass(opt));

  GpuModuleToBinaryPassOptions gpuModuleToBinaryPassOptions;
  gpuModuleToBinaryPassOptions.compilationTarget = options.binaryFormat;
  gpuModuleToBinaryPassOptions.cmdOptions = options.cmdOptions;
  pm.addPass(createGpuModuleToBinaryPass(gpuModuleToBinaryPassOptions));
  pm.addPass(createConvertMathToLLVMPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(createReconcileUnrealizedCastsPass());
}

} // namespace

void mlir::gpu::buildLowerToROCDLPassPipeline(
    OpPassManager &pm, const GPUToROCDLPipelineOptions &options) {
  // Common pipelines
  buildCommonPassPipeline(pm, options);

  // GPUModule-specific stuff
  buildGpuPassPipeline(pm, options);

  // Host post-GPUModule-specific stuff
  buildHostPostPipeline(pm, options);
}

void mlir::gpu::registerGPUToROCDLPipeline() {
  PassPipelineRegistration<GPUToROCDLPipelineOptions>(
      "gpu-lower-to-rocdl-pipeline",
      "The default pipeline lowers main dialects (arith, memref, scf, vector, "
      "gpu) to ROCDL. It starts by lowering GPU code to the specified "
      "compilation target (default is fatbin) then lowers the host code.",
      buildLowerToROCDLPassPipeline);
}
