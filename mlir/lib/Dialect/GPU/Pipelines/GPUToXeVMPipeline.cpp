//===- GPUToXeVMPipeline.cpp - Lowering pipeline to XeVM/LLVM -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass for testing the lowering to XeVM as a generally
// usable sink pass. If XeGPU ops are used, it expects the MLIR code to have
// XeGPU ops already embedded in gpu code.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/MathToXeVM/MathToXeVM.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Conversion/XeGPUToXeVM/XeGPUToXeVM.h"
#include "mlir/Conversion/XeVMToLLVM/XeVMToLLVM.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Pipelines/Passes.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/Transforms/RequestCWrappers.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/XeGPU/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Target/LLVM/XeVM/Target.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace {
//===----------------------------------------------------------------------===//
// Pre-GPU common pipeline for both Host and GPU.
//===----------------------------------------------------------------------===//
void buildPreGPUCommonPassPipeline(
    OpPassManager &pm, const mlir::gpu::GPUToXeVMPipelineOptions &options) {
  // builtin.module scope passes.
  pm.addPass(createCSEPass());
  pm.addPass(createConvertVectorToSCFPass());
  {
    GpuXeVMAttachTargetOptions xevmTargetOptions;
    xevmTargetOptions.moduleMatcher = options.xevmModuleMatcher;
    xevmTargetOptions.triple = options.zebinTriple;
    xevmTargetOptions.chip = options.zebinChip;
    xevmTargetOptions.optLevel = options.optLevel;
    xevmTargetOptions.cmdOptions = options.cmdOptions;
    pm.addPass(createGpuXeVMAttachTarget(xevmTargetOptions));
  }
  pm.addPass(createLowerAffinePass());
  pm.addNestedPass<func::FuncOp>(createGpuAsyncRegionPass());
}

//===----------------------------------------------------------------------===//
// GPUModule-specific stuff.
//===----------------------------------------------------------------------===//
void buildGPUPassPipeline(OpPassManager &pm,
                          const mlir::gpu::GPUToXeVMPipelineOptions &options) {
  if (options.xegpuOpLevel == "workgroup") {
    pm.addNestedPass<gpu::GPUModuleOp>(xegpu::createXeGPUWgToSgDistribute());
    pm.addNestedPass<gpu::GPUModuleOp>(createCSEPass());
    pm.addNestedPass<gpu::GPUModuleOp>(xegpu::createXeGPUBlocking());
    pm.addNestedPass<gpu::GPUModuleOp>(createCanonicalizerPass());
    pm.addNestedPass<gpu::GPUModuleOp>(createCSEPass());
  }
  if (options.xegpuOpLevel == "subgroup" ||
      options.xegpuOpLevel == "workgroup") {
    pm.addNestedPass<gpu::GPUModuleOp>(xegpu::createXeGPUPropagateLayout());
    pm.addNestedPass<gpu::GPUModuleOp>(xegpu::createXeGPUSubgroupDistribute());
    pm.addNestedPass<gpu::GPUModuleOp>(createCanonicalizerPass());
    pm.addNestedPass<gpu::GPUModuleOp>(createCSEPass());
    pm.addNestedPass<gpu::GPUModuleOp>(createLoopInvariantCodeMotionPass());
    pm.addNestedPass<gpu::GPUModuleOp>(createCSEPass());
    pm.addNestedPass<gpu::GPUModuleOp>(xegpu::createXeGPUVectorLinearize());
  }
  pm.addNestedPass<gpu::GPUModuleOp>(createConvertMathToXeVM());
  pm.addNestedPass<gpu::GPUModuleOp>(createConvertXeGPUToXeVMPass());
  {
    ConvertGpuOpsToLLVMSPVOpsOptions gpuToLLVMSPVOptions;
    gpuToLLVMSPVOptions.use64bitIndex = options.use64bitIndex;
    pm.addNestedPass<gpu::GPUModuleOp>(
        createConvertGpuOpsToLLVMSPVOps(gpuToLLVMSPVOptions));
  }
  pm.addNestedPass<gpu::GPUModuleOp>(createCSEPass());
  pm.addNestedPass<gpu::GPUModuleOp>(createReconcileUnrealizedCastsPass());
}

//===----------------------------------------------------------------------===//
// Post-GPU pipeline for both Host and GPU.
//===----------------------------------------------------------------------===//
void buildPostGPUCommonPassPipeline(
    OpPassManager &pm, const mlir::gpu::GPUToXeVMPipelineOptions &options) {
  // builtin.module scope passes.
  pm.addPass(createSCFToControlFlowPass());
  pm.addPass(memref::createExpandStridedMetadataPass());
  {
    GpuToLLVMConversionPassOptions gpuToLLVMOptions;
    gpuToLLVMOptions.hostBarePtrCallConv = options.hostBarePtrCallConv;
    gpuToLLVMOptions.kernelBarePtrCallConv = options.kernelBarePtrCallConv;
    pm.addPass(createGpuToLLVMConversionPass(gpuToLLVMOptions));
  }
  pm.addPass(createLowerAffinePass());
  pm.addPass(createConvertToLLVMPass());
  pm.addPass(createReconcileUnrealizedCastsPass());
  // gpu-module-to-binary
  {
    GpuModuleToBinaryPassOptions gpuToModuleBinOptions;
    gpuToModuleBinOptions.compilationTarget = options.binaryFormat;
    gpuToModuleBinOptions.cmdOptions = options.cmdOptions;
    pm.addPass(createGpuModuleToBinaryPass(gpuToModuleBinOptions));
  }
}
} // namespace

void mlir::gpu::buildLowerToXeVMPassPipeline(
    OpPassManager &pm, const GPUToXeVMPipelineOptions &options) {
  // Pre-GPU common pipelines.
  buildPreGPUCommonPassPipeline(pm, options);

  // GPUModule-specific stuff.
  buildGPUPassPipeline(pm, options);

  // Post-GPU pipeline for both Host and GPU.
  buildPostGPUCommonPassPipeline(pm, options);
}

void mlir::gpu::registerGPUToXeVMPipeline() {
  PassPipelineRegistration<GPUToXeVMPipelineOptions>(
      "gpu-lower-to-xevm-pipeline",
      "The default GPU to XeVM lowering pipeline. It starts by lowering GPU "
      "code to the "
      "specified compilation target (default is fatbin) then lowers the host "
      "code.",
      buildLowerToXeVMPassPipeline);
}
