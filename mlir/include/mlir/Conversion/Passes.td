//===-- Passes.td - Conversion pass definition file --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_PASSES
#define MLIR_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// AffineToStandard
//===----------------------------------------------------------------------===//

def ConvertAffineToStandard : Pass<"lower-affine"> {
  let summary = "Lower Affine operations to a combination of Standard and Loop "
                "operations";
  let constructor = "mlir::createLowerAffinePass()";
}

//===----------------------------------------------------------------------===//
// AVX512ToLLVM
//===----------------------------------------------------------------------===//

def ConvertAVX512ToLLVM : Pass<"convert-avx512-to-llvm"> {
  let summary = "Convert the operations from the avx512 dialect into the LLVM "
                "dialect";
  let constructor = "mlir::createConvertAVX512ToLLVMPass()";
}

//===----------------------------------------------------------------------===//
// GPUToCUDA
//===----------------------------------------------------------------------===//

def ConvertGpuLaunchFuncToCudaCalls : Pass<"launch-func-to-cuda"> {
  let summary = "Convert all launch_func ops to CUDA runtime calls";
  let constructor = "mlir::createConvertGpuLaunchFuncToCudaCallsPass()";
}

//===----------------------------------------------------------------------===//
// GPUToNVVM
//===----------------------------------------------------------------------===//

def ConvertGpuOpsToNVVMOps : Pass<"convert-gpu-to-nvvm"> {
  let summary = "Generate NVVM operations for gpu operations";
  let constructor = "mlir::createLowerGpuOpsToNVVMOpsPass()";
}

//===----------------------------------------------------------------------===//
// GPUToROCDL
//===----------------------------------------------------------------------===//

def ConvertGpuOpsToROCDLOps : Pass<"convert-gpu-to-rocdl"> {
  let summary = "Generate ROCDL operations for gpu operations";
  let constructor = "mlir::createLowerGpuOpsToROCDLOpsPass()";
}

//===----------------------------------------------------------------------===//
// GPUToSPIRV
//===----------------------------------------------------------------------===//

def ConvertGPUToSPIRV : Pass<"convert-gpu-to-spirv"> {
  let summary = "Convert GPU dialect to SPIR-V dialect";
  let constructor = "mlir::createConvertGPUToSPIRVPass()";
}

//===----------------------------------------------------------------------===//
// GPUToVulkan
//===----------------------------------------------------------------------===//

def ConvertGpuLaunchFuncToVulkanLaunchFunc
    : Pass<"convert-gpu-launch-to-vulkan-launch"> {
  let summary = "Convert gpu.launch_func to vulkanLaunch external call";
  let constructor = "mlir::createConvertGpuLaunchFuncToVulkanLaunchFuncPass()";
}

def ConvertVulkanLaunchFuncToVulkanCalls : Pass<"launch-func-to-vulkan"> {
  let summary = "Convert vulkanLaunch external call to Vulkan runtime external "
                "calls";
  let constructor = "mlir::createConvertVulkanLaunchFuncToVulkanCallsPass()";
}

//===----------------------------------------------------------------------===//
// LinalgToLLVM
//===----------------------------------------------------------------------===//

def ConvertLinalgToLLVM : Pass<"convert-linalg-to-llvm"> {
  let summary = "Convert the operations from the linalg dialect into the LLVM "
                "dialect";
  let constructor = "mlir::createConvertLinalgToLLVMPass()";
}

//===----------------------------------------------------------------------===//
// LinalgToSPIRV
//===----------------------------------------------------------------------===//

def ConvertLinalgToSPIRV : Pass<"convert-linalg-to-spirv"> {
  let summary = "Convert Linalg ops to SPIR-V ops";
  let constructor = "mlir::createLinalgToSPIRVPass()";
}

//===----------------------------------------------------------------------===//
// LoopToStandard
//===----------------------------------------------------------------------===//

def ConvertLoopToStandard : Pass<"convert-loop-to-std"> {
  let summary = "Convert Loop dialect to Standard dialect, replacing structured"
                " control flow with a CFG";
  let constructor = "mlir::createLowerToCFGPass()";
}

//===----------------------------------------------------------------------===//
// LoopsToGPU
//===----------------------------------------------------------------------===//

def ConvertSimpleLoopsToGPU : Pass<"convert-loops-to-gpu"> {
  let summary = "Convert top-level loops to GPU kernels";
  let constructor = "mlir::createSimpleLoopsToGPUPass()";
}

def ConvertLoopsToGPU : Pass<"convert-loop-op-to-gpu"> {
  let summary = "Convert top-level loop::ForOp to GPU kernels";
  let constructor = "mlir::createLoopToGPUPass()";
}

def ConvertParallelLoopToGpu : Pass<"convert-parallel-loops-to-gpu"> {
  let summary = "Convert mapped loop.parallel ops to gpu launch operations";
  let constructor = "mlir::createParallelLoopToGpuPass()";
}

//===----------------------------------------------------------------------===//
// StandardToLLVM
//===----------------------------------------------------------------------===//

def ConvertStandardToLLVM : Pass<"convert-std-to-llvm"> {
  let summary = "Convert scalar and vector operations from the Standard to the "
                "LLVM dialect";
  let constructor = "mlir::createLowerToLLVMPass()";
}

//===----------------------------------------------------------------------===//
// StandardToSPIRV
//===----------------------------------------------------------------------===//

def LegalizeStandardForSPIRV : Pass<"legalize-std-for-spirv"> {
  let summary = "Legalize standard ops for SPIR-V lowering";
  let constructor = "mlir::createLegalizeStdOpsForSPIRVLoweringPass()";
}

def ConvertStandardToSPIRV : Pass<"convert-std-to-spirv"> {
  let summary = "Convert Standard Ops to SPIR-V dialect";
  let constructor = "mlir::createConvertStandardToSPIRVPass()";
}

//===----------------------------------------------------------------------===//
// VectorToLLVM
//===----------------------------------------------------------------------===//

def ConvertVectorToLLVM : Pass<"convert-vector-to-llvm"> {
  let summary = "Lower the operations from the vector dialect into the LLVM "
                "dialect";
  let constructor = "mlir::createConvertVectorToLLVMPass()";
}

#endif // MLIR_CONVERSION_PASSES
