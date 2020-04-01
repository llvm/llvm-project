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
  let options = [
    Option<"numBlockDims", "gpu-block-dims", "unsigned", /*default=*/"1u",
           "Number of GPU block dimensions for mapping">,
    Option<"numThreadDims", "gpu-thread-dims", "unsigned", /*default=*/"1u",
           "Number of GPU thread dimensions for mapping">
  ];
}

def ConvertLoopsToGPU : Pass<"convert-loop-op-to-gpu"> {
  let summary = "Convert top-level loop::ForOp to GPU kernels";
  let constructor = "mlir::createLoopToGPUPass()";
  let options = [
    ListOption<"numWorkGroups", "gpu-num-workgroups", "int64_t",
               "Num workgroups in the GPU launch",
               "llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated">,
    ListOption<"workGroupSize", "gpu-workgroup-size", "int64_t",
               "Workgroup Size in the GPU launch",
               "llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated">
  ];
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
  let options = [
    Option<"useAlloca", "use-alloca", "bool", /*default=*/"false",
           "Use `alloca` instead of `call @malloc` for converting std.alloc">,
    Option<"useBarePtrCallConv", "use-bare-ptr-memref-call-conv", "bool",
           /*default=*/"false",
           "Replace FuncOp's MemRef arguments with bare pointers to the MemRef "
           "element types">,
    Option<"emitCWrappers", "emit-c-wrappers", "bool", /*default=*/"false",
           "Emit wrappers for C-compatible pointer-to-struct memref "
           "descriptors">,
    Option<"indexBitwidth", "index-bitwidth", "unsigned",
           /*default=*/"kDeriveIndexBitwidthFromDataLayout",
           "Bitwidth of the index type, 0 to use size of machine word">,
  ];
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
