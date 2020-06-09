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

def ConvertAffineToStandard : FunctionPass<"lower-affine"> {
  let summary = "Lower Affine operations to a combination of Standard and SCF "
                "operations";
  let description = [{

    Convert operations from the affine dialect into operations from the loop and
    standard dialects.

    `affine.for` operations are converted to `scf.for` operations that are free
    of certain structural restrictions (on their bounds and step). `affine.if`
    is similarly converted to the `scf.if` operation. `affine.apply` operations
    are converted into sequences of primitive arithmetic operations from the
    standard dialect that have the same effect, using operands of the `index`
    type. Consequently, named maps and sets thare are no longer in use may be
    removed from the module.

    For example, `%r = affine.apply affine_map<(d0, d1)[s0] -> (d0 + 2*d1 +
    s0)>(%d0, %d1)[%s0]`
    can be converted into:

    ```mlir
    %d0 = <...>
    %d1 = <...>
    %s0 = <...>
    %0 = constant 2 : index
    %1 = muli %0, %d1
    %2 = addi %d0, %1
    %r = addi %2, %s0
    ```

    #### Input invariant

    -   no `Tensor` types;

    These restrictions may be lifted in the future.

    #### Output IR

    Functions with `affine.for` and `affine.if` operations eliminated. These
    functions may contain operations from the Standard dialect in addition to
    those already present before the pass.

    #### Invariants

    -   Functions without a body are not modified.
    -   The semantics of the other functions is preserved.
    -   Individual operations other than those mentioned above are not modified
        if they do not depend on the loop iterator value or on the result of
        `affine.apply`.
  }];
  let constructor = "mlir::createLowerAffinePass()";
}

//===----------------------------------------------------------------------===//
// AVX512ToLLVM
//===----------------------------------------------------------------------===//

def ConvertAVX512ToLLVM : Pass<"convert-avx512-to-llvm", "ModuleOp"> {
  let summary = "Convert the operations from the avx512 dialect into the LLVM "
                "dialect";
  let constructor = "mlir::createConvertAVX512ToLLVMPass()";
}

//===----------------------------------------------------------------------===//
// GPUCommon
//===----------------------------------------------------------------------===//

def ConvertGpuLaunchFuncToGpuRuntimeCalls : Pass<"launch-func-to-gpu-runtime",
                                                 "ModuleOp"> {
  let summary = "Convert all launch_func ops to GPU runtime calls";
  let constructor = "mlir::createConvertGpuLaunchFuncToGpuRuntimeCallsPass()";
  let options = [
    Option<"gpuBinaryAnnotation", "gpu-binary-annotation", "std::string",
           "\"nvvm.cubin\"",
           "Annotation attribute string for GPU binary">,
  ];
}

//===----------------------------------------------------------------------===//
// GPUToNVVM
//===----------------------------------------------------------------------===//

def ConvertGpuOpsToNVVMOps : Pass<"convert-gpu-to-nvvm", "gpu::GPUModuleOp"> {
  let summary = "Generate NVVM operations for gpu operations";
  let constructor = "mlir::createLowerGpuOpsToNVVMOpsPass()";
}

//===----------------------------------------------------------------------===//
// GPUToROCDL
//===----------------------------------------------------------------------===//

def ConvertGpuOpsToROCDLOps : Pass<"convert-gpu-to-rocdl", "gpu::GPUModuleOp"> {
  let summary = "Generate ROCDL operations for gpu operations";
  let constructor = "mlir::createLowerGpuOpsToROCDLOpsPass()";
}

//===----------------------------------------------------------------------===//
// GPUToSPIRV
//===----------------------------------------------------------------------===//

def ConvertGPUToSPIRV : Pass<"convert-gpu-to-spirv", "ModuleOp"> {
  let summary = "Convert GPU dialect to SPIR-V dialect";
  let constructor = "mlir::createConvertGPUToSPIRVPass()";
}

//===----------------------------------------------------------------------===//
// GPUToVulkan
//===----------------------------------------------------------------------===//

def ConvertGpuLaunchFuncToVulkanLaunchFunc
    : Pass<"convert-gpu-launch-to-vulkan-launch", "ModuleOp"> {
  let summary = "Convert gpu.launch_func to vulkanLaunch external call";
  let constructor = "mlir::createConvertGpuLaunchFuncToVulkanLaunchFuncPass()";
}

def ConvertVulkanLaunchFuncToVulkanCalls
    : Pass<"launch-func-to-vulkan", "ModuleOp"> {
  let summary = "Convert vulkanLaunch external call to Vulkan runtime external "
                "calls";
  let constructor = "mlir::createConvertVulkanLaunchFuncToVulkanCallsPass()";
}

//===----------------------------------------------------------------------===//
// LinalgToLLVM
//===----------------------------------------------------------------------===//

def ConvertLinalgToLLVM : Pass<"convert-linalg-to-llvm", "ModuleOp"> {
  let summary = "Convert the operations from the linalg dialect into the LLVM "
                "dialect";
  let constructor = "mlir::createConvertLinalgToLLVMPass()";
}

//===----------------------------------------------------------------------===//
// LinalgToStandard
//===----------------------------------------------------------------------===//

def ConvertLinalgToStandard : Pass<"convert-linalg-to-std", "ModuleOp"> {
  let summary = "Convert the operations from the linalg dialect into the "
                "Standard dialect";
  let constructor = "mlir::createConvertLinalgToStandardPass()";
}

//===----------------------------------------------------------------------===//
// LinalgToSPIRV
//===----------------------------------------------------------------------===//

def ConvertLinalgToSPIRV : Pass<"convert-linalg-to-spirv", "ModuleOp"> {
  let summary = "Convert Linalg ops to SPIR-V ops";
  let constructor = "mlir::createLinalgToSPIRVPass()";
}

//===----------------------------------------------------------------------===//
// SCFToStandard
//===----------------------------------------------------------------------===//

def SCFToStandard : Pass<"convert-scf-to-std"> {
  let summary = "Convert SCF dialect to Standard dialect, replacing structured"
                " control flow with a CFG";
  let constructor = "mlir::createLowerToCFGPass()";
}

//===----------------------------------------------------------------------===//
// SCFToGPU
//===----------------------------------------------------------------------===//

def ConvertAffineForToGPU : FunctionPass<"convert-affine-for-to-gpu"> {
  let summary = "Convert top-level AffineFor Ops to GPU kernels";
  let constructor = "mlir::createAffineForToGPUPass()";
  let options = [
    Option<"numBlockDims", "gpu-block-dims", "unsigned", /*default=*/"1u",
           "Number of GPU block dimensions for mapping">,
    Option<"numThreadDims", "gpu-thread-dims", "unsigned", /*default=*/"1u",
           "Number of GPU thread dimensions for mapping">
  ];
}

def ConvertParallelLoopToGpu : Pass<"convert-parallel-loops-to-gpu"> {
  let summary = "Convert mapped scf.parallel ops to gpu launch operations";
  let constructor = "mlir::createParallelLoopToGpuPass()";
}

//===----------------------------------------------------------------------===//
// ShapeToStandard
//===----------------------------------------------------------------------===//

def ConvertShapeToStandard : Pass<"convert-shape-to-std", "ModuleOp"> {
  let summary = "Convert operations from the shape dialect into the standard "
                "dialect";
  let constructor = "mlir::createConvertShapeToStandardPass()";
}

//===----------------------------------------------------------------------===//
// SPIRVToLLVM
//===----------------------------------------------------------------------===//

def ConvertSPIRVToLLVM : Pass<"convert-spirv-to-llvm", "ModuleOp"> {
  let summary = "Convert SPIR-V dialect to LLVM dialect";
  let constructor = "mlir::createConvertSPIRVToLLVMPass()";
}

//===----------------------------------------------------------------------===//
// StandardToLLVM
//===----------------------------------------------------------------------===//

def ConvertStandardToLLVM : Pass<"convert-std-to-llvm", "ModuleOp"> {
  let summary = "Convert scalar and vector operations from the Standard to the "
                "LLVM dialect";
  let description = [{
    Convert standard operations into the LLVM IR dialect operations.

    #### Input invariant

    -   operations including: arithmetic on integers and floats, constants,
        direct calls, returns and branches;
    -   no `tensor` types;
    -   all `vector` are one-dimensional;
    -   all blocks are reachable by following the successors of the first basic
        block;

    If other operations are present and their results are required by the LLVM
    IR dialect operations, the pass will fail.  Any LLVM IR operations or types
    already present in the IR will be kept as is.

    #### Output IR

    Functions converted to LLVM IR. Function arguments types are converted
    one-to-one. Function results are converted one-to-one and, in case more than
    1 value is returned, packed into an LLVM IR struct type. Function calls and
    returns are updated accordingly. Block argument types are updated to use
    LLVM IR types.
  }];
  let constructor = "mlir::createLowerToLLVMPass()";
  let options = [
    Option<"useAlignedAlloc", "use-aligned-alloc", "bool", /*default=*/"false",
           "Use aligned_alloc in place of malloc for heap allocations">,
    Option<"useBarePtrCallConv", "use-bare-ptr-memref-call-conv", "bool",
           /*default=*/"false",
           "Replace FuncOp's MemRef arguments with bare pointers to the MemRef "
           "element types">,
    Option<"emitCWrappers", "emit-c-wrappers", "bool", /*default=*/"false",
           "Emit wrappers for C-compatible pointer-to-struct memref "
           "descriptors">,
    Option<"indexBitwidth", "index-bitwidth", "unsigned",
           /*default=kDeriveIndexBitwidthFromDataLayout*/"0",
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

def ConvertStandardToSPIRV : Pass<"convert-std-to-spirv", "ModuleOp"> {
  let summary = "Convert Standard Ops to SPIR-V dialect";
  let constructor = "mlir::createConvertStandardToSPIRVPass()";
}

//===----------------------------------------------------------------------===//
// VectorToSCF
//===----------------------------------------------------------------------===//

def ConvertVectorToSCF : FunctionPass<"convert-vector-to-scf"> {
  let summary = "Lower the operations from the vector dialect into the SCF "
                "dialect";
  let constructor = "mlir::createConvertVectorToSCFPass()";
  let options = [
    Option<"fullUnroll", "full-unroll", "bool", /*default=*/"false",
           "Perform full unrolling when converting vector transfers to SCF">,
  ];
}

//===----------------------------------------------------------------------===//
// VectorToLLVM
//===----------------------------------------------------------------------===//

def ConvertVectorToLLVM : Pass<"convert-vector-to-llvm", "ModuleOp"> {
  let summary = "Lower the operations from the vector dialect into the LLVM "
                "dialect";
  let constructor = "mlir::createConvertVectorToLLVMPass()";
}

#endif // MLIR_CONVERSION_PASSES
