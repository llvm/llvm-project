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
  let summary = "Lower Affine operations to a combination of Standard and SCF "
                "operations";
  let description = [{

    Convert operations from the affine dialect into operations from the SCF and
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
    %0 = arith.constant 2 : index
    %1 = arith.muli %0, %d1
    %2 = arith.addi %d0, %1
    %r = arith.addi %2, %s0
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
  let dependentDialects = [
    "memref::MemRefDialect",
    "scf::SCFDialect",
    "vector::VectorDialect"
  ];
}

//===----------------------------------------------------------------------===//
// AMDGPUToROCDL
//===----------------------------------------------------------------------===//

def ConvertAMDGPUToROCDL : Pass<"convert-amdgpu-to-rocdl"> {
  let summary = "Convert AMDGPU dialect to ROCDL dialect";
  let description = [{
    This pass converts supported AMDGPU ops to ROCDL dialect intrinsics.
  }];
  let constructor = "mlir::createConvertAMDGPUToROCDLPass()";
  let dependentDialects = [
    "LLVM::LLVMDialect",
    "ROCDL::ROCDLDialect",
  ];
  let options = [Option<"chipset", "chipset", "std::string",
                        /*default=*/"\"gfx000\"",
                        "Chipset that these operations will run on">];
}

//===----------------------------------------------------------------------===//
// ArithToLLVM
//===----------------------------------------------------------------------===//

def ArithToLLVMConversionPass : Pass<"convert-arith-to-llvm"> {
  let summary = "Convert Arith dialect to LLVM dialect";
  let description = [{
    This pass converts supported Arith ops to LLVM dialect instructions.
  }];
  let dependentDialects = ["LLVM::LLVMDialect"];
  let options = [
    Option<"indexBitwidth", "index-bitwidth", "unsigned",
           /*default=kDeriveIndexBitwidthFromDataLayout*/"0",
           "Bitwidth of the index type, 0 to use size of machine word">,
  ];
}

//===----------------------------------------------------------------------===//
// ArithToSPIRV
//===----------------------------------------------------------------------===//

def ConvertArithToSPIRV : Pass<"convert-arith-to-spirv"> {
  let summary = "Convert Arith dialect to SPIR-V dialect";
  let constructor = "mlir::arith::createConvertArithToSPIRVPass()";
  let dependentDialects = ["spirv::SPIRVDialect"];
  let options = [
    Option<"emulateLT32BitScalarTypes", "emulate-lt-32-bit-scalar-types",
           "bool", /*default=*/"true",
           "Emulate narrower scalar types with 32-bit ones if not supported by "
           "the target">,
    Option<"enableFastMath", "enable-fast-math",
           "bool", /*default=*/"false",
           "Enable fast math mode (assuming no NaN and infinity for floating "
           "point values) when performing conversion">
  ];
}

//===----------------------------------------------------------------------===//
// ArmNeon2dToIntr
//===----------------------------------------------------------------------===//

def ConvertArmNeon2dToIntr : Pass<"arm-neon-2d-to-intr"> {
  let summary = "Convert Arm NEON structured ops to intrinsics";
  let constructor = "mlir::createConvertArmNeon2dToIntrPass()";
  let dependentDialects = ["arm_neon::ArmNeonDialect", "vector::VectorDialect"];
}

//===----------------------------------------------------------------------===//
// AsyncToLLVM
//===----------------------------------------------------------------------===//

def ConvertAsyncToLLVMPass : Pass<"convert-async-to-llvm", "ModuleOp"> {
  let summary = "Convert the operations from the async dialect into the LLVM "
                "dialect";
  let description = [{
    Convert `async.execute` operations to LLVM coroutines and use async runtime
    API to execute them.
  }];
  let dependentDialects = [
    "arith::ArithDialect",
    "async::AsyncDialect",
    "LLVM::LLVMDialect",
    "func::FuncDialect",
  ];
  let options = [
    Option<"useOpaquePointers", "use-opaque-pointers", "bool",
           /*default=*/"false", "Generate LLVM IR using opaque pointers "
           "instead of typed pointers">,
  ];
}

//===----------------------------------------------------------------------===//
// BufferizationToMemRef
//===----------------------------------------------------------------------===//

def ConvertBufferizationToMemRef : Pass<"convert-bufferization-to-memref"> {
  let summary = "Convert operations from the Bufferization dialect to the "
                "MemRef dialect";
  let description = [{

    This pass converts bufferization operations into memref operations.

    In the current state, this pass only transforms a `bufferization.clone`
    operation into `memref.alloc` and `memref.copy` operations. This conversion
    is needed, since some clone operations could remain after applying several
    transformation processes. Currently, only `canonicalize` transforms clone
    operations or even eliminates them. This can lead to errors if any clone op
    survived after all conversion passes (starting from the bufferization
    dialect) are performed.

    See:
    https://llvm.discourse.group/t/bufferization-error-related-to-memref-clone/4665

    To avoid these errors, this pass can be performed as a last clean-up pass to
    transform remaining operations and to proceed in other dialects (memref
    e.g.).

    Note that this pass only transforms the operation without any further
    analyses. This pass does not consider any memory analysis or optimization
    and hence does not resolve any memory leaks.

  }];
  let constructor = "mlir::createBufferizationToMemRefPass()";
  let dependentDialects = ["arith::ArithDialect", "memref::MemRefDialect"];
}

//===----------------------------------------------------------------------===//
// ComplexToLLVM
//===----------------------------------------------------------------------===//

def ConvertComplexToLLVMPass : Pass<"convert-complex-to-llvm"> {
  let summary = "Convert Complex dialect to LLVM dialect";
  let dependentDialects = ["LLVM::LLVMDialect"];
}

//===----------------------------------------------------------------------===//
// ComplexToLibm
//===----------------------------------------------------------------------===//

def ConvertComplexToLibm : Pass<"convert-complex-to-libm", "ModuleOp"> {
  let summary = "Convert Complex dialect to libm calls";
  let description = [{
    This pass converts supported Complex ops to libm calls.
  }];
  let constructor = "mlir::createConvertComplexToLibmPass()";
  let dependentDialects = [
    "func::FuncDialect",
  ];
}

//===----------------------------------------------------------------------===//
// ComplexToStandard
//===----------------------------------------------------------------------===//

def ConvertComplexToStandard : Pass<"convert-complex-to-standard"> {
  let summary = "Convert Complex dialect to standard dialect";
  let constructor = "mlir::createConvertComplexToStandardPass()";
  let dependentDialects = ["math::MathDialect"];
}

//===----------------------------------------------------------------------===//
// ControlFlowToLLVM
//===----------------------------------------------------------------------===//

def ConvertControlFlowToLLVMPass : Pass<"convert-cf-to-llvm", "ModuleOp"> {
  let summary = "Convert ControlFlow operations to the LLVM dialect";
  let description = [{
    Convert ControlFlow operations into LLVM IR dialect operations.

    If other operations are present and their results are required by the LLVM
    IR dialect operations, the pass will fail.  Any LLVM IR operations or types
    already present in the IR will be kept as is.
  }];
  let dependentDialects = ["LLVM::LLVMDialect"];
  let options = [
    Option<"indexBitwidth", "index-bitwidth", "unsigned",
           /*default=kDeriveIndexBitwidthFromDataLayout*/"0",
           "Bitwidth of the index type, 0 to use size of machine word">,
    Option<"useOpaquePointers", "use-opaque-pointers", "bool",
                   /*default=*/"false", "Generate LLVM IR using opaque pointers "
                   "instead of typed pointers">,
  ];
}

//===----------------------------------------------------------------------===//
// ControlFlowToSPIRV
//===----------------------------------------------------------------------===//

def ConvertControlFlowToSPIRV : Pass<"convert-cf-to-spirv"> {
  let summary = "Convert ControlFlow dialect to SPIR-V dialect";
  let constructor = "mlir::createConvertControlFlowToSPIRVPass()";
  let dependentDialects = ["spirv::SPIRVDialect"];
  let options = [
    Option<"emulateLT32BitScalarTypes", "emulate-lt-32-bit-scalar-types",
           "bool", /*default=*/"true",
           "Emulate narrower scalar types with 32-bit ones if not supported by"
           " the target">
  ];
}

//===----------------------------------------------------------------------===//
// FuncToLLVM
//===----------------------------------------------------------------------===//

def ConvertFuncToLLVMPass : Pass<"convert-func-to-llvm", "ModuleOp"> {
  let summary = "Convert from the Func dialect to the LLVM dialect";
  let description = [{
    Convert Func dialect operations into the LLVM IR dialect operations.

    #### Input invariant

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
  let dependentDialects = ["LLVM::LLVMDialect"];
  let options = [
    Option<"useBarePtrCallConv", "use-bare-ptr-memref-call-conv", "bool",
           /*default=*/"false",
           "Replace FuncOp's MemRef arguments with bare pointers to the MemRef "
           "element types">,
    Option<"indexBitwidth", "index-bitwidth", "unsigned",
           /*default=kDeriveIndexBitwidthFromDataLayout*/"0",
           "Bitwidth of the index type, 0 to use size of machine word">,
    Option<"dataLayout", "data-layout", "std::string",
           /*default=*/"\"\"",
           "String description (LLVM format) of the data layout that is "
           "expected on the produced module">,
    Option<"useOpaquePointers", "use-opaque-pointers", "bool",
                       /*default=*/"false", "Generate LLVM IR using opaque pointers "
                       "instead of typed pointers">,
  ];
}

//===----------------------------------------------------------------------===//
// FuncToSPIRV
//===----------------------------------------------------------------------===//

def ConvertFuncToSPIRV : Pass<"convert-func-to-spirv"> {
  let summary = "Convert Func dialect to SPIR-V dialect";
  let constructor = "mlir::createConvertFuncToSPIRVPass()";
  let dependentDialects = ["spirv::SPIRVDialect"];
  let options = [
    Option<"emulateLT32BitScalarTypes", "emulate-lt-32-bit-scalar-types",
           "bool", /*default=*/"true",
           "Emulate narrower scalar types with 32-bit ones if not supported by"
           " the target">
  ];
}

//===----------------------------------------------------------------------===//
// GPUCommon
//===----------------------------------------------------------------------===//

def GpuToLLVMConversionPass : Pass<"gpu-to-llvm", "ModuleOp"> {
  let summary = "Convert GPU dialect to LLVM dialect with GPU runtime calls";

  let description = [{
    Creates a pass to convert a GPU operations into a sequence of GPU runtime
    calls.

    This pass does not generate code to call GPU runtime APIs directly but
    instead uses a small wrapper library that exports a stable and conveniently
    typed ABI on top of GPU runtimes such as CUDA or ROCm (HIP).
  }];

  let options = [
    Option<"kernelBarePtrCallConv", "use-bare-pointers-for-kernels", "bool",
           /*default=*/"false",
             "Use bare pointers to pass memref arguments to kernels. "
             "The kernel must use the same setting for this option."
           >,
    Option<"gpuBinaryAnnotation", "gpu-binary-annotation", "std::string",
               /*default=*/"gpu::getDefaultGpuBinaryAnnotation()",
               "Annotation attribute string for GPU binary"
               >,
  ];

  let dependentDialects = [
    "LLVM::LLVMDialect",
    "memref::MemRefDialect",
  ];
}

def LowerHostCodeToLLVMPass : Pass<"lower-host-to-llvm", "ModuleOp"> {
  let summary = "Lowers the host module code and `gpu.launch_func` to LLVM";

  let description = [{
    Creates a pass to emulate `gpu.launch_func` call in LLVM dialect and lower
    the host module code to LLVM.

    This transformation creates a sequence of global variables that are later
    linked to the variables in the kernel module, and a series of copies to/from
    them to emulate the memory transfer from the host or to the device sides. It
    also converts the remaining Arithmetic, Func, and MemRef dialects into LLVM
    dialect, emitting C wrappers.
  }];

  let options = [
    Option<"useOpaquePointers", "use-opaque-pointers", "bool",
                 /*default=*/"false", "Generate LLVM IR using opaque pointers "
                 "instead of typed pointers">
  ];

  let dependentDialects = ["LLVM::LLVMDialect"];
}

//===----------------------------------------------------------------------===//
// GPUToNVVM
//===----------------------------------------------------------------------===//

def ConvertGpuOpsToNVVMOps : Pass<"convert-gpu-to-nvvm", "gpu::GPUModuleOp"> {
  let summary = "Generate NVVM operations for gpu operations";
  let constructor = "mlir::createLowerGpuOpsToNVVMOpsPass()";
  let dependentDialects = [
    "cf::ControlFlowDialect",
    "memref::MemRefDialect",
    "NVVM::NVVMDialect",
  ];
  let options = [
    Option<"indexBitwidth", "index-bitwidth", "unsigned",
           /*default=kDeriveIndexBitwidthFromDataLayout*/"0",
           "Bitwidth of the index type, 0 to use size of machine word">,
    Option<"hasRedux", "has-redux", "bool", /*default=*/"false",
           "Target gpu supports redux">,
  ];
}

//===----------------------------------------------------------------------===//
// GPUToROCDL
//===----------------------------------------------------------------------===//

def ConvertGpuOpsToROCDLOps : Pass<"convert-gpu-to-rocdl", "gpu::GPUModuleOp"> {
  let summary = "Generate ROCDL operations for gpu operations";
  let constructor = "mlir::createLowerGpuOpsToROCDLOpsPass()";
  let dependentDialects = [
    "ROCDL::ROCDLDialect",
    "cf::ControlFlowDialect",
    "memref::MemRefDialect",
  ];
  let options = [
    Option<"chipset", "chipset", "std::string",
           /*default=*/"\"gfx000\"",
           "Chipset that these operations will run on">,
    Option<"indexBitwidth", "index-bitwidth", "unsigned",
           /*default=kDeriveIndexBitwidthFromDataLayout*/"0",
           "Bitwidth of the index type, 0 to use size of machine word">,
    Option<"useBarePtrCallConv", "use-bare-ptr-memref-call-conv", "bool",
           /*default=*/"false",
           "Replace memref arguments in GPU functions with bare pointers."
           "All memrefs must have static shape">,
    Option<"runtime", "runtime", "::mlir::gpu::amd::Runtime",
          "::mlir::gpu::amd::Runtime::Unknown",
          "Runtime code will be run on (default is Unknown, can also use HIP or OpenCl)",
          [{::llvm::cl::values(
            clEnumValN(::mlir::gpu::amd::Runtime::Unknown, "unknown", "Unknown (default)"),
            clEnumValN(::mlir::gpu::amd::Runtime::HIP, "HIP", "HIP"),
            clEnumValN(::mlir::gpu::amd::Runtime::OpenCL, "OpenCL", "OpenCL")
          )}]>
  ];
}

//===----------------------------------------------------------------------===//
// GPUToSPIRV
//===----------------------------------------------------------------------===//

def ConvertGPUToSPIRV : Pass<"convert-gpu-to-spirv", "ModuleOp"> {
  let summary = "Convert GPU dialect to SPIR-V dialect";
  let description = [{
    This pass converts supported GPU device ops to SPIR-V ops. It does not
    handle GPU host ops.

    A `gpu.func` op can have parameters to pass in resources. But in SPIR-V
    entry functions cannot take parameters; they use descriptors to access
    resources. By default, parameters to a `gpu.func` op will be converted to
    global variables. These global variables will be assigned sequential binding
    numbers following their order in the original `gpu.func` op, starting from
    0, in set 0. One can attach `spirv.interface_var_abi` to those parameters
    to control the set and binding if wanted.
  }];
  let constructor = "mlir::createConvertGPUToSPIRVPass()";
  let dependentDialects = ["spirv::SPIRVDialect"];
}

//===----------------------------------------------------------------------===//
// GPUToVulkan
//===----------------------------------------------------------------------===//

def ConvertGpuLaunchFuncToVulkanLaunchFunc
    : Pass<"convert-gpu-launch-to-vulkan-launch", "ModuleOp"> {
  let summary = "Convert gpu.launch_func to vulkanLaunch external call";
  let description = [{
    This pass is only intended for the mlir-vulkan-runner.
  }];
  let constructor = "mlir::createConvertGpuLaunchFuncToVulkanLaunchFuncPass()";
  let dependentDialects = ["spirv::SPIRVDialect"];
}

def ConvertVulkanLaunchFuncToVulkanCalls
    : Pass<"launch-func-to-vulkan", "ModuleOp"> {
  let summary = "Convert vulkanLaunch external call to Vulkan runtime external "
                "calls";
  let description = [{
    This pass is only intended for the mlir-vulkan-runner.
  }];
  let constructor = "mlir::createConvertVulkanLaunchFuncToVulkanCallsPass()";
  let dependentDialects = ["LLVM::LLVMDialect"];
}

//===----------------------------------------------------------------------===//
// ConvertIndexToLLVMPass
//===----------------------------------------------------------------------===//

def ConvertIndexToLLVMPass : Pass<"convert-index-to-llvm"> {
  let summary = "Lower the `index` dialect to the `llvm` dialect.";
  let description = [{
    This pass lowers Index dialect operations to LLVM dialect operations.
    Operation conversions are 1-to-1 except for the exotic divides: `ceildivs`,
    `ceildivu`, and `floordivs`, which expand to series of LLVM operations.
    Importantly, the index bitwidth should be correctly set to the target
    pointer width via `index-bitwidth`.
  }];

  let dependentDialects = ["::mlir::LLVM::LLVMDialect"];

  let options = [
    Option<"indexBitwidth", "index-bitwidth", "unsigned",
           /*default=kDeriveIndexBitwidthFromDataLayout*/"0",
           "Bitwidth of the index type, 0 to use size of machine word">,
  ];
}

//===----------------------------------------------------------------------===//
// LinalgToLLVM
//===----------------------------------------------------------------------===//

def ConvertLinalgToLLVMPass : Pass<"convert-linalg-to-llvm", "ModuleOp"> {
  let summary = "Convert the operations from the linalg dialect into the LLVM "
                "dialect";
  let dependentDialects = ["scf::SCFDialect", "LLVM::LLVMDialect"];
}

//===----------------------------------------------------------------------===//
// LinalgToStandard
//===----------------------------------------------------------------------===//

def ConvertLinalgToStandard : Pass<"convert-linalg-to-std", "ModuleOp"> {
  let summary = "Convert the operations from the linalg dialect into the "
                "Standard dialect";
  let constructor = "mlir::createConvertLinalgToStandardPass()";
  let dependentDialects = ["func::FuncDialect", "memref::MemRefDialect"];
}

//===----------------------------------------------------------------------===//
// MathToLibm
//===----------------------------------------------------------------------===//

def ConvertMathToLibm : Pass<"convert-math-to-libm", "ModuleOp"> {
  let summary = "Convert Math dialect to libm calls";
  let description = [{
    This pass converts supported Math ops to libm calls.
  }];
  let constructor = "mlir::createConvertMathToLibmPass()";
  let dependentDialects = [
    "arith::ArithDialect",
    "func::FuncDialect",
    "vector::VectorDialect",
  ];
}

//===----------------------------------------------------------------------===//
// MathToLLVM
//===----------------------------------------------------------------------===//

def ConvertMathToLLVMPass : Pass<"convert-math-to-llvm"> {
  let summary = "Convert Math dialect to LLVM dialect";
  let dependentDialects = ["LLVM::LLVMDialect"];
  let options = [
    Option<"approximateLog1p", "approximate-log1p", "bool", "true",
           "Enable approximation of Log1p.">
  ];
}

//===----------------------------------------------------------------------===//
// MathToSPIRV
//===----------------------------------------------------------------------===//

def ConvertMathToSPIRV : Pass<"convert-math-to-spirv"> {
  let summary = "Convert Math dialect to SPIR-V dialect";
  let constructor = "mlir::createConvertMathToSPIRVPass()";
  let dependentDialects = ["spirv::SPIRVDialect"];
}

//===----------------------------------------------------------------------===//
// MathToFuncs
//===----------------------------------------------------------------------===//

def ConvertMathToFuncs : Pass<"convert-math-to-funcs", "ModuleOp"> {
  let summary = "Convert Math operations to calls of outlined implementations.";
  let description = [{
    This pass converts supported Math ops to calls of compiler generated
    functions implementing these operations in software.
    The LLVM dialect is used for LinkonceODR linkage of the generated functions.
  }];
  let dependentDialects = [
    "arith::ArithDialect",
    "cf::ControlFlowDialect",
    "func::FuncDialect",
    "vector::VectorDialect",
    "LLVM::LLVMDialect",
  ];
  let options = [
    Option<"minWidthOfFPowIExponent", "min-width-of-fpowi-exponent", "unsigned",
           /*default=*/"1",
           "Convert FPowI only if the width of its exponent's integer type "
           "is greater than or equal to this value">
  ];
}

//===----------------------------------------------------------------------===//
// MemRefToLLVM
//===----------------------------------------------------------------------===//

def FinalizeMemRefToLLVMConversionPass :
    Pass<"finalize-memref-to-llvm", "ModuleOp"> {
  let summary = "Finalize the conversion of the operations from the MemRef "
                "dialect to the LLVM dialect."
                "This conversion will not convert some complex MemRef "
                "operations. Make sure to run `expand-strided-metadata` "
                "beforehand for these.";
  let dependentDialects = ["LLVM::LLVMDialect"];
  let options = [
    Option<"useAlignedAlloc", "use-aligned-alloc", "bool", /*default=*/"false",
           "Use aligned_alloc in place of malloc for heap allocations">,
    Option<"indexBitwidth", "index-bitwidth", "unsigned",
           /*default=kDeriveIndexBitwidthFromDataLayout*/"0",
           "Bitwidth of the index type, 0 to use size of machine word">,
    Option<"useGenericFunctions", "use-generic-functions",
           "bool",
           /*default=*/"false",
           "Use generic allocation and deallocation functions instead of the "
           "classic 'malloc', 'aligned_alloc' and 'free' functions">,
    Option<"useOpaquePointers", "use-opaque-pointers", "bool",
               /*default=*/"false", "Generate LLVM IR using opaque pointers "
               "instead of typed pointers">
  ];
}

//===----------------------------------------------------------------------===//
// MemRefToSPIRV
//===----------------------------------------------------------------------===//

def MapMemRefStorageClass : Pass<"map-memref-spirv-storage-class"> {
  let summary = "Map numeric MemRef memory spaces to SPIR-V storage classes";
  let constructor = "mlir::createMapMemRefStorageClassPass()";
  let dependentDialects = ["spirv::SPIRVDialect"];
  let options = [
    Option<"clientAPI", "client-api", "std::string", /*default=*/"\"vulkan\"",
           "The client API to use for populating mappings">
  ];
}

def ConvertMemRefToSPIRV : Pass<"convert-memref-to-spirv"> {
  let summary = "Convert MemRef dialect to SPIR-V dialect";
  let constructor = "mlir::createConvertMemRefToSPIRVPass()";
  let dependentDialects = ["spirv::SPIRVDialect"];
  let options = [
    Option<"boolNumBits", "bool-num-bits",
           "int", /*default=*/"8",
           "The number of bits to store a boolean value">
  ];
}

//===----------------------------------------------------------------------===//
// NVGPUToNVVM
//===----------------------------------------------------------------------===//

def ConvertNVGPUToNVVM : Pass<"convert-nvgpu-to-nvvm"> {
  let summary = "Convert NVGPU dialect to NVVM dialect";
  let description = [{
    This pass converts supported NVGPU ops to NVVM dialect intrinsics.
  }];
  let constructor = "mlir::createConvertNVGPUToNVVMPass()";
  let dependentDialects = [
    "NVVM::NVVMDialect",
  ];
}


//===----------------------------------------------------------------------===//
// OpenACCToSCF
//===----------------------------------------------------------------------===//

def ConvertOpenACCToSCF : Pass<"convert-openacc-to-scf", "ModuleOp"> {
  let summary = "Convert the OpenACC ops to OpenACC with SCF dialect";
  let constructor = "mlir::createConvertOpenACCToSCFPass()";
  let dependentDialects = ["scf::SCFDialect", "acc::OpenACCDialect"];
}

//===----------------------------------------------------------------------===//
// OpenACCToLLVM
//===----------------------------------------------------------------------===//

def ConvertOpenACCToLLVMPass : Pass<"convert-openacc-to-llvm", "ModuleOp"> {
  let summary = "Convert the OpenACC ops to LLVM dialect";
  let dependentDialects = ["LLVM::LLVMDialect"];
  let options = [
    Option<"useOpaquePointers", "use-opaque-pointers", "bool",
           /*default=*/"false", "Generate LLVM IR using opaque pointers "
           "instead of typed pointers">,
  ];
}

//===----------------------------------------------------------------------===//
// OpenMPToLLVM
//===----------------------------------------------------------------------===//

def ConvertOpenMPToLLVMPass : Pass<"convert-openmp-to-llvm", "ModuleOp"> {
  let summary = "Convert the OpenMP ops to OpenMP ops with LLVM dialect";
  let dependentDialects = ["LLVM::LLVMDialect"];
}

//===----------------------------------------------------------------------===//
// PDLToPDLInterp
//===----------------------------------------------------------------------===//

def ConvertPDLToPDLInterp : Pass<"convert-pdl-to-pdl-interp", "ModuleOp"> {
  let summary = "Convert PDL ops to PDL interpreter ops";
  let constructor = "mlir::createPDLToPDLInterpPass()";
  let dependentDialects = ["pdl_interp::PDLInterpDialect"];
}

//===----------------------------------------------------------------------===//
// ReconcileUnrealizedCasts
//===----------------------------------------------------------------------===//

def ReconcileUnrealizedCasts : Pass<"reconcile-unrealized-casts"> {
  let summary = "Simplify and eliminate unrealized conversion casts";
  let description = [{
    Eliminate `unrealized_conversion_cast` operations, commonly introduced by
    partial dialect conversions, that transitively convert a value to another
    value of the same type, that is:

    ```
    %0 = "producer.op"() : () -> !type.A
    %1 = unrealized_conversion_cast %0 : !type.A to !type.B
    %2 = unrealized_conversion_cast %1 : !type.B to !type.C
    %3 = unrealized_conversion_cast %2 : !type.C to !type.A
    "consumer.op"(%3) : (!type.A) -> ()
    ```

    Such situations appear when the consumer operation is converted by one pass
    and the producer operation is converted by another pass, each of which
    produces an unrealized cast. This pass can be used to clean up the IR.
  }];
  let constructor = "mlir::createReconcileUnrealizedCastsPass()";
}

//===----------------------------------------------------------------------===//
// SCFToControlFlow
//===----------------------------------------------------------------------===//

def SCFToControlFlow : Pass<"convert-scf-to-cf"> {
  let summary = "Convert SCF dialect to ControlFlow dialect, replacing structured"
                " control flow with a CFG";
  let constructor = "mlir::createConvertSCFToCFPass()";
  let dependentDialects = ["cf::ControlFlowDialect"];
}

//===----------------------------------------------------------------------===//
// SCFToOpenMP
//===----------------------------------------------------------------------===//

def ConvertSCFToOpenMPPass : Pass<"convert-scf-to-openmp", "ModuleOp"> {
  let summary = "Convert SCF parallel loop to OpenMP parallel + workshare "
                "constructs.";

  let options = [
    Option<"useOpaquePointers", "use-opaque-pointers", "bool",
                 /*default=*/"false", "Generate LLVM IR using opaque pointers "
                 "instead of typed pointers">
  ];

  let dependentDialects = ["omp::OpenMPDialect", "LLVM::LLVMDialect",
                           "memref::MemRefDialect"];
}

//===----------------------------------------------------------------------===//
// SCFToSPIRV
//===----------------------------------------------------------------------===//

def SCFToSPIRV : Pass<"convert-scf-to-spirv"> {
  let summary = "Convert SCF dialect to SPIR-V dialect.";
  let description = [{
    This pass converts SCF ops into SPIR-V structured control flow ops.
    SPIR-V structured control flow ops does not support yielding values.
    So for SCF ops yielding values, SPIR-V variables are created for
    holding the values and load/store operations are emitted for updating
    them.
  }];
  let constructor = "mlir::createConvertSCFToSPIRVPass()";
  let dependentDialects = ["spirv::SPIRVDialect"];
}

//===----------------------------------------------------------------------===//
// SCFToGPU
//===----------------------------------------------------------------------===//

def ConvertAffineForToGPU
    : InterfacePass<"convert-affine-for-to-gpu", "FunctionOpInterface"> {
  let summary = "Convert top-level AffineFor Ops to GPU kernels";
  let constructor = "mlir::createAffineForToGPUPass()";
  let dependentDialects = ["gpu::GPUDialect"];
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
  let dependentDialects = ["AffineDialect", "gpu::GPUDialect"];
}

//===----------------------------------------------------------------------===//
// ShapeToStandard
//===----------------------------------------------------------------------===//

def ConvertShapeToStandard : Pass<"convert-shape-to-std", "ModuleOp"> {
  let summary = "Convert operations from the shape dialect into the standard "
                "dialect";
  let constructor = "mlir::createConvertShapeToStandardPass()";
  let dependentDialects = [
    "scf::SCFDialect",
  ];
}

def ConvertShapeConstraints : Pass<"convert-shape-constraints"> {
  let summary = "Convert shape constraint operations to the standard dialect";
  let description = [{
    This pass eliminates shape constraints from the program, converting them to
    eager (side-effecting) error handling code.

    This pass is separate from the regular convert-shape-to-standard, despite
    converting between the same dialects, because converting shape constraints
    can happen at a different part of the program than general shape
    computation lowering.
  }];
  let constructor = "mlir::createConvertShapeConstraintsPass()";
  let dependentDialects = ["cf::ControlFlowDialect", "scf::SCFDialect"];
}

//===----------------------------------------------------------------------===//
// SPIRVToLLVM
//===----------------------------------------------------------------------===//

def ConvertSPIRVToLLVMPass : Pass<"convert-spirv-to-llvm", "ModuleOp"> {
  let summary = "Convert SPIR-V dialect to LLVM dialect";
  let description = [{
    See https://mlir.llvm.org/docs/SPIRVToLLVMDialectConversion/
    for more details.
  }];
  let dependentDialects = ["LLVM::LLVMDialect"];

  let options = [
    Option<"useOpaquePointers", "use-opaque-pointers", "bool",
                 /*default=*/"false", "Generate LLVM IR using opaque pointers "
                 "instead of typed pointers">
  ];
}

//===----------------------------------------------------------------------===//
// TensorToLinalg
//===----------------------------------------------------------------------===//

def ConvertTensorToLinalg : Pass<"convert-tensor-to-linalg", "ModuleOp"> {
  let summary = "Convert some Tensor dialect ops to Linalg dialect";
  let constructor = "mlir::createConvertTensorToLinalgPass()";
  let dependentDialects = [
    "arith::ArithDialect",
    "linalg::LinalgDialect",
  ];
}

//===----------------------------------------------------------------------===//
// TensorToSPIRV
//===----------------------------------------------------------------------===//

def ConvertTensorToSPIRV : Pass<"convert-tensor-to-spirv"> {
  let summary = "Convert Tensor dialect to SPIR-V dialect";
  let constructor = "mlir::createConvertTensorToSPIRVPass()";
  let dependentDialects = ["spirv::SPIRVDialect"];
  let options = [
    Option<"emulateLT32BitScalarTypes", "emulate-lt-32-bit-scalar-types",
           "bool", /*default=*/"true",
           "Emulate narrower scalar types with 32-bit ones if not supported by"
           " the target">
  ];
}

//===----------------------------------------------------------------------===//
// TosaToArith
//===----------------------------------------------------------------------===//

def TosaToArith : Pass<"tosa-to-arith"> {
  let summary = "Lower TOSA to the Arith dialect";
  let dependentDialects = [
    "arith::ArithDialect",
  ];
  let description = [{
    Pass that converts TOSA operations to the equivalent operations using the
    operations in the Arith dialect. The ApplyScale operator is optionally
    included as it is often preserved until the final invocation.
  }];

  let options = [
    Option<"includeApplyRescale", "include-apply-rescale",
           "bool", /*default=*/"false",
           "Whether to include the lowering for tosa.apply_rescale to arith">,
    Option<"use32Bit", "use-32-bit",
           "bool", /*default=*/"false",
           "Whether to prioritze lowering to 32-bit operations">
  ];

  let constructor = "tosa::createTosaToArith()";
}

//===----------------------------------------------------------------------===//
// TosaToLinalg
//===----------------------------------------------------------------------===//

def TosaToLinalg
    : InterfacePass<"tosa-to-linalg", "FunctionOpInterface"> {
  let summary = "Lower TOSA to LinAlg on tensors";
  let description = [{
    Pass that converts TOSA operations to the equivalent operations using the
    tensor operations in LinAlg.
  }];

  let constructor = "tosa::createTosaToLinalg()";
}

//===----------------------------------------------------------------------===//
// TosaToLinalgNamed
//===----------------------------------------------------------------------===//

def TosaToLinalgNamed
    : InterfacePass<"tosa-to-linalg-named", "FunctionOpInterface"> {
  let summary = "Lower TOSA to LinAlg named operations";
  let description = [{
    Pass that converts TOSA operations to the equivalent operations using the
    Linalg named operations.
  }];

  let constructor = "tosa::createTosaToLinalgNamed()";
}

//===----------------------------------------------------------------------===//
// TosaToSCF
//===----------------------------------------------------------------------===//

def TosaToSCF : Pass<"tosa-to-scf"> {
  let summary = "Lower TOSA to the SCF dialect";
  let dependentDialects = ["tensor::TensorDialect, scf::SCFDialect"];
  let description = [{
    Pass that converts TOSA's control flow operations to the equivalent SCF
    operations.
  }];

  let constructor = "tosa::createTosaToSCF()";
}

//===----------------------------------------------------------------------===//
// TosaToTensor
//===----------------------------------------------------------------------===//

def TosaToTensor : Pass<"tosa-to-tensor"> {
  let summary = "Lower TOSA to the Tensor dialect";
  let dependentDialects = [
    "tensor::TensorDialect",
  ];
  let description = [{
    Pass that converts TOSA operations to the equivalent operations using the
    operations in the Tensor dialect.
  }];

  let constructor = "tosa::createTosaToTensor()";
}

//===----------------------------------------------------------------------===//
// VectorToGPU
//===----------------------------------------------------------------------===//

def ConvertVectorToGPU : Pass<"convert-vector-to-gpu"> {
  let summary = "Lower the operations from the vector dialect into the GPU "
                "dialect";
  let constructor = "mlir::createConvertVectorToGPUPass()";
  let dependentDialects = [
    "memref::MemRefDialect", "gpu::GPUDialect", "AffineDialect",
    "vector::VectorDialect", "nvgpu::NVGPUDialect"
  ];

  let options = [
    Option<"useNvGpu", "use-nvgpu", "bool", /*default=*/"false",
      "convert to NvGPU ops instead of GPU dialect ops">
  ];
}

//===----------------------------------------------------------------------===//
// VectorToSCF
//===----------------------------------------------------------------------===//

def ConvertVectorToSCF : Pass<"convert-vector-to-scf"> {
  let summary = "Lower the operations from the vector dialect into the SCF "
                "dialect";
  let constructor = "mlir::createConvertVectorToSCFPass()";
  let dependentDialects = [
    "AffineDialect",
    "memref::MemRefDialect",
    "scf::SCFDialect",
    "tensor::TensorDialect"
  ];
  let options = [
    Option<"fullUnroll", "full-unroll", "bool", /*default=*/"false",
           "Perform full unrolling when converting vector transfers to SCF">,
    Option<"targetRank", "target-rank", "unsigned", /*default=*/"1",
           "Target vector rank to which transfer ops should be lowered">,
    Option<"lowerTensors", "lower-tensors", "bool", /*default=*/"false",
           "Lower transfer ops that operate on tensors">
  ];
}

//===----------------------------------------------------------------------===//
// VectorToLLVM
//===----------------------------------------------------------------------===//

def ConvertVectorToLLVMPass : Pass<"convert-vector-to-llvm", "ModuleOp"> {
  let summary = "Lower the operations from the vector dialect into the LLVM "
                "dialect";
  let description = [{

    Convert operations from the vector dialect into the LLVM IR dialect
    operations. The lowering pass provides several options to control
    the kinds of optimizations that are allowed. It also provides options
    that enable the use of one or more architectural-specific dialects
    (AMX, X86Vector, ArmNeon, ArmSVE, etc.) in combination with the
    architectural-neutral vector dialect lowering.

  }];
  // Override explicitly in C++ to allow conditional dialect dependence.
  // let dependentDialects;
  let options = [
    Option<"reassociateFPReductions", "reassociate-fp-reductions",
           "bool", /*default=*/"false",
           "Allows llvm to reassociate floating-point reductions for speed">,
    Option<"force32BitVectorIndices", "force-32bit-vector-indices",
           "bool", /*default=*/"true",
           "Allows compiler to assume vector indices fit in 32-bit if that "
     "yields faster code">,
    Option<"amx", "enable-amx",
           "bool", /*default=*/"false",
           "Enables the use of AMX dialect while lowering the vector "
	   "dialect.">,
    Option<"armNeon", "enable-arm-neon",
           "bool", /*default=*/"false",
           "Enables the use of ArmNeon dialect while lowering the vector "
	   "dialect.">,
    Option<"armSVE", "enable-arm-sve",
           "bool", /*default=*/"false",
           "Enables the use of ArmSVE dialect while lowering the vector "
       "dialect.">,
    Option<"x86Vector", "enable-x86vector",
           "bool", /*default=*/"false",
           "Enables the use of X86Vector dialect while lowering the vector "
	   "dialect.">,
	  Option<"useOpaquePointers", "use-opaque-pointers", "bool",
               /*default=*/"false", "Generate LLVM IR using opaque pointers "
               "instead of typed pointers">
  ];
}

//===----------------------------------------------------------------------===//
// VectorToSPIRV
//===----------------------------------------------------------------------===//

def ConvertVectorToSPIRV : Pass<"convert-vector-to-spirv"> {
  let summary = "Convert Vector dialect to SPIR-V dialect";
  let constructor = "mlir::createConvertVectorToSPIRVPass()";
  let dependentDialects = ["spirv::SPIRVDialect"];
}

#endif // MLIR_CONVERSION_PASSES
