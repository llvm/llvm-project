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
include "mlir/Dialect/Vector/Transforms/VectorTransformsBase.td"

//===----------------------------------------------------------------------===//
// ToEmitC
//===----------------------------------------------------------------------===//

def ConvertToEmitC : Pass<"convert-to-emitc"> {
  let summary = "Convert to EmitC dialect via dialect interfaces";
  let description = [{
    This is a generic pass to convert to the EmitC dialect, it uses the
    `ConvertToEmitCPatternInterface` dialect interface to delegate to dialects
    the injection of conversion patterns.
  }];
  let options = [
    ListOption<"filterDialects", "filter-dialects", "std::string",
               "Test conversion patterns of only the specified dialects">,
  ];
}

//===----------------------------------------------------------------------===//
// ToLLVM
//===----------------------------------------------------------------------===//

def ConvertToLLVMPass : Pass<"convert-to-llvm"> {
  let summary = "Convert to LLVM via dialect interfaces found in the input IR";
  let description = [{
    This is a generic pass to convert to LLVM, it uses the
    `ConvertToLLVMPatternInterface` dialect interface to delegate to dialects
    the injection of conversion patterns.

    If `dynamic` is set to `true`, the pass will look for
    `ConvertToLLVMAttrInterface` attributes and use them to further configure
    the conversion process. This option also uses the `DataLayoutAnalysis`
    analysis to configure the type converter. Enabling this option incurs in
    extra overhead.
  }];

  let options = [
    ListOption<"filterDialects", "filter-dialects", "std::string",
               "Test conversion patterns of only the specified dialects">,
    Option<"useDynamic", "dynamic", "bool", "false",
           "Use op conversion attributes to configure the conversion">,
    Option<"allowPatternRollback", "allow-pattern-rollback", "bool", "true",
           "Experimental performance flag to disallow pattern rollback">
  ];
}

//===----------------------------------------------------------------------===//
// LowerAffine
//===----------------------------------------------------------------------===//

def LowerAffinePass : Pass<"lower-affine"> {
  let summary = "Lower Affine operations to a combination of Arith and SCF "
                "operations";
  let description = [{

    Convert operations from the affine dialect into operations from the SCF and
    standard dialects.

    `affine.for` operations are converted to `scf.for` operations that are free
    of certain structural restrictions (on their bounds and step). `affine.if`
    is similarly converted to the `scf.if` operation. `affine.apply` operations
    are converted into sequences of primitive arithmetic operations from the
    arith dialect that have the same effect, using operands of the `index`
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
  let dependentDialects = ["arith::ArithDialect", "memref::MemRefDialect",
                           "scf::SCFDialect", "vector::VectorDialect"];
}

//===----------------------------------------------------------------------===//
// AMDGPUToROCDL
//===----------------------------------------------------------------------===//

def ConvertAMDGPUToROCDLPass : Pass<"convert-amdgpu-to-rocdl"> {
  let summary = "Convert AMDGPU dialect to ROCDL dialect";
  let description = [{
    This pass converts supported AMDGPU ops to ROCDL dialect intrinsics.
  }];
  let dependentDialects = [
    "LLVM::LLVMDialect",
    "ROCDL::ROCDLDialect",
  ];
  let options = [Option<"chipset", "chipset", "std::string",
                        /*default=*/"\"gfx000\"",
                        "Chipset that these operations will run on">];
}

//===----------------------------------------------------------------------===//
// ArithToAMDGPU
//===----------------------------------------------------------------------===//
def ArithToAMDGPUConversionPass : Pass<"convert-arith-to-amdgpu"> {
  let summary = "Convert Arith operations to AMDGPU-specific implementations";
  let description = [{
    Convert `arith` operations (currently extf and truncf on 8-bit floats)
    to operations in the `amdgpu` dialect. This pass is done in two steps
    in order to avoid running a notional arith-to-rocdl and arith-to-llvm
    simultaniously.
  }];

  let dependentDialects = ["amdgpu::AMDGPUDialect", "vector::VectorDialect"];

  let options = [
    Option<"chipset", "chipset", "std::string",
                        /*default=*/"\"gfx000\"",
                        "Chipset that these operations will run on">,
    Option<"saturateFP8Truncf", "saturate-fp8-truncf", "bool",
           /*default=*/"false",
           "Use saturating truncation for 8-bit float types">,
    Option<"allowPackedF16Rtz", "allow-packed-f16-round-to-zero", "bool",
           /*default=*/"false",
           "Whether we should allow f32->f16 packed round-to-zero conversion">,
  ];
}

//===----------------------------------------------------------------------===//
// ArithToEmitC
//===----------------------------------------------------------------------===//

def ConvertArithToEmitC : Pass<"convert-arith-to-emitc"> {
  let summary = "Convert Arith dialect to EmitC dialect";
  let dependentDialects = ["emitc::EmitCDialect"];
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

def ConvertArithToSPIRVPass : Pass<"convert-arith-to-spirv"> {
  let summary = "Convert Arith dialect to SPIR-V dialect";
  let dependentDialects = ["spirv::SPIRVDialect"];
  let options = [
    Option<"emulateLT32BitScalarTypes", "emulate-lt-32-bit-scalar-types",
           "bool", /*default=*/"true",
           "Emulate narrower scalar types with 32-bit ones if not supported by "
           "the target">,
    Option<"emulateUnsupportedFloatTypes", "emulate-unsupported-float-types",
           "bool", /*default=*/"true",
           "Emulate unsupported float types by representing them with integer "
           "types of same bit width">
  ];
}

//===----------------------------------------------------------------------===//
// ArithToArmSME
//===----------------------------------------------------------------------===//

def ArithToArmSMEConversionPass : Pass<"convert-arith-to-arm-sme"> {
  let summary = "Convert Arith dialect to ArmSME dialect";
  let dependentDialects = ["arm_sme::ArmSMEDialect"];
}

//===----------------------------------------------------------------------===//
// ArmNeon2dToIntr
//===----------------------------------------------------------------------===//

def ConvertArmNeon2dToIntrPass : Pass<"arm-neon-2d-to-intr"> {
  let summary = "Convert Arm NEON structured ops to intrinsics";
  let description = [{
    Creates a pass to lower Arm NEON 2D ops to intrinsics, i.e.
    equivalent ops operating on flattened 1D vectors and mapping more
    directly to the corresponding Arm NEON instruction.
  }];
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
}

//===----------------------------------------------------------------------===//
// BufferizationToMemRef
//===----------------------------------------------------------------------===//

def ConvertBufferizationToMemRefPass : Pass<"convert-bufferization-to-memref"> {
  let summary = "Convert operations from the Bufferization dialect to the "
                "MemRef dialect";
  let description = [{

    This pass converts bufferization operations into memref operations.

    In the current state, this pass only transforms a `bufferization.clone`
    operation into `memref.alloc` and `memref.copy` operations and
    `bufferization.dealloc` operations (the same way as the
    `-bufferization-lower-deallocations` pass). The conversion of `clone`
    operations is needed, since some clone operations could remain after
    applying several transformation processes. Currently, only `canonicalize`
    transforms clone operations or even eliminates them. This can lead to errors
    if any clone op survived after all conversion passes (starting from the
    bufferization dialect) are performed.

    See:
    https://llvm.discourse.group/t/bufferization-error-related-to-memref-clone/4665

    To avoid these errors, this pass can be performed as a last clean-up pass to
    transform remaining operations and to proceed in other dialects (memref
    e.g.).

    Note that this pass only transforms the operation without any further
    analyses. This pass does not consider any memory analysis or optimization
    and hence does not resolve any memory leaks.

  }];
  let dependentDialects = [
    "arith::ArithDialect", "memref::MemRefDialect", "scf::SCFDialect",
    "func::FuncDialect"
  ];
}

//===----------------------------------------------------------------------===//
// ComplexToLLVM
//===----------------------------------------------------------------------===//

def ConvertComplexToLLVMPass : Pass<"convert-complex-to-llvm"> {
  let summary = "Convert Complex dialect to LLVM dialect";
  let dependentDialects = ["LLVM::LLVMDialect"];

  let options = [
    Option<"complexRange", "complex-range", "::mlir::complex::ComplexRangeFlags",
      /*default=*/"::mlir::complex::ComplexRangeFlags::basic",
      "Control the intermediate calculation of complex number division",
      [{::llvm::cl::values(
        clEnumValN(::mlir::complex::ComplexRangeFlags::improved, "improved", "improved"),
        clEnumValN(::mlir::complex::ComplexRangeFlags::basic, "basic", "basic (default)"),
        clEnumValN(::mlir::complex::ComplexRangeFlags::none, "none", "none")
      )}]>,
  ];
}

//===----------------------------------------------------------------------===//
// ComplexToLibm
//===----------------------------------------------------------------------===//

def ConvertComplexToLibm : Pass<"convert-complex-to-libm", "ModuleOp"> {
  let summary = "Convert Complex dialect to libm calls";
  let description = [{
    This pass converts supported Complex ops to libm calls.
  }];
  let dependentDialects = ["func::FuncDialect"];
}

//===----------------------------------------------------------------------===//
// ComplexToROCDLLibraryCalls
//===----------------------------------------------------------------------===//

def ConvertComplexToROCDLLibraryCalls : Pass<"convert-complex-to-rocdl-library-calls", "ModuleOp"> {
  let summary = "Convert Complex dialect to ROCDL library calls";
  let description = [{
    This pass converts supported Complex ops to calls to the AMD device library.
  }];
  let dependentDialects = ["func::FuncDialect"];
}

//===----------------------------------------------------------------------===//
// ComplexToSPIRV
//===----------------------------------------------------------------------===//

def ConvertComplexToSPIRVPass : Pass<"convert-complex-to-spirv"> {
  let summary = "Convert Complex dialect to SPIRV dialect";
  let dependentDialects = ["spirv::SPIRVDialect"];
}

//===----------------------------------------------------------------------===//
// ComplexToStandard
//===----------------------------------------------------------------------===//

def ConvertComplexToStandardPass : Pass<"convert-complex-to-standard"> {
  let summary = "Convert Complex dialect to standard dialect";
  let dependentDialects = ["math::MathDialect"];

  let options = [
    Option<"complexRange", "complex-range", "::mlir::complex::ComplexRangeFlags",
      /*default=*/"::mlir::complex::ComplexRangeFlags::improved",
      "Control the intermediate calculation of complex number division",
      [{::llvm::cl::values(
        clEnumValN(::mlir::complex::ComplexRangeFlags::improved, "improved", "improved (default)"),
        clEnumValN(::mlir::complex::ComplexRangeFlags::basic, "basic", "basic"),
        clEnumValN(::mlir::complex::ComplexRangeFlags::none, "none", "none")
      )}]>,
  ];
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
           "Bitwidth of the index type, 0 to use size of machine word">
  ];
}

//===----------------------------------------------------------------------===//
// ControlFlowToSCF
//===----------------------------------------------------------------------===//

def LiftControlFlowToSCFPass : Pass<"lift-cf-to-scf"> {
  let summary = "Lift ControlFlow dialect to SCF dialect";
  let description = [{
    Lifts ControlFlow operations to SCF dialect operations.

    This pass is prefixed with "lift" instead of "convert" as it is not always
    guaranteed to replace all ControlFlow ops.
    If a region contains only a single kind of return-like operation, all
    ControlFlow operations will be replaced successfully.
    Otherwise a single ControlFlow switch branching to one block per return-like
    operation kind remains.

    This pass may need to create unreachable terminators in case of infinite
    loops, which is only supported for 'func.func' for now. If you potentially
    have infinite loops inside CFG regions not belonging to 'func.func',
    consider using `transformCFGToSCF` function directly with corresponding
    `CFGToSCFInterface::createUnreachableTerminator` implementation.
  }];

  let dependentDialects = ["scf::SCFDialect",
                           "arith::ArithDialect",
                           "ub::UBDialect",
                           // TODO: This is only necessary until we have a
                           //       ub.unreachable op.
                           "func::FuncDialect"];
}

//===----------------------------------------------------------------------===//
// ControlFlowToSPIRV
//===----------------------------------------------------------------------===//

def ConvertControlFlowToSPIRVPass : Pass<"convert-cf-to-spirv"> {
  let summary = "Convert ControlFlow dialect to SPIR-V dialect";
  let dependentDialects = ["spirv::SPIRVDialect"];
  let options = [
    Option<"emulateLT32BitScalarTypes", "emulate-lt-32-bit-scalar-types",
           "bool", /*default=*/"true",
           "Emulate narrower scalar types with 32-bit ones if not supported by"
           " the target">,
    Option<"emulateUnsupportedFloatTypes", "emulate-unsupported-float-types",
           "bool", /*default=*/"true",
           "Emulate unsupported float types by representing them with integer "
           "types of same bit width">
  ];
}

//===----------------------------------------------------------------------===//
// FuncToEmitC
//===----------------------------------------------------------------------===//

def ConvertFuncToEmitC : Pass<"convert-func-to-emitc", "ModuleOp"> {
  let summary = "Convert Func dialect to EmitC dialect";
  let dependentDialects = ["emitc::EmitCDialect"];
}

//===----------------------------------------------------------------------===//
// FuncToLLVM
//===----------------------------------------------------------------------===//

def SetLLVMModuleDataLayoutPass : Pass<"set-llvm-module-datalayout", "ModuleOp"> {
  let summary = "Attach a datalayout string as a module attribute";
  let description = [{
    Verify that the dataLayout string is a valid LLVM datalayout string and
    attach it as an attribute `LLVMDialect::getDataLayoutAttrName()` to the
    module, overriding the existing one.
  }];
  let options = [
    Option<"dataLayout", "data-layout", "std::string",
           /*default=*/"\"\"",
           "String description (LLVM format) of the data layout that is "
           "expected on the produced module">,
  ];
}

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

    An LLVM datalayout string can be attached as an attribute to the module on
    which the pass anchors. Such an attribute is attached by calling the
    set-module-datalayout pass. If present, an llvm::DataLayout object is
    created from this attribute and used in the conversion to LLVM.

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
  ];
}

//===----------------------------------------------------------------------===//
// FuncToSPIRV
//===----------------------------------------------------------------------===//

def ConvertFuncToSPIRVPass : Pass<"convert-func-to-spirv"> {
  let summary = "Convert Func dialect to SPIR-V dialect";
  let dependentDialects = ["spirv::SPIRVDialect"];
  let options = [
    Option<"emulateLT32BitScalarTypes", "emulate-lt-32-bit-scalar-types",
           "bool", /*default=*/"true",
           "Emulate narrower scalar types with 32-bit ones if not supported by"
           " the target">,
    Option<"emulateUnsupportedFloatTypes", "emulate-unsupported-float-types",
           "bool", /*default=*/"true",
           "Emulate unsupported float types by representing them with integer "
           "types of same bit width">
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
    Option<"hostBarePtrCallConv", "use-bare-pointers-for-host", "bool",
           /*default=*/"false",
           "Use bare pointers to pass memref arguments to host functions. "
           "All memrefs must have static shape.">,
    Option<"kernelBarePtrCallConv", "use-bare-pointers-for-kernels", "bool",
           /*default=*/"false",
             "Use bare pointers to pass memref arguments to kernels. "
             "The kernel must use the same setting for this option."
          >,
    Option<"kernelIntersperseSizeCallConv", "intersperse-sizes-for-kernels", "bool",
           /*default=*/"false",
           "Inserts a size_t argument following each memref argument, "
           "containing the static size in bytes of the buffer. Incompatible "
           "arguments are rejected. This is intended for use by the Vulkan "
           "runtime with the kernel bare pointer calling convention, to enable "
           "dynamic binding of buffers as arguments without static type info."
          >
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

  let dependentDialects = ["LLVM::LLVMDialect"];
}

//===----------------------------------------------------------------------===//
// GPUToLLVMSPV
//===----------------------------------------------------------------------===//

def ConvertGpuOpsToLLVMSPVOps : Pass<"convert-gpu-to-llvm-spv", "gpu::GPUModuleOp"> {
  let summary =
    "Generate LLVM operations to be ingested by a SPIR-V backend for gpu operations";
  let dependentDialects = ["LLVM::LLVMDialect"];
  let options = [
    Option<"use64bitIndex", "use-64bit-index",
           "bool", /*default=*/"false",
           "Use 64-bit integers to convert index types">,
  ];
}

//===----------------------------------------------------------------------===//
// GPUToNVVM
//===----------------------------------------------------------------------===//

def ConvertGpuOpsToNVVMOps : Pass<"convert-gpu-to-nvvm", "gpu::GPUModuleOp"> {
  let summary = "Generate NVVM operations for gpu operations";
  let dependentDialects = [
    "cf::ControlFlowDialect",
    "memref::MemRefDialect",
    "NVVM::NVVMDialect",
  ];
  let options = [
    Option<"indexBitwidth", "index-bitwidth", "unsigned",
           /*default=kDeriveIndexBitwidthFromDataLayout*/ "0",
           "Bitwidth of the index type, 0 to use size of machine word">,
    Option<"hasRedux", "has-redux", "bool", /*default=*/"false",
           "Target gpu supports redux">,
    Option<"useBarePtrCallConv", "use-bare-ptr-memref-call-conv", "bool",
           /*default=*/"false",
           "Replace memref arguments in GPU functions with bare pointers. "
           "All memrefs must have static shape.">,
    ListOption<"allowedDialects", "allowed-dialects", "std::string",
               "Run conversion patterns of only the specified dialects">,
  ];
}

//===----------------------------------------------------------------------===//
// GPUToROCDL
//===----------------------------------------------------------------------===//

def ConvertGpuOpsToROCDLOps : Pass<"convert-gpu-to-rocdl", "gpu::GPUModuleOp"> {
  let summary = "Generate ROCDL operations for gpu operations";
  let dependentDialects = [
    "ROCDL::ROCDLDialect",
    "amdgpu::AMDGPUDialect",
    "cf::ControlFlowDialect",
    "memref::MemRefDialect",
  ];
  let options = [
    Option<"chipset", "chipset", "std::string",
           /*default=*/"\"gfx000\"",
           "Chipset that these operations will run on">,
    Option<"indexBitwidth", "index-bitwidth", "unsigned",
           /*default=kDeriveIndexBitwidthFromDataLayout*/ "0",
           "Bitwidth of the index type, 0 to use size of machine word">,
    Option<"useBarePtrCallConv", "use-bare-ptr-memref-call-conv", "bool",
           /*default=*/"false",
           "Replace memref arguments in GPU functions with bare pointers."
           "All memrefs must have static shape">,
    Option<"runtime", "runtime", "::mlir::gpu::amd::Runtime",
           "::mlir::gpu::amd::Runtime::Unknown",
           "Runtime code will be run on (default is Unknown, can also use HIP "
           "or OpenCL)",
           [{::llvm::cl::values(
               clEnumValN(::mlir::gpu::amd::Runtime::Unknown, "unknown",
                          "Unknown (default)"),
               clEnumValN(::mlir::gpu::amd::Runtime::HIP, "HIP", "HIP"),
               clEnumValN(::mlir::gpu::amd::Runtime::OpenCL, "OpenCL",
                          "OpenCL"))}]>,
    ListOption<"allowedDialects", "allowed-dialects", "std::string",
               "Run conversion patterns of only the specified dialects">,
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
  let dependentDialects = [
    "func::FuncDialect",
    "spirv::SPIRVDialect",
  ];
  let options = [
    Option<"use64bitIndex", "use-64bit-index",
           "bool", /*default=*/"false",
           "Use 64-bit integers to convert index types">,
  ];
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
// ConvertIndexToSPIRVPass
//===----------------------------------------------------------------------===//

def ConvertIndexToSPIRVPass : Pass<"convert-index-to-spirv"> {
  let summary = "Lower the `index` dialect to the `spirv` dialect.";
  let description = [{
    This pass lowers Index dialect operations to SPIR-V dialect operations.
    Operation conversions are 1-to-1 except for the exotic divides: `ceildivs`,
    `ceildivu`, and `floordivs`. The index bitwidth will be 32 or 64 as
    specified by use-64bit-index.
  }];

  let dependentDialects = ["::mlir::spirv::SPIRVDialect"];

  let options = [
    Option<"use64bitIndex", "use-64bit-index",
           "bool", /*default=*/"false",
           "Use 64-bit integers to convert index types">
  ];
}

//===----------------------------------------------------------------------===//
// LinalgToStandard
//===----------------------------------------------------------------------===//

def ConvertLinalgToStandardPass : Pass<"convert-linalg-to-std", "ModuleOp"> {
  let summary = "Convert the operations from the linalg dialect into the "
                "Standard dialect";
  let dependentDialects = ["func::FuncDialect", "memref::MemRefDialect"];
}

//===----------------------------------------------------------------------===//
// MathToLibm
//===----------------------------------------------------------------------===//

def ConvertMathToLibmPass : Pass<"convert-math-to-libm", "ModuleOp"> {
  let summary = "Convert Math dialect to libm calls";
  let description = [{
    This pass converts supported Math ops to libm calls.
  }];
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
// MathToLibm
//===----------------------------------------------------------------------===//

def ConvertMathToROCDL : Pass<"convert-math-to-rocdl", "ModuleOp"> {
  let summary = "Convert Math dialect to ROCDL library calls";
  let description = [{
    This pass converts supported Math ops to ROCDL library calls.

    The chipset option specifies the target AMDGPU architecture. If the chipset
    is empty, none of the chipset-dependent patterns are added and the pass
    will not attempt to parse the chipset.
  }];
  let dependentDialects = [
    "arith::ArithDialect",
    "func::FuncDialect",
    "ROCDL::ROCDLDialect",
    "vector::VectorDialect",
  ];
  let options = [Option<"chipset", "chipset", "std::string",
                        /*default=*/"\"gfx000\"",
                        "Chipset that these operations will run on">];
}

//===----------------------------------------------------------------------===//
// MathToSPIRV
//===----------------------------------------------------------------------===//

def ConvertMathToSPIRVPass : Pass<"convert-math-to-spirv"> {
  let summary = "Convert Math dialect to SPIR-V dialect";
  let dependentDialects = ["spirv::SPIRVDialect"];
}

//===----------------------------------------------------------------------===//
// MathToXeVM
//===----------------------------------------------------------------------===//

def ConvertMathToXeVM : Pass<"convert-math-to-xevm", "ModuleOp"> {
  let summary =
      "Convert (fast) math operations to native XeVM/SPIRV equivalents";
  let description = [{
    This pass converts supported math ops marked with the `afn` fastmath flag
    to function calls for OpenCL `native_` math intrinsics: These intrinsics
    are typically mapped directly to native device instructions, often resulting
    in better performance. However, the precision/error of these intrinsics
    are implementation-defined, and thus math ops are only converted when they
    have the `afn` fastmath flag enabled.
  }];
  let options = [Option<
      "convertArith", "convert-arith", "bool", /*default=*/"true",
      "Convert supported Arith ops (e.g. arith.divf) as well.">];
  let dependentDialects = [
    "arith::ArithDialect",
    "xevm::XeVMDialect",
    "LLVM::LLVMDialect",
  ];
}

//===----------------------------------------------------------------------===//
// MathToEmitC
//===----------------------------------------------------------------------===//

def ConvertMathToEmitC  : Pass<"convert-math-to-emitc"> {
  let summary = "Convert some Math operations to EmitC call_opaque operations";
  let description = [{
    This pass converts supported Math ops to `call_opaque` ops targeting libc/libm
    functions. Unlike convert-math-to-funcs pass, converting to `call_opaque` ops
    allows to overload the same function with different argument types.
  }];
  let dependentDialects = ["emitc::EmitCDialect"];
  let options = [
    Option<"languageTarget", "language-target", "::mlir::emitc::LanguageTarget",
      /*default=*/"::mlir::emitc::LanguageTarget::c99", "Select the language standard target for callees (c99 or cpp11).",
      [{::llvm::cl::values(
            clEnumValN(::mlir::emitc::LanguageTarget::c99, "c99", "c99"),
            clEnumValN(::mlir::emitc::LanguageTarget::cpp11, "cpp11", "cpp11")
          )}]>
  ];
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
    "scf::SCFDialect",
    "vector::VectorDialect",
    "LLVM::LLVMDialect",
  ];
  let options = [
    Option<"minWidthOfFPowIExponent", "min-width-of-fpowi-exponent", "unsigned",
           /*default=*/"1",
           "Convert FPowI only if the width of its exponent's integer type "
           "is greater than or equal to this value">,
    // Most backend targets support a native ctlz operation, so by default
    // ctrlz conversion is disabled.
    Option<"convertCtlz", "convert-ctlz", "bool", /*default=*/"false",
           "Convert math.ctlz to a software implementation. Enable "
           "for targets that do not natively support ctlz.">,
  ];
}

//===----------------------------------------------------------------------===//
// MemRefToEmitC
//===----------------------------------------------------------------------===//

def ConvertMemRefToEmitC : Pass<"convert-memref-to-emitc", "ModuleOp"> {
  let summary = "Convert MemRef dialect to EmitC dialect";
  let dependentDialects = ["emitc::EmitCDialect"];
  let options = [Option<
      "lowerToCpp", "lower-to-cpp", "bool",
      /*default=*/"false",
      /*description=*/"Target C++ (true) instead of C (false)">];
}

//===----------------------------------------------------------------------===//
// MemRefToLLVM
//===----------------------------------------------------------------------===//

def FinalizeMemRefToLLVMConversionPass :
    Pass<"finalize-memref-to-llvm", "ModuleOp"> {
  let summary = "Finalize MemRef dialect to LLVM dialect conversion";
  let description = [{
    Finalize the conversion of the operations from the MemRef
    dialect to the LLVM dialect.
    This conversion will not convert some complex MemRef
    operations. Make sure to run `expand-strided-metadata`
    beforehand for these.
  }];
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
           "classic 'malloc', 'aligned_alloc' and 'free' functions">
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

def ConvertMemRefToSPIRVPass : Pass<"convert-memref-to-spirv"> {
  let summary = "Convert MemRef dialect to SPIR-V dialect";
  let dependentDialects = ["spirv::SPIRVDialect"];
  let options = [
    Option<"boolNumBits", "bool-num-bits",
           "int", /*default=*/"8",
           "The number of bits to store a boolean value">,
    Option<"use64bitIndex", "use-64bit-index",
           "bool", /*default=*/"false",
           "Use 64-bit integers to convert index types">
  ];
}

//===----------------------------------------------------------------------===//
// ShardToMPI
//===----------------------------------------------------------------------===//

def ConvertShardToMPIPass : Pass<"convert-shard-to-mpi"> {
  let summary = "Convert Shard dialect to MPI dialect.";
  let description = [{
    This pass converts communication operations from the Shard dialect to the
    MPI dialect.
    If it finds the DLTI attribute "MPI:comm_world-rank" on the module it will
    use that integer value instead of calling MPI_Comm_rank. This allows
    optimizations like constant shape propagation and fusion because
    shard/partition sizes depend on the rank.
  }];
  let dependentDialects = [
    "affine::AffineDialect",
    "arith::ArithDialect",
    "memref::MemRefDialect",
    "mpi::MPIDialect",
    "scf::SCFDialect",
    "bufferization::BufferizationDialect"
  ];
}

//===----------------------------------------------------------------------===//
// NVVMToLLVM
//===----------------------------------------------------------------------===//

def ConvertNVVMToLLVMPass : Pass<"convert-nvvm-to-llvm"> {
  let summary = "Convert NVVM to PTX with Inline Assembly in LLVM dialect";
  let description = [{
    This pass generates PTX instructions using inline assembly for NVVM
    operations implements `BasicPtxBuilderInterface`.
  }];
  let dependentDialects = [
    "NVVM::NVVMDialect",
  ];
}

//===----------------------------------------------------------------------===//
// NVGPUToNVVM
//===----------------------------------------------------------------------===//

def ConvertNVGPUToNVVMPass : Pass<"convert-nvgpu-to-nvvm"> {
  let summary = "Convert NVGPU dialect to NVVM dialect";
  let description = [{
    This pass converts supported NVGPU ops to NVVM dialect intrinsics.
  }];

  let dependentDialects = [
    "arith::ArithDialect",
    "LLVM::LLVMDialect",
    "memref::MemRefDialect",
    "NVVM::NVVMDialect"
  ];
}


//===----------------------------------------------------------------------===//
// OpenACCToSCF
//===----------------------------------------------------------------------===//

def ConvertOpenACCToSCFPass : Pass<"convert-openacc-to-scf", "ModuleOp"> {
  let summary = "Convert the OpenACC ops to OpenACC with SCF dialect";
  let dependentDialects = ["scf::SCFDialect", "acc::OpenACCDialect"];
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

def ConvertPDLToPDLInterpPass : Pass<"convert-pdl-to-pdl-interp", "ModuleOp"> {
  let summary = "Convert PDL ops to PDL interpreter ops";
  let dependentDialects = ["pdl_interp::PDLInterpDialect"];
}

//===----------------------------------------------------------------------===//
// ReconcileUnrealizedCasts
//===----------------------------------------------------------------------===//

def ReconcileUnrealizedCastsPass : Pass<"reconcile-unrealized-casts"> {
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
}

//===----------------------------------------------------------------------===//
// SCFToControlFlow
//===----------------------------------------------------------------------===//

def SCFToControlFlowPass : Pass<"convert-scf-to-cf"> {
  let summary = "Convert SCF dialect to ControlFlow dialect, replacing structured"
                " control flow with a CFG";
  let dependentDialects = ["cf::ControlFlowDialect"];
}

//===----------------------------------------------------------------------===//
// SCFToOpenMP
//===----------------------------------------------------------------------===//

def ConvertSCFToOpenMPPass : Pass<"convert-scf-to-openmp", "ModuleOp"> {
  let summary = "Convert SCF parallel loop to OpenMP parallel + workshare "
                "constructs.";

  let options = [
    Option<"numThreads", "num-threads", "unsigned",
           /*default=kUseOpenMPDefaultNumThreads*/"0",
           "Number of threads to use">
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
    Converts SCF ops into SPIR-V structured control flow ops.
    SPIR-V structured control flow ops do not support yielding values.
    So for SCF ops yielding values, SPIR-V variables are created for
    holding the values and load/store operations are emitted for updating
    them.
  }];
  let dependentDialects = ["spirv::SPIRVDialect"];
}

//===----------------------------------------------------------------------===//
// SCFToGPU
//===----------------------------------------------------------------------===//

def ConvertAffineForToGPUPass
    : InterfacePass<"convert-affine-for-to-gpu", "FunctionOpInterface"> {
  let summary = "Convert top-level AffineFor Ops to GPU kernels";
  let dependentDialects = ["gpu::GPUDialect"];
  let options = [
    Option<"numBlockDims", "gpu-block-dims", "unsigned", /*default=*/"1u",
           "Number of GPU block dimensions for mapping">,
    Option<"numThreadDims", "gpu-thread-dims", "unsigned", /*default=*/"1u",
           "Number of GPU thread dimensions for mapping">
  ];
}

def ConvertParallelLoopToGpuPass : Pass<"convert-parallel-loops-to-gpu"> {
  let summary = "Convert mapped scf.parallel ops to gpu launch operations";
  let description = [{
    Creates a pass that converts scf.parallel operations into a gpu.launch
    operation. The mapping of loop dimensions to launch dimensions is derived
    from mapping attributes. See ParallelToGpuLaunchLowering::matchAndRewrite
    for a description of the used attributes.
  }];
  let dependentDialects = ["affine::AffineDialect", "gpu::GPUDialect"];
}

//===----------------------------------------------------------------------===//
// SCFToEmitC
//===----------------------------------------------------------------------===//

def SCFToEmitC : Pass<"convert-scf-to-emitc"> {
  let summary = "Convert SCF dialect to EmitC dialect, maintaining structured"
                " control flow";
  let dependentDialects = ["emitc::EmitCDialect"];
}

//===----------------------------------------------------------------------===//
// ShapeToStandard
//===----------------------------------------------------------------------===//

def ConvertShapeToStandardPass : Pass<"convert-shape-to-std", "ModuleOp"> {
  let summary = "Convert operations from the shape dialect into the standard "
                "dialect";
  let dependentDialects = ["scf::SCFDialect"];
}

def ConvertShapeConstraintsPass : Pass<"convert-shape-constraints"> {
  let summary = "Convert shape constraint operations to the standard dialect";
  let description = [{
    This pass eliminates shape constraints from the program, converting them to
    eager (side-effecting) error handling code.

    This pass is separate from the regular convert-shape-to-standard, despite
    converting between the same dialects, because converting shape constraints
    can happen at a different part of the program than general shape
    computation lowering.
  }];
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
    Option<"clientAPI", "client-api", "::mlir::spirv::ClientAPI",
	   /*default=*/"::mlir::spirv::ClientAPI::Unknown",
	   "Derive StorageClass to address space mapping from the client API",
	   [{::llvm::cl::values(
	     clEnumValN(::mlir::spirv::ClientAPI::Unknown, "Unknown", "Unknown (default)"),
	     clEnumValN(::mlir::spirv::ClientAPI::Metal, "Metal", "Metal"),
	     clEnumValN(::mlir::spirv::ClientAPI::OpenCL, "OpenCL", "OpenCL"),
	     clEnumValN(::mlir::spirv::ClientAPI::Vulkan, "Vulkan", "Vulkan"),
	     clEnumValN(::mlir::spirv::ClientAPI::WebGPU, "WebGPU", "WebGPU")
	   )}]>,
  ];
}

//===----------------------------------------------------------------------===//
// TensorToLinalg
//===----------------------------------------------------------------------===//

def ConvertTensorToLinalgPass : Pass<"convert-tensor-to-linalg", "ModuleOp"> {
  let summary = "Convert some Tensor dialect ops to Linalg dialect";
  let dependentDialects = [
    "arith::ArithDialect",
    "linalg::LinalgDialect",
  ];
}

//===----------------------------------------------------------------------===//
// TensorToSPIRV
//===----------------------------------------------------------------------===//

def ConvertTensorToSPIRVPass : Pass<"convert-tensor-to-spirv"> {
  let summary = "Convert Tensor dialect to SPIR-V dialect";
  let dependentDialects = ["spirv::SPIRVDialect"];
  let options = [
    Option<"emulateLT32BitScalarTypes", "emulate-lt-32-bit-scalar-types",
           "bool", /*default=*/"true",
           "Emulate narrower scalar types with 32-bit ones if not supported by"
           " the target">,
    Option<"emulateUnsupportedFloatTypes", "emulate-unsupported-float-types",
           "bool", /*default=*/"true",
           "Emulate unsupported float types by representing them with integer "
           "types of same bit width">
  ];
}

//===----------------------------------------------------------------------===//
// TosaToArith
//===----------------------------------------------------------------------===//

def TosaToArithPass : Pass<"tosa-to-arith"> {
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
  let options = [
    Option<"disableTosaDecompositions", "disable-tosa-decompositions",
           "bool", /*default=*/"false",
           "Disable tosa decompositions pass">,
    Option<"aggressiveReduceConstant", "aggressive-reduce-constant",
           "bool", /*default=*/"false",
           "Always perform the reduce constant optimization">
  ];
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

  let options = [
      Option<"preferConv2DKernelLayoutHWCF", "prefer-conv2d-kernel-layout-hwcf",
           "bool", /*default=*/"false",
           "Prefer generating linalg.conv_2d_nhwc_hwcf over linalg.conv_2d_nhwc_fhwc">
  ];

  let constructor = "tosa::createTosaToLinalgNamed()";
}

//===----------------------------------------------------------------------===//
// TosaToMLProgram
//===----------------------------------------------------------------------===//

def TosaToMLProgram : Pass<"tosa-to-mlprogram", "ModuleOp"> {
  let summary = "Lower TOSA to the MLProgram dialect";
  let dependentDialects = ["ml_program::MLProgramDialect"];
  let description = [{
    Pass that converts TOSA's variable operator operations to the equivalent
    MLProgram operations.
  }];
}

//===----------------------------------------------------------------------===//
// TosaToSCF
//===----------------------------------------------------------------------===//

def TosaToSCFPass : Pass<"tosa-to-scf"> {
  let summary = "Lower TOSA to the SCF dialect";
  let dependentDialects = ["tensor::TensorDialect, scf::SCFDialect"];
  let description = [{
    Pass that converts TOSA's control flow operations to the equivalent SCF
    operations.
  }];
}

//===----------------------------------------------------------------------===//
// TosaToTensor
//===----------------------------------------------------------------------===//

def TosaToTensorPass : Pass<"tosa-to-tensor"> {
  let summary = "Lower TOSA to the Tensor dialect";
  let dependentDialects = [
    "tensor::TensorDialect",
  ];
  let description = [{
    Pass that converts TOSA operations to the equivalent operations using the
    operations in the Tensor dialect.
  }];
}

//===----------------------------------------------------------------------===//
// UBToLLVM
//===----------------------------------------------------------------------===//

def UBToLLVMConversionPass : Pass<"convert-ub-to-llvm"> {
  let summary = "Convert UB dialect to LLVM dialect";
  let description = [{
    This pass converts supported UB ops to LLVM dialect instructions.
  }];
  let dependentDialects = ["LLVM::LLVMDialect"];
  let options = [
    Option<"indexBitwidth", "index-bitwidth", "unsigned",
           /*default=kDeriveIndexBitwidthFromDataLayout*/"0",
           "Bitwidth of the index type, 0 to use size of machine word">,
  ];
}

//===----------------------------------------------------------------------===//
// UBToSPIRV
//===----------------------------------------------------------------------===//

def UBToSPIRVConversionPass : Pass<"convert-ub-to-spirv"> {
  let summary = "Convert UB dialect to SPIR-V dialect";
  let description = [{
    This pass converts supported UB ops to SPIR-V dialect ops.
  }];
  let dependentDialects = ["spirv::SPIRVDialect"];
}

//===----------------------------------------------------------------------===//
// VectorToGPU
//===----------------------------------------------------------------------===//

def ConvertVectorToGPU : Pass<"convert-vector-to-gpu"> {
  let summary = "Lower the operations from the vector dialect into the GPU "
                "dialect";
  let constructor = "mlir::createConvertVectorToGPUPass()";
  let dependentDialects = [
    "memref::MemRefDialect", "gpu::GPUDialect", "affine::AffineDialect",
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
    "affine::AffineDialect",
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
           "Lower transfer ops that operate on tensors">,
    Option<"lowerScalable", "lower-scalable", "bool", /*default=*/"false",
           "Add scalable vector specific lowerings (that introduce loops)">
  ];
}

//===----------------------------------------------------------------------===//
// VectorToArmSME
//===----------------------------------------------------------------------===//

def ConvertVectorToArmSMEPass : Pass<"convert-vector-to-arm-sme"> {
  let summary = "Lower the operations from the vector dialect into the ArmSME "
                "dialect";
  let description = [{
    Pass that converts vector dialect operations into equivalent ArmSME dialect
    operations.
  }];
  let dependentDialects = ["arm_sme::ArmSMEDialect", "arm_sve::ArmSVEDialect"];
}

//===----------------------------------------------------------------------===//
// ArmSMEToSCF
//===----------------------------------------------------------------------===//

def ConvertArmSMEToSCFPass : Pass<"convert-arm-sme-to-scf"> {
  let summary = "Lower the operations from the ArmSME dialect into the SCF "
                "dialect";
  let dependentDialects = [
    "scf::SCFDialect",
    "arith::ArithDialect",
    "vector::VectorDialect",
    "arm_sme::ArmSMEDialect"
  ];
}

//===----------------------------------------------------------------------===//
// ArmSMEToLLVM
//===----------------------------------------------------------------------===//

def ConvertArmSMEToLLVM : InterfacePass<"convert-arm-sme-to-llvm", "FunctionOpInterface"> {
  let summary = "Lower the operations from the ArmSME dialect into the LLVM "
                "dialect";
  let constructor = "mlir::createConvertArmSMEToLLVMPass()";
  let dependentDialects = [
    "arm_sme::ArmSMEDialect",
    "LLVM::LLVMDialect"
  ];
  let options = [
    Option<"dumpTileLiveRanges", "dump-tile-live-ranges",
           "bool", /*default=*/"false",
           "Dump the live ranges of SME tiles (for debugging)">
  ];
}

//===----------------------------------------------------------------------===//
// VectorToLLVM
//===----------------------------------------------------------------------===//

def ConvertVectorToLLVMPass : Pass<"convert-vector-to-llvm"> {
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
    Option<"useVectorAlignment", "use-vector-alignment",
           "bool", /*default=*/"false",
           "Use the preferred alignment of a vector type in load/store "
           "operations instead of the alignment of the element type of the "
           "memref. This flag is intended for use with hardware which requires"
           "vector alignment, or in application contexts where it is known all "
           "vector access are naturally aligned. If operations have an "
           "alignment attribute set, the alignment attribute takes priority "
           "over this option ">,
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
    Option<"armI8MM", "enable-arm-i8mm",
           "bool", /*default=*/"false",
           "Enables the use of Arm FEAT_I8MM instructions while lowering "
           "the vector dialect.">,
    Option<"armBF16", "enable-arm-bf16",
           "bool", /*default=*/"false",
           "Enables the use of Arm FEAT_BF16 instructions while lowering "
           "the vector dialect.">,
    Option<"x86Vector", "enable-x86vector",
           "bool", /*default=*/"false",
           "Enables the use of X86Vector dialect while lowering the vector "
	   "dialect.">,
    Option<"vectorContractLowering", "vector-contract-lowering",
           "vector::VectorContractLowering",
           /*default=*/"vector::VectorContractLowering::Dot",
           VectorContractLoweringAttr.summary, [{::llvm::cl::values(
           clEnumValN(::mlir::vector::VectorContractLowering::Dot, "dot",
            "Progressively lower to finer grained `vector.contract` and dot-products. (default)"),
           clEnumValN(::mlir::vector::VectorContractLowering::LLVMIntr, "llvmintr",
            "Lower directly to `llvm.intr.matrix.multiply`."),
           clEnumValN(::mlir::vector::VectorContractLowering::OuterProduct, "outerproduct",
            "Lower to `vector.outerproduct`."),
           clEnumValN(::mlir::vector::VectorContractLowering::ParallelArith, "parallelarith",
            "Lower contract with all reduction dimensions unrolled to 1 to a vector elementwise operations.")
	        )}]>,
    Option<"vectorTransposeLowering", "vector-transpose-lowering",
           "vector::VectorTransposeLowering",
           /*default=*/"vector::VectorTransposeLowering::EltWise",
           VectorTransposeLoweringAttr.summary, [{::llvm::cl::values(
           clEnumValN(::mlir::vector::VectorTransposeLowering::EltWise, "eltwise",
            "Lower transpose into element-wise extract and inserts (default)"),
           clEnumValN(::mlir::vector::VectorTransposeLowering::LLVMIntr, "llvmintr",
            "Lower 2-D transpose directly to `llvm.intr.matrix.transpose`"),
           clEnumValN(::mlir::vector::VectorTransposeLowering::Shuffle1D, "shuffle1d",
            "Lower 2-D transpose to `vector.shuffle` on 1-D vector."),
           clEnumValN(::mlir::vector::VectorTransposeLowering::Shuffle16x16, "shuffle16x16",
            "Lower 2-D transpose to `vector.shuffle` on 16x16 vector.")
          )}]>,
  ];
}

//===----------------------------------------------------------------------===//
// VectorToSPIRV
//===----------------------------------------------------------------------===//

def ConvertVectorToSPIRVPass : Pass<"convert-vector-to-spirv"> {
  let summary = "Convert Vector dialect to SPIR-V dialect";
  let dependentDialects = [
    "spirv::SPIRVDialect",
    "ub::UBDialect"
  ];
}

//===----------------------------------------------------------------------===//
// VectorToXeGPU
//===----------------------------------------------------------------------===//

def ConvertVectorToXeGPU : Pass<"convert-vector-to-xegpu"> {
  let summary = "Lower the operations from the vector dialect into the XeGPU "
                "dialect";
  let dependentDialects = [
    "memref::MemRefDialect", "arith::ArithDialect",
    "vector::VectorDialect", "xegpu::XeGPUDialect"
  ];
}

//===----------------------------------------------------------------------===//
// VectorToAMX
//===----------------------------------------------------------------------===//

def ConvertVectorToAMX : Pass<"convert-vector-to-amx"> {
  let summary = "Lower the operations from the vector dialect into the AMX "
                "dialect";
  let dependentDialects = [
    "affine::AffineDialect", "amx::AMXDialect", "arith::ArithDialect",
    "memref::MemRefDialect", "scf::SCFDialect", "vector::VectorDialect"
  ];
}

//===----------------------------------------------------------------------===//
// XeVMToLLVM
//===----------------------------------------------------------------------===//

def ConvertXeVMToLLVMPass : Pass<"convert-xevm-to-llvm"> {
  let summary = "Convert XeVM to LLVM dialect";
  let dependentDialects = ["LLVM::LLVMDialect"];
}

//===----------------------------------------------------------------------===//
// XeGPUToXeVM
//===----------------------------------------------------------------------===//

def ConvertXeGPUToXeVMPass : Pass<"convert-xegpu-to-xevm"> {
  let summary = "Convert XeGPU to XeVM dialect";
  let dependentDialects = ["xevm::XeVMDialect", "vector::VectorDialect",
                           "memref::MemRefDialect", "arith::ArithDialect",
                           "LLVM::LLVMDialect", "index::IndexDialect",
                           "gpu::GPUDialect", "scf::SCFDialect"];
}

#endif // MLIR_CONVERSION_PASSES
