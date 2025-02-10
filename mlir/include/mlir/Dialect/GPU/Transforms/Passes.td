//===-- Passes.td - GPU pass definition file ---------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_GPU_PASSES
#define MLIR_DIALECT_GPU_PASSES

include "mlir/Pass/PassBase.td"

def GpuLaunchSinkIndexComputations : Pass<"gpu-launch-sink-index-computations"> {
  let summary = "Sink index computations into gpu.launch body";
  let constructor = "mlir::createGpuLauchSinkIndexComputationsPass()";
  let dependentDialects = ["mlir::gpu::GPUDialect"];
}

def GpuKernelOutlining : Pass<"gpu-kernel-outlining", "ModuleOp"> {
  let summary = "Outline gpu.launch bodies to kernel functions";
  let constructor = "mlir::createGpuKernelOutliningPass()";
  let dependentDialects = ["mlir::DLTIDialect", "cf::ControlFlowDialect"];
}

def GpuAsyncRegionPass : Pass<"gpu-async-region", "func::FuncOp"> {
  let summary = "Make GPU ops async";
  let constructor = "mlir::createGpuAsyncRegionPass()";
  let dependentDialects = ["async::AsyncDialect"];
}

def GpuMapParallelLoopsPass
    : Pass<"gpu-map-parallel-loops", "mlir::func::FuncOp"> {
  let summary = "Greedily maps loops to GPU hardware dimensions.";
  let constructor = "mlir::createGpuMapParallelLoopsPass()";
  let description = "Greedily maps loops to GPU hardware dimensions.";
  let dependentDialects = ["mlir::gpu::GPUDialect"];
}

def GpuEliminateBarriers
    : Pass<"gpu-eliminate-barriers", "mlir::func::FuncOp"> {
  let summary = "Erase unnecessary barriers";
  let description = [{
    Barrier elimination pass. If a barrier does not enforce any conflicting
    pair of memory effects, including a pair that is enforced by another
    barrier, it is unnecessary and can be removed. Adapted from
    "High-Performance GPU-to-CPU Transpilation and Optimization via High-Level
    Parallel Constructs" by Moses, Ivanov, Domke, Endo, Doerfert, and Zinenko in
    PPoPP 2023 and implementation in Polygeist.
  }];
  let dependentDialects = [
    "mlir::gpu::GPUDialect",
    "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect"
  ];
}

def GpuDecomposeMemrefsPass : Pass<"gpu-decompose-memrefs"> {
  let summary = "Decomposes memref index computation into explicit ops.";
  let description = [{
    This pass decomposes memref index computation into explicit computations on
    sizes/strides, obtained from `memref.extract_memref_metadata` which it tries
    to place outside of `gpu.launch` body. Memrefs are then reconstructed using
    `memref.reinterpret_cast`.
    This is needed for as some targets (SPIR-V) lower memrefs to bare pointers
    and sizes/strides for dynamically-sized memrefs are not available inside
    `gpu.launch`.
  }];
  let constructor = "mlir::createGpuDecomposeMemrefsPass()";
  let dependentDialects = [
    "mlir::gpu::GPUDialect", "mlir::memref::MemRefDialect",
    "mlir::affine::AffineDialect"
  ];
}

def GpuModuleToBinaryPass
    : Pass<"gpu-module-to-binary", ""> {
  let summary = "Transforms a GPU module into a GPU binary.";
  let description = [{
    This pass searches for all nested GPU modules and serializes the module
    using the target attributes attached to the module, producing a GPU binary
    with an object for every target.

    The `format` argument can have the following values:
    1. `offloading`, `llvm`: produces an offloading representation.
    2. `assembly`, `isa`: produces assembly code.
    3. `binary`, `bin`: produces binaries.
    4. `fatbinary`, `fatbin`: produces fatbinaries.
  }];
  let options = [
    Option<"toolkitPath", "toolkit", "std::string", [{""}],
           "Toolkit path.">,
    ListOption<"linkFiles", "l", "std::string",
           "Extra files to link to.">,
    Option<"cmdOptions", "opts", "std::string", [{""}],
           "Command line options to pass to the tools.">,
    Option<"compilationTarget", "format", "std::string", [{"fatbin"}],
           "The target representation of the compilation process.">,
    Option<"elfSection", "section", "std::string", [{""}],
           "ELF section where binary is to be located.">
  ];
}

def GpuNVVMAttachTarget: Pass<"nvvm-attach-target", ""> {
  let summary = "Attaches an NVVM target attribute to a GPU Module.";
  let description = [{
    This pass searches for all GPU Modules in the immediate regions and attaches
    an NVVM target if the module matches the name specified by the `module` argument.

    Example:
    ```
    // File: in.mlir:
    gpu.module @nvvm_module_1 {...}
    gpu.module @nvvm_module_2 {...}
    gpu.module @rocdl_module_1 {...}
    // mlir-opt --nvvm-attach-target="module=nvvm.* chip=sm_90" in.mlir
    gpu.module @nvvm_module_1 [#nvvm.target<chip = "sm_90">] {...}
    gpu.module @nvvm_module_2 [#nvvm.target<chip = "sm_90">] {...}
    gpu.module @rocdl_module_1 {...}
    ```
  }];
  let options = [
    Option<"moduleMatcher", "module", "std::string",
           /*default=*/ [{""}],
           "Regex used to identify the modules to attach the target to.">,
    Option<"triple", "triple", "std::string",
           /*default=*/ "\"nvptx64-nvidia-cuda\"",
           "Target triple.">,
    Option<"chip", "chip", "std::string",
           /*default=*/"\"sm_50\"",
           "Target chip.">,
    Option<"features", "features", "std::string",
           /*default=*/"\"+ptx60\"",
           "Target features.">,
    Option<"optLevel", "O", "unsigned",
           /*default=*/"2",
           "Optimization level.">,
    Option<"fastFlag", "fast", "bool",
           /*default=*/"false",
           "Enable fast math mode.">,
    Option<"ftzFlag", "ftz", "bool",
           /*default=*/"false",
           "Enable flush to zero for denormals.">,
    Option<"nvvmOptions", "nvvm-options", "std::string",
           /*default=*/ [{""}],
           "Options passed to libnvvm">,
    ListOption<"linkLibs", "l", "std::string",
           "Extra bitcode libraries paths to link to.">,
  ];
}

def GpuROCDLAttachTarget: Pass<"rocdl-attach-target", ""> {
  let summary = "Attaches a ROCDL target attribute to a GPU Module.";
  let description = [{
    This pass searches for all GPU Modules in the immediate regions and attaches
    a ROCDL target if the module matches the name specified by the `module` argument.

    Example:
    ```
    // File: in.mlir:
    gpu.module @nvvm_module_1 {...}
    gpu.module @nvvm_module_2 {...}
    gpu.module @rocdl_module_1 {...}
    // mlir-opt --nvvm-attach-target="module=rocdl.* chip=gfx90a" in.mlir
    gpu.module @nvvm_module_1 {...}
    gpu.module @nvvm_module_2 {...}
    gpu.module @rocdl_module_1 [#rocdl.target<chip = "gfx90a">] {...}
    ```
  }];
  let options = [
    Option<"moduleMatcher", "module", "std::string",
           /*default=*/ [{""}],
           "Regex used to identify the modules to attach the target to.">,
    Option<"triple", "triple", "std::string",
           /*default=*/ "\"amdgcn-amd-amdhsa\"",
           "Target triple.">,
    Option<"chip", "chip", "std::string",
           /*default=*/"\"gfx900\"",
           "Target chip.">,
    Option<"features", "features", "std::string",
           /*default=*/"\"\"",
           "Target features.">,
    Option<"abiVersion", "abi", "std::string",
           /*default=*/"\"500\"",
           "ABI version.">,
    Option<"optLevel", "O", "unsigned",
           /*default=*/"2",
           "Optimization level.">,
    Option<"wave64Flag", "wave64", "bool",
           /*default=*/"true",
           "Use Wave64 mode.">,
    Option<"fastFlag", "fast", "bool",
           /*default=*/"false",
           "Enable fast relaxed math opt.">,
    Option<"dazFlag", "daz", "bool",
           /*default=*/"false",
           "Enable denormals are zero opt.">,
    Option<"finiteOnlyFlag", "finite-only", "bool",
           /*default=*/"false",
           "Enable finite only opt.">,
    Option<"unsafeMathFlag", "unsafe-math", "bool",
           /*default=*/"false",
           "Enable unsafe math opt.">,
    Option<"correctSqrtFlag", "correct-sqrt", "bool",
           /*default=*/"true",
           "Enable correct rounded sqrt.">,
    ListOption<"linkLibs", "l", "std::string",
           "Extra bitcode libraries paths to link to.">,
  ];
}

def GpuSPIRVAttachTarget: Pass<"spirv-attach-target", ""> {
  let summary = "Attaches an SPIR-V target attribute to a GPU Module.";
  let description = [{
    This pass searches for all GPU Modules in the immediate regions and attaches
    an SPIR-V target if the module matches the name specified by the `module` argument.

    Example:
    ```
    // Given the following file: in1.mlir:
    gpu.module @nvvm_module_1 {...}
    gpu.module @spirv_module_1 {...}
    // With
    // mlir-opt --spirv-attach-target="module=spirv.* ver=v1.0 caps=Kernel" in1.mlir
    // it will generate,
    gpu.module @nvvm_module_1 {...}
    gpu.module @spirv_module_1 [#spirv.target<#spirv.vce<v1.0, [Kernel], []>, #spirv.resource_limits<>>] {...}
    ```
  }];
  let options = [
    Option<"moduleMatcher", "module", "std::string",
           /*default=*/ [{""}],
           "Regex used to identify the modules to attach the target to.">,
    Option<"spirvVersion", "ver", "std::string",
           /*default=*/ "\"v1.0\"",
           "SPIR-V Version.">,
    ListOption<"spirvCapabilities", "caps", "std::string",
           "List of supported SPIR-V Capabilities">,
    ListOption<"spirvExtensions", "exts", "std::string",
           "List of supported SPIR-V Extensions">,
    Option<"clientApi", "client_api", "std::string",
           /*default=*/ "\"Unknown\"",
           "Client API">,
    Option<"deviceVendor", "vendor", "std::string",
           /*default=*/ "\"Unknown\"",
           "Device Vendor">,
    Option<"deviceType", "device_type", "std::string",
           /*default=*/ "\"Unknown\"",
           "Device Type">,
    Option<"deviceId", "device_id", "uint32_t",
           /*default=*/ "",
           "Device ID">,
  ];
}

#endif // MLIR_DIALECT_GPU_PASSES
