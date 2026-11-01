//===-- Passes.td - GPU NVVM pass definition file ----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_GPU_TRANSFORMS_NVVM_PASSES
#define MLIR_DIALECT_GPU_TRANSFORMS_NVVM_PASSES

include "mlir/Pass/PassBase.td"

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
           /*default=*/"\"sm_75\"",
           "Target chip.">,
    Option<"features", "features", "std::string",
           /*default=*/"\"\"",
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
    Option<"compilerDiagnosticsFlag", "collect-compiler-diagnostics", "bool",
           /*default=*/"false",
           "Enable collection of compiler diagnostics.">,
    ListOption<"linkLibs", "l", "std::string",
           "Extra bitcode libraries paths to link to.">,
    Option<"cmdOptions", "ptxas-cmd-options", "std::string",
           /*default=*/ [{""}],
           "Command line options passed to downstream compiler">,
    Option<"verifyTarget", "verify-target-arch", "bool",
           /*default=*/"true",
           "Enable verification of the target architecture">,
  ];
}

#endif // MLIR_DIALECT_GPU_TRANSFORMS_NVVM_PASSES
