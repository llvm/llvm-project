//===-- Passes.td - GPU XeVM pass definition file ----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_GPU_TRANSFORMS_XEVM_PASSES
#define MLIR_DIALECT_GPU_TRANSFORMS_XEVM_PASSES

include "mlir/Pass/PassBase.td"

def GpuXeVMAttachTarget : Pass<"xevm-attach-target", ""> {
  let summary = "Attaches a XeVM target attribute to a GPU Module.";
  let description = [{
    This pass searches for all GPU Modules in the immediate regions and attaches
    a XeVM target if the module matches the name specified by the `module` argument.

    Example:
    ```
    // File: in.mlir:
    gpu.module @nvvm_module_1 {...}
    gpu.module @rocdl_module_2 {...}
    gpu.module @xevm_module_3 {...}
    // mlir-opt --xevm-attach-target="module=xevm.* chip=pvc" in.mlir
    gpu.module @nvvm_module_1 {...}
    gpu.module @rocdl_module_2 {...}
    gpu.module @xevm_module_3 [#xevm.target<chip = "pvc">] {...}
    ```
  }];
  let options =
      [Option<"moduleMatcher", "module", "std::string",
              /*default=*/[{""}],
              "Regex used to identify the modules to attach the target to.">,
       Option<"triple", "triple", "std::string",
              /*default=*/"\"spirv64-unknown-unknown\"", "Target triple.">,
       Option<"chip", "chip", "std::string",
              /*default=*/"\"bmg\"", "Target chip.">,
       Option<"optLevel", "O", "unsigned",
              /*default=*/"2", "Optimization level.">,
       ListOption<"linkLibs", "l", "std::string",
                  "Extra bitcode libraries paths to link to.">,
       Option<"cmdOptions", "cmd-options", "std::string",
              /*default=*/[{""}],
              "Command line options passed to downstream compiler">];
}

#endif // MLIR_DIALECT_GPU_TRANSFORMS_XEVM_PASSES
