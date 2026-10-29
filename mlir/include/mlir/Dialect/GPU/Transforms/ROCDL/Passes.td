//===-- Passes.td - GPU ROCDL pass definition file ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_GPU_TRANSFORMS_ROCDL_PASSES
#define MLIR_DIALECT_GPU_TRANSFORMS_ROCDL_PASSES

include "mlir/Pass/PassBase.td"

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
    Option<"arch", "arch", "std::string",
           /*default=*/"\"\"",
           "Target architecture, in the spelling the conversion passes take "
           "(e.g. gfx942, gfx90a:xnack+, amdgpu9.4-amd-amdhsa, "
           "amdgcn-amd-amdhsa--gfx90a:xnack-). "
           "When non-empty this supersedes 'triple' and 'chip', and its "
           "target-ID modifiers become the module's 'rocdl.xnack' and "
           "'rocdl.sramecc' attributes rather than target features.">,
    Option<"triple", "triple", "std::string",
           /*default=*/ "\"amdgcn-amd-amdhsa\"",
           "Target triple. Ignored when 'arch' is given.">,
    Option<"chip", "chip", "std::string",
           /*default=*/"\"gfx900\"",
           "Target chip. Ignored when 'arch' is given.">,
    Option<"features", "features", "std::string",
           /*default=*/"\"\"",
           "Target features.">,
    Option<"abiVersion", "abi", "std::string",
           /*default=*/"\"600\"",
           "ABI version.">,
    Option<"optLevel", "O", "unsigned",
           /*default=*/"2",
           "Optimization level.">,
    Option<"waveSize", "wavesize", "unsigned",
           /*default=*/"0",
           "Wavefront size (32 or 64). 0 takes the size from the target when "
           "'arch' is given, and Wave64 otherwise.">,
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

#endif // MLIR_DIALECT_GPU_TRANSFORMS_ROCDL_PASSES
