//===-- Passes.td - GPU SPIR-V pass definition file --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_GPU_TRANSFORMS_SPIRV_PASSES
#define MLIR_DIALECT_GPU_TRANSFORMS_SPIRV_PASSES

include "mlir/Pass/PassBase.td"

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

#endif // MLIR_DIALECT_GPU_TRANSFORMS_SPIRV_PASSES
