//===-- Passes.td - SPIRV pass definition file -------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SPIRV_TRANSFORMS_PASSES
#define MLIR_DIALECT_SPIRV_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def SPIRVCanonicalizeGLPass : Pass<"spirv-canonicalize-gl", ""> {
  let summary = "Canonicalize GLSL ops";
  let description = [{
    Pass to run canoncalization patterns that involve GL ops.
    These patterns cannot be run in default canonicalization because GL ops
    aren't always available. So they should be involed specifically when needed.
  }];
}

def SPIRVCompositeTypeLayoutPass
    : Pass<"decorate-spirv-composite-type-layout", "mlir::ModuleOp"> {
  let summary = "Decorate SPIR-V composite type with layout info";
  let description = [{
    Module pass that converts composite types used by objects in the
    StorageBuffer, PhysicalStorageBuffer, Uniform, and PushConstant storage
    classes to attatch layout information.
    Right now this pass only supports Vulkan layout rules.
  }];
}

def SPIRVLowerABIAttributesPass : Pass<"spirv-lower-abi-attrs", "spirv::ModuleOp"> {
  let summary = "Decorate SPIR-V composite type with layout info";
  let description = [{
    Operation pass that lowers the ABI attributes specified during
    SPIR-V Lowering. Specifically:
    1. Creates the global variables for arguments of entry point function using
      the specification in the `spirv.interface_var_abi` attribute for each
      argument.
    2. Inserts the EntryPointOp and the ExecutionModeOp for entry point
      functions using the specification in the `spirv.entry_point_abi`
      attribute.
  }];
}

def SPIRVRewriteInsertsPass : Pass<"spirv-rewrite-inserts", "spirv::ModuleOp"> {
  let summary = "Rewrite sequential chains of `spirv.CompositeInsert` operations "
                "into `spirv.CompositeConstruct` operations";
}

def SPIRVUnifyAliasedResourcePass
    : Pass<"spirv-unify-aliased-resource", "spirv::ModuleOp"> {
  let summary = "Unify access of multiple aliased resources into access of one "
                "single resource";
  let constructor = "mlir::spirv::createUnifyAliasedResourcePass()";
}

def SPIRVUpdateVCEPass : Pass<"spirv-update-vce", "spirv::ModuleOp"> {
  let summary = "Deduce and attach minimal (version, capabilities, extensions) "
                "requirements to spirv.module ops";
  let description = [{
    Operation pass that deduces and attaches the minimal version/
    capabilities/extensions requirements for spirv.module ops.
    For each spirv.module op, this pass requires a `spirv.target_env` attribute
    on it or an enclosing module-like op to drive the deduction. The reason is
    that an op can be enabled by multiple extensions/capabilities. So we need
    to know which one to pick. `spirv.target_env` gives the hard limit as for
    what the target environment can support; this pass deduces what are
    actually needed for a specific spirv.module op.
  }];
}

def SPIRVWebGPUPreparePass : Pass<"spirv-webgpu-prepare", "spirv::ModuleOp"> {
  let summary = "Prepare SPIR-V to target WebGPU by expanding unsupported ops "
                "and replacing with supported ones";
}

#endif // MLIR_DIALECT_SPIRV_TRANSFORMS_PASSES
