//===-- Passes.td - SPIRV pass definition file -------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SPIRV_PASSES
#define MLIR_DIALECT_SPIRV_PASSES

include "mlir/Pass/PassBase.td"

def SPIRVCompositeTypeLayout : Pass<"decorate-spirv-composite-type-layout"> {
  let summary = "Decorate SPIR-V composite type with layout info";
  let constructor = "mlir::spirv::createDecorateSPIRVCompositeTypeLayoutPass()";
}

def SPIRVLowerABIAttributes : Pass<"spirv-lower-abi-attrs"> {
  let summary = "Decorate SPIR-V composite type with layout info";
  let constructor = "mlir::spirv::createLowerABIAttributesPass()";
}

def SPIRVUpdateVCE : Pass<"spirv-update-vce"> {
  let summary = "Deduce and attach minimal (version, capabilities, extensions) "
                "requirements to spv.module ops";
  let constructor = "mlir::spirv::createUpdateVersionCapabilityExtensionPass()";
}

#endif // MLIR_DIALECT_SPIRV_PASSES
