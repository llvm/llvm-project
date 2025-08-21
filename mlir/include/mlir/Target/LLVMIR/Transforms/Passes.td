//===-- Passes.td - LLVM Target pass definition file -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TARGET_LLVMIR_TRANSFORMS_PASSES
#define MLIR_TARGET_LLVMIR_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def LLVMTargetToDataLayout : Pass<"llvm-target-to-data-layout"> {
  let summary = "Derive data layout attributes from LLVM target attributes";
  let dependentDialects = ["mlir::DLTIDialect"];
  let description = [{
    Derive a `DataLayoutSpecInterface`-implementing data layout attribute from
    the LLVM-backend target specified by the `TargetAttrInterface`-implementing
    attribute attached to the target op at the name `llvm.target`.
  }];
  let options = [
    Option<"initializeLLVMTargets", "initialize-llvm-targets", "bool",
           /*default=*/"true",
           "Whether to pre-load all available target machines, that LLVM is "
           "configured to support, into the TargetRegistry.">
  ];
}

#endif // MLIR_TARGET_LLVMIR_TRANSFORMS_PASSES
