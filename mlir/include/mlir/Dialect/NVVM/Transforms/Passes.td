//===-- Passes.td - NVVM pass definition file --------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_NVVM_TRANSFORMS_PASSES
#define MLIR_DIALECT_NVVM_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def NVVMOptimizeForTargetPass : Pass<"llvm-optimize-for-nvvm-target"> {
  let summary = "Optimize NVVM IR";
}

#endif // MLIR_DIALECT_NVVM_TRANSFORMS_PASSES
