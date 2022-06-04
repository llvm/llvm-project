//===-- Passes.td - LLVM pass definition file --------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LLVMIR_TRANSFORMS_PASSES
#define MLIR_DIALECT_LLVMIR_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def LLVMLegalizeForExport : Pass<"llvm-legalize-for-export"> {
  let summary = "Legalize LLVM dialect to be convertible to LLVM IR";
  let constructor = "mlir::LLVM::createLegalizeForExportPass()";
}

def NVVMOptimizeForTarget : Pass<"llvm-optimize-for-nvvm-target"> {
  let summary = "Optimize NVVM IR";
  let constructor = "mlir::NVVM::createOptimizeForTargetPass()";
}

#endif // MLIR_DIALECT_LLVMIR_TRANSFORMS_PASSES
