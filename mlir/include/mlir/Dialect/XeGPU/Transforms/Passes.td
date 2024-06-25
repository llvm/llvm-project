//===-- Passes.td - XeGPU transformation definition file ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//


#ifndef MLIR_DIALECT_XEGPU_TRANSFORMS_PASSES_TD
#define MLIR_DIALECT_XEGPU_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def XeGPUFoldAliasOps : Pass<"xegpu-fold-alias-ops"> {
  let summary = "Fold alias ops into XeGPU ops";
  let description = [{
    The pass folds aliasing ops into XeGPU ops that they operate on the original
    source references.
  }];
  let dependentDialects = [
      "memref::MemRefDialect", "xegpu::XeGPUDialect"
  ];
}

#endif // MLIR_DIALECT_XEGPU_TRANSFORMS_PASSES_TD
