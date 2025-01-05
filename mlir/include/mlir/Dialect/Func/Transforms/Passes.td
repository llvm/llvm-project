//===-- Passes.td - Func pass definition file --------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_FUNC_TRANSFORMS_PASSES_TD
#define MLIR_DIALECT_FUNC_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def DuplicateFunctionEliminationPass : Pass<"duplicate-function-elimination",
    "ModuleOp"> {
  let summary = "Deduplicate functions";
  let description = [{
    Deduplicate functions that are equivalent in all aspects but their symbol
    name. The pass chooses one representative per equivalence class, erases
    the remainder, and updates function calls accordingly.
  }];
  let constructor = "mlir::func::createDuplicateFunctionEliminationPass()";
}

#endif // MLIR_DIALECT_FUNC_TRANSFORMS_PASSES_TD
