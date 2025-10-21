//===-- Passes.td - OpenMP pass definition file ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_OPENMP_TRANSFORMS_PASSES
#define MLIR_DIALECT_OPENMP_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def PrepareForOMPOffloadPrivatizationPass : Pass<"omp-offload-privatization-prepare", "ModuleOp"> {
    let summary = "Prepare OpenMP maps for privatization for deferred target tasks";
    let description = [{
      When generating LLVMIR for privatized variables in an OpenMP offloading directive (eg. omp::TargetOp)
      that creates a deferred target task (when the nowait clause is used), we need to copy the privatized
      variable out of the stack of the generating task and into the heap so that the deferred target task
      can still access it. However, if such a privatized variable is also mapped, typically the case for
      allocatables, then the corresponding `omp::MapInfoOp` needs to be fixed up to map the new heap-allocated
      variable and not the original variable.
    }];
  let dependentDialects = ["LLVM::LLVMDialect"];
}
#endif // MLIR_DIALECT_OPENMP_TRANSFORMS_PASSES
