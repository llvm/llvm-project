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

def StackToSharedPass : Pass<"omp-stack-to-shared", "mlir::LLVM::LLVMFuncOp"> {
  let summary = "Replaces stack allocations target devices with shared memory.";
  let description = [{
    `llvm.alloca` operations defining values in a non-SPMD target region and
    then potentially used inside of an `omp.parallel` region are replaced by
    this pass with `omp.alloc_shared_mem` and `omp.free_shared_mem`. This is
    also done for top-level function `llvm.alloca`s used in the same way when
    the parent function is a target device function.

    This ensures that explicit private allocations, intended to be shared across
    threads, use the proper memory space on a target device while supporting the
    case of parallel regions indirectly reached from within a target region via
    function calls.
  }];
  let dependentDialects = ["mlir::omp::OpenMPDialect"];
}

#endif // MLIR_DIALECT_OPENMP_TRANSFORMS_PASSES
