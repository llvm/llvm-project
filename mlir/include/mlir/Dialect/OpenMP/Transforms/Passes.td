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

def HostOpFilteringPass : Pass<"omp-host-op-filter"> {
  let summary = "Removes all non target-related ops from host functions.";
  let description = [{
    When compiling for an OpenMP offloading target, host functions holding
    target regions must be maintained in order for the `omp.target` ops to be
    translated to LLVM IR as functions.
    
    However, these host functions might contain more host code than just what's
    necessary for generating code for their contained target regions on the
    target device, potentially causing problematic artifacts to be generated
    into the target device's module.

    This pass strips out all non-essential operations from host functions when
    compiling for an OpenMP target device, leaving just `omp.target` operations,
    related `omp.map.info` and placeholders in the form of function arguments to
    keep the MLIR representation legal.

    It must run on a module at a low level of abstraction (i.e. directly
    translatable to LLVM IR), after any passes that might produce new OpenMP
    operations have already completed.
  }];
  let dependentDialects = ["LLVM::LLVMDialect"];
}

def MarkDeclareTargetPass : Pass<"omp-mark-declare-target", "ModuleOp"> {
  let summary = "Marks all functions called by an OpenMP declare target "
                "function as declare target";
  let description = [{
    Marks functions contained within the module as declare target if they are
    called from within an explicitly marked declare target function or a target
    region (omp.target).
    }];
  let dependentDialects = ["mlir::omp::OpenMPDialect"];
}

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
    This pass replaces `llvm.alloca` operations located in a non-SPMD target
    region and then potentially used inside of an `omp.parallel` region with
    `omp.alloc_shared_mem` and `omp.free_shared_mem`. This is also done for
    top-level function `llvm.alloca`s used in the same way when the parent
    function is a target device function.

    This ensures that explicit private allocations, intended to be shared across
    threads, use the proper memory space on a target device while supporting the
    case of parallel regions indirectly reached from within a target region via
    function calls.
  }];
  let dependentDialects = ["mlir::omp::OpenMPDialect"];
}

#endif // MLIR_DIALECT_OPENMP_TRANSFORMS_PASSES
