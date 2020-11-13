//===-- Passes.td - StandardOps pass definition file -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_STANDARD_TRANSFORMS_PASSES
#define MLIR_DIALECT_STANDARD_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def StdBufferize : FunctionPass<"std-bufferize"> {
  let summary = "Bufferize the std dialect";
  let constructor = "mlir::createStdBufferizePass()";
  let dependentDialects = ["scf::SCFDialect"];
}

def StdExpandOps : FunctionPass<"std-expand"> {
  let summary = "Legalize std operations to be convertible to LLVM.";
  let constructor = "mlir::createStdExpandOpsPass()";
}

def FuncBufferize : Pass<"func-bufferize", "ModuleOp"> {
  let summary = "Bufferize func/call/return ops";
  let description = [{
    A finalizing bufferize pass that bufferizes std.func and std.call ops.

    Because this pass updates std.func ops, it must be a module pass. It is
    useful to keep this pass separate from other bufferizations so that the
    other ones can be run at function-level in parallel.

    This pass must be done atomically for two reasons:
    1. This pass changes func op signatures, which requires atomically updating
       calls as well throughout the entire module.
    2. This pass changes the type of block arguments, which requires that all
       successor arguments of predecessors be converted. Terminators are not
       a closed universe (and need not implement BranchOpInterface), and so we
       cannot in general rewrite them.

    Note, because this is a "finalizing" bufferize step, it can create
    invalid IR because it will not create materializations. To avoid this
    situation, the pass must only be run when the only SSA values of
    tensor type are:
    - block arguments
    - the result of tensor_load
    Other values of tensor type should be eliminated by earlier
    bufferization passes.
  }];
  let constructor = "mlir::createFuncBufferizePass()";
}

def TensorConstantBufferize : Pass<"tensor-constant-bufferize", "ModuleOp"> {
  let summary = "Bufferize tensor constants.";
  let description = [{
    This pass bufferizes tensor constants.

    This pass needs to be a module pass because it inserts std.global_memref
    ops into the module, which cannot be done safely from a function pass due to
    multi-threading. Most other bufferization passes can run in parallel at
    function granularity.
  }];
  let constructor = "mlir::createTensorConstantBufferizePass()";
}

#endif // MLIR_DIALECT_STANDARD_TRANSFORMS_PASSES
