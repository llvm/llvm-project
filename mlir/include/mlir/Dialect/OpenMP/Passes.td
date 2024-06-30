//===-- Passes.td - OpenMP pass definition file -------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_OPENMP_PASSES
#define MLIR_DIALECT_OPENMP_PASSES

include "mlir/Pass/PassBase.td"

def OpenMPTaskBasedTarget : Pass<"openmp-task-based-target", "func::FuncOp"> {
  let summary = "Nest certain instances of mlir::omp::TargetOp inside mlir::omp::TaskOp";

  let constructor = "mlir::createOpenMPTaskBasedTargetPass()";

  let description = [{
  This pass transforms `omp.target`, `omp.target_enter_data`,
  `omp.target_update_data` and `omp.target_exit_data` whenever these operations
  have the `depend` clause on them.

  These operations are transformed by enclosing them inside a new `omp.task`
  operation. The `depend` clause related arguments are moved to the new `omp.task`
  operation from the original 'target' operation.

  Example:
  Input:
  ```mlir
  omp.target map_entries(%map_a -> %arg0, %map_c_from -> %arg1 : memref<?xi32>, memref<?xi32>) depend(taskdependout -> %c : memref<?xi32>) {
    ^bb0(%arg0 : memref<?xi32>, %arg1 : memref<?xi32>) :
        "test.foobar"() : ()->()
         omp.terminator
  }
  ```
  Output:
  ```mlir
  omp.task depend(taskdependout -> %c : memref<?xi32>) {
    omp.target map_entries(%map_a -> %arg0, %map_c_from -> %arg1 : memref<?xi32>, memref<?xi32>) {
      ^bb0(%arg0 : memref<?xi32>, %arg1 : memref<?xi32>) :
          "test.foobar"() : ()->()
           omp.terminator
    }
    omp.terminator
  }
  ```
  The intent is to make it easier to translate to LLVMIR by avoiding the
  creation of such tasks in the OMPIRBuilder.
  }];

  let dependentDialects = ["omp::OpenMPDialect"];
}
#endif
