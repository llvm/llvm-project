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

  let description = [{ First pass attempt}];

  let dependentDialects = ["omp::OpenMPDialect"];
}
#endif
