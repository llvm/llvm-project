//===-- Passes.td - OMP pass definition file ---------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_OPENMP_PASSES
#define MLIR_DIALECT_OPENMP_PASSES

include "mlir/Pass/PassBase.td"

def OMPTargetGPUOutlining : Pass<"omp-target-outline-to-gpu", "ModuleOp"> {
  let summary = "Outline omp.target regions into GPU modules.";
  let description = [{

  }];
  let dependentDialects = ["mlir::gpu::GPUDialect", "mlir::func::FuncDialect"];
  let options = [
    Option<"moduleName", "name", "std::string", [{"omp_offload"}],
           "Name of the GPU module.">
  ];
}

#endif // MLIR_DIALECT_OPENMP_PASSES
