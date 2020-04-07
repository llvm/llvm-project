//===-- Passes.td - Loop pass definition file --------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LOOP_PASSES
#define MLIR_DIALECT_LOOP_PASSES

include "mlir/Pass/PassBase.td"

def LoopParallelLoopFusion : Pass<"parallel-loop-fusion"> {
  let summary = "Fuse adjacent parallel loops";
  let constructor = "mlir::createParallelLoopFusionPass()";
}

def LoopParallelLoopSpecialization
    : FunctionPass<"parallel-loop-specialization"> {
  let summary = "Specialize parallel loops for vectorization";
  let constructor = "mlir::createParallelLoopSpecializationPass()";
}

def LoopParallelLoopTiling : FunctionPass<"parallel-loop-tiling"> {
  let summary = "Tile parallel loops";
  let constructor = "mlir::createParallelLoopTilingPass()";
  let options = [
    ListOption<"tileSizes", "parallel-loop-tile-sizes", "int64_t",
               "Factors to tile parallel loops by",
               "llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated">
  ];
}

#endif // MLIR_DIALECT_LOOP_PASSES
