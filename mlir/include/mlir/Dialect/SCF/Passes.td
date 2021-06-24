//===-- Passes.td - SCF pass definition file ---------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SCF_PASSES
#define MLIR_DIALECT_SCF_PASSES

include "mlir/Pass/PassBase.td"

def SCFBufferize : FunctionPass<"scf-bufferize"> {
  let summary = "Bufferize the scf dialect.";
  let constructor = "mlir::createSCFBufferizePass()";
  let dependentDialects = ["memref::MemRefDialect"];
}

def SCFForLoopSpecialization
    : FunctionPass<"for-loop-specialization"> {
  let summary = "Specialize `for` loops for vectorization";
  let constructor = "mlir::createForLoopSpecializationPass()";
}

def SCFParallelLoopFusion : Pass<"parallel-loop-fusion"> {
  let summary = "Fuse adjacent parallel loops";
  let constructor = "mlir::createParallelLoopFusionPass()";
}

def SCFParallelLoopSpecialization
    : FunctionPass<"parallel-loop-specialization"> {
  let summary = "Specialize parallel loops for vectorization";
  let constructor = "mlir::createParallelLoopSpecializationPass()";
}

def SCFParallelLoopTiling : FunctionPass<"parallel-loop-tiling"> {
  let summary = "Tile parallel loops";
  let constructor = "mlir::createParallelLoopTilingPass()";
  let options = [
    ListOption<"tileSizes", "parallel-loop-tile-sizes", "int64_t",
               "Factors to tile parallel loops by",
               "llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated">
  ];
  let dependentDialects = ["AffineDialect"];
}

def SCFForLoopRangeFolding
    : Pass<"for-loop-range-folding"> {
  let summary = "Fold add/mul ops into loop range";
  let constructor = "mlir::createForLoopRangeFoldingPass()";
}

#endif // MLIR_DIALECT_SCF_PASSES
