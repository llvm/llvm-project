//===-- Passes.td - Affine pass definition file ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions for passes within the Affine/ directory.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_AFFINE_PASSES
#define MLIR_DIALECT_AFFINE_PASSES

include "mlir/Pass/PassBase.td"

def AffineDataCopyGeneration : Pass<"affine-data-copy-generate"> {
  let summary = "Generate explicit copying for affine memory operations";
  let constructor = "mlir::createAffineDataCopyGenerationPass()";
}

def AffineLoopInvariantCodeMotion : Pass<"affine-loop-invariant-code-motion"> {
  let summary = "Hoist loop invariant instructions outside of affine loops";
  let constructor = "mlir::createAffineLoopInvariantCodeMotionPass()";
}

def AffineLoopTiling : Pass<"affine-loop-tile"> {
  let summary = "Tile affine loop nests";
  let constructor = "mlir::createLoopTilingPass()";
}

def AffineLoopUnroll : Pass<"affine-loop-unroll"> {
  let summary = "Unroll affine loops";
  let constructor = "mlir::createLoopUnrollPass()";
}

def AffineLoopUnrollAndJam : Pass<"affine-loop-unroll-jam"> {
  let summary = "Unroll and jam affine loops";
  let constructor = "mlir::createLoopUnrollAndJamPass()";
}

def AffineVectorize : Pass<"affine-super-vectorize"> {
  let summary = "Vectorize to a target independent n-D vector abstraction";
  let constructor = "mlir::createSuperVectorizePass()";
  let options = [
    ListOption<"vectorSizes", "virtual-vector-size", "int64_t",
               "Specify an n-D virtual vector size for vectorization",
               "llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated">,
    // Optionally, the fixed mapping from loop to fastest varying MemRef
    // dimension for all the MemRefs within a loop pattern:
    //   the index represents the loop depth, the value represents the k^th
    //   fastest varying memory dimension.
    // This is voluntarily restrictive and is meant to precisely target a
    // particular loop/op pair, for testing purposes.
    ListOption<"fastestVaryingPattern", "test-fastest-varying", "int64_t",
               "Specify a 1-D, 2-D or 3-D pattern of fastest varying memory "
               "dimensions to match. See defaultPatterns in Vectorize.cpp for "
               "a description and examples. This is used for testing purposes",
               "llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated">
  ];
}

def SimplifyAffineStructures : Pass<"simplify-affine-structures"> {
  let summary = "Simplify affine expressions in maps/sets and normalize "
                "memrefs";
  let constructor = "mlir::createSimplifyAffineStructuresPass()";
}

#endif // MLIR_DIALECT_AFFINE_PASSES
