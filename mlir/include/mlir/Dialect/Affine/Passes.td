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
}

def SimplifyAffineStructures : Pass<"simplify-affine-structures"> {
  let summary = "Simplify affine expressions in maps/sets and normalize "
                "memrefs";
  let constructor = "mlir::createSimplifyAffineStructuresPass()";
}

#endif // MLIR_DIALECT_AFFINE_PASSES
