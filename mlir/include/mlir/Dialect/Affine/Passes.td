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

def AffineDataCopyGeneration : FunctionPass<"affine-data-copy-generate"> {
  let summary = "Generate explicit copying for affine memory operations";
  let constructor = "mlir::createAffineDataCopyGenerationPass()";
  let options = [
    Option<"fastMemoryCapacity", "fast-mem-capacity", "uint64_t",
           /*default=*/"std::numeric_limits<uint64_t>::max()",
           "Set fast memory space capacity in KiB (default: unlimited)">,
    Option<"fastMemorySpace", "fast-mem-space", "unsigned",
           /*default=*/"1",
           "Fast memory space identifier for copy generation (default: 1)">,
    Option<"generateDma", "generate-dma", "bool",
           /*default=*/"true", "Generate DMA instead of point-wise copy">,
    Option<"minDmaTransferSize", "min-dma-transfer", "int",
           /*default=*/"1024",
           "Minimum DMA transfer size supported by the target in bytes">,
    Option<"slowMemorySpace", "slow-mem-space", "unsigned",
           /*default=*/"0",
           "Slow memory space identifier for copy generation (default: 0)">,
    Option<"skipNonUnitStrideLoops", "skip-non-unit-stride-loops", "bool",
           /*default=*/"false", "Testing purposes: avoid non-unit stride loop "
                                "choice depths for copy placement">,
    Option<"tagMemorySpace", "tag-mem-space", "unsigned",
           /*default=*/"0",
           "Tag memory space identifier for copy generation (default: 0)">,
  ];
}

def AffineLoopInvariantCodeMotion
    : FunctionPass<"affine-loop-invariant-code-motion"> {
  let summary = "Hoist loop invariant instructions outside of affine loops";
  let constructor = "mlir::createAffineLoopInvariantCodeMotionPass()";
}

def AffineLoopTiling : FunctionPass<"affine-loop-tile"> {
  let summary = "Tile affine loop nests";
  let constructor = "mlir::createLoopTilingPass()";
  let options = [
    Option<"cacheSizeInKiB", "cache-size", "uint64_t", /*default=*/"512",
           "Set size of cache to tile for in KiB">,
    Option<"separate", "separate", "bool", /*default=*/"",
           "Separate full and partial tiles">,
    Option<"tileSize", "tile-size", "unsigned", /*default=*/"",
           "Use this tile size for all loops">,
    ListOption<"tileSizes", "tile-sizes", "unsigned",
               "List of tile sizes for each perfect nest "
               "(overridden by -tile-size)",
               "llvm::cl::ZeroOrMore">,
  ];
}

def AffineLoopUnroll : FunctionPass<"affine-loop-unroll"> {
  let summary = "Unroll affine loops";
  let constructor = "mlir::createLoopUnrollPass()";
  let options = [
    Option<"unrollFactor", "unroll-factor", "unsigned", /*default=*/"4",
           "Use this unroll factor for all loops being unrolled">,
    Option<"unrollUpToFactor", "unroll-up-to-factor", "bool",
           /*default=*/"false", "Allow unrolling up to the factor specified">,
    Option<"unrollFull", "unroll-full", "bool", /*default=*/"false",
           "Fully unroll loops">,
    Option<"numRepetitions", "unroll-num-reps", "unsigned", /*default=*/"1",
           "Unroll innermost loops repeatedly this many times">,
    Option<"unrollFullThreshold", "unroll-full-threshold", "unsigned",
           /*default=*/"1",
           "Unroll all loops with trip count less than or equal to this">,
  ];
}

def AffineLoopUnrollAndJam : FunctionPass<"affine-loop-unroll-jam"> {
  let summary = "Unroll and jam affine loops";
  let constructor = "mlir::createLoopUnrollAndJamPass()";
  let options = [
    Option<"unrollJamFactor", "unroll-jam-factor", "unsigned",
           /*default=*/"4",
           "Use this unroll jam factor for all loops (default 4)">,
  ];
}

def AffineVectorize : FunctionPass<"affine-super-vectorize"> {
  let summary = "Vectorize to a target independent n-D vector abstraction";
  let constructor = "mlir::createSuperVectorizePass()";
  let dependentDialects = ["vector::VectorDialect"];
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

def AffineParallelize : FunctionPass<"affine-parallelize"> {
  let summary = "Convert affine.for ops into 1-D affine.parallel";
  let constructor = "mlir::createAffineParallelizePass()";
}

def AffineParallelNormalize : FunctionPass<"affine-parallel-normalize"> {
  let summary = "Normalize affine.parallel ops so that lower bounds are 0 and "
                "steps are 1";
  let constructor = "mlir::createAffineParallelNormalizePass()";
}

def SimplifyAffineStructures : FunctionPass<"simplify-affine-structures"> {
  let summary = "Simplify affine expressions in maps/sets and normalize "
                "memrefs";
  let constructor = "mlir::createSimplifyAffineStructuresPass()";
}

#endif // MLIR_DIALECT_AFFINE_PASSES
