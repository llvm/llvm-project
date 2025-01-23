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

def AffineDataCopyGeneration : Pass<"affine-data-copy-generate", "func::FuncOp"> {
  let summary = "Generate explicit copying for affine memory operations";
  let constructor = "mlir::affine::createAffineDataCopyGenerationPass()";
  let dependentDialects = ["memref::MemRefDialect"];
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

def AffineLoopFusion : Pass<"affine-loop-fusion"> {
  let summary = "Fuse affine loop nests";
  let description = [{
    This pass performs fusion of loop nests using a slicing-based approach. The
    transformation works on an MLIR `Block` granularity and applies to all
    blocks of the pass is run on. It combines two fusion strategies:
    producer-consumer fusion and sibling fusion. Producer-consumer fusion is
    aimed at fusing pairs of loops where the first one writes to a memref that
    the second reads. Sibling fusion targets pairs of loops that share no
    dependences between them but that load from the same memref. The fused loop
    nests, when possible, are rewritten to access significantly smaller local
    buffers instead of the original memref's, and the latter are often either
    completely optimized away or contracted. This transformation leads to
    enhanced locality and lower memory footprint through the elimination or
    contraction of temporaries/intermediate memref's. These benefits are
    sometimes achieved at the expense of redundant computation through a cost
    model that evaluates available choices such as the depth at which a source
    slice should be materialized in the designation slice.

    Example 1: Producer-consumer fusion.
    Input:
    ```mlir
    func.func @producer_consumer_fusion(%arg0: memref<10xf32>, %arg1: memref<10xf32>) {
      %0 = memref.alloc() : memref<10xf32>
      %1 = memref.alloc() : memref<10xf32>
      %cst = arith.constant 0.000000e+00 : f32
      affine.for %arg2 = 0 to 10 {
        affine.store %cst, %0[%arg2] : memref<10xf32>
        affine.store %cst, %1[%arg2] : memref<10xf32>
      }
      affine.for %arg2 = 0 to 10 {
        %2 = affine.load %0[%arg2] : memref<10xf32>
        %3 = arith.addf %2, %2 : f32
        affine.store %3, %arg0[%arg2] : memref<10xf32>
      }
      affine.for %arg2 = 0 to 10 {
        %2 = affine.load %1[%arg2] : memref<10xf32>
        %3 = arith.mulf %2, %2 : f32
        affine.store %3, %arg1[%arg2] : memref<10xf32>
      }
      return
    }
    ```
    Output:
    ```mlir
    func.func @producer_consumer_fusion(%arg0: memref<10xf32>, %arg1: memref<10xf32>) {
      %0 = memref.alloc() : memref<1xf32>
      %1 = memref.alloc() : memref<1xf32>
      %cst = arith.constant 0.000000e+00 : f32
      affine.for %arg2 = 0 to 10 {
        affine.store %cst, %0[0] : memref<1xf32>
        affine.store %cst, %1[0] : memref<1xf32>
        %2 = affine.load %1[0] : memref<1xf32>
        %3 = arith.mulf %2, %2 : f32
        affine.store %3, %arg1[%arg2] : memref<10xf32>
        %4 = affine.load %0[0] : memref<1xf32>
        %5 = arith.addf %4, %4 : f32
        affine.store %5, %arg0[%arg2] : memref<10xf32>
      }
      return
    }
    ```

    Example 2: Sibling fusion.
    Input:
    ```mlir
    func.func @sibling_fusion(%arg0: memref<10x10xf32>, %arg1: memref<10x10xf32>,
                         %arg2: memref<10x10xf32>, %arg3: memref<10x10xf32>,
                         %arg4: memref<10x10xf32>) {
      affine.for %arg5 = 0 to 3 {
        affine.for %arg6 = 0 to 3 {
          %0 = affine.load %arg0[%arg5, %arg6] : memref<10x10xf32>
          %1 = affine.load %arg1[%arg5, %arg6] : memref<10x10xf32>
          %2 = arith.mulf %0, %1 : f32
          affine.store %2, %arg3[%arg5, %arg6] : memref<10x10xf32>
        }
      }
      affine.for %arg5 = 0 to 3 {
        affine.for %arg6 = 0 to 3 {
          %0 = affine.load %arg0[%arg5, %arg6] : memref<10x10xf32>
          %1 = affine.load %arg2[%arg5, %arg6] : memref<10x10xf32>
          %2 = arith.addf %0, %1 : f32
          affine.store %2, %arg4[%arg5, %arg6] : memref<10x10xf32>
        }
      }
      return
    }
    ```
    Output:
    ```mlir
    func.func @sibling_fusion(%arg0: memref<10x10xf32>, %arg1: memref<10x10xf32>,
                         %arg2: memref<10x10xf32>, %arg3: memref<10x10xf32>,
                         %arg4: memref<10x10xf32>) {
      affine.for %arg5 = 0 to 3 {
        affine.for %arg6 = 0 to 3 {
          %0 = affine.load %arg0[%arg5, %arg6] : memref<10x10xf32>
          %1 = affine.load %arg1[%arg5, %arg6] : memref<10x10xf32>
          %2 = arith.mulf %0, %1 : f32
          affine.store %2, %arg3[%arg5, %arg6] : memref<10x10xf32>
          %3 = affine.load %arg0[%arg5, %arg6] : memref<10x10xf32>
          %4 = affine.load %arg2[%arg5, %arg6] : memref<10x10xf32>
          %5 = arith.addf %3, %4 : f32
          affine.store %5, %arg4[%arg5, %arg6] : memref<10x10xf32>
        }
      }
      return
    }
    ```
  }];
  let constructor = "mlir::affine::createLoopFusionPass()";
  let options = [
    Option<"computeToleranceThreshold", "fusion-compute-tolerance", "double",
           /*default=*/"0.30f", "Fractional increase in additional computation "
                                "tolerated while fusing">,
    Option<"fastMemorySpace", "fusion-fast-mem-space", "unsigned",
           /*default=*/"0",
           "Faster memory space number to promote fusion buffers to">,
    Option<"localBufSizeThreshold", "fusion-local-buf-threshold", "uint64_t",
           /*default=*/"0", "Threshold size (KiB) for promoting local buffers "
                            "to fast memory space">,
    Option<"maximalFusion", "fusion-maximal", "bool", /*default=*/"false",
           "Enables maximal loop fusion">,
    Option<"affineFusionMode", "mode", "enum FusionMode",
           "mlir::affine::FusionMode::Greedy", "fusion mode to attempt",
           "llvm::cl::values(clEnumValN(mlir::affine::FusionMode::Greedy,"
           " \"greedy\", \"Perform greedy (both producer-consumer and sibling)  fusion\"), "
           "clEnumValN( mlir::affine::FusionMode::ProducerConsumer, "
           "\"producer\", \"Perform only producer-consumer fusion\"), "
           "clEnumValN( mlir::affine::FusionMode::Sibling, "
           "\"sibling\", \"Perform only sibling fusion\"))">,
    ];
  let dependentDialects = ["memref::MemRefDialect"];
}

def AffineLoopInvariantCodeMotion
    : Pass<"affine-loop-invariant-code-motion", "func::FuncOp"> {
  let summary = "Hoist loop invariant instructions outside of affine loops";
  let constructor = "mlir::affine::createAffineLoopInvariantCodeMotionPass()";
}

def AffineLoopTiling : Pass<"affine-loop-tile", "func::FuncOp"> {
  let summary = "Tile affine loop nests";
  let constructor = "mlir::affine::createLoopTilingPass()";
  let options = [
    Option<"cacheSizeInKiB", "cache-size", "uint64_t", /*default=*/"512",
           "Set size of cache to tile for in KiB (default: 512)">,
    Option<"separate", "separate", "bool", /*default=*/"false",
           "Separate full and partial tiles (default: false)">,
    Option<"tileSize", "tile-size", "unsigned", /*default=*/"",
           "Use this tile size for all loops">,
    ListOption<"tileSizes", "tile-sizes", "unsigned",
               "List of tile sizes for each perfect nest "
               "(overridden by -tile-size)">,
  ];
}

def AffineLoopUnroll : Pass<"affine-loop-unroll", "func::FuncOp"> {
  let summary = "Unroll affine loops";
  let constructor = "mlir::affine::createLoopUnrollPass()";
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
    Option<"cleanUpUnroll", "cleanup-unroll", "bool", /*default=*/"false",
           "Fully unroll the cleanup loop when possible.">,
  ];
}

def AffineLoopUnrollAndJam : Pass<"affine-loop-unroll-jam", "func::FuncOp"> {
  let summary = "Unroll and jam affine loops";
  let constructor = "mlir::affine::createLoopUnrollAndJamPass()";
  let options = [
    Option<"unrollJamFactor", "unroll-jam-factor", "unsigned",
           /*default=*/"4",
           "Use this unroll jam factor for all loops (default 4)">,
  ];
}

def AffinePipelineDataTransfer
    : Pass<"affine-pipeline-data-transfer", "func::FuncOp"> {
  let summary = "Pipeline non-blocking data transfers between explicitly "
                "managed levels of the memory hierarchy";
  let description = [{
    This pass performs a transformation to overlap non-blocking DMA operations
    in a loop with computations through double buffering. This is achieved by
    advancing dma_start operations with respect to other operations.

    Input

    ```mlir
    func.func @pipelinedatatransfer() {
      %0 = memref.alloc() : memref<256xf32>
      %1 = memref.alloc() : memref<32xf32, 1>
      %2 = memref.alloc() : memref<1xf32>
      %c0 = arith.constant 0 : index
      %c128 = arith.constant 128 : index
      affine.for %i0 = 0 to 8 {
        affine.dma_start %0[%i0], %1[%i0], %2[%c0], %c128 : memref<256xf32>, memref<32xf32, 1>, memref<1xf32>
        affine.dma_wait %2[%c0], %c128 : memref<1xf32>
        %3 = affine.load %1[%i0] : memref<32xf32, 1>
        %4 = "compute"(%3) : (f32) -> f32
        affine.store %4, %1[%i0] : memref<32xf32, 1>
      }
      return
    }
    ```

    Output

    ```mlir
    module {
      func.func @pipelinedatatransfer() {
        %c8 = arith.constant 8 : index
        %c0 = arith.constant 0 : index
        %0 = memref.alloc() : memref<256xf32>
        %c0_0 = arith.constant 0 : index
        %c128 = arith.constant 128 : index
        %1 = memref.alloc() : memref<2x32xf32, 1>
        %2 = memref.alloc() : memref<2x1xf32>
        affine.dma_start %0[%c0], %1[%c0 mod 2, %c0], %2[%c0 mod 2, symbol(%c0_0)], %c128 : memref<256xf32>, memref<2x32xf32, 1>, memref<2x1xf32>
        affine.for %arg0 = 1 to 8 {
          affine.dma_start %0[%arg0], %1[%arg0 mod 2, %arg0], %2[%arg0 mod 2, symbol(%c0_0)], %c128 : memref<256xf32>, memref<2x32xf32, 1>, memref<2x1xf32>
          %8 = affine.apply #map3(%arg0)
          %9 = affine.apply #map4(%8)
          %10 = affine.apply #map4(%8)
          affine.dma_wait %2[%8 mod 2, symbol(%c0_0)], %c128 : memref<2x1xf32>
          %11 = affine.load %1[%8 mod 2, %8] : memref<2x32xf32, 1>
          %12 = "compute"(%11) : (f32) -> f32
          affine.store %12, %1[%8 mod 2, %8] : memref<2x32xf32, 1>
        }
        %3 = affine.apply #map3(%c8)
        %4 = affine.apply #map4(%3)
        %5 = affine.apply #map4(%3)
        affine.dma_wait %2[%3 mod 2, symbol(%c0_0)], %c128 : memref<2x1xf32>
        %6 = affine.load %1[%3 mod 2, %3] : memref<2x32xf32, 1>
        %7 = "compute"(%6) : (f32) -> f32
        affine.store %7, %1[%3 mod 2, %3] : memref<2x32xf32, 1>
        memref.dealloc %2 : memref<2x1xf32>
        memref.dealloc %1 : memref<2x32xf32, 1>
        return
      }
    }
    ```
  }];
  let constructor = "mlir::affine::createPipelineDataTransferPass()";
}

def AffineScalarReplacement : Pass<"affine-scalrep", "func::FuncOp"> {
  let summary = "Replace affine memref accesses by scalars by forwarding stores "
                "to loads and eliminating redundant loads";
  let description = [{
    This pass performs store to load forwarding and redundant load elimination
    for affine memref accesses and potentially eliminates the entire memref
    if all its accesses are forwarded.

    Input

    ```mlir
    func.func @store_load_affine_apply() -> memref<10x10xf32> {
      %cf7 = arith.constant 7.0 : f32
      %m = memref.alloc() : memref<10x10xf32>
      affine.for %i0 = 0 to 10 {
        affine.for %i1 = 0 to 10 {
          affine.store %cf7, %m[%i0, %i1] : memref<10x10xf32>
          %v0 = affine.load %m[%i0, %i1] : memref<10x10xf32>
          %v1 = arith.addf %v0, %v0 : f32
        }
      }
      return %m : memref<10x10xf32>
    }
    ```

    Output

    ```mlir
    module {
      func.func @store_load_affine_apply() -> memref<10x10xf32> {
        %cst = arith.constant 7.000000e+00 : f32
        %0 = memref.alloc() : memref<10x10xf32>
        affine.for %arg0 = 0 to 10 {
          affine.for %arg1 = 0 to 10 {
            affine.store %cst, %0[%arg0, %arg1] : memref<10x10xf32>
            %1 = arith.addf %cst, %cst : f32
          }
        }
        return %0 : memref<10x10xf32>
      }
    }
    ```
  }];
  let constructor = "mlir::affine::createAffineScalarReplacementPass()";
}

def AffineVectorize : Pass<"affine-super-vectorize", "func::FuncOp"> {
  let summary = "Vectorize to a target independent n-D vector abstraction";
  let dependentDialects = ["vector::VectorDialect"];
  let options = [
    ListOption<"vectorSizes", "virtual-vector-size", "int64_t",
               "Specify an n-D virtual vector size for vectorization. "
               "This must be greater than zero.">,
    // Optionally, the fixed mapping from loop to fastest varying MemRef
    // dimension for all the MemRefs within a loop pattern:
    //   the index represents the loop depth, the value represents the k^th
    //   fastest varying memory dimension.
    // This is voluntarily restrictive and is meant to precisely target a
    // particular loop/op pair, for testing purposes.
    ListOption<"fastestVaryingPattern", "test-fastest-varying", "int64_t",
               "Specify a 1-D, 2-D or 3-D pattern of fastest varying memory "
               "dimensions to match. See defaultPatterns in Vectorize.cpp for "
               "a description and examples. This is used for testing purposes">,
    Option<"vectorizeReductions", "vectorize-reductions", "bool",
           /*default=*/"false",
           "Vectorize known reductions expressed via iter_args. "
           "Switched off by default.">
  ];
}

def AffineParallelize : Pass<"affine-parallelize", "func::FuncOp"> {
  let summary = "Convert affine.for ops into 1-D affine.parallel";
  let constructor = "mlir::affine::createAffineParallelizePass()";
  let options = [
    Option<"maxNested", "max-nested", "unsigned", /*default=*/"-1u",
           "Maximum number of nested parallel loops to produce. "
           "Defaults to unlimited (UINT_MAX).">,
    Option<"parallelReductions", "parallel-reductions", "bool",
           /*default=*/"false",
           "Whether to parallelize reduction loops. Defaults to false.">
  ];
}

def AffineLoopNormalize : Pass<"affine-loop-normalize", "func::FuncOp"> {
  let summary = "Apply normalization transformations to affine loop-like ops";
  let constructor = "mlir::affine::createAffineLoopNormalizePass()";
  let options = [
    Option<"promoteSingleIter", "promote-single-iter", "bool",
           /*default=*/"true", "Promote single iteration loops">,
  ];
}

def LoopCoalescing : Pass<"affine-loop-coalescing", "func::FuncOp"> {
  let summary = "Coalesce nested loops with independent bounds into a single "
                "loop";
  let constructor = "mlir::affine::createLoopCoalescingPass()";
  let dependentDialects = ["affine::AffineDialect","arith::ArithDialect"];
}

def SimplifyAffineStructures : Pass<"affine-simplify-structures", "func::FuncOp"> {
  let summary = "Simplify affine expressions in maps/sets and normalize "
                "memrefs";
  let constructor = "mlir::affine::createSimplifyAffineStructuresPass()";
}

def AffineExpandIndexOps : Pass<"affine-expand-index-ops"> {
  let summary = "Lower affine operations operating on indices into more fundamental operations";
  let constructor = "mlir::affine::createAffineExpandIndexOpsPass()";
}

def AffineExpandIndexOpsAsAffine : Pass<"affine-expand-index-ops-as-affine"> {
  let summary = "Lower affine operations operating on indices into affine.apply operations";
  let constructor = "mlir::affine::createAffineExpandIndexOpsAsAffinePass()";
}

#endif // MLIR_DIALECT_AFFINE_PASSES
