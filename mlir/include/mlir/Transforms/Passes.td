//===-- Passes.td - Transforms pass definition file --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions for passes within the Transforms/ directory.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TRANSFORMS_PASSES
#define MLIR_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"
include "mlir/Rewrite/PassUtil.td"

def AffineLoopFusion : FunctionPass<"affine-loop-fusion"> {
  let summary = "Fuse affine loop nests";
  let description = [{
    This pass performs fusion of loop nests using a slicing-based approach. It
    combines two fusion strategies: producer-consumer fusion and sibling fusion.
    Producer-consumer fusion is aimed at fusing pairs of loops where the first
    one writes to a memref that the second reads. Sibling fusion targets pairs
    of loops that share no dependences between them but that load from the same
    memref. The fused loop nests, when possible, are rewritten to access
    significantly smaller local buffers instead of the original memref's, and
    the latter are often either completely optimized away or contracted. This
    transformation leads to enhanced locality and lower memory footprint through
    the elimination or contraction of temporaries/intermediate memref's. These
    benefits are sometimes achieved at the expense of redundant computation
    through a cost model that evaluates available choices such as the depth at
    which a source slice should be materialized in the designation slice.

    Example 1: Producer-consumer fusion.
    Input:
    ```mlir
    func @producer_consumer_fusion(%arg0: memref<10xf32>, %arg1: memref<10xf32>) {
      %0 = alloc() : memref<10xf32>
      %1 = alloc() : memref<10xf32>
      %cst = constant 0.000000e+00 : f32
      affine.for %arg2 = 0 to 10 {
        affine.store %cst, %0[%arg2] : memref<10xf32>
        affine.store %cst, %1[%arg2] : memref<10xf32>
      }
      affine.for %arg2 = 0 to 10 {
        %2 = affine.load %0[%arg2] : memref<10xf32>
        %3 = addf %2, %2 : f32
        affine.store %3, %arg0[%arg2] : memref<10xf32>
      }
      affine.for %arg2 = 0 to 10 {
        %2 = affine.load %1[%arg2] : memref<10xf32>
        %3 = mulf %2, %2 : f32
        affine.store %3, %arg1[%arg2] : memref<10xf32>
      }
      return
    }
    ```
    Output:
    ```mlir
    func @producer_consumer_fusion(%arg0: memref<10xf32>, %arg1: memref<10xf32>) {
      %0 = alloc() : memref<1xf32>
      %1 = alloc() : memref<1xf32>
      %cst = constant 0.000000e+00 : f32
      affine.for %arg2 = 0 to 10 {
        affine.store %cst, %0[0] : memref<1xf32>
        affine.store %cst, %1[0] : memref<1xf32>
        %2 = affine.load %1[0] : memref<1xf32>
        %3 = mulf %2, %2 : f32
        affine.store %3, %arg1[%arg2] : memref<10xf32>
        %4 = affine.load %0[0] : memref<1xf32>
        %5 = addf %4, %4 : f32
        affine.store %5, %arg0[%arg2] : memref<10xf32>
      }
      return
    }
    ```

    Example 2: Sibling fusion.
    Input:
    ```mlir
    func @sibling_fusion(%arg0: memref<10x10xf32>, %arg1: memref<10x10xf32>,
                         %arg2: memref<10x10xf32>, %arg3: memref<10x10xf32>,
                         %arg4: memref<10x10xf32>) {
      affine.for %arg5 = 0 to 3 {
        affine.for %arg6 = 0 to 3 {
          %0 = affine.load %arg0[%arg5, %arg6] : memref<10x10xf32>
          %1 = affine.load %arg1[%arg5, %arg6] : memref<10x10xf32>
          %2 = mulf %0, %1 : f32
          affine.store %2, %arg3[%arg5, %arg6] : memref<10x10xf32>
        }
      }
      affine.for %arg5 = 0 to 3 {
        affine.for %arg6 = 0 to 3 {
          %0 = affine.load %arg0[%arg5, %arg6] : memref<10x10xf32>
          %1 = affine.load %arg2[%arg5, %arg6] : memref<10x10xf32>
          %2 = addf %0, %1 : f32
          affine.store %2, %arg4[%arg5, %arg6] : memref<10x10xf32>
        }
      }
      return
    }
    ```
    Output:
    ```mlir
    func @sibling_fusion(%arg0: memref<10x10xf32>, %arg1: memref<10x10xf32>,
                         %arg2: memref<10x10xf32>, %arg3: memref<10x10xf32>,
                         %arg4: memref<10x10xf32>) {
      affine.for %arg5 = 0 to 3 {
        affine.for %arg6 = 0 to 3 {
          %0 = affine.load %arg0[%arg5, %arg6] : memref<10x10xf32>
          %1 = affine.load %arg1[%arg5, %arg6] : memref<10x10xf32>
          %2 = mulf %0, %1 : f32
          affine.store %2, %arg3[%arg5, %arg6] : memref<10x10xf32>
          %3 = affine.load %arg0[%arg5, %arg6] : memref<10x10xf32>
          %4 = affine.load %arg2[%arg5, %arg6] : memref<10x10xf32>
          %5 = addf %3, %4 : f32
          affine.store %5, %arg4[%arg5, %arg6] : memref<10x10xf32>
        }
      }
      return
    }
    ```
  }];
  let constructor = "mlir::createLoopFusionPass()";
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
           "mlir::FusionMode::Greedy", "fusion mode to attempt",
           "llvm::cl::values(clEnumValN(mlir::FusionMode::Greedy,"
           " \"greedy\", \"Perform greedy (both producer-consumer and sibling)  fusion\"), "
           "clEnumValN( mlir::FusionMode::ProducerConsumer, "
           "\"producer\", \"Perform only producer-consumer fusion\"), "
           "clEnumValN( mlir::FusionMode::Sibling, "
           "\"sibling\", \"Perform only sibling fusion\"))">,
    ];
  let dependentDialects = ["memref::MemRefDialect"];
}

def AffinePipelineDataTransfer
    : FunctionPass<"affine-pipeline-data-transfer"> {
  let summary = "Pipeline non-blocking data transfers between explicitly "
                "managed levels of the memory hierarchy";
  let description = [{
    This pass performs a transformation to overlap non-blocking DMA operations
    in a loop with computations through double buffering. This is achieved by
    advancing dma_start operations with respect to other operations.

    Input

    ```mlir
    func @pipelinedatatransfer() {
      %0 = alloc() : memref<256xf32>
      %1 = alloc() : memref<32xf32, 1>
      %2 = alloc() : memref<1xf32>
      %c0 = constant 0 : index
      %c128 = constant 128 : index
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
      func @pipelinedatatransfer() {
        %c8 = constant 8 : index
        %c0 = constant 0 : index
        %0 = alloc() : memref<256xf32>
        %c0_0 = constant 0 : index
        %c128 = constant 128 : index
        %1 = alloc() : memref<2x32xf32, 1>
        %2 = alloc() : memref<2x1xf32>
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
        dealloc %2 : memref<2x1xf32>
        dealloc %1 : memref<2x32xf32, 1>
        return
      }
    }
    ```
  }];
  let constructor = "mlir::createPipelineDataTransferPass()";
}

def BufferDeallocation : FunctionPass<"buffer-deallocation"> {
  let summary = "Adds all required dealloc operations for all allocations in the "
                "input program";
  let description = [{
    This pass implements an algorithm to automatically introduce all required
    deallocation operations for all buffers in the input program. This ensures that
    the resulting program does not have any memory leaks.


    Input

    ```mlir
    #map0 = affine_map<(d0) -> (d0)>
    module {
      func @condBranch(%arg0: i1, %arg1: memref<2xf32>, %arg2: memref<2xf32>) {
        cond_br %arg0, ^bb1, ^bb2
      ^bb1:
        br ^bb3(%arg1 : memref<2xf32>)
      ^bb2:
        %0 = alloc() : memref<2xf32>
        linalg.generic {
          args_in = 1 : i64,
          args_out = 1 : i64,
          indexing_maps = [#map0, #map0],
          iterator_types = ["parallel"]} %arg1, %0 {
        ^bb0(%gen1_arg0: f32, %gen1_arg1: f32):
          %tmp1 = exp %gen1_arg0 : f32
          linalg.yield %tmp1 : f32
        }: memref<2xf32>, memref<2xf32>
        br ^bb3(%0 : memref<2xf32>)
      ^bb3(%1: memref<2xf32>):
        "linalg.copy"(%1, %arg2) : (memref<2xf32>, memref<2xf32>) -> ()
        return
      }
    }

    ```

    Output

    ```mlir
    #map0 = affine_map<(d0) -> (d0)>
    module {
      func @condBranch(%arg0: i1, %arg1: memref<2xf32>, %arg2: memref<2xf32>) {
        cond_br %arg0, ^bb1, ^bb2
      ^bb1:  // pred: ^bb0
        %0 = alloc() : memref<2xf32>
        linalg.copy(%arg1, %0) : memref<2xf32>, memref<2xf32>
        br ^bb3(%0 : memref<2xf32>)
      ^bb2:  // pred: ^bb0
        %1 = alloc() : memref<2xf32>
        linalg.generic {
          args_in = 1 : i64,
          args_out = 1 : i64,
          indexing_maps = [#map0, #map0],
          iterator_types = ["parallel"]} %arg1, %1 {
        ^bb0(%arg3: f32, %arg4: f32):  // no predecessors
          %4 = exp %arg3 : f32
          linalg.yield %4 : f32
        }: memref<2xf32>, memref<2xf32>
        %2 = alloc() : memref<2xf32>
        linalg.copy(%1, %2) : memref<2xf32>, memref<2xf32>
        dealloc %1 : memref<2xf32>
        br ^bb3(%2 : memref<2xf32>)
      ^bb3(%3: memref<2xf32>):  // 2 preds: ^bb1, ^bb2
        linalg.copy(%3, %arg2) : memref<2xf32>, memref<2xf32>
        dealloc %3 : memref<2xf32>
        return
      }

    }
    ```

  }];
  let constructor = "mlir::createBufferDeallocationPass()";
}

def BufferHoisting : FunctionPass<"buffer-hoisting"> {
  let summary = "Optimizes placement of allocation operations by moving them "
                "into common dominators and out of nested regions";
  let description = [{
    This pass implements an approach to aggressively move allocations upwards
    into common dominators and out of nested regions.
  }];
  let constructor = "mlir::createBufferHoistingPass()";
}

def BufferLoopHoisting : FunctionPass<"buffer-loop-hoisting"> {
  let summary = "Optimizes placement of allocation operations by moving them "
                "out of loop nests";
  let description = [{
    This pass implements an approach to aggressively move allocations upwards
    out of loop nests. It does not move allocations into common dominators.
  }];
  let constructor = "mlir::createBufferLoopHoistingPass()";
}

def PromoteBuffersToStack : FunctionPass<"promote-buffers-to-stack"> {
  let summary = "Promotes heap-based allocations to automatically managed "
                "stack-based allocations";
  let description = [{
    This pass implements a simple algorithm to convert heap-based memory
    allocations to stack-based ones. It uses a built-in heuristic to decide
    whether it makes sense to convert an allocation. Furthermore, dynamic
    shaped buffers that are limited by the rank of the tensor can be
    converted. They are only transformed if they are considered to be small.
  }];
  let constructor = "mlir::createPromoteBuffersToStackPass()";
  let options = [
    Option<"maxAllocSizeInBytes", "max-alloc-size-in-bytes", "unsigned",
           /*default=*/"1024",
           "Maximal size in bytes to promote allocations to stack.">,
    Option<"bitwidthOfIndexType", "bitwidth-of-index-type", "unsigned",
           /*default=*/"64",
           "Bitwidth of the index type. Used for size estimation.">,
    Option<"maxRankOfAllocatedMemRef", "max-rank-of-allocated-memref", "unsigned",
           /*default=*/"1",
           "Maximal memref rank to promote dynamic buffers.">,
  ];
}

def BufferResultsToOutParams : Pass<"buffer-results-to-out-params", "ModuleOp">  {
  let summary = "Converts memref-typed function results to out-params";
  let description = [{
    Some calling conventions prefer to pass output memrefs as "out params". The
    conversion to this calling convention must be done as an atomic
    transformation of the entire program (hence this is a module pass).

    For example, if a call is rewritten, the callee needs to be rewritten
    otherwise the IR will end up invalid. Thus, this transformation
    require an atomic change to the entire program (e.g. the whole module).

    This pass is expected to run immediately after bufferization is finished.
    At that point, tensor-typed results will have been converted to memref-typed
    results, and can be consistently converted to out params.

    All memref-typed results are appended to the function argument list.

    The main issue with this pass (and the out-param calling convention) is that
    buffers for results need to be allocated in the caller. This currently only
    works for static shaped memrefs.
  }];
  let constructor = "mlir::createBufferResultsToOutParamsPass()";
  let dependentDialects = ["memref::MemRefDialect"];
}

def Canonicalizer : Pass<"canonicalize"> {
  let summary = "Canonicalize operations";
  let description = [{
    This pass performs various types of canonicalizations over a set of
    operations. See [Operation Canonicalization](Canonicalization.md) for more
    details.
  }];
  let constructor = "mlir::createCanonicalizerPass()";
  let options = [
    Option<"topDownProcessingEnabled", "top-down", "bool",
           /*default=*/"true",
           "Seed the worklist in general top-down order">,
    Option<"enableRegionSimplification", "region-simplify", "bool",
           /*default=*/"true",
           "Seed the worklist in general top-down order">,
    Option<"maxIterations", "max-iterations", "int64_t",
           /*default=*/"10",
           "Seed the worklist in general top-down order">
  ] # RewritePassUtils.options;
}

def CSE : Pass<"cse"> {
  let summary = "Eliminate common sub-expressions";
  let description = [{
    This pass implements a generalized algorithm for common sub-expression
    elimination. This pass relies on information provided by the
    `Memory SideEffect` interface to identify when it is safe to eliminate
    operations. See [Common subexpression elimination](https://en.wikipedia.org/wiki/Common_subexpression_elimination)
    for more general details on this optimization.
  }];
  let constructor = "mlir::createCSEPass()";
  let statistics = [
    Statistic<"numCSE", "num-cse'd", "Number of operations CSE'd">,
    Statistic<"numDCE", "num-dce'd", "Number of operations DCE'd">
  ];
}

def Inliner : Pass<"inline"> {
  let summary = "Inline function calls";
  let constructor = "mlir::createInlinerPass()";
  let options = [
    Option<"defaultPipelineStr", "default-pipeline", "std::string",
           /*default=*/"", "The default optimizer pipeline used for callables">,
    ListOption<"opPipelineStrs", "op-pipelines", "std::string",
               "Callable operation specific optimizer pipelines (in the form "
               "of `dialect.op(pipeline)`)",
               "llvm::cl::MiscFlags::CommaSeparated">,
    Option<"maxInliningIterations", "max-iterations", "unsigned",
           /*default=*/"4",
           "Maximum number of iterations when inlining within an SCC">,
  ];
}

def FinalizingBufferize : FunctionPass<"finalizing-bufferize"> {
  let summary = "Finalize a partial bufferization";
  let description = [{
    A bufferize pass that finalizes a partial bufferization by removing
    remaining `memref.tensor_load` and `memref.buffer_cast` operations.

    The removal of those operations is only possible if the operations only
    exist in pairs, i.e., all uses of `memref.tensor_load` operations are
    `memref.buffer_cast` operations.

    This pass will fail if not all operations can be removed or if any operation
    with tensor typed operands remains.
  }];
  let constructor = "mlir::createFinalizingBufferizePass()";
}

def LocationSnapshot : Pass<"snapshot-op-locations"> {
  let summary = "Generate new locations from the current IR";
  let description = [{
    This pass allows for generating new locations from the IR during any stage
    of compilation, by snapshotting the IR to a file and using that file to
    generate new locations for the operations.

    Depending on the value of the `tag` option, different resulting locations
    may be generated:

    * If unset, the original location of the operation is replaced.

    Example:

    ```mlir
    // old:
    ... loc("original_source.cpp":1:1)

    // new:
    ... loc("snapshot_source.mlir":10:10)
    ```

    * If set, the new location is fused with the original location in the form
    of a [`Name Location`](Diagnostics.md#name-location) with the specified tag.

    Example:

    ```mlir
    // old:
    ... loc("original_source.cpp":1:1)

    // new:
    ... loc(fused["original_source.cpp":1:1, "snapshot"("snapshot_source.mlir":10:10)])
    ```
  }];
  let constructor = "mlir::createLocationSnapshotPass()";
  let options = [
    Option<"fileName", "filename", "std::string", /*default=*/"",
           "The filename to print the generated IR">,
    Option<"tag", "tag", "std::string", /*default=*/"",
           "A tag to use when fusing the new locations with the "
           "original. If unset, the locations are replaced.">,
  ];
}

def LoopCoalescing : FunctionPass<"loop-coalescing"> {
  let summary = "Coalesce nested loops with independent bounds into a single "
                "loop";
  let constructor = "mlir::createLoopCoalescingPass()";
}

def LoopInvariantCodeMotion : Pass<"loop-invariant-code-motion"> {
  let summary = "Hoist loop invariant instructions outside of the loop";
  let constructor = "mlir::createLoopInvariantCodeMotionPass()";
}

def NormalizeMemRefs : Pass<"normalize-memrefs", "ModuleOp"> {
  let summary = "Normalize memrefs";
   let description = [{
    This pass transforms memref types with a non-trivial
    [layout map](https://mlir.llvm.org/docs/LangRef/#layout-map) into
    memref types with an identity layout map, e.g. (i, j) -> (i, j). This
    pass is inter-procedural, in the sense that it can modify function
    interfaces and call sites that pass memref types. In order to modify
    memref types while preserving the original behavior, users of those
    memref types are also modified to incorporate the resulting layout map.
    For instance, an [AffineLoadOp]
    (https://mlir.llvm.org/docs/Dialects/Affine/#affineload-affineloadop)
    will be updated to compose the layout map with with the affine expression
    contained in the op. Operations marked with the [MemRefsNormalizable]
    (https://mlir.llvm.org/docs/Traits/#memrefsnormalizable) trait are
    expected to be normalizable. Supported operations include affine
    operations, memref.alloc, memref.dealloc, and std.return.

    Given an appropriate layout map specified in the code, this transformation
    can express tiled or linearized access to multi-dimensional data
    structures, but will not modify memref types without an explicit layout
    map.

    Currently this pass is limited to only modify
    functions where all memref types can be normalized. If a function
    contains any operations that are not MemRefNormalizable, then the function
    and any functions that call or call it will not be modified.

    Input

    ```mlir
    #tile = affine_map<(i) -> (i floordiv 4, i mod 4)>
    func @matmul(%A: memref<16xf64, #tile>,
                 %B: index, %C: memref<16xf64>) -> (memref<16xf64, #tile>) {
      affine.for %arg3 = 0 to 16 {
            %a = affine.load %A[%arg3] : memref<16xf64, #tile>
            %p = mulf %a, %a : f64
            affine.store %p, %A[%arg3] : memref<16xf64, #tile>
      }
      %c = alloc() : memref<16xf64, #tile>
      %d = affine.load %c[0] : memref<16xf64, #tile>
      return %A: memref<16xf64, #tile>
    }
    ```

    Output

    ```mlir
    func @matmul(%arg0: memref<4x4xf64>, %arg1: index, %arg2: memref<16xf64>)
      -> memref<4x4xf64> {
      affine.for %arg3 = 0 to 16 {
        %3 = affine.load %arg0[%arg3 floordiv 4, %arg3 mod 4]: memref<4x4xf64>
        %4 = mulf %3, %3 : f64
        affine.store %4, %arg0[%arg3 floordiv 4, %arg3 mod 4]: memref<4x4xf64>
      }
      %0 = alloc() : memref<4x4xf64>
      %1 = affine.apply #map1()
      %2 = affine.load %0[0, 0] : memref<4x4xf64>
      return %arg0 : memref<4x4xf64>
    }
    ```

    Input

    ```
    #linear8 = affine_map<(i, j) -> (i * 8 + j)>
    func @linearize(%arg0: memref<8x8xi32, #linear8>,
                    %arg1: memref<8x8xi32, #linear8>,
                    %arg2: memref<8x8xi32, #linear8>) {
      %c8 = constant 8 : index
      %c0 = constant 0 : index
      %c1 = constant 1 : index
      affine.for %arg3 = %c0 to %c8  {
      affine.for %arg4 = %c0 to %c8  {
        affine.for %arg5 = %c0 to %c8 {
          %0 = affine.load %arg0[%arg3, %arg5] : memref<8x8xi32, #linear8>
          %1 = affine.load %arg1[%arg5, %arg4] : memref<8x8xi32, #linear8>
          %2 = affine.load %arg2[%arg3, %arg4] : memref<8x8xi32, #linear8>
          %3 = muli %0, %1 : i32
          %4 = addi %2, %3 : i32
          affine.store %4, %arg2[%arg3, %arg4] : memref<8x8xi32, #linear8>
        }
      }
      }
      return
    }
    ```

    Output

    ```mlir
    func @linearize(%arg0: memref<64xi32>,
                    %arg1: memref<64xi32>,
                    %arg2: memref<64xi32>) {
    %c8 = constant 8 : index
    %c0 = constant 0 : index
    affine.for %arg3 = %c0 to %c8 {
      affine.for %arg4 = %c0 to %c8 {
        affine.for %arg5 = %c0 to %c8 {
          %0 = affine.load %arg0[%arg3 * 8 + %arg5] : memref<64xi32>
          %1 = affine.load %arg1[%arg5 * 8 + %arg4] : memref<64xi32>
          %2 = affine.load %arg2[%arg3 * 8 + %arg4] : memref<64xi32>
          %3 = muli %0, %1 : i32
          %4 = addi %2, %3 : i32
          affine.store %4, %arg2[%arg3 * 8 + %arg4] : memref<64xi32>
        }
      }
    }
    return
  }
  ```
  }];
  let constructor = "mlir::createNormalizeMemRefsPass()";
  let dependentDialects = ["AffineDialect"];
}

def ParallelLoopCollapsing : Pass<"parallel-loop-collapsing"> {
  let summary = "Collapse parallel loops to use less induction variables";
  let constructor = "mlir::createParallelLoopCollapsingPass()";
  let options = [
    ListOption<"clCollapsedIndices0", "collapsed-indices-0", "unsigned",
               "Which loop indices to combine 0th loop index",
               "llvm::cl::MiscFlags::CommaSeparated">,
    ListOption<"clCollapsedIndices1", "collapsed-indices-1", "unsigned",
               "Which loop indices to combine into the position 1 loop index",
               "llvm::cl::MiscFlags::CommaSeparated">,
    ListOption<"clCollapsedIndices2", "collapsed-indices-2", "unsigned",
               "Which loop indices to combine into the position 2 loop index",
               "llvm::cl::MiscFlags::CommaSeparated">,
  ];
}

def PrintOpStats : Pass<"print-op-stats"> {
  let summary = "Print statistics of operations";
  let constructor = "mlir::createPrintOpStatsPass()";
}

def SCCP : Pass<"sccp"> {
  let summary = "Sparse Conditional Constant Propagation";
  let description = [{
    This pass implements a general algorithm for sparse conditional constant
    propagation. This algorithm detects values that are known to be constant and
    optimistically propagates this throughout the IR. Any values proven to be
    constant are replaced, and removed if possible.

    This implementation is based on the algorithm described by Wegman and Zadeck
    in [“Constant Propagation with Conditional Branches”](https://dl.acm.org/doi/10.1145/103135.103136) (1991).
  }];
  let constructor = "mlir::createSCCPPass()";
}

def StripDebugInfo : Pass<"strip-debuginfo"> {
  let summary = "Strip debug info from all operations";
  let description = [{
    This pass strips the IR of any location information, by replacing all
    operation locations with [`unknown`](Diagnostics.md#unknown-location).
  }];
  let constructor = "mlir::createStripDebugInfoPass()";
}

def SymbolDCE : Pass<"symbol-dce"> {
  let summary = "Eliminate dead symbols";
  let description = [{
    This pass deletes all symbols that are found to be unreachable. This is done
    by computing the set of operations that are known to be live, propagating
    that liveness to other symbols, and then deleting all symbols that are not
    within this live set. Live symbols are those that have a
    [visibility](SymbolsAndSymbolTables.md#symbol-visibility) that extends
    beyond the IR, e.g. `public`, or those that are referenced by live symbols
    or other non-Symbol operations.

    For example, consider the following input:

    ```mlir
    func private @dead_private_function()
    func private @live_private_function()

    // Note: The `public` isn't necessary here, as this is the default.
    func public @public_function() {
      "foo.return"() {uses = [@live_private_function]} : () -> ()
    }
    ```

    A known live function, `public_function`, contains a reference to an
    otherwise non-live function `live_private_function`. After running
    `symbol-dce`, only these two symbols should remain, as the final symbol
    `dead_private_function` is not visible outside of the current IR and there
    are no links to known-live operations. After running, we get the expected:

    ```mlir
    func private @live_private_function()

    func public @public_function() {
      "foo.return"() {uses = [@live_private_function]} : () -> ()
    }
    ```

    See [Symbols and SymbolTables](SymbolsAndSymbolTables.md) for more
    information on `Symbols`.
  }];
  let constructor = "mlir::createSymbolDCEPass()";
}

def ViewOpGraphPass : Pass<"view-op-graph"> {
  let summary = "Print Graphviz visualization of an operation";
  let description = [{
    This pass prints a Graphviz graph of a module.

    - Operations are represented as nodes;
    - Uses (data flow) as edges;
    - Control flow as dashed edges;
    - Regions/blocks as subgraphs.

    By default, only data flow edges are printed.

    Note: See https://www.graphviz.org/doc/info/lang.html for more information
    about the Graphviz DOT language.
  }];
  let options = [
    Option<"maxLabelLen", "max-label-len", "unsigned",
            /*default=*/"20", "Limit attribute/type length to number of chars">,
    Option<"printAttrs", "print-attrs", "bool",
           /*default=*/"true", "Print attributes of operations">,
    Option<"printControlFlowEdges", "print-control-flow-edges", "bool",
           /*default=*/"false", "Print control flow edges">,
    Option<"printDataFlowEdges", "print-data-flow-edges", "bool",
           /*default=*/"true", "Print data flow edges">,
    Option<"printResultTypes", "print-result-types", "bool",
            /*default=*/"true", "Print result types of operations">
  ];
  let constructor = "mlir::createPrintOpGraphPass()";
}

#endif // MLIR_TRANSFORMS_PASSES
