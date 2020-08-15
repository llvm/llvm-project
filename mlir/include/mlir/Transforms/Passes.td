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

def AffineLoopFusion : FunctionPass<"affine-loop-fusion"> {
  let summary = "Fuse affine loop nests";
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
  ];
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

def BufferPlacement : FunctionPass<"buffer-placement"> {
  let summary = "Optimizes placement of alloc and dealloc operations";
  let description = [{
    This pass implements an algorithm to optimize the placement of alloc and
    dealloc operations. This pass also inserts missing dealloc operations
    automatically to reclaim memory.


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
        linalg.generic {args_in = 1 : i64, args_out = 1 : i64, indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} %arg1, %0 {
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
        %0 = alloc() : memref<2xf32>
        cond_br %arg0, ^bb1, ^bb2
      ^bb1: // pred: ^bb0
        br ^bb3(%arg1 : memref<2xf32>)
      ^bb2: // pred: ^bb0
        linalg.generic {args_in = 1 : i64, args_out = 1 : i64, indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} %arg1, %0 {
        ^bb0(%arg3: f32, %arg4: f32):       // no predecessors
          %2 = exp %arg3 : f32
          linalg.yield %2 : f32
        }: memref<2xf32>, memref<2xf32>
        br ^bb3(%0 : memref<2xf32>)
      ^bb3(%1: memref<2xf32>):      // 2 preds: ^bb1, ^bb2
        linalg.copy(%1, %arg2) : memref<2xf32>, memref<2xf32>
        dealloc %0 : memref<2xf32>
        return
      }
    }
    ```

  }];
  let constructor = "mlir::createBufferPlacementPass()";
}

def Canonicalizer : Pass<"canonicalize"> {
  let summary = "Canonicalize operations";
  let description = [{
    This pass performs various types of canonicalizations over a set of
    operations. See [Operation Canonicalization](Canonicalization.md) for more
    details.
  }];
  let constructor = "mlir::createCanonicalizerPass()";
}

def CopyRemoval : FunctionPass<"copy-removal"> {
  let summary = "Remove the redundant copies from input IR";
  let constructor = "mlir::createCopyRemovalPass()";
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
    Option<"disableCanonicalization", "disable-simplify", "bool",
           /*default=*/"false",
           "Disable running simplifications during inlining">,
    Option<"maxInliningIterations", "max-iterations", "unsigned",
           /*default=*/"4",
           "Maximum number of iterations when inlining within an SCC">,
  ];
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

def MemRefDataFlowOpt : FunctionPass<"memref-dataflow-opt"> {
  let summary = "Perform store/load forwarding for memrefs";
  let description = [{
    This pass performs store to load forwarding for memref's to eliminate memory
    accesses and potentially the entire memref if all its accesses are
    forwarded.

    Input

    ```mlir
    func @store_load_affine_apply() -> memref<10x10xf32> {
      %cf7 = constant 7.0 : f32
      %m = alloc() : memref<10x10xf32>
      affine.for %i0 = 0 to 10 {
        affine.for %i1 = 0 to 10 {
          affine.store %cf7, %m[%i0, %i1] : memref<10x10xf32>
          %v0 = affine.load %m[%i0, %i1] : memref<10x10xf32>
          %v1 = addf %v0, %v0 : f32
        }
      }
      return %m : memref<10x10xf32>
    }
    ```

    Output

    ```mlir
    module {
      func @store_load_affine_apply() -> memref<10x10xf32> {
        %cst = constant 7.000000e+00 : f32
        %0 = alloc() : memref<10x10xf32>
        affine.for %arg0 = 0 to 10 {
          affine.for %arg1 = 0 to 10 {
            affine.store %cst, %0[%arg0, %arg1] : memref<10x10xf32>
            %1 = addf %cst, %cst : f32
          }
        }
        return %0 : memref<10x10xf32>
      }
    }
    ```
  }];
  let constructor = "mlir::createMemRefDataFlowOptPass()";
}

def NormalizeMemRefs : Pass<"normalize-memrefs", "ModuleOp"> {
  let summary = "Normalize memrefs";
  let constructor = "mlir::createNormalizeMemRefsPass()";
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

def PrintCFG : FunctionPass<"print-cfg-graph"> {
  let summary = "Print CFG graph per-Region";
  let constructor = "mlir::createPrintCFGGraphPass()";
}

def PrintOpStats : Pass<"print-op-stats", "ModuleOp"> {
  let summary = "Print statistics of operations";
  let constructor = "mlir::createPrintOpStatsPass()";
}

def PrintOp : Pass<"print-op-graph", "ModuleOp"> {
  let summary = "Print op graph per-Region";
  let constructor = "mlir::createPrintOpGraphPass()";
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
    func @dead_private_function() attributes { sym_visibility = "private" }
    func @live_private_function() attributes { sym_visibility = "private" }

    // Note: The `public` isn't necessary here, as this is the default.
    func @public_function() attributes { sym_visibility = "public" } {
      "foo.return"() {uses = [@live_private_function]} : () -> ()
    }
    ```

    A known live function, `public_function`, contains a reference to an
    otherwise non-live function `live_private_function`. After running
    `symbol-dce`, only these two symbols should remain, as the final symbol
    `dead_private_function` is not visible outside of the current IR and there
    are no links to known-live operations. After running, we get the expected:

    ```mlir
    func @live_private_function() attributes { sym_visibility = "private" }

    func @public_function() attributes { sym_visibility = "public" } {
      "foo.return"() {uses = [@live_private_function]} : () -> ()
    }
    ```

    See [Symbols and SymbolTables](SymbolsAndSymbolTables.md) for more
    information on `Symbols`.
  }];
  let constructor = "mlir::createSymbolDCEPass()";
}
#endif // MLIR_TRANSFORMS_PASSES
