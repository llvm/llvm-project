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

def AffineLoopFusion : Pass<"affine-loop-fusion", "FuncOp"> {
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
    func @producer_consumer_fusion(%arg0: memref<10xf32>, %arg1: memref<10xf32>) {
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
    func @sibling_fusion(%arg0: memref<10x10xf32>, %arg1: memref<10x10xf32>,
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
    func @sibling_fusion(%arg0: memref<10x10xf32>, %arg1: memref<10x10xf32>,
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
    : Pass<"affine-pipeline-data-transfer", "FuncOp"> {
  let summary = "Pipeline non-blocking data transfers between explicitly "
                "managed levels of the memory hierarchy";
  let description = [{
    This pass performs a transformation to overlap non-blocking DMA operations
    in a loop with computations through double buffering. This is achieved by
    advancing dma_start operations with respect to other operations.

    Input

    ```mlir
    func @pipelinedatatransfer() {
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
      func @pipelinedatatransfer() {
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
  let constructor = "mlir::createPipelineDataTransferPass()";
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

def ControlFlowSink : Pass<"control-flow-sink"> {
  let summary = "Sink operations into conditional blocks";
  let description = [{
    This pass implements control-flow sink on operations that implement
    `RegionBranchOpInterface` by moving dominating operations whose only uses
    are in a conditionally-executed regions into those regions so that
    executions paths where their results are not needed do not perform
    unnecessary computations.

    This is similar (but opposite) to loop-invariant code motion, which hoists
    operations out of regions executed more than once. The implementation of
    control-flow sink uses a simple and conversative cost model: operations are
    never duplicated and are only moved into singly-executed regions.

    It is recommended to run canonicalization first to remove unreachable
    blocks: ops in unreachable blocks may prevent other operations from being
    sunk as they may contain uses of their results
  }];
  let constructor = "::mlir::createControlFlowSinkPass()";
  let statistics = [
    Statistic<"numSunk", "num-sunk", "Number of operations sunk">,
  ];
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

def LoopCoalescing : Pass<"loop-coalescing", "FuncOp"> {
  let summary = "Coalesce nested loops with independent bounds into a single "
                "loop";
  let constructor = "mlir::createLoopCoalescingPass()";
  let dependentDialects = ["arith::ArithmeticDialect"];
}

def LoopInvariantCodeMotion : Pass<"loop-invariant-code-motion"> {
  let summary = "Hoist loop invariant instructions outside of the loop";
  let constructor = "mlir::createLoopInvariantCodeMotionPass()";
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

def ViewOpGraph : Pass<"view-op-graph"> {
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
