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

def Canonicalizer : Pass<"canonicalize"> {
  let summary = "Canonicalize operations";
  let constructor = "mlir::createCanonicalizerPass()";
}

def CSE : Pass<"cse"> {
  let summary = "Eliminate common sub-expressions";
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

def StripDebugInfo : Pass<"strip-debuginfo"> {
  let summary = "Strip debug info from all operations";
  let constructor = "mlir::createStripDebugInfoPass()";
}

def SymbolDCE : Pass<"symbol-dce"> {
  let summary = "Eliminate dead symbols";
  let constructor = "mlir::createSymbolDCEPass()";
}

#endif // MLIR_TRANSFORMS_PASSES
