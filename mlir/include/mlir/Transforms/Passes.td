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

def AffinePipelineDataTransfer : Pass<
    "affine-pipeline-data-transfer"> {
  let summary = "Pipeline non-blocking data transfers between explicitly "
                "managed levels of the memory hierarchy";
  let constructor = "mlir::createPipelineDataTransferPass()";
}

def AffineLoopFusion : Pass<"affine-loop-fusion"> {
  let summary = "Fuse affine loop nests";
  let constructor = "mlir::createLoopFusionPass()";
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
}

def LocationSnapshot : Pass<"snapshot-op-locations"> {
  let summary = "Generate new locations from the current IR";
  let constructor = "mlir::createLocationSnapshotPass()";
  let options = [
    Option<"fileName", "filename", "std::string", /*default=*/"",
           "The filename to print the generated IR.">,
    Option<"tag", "tag", "std::string", /*default=*/"",
           "A tag to use when fusing the new locations with the "
           "original. If unset, the locations are replaced.">,
  ];
}

def LoopCoalescing : Pass<"loop-coalescing"> {
  let summary = "Coalesce nested loops with independent bounds into a single "
                "loop";
  let constructor = "mlir::createLoopCoalescingPass()";
}

def LoopInvariantCodeMotion : Pass<"loop-invariant-code-motion"> {
  let summary = "Hoist loop invariant instructions outside of the loop";
  let constructor = "mlir::createLoopInvariantCodeMotionPass()";
}

def MemRefDataFlowOpt : Pass<"memref-dataflow-opt"> {
  let summary = "Perform store/load forwarding for memrefs";
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

def PrintCFG : Pass<"print-cfg-graph"> {
  let summary = "Print CFG graph per-Region";
  let constructor = "mlir::createPrintCFGGraphPass()";
}

def PrintOpStats : Pass<"print-op-stats"> {
  let summary = "Print statistics of operations";
  let constructor = "mlir::createPrintOpStatsPass()";
}

def PrintOp : Pass<"print-op-graph"> {
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
