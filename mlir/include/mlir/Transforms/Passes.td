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

def LoopInvariantCodeMotion : Pass<"loop-invariant-code-motion"> {
  let summary = "Hoist loop invariant instructions outside of the loop";
  let constructor = "mlir::createLoopInvariantCodeMotionPass()";
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

def SymbolPrivatize : Pass<"symbol-privatize"> {
  let summary = "Mark symbols private";
  let description = [{
    This pass marks all top-level symbols of the operation run as `private`
    except if listed in `exclude` pass option.
  }];
  let options = [
    ListOption<"exclude", "exclude", "std::string",
       "Comma separated list of symbols that should not be marked private",
       "llvm::cl::MiscFlags::CommaSeparated">
  ];
  let constructor = "mlir::createSymbolPrivatizePass()";
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
