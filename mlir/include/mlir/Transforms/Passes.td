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
    operations by iteratively applying the canonicalization patterns of all
    loaded dialects until either a fixpoint is reached or the maximum number of
    iterations/rewrites is exhausted. Canonicalization is best-effort and does
    not guarantee that the entire IR is in a canonical form after running this
    pass. See [Operation Canonicalization](Canonicalization.md) for more
    details.
  }];
  let constructor = "mlir::createCanonicalizerPass()";
  let options = [
    Option<"topDownProcessingEnabled", "top-down", "bool",
           /*default=*/"true",
           "Seed the worklist in general top-down order">,
    Option<"enableRegionSimplification", "region-simplify", "bool",
           /*default=*/"true",
           "Perform control flow optimizations to the region tree">,
    Option<"maxIterations", "max-iterations", "int64_t",
           /*default=*/"10",
           "Max. iterations between applying patterns / simplifying regions">,
    Option<"maxNumRewrites", "max-num-rewrites", "int64_t", /*default=*/"-1",
           "Max. number of pattern rewrites within an iteration">,
    Option<"testConvergence", "test-convergence", "bool", /*default=*/"false",
           "Test only: Fail pass on non-convergence to detect cyclic pattern">
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

def PrintIRPass : Pass<"print-ir"> {
  let summary = "Print IR on the debug stream";
  let description = [{
    Print the entire IR on the debug stream. This is meant for debugging
    purposes to inspect the IR at a specific point in the pipeline.
  }];
  let constructor = "mlir::createPrintIRPass()";
  let options = [
    Option<"label", "label", "std::string", /*default=*/"", "Label">,
  ];
}

def GenerateRuntimeVerification : Pass<"generate-runtime-verification"> {
  let summary = "Generate additional runtime op verification checks";
  let description = [{
    This pass generates op-specific runtime checks using the
    `RuntimeVerifiableOpInterface`. It can be run for debugging purposes after
    passes that are suspected to introduce faulty IR.
  }];
  let constructor = "mlir::createGenerateRuntimeVerificationPass()";
}

def Inliner : Pass<"inline"> {
  let summary = "Inline function calls";
  let constructor = "mlir::createInlinerPass()";
  let options = [
    Option<"defaultPipelineStr", "default-pipeline", "std::string",
           /*default=*/"\"canonicalize\"", "The default optimizer pipeline used for callables">,
    ListOption<"opPipelineList", "op-pipelines", "OpPassManager",
               "Callable operation specific optimizer pipelines (in the form "
               "of `dialect.op(pipeline)`)">,
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
    of a [`Name Location`](Dialects/Builtin.md/#nameloc) with the specified tag.

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

def Mem2Reg : Pass<"mem2reg"> {
  let summary = "Promotes memory slots into values.";
  let description = [{
    This pass removes loads out of and stores into a memory slot, and turns
    them into direct uses of SSA values. This is done generically using the
    `PromoteAllocationOpInterface`, `PromoteOpInterface` and
    `PromoteMemOpInterface` interfaces.

    This pass will attempt to compute which definitions of the content of
    the memory slot reach operations that use the memory slot pointer. It
    will rewire or remove operations that use the slot pointer so they no
    longer use it. If any of this is not possible, the IR will be left
    without mutation.

    This pass only supports unstructured control-flow. Promotion of operations
    within subregions will not happen.
  }];

  let options = [
    Option<"enableRegionSimplification", "region-simplify", "bool",
       /*default=*/"true",
       "Perform control flow optimizations to the region tree">,
  ];

  let statistics = [
    Statistic<"promotedAmount",
              "promoted slots",
              "Total amount of memory slot promoted">,
    Statistic<"newBlockArgumentAmount",
              "new block args",
              "Total amount of new block argument inserted in blocks">,
  ];
}

def PrintOpStats : Pass<"print-op-stats"> {
  let summary = "Print statistics of operations";
  let constructor = "mlir::createPrintOpStatsPass()";
  let options = [
    Option<"printAsJSON", "json", "bool", /*default=*/"false",
           "print the stats as JSON">
  ];
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

def SROA : Pass<"sroa"> {
  let summary = "Scalar Replacement of Aggregates";
  let description = [{
    Scalar Replacement of Aggregates. Replaces allocations of aggregates into
    independant allocations of its elements.

    Allocators must implement `DestructurableAllocationOpInterface` to provide
    the list of memory slots for which destructuring should be attempted.

    This pass will only be applied if all accessors of the aggregate implement
    the `DestructurableAccessorOpInterface`. If the accessors provide a view
    into the struct, users of the view must ensure it is used in a type-safe
    manner and within bounds by implementing `TypeSafeOpInterface`.
  }];

  let statistics = [
    Statistic<
      "destructuredAmount",
      "destructured slots",
      "Total amount of memory slots destructured"
    >,
    Statistic<
      "slotsWithMemoryBenefit",
      "slots with memory benefit",
      "Total amount of memory slots in which the destructured size was smaller "
      "than the total size after eliminating unused fields"
    >,
    Statistic<
      "maxSubelementAmount",
      "max subelement number",
      "Maximal number of sub-elements a successfully destructured slot "
      "initially had"
    >,
  ];
}

def StripDebugInfo : Pass<"strip-debuginfo"> {
  let summary = "Strip debug info from all operations";
  let description = [{
    This pass strips the IR of any location information, by replacing all
    operation locations with [`unknown`](Dialects/Builtin.md/#unknownloc).
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
    [visibility](SymbolsAndSymbolTables.md/#symbol-visibility) that extends
    beyond the IR, e.g. `public`, or those that are referenced by live symbols
    or other non-Symbol operations.

    For example, consider the following input:

    ```mlir
    func.func private @dead_private_function()
    func.func private @live_private_function()

    // Note: The `public` isn't necessary here, as this is the default.
    func.func public @public_function() {
      "foo.return"() {uses = [@live_private_function]} : () -> ()
    }
    ```

    A known live function, `public_function`, contains a reference to an
    otherwise non-live function `live_private_function`. After running
    `symbol-dce`, only these two symbols should remain, as the final symbol
    `dead_private_function` is not visible outside of the current IR and there
    are no links to known-live operations. After running, we get the expected:

    ```mlir
    func.func private @live_private_function()

    func.func public @public_function() {
      "foo.return"() {uses = [@live_private_function]} : () -> ()
    }
    ```

    See [Symbols and SymbolTables](SymbolsAndSymbolTables.md) for more
    information on `Symbols`.
  }];
  let constructor = "mlir::createSymbolDCEPass()";

  let statistics = [
    Statistic<"numDCE", "num-dce'd", "Number of symbols DCE'd">,
  ];
}

def SymbolPrivatize : Pass<"symbol-privatize"> {
  let summary = "Mark symbols private";
  let description = [{
    This pass marks all top-level symbols of the operation run as `private`
    except if listed in `exclude` pass option.
  }];
  let options = [
    ListOption<"exclude", "exclude", "std::string",
       "Comma separated list of symbols that should not be marked private">
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

def TopologicalSort : Pass<"topological-sort"> {
  let summary = "Sort regions without SSA dominance in topological order";
  let description = [{
    Recursively sorts all nested regions without SSA dominance in topological
    order. The main purpose is readability, as well as potentially processing of
    certain transformations and analyses. The function sorts the operations in
    all nested regions such that, as much as possible, all users appear after
    their producers.

    This sort is stable. If the block is already topologically sorted, the IR
    is not changed. Operations that form a cycle are moved to the end of the
    regions in a stable order.
  }];

  let constructor = "mlir::createTopologicalSortPass()";
}

#endif // MLIR_TRANSFORMS_PASSES
