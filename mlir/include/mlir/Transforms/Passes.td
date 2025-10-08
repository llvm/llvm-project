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
    Option<"regionSimplifyLevel", "region-simplify", "mlir::GreedySimplifyRegionLevel",
           /*default=*/"mlir::GreedySimplifyRegionLevel::Normal",
           "Perform control flow optimizations to the region tree",
             [{::llvm::cl::values(
               clEnumValN(mlir::GreedySimplifyRegionLevel::Disabled, "disabled",
                "Don't run any control-flow simplification."),
               clEnumValN(mlir::GreedySimplifyRegionLevel::Normal, "normal",
                "Perform simple control-flow simplifications (e.g. dead args elimination)."),
               clEnumValN(mlir::GreedySimplifyRegionLevel::Aggressive, "aggressive",
                "Perform aggressive control-flow simplification (e.g. block merging).")
              )}]>,
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

def RemoveDeadValues : Pass<"remove-dead-values"> {
  let summary = "Remove dead values";
  let description = [{
    The goal of this pass is optimization (reducing runtime) by removing
    unnecessary instructions. Unlike other passes that rely on local information
    gathered from patterns to accomplish optimization, this pass uses a full
    analysis of the IR, specifically, liveness analysis, and is thus more
    powerful.

    Currently, this pass performs the following optimizations:
    (A) Removes function arguments that are not live,
    (B) Removes function return values that are not live across all callers of
    the function,
    (C) Removes unneccesary operands, results, region arguments, and region
    terminator operands of region branch ops, and,
    (D) Removes simple and region branch ops that have all non-live results and
    don't affect memory in any way,

    iff

    the IR doesn't have any non-function symbol ops, non-call symbol user ops
    and branch ops.

    Here, a "simple op" refers to an op that isn't a symbol op, symbol-user op,
    region branch op, branch op, region branch terminator op, or return-like.

    It is noteworthy that we do not refer to non-live values as "dead" in this
    file to avoid confusing it with dead code analysis's "dead", which refers to
    unreachable code (code that never executes on hardware) while "non-live"
    refers to code that executes on hardware but is unnecessary. Thus, while the
    removal of dead code helps little in reducing runtime, removing non-live
    values should theoretically have significant impact (depending on the amount
    removed).

    It is also important to note that unlike other passes (like `canonicalize`)
    that apply op-specific optimizations through patterns, this pass uses
    different interfaces to handle various types of ops and tries to cover all
    existing ops through these interfaces.

    It is because of its reliance on (a) liveness analysis and (b) interfaces
    that makes it so powerful that it can optimize ops that don't have a
    canonicalizer and even when an op does have a canonicalizer, it can perform
    more aggressive optimizations, as observed in the test files associated with
    this pass.

    Example of optimization (A):-

    ```
    int add_2_to_y(int x, int y) {
      return 2 + y
    }

    print(add_2_to_y(3, 4))
    print(add_2_to_y(5, 6))
    ```

    becomes

    ```
    int add_2_to_y(int y) {
      return 2 + y
    }

    print(add_2_to_y(4))
    print(add_2_to_y(6))
    ```

    Example of optimization (B):-

    ```
    int, int get_incremented_values(int y) {
      store y somewhere in memory
      return y + 1, y + 2
    }

    y1, y2 = get_incremented_values(4)
    y3, y4 = get_incremented_values(6)
    print(y2)
    ```

    becomes

    ```
    int get_incremented_values(int y) {
      store y somewhere in memory
      return y + 2
    }

    y2 = get_incremented_values(4)
    y4 = get_incremented_values(6)
    print(y2)
    ```

    Example of optimization (C):-

    Assume only `%result1` is live here. Then,

    ```
    %result1, %result2, %result3 = scf.while (%arg1 = %operand1, %arg2 = %operand2) {
      %terminator_operand2 = add %arg2, %arg2
      %terminator_operand3 = mul %arg2, %arg2
      %terminator_operand4 = add %arg1, %arg1
      scf.condition(%terminator_operand1) %terminator_operand2, %terminator_operand3, %terminator_operand4
    } do {
    ^bb0(%arg3, %arg4, %arg5):
      %terminator_operand6 = add %arg4, %arg4
      %terminator_operand5 = add %arg5, %arg5
      scf.yield %terminator_operand5, %terminator_operand6
    }
    ```

    becomes

    ```
    %result1, %result2 = scf.while (%arg2 = %operand2) {
      %terminator_operand2 = add %arg2, %arg2
      %terminator_operand3 = mul %arg2, %arg2
      scf.condition(%terminator_operand1) %terminator_operand2, %terminator_operand3
    } do {
    ^bb0(%arg3, %arg4):
      %terminator_operand6 = add %arg4, %arg4
      scf.yield %terminator_operand6
    }
    ```

    It is interesting to see that `%result2` won't be removed even though it is
    not live because `%terminator_operand3` forwards to it and cannot be
    removed. And, that is because it also forwards to `%arg4`, which is live.

    Example of optimization (D):-

    ```
    int square_and_double_of_y(int y) {
      square = y ^ 2
      double = y * 2
      return square, double
    }

    sq, do = square_and_double_of_y(5)
    print(do)
    ```

    becomes

    ```
    int square_and_double_of_y(int y) {
      double = y * 2
      return double
    }

    do = square_and_double_of_y(5)
    print(do)
    ```
  }];
  let constructor = "mlir::createRemoveDeadValuesPass()";
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
  let options = [
    Option<"verboseLevel", "verbose-level", "unsigned", /*default=*/"1",
           "Verbosity level for runtime verification messages: "
           "0 = Minimum (only source location), "
           "1 = Detailed (include full operation details, names, types, shapes, etc.)">
  ];
}


def Inliner : Pass<"inline"> {
  let summary = "Inline function calls";
  let constructor = "mlir::createInlinerPass()";
  let options = [
    Option<"defaultPipelineStr", "default-pipeline", "std::string",
           /*default=*/"\"canonicalize\"",
           "The optimizer pipeline used for callables that do not have "
           "a dedicated optimizer pipeline in opPipelineList">,
    ListOption<"opPipelineList", "op-pipelines", "OpPassManager",
               "Callable operation specific optimizer pipelines (in the form "
               "of `dialect.op(pipeline)`)">,
    Option<"maxInliningIterations", "max-iterations", "unsigned",
           /*default=*/"4",
           "Maximum number of iterations when inlining within an SCC">,
    Option<"inliningThreshold", "inlining-threshold", "unsigned",
           /*default=*/"-1U",
           "If the ratio between the number of the operations "
           "in the callee and the number of the operations "
           "in the caller exceeds this value (in percentage), "
           "then the callee is not inlined even if it is legal "
           "to inline it">,
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
  let options = [
    Option<"fileName", "filename", "std::string", /*default=*/"",
           "The filename to print the generated IR">,
    Option<"tag", "tag", "std::string", /*default=*/"",
           "A tag to use when fusing the new locations with the "
           "original. If unset, the locations are replaced.">,
    Option<"enableDebugInfo", "print-debuginfo", "bool", /*default=*/"false",
           "Print debug info in MLIR output">,
    Option<"printGenericOpForm", "print-op-generic", "bool", /*default=*/"false",
           "Print the generic op form">,
    Option<"useLocalScope", "print-local-scope", "bool", /*default=*/"false",
           "Print with local scope and inline information (eliding "
           "aliases for attributes, types, and locations">,
    Option<"printPrettyDebugInfo", "pretty-debuginfo", "bool", /*default=*/"false",
           "Print pretty debug info in MLIR output">,
  ];
}

def LoopInvariantCodeMotion : Pass<"loop-invariant-code-motion"> {
  let summary = "Hoist loop invariant instructions outside of the loop";
  let constructor = "mlir::createLoopInvariantCodeMotionPass()";
}

def LoopInvariantSubsetHoisting : Pass<"loop-invariant-subset-hoisting"> {
  let summary = "Hoist loop invariant subset ops outside of the loop";
  let constructor = "mlir::createLoopInvariantSubsetHoistingPass()";
}

def Mem2Reg : Pass<"mem2reg"> {
  let summary = "Promotes memory slots into values.";
  let description = [{
    This pass removes loads out of and stores into a memory slot, and turns
    them into direct uses of SSA values. This is done generically using the
    `PromotableAllocationOpInterface`, `PromotableOpInterface` and
    `PromotableMemOpInterface` interfaces.

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

def CompositeFixedPointPass : Pass<"composite-fixed-point-pass"> {
  let summary = "Composite fixed point pass";
  let description = [{
    Composite pass runs provided set of passes until fixed point or maximum
    number of iterations reached.
  }];

  let options = [
    Option<"name", "name", "std::string", /*default=*/"\"CompositeFixedPointPass\"",
      "Composite pass display name">,
    Option<"pipelineStr", "pipeline", "std::string", /*default=*/"",
      "Composite pass inner pipeline">,
    Option<"maxIter", "max-iterations", "int", /*default=*/"10",
      "Maximum number of iterations if inner pipeline">,
  ];
}

def BubbleDownMemorySpaceCasts :
    Pass<"bubble-down-memory-space-casts"> {
  let summary = "Bubbles down memory-space cast operations.";
  let description = [{
    This pass tries to iteratively bubble down all possible memory-space cast
    operations. It is important to note that the determination of which casts
    are bubbled down is based on the interfaces
    `MemorySpaceCastConsumerOpInterface`, and `MemorySpaceCastOpInterface`, and
    not the pass. The pass only looks for operations implementing the
    `MemorySpaceCastConsumerOpInterface` interface, and invoking the interface
    methods to perform the bubbling down.

    Example:

    ```mlir
    func.func @op_with_cast_sequence(%arg0: memref<4x4xf32, 1>, %arg1: index, %arg2: f32) -> memref<16xf32> {
      %memspacecast = memref.memory_space_cast %arg0 : memref<4x4xf32, 1> to memref<4x4xf32>
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %expanded = memref.expand_shape %memspacecast [[0], [1, 2]] output_shape [4, 2, 2] : memref<4x4xf32> into memref<4x2x2xf32>
      %collapsed = memref.collapse_shape %expanded [[0, 1, 2]] : memref<4x2x2xf32> into memref<16xf32>
      %loaded = memref.load %collapsed[%c0] : memref<16xf32>
      %added = arith.addf %loaded, %arg2 : f32
      memref.store %added, %collapsed[%c0] : memref<16xf32>
      %atomic_result = memref.atomic_rmw addf %arg2, %collapsed[%c4] : (f32, memref<16xf32>) -> f32
      return %collapsed : memref<16xf32>
    }
    // mlir-opt --bubble-down-memory-space-casts
    func.func @op_with_cast_sequence(%arg0: memref<4x4xf32, 1>, %arg1: index, %arg2: f32) -> memref<16xf32> {
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %expand_shape = memref.expand_shape %arg0 [[0], [1, 2]] output_shape [4, 2, 2] : memref<4x4xf32, 1> into memref<4x2x2xf32, 1>
      %collapse_shape = memref.collapse_shape %expand_shape [[0, 1, 2]] : memref<4x2x2xf32, 1> into memref<16xf32, 1>
      %memspacecast = memref.memory_space_cast %collapse_shape : memref<16xf32, 1> to memref<16xf32>
      %0 = memref.load %collapse_shape[%c0] : memref<16xf32, 1>
      %1 = arith.addf %0, %arg2 : f32
      memref.store %1, %collapse_shape[%c0] : memref<16xf32, 1>
      %2 = memref.atomic_rmw addf %arg2, %collapse_shape[%c4] : (f32, memref<16xf32, 1>) -> f32
      return %memspacecast : memref<16xf32>
    }
    ```
  }];
}

#endif // MLIR_TRANSFORMS_PASSES
