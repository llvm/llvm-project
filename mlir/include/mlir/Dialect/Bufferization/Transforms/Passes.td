//===-- Passes.td - Bufferization passes definition file ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_BUFFERIZATION_TRANSFORMS_PASSES
#define MLIR_DIALECT_BUFFERIZATION_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def OwnershipBasedBufferDeallocation : Pass<
    "ownership-based-buffer-deallocation"> {
  let summary = "Adds all required dealloc operations for all allocations in "
                "the input program";
  let description = [{
    This pass implements an algorithm to automatically introduce all required
    deallocation operations for all buffers in the input program. This ensures
    that the resulting program does not have any memory leaks.

    The Buffer Deallocation pass operates on the level of operations
    implementing the FunctionOpInterface. Such operations can take MemRefs as
    arguments, but also return them. To ensure compatibility among all functions
    (including external ones), some rules have to be enforced. They are just
    assumed to hold for all external functions. Functions for which the
    definition is available ideally also already adhere to the ABI.
    Otherwise, all MemRef write operations in the input IR must dominate all
    MemRef read operations in the input IR. Then, the pass may modify the input
    IR by inserting `bufferization.clone` operations such that the output IR
    adheres to the function boundary ABI:
    * When a MemRef is passed as a function argument, ownership is never
      acquired. It is always the caller's responsibility to deallocate such
      MemRefs.
    * Returning a MemRef from a function always passes ownership to the caller,
      i.e., it is also the caller's responsibility to deallocate MemRefs
      returned from a called function.
    * A function must not return a MemRef with the same allocated base buffer as
      one of its arguments (in this case a copy has to be created). Note that in
      this context two subviews of the same buffer that don't overlap are also
      considered an alias.

    It is recommended to bufferize all operations first such that no tensor
    values remain in the IR once this pass is applied. That way all allocated
    MemRefs will be properly deallocated without any additional manual work.
    Otherwise, the pass that bufferizes the remaining tensors is responsible to
    add the corresponding deallocation operations. Note that this pass does not
    consider any values of tensor type and assumes that MemRef values defined by
    `bufferization.to_memref` do not return ownership and do not have to be
    deallocated. `bufferization.to_tensor` operations are handled similarly to
    `bufferization.clone` operations with the exception that the result value is
    not handled because it's a tensor (not a MemRef).

    Input

    ```mlir
    #map0 = affine_map<(d0) -> (d0)>
    module {
      func.func @condBranch(%arg0: i1,
                            %arg1: memref<2xf32>,
                            %arg2: memref<2xf32>) {
        cf.cond_br %arg0, ^bb1, ^bb2
      ^bb1:
        cf.br ^bb3(%arg1 : memref<2xf32>)
      ^bb2:
        %0 = memref.alloc() : memref<2xf32>
        linalg.generic {
          indexing_maps = [#map0, #map0],
          iterator_types = ["parallel"]}
        outs(%arg1, %0 : memref<2xf32>, memref<2xf32>) {
        ^bb0(%gen1_arg0: f32, %gen1_arg1: f32):
          %tmp1 = exp %gen1_arg0 : f32
          linalg.yield %tmp1 : f32
        }
        cf.br ^bb3(%0 : memref<2xf32>)
      ^bb3(%1: memref<2xf32>):
        "memref.copy"(%1, %arg2) : (memref<2xf32>, memref<2xf32>) -> ()
        return
      }
    }
    ```

    Output

    ```mlir
    #map = affine_map<(d0) -> (d0)>
    module {
      func.func @condBranch(%arg0: i1,
                            %arg1: memref<2xf32>,
                            %arg2: memref<2xf32>) {
        %false = arith.constant false
        %true = arith.constant true
        cf.cond_br %arg0, ^bb1, ^bb2
      ^bb1:  // pred: ^bb0
        cf.br ^bb3(%arg1, %false : memref<2xf32>, i1)
      ^bb2:  // pred: ^bb0
        %alloc = memref.alloc() : memref<2xf32>
        linalg.generic {
          indexing_maps = [#map, #map],
          iterator_types = ["parallel"]}
        outs(%arg1, %alloc : memref<2xf32>, memref<2xf32>)
        ^bb0(%out: f32, %out_0: f32):
          %2 = math.exp %out : f32
          linalg.yield %2, %out_0 : f32, f32
        }
        cf.br ^bb3(%alloc, %true : memref<2xf32>, i1)
      ^bb3(%0: memref<2xf32>, %1: i1):  // 2 preds: ^bb1, ^bb2
        memref.copy %0, %arg2 : memref<2xf32> to memref<2xf32>
        %base_buffer, %offset, %sizes, %strides =
          memref.extract_strided_metadata %0 :
          memref<2xf32> -> memref<f32>, index, index, index
        bufferization.dealloc (%base_buffer : memref<f32>) if (%1)
        return
      }
    }
    ```

    The `private-function-dynamic-ownership` pass option allows the pass to add
    additional arguments to private functions to dynamically give ownership of
    MemRefs to callees. This can enable earlier deallocations and allows the
    pass to by-pass the function boundary ABI and thus potentially leading to
    fewer MemRef clones being inserted. For example, the private function
    ```mlir
    func.func private @passthrough(%memref: memref<2xi32>) -> memref<2xi32> {
      return %memref : memref<2xi32>
    }
    ```
    would be converted to
    ```mlir
    func.func private @passthrough(%memref: memref<2xi32>,
                                   %ownership: i1) -> (memref<2xi32>, i1) {
      return %memref, %ownership : memref<2xi32>, i1
    }
    ```
    and thus allows the returned MemRef to alias with the MemRef passed as
    argument (which would otherwise be forbidden according to the function
    boundary ABI).
  }];
  let options = [
    Option<"privateFuncDynamicOwnership", "private-function-dynamic-ownership",
           "bool", /*default=*/"false",
           "Allows to add additional arguments to private functions to "
           "dynamically pass ownership of memrefs to callees. This can enable "
           "earlier deallocations.">,
  ];
  let constructor = "mlir::bufferization::createOwnershipBasedBufferDeallocationPass()";

  let dependentDialects = [
    "mlir::bufferization::BufferizationDialect", "mlir::arith::ArithDialect",
    "mlir::memref::MemRefDialect", "mlir::scf::SCFDialect"
  ];
}

def BufferDeallocationSimplification :
    Pass<"buffer-deallocation-simplification"> {
  let summary = "Optimizes `bufferization.dealloc` operation for more "
                "efficient codegen";
  let description = [{
    This pass uses static alias analysis to reduce the number of alias checks
    required at runtime. Such checks are sometimes necessary to make sure that
    memrefs aren't deallocated before their last usage (use after free) or that
    some memref isn't deallocated twice (double free).
  }];

  let constructor =
    "mlir::bufferization::createBufferDeallocationSimplificationPass()";

  let dependentDialects = [
    "mlir::bufferization::BufferizationDialect", "mlir::arith::ArithDialect",
    "mlir::memref::MemRefDialect"
  ];
}

def OptimizeAllocationLiveness
    : Pass<"optimize-allocation-liveness", "func::FuncOp"> {
  let summary = "This pass optimizes the liveness of temp allocations in the "
                "input function";
  let description =
       [{This pass will find all operations that have a memory allocation effect.
       It will search for the corresponding deallocation and move it right after
       the last user of the allocation.
       This will optimize the liveness of the allocations.

       The pass is expected to run after the deallocation pipeline.}];
  let constructor =
      "mlir::bufferization::createOptimizeAllocationLivenessPass()";
  let dependentDialects = ["mlir::memref::MemRefDialect"];
}

def LowerDeallocations : Pass<"bufferization-lower-deallocations"> {
  let summary = "Lowers `bufferization.dealloc` operations to `memref.dealloc`"
                "operations";
  let description = [{
    This pass lowers `bufferization.dealloc` operations to the `memref` dialect.
    It can be applied to a `builtin.module` or operations implementing the
    `FunctionOpInterface`. For the latter, only simple `dealloc` operations can
    be lowered because the library function necessary for the fully generic
    lowering cannot be inserted. In this case, an error will be emitted.
    Next to `memref.dealloc` operations, it may also emit operations from the
    `arith`, `scf`, and `func` dialects to build conditional deallocations and
    library functions to avoid code-size blow-up.
  }];

  let constructor =
    "mlir::bufferization::createLowerDeallocationsPass()";

  let dependentDialects = [
    "arith::ArithDialect", "memref::MemRefDialect", "scf::SCFDialect",
    "func::FuncDialect"
  ];
}

def BufferHoisting : Pass<"buffer-hoisting", "func::FuncOp"> {
  let summary = "Optimizes placement of allocation operations by moving them "
                "into common dominators and out of nested regions";
  let description = [{
    This pass implements an approach to aggressively move allocations upwards
    into common dominators and out of nested regions.
  }];
  let constructor = "mlir::bufferization::createBufferHoistingPass()";
}

def BufferLoopHoisting : Pass<"buffer-loop-hoisting", "func::FuncOp"> {
  let summary = "Optimizes placement of allocation operations by moving them "
                "out of loop nests";
  let description = [{
    This pass implements an approach to aggressively move allocations upwards
    out of loop nests. It does not move allocations into common dominators.
  }];
  let constructor = "mlir::bufferization::createBufferLoopHoistingPass()";
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

    If the hoist-static-allocs option is on, the pass tries to eliminate the
    allocation for the returned memref and avoid the memory-copy if possible.
    This optimization applies on the returned memref which has static shape and
    is allocated by memref.alloc in the function. It will use the memref given
    in function argument to replace the allocated memref.
  }];
  let options = [
    Option<"addResultAttribute", "add-result-attr", "bool",
       /*default=*/"false",
       "Add the attribute 'bufferize.result' to all output parameters.">,
    Option<"hoistStaticAllocs", "hoist-static-allocs",
       "bool", /*default=*/"false",
       "Hoist static allocations to call sites.">,
  ];
  let constructor = "mlir::bufferization::createBufferResultsToOutParamsPass()";
  let dependentDialects = ["memref::MemRefDialect"];
}

def DropEquivalentBufferResults : Pass<"drop-equivalent-buffer-results", "ModuleOp">  {
  let summary = "Remove MemRef return values that are equivalent to a bbArg";
  let description = [{
    This pass removes MemRef return values from functions if they are equivalent
    to a function bbArg. In that case, the return value is redundant and the
    respective CallOp operand can be used at the call site.

    Note: If a bbArg buffer is not returned directly but casted to beforehand,
    the buffer is still considered equivalent.
  }];
  let constructor = "mlir::bufferization::createDropEquivalentBufferResultsPass()";
  let dependentDialects = ["memref::MemRefDialect"];
}

def EmptyTensorToAllocTensor : Pass<"empty-tensor-to-alloc-tensor"> {
  let summary = "Replace all empty ops by alloc_tensor ops.";
  let description = [{
    tensor.empty ops return a tensor of unspecified contents who's only purpose
    is to carry the tensor shape. This pass converts such ops to
    bufferization.alloc_tensor ops, which bufferize to buffer allocations.
  }];
  let constructor = "mlir::bufferization::createEmptyTensorToAllocTensorPass()";
  let dependentDialects = ["tensor::TensorDialect"];
}

def OneShotBufferize : Pass<"one-shot-bufferize", "ModuleOp"> {
  let summary = "One-Shot Bufferize";
  let description = [{
    This pass bufferizes all ops that implement `BufferizableOpInterface`. It
    first performs an inplacability analysis on SSA use-def chains of tensor
    values to determine which OpOperands may bufferize in-place, i.e., without
    inserting a buffer copy. It then rewrites the IR, inserting a buffer
    allocation and copy for each OpOperand that was decided to bufferize
    out-of-place.

    One-Shot Bufferize (and `BufferizableOpInterface`) was designed for ops that
    are in destination-passing style. When bufferizing such ops, it is possible
    to reuse the buffer of a tensor OpOperand for a tensor OpResult. In essence,
    a possible destination of an operation is already passed as an SSA value.

    `tensor.insert` is an example for an op in destination-passing style. E.g.,
    when bufferizing `%t0 = tensor.insert %f into %dest[%idx]`, `buffer(%t0)` is
    identical to `buffer(%dest)` in the absence of RaW conflicts. As a counter
    example, `tensor.generate` is not in destination-passing style and always
    results in a new buffer allocation.

    One-Shot Bufferize does not deallocate any buffers that it allocates. The
    `-buffer-deallocation-pipeline` pipeline should be run after One-Shot
    Bufferize to insert the deallocation operations necessary to eliminate
    memory leaks.

    One-Shot Bufferize will by default reject IR that contains non-bufferizable
    op, i.e., ops that do not implemement BufferizableOpInterface. Such IR can
    be allowed with `allow-unknown-ops=1`. In that case, to_memref and to_tensor
    ops will be generated at the bufferization boundary. This is useful for
    compatibility with existing partial bufferization passes: These can
    bufferize the remaining IR after running One-Shot Bufferize.

    Note: Running One-Shot Bufferize after a partial bufferization pass is
    currently not supported. Running partial bufferization passes after running
    One-Shot Bufferize is supported and the recommended way to gradually
    migrate from partial bufferization to One-Shot Bufferize.

    With `dialect-filter`, bufferization can be restricted to a set of dialects.
    If no filter is specified, all ops that implement `BufferizableOpInterface`
    are bufferized. Ops from the `std` dialect are an exception: These ops are
    always ignored, even if no filter is specified. When specifying a dialect
    filter and `allow-unknown-ops` is not turned on, bufferization would fail
    when encountering an op that is not included in the filter (even if it is
    bufferizable).

    One-Shot Bufferize will by default assume memref types with fully dynamic
    layout maps when a precise layout cannot be inferred. E.g., this is the case
    when wrapping a non-bufferizable op in to_memref/to_tensor ops. This
    behavior can be overridden with `unknown-type-conversion`. Valid values are
    `fully-dynamic-layout-map` and `identity-layout-map`.

    For testing/debugging purposes, `test-analysis-only=1 print-conflicts=1`
    prints analysis results and explains why an OpOperand was decided to
    bufferize out-of-place. This is useful for understanding why One-Shot
    Bufferize chose to insert a certain buffer copy.

    `bufferize-function-boundaries` is an experimental flag for bufferizing
    `FuncOp`, `ReturnOp` and `CallOp`. This feature is still under development
    and supports only simple cases at the moment. In particular:

    * Recursive or circular function call graphs are not supported.
    * External functions (without bodies) that return a tensor are not
      supported.
    * Function with multiple blocks or multiple ReturnOps are not supported.
    * Layout maps on function signatures can be controlled with a separate
      `function-boundary-type-conversion` option, which is similar to
      `unknown-type-conversion` but supports an additional `infer-layout-map`
      option. `fully-dynamic-layout-map` and `identity-layout-map` ensure that
      function signatures bufferize to easily predictable types, potentially at
      the cost of additional casts and copies, respectively. When layout maps
      are inferred, function return types may be more precise, but less
      predictable. Function argument types cannot be inferred and always have
      fully dynamic layout maps with `infer-layout-map`.

    One-Shot Bufferize implements the following contract around function calls:
    The buffer of function arguments is always writable (unless annotated with
    `bufferization.writable = false`). A buffer copy may be inserted at the call
    site where necessary. Alias sets and equivalence info is propagated through
    function calls. Whenever a function is bufferized, all other functions that
    are being called were already analyzed and bufferized, so exact alias and
    equivalence information is available. This is why recursive function calls
    are not yet supported.

    One-Shot Bufferize gathers additional information during the analysis phase
    when function boundary bufferization is activated. E.g., whether a function
    argument is read/written and which returned values are aliasing/equivalent.
    For debugging purposes, such information can be printed with
    `test-analysis-only`.

    The order in which ops are analyzed is important. The analysis is greedy and
    ops that are analyzed earlier are more likely to bufferize in-place. The
    heuristic can be set with `analysis-heuristic`. At the moment, the following
    heuristics are available:

    * `bottom-up` (default): Analyze ops from bottom to top.
    * `top-down`: Analyze ops from top to bottom.
    * `fuzzer`: Randomize the ordering of ops with `analysis-fuzzer-seed`.
    * `bottom-up-from-terminators`: Traverse the reverse use-def chains of
      tensor IR, starting from region branch terminators (bottom-up). Nested
      regions are traversed before enclosing regions. Analyze the traversed ops
      first, then analyze the remaining ops bottom-up. This heuristic is useful
      for bufferizing loop constructs. One-Shot Bufferize currently supports
      only such IR where yielded tensor values bufferize to equivalent region
      iter_args, and first analyzing all ops on the path from the "yielding" op
      to the beginning of the loop body makes it more likely for the region
      iter_args and yielded values to bufferize to equivalent buffers.
  }];
  let options = [
    Option<"allowReturnAllocsFromLoops", "allow-return-allocs-from-loops",
           "bool", /*default=*/"false",
           "Allows returning/yielding new allocations from a loop.">,
    Option<"allowUnknownOps", "allow-unknown-ops", "bool",
           /*default=*/"false",
           "Allows unknown (not bufferizable) ops in the input IR.">,
    Option<"analysisFuzzerSeed", "analysis-fuzzer-seed", "unsigned",
           /*default=*/"0",
           "Test only: Analyze ops in random order with a given seed (fuzzer)">,
    Option<"analysisHeuristic", "analysis-heuristic", "std::string",
           /*default=*/"\"bottom-up\"",
           "Heuristic that control the IR traversal during analysis">,
    Option<"bufferizeFunctionBoundaries", "bufferize-function-boundaries",
           "bool", /*default=*/"0",
           "Bufferize function boundaries (experimental).">,
    Option<"checkParallelRegions", "check-parallel-regions", "bool",
           /*default=*/"true", "Account for parallel regions in RaW analysis.">,
    Option<"copyBeforeWrite", "copy-before-write", "bool", /*default=*/"false",
           "Skip the analysis. Make a buffer copy on every write.">,
    ListOption<"dialectFilter", "dialect-filter", "std::string",
               "Restrict bufferization to ops from these dialects.">,
    Option<"dumpAliasSets", "dump-alias-sets", "bool", /*default=*/"false",
           "Test only: Annotate tensor IR with alias sets">,
    ListOption<"noAnalysisFuncFilter", "no-analysis-func-filter", "std::string",
               "Skip analysis of functions with these symbol names."
               "Set copyBeforeWrite to true when bufferizing them.">,
    Option<"functionBoundaryTypeConversion",
           "function-boundary-type-conversion", "std::string",
           /*default=*/"\"infer-layout-map\"",
           "Controls layout maps when bufferizing function signatures.">,
    Option<"mustInferMemorySpace", "must-infer-memory-space", "bool",
           /*default=*/"false",
           "The memory space of an memref types must always be inferred. If "
           "unset, a default memory space of 0 is used otherwise.">,
    Option<"useEncodingForMemorySpace", "use-encoding-for-memory-space", "bool",
            /*default=*/"false",
            "Use the Tensor encoding attribute for the memory space. Exclusive to"
            " the 'must-infer-memory-space' option">,
    Option<"testAnalysisOnly", "test-analysis-only", "bool",
            /*default=*/"false",
           "Test only: Only run inplaceability analysis and annotate IR">,
    Option<"printConflicts", "print-conflicts", "bool",
            /*default=*/"false",
           "Test only: Annotate IR with RaW conflicts. Requires "
           "test-analysis-only.">,
    Option<"unknownTypeConversion", "unknown-type-conversion", "std::string",
           /*default=*/"\"fully-dynamic-layout-map\"",
           "Controls layout maps for non-inferrable memref types.">,
    Option<"bufferAlignment", "buffer-alignment", "uint64_t", /*default=*/"64",
           "Sets the alignment of newly allocated buffers.">,
  ];
  let constructor = "mlir::bufferization::createOneShotBufferizePass()";

  let statistics = [
    Statistic<"numBufferAlloc", "num-buffer-alloc",
              "Number of buffer allocations">,
    Statistic<"numTensorInPlace", "num-tensor-in-place",
              "Number of in-place tensor OpOperands">,
    Statistic<"numTensorOutOfPlace", "num-tensor-out-of-place",
              "Number of out-of-place tensor OpOperands">,
  ];
}

def PromoteBuffersToStack : Pass<"promote-buffers-to-stack", "func::FuncOp"> {
  let summary = "Promotes heap-based allocations to automatically managed "
                "stack-based allocations";
  let description = [{
    This pass implements a simple algorithm to convert heap-based memory
    allocations to stack-based ones. It uses a built-in heuristic to decide
    whether it makes sense to convert an allocation. Furthermore, dynamic
    shaped buffers that are limited by the rank of the tensor can be
    converted. They are only transformed if they are considered to be small.
  }];
  let constructor = "mlir::bufferization::createPromoteBuffersToStackPass()";
  let options = [
    Option<"maxAllocSizeInBytes", "max-alloc-size-in-bytes", "unsigned",
           /*default=*/"1024",
           "Maximal size in bytes to promote allocations to stack.">,
    Option<"maxRankOfAllocatedMemRef", "max-rank-of-allocated-memref", "unsigned",
           /*default=*/"1",
           "Maximal memref rank to promote dynamic buffers.">,
  ];
}

def EmptyTensorElimination : Pass<"eliminate-empty-tensors"> {
  let summary = "Try to eliminate all tensor.empty ops.";
  let description = [{
    Try to eliminate "tensor.empty" ops inside `op`. This transformation looks
    for subset ops that insert a tensor that originates from a "tensor.empty"
    (as per the reverse use-def chain). Such "tensor.empty" ops are replaced
    with the destination subset.

    E.g.:
    ```
    %0 = tensor.empty() : tensor<10xf32>
    %1 = linalg.fill ... outs(%0 : tensor<10xf32>)
    %2 = tensor.insert_slice %1 into %t ...
    ```

    In the above example, the subset op is "tensor.insert_slice". When tracing
    back the reverse use-def chain of a the source, we end up at a
    "tensor.empty" op. The "tensor.empty" op is replaced with a
    "tensor.extract_slice" op.
  }];
  let constructor = "mlir::bufferization::createEmptyTensorEliminationPass()";
}

#endif // MLIR_DIALECT_BUFFERIZATION_TRANSFORMS_PASSES
