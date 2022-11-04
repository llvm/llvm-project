//===- TransformOps.td - Transform dialect operations ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_TRANSFORM_IR_TRANSFORMOPS
#define MLIR_DIALECT_TRANSFORM_IR_TRANSFORMOPS

include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Dialect/Transform/IR/TransformAttrs.td"
include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/IR/TransformEffects.td"
include "mlir/Dialect/Transform/IR/TransformInterfaces.td"

def AlternativesOp : TransformDialectOp<"alternatives",
    [DeclareOpInterfaceMethods<RegionBranchOpInterface,
        ["getSuccessorEntryOperands", "getSuccessorRegions",
         "getRegionInvocationBounds"]>,
     DeclareOpInterfaceMethods<TransformOpInterface>,
     FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
     IsolatedFromAbove, PossibleTopLevelTransformOpTrait,
     SingleBlockImplicitTerminator<"::mlir::transform::YieldOp">]> {
  let summary = "Attempts sequences of transforms until one succeeds";
  let description = [{
    This op may have an arbitrary number of regions, each of which represents a
    sequence of transform operations to be applied to the same payload IR. The
    regions are visited in order of appearance, and transforms in them are
    applied in their respective order of appearance. If one of these transforms
    fails to apply, the remaining ops in the same region are skipped an the next
    region is attempted. If all transformations in a region succeed, the
    remaining regions are skipped and the entire "alternatives" transformation
    succeeds. If all regions contained a failing transformation, the entire
    "alternatives" transformation fails.

    It is up to the nested operations to define which errors are "recoverable"
    (or "silenceable") and allow another alternatives to be attempted, and which
    errors should be propagated without attempting the other alternatives.

    The single operand of this operation is the scope in which the alternative
    transformation sequences are attempted, that is, an operation in the payload
    IR that contains all the other operations that may be modified by the
    transformations. The scope operation must be isolated from above. There is
    no check that the transforms are indeed scoped as their "apply" methods can
    be arbitrarily complex. Therefore it is the responsibility of the user to
    ensure that the transforms are scoped correctly, or to produce an
    irrecoverable error and thus abort the execution without attempting the
    remaining alternatives. Note that the payload IR outside of the given scope
    is not necessarily in the valid state, or even accessible to the
    transformation.
    
    The changes to the IR within the scope performed by transforms in the failed
    alternative region are reverted before attempting the next region.
    Practically, this is achieved by cloning the scope. Therefore it is advised
    to limit the scope as much as possible and place the most likely
    alternatives early in the region list. The operation is also isolated from
    above and requires rediscovering the operations within the given scope to
    avoid additional handle invalidation. The latter restriction may be lifted
    in the future.

    Each of the regions may yield transform IR handles. The handles of the first
    successful alternative region are returned as the results of the
    "alternatives" op. Therefore, each alternative region must yield the same
    number of results, which should also match the number and the types of the
    "alternatives" op results.

    Remark: this op allows one to implement a simple "try" construct as follows:

    ```mlir
    %result = transform.alternatives %scope {
    ^bb0(%arg0: !pdl.operation):
      // Try a fallible transformation.
      %0 = transform.fallible %arg0 // ...
      // If succeeded, yield the the result of the transformation.
      transform.yield %0 : !pdl.operation
    }, {
    ^bb0(%arg0: !pdl.operation):
      // Otherwise, the second alternative is tried and it always succeeds by
      // returning the original handle.
      transform.yield %arg0 : !pdl.operation
    }
    ```
  }];

  let arguments = (ins Optional<TransformTypeInterface>:$scope);
  let results = (outs Variadic<TransformTypeInterface>:$results);
  let regions = (region VariadicRegion<SizedRegion<1>>:$alternatives);

  let assemblyFormat =
    "($scope^ `:` type($scope))? (`->` type($results)^)? "
    "attr-dict-with-keyword regions";
  let hasVerifier = 1;
}

def CastOp : TransformDialectOp<"cast",
    [TransformOpInterface, TransformEachOpTrait,
     DeclareOpInterfaceMethods<CastOpInterface>,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let arguments = (ins TransformTypeInterface:$input);
  let results = (outs TransformTypeInterface:$output);
  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($output)";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
      ::mlir::Operation *target,
      ::llvm::SmallVectorImpl<::mlir::Operation *> &results,
      ::mlir::transform::TransformState &state);
  }];
}

def ForeachOp : TransformDialectOp<"foreach",
    [DeclareOpInterfaceMethods<TransformOpInterface>,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     DeclareOpInterfaceMethods<RegionBranchOpInterface, [
         "getSuccessorRegions", "getSuccessorEntryOperands"]>,
     SingleBlockImplicitTerminator<"::mlir::transform::YieldOp">
    ]> {
  let summary = "Executes the body for each payload op";
  let description = [{
    This op has exactly one region with exactly one block ("body"). The body is
    executed for each payload op that is associated to the target operand in an
    unbatched fashion. I.e., the block argument ("iteration variable") is always
    mapped to exactly one payload op.

    This op always reads the target handle. Furthermore, it consumes the handle
    if there is a transform op in the body that consumes the iteration variable.
    This op does not return anything.

    The transformations inside the body are applied in order of their
    appearance. During application, if any transformation in the sequence fails,
    the entire sequence fails immediately leaving the payload IR in potentially
    invalid state, i.e., this operation offers no transformation rollback
    capabilities.

    This op generates as many handles as the terminating YieldOp has operands.
    For each result, the payload ops of the corresponding YieldOp operand are
    merged and mapped to the same resulting handle.
  }];

  let arguments = (ins TransformTypeInterface:$target);
  let results = (outs Variadic<TransformTypeInterface>:$results);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = 
    "$target `:` type($target) (`->` type($results)^)? $body attr-dict";
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    /// Allow the dialect prefix to be omitted.
    static StringRef getDefaultDialect() { return "transform"; }

    BlockArgument getIterationVariable() {
      return getBody().front().getArgument(0);
    }

    transform::YieldOp getYieldOp();
  }];
}

def GetClosestIsolatedParentOp : TransformDialectOp<"get_closest_isolated_parent",
    [DeclareOpInterfaceMethods<TransformOpInterface>,
     NavigationTransformOpTrait, MemoryEffectsOpInterface]> {
  let summary = "Gets handles to the closest isolated-from-above parents";
  let description = [{
    The handles defined by this Transform op correspond to the closest isolated
    from above ancestor of the Payload IR operations associated with its
    operand. If any of the given Payload IR ops has no such parent (unlikely as
    there usually is a top-level ModuleOp), the transformation is considered to
    have failed.

    Ancestor ops follow the same order as the ops associated with the
    operand, except for potential duplicates (multiple Payload IR ops associated
    with the operand have the same parent) for which the ancestor will only be
    listed once for the first time it occurs. For example, given the list
    "(childof(A), childof(B), childof(B), childof(A), childof(B))", the
    resulting list will be just "(A, B)". Note that no other semantic ordering
    is applied, e.g., "B" may itself be a parent of "A". This may have an impact
    on the further transformation applied to the handle produced here.
  }];

  let arguments = (ins TransformTypeInterface:$target);
  let results = (outs TransformTypeInterface:$parent);
  let assemblyFormat =
    "$target attr-dict `:` functional-type(operands, results)";
}

def GetProducerOfOperand : TransformDialectOp<"get_producer_of_operand",
    [DeclareOpInterfaceMethods<TransformOpInterface>,
     NavigationTransformOpTrait, MemoryEffectsOpInterface]> {
  let summary = "Get handle to the producer of this operation's operand number";
  let description = [{
    The handle defined by this Transform op corresponds to operation that
    produces the SSA value defined by the `target` and `operand_number`
    arguments. If the origin of the SSA value is not an operations (i.e. it is
    a block argument), the transform silently fails.
    The return handle points to only the subset of successfully produced
    computational operations, which can be empty.
  }];

  let arguments = (ins TransformTypeInterface:$target,
                       I64Attr:$operand_number);
  let results = (outs TransformTypeInterface:$parent);
  let assemblyFormat = "$target `[` $operand_number `]` attr-dict `:` "
                       "functional-type(operands, results)";
}

def MergeHandlesOp : TransformDialectOp<"merge_handles",
    [DeclareOpInterfaceMethods<TransformOpInterface>,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     SameOperandsAndResultType]> {
  let summary = "Merges handles into one pointing to the union of payload ops";
  let description = [{
    Creates a new Transform IR handle value that points to the same Payload IR
    operations as the operand handles. The Payload IR operations are listed
    in the same order as they are in the operand handles, grouped by operand
    handle, e.g., all Payload IR operations associated with the first handle
    come first, then all Payload IR operations associated with the second handle
    and so on. If `deduplicate` is set, do not add the given Payload IR
    operation more than once to the final list regardless of it coming from the
    same or different handles. Consumes the operands and produces a new handle.
  }];

  let arguments = (ins Variadic<TransformTypeInterface>:$handles,
                       UnitAttr:$deduplicate);
  let results = (outs TransformTypeInterface:$result);
  let assemblyFormat = "($deduplicate^)? $handles attr-dict `:` type($result)";
  let hasFolder = 1;
}

def SplitHandlesOp : TransformDialectOp<"split_handles",
    [FunctionalStyleTransformOpTrait,
     DeclareOpInterfaceMethods<TransformOpInterface>,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Splits handles from a union of payload ops to a list";
  let description = [{
    Creates `num_result_handles` transform IR handles extracted from the 
    `handle` operand. The resulting Payload IR operation handles are listed
    in the same order as the operations appear in the source `handle`.
    This is useful for ensuring a statically known number of operations are
    tracked by the source `handle` and to extract them into individual handles
    that can be further manipulated in isolation.

    This operation succeeds and returns `num_result_handles` if the statically
    specified `num_result_handles` corresponds to the dynamic number of 
    operations contained in the source `handle`. Otherwise it silently fails.
  }];

  let arguments = (ins TransformTypeInterface:$handle,
                       I64Attr:$num_result_handles);
  let results = (outs Variadic<TransformTypeInterface>:$results);

  let builders = [
    OpBuilder<(ins "Value":$handle, "int64_t":$numResultHandles)>
  ];

  let assemblyFormat = [{
    $handle `in` `[` $num_result_handles `]` 
    attr-dict `:` functional-type(operands, results)
  }];
}

def PDLMatchOp : TransformDialectOp<"pdl_match",
    [DeclareOpInterfaceMethods<TransformOpInterface>,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Finds ops that match the named PDL pattern";
  let description = [{
    Find Payload IR ops nested within the Payload IR op associated with the
    operand that match the PDL pattern identified by its name. The pattern is
    expected to be defined in the closest surrounding `WithPDLPatternsOp`.

    Produces a Transform IR value associated with the list of Payload IR ops
    that matched the pattern. The order of results in the list is that of the
    Operation::walk, clients are advised not to rely on a specific order though.
    If the operand is associated with multiple Payload IR ops, finds matching
    ops nested within each of those and produces a single list containing all
    of the matched ops.

    The transformation is considered successful regardless of whether some
    Payload IR ops actually matched the pattern and only fails if the pattern
    could not be looked up or compiled.
  }];

  let arguments = (ins
    Arg<TransformTypeInterface, "Payload IR scope to match within">:$root, 
    SymbolRefAttr:$pattern_name);
  let results = (outs
    Res<TransformTypeInterface, "Handle to the matched Payload IR ops">:$matched);

  let assemblyFormat = "$pattern_name `in` $root attr-dict `:` "
                       "functional-type(operands, results)";
}

def PrintOp : TransformDialectOp<"print",
    [DeclareOpInterfaceMethods<TransformOpInterface>,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Dump each payload op";
  let description = [{
    This op dumps each payload op that is associated with the `target` operand
    to stderr. It also prints the `name` string attribute. If no target is
    specified, the top-level op is dumped.

    This op is useful for printf-style debugging.
  }];

  let arguments = (ins Optional<TransformTypeInterface>:$target,
                       OptionalAttr<StrAttr>:$name);
  let results = (outs);

  let builders = [
    OpBuilder<(ins CArg<"StringRef", "StringRef()">:$name)>,
    OpBuilder<(ins "Value":$target, CArg<"StringRef", "StringRef()">:$name)>
  ];

  let assemblyFormat = "$target attr-dict (`:` type($target)^)?";
}


def ReplicateOp : TransformDialectOp<"replicate",
    [DeclareOpInterfaceMethods<TransformOpInterface>,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     AllTypesMatch<["handles", "replicated"]>]> {
  let summary = "Lists payload ops multiple times in the new handle";
  let description = [{
    Produces a new handle associated with a list of payload IR ops that is
    computed by repeating the list of payload IR ops associated with the
    operand handle as many times as the "pattern" handle has associated
    operations. For example, if pattern is associated with [op1, op2] and the
    operand handle is associated with [op3, op4, op5], the resulting handle
    will be associated with [op3, op4, op5, op3, op4, op5].

    This transformation is useful to "align" the sizes of payload IR lists
    before a transformation that expects, e.g., identically-sized lists. For
    example, a transformation may be parameterized by same notional per-target 
    size computed at runtime and supplied as another handle, the replication
    allows this size to be computed only once and used for every target instead
    of replicating the computation itself.

    Note that it is undesirable to pass a handle with duplicate operations to
    an operation that consumes the handle. Handle consumption often indicates
    that the associated payload IR ops are destroyed, so having the same op
    listed more than once will lead to double-free. Single-operand
    MergeHandlesOp may be used to deduplicate the associated list of payload IR
    ops when necessary. Furthermore, a combination of ReplicateOp and
    MergeHandlesOp can be used to construct arbitrary lists with repetitions.
  }];

  let arguments = (ins TransformTypeInterface:$pattern,
                       Variadic<TransformTypeInterface>:$handles);
  let results = (outs Variadic<TransformTypeInterface>:$replicated);
  let assemblyFormat = "`num` `(` $pattern `)` $handles attr-dict `:` "
                       "type($pattern) `,` type($handles)";
}

def SequenceOp : TransformDialectOp<"sequence",
    [DeclareOpInterfaceMethods<RegionBranchOpInterface,
        ["getSuccessorEntryOperands", "getSuccessorRegions",
         "getRegionInvocationBounds"]>,
     DeclareOpInterfaceMethods<TransformOpInterface>,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     OpAsmOpInterface, PossibleTopLevelTransformOpTrait,
     SingleBlockImplicitTerminator<"::mlir::transform::YieldOp">]> {
  let summary = "Contains a sequence of other transform ops to apply";
  let description = [{
    The transformations indicated by the sequence are applied in order of their
    appearance. Each value produced by a transformation within the sequence
    corresponds to an operation or a group of operations in the payload IR.
    The behavior of the operation when a nested transformation produces a
    silenceable error is controlled by the `failure_propagation_mode` attribute.
    When set to `propagate`, the failure of any nested transformation in the
    sequence implies immediate failure of the entire sequence with a silenceable
    error, and no further transformation is attempted. When set to `suppress`,
    silenceable errors in nested operations are ignored and further
    transformations are applied. Beware that even silenceable errors may leave
    the payload IR in a state unsuitable for further transformations. It is
    the responsibility of the caller to ensure the following transformations
    are robust enough when errors are suppressed. Definite errors reported by
    nested transformations abort the sequence regardless of the propagation
    mode. The set of modes may be extended in the future, e.g., to collect
    silenceable errors and report them after attempting all transformations in
    the sequence.

    The entry block of this operation has a single argument that maps to either
    the operand if provided or the top-level container operation of the payload
    IR, typically the root operation of the pass interpreting the transform
    dialect. Operand omission is only allowed for sequences not contained in
    another sequence.
  }];

  let arguments = (ins FailurePropagationMode:$failure_propagation_mode,
                       Optional<TransformTypeInterface>:$root);
  let results = (outs Variadic<TransformTypeInterface>:$results);
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat =
    "($root^ `:` type($root))? (`->` type($results)^)? `failures` `(` "
    "$failure_propagation_mode `)` attr-dict-with-keyword regions";

  let extraClassDeclaration = [{
    /// Allow the dialect prefix to be omitted.
    static StringRef getDefaultDialect() { return "transform"; }
  }];

  let hasVerifier = 1;
}

def WithPDLPatternsOp : TransformDialectOp<"with_pdl_patterns",
    [DeclareOpInterfaceMethods<TransformOpInterface>, NoTerminator,
     OpAsmOpInterface, PossibleTopLevelTransformOpTrait, RecursiveMemoryEffects,
     SymbolTable]> {
  let summary = "Contains PDL patterns available for use in transforms";
  let description = [{
    This op contains a set of named PDL patterns that are available for the
    Transform dialect operations to be used for pattern matching. For example,
    PDLMatchOp can be used to produce a Transform IR value associated with all
    Payload IR operations that match the pattern as follows:

    ```mlir
    transform.with_pdl_patterns {
    ^bb0(%arg0: !pdl.operation):
      pdl.pattern @my_pattern : benefit(1) {
        %0 = pdl.operation //...
        // Regular PDL goes here.
        pdl.rewrite %0 with "transform.dialect"
      }

      sequence %arg0 failures(propagate) {
      ^bb0(%arg1: !pdl.operation):
        %1 = pdl_match @my_pattern in %arg1
        // Use %1 as handle
      }
    }
    ```

    Note that the pattern is expected to finish with a `pdl.rewrite` terminator
    that points to the custom rewriter named "transform.dialect". The rewriter
    actually does nothing, but the transform application will keep track of the
    operations that matched the pattern.

    This op is expected to contain `pdl.pattern` operations and exactly one
    another Transform dialect operation that gets executed with all patterns
    available. This op is a possible top-level Transform IR op, the argument of
    its entry block corresponds to either the root op of the payload IR or the
    ops associated with its operand when provided.
  }];

  let arguments = (ins
    Arg<Optional<TransformTypeInterface>, "Root operation of the Payload IR",
        [TransformMappingRead]>:$root);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "($root^ `:` type($root))? attr-dict-with-keyword regions";

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    /// Allow the dialect prefix to be omitted.
    static StringRef getDefaultDialect() { return "transform"; }
  }];
}

def YieldOp : TransformDialectOp<"yield", [Terminator]> {
  let summary = "Yields operation handles from a transform IR region";
  let description = [{
    This terminator operation yields operation handles from regions of the
    transform IR ops back to the containing op. It is not itself associated with
    any transformation on the payload IR and is used for flow purposes only.
  }];

  let arguments = (ins 
    Arg<Variadic<TransformTypeInterface>, "Operation handles yielded back to the parent",
        [TransformMappingRead]>:$operands);
  let assemblyFormat = "operands attr-dict (`:` type($operands)^)?";

  let builders = [
    OpBuilder<(ins), [{
      return build($_builder, $_state, ::mlir::ValueRange());
    }]>
  ];
}

#endif // MLIR_DIALECT_TRANSFORM_IR_TRANSFORMOPS
