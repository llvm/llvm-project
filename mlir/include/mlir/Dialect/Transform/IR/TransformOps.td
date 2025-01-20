//===- TransformOps.td - Transform dialect operations ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_TRANSFORM_IR_TRANSFORMOPS
#define MLIR_DIALECT_TRANSFORM_IR_TRANSFORMOPS

include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Dialect/Transform/Interfaces/MatchInterfaces.td"
include "mlir/Dialect/Transform/IR/TransformAttrs.td"
include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.td"

def AlternativesOp : TransformDialectOp<"alternatives",
    [DeclareOpInterfaceMethods<RegionBranchOpInterface,
        ["getEntrySuccessorOperands", "getSuccessorRegions",
         "getRegionInvocationBounds"]>,
     DeclareOpInterfaceMethods<TransformOpInterface>,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
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
    ^bb0(%arg0: !transform.any_op):
      // Try a fallible transformation.
      %0 = transform.fallible %arg0 // ...
      // If succeeded, yield the the result of the transformation.
      transform.yield %0 : !transform.any_op
    }, {
    ^bb0(%arg0: !transform.any_op):
      // Otherwise, the second alternative is tried and it always succeeds by
      // returning the original handle.
      transform.yield %arg0 : !transform.any_op
    }
    ```
  }];

  let arguments = (ins Optional<TransformHandleTypeInterface>:$scope);
  let results = (outs Variadic<TransformHandleTypeInterface>:$results);
  let regions = (region VariadicRegion<SizedRegion<1>>:$alternatives);

  let assemblyFormat =
    "($scope^ `:` type($scope))? (`->` type($results)^)? "
    "attr-dict-with-keyword regions";
  let hasVerifier = 1;
}

def AnnotateOp : TransformDialectOp<"annotate",
    [DeclareOpInterfaceMethods<TransformOpInterface>,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Annotates the target operation with an attribute by name";
  let description = [{
    Adds an attribute with the given `name` to the `target` operation. An
    optional `param` handle can be provided to give the attribute a specific
    value, else a UnitAttr is added. A single attribute will be broadcasted to
    all target operations, otherwise the attributes will be mapped 1:1 based on
    the order within the handles.

    Produces a silenceable failure if the length of the parameter payload does
    not match the length of the target payload. Does not consume the provided
    handles.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target,
                       StrAttr:$name,
                       Optional<TransformParamTypeInterface>:$param);
  let results = (outs);

  let assemblyFormat =
    "$target $name attr-dict (`=` $param^)?"
    "`:` type($target) (`,` type($param)^)?";
}

def ApplyCommonSubexpressionEliminationOp : TransformDialectOp<"apply_cse",
    [TransformOpInterface, TransformEachOpTrait,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     ReportTrackingListenerFailuresOpTrait]> {
  let summary = "Eliminate common subexpressions in the body of the target op";
  let description = [{
    This transform applies common subexpression elimination (CSE) to the body
    of the targeted op.

    This transform reads the target handle and modifies the payload. Existing
    handles to operations inside of the targeted op are retained and updated if
    necessary. Note that this can lead to situations where a handle, that was
    previously mapped to multiple distinct (but equivalent) operations, is now
    mapped to the same operation multiple times.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target);
  let results = (outs);
  let assemblyFormat = "`to` $target attr-dict `:` type($target)";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
      ::mlir::transform::TransformRewriter &rewriter,
      ::mlir::Operation *target,
      ::mlir::transform::ApplyToEachResultList &results,
      ::mlir::transform::TransformState &state);
  }];
}

def ApplyConversionPatternsOp : TransformDialectOp<"apply_conversion_patterns",
    [DeclareOpInterfaceMethods<TransformOpInterface>,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     ReportTrackingListenerFailuresOpTrait]
        # GraphRegionNoTerminator.traits> {
  let summary = "Applies conversion patterns to the body of the targeted op";
  let description = [{
    This transform applies the specified conversion patterns to the targeted op
    and all nested ops. By default, this transform applies a "full" dialect
    conversion. If the `partial_conversion` unit attribute is present, this
    transform applies a partial dialect conversion.

    The patterns that should be applied are specified in the first graph region
    of this op. They must implement the
    `ConversionPatternDescriptorOpInterface`. The order in which patterns are
    applied is unspecified; i.e., the ordering of ops in the region of this op
    is irrelevant.

    The second, optional graph region contains exactly one op that specifies
    default type converter that should be used with this dialect conversion. If
    provided, this op must implement the `TypeConverterBuilderOpInterface`.
    Type converters are a property of conversion patterns: each conversion
    pattern stores the type converter that should be used in its C++ class. Each
    conversion pattern descriptor can optionally specify a type converter in its
    `getTypeConverter` interface method. If no type converter is specified in
    this method, the default type converter of the dialect conversion is used.
    Default type converters are useful if the same type converter should be used
    for multiple sets of conversion patterns. (Patterns that should not use this
    default type converter specify their own type converter.)

    The `legal_ops`, `illegal_ops`, `legal_dialects`, `illegal_dialects`
    attributes specify the conversion target.

    This transform modifies the payload. By default, it consumes the `target`
    handle. It does not produce any handles.

    If the `preserve_handles` attribute is set, this transform does not consume
    the `target` handle and instead updates handles based on notifications from
    a tracking listener that is attached to the dialect conversion, similar to
    `transform.apply_patterns`. Only replacements via `RewriterBase::replaceOp`
    or `replaceOpWithNewOp` are considered "payload op replacements". In
    contrast to `transform.apply_patterns`, we allow replacement ops even if the
    op name has changed. This is because conversion patterns are expected to
    lower ops to different ops (from a different dialect). More details can be
    found at the documentation site of `TrackingListener`.

    This transform produces a silenceable failure if the dialect conversion was
    unsuccessful or the tracking listener failed to find a replacement op.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target,
                       OptionalAttr<StrArrayAttr>:$legal_ops,
                       OptionalAttr<StrArrayAttr>:$illegal_ops,
                       OptionalAttr<StrArrayAttr>:$legal_dialects,
                       OptionalAttr<StrArrayAttr>:$illegal_dialects,
                       UnitAttr:$partial_conversion,
                       UnitAttr:$preserve_handles);
  let results = (outs);
  let regions = (region
      MaxSizedRegion<1>:$patterns,
      VariadicRegion<MaxSizedRegion<1>>:$default_type_converter_region);

  let assemblyFormat = [{
    `to` $target $patterns
    (`with` `type_converter` $default_type_converter_region^)?
    attr-dict `:` type($target)
  }];
  let hasVerifier = 1;

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins
        "Value":$target,
        CArg<"function_ref<void(OpBuilder &, Location)>", "nullptr">:
            $patternsBodyBuilder,
        CArg<"function_ref<void(OpBuilder &, Location)>", "nullptr">:
            $typeConverterBodyBuilder)>,
  ];

  let extraClassDeclaration = [{
    ::mlir::transform::TypeConverterBuilderOpInterface getDefaultTypeConverter() {
      if (getDefaultTypeConverterRegion().size() == 0)
        return {};
      return ::llvm::cast<::mlir::transform::TypeConverterBuilderOpInterface>(
          &getDefaultTypeConverterRegion()[0].front().front());
    }
  }];
}

def ApplyToLLVMConversionPatternsOp : Op<Transform_Dialect,
    "apply_conversion_patterns.dialect_to_llvm",
    [DeclareOpInterfaceMethods<ConversionPatternDescriptorOpInterface,
                               ["verifyTypeConverter"]>]> {
  let description = [{
    Collects patterns that convert ops from the specified dialect to LLVM
    dialect ops. These patterns require an "LLVMTypeConverter".

    Note: Only dialects that implement the `ConvertToLLVMPatternInterface` are
    supported. Any conversion target modifications by interface implementations
    are currently ignored. The conversion target is fully specified by the
    enclosing "apply_conversion_patterns" op.
  }];

  let arguments = (ins StrAttr:$dialect_name);
  let assemblyFormat = "$dialect_name attr-dict";
  let hasVerifier = 1;
}

def ApplyDeadCodeEliminationOp : TransformDialectOp<"apply_dce",
    [TransformOpInterface, TransformEachOpTrait,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     ReportTrackingListenerFailuresOpTrait]> {
  let summary = "Eliminate dead operations in the body of the target op";
  let description = [{
    This transform applies dead code elimination (DCE) to the body of the
    targeted op.

    Note: "transform.apply_patterns" with an empty region can also be used to
    remove dead ops. However, that op applies additional simplifications such as
    op folding and region simplification.

    This transform reads the target handle and modifies the payload. Note that
    this transform may silently remove payload ops from handles.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target);
  let results = (outs);
  let assemblyFormat = "`to` $target attr-dict `:` type($target)";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
      ::mlir::transform::TransformRewriter &rewriter,
      ::mlir::Operation *target,
      ::mlir::transform::ApplyToEachResultList &results,
      ::mlir::transform::TransformState &state);
  }];
}

def ApplyPatternsOp : TransformDialectOp<"apply_patterns",
    [TransformOpInterface, TransformEachOpTrait,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     ReportTrackingListenerFailuresOpTrait]
        # GraphRegionNoTerminator.traits> {
  let summary = "Greedily applies patterns to the body of the targeted op";
  let description = [{
    This transform greedily applies the specified patterns to the body of the
    targeted op until a fixpoint was reached. Patterns are not applied to the
    targeted op itself.

    The patterns that should be applied are specified in the graph region of
    this op. They must implement the `PatternDescriptorOpInterface`. The order
    in which patterns are applied is unspecified; i.e., the ordering of ops in
    the region of this op is irrelevant.

    If `apple_cse` is set, the greedy pattern rewrite is interleaved with
    common subexpression elimination (CSE): both are repeated until a fixpoint
    is reached.

    This transform only reads the target handle and modifies the payload. If a
    pattern erases or replaces a tracked op, the mapping is updated accordingly.

    Only replacements via `RewriterBase::replaceOp` or `replaceOpWithNewOp` are
    considered "payload op replacements". Furthermore, only if the replacement
    values are defined by the same op and that op has the same type as the
    original op, the mapping is updated. Otherwise, this transform produces a
    silenceable failure. More details can be found at the documentation site of
    `TrackingListener`.

    This transform also produces a silenceable failure if the pattern
    application did not converge within the default number of
    iterations/rewrites of the greedy pattern rewrite driver.
  }];

  let arguments = (ins
    TransformHandleTypeInterface:$target,
    UnitAttr:$apply_cse,
    DefaultValuedAttr<I64Attr, "static_cast<uint64_t>(-1)">:$max_iterations,
    DefaultValuedAttr<I64Attr, "static_cast<uint64_t>(-1)">:$max_num_rewrites);
  let results = (outs);
  let regions = (region MaxSizedRegion<1>:$patterns);

  let assemblyFormat = "`to` $target $patterns attr-dict `:` type($target)";
  let hasVerifier = 1;

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins
        "Value":$target,
        CArg<"function_ref<void(OpBuilder &, Location)>", "nullptr">:
            $bodyBuilder)>,
  ];

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
      ::mlir::transform::TransformRewriter &rewriter,
      ::mlir::Operation *target,
      ::mlir::transform::ApplyToEachResultList &results,
      ::mlir::transform::TransformState &state);
  }];
}

def ApplyCanonicalizationPatternsOp
    : TransformDialectOp<"apply_patterns.canonicalization",
        [DeclareOpInterfaceMethods<PatternDescriptorOpInterface>]> {
  let summary = "Populates canonicalization patterns";
  let description = [{
    This op populates all canonicalization patterns of all loaded dialects in
    an `apply_patterns` transform.
  }];
  let assemblyFormat = "attr-dict";
}

def ApplyLoopInvariantCodeMotionOp : TransformDialectOp<"apply_licm",
    [TransformOpInterface, TransformEachOpTrait,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     ReportTrackingListenerFailuresOpTrait]> {
  let summary = "Move loop-invariant code out of a loop-like op";
  let description = [{
    This transform moves side-effect free, loop invariant code out of the
    targeted loop-like op. The targeted op must implement the
    `LoopLikeOpInterface`.

    Note: To move invariant ops from a loop nest, this transform must be applied
    to each loop of the loop nest, starting with the inner-most loop.

    This transform reads the target handle and modifies the payload.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target);
  let results = (outs);
  let assemblyFormat = "`to` $target attr-dict `:` type($target)";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
      ::mlir::transform::TransformRewriter &rewriter,
      ::mlir::LoopLikeOpInterface target,
      ::mlir::transform::ApplyToEachResultList &results,
      ::mlir::transform::TransformState &state);
  }];
}

def ApplyRegisteredPassOp : TransformDialectOp<"apply_registered_pass",
    [TransformOpInterface, TransformEachOpTrait,
     FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface]> {
  let summary = "Applies the specified registered pass or pass pipeline";
  let description = [{
    This transform applies the specified pass or pass pipeline to the targeted
    ops. The name of the pass/pipeline is specified as a string attribute, as
    set during pass/pipeline registration. Optionally, pass options may be
    specified as a string attribute. The pass options syntax is identical to the
    one used with "mlir-opt".

    This op first looks for a pass pipeline with the specified name. If no such
    pipeline exists, it looks for a pass with the specified name. If no such
    pass exists either, this op fails definitely.

    This transform consumes the target handle and produces a new handle that is
    mapped to the same op. Passes are not allowed to remove/modify the operation
    that they operate on, so the target op is guaranteed to still exist. The
    target handle is invalidated because a pass may arbitrarily modify the body
    of targeted ops.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target,
                       StrAttr:$pass_name,
                       DefaultValuedAttr<StrAttr, "\"\"">:$options);
  let results = (outs TransformHandleTypeInterface:$result);
  let assemblyFormat = [{
    $pass_name `to` $target attr-dict `:` functional-type(operands, results)
  }];

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
      ::mlir::transform::TransformRewriter &rewriter,
      ::mlir::Operation *target,
      ::mlir::transform::ApplyToEachResultList &results,
      ::mlir::transform::TransformState &state);
  }];
}

def CastOp : TransformDialectOp<"cast",
    [TransformOpInterface, TransformEachOpTrait,
     DeclareOpInterfaceMethods<CastOpInterface>,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let arguments = (ins TransformHandleTypeInterface:$input);
  let results = (outs TransformHandleTypeInterface:$output);
  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($output)";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
      ::mlir::transform::TransformRewriter &rewriter,
      ::mlir::Operation *target,
      ::mlir::transform::ApplyToEachResultList &results,
      ::mlir::transform::TransformState &state);
  }];
}

def NumAssociationsOp : TransformDialectOp<"num_associations",
    [MemoryEffectsOpInterface, ParamProducerTransformOpTrait,
     DeclareOpInterfaceMethods<TransformOpInterface>,
     MatchOpInterface]> {
  let summary =
    "Returns the number of payload objects associated with the argument";
  let description = [{
    Given an argument, handle or parameter, returns a new parameter associated
    with a single 64-bit number that corresponds to the number of payload
    objects (operations or values for a handle, attributes for a parameter)
    associated with the argument.

    Always succeeds.
  }];
  let arguments = (ins Transform_AnyHandleOrParamType:$handle);
  let results = (outs TransformParamTypeInterface:$num);
  let assemblyFormat = [{
    $handle attr-dict `:` functional-type(operands, results)
  }];
  let hasVerifier = 1;
}

def CollectMatchingOp : TransformDialectOp<"collect_matching", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>,
    DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let summary = "Collects all payload ops that match the given named matcher";
  let description = [{
    Collects operations or other payload IR objects nested under `root`
    (inclusive) that match the given matcher expressed as a named sequence. The
    matcher sequence must accept exactly one argument that it is not allowed to
    modify. It must yield as many values as this op has results. Each of the
    yielded values must be associated with exactly one payload object. If any
    operation in the matcher sequence produces a silenceable failure, the
    matcher advances to the next payload operation in the walk order without
    finishing the sequence.

    The i-th result of this operation is constructed by concatenating the i-th
    yielded payload IR objects of all successful matcher sequence applications.
    All results are guaranteed to be mapped to the same number of payload IR
    objects.

    The operation succeeds unless the matcher sequence produced a definite
    failure for any invocation.
  }];

  let arguments = (ins TransformHandleTypeInterface:$root,
                       SymbolRefAttr:$matcher);
  let results = (outs Variadic<Transform_AnyHandleOrParamType>:$results);

  let assemblyFormat = [{
    $matcher `in` $root attr-dict `:` functional-type($root, $results)
  }];
}

def ForeachMatchOp : TransformDialectOp<"foreach_match", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>,
    DeclareOpInterfaceMethods<TransformOpInterface,
                              ["allowsRepeatedHandleOperands"]>,
    DeclareOpInterfaceMethods<OpAsmOpInterface,
                              ["getAsmResultNames"]>]> {
  let summary = "Applies named sequences when a named matcher succeeds";
  let description = [{
    Given a pair of co-indexed lists of transform dialect symbols (such as
    `transform.named_sequence`), walks the payload IR associated with the root
    handle and interprets the symbols as matcher/action pairs by applying the
    body of the corresponding symbol definition. The symbol from the first list
    is the matcher part: if it results in a silenceable error, the error is
    silenced and the next matcher is attempted. Definite failures from any
    matcher stop the application immediately and are propagated unconditionally.
    If none of the matchers succeeds, the next payload operation in walk order
    (post-order at the moment of writing, double check `Operation::walk`) is
    matched. If a matcher succeeds, the co-indexed action symbol is applied and
    the following matchers are not applied to the same payload operation. If the
    action succeeds, the next payload operation in walk order is matched. If it
    fails, both silenceable and definite errors are propagated as the result of
    this op; propagation of silenceable errors is postponed until the end of the
    walk.

    The matcher symbol must take at least one operand of a type that implements
    the same transform dialect interface as the `root` operand (a check is
    performed at application time to see if the associated payload satisfies the
    constraints of the actual type), and may take additional operands with a
    similar type requirement. It must not consume operands as multiple matchers
    may be applied. The matcher may produce any number of results. The action
    symbol paired with the matcher must take the same number of arguments as the
    matcher has results, and these arguments must implement the same transform
    dialect interfaces, but not necessarily have the exact same type (again, a
    check is performed at application time to see if the associated payload
    satisfies the constraints of actual types on both sides).

    The action symbol may have results that are accumulated from all actions and
    returned from the `foreach_match` operation on success. Unless the
    `flatten_results` attribute is present, each action result must be
    associated with exactly one payload entity. The actions are expected to only
    modify payload operations nested in the `root` payload operations associated
    with the operand of this transform operation. Furthermore, the actions may
    not modify operations outside of the currently matched payload operation,
    e.g., they may not modify sibling or parent operations. If such behavior is
    desired, the parent must be matched first and the nested operations obtained
    by traversing the IR from the parent. This is due to the matching being
    performed as a post-order IR walk.

    This operation consumes the operand and produces a new handle associated
    with the same payload. This is necessary to trigger invalidation of handles
    to any of the payload operations nested in the payload operations associated
    with the operand, as those are likely to be modified by actions.

    By default, the root payload operation associated with the operand is not
    matched. This is to support the conservative case where applied actions may
    invalidate the root payload operation. If the optional `restrict_root`
    attribute is set, the root operand is guaranteed to not be invalidated by any
    of the applied actions. In such cases, the root payload operation is also
    matched. This is useful because matching the root payload operation is a
    common idiom, when e.g. matching a func.func directly and operations nested
    under it.

    The operation succeeds if none of the matchers produced a definite failure
    during application and if all of the applied actions produced success. Note
    that it also succeeds if all the matchers failed on all payload operations,
    i.e. failure to apply is not an error. The operation produces a silenceable
    failure if any applied action produced a silenceable failure. In this case,
    the resulting handle is associated with an empty payload. The operation
    produces a definite failure if any of the applied matchers or actions
    produced a definite failure.
  }];

  let arguments =
      (ins TransformHandleTypeInterface:$root,
           Variadic<Transform_AnyHandleOrParamType>:$forwarded_inputs,
           UnitAttr:$restrict_root,
           UnitAttr:$flatten_results,
           SymbolRefArrayAttr:$matchers,
           SymbolRefArrayAttr:$actions);
  let results =
      (outs TransformHandleTypeInterface:$updated,
            Variadic<Transform_AnyHandleOrParamType>:$forwarded_outputs);

  let assemblyFormat = [{
    oilist( `restrict_root` $restrict_root
          | `flatten_results` $flatten_results
          )
    `in`
    $root (`,` $forwarded_inputs^)?
    custom<ForeachMatchSymbols>($matchers, $actions)
    attr-dict
    `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

def ForeachOp : TransformDialectOp<"foreach",
    [DeclareOpInterfaceMethods<TransformOpInterface>,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     DeclareOpInterfaceMethods<RegionBranchOpInterface, [
         "getSuccessorRegions", "getEntrySuccessorOperands"]>,
     SingleBlockImplicitTerminator<"::mlir::transform::YieldOp">
    ]> {
  let summary = "Executes the body for each element of the payload";
  let description = [{
    Execute the op's body - its single region block - exactly once per
    element of the payload associated to a target handle. The body's
    transformations are applied in order of appearance until reaching the
    (implicit) YieldOp terminator.

    Each iteration gets executed by co-indexing the payloads of the arguments
    and mapping the body's arguments to these tuples, as though iterating over
    the zipped together `targets`. As such, in each iteration, the size of the
    payload of each of the body's block arguments is exactly one. The attribute
    `zip_shortest` can be used if the targets vary in their number of payloads;
    this will limit the iterations to only the number of payloads found in the
    shortest target.

    This op always reads the target handles. Furthermore, it consumes a handle
    if there is a transform op in the body that consumes the corresponding
    block argument. Handles can point to ops, values, or parameters.

    #### Return Modes

    This op produces as many result handles as the body's terminating YieldOp
    has operands. For each result, the payloads of the corresponding YieldOp
    operand are merged and mapped to the same resulting handle.

    If the target handles do not associate payloads of the same size, a
    silencable failure will be generated.

    During application, if any transformation in the sequence fails, the entire
    sequence fails immediately with the same failure, leaving the payload IR in
    a potentially invalid state, i.e., this operation offers no transformation
    rollback capabilities.
  }];

  let arguments = (ins Variadic<Transform_AnyHandleOrParamType>:$targets,
                       UnitAttr:$with_zip_shortest);
  let results = (outs Variadic<Transform_AnyHandleOrParamType>:$results);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat =
    "$targets oilist(`with_zip_shortest` $with_zip_shortest) `:` "
    "type($targets) (`->` type($results)^)? $body attr-dict";
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    /// Allow the dialect prefix to be omitted.
    static StringRef getDefaultDialect() { return "transform"; }

    transform::YieldOp getYieldOp();
  }];
}

def GetConsumersOfResult : TransformDialectOp<"get_consumers_of_result",
    [DeclareOpInterfaceMethods<TransformOpInterface>,
     NavigationTransformOpTrait, MemoryEffectsOpInterface]> {
  let summary = "Get handle to the consumers of this operation's result number";
  let description = [{
    The handle defined by this Transform op corresponds to all operations that
    consume the SSA value defined by the `target` and `result_number`
    arguments.
    This operation applies to a single payload operation, otherwise it produces
    a definite failure.
    The return handle points to the consuming operations operations, which can
    be empty.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target,
                       I64Attr:$result_number);
  let results = (outs TransformHandleTypeInterface:$consumers);
  let assemblyFormat = "$target `[` $result_number `]` attr-dict `:` "
                       "functional-type(operands, results)";
}

def GetDefiningOp : TransformDialectOp<"get_defining_op",
    [DeclareOpInterfaceMethods<TransformOpInterface>,
     MatchOpInterface,
     NavigationTransformOpTrait, MemoryEffectsOpInterface]> {
  let summary = "Get handle to the defining op of a value";
  let description = [{
    The handle defined by this Transform op corresponds to the defining op of
    the targeted value.

    This transform produces a silenceable failure if the targeted value is a
    block argument.
  }];

  let arguments = (ins TransformValueHandleTypeInterface:$target);
  let results = (outs TransformHandleTypeInterface:$result);
  let assemblyFormat = "$target attr-dict `:` "
                       "functional-type(operands, results)";
}

def GetParentOp : TransformDialectOp<"get_parent_op",
    [DeclareOpInterfaceMethods<TransformOpInterface>,
     MatchOpInterface,
     NavigationTransformOpTrait, MemoryEffectsOpInterface]> {
  let summary = "Gets handles to the closest parent ops";
  let description = [{
    The handle defined by this Transform op corresponds to the parents of the
    targeted payload ops (in the same order).

    Requirements that parent ops must fulfill can be optionally specified. In
    that case for each target op, the closest parent op that fulfills all
    requirements, is returned.
    - `isolated_from_above`: the parent op must be isolated from above
    - `allow_empty_results`: get_parent_op is allowed to return an empty list
      and still succeeds. In such a case, if `get_parent_op` fails for any
      operation in the list, the entire transform returns an empty handle.
    - `op_name`: the parent op must have the specified name
    - `nth_parent`: get the n-th parent of that satisfies the above requirements

    If `deduplicate` is set, the result handle does not contain any duplicate
    ops. For example, given the list
    "(childof(A), childof(B), childof(B), childof(A), childof(B))", the
    resulting list will be just "(A, B)". Note that no other semantic ordering
    is applied, e.g., "B" may itself be a parent of "A". This may have an impact
    on the further transformation applied to the handle produced here.

    If any of the given Payload IR ops has no such suitable parent, then:
      - if `allow_empty_results` is set, the result handle is empty
      - otherwise, the transformation produces a silenceable failure.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target,
                       UnitAttr:$isolated_from_above,
                       UnitAttr:$allow_empty_results,
                       OptionalAttr<StrAttr>:$op_name,
                       UnitAttr:$deduplicate,
                       DefaultValuedAttr<ConfinedAttr<I64Attr, [IntPositive]>,
                                         "1">:$nth_parent);
  let results = (outs TransformHandleTypeInterface:$parent);
  let assemblyFormat =
    "$target attr-dict `:` functional-type(operands, results)";
}

def GetProducerOfOperand : TransformDialectOp<"get_producer_of_operand",
    [DeclareOpInterfaceMethods<TransformOpInterface>,
     NavigationTransformOpTrait, MatchOpInterface, MemoryEffectsOpInterface]> {
  let summary = "Get handle to the producer of this operation's operand number";
  let description = [{
    The handle defined by this Transform op corresponds to operation that
    produces the SSA value defined by the `target` and `operand_number`
    arguments. If the origin of the SSA value is not an operations (i.e. it is
    a block argument), the transform produces a silenceable failure.
    The return handle points to only the subset of successfully produced
    computational operations, which can be empty.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target,
                       I64Attr:$operand_number);
  let results = (outs TransformHandleTypeInterface:$producer);
  let assemblyFormat = "$target `[` $operand_number `]` attr-dict `:` "
                       "functional-type(operands, results)";
}

def GetOperandOp : TransformDialectOp<"get_operand",
    [DeclareOpInterfaceMethods<TransformOpInterface>,
     NavigationTransformOpTrait, MatchOpInterface, MemoryEffectsOpInterface]> {
  let summary = "Get a handle to the operand(s) of the targeted op";
  let description = [{
    The handle defined by this Transform op corresponds to the operands of the
    given `target` operation specified by the given set of positions. There are
    three possible modes:

     - Position list directly, i.e. `%target[0, 1, 2]`. This will return the
       operands at the specified positions.
     - Inverted position list, i.e. `%target[except(0, 1, 2)]`. This will return
       all operands except those at the given positions.
     - All, i.e. `%target[all]`. This will return all operands of the operation.
    
    This transform produces a silenceable failure if any of the operand indices
    exceeds the number of operands in the target. It reads the target handle and
    produces the result handle.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target,
                       DenseI64ArrayAttr:$raw_position_list,
                       UnitAttr:$is_inverted,
                       UnitAttr:$is_all);
  let results = (outs TransformValueHandleTypeInterface:$result);
  let assemblyFormat =
      "$target `[`"
      "custom<TransformMatchDims>($raw_position_list, $is_inverted, $is_all)"
      "`]` attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def GetResultOp : TransformDialectOp<"get_result",
    [DeclareOpInterfaceMethods<TransformOpInterface>,
     NavigationTransformOpTrait, MemoryEffectsOpInterface]> {
  let summary = "Get a handle to the result(s) of the targeted op";
  let description = [{
    The handle defined by this Transform op correspond to the OpResults of the
    given `target` operation. Optionally `result_number` can be specified to
    select a specific result.
    
    This transform fails silently if the targeted operation does not have enough
    results. It reads the target handle and produces the result handle.

    The handle defined by this Transform op corresponds to the results of the
    given `target` operation specified by the given set of positions. There are
    three possible modes:

     - Position list directly, i.e. `%target[0, 1, 2]`. This will return the
       results at the specified positions.
     - Inverted position list, i.e. `%target[except(0, 1, 2)]`. This will return
       all results except those at the given positions.
     - All, i.e. `%target[all]`. This will return all results of the operation.
    
    This transform produces a silenceable failure if any of the result indices
    exceeds the number of results returned by the target. It reads the target
    handle and produces the result handle.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target,
                       DenseI64ArrayAttr:$raw_position_list,
                       UnitAttr:$is_inverted,
                       UnitAttr:$is_all);
  let results = (outs TransformValueHandleTypeInterface:$result);
  let assemblyFormat =
      "$target `[`"
      "custom<TransformMatchDims>($raw_position_list, $is_inverted, $is_all)"
      "`]` attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def GetTypeOp : TransformDialectOp<"get_type",
    [DeclareOpInterfaceMethods<TransformOpInterface>,
     MatchOpInterface,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Get a parameter containing the type of the given value";
  let description = [{
    This operation creates a new Transform parameter containing the
    type(s) of the value(s) associated with the operand handle.

    This transform never fails.
  }];

  let arguments = (ins TransformValueHandleTypeInterface:$value,
                       UnitAttr:$elemental);
  let results = (outs TransformParamTypeInterface:$type_param);
  let assemblyFormat = "(`elemental` $elemental^)? $value attr-dict `:`"
                       "functional-type(operands, results)";
}

def IncludeOp : TransformDialectOp<"include",
    [CallOpInterface,
     MatchOpInterface,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     DeclareOpInterfaceMethods<SymbolUserOpInterface>,
     DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let summary = "Includes a named transform sequence";
  let description = [{
    The application of this transform operation is equivalent to applying the
    operations contained in the named transform sequence with operands being
    remapped to block arguments. The behavior of the operation when a
    transformation in the included named sequence produces a silenceable error
    is controlled by the `failure_propagation_mode` attribute. When set to
    `propagate`, the failure of any nested transformation in the sequence
    implies immediate failure of the entire sequence with a silenceable error,
    and no further transformation is attempted. When set to `suppress`,
    silenceable errors in nested operations are ignored and further
    transformations are applied. Beware that even silenceable errors may leave
    the payload IR in a state unsuitable for further transformations. It is the
    responsibility of the user to ensure the following transformations are
    robust enough when errors are suppressed. Definite errors are propagated
    immediately regardless of the mode. The objects associated with the results
    of this operation are the same as those associated with the operands of the
    `transform.yield` in the referenced named sequence.
  }];

  let arguments = (ins SymbolRefAttr:$target,
                       FailurePropagationMode:$failure_propagation_mode,
                       Variadic<Transform_AnyHandleOrParamType>:$operands);
  let results = (outs Variadic<Transform_AnyHandleOrParamType>:$results);

  let assemblyFormat =
      "$target `failures` `(` $failure_propagation_mode `)`"
      "`(` $operands `)` attr-dict `:` functional-type($operands, $results)";

  let extraClassDeclaration = [{
    ::mlir::CallInterfaceCallable getCallableForCallee() {
      return getTarget();
    }

    void setCalleeFromCallable(::mlir::CallInterfaceCallable callee) {
      setTargetAttr(cast<SymbolRefAttr>(callee));
    }

    ::mlir::Operation::operand_range getArgOperands() {
      return getOperands();
    }

    ::mlir::MutableOperandRange getArgOperandsMutable() {
      return getOperandsMutable();
    }
  }];
}

def MatchOperationEmptyOp : Op<Transform_Dialect, "match.operation_empty", [
    AtMostOneOpMatcher,
    MatchOpInterface,
    MemoryEffectsOpInterface]> {
  let summary =
    "Matches if the handle is not associated to any op";
  let description = [{
    Succeeds if the handle is not associated to any op.
  }];
  let arguments = (ins TransformHandleTypeInterface:$operand_handle);
  let assemblyFormat =
      "$operand_handle attr-dict `:` type($operand_handle)";
  let extraClassDeclaration = AtMostOneOpMatcher.extraDeclaration;
}

def MatchOperationNameOp : TransformDialectOp<"match.operation_name",
    [SingleOpMatcher,
     MatchOpInterface,
     MemoryEffectsOpInterface]> {
  let summary = "Matches a single operation of one of the given kinds";
  let description = [{
    Succeeds if the operation associated with the operand handle has one of the
    given operation names. Produces a silenceable failure otherwise.

    If more than one payload operation is associated with the operand handle,
    produces a definite failure.
  }];

  let arguments = (ins TransformHandleTypeInterface:$operand_handle,
                       StrArrayAttr:$op_names);
  let assemblyFormat =
      "$operand_handle $op_names attr-dict `:` type($operand_handle)";
  let extraClassDeclaration = SingleOpMatcher.extraDeclaration;
}

def MatchParamCmpIOp : Op<Transform_Dialect, "match.param.cmpi", [
    DeclareOpInterfaceMethods<TransformOpInterface>,
    MatchOpInterface,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    SameTypeOperands]> {
  let summary =
    "Matches if two parameter lists are associated with the same value";
  let description = [{
    Succeeds if all of the co-indexed values associated with the given
    parameters relate as specified by the predicate (greater than, less than,
    equal to, or their combinations). Comparison treats all values as signed.
    Produces a silenceable failure otherwise.
  }];
  let arguments = (ins TransformParamTypeInterface:$param,
                       TransformParamTypeInterface:$reference,
                       MatchCmpIPredicateAttr:$predicate);
  let assemblyFormat =
      "$predicate $param `,` $reference attr-dict `:` type($param)";
}

def MergeHandlesOp : TransformDialectOp<"merge_handles",
    [DeclareOpInterfaceMethods<TransformOpInterface, ["allowsRepeatedHandleOperands"]>,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     MatchOpInterface, SameOperandsAndResultType]> {
  let summary = "Merges handles into one pointing to the union of payload ops";
  let description = [{
    Creates a new Transform IR handle value that points to the same Payload IR
    operations/values/parameters as the operand handles. The Payload IR elements
    are listed in the same order as they are in the operand handles, grouped by
    operand handle, e.g., all Payload IR associated with the first handle comes
    first, then all Payload IR associated with the second handle and so on. If
    `deduplicate` is set, do not add the given Payload IR operation, value, or
    parameter more than once to the final list regardless of it coming from the
    same or different handles. Consumes the operands and produces a new handle.
  }];

  let arguments = (ins Variadic<Transform_AnyHandleOrParamType>:$handles,
                       UnitAttr:$deduplicate);
  let results = (outs Transform_AnyHandleOrParamType:$result);
  let assemblyFormat = "(`deduplicate` $deduplicate^)? $handles attr-dict `:` type($result)";
  let hasFolder = 1;
}

def NamedSequenceOp : TransformDialectOp<"named_sequence",
    [FunctionOpInterface,
     IsolatedFromAbove,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let summary = "Named transform sequence that can be included elsewhere";
  let description = [{
    Defines a named (callable, function-like) sequence of other Transform
    dialect operations that can be included using `transform.include` as part of
    another Transform dialect construct. This sequence is not processed
    immediately but rather dispatched to when the inclusion is processed. The
    arguments and results can be used to communicate a subset of mapping into
    the named sequence. The sequence must consist of a single block and end with
    a `transform.yield` terminator. The operands of the terminator become the
    results of the `transform.include`.

    When dispatched to, the operations in the named sequence are executed one by
    one, similarly to the regular unnamed sequence. The failure propagation mode
    is specified on the `transform.include`. Different inclusions may use
    different failure propagation modes. This transform operation always
    succeeds by itself, but the inclusion may fail if any of the operations
    fail.

    Named sequences can only appear at the top-level of the Transform dialect
    nesting structure. That is, they cannot be nested in other Transform dialect
    operations. Furthermore, one of the ancestors must have the `SymbolTable`
    trait and have the `transform.with_named_sequence` attribute attached.

    Named sequences may include other named sequences via `transform.include`,
    but recursion is *not* allowed.
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrBase<"::mlir::FunctionType",
                 "function type attribute">:$function_type,
    OptionalAttr<StrAttr>:$sym_visibility,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs);
  let regions = (region MaxSizedRegion<1>:$body);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

  let builders = [
    // Build a named sequence.
    OpBuilder<(ins
      "StringRef":$symName,
      "Type":$rootType,
      "TypeRange":$resultType,
      "SequenceBodyBuilderFn":$bodyBuilder,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs,
      CArg<"ArrayRef<DictionaryAttr>", "{}">:$argAttrs)>
  ];

  let extraClassDeclaration = [{
    ::llvm::ArrayRef<::mlir::Type> getArgumentTypes() {
      return getFunctionType().getInputs();
    }
    ::llvm::ArrayRef<::mlir::Type> getResultTypes() {
      return getFunctionType().getResults();
    }
    ::mlir::Region *getCallableRegion() {
      return &getBody();
    }
  }];
}

def SplitHandleOp : TransformDialectOp<"split_handle",
    [FunctionalStyleTransformOpTrait,
     DeclareOpInterfaceMethods<TransformOpInterface>,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Splits a handle or parameter into multiple values";
  let description = [{
    Splits `handle` into one or multiple handles, as specified by the number
    of results of this operation. `handle` should be mapped to as many payload
    ops, values or parameteres as there are results. Otherwise, this transform
    will fail producing a silenceable failure by default. Each result handle
    is mapped to exactly one payload unless specified otherwise by attributes
    described below. The order of the payloads is preserved,  i.e., the i-th
    payload is mapped to the i-th result handle.

    This operation is useful for ensuring a statically known number of
    payloads are tracked by the source `handle` and to extract them into
    individual handles that can be further manipulated in isolation.

    If there are more payloads than results, the remaining payloads are mapped to
    the result with index `overflow_result`. If no `overflow_result` is
    specified, the transform produces a silenceable failure.

    If there are fewer payload ops than results, the transform produces a
    silenceable failure if `fail_on_payload_too_small` is set to "true".
    Otherwise, it succeeds and the remaining result handles are not mapped to
    anything. It also succeeds if `handle` is empty and
    `pass_through_empty_handle` is set to "true", regardless of
    `fail_on_payload_too_small`.
  }];

  let arguments = (ins Transform_AnyHandleOrParamType:$handle,
                       DefaultValuedAttr<BoolAttr, "true">:$pass_through_empty_handle,
                       DefaultValuedAttr<BoolAttr, "true">:$fail_on_payload_too_small,
                       OptionalAttr<I64Attr>:$overflow_result);
  let results = (outs Variadic<Transform_AnyHandleOrParamType>:$results);
  let hasVerifier = 1;

  let builders = [
    OpBuilder<(ins "Value":$handle, "int64_t":$numResultHandles)>
  ];

  let assemblyFormat = [{
    $handle attr-dict `:` functional-type(operands, results)
  }];
}

def ParamConstantOp : Op<Transform_Dialect, "param.constant", [
    MatchOpInterface,
    DeclareOpInterfaceMethods<TransformOpInterface>,
    MemoryEffectsOpInterface,
    ParamProducerTransformOpTrait]> {
  let summary = "Produces a new transform dialect parameter value associated "
                "with the given attribute";
  let description = [{
    Produces a new transform dialect parameter associated with the singleton
    list containing the given attribute. The operation itself always succeeds,
    but the general association check may fail if the parameter type does not
    accept the given kind of attribute as valid.
  }];
  let arguments = (ins AnyAttr:$value);
  let results = (outs TransformParamTypeInterface:$param);
  let assemblyFormat = "$value attr-dict `->` type($param)";
}

def PrintOp : TransformDialectOp<"print",
    [DeclareOpInterfaceMethods<TransformOpInterface>,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     MatchOpInterface]> {
  let summary = "Dump each payload op";
  let description = [{
    Prints each payload op that is associated with the `target` operand to
    `stdout`. It also prints the `name` string attribute. If no target is
    specified, the top-level op is dumped.

    This op is useful for printf-style debugging.

    Supported printing flag attributes:
    * `assume_verified` -- skips verification when the unit attribute is
      specified. This improves performace but may lead to crashes and
      unexpected behavior when the printed payload op is invalid.
    * `use_local_scope` -- prints in local scope when the unit attribute is
      specified. This improves performance but may not be identical to
      printing within the full module.
    * `skip_regions` -- does not print regions of operations when the unit
      attribute is specified.
  }];

  let arguments = (ins Optional<TransformHandleTypeInterface>:$target,
                       OptionalAttr<StrAttr>:$name,
                       OptionalAttr<UnitAttr>:$assume_verified,
                       OptionalAttr<UnitAttr>:$use_local_scope,
                       OptionalAttr<UnitAttr>:$skip_regions);
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

  let arguments = (ins TransformHandleTypeInterface:$pattern,
                       Variadic<Transform_AnyHandleOrParamType>:$handles);
  let results = (outs Variadic<Transform_AnyHandleOrParamType>:$replicated);
  let assemblyFormat = "`num` `(` $pattern `)` $handles attr-dict `:` "
                       "type($pattern) `,` type($handles)";
}

def SelectOp : TransformDialectOp<"select",
    [DeclareOpInterfaceMethods<TransformOpInterface>,
     NavigationTransformOpTrait, MemoryEffectsOpInterface]> {
  let summary = "Select payload ops by name";
  let description = [{
    The handle defined by this Transform op corresponds to all operations among
    `target` that have the specified properties. Currently the following
    properties are supported:

    - `op_name`: The op must have the specified name.

    The result payload ops are in the same relative order as the targeted ops.
    This transform op reads the `target` handle and produces the `result`
    handle. It reads the payload, but does not modify it.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target,
                       StrAttr:$op_name);
  let results = (outs TransformHandleTypeInterface:$result);
  let assemblyFormat = [{
    $op_name `in` $target attr-dict `:` functional-type(operands, results)
  }];
}

def SequenceOp : TransformDialectOp<"sequence",
    [DeclareOpInterfaceMethods<RegionBranchOpInterface,
        ["getEntrySuccessorOperands", "getSuccessorRegions",
         "getRegionInvocationBounds"]>,
     MatchOpInterface,
     DeclareOpInterfaceMethods<TransformOpInterface>,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     OpAsmOpInterface, PossibleTopLevelTransformOpTrait,
     SingleBlockImplicitTerminator<"::mlir::transform::YieldOp">,
     AttrSizedOperandSegments]> {
  let summary = "Contains a sequence of other transform ops to apply";
  let description = [{
    The transformations indicated by the sequence are applied in order of their
    appearance. Each value produced by a transformation within the sequence
    corresponds to a group of operations or values in the payload IR, or to a
    group of parameters, depending on the type of the value. The behavior of the
    operation when a nested transformation produces a silenceable error is
    controlled by the `failure_propagation_mode` attribute. When set to
    `propagate`, the failure of any nested transformation in the sequence
    implies immediate failure of the entire sequence with a silenceable error,
    and no further transformation is attempted. When set to `suppress`,
    silenceable errors in nested operations are ignored and further
    transformations are applied. Beware that even silenceable errors may leave
    the payload IR in a state unsuitable for further transformations. It is the
    responsibility of the caller to ensure the following transformations are
    robust enough when errors are suppressed. Definite errors reported by nested
    transformations abort the sequence regardless of the propagation mode. The
    set of modes may be extended in the future, e.g., to collect silenceable
    errors and report them after attempting all transformations in the sequence.

    The entry block of this operation has a single argument that maps to either
    the operand if provided or the top-level container operation of the payload
    IR, typically the root operation of the pass interpreting the transform
    dialect. Operand omission is only allowed for sequences not contained in
    another sequence.

    The type of the block argument must match the type of the operand. If the
    sequence is a top-level transform (without an operand), it can be used for
    matching operations if the specified type within the top-level container
    payload IR (including the container op itself). E.g.:

    ```mlir
    transform.sequence failures(propagate) {
    ^bb1(%arg1: !transform.any_op):
      // %arg1 is mapped to the top-level container of the payload IR, which is
      // typically a module
    }

    transform.sequence failures(propagate) {
    ^bb1(%arg1: !transform.op<"func.func>"):
      // %arg1 is mapped to all "func.func" ops within and including the
      // top-level container of the payload IR. Nested operations that have the
      // specified op type are not included.
    }
    ```

    The body of the sequence terminates with an implicit or explicit
    `transform.yield` op. The operands of the terminator are returned as the
    results of the sequence op.
  }];

  let arguments = (ins FailurePropagationMode:$failure_propagation_mode,
                       Optional<TransformHandleTypeInterface>:$root,
                       Variadic<Transform_AnyHandleOrParamType>:$extra_bindings);
  let results = (outs Variadic<TransformHandleTypeInterface>:$results);
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat =
    "custom<SequenceOpOperands>($root, type($root), $extra_bindings, type($extra_bindings))"
    " (`->` type($results)^)? `failures` `(` "
    "$failure_propagation_mode `)` attr-dict-with-keyword regions";

  let builders = [
    // Build a sequence with a root.
    OpBuilder<(ins
        "::mlir::TypeRange":$resultTypes,
        "::mlir::transform::FailurePropagationMode":$failure_propagation_mode,
        "::mlir::Value":$root, "SequenceBodyBuilderFn":$bodyBuilder)>,

    // Build a sequence with a root and additional arguments.
    OpBuilder<(ins
        "::mlir::TypeRange":$resultTypes,
        "::mlir::transform::FailurePropagationMode":$failure_propagation_mode,
        "::mlir::Value":$root, "::mlir::ValueRange":$extraBindings,
        "SequenceBodyBuilderArgsFn":$bodyBuilder)>,

    // Build a top-level sequence (no root).
    OpBuilder<(ins
        "::mlir::TypeRange":$resultTypes,
        "::mlir::transform::FailurePropagationMode":$failure_propagation_mode,
        "::mlir::Type":$bbArgType, "SequenceBodyBuilderFn":$bodyBuilder)>,

    // Build a top-level sequence (no root) with extra arguments.
    OpBuilder<(ins
        "::mlir::TypeRange":$resultTypes,
        "::mlir::transform::FailurePropagationMode":$failure_propagation_mode,
        "::mlir::Type":$bbArgType, "::mlir::TypeRange":$extraBindingTypes,
        "SequenceBodyBuilderArgsFn":$bodyBuilder)>
  ];

  let extraClassDeclaration = [{
    /// Allow the dialect prefix to be omitted.
    static StringRef getDefaultDialect() { return "transform"; }
  }];

  let hasVerifier = 1;
}

def VerifyOp : TransformDialectOp<"verify",
    [TransformOpInterface, TransformEachOpTrait,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     ReportTrackingListenerFailuresOpTrait]> {
  let summary = "Verifies the targeted ops";
  let description = [{
    This transform verifies the targeted ops. If at least one op fails to
    verify, the transform produces a definite failure.

    Note: This op was designed for debugging purposes and should be used like an
    assertion. It is intentional that this op produces a definite failure and
    not a silenceable one. Correctness of the program should not depend on this
    op.

    This transform reads the target handle.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target);
  let results = (outs);
  let assemblyFormat = "$target attr-dict `:` type($target)";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
      ::mlir::transform::TransformRewriter &rewriter,
      ::mlir::Operation *target,
      ::mlir::transform::ApplyToEachResultList &results,
      ::mlir::transform::TransformState &state);
  }];
}

def YieldOp : TransformDialectOp<"yield",
    [Terminator, DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Yields operation handles from a transform IR region";
  let description = [{
    This terminator operation yields operation handles from regions of the
    transform IR ops back to the containing op. It is not itself associated with
    any transformation on the payload IR and is used for flow purposes only.
  }];

  let arguments = (ins
    Arg<Variadic<Transform_AnyHandleOrParamType>,
        "Transform values yielded back to the parent"
        >:$operands);
  let assemblyFormat = "operands attr-dict (`:` type($operands)^)?";

  let builders = [
    OpBuilder<(ins), [{
      return build($_builder, $_state, ::mlir::ValueRange());
    }]>
  ];
}

#endif // MLIR_DIALECT_TRANSFORM_IR_TRANSFORMOPS
