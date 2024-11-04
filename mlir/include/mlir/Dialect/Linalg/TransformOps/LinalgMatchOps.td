//===- LinalgMatchOps.td - Linalg transform matcher ops ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LINALG_MATCH_OPS
#define LINALG_MATCH_OPS

include "mlir/Dialect/Linalg/TransformOps/LinalgTransformEnums.td"
include "mlir/Dialect/Transform/IR/MatchInterfaces.td"
include "mlir/Dialect/Transform/IR/TransformAttrs.td"
include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/IR/TransformTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Structured match op and predicates usable inside it.
//===----------------------------------------------------------------------===//

def MatchStructuredOp : Op<Transform_Dialect, "match.structured", [
    MatchOpInterface,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    SingleOpMatcher,
    SingleBlockImplicitTerminator<"::mlir::transform::MatchStructuredYieldOp">]> {
  let summary =
      "Matches a structured (linalg) operation with additional conditions";
  let description = [{
    Checks if the payload operation associated with the operand handle is a
    structured operation, that is, an operation that implements
    `LinalgOpInterface`, and that all conditions listed in the body of this
    operation are satisfied. Produces a silenceable failure if the payload
    operation is not structured.

    The transform operations nested in the body region are applied one by one.
    If any of them produces a failure, silenceable or definite, the following
    operations are not applied. If the failure propagation mode is "propagate",
    silenceable failures are forwarded as the result of this operation. If it is
    "suppress", they are ignored and this operation immediately succeeds.
    Definite failures are always propagated immediately.

    In case of success, the transform values produced by this operation are
    associated with the same payload as the operands of the block terminator. If
    any of the nested operations produced a silenceable failure, regardless of
    the failure propagation mode, the transform values produced by this
    operation that correspond to the already defined terminator operands are
    associated with the same payload as the already defined terminator operands.
    Other values produced by this operation are associated with empty payloads.

    If the failure propagation mode is not specified, it is considered
    "propagate" by default. The "suppress" mode can be used to specify optional
    matches.

    #### Return modes

    This operation only reads all operand handles and produces all resulting
    handles. It succeeds in "propagate" mode if the payload operation is a
    structured operation and if all the nested operations succeed. It succeeds
    in "suppress" mode as long as the operand handle is associated with exactly
    one payload operation. It produces a definite failure when the handle is
    not associated with exactly one payload operation.
  }];

  let arguments = (ins TransformHandleTypeInterface:$current,
                       OptionalAttr<FailurePropagationMode>:$failure_propagation_mode);
  let results = (outs Variadic<Transform_AnyHandleOrParamType>:$outputs);

  let regions = (region SizedRegion<1>:$body_region);
  let assemblyFormat =
    "(`failures` `(` $failure_propagation_mode^ `)`)?"
    "$current `:` custom<SemiFunctionType>(type($current), type($outputs))"
    "attr-dict-with-keyword regions";
  let hasVerifier = 1;

  let extraClassDeclaration = SingleOpMatcher.extraDeclaration # [{
    ::mlir::Value getOperandHandle() { return getCurrent(); }
  }];
}

def StructuredPredicate : NativeOpTrait<"StructuredOpPredicateOpTrait"> {
  let cppNamespace = "::mlir::transform";
  string extraDescription = [{
    This op can only appear immediately inside a `transform.match.structured`
    op and apply to its first block argument because it assumes the payload
    to have been already checked for being a single structured op.
  }];
}

def MatchStructuredBodyOp : Op<Transform_Dialect, "match.structured.body", [
    SingleOpMatcher,
    StructuredPredicate,
    MatchOpInterface,
    MemoryEffectsOpInterface]> {
  let summary =
      "Checks if the body of the structured op satisfies some criteria";
  let description = !strconcat([{
    Checks if the body of the structured payload op satisfies one of the
    following mutually exclusive criteria specified by attributes:

      * `reduction_position`: the body of the structured payload op implements
        a reduction of the `n`-th operand (`n` is the value of the attribute)
        using a single combiner operation;

      * `passthrough`: the body of the structured payload op only forwards
        inputs to the outputs (copy or broadcast).

      * `elementwise`: the body of the structured payload op represents an
        elementwise operation.

      * `contraction`: the body of the structured payload op is a contraction
        of the form `<red>(<elem>(bbarg0, bbarg1), bbarg2)` where `<elem>` and
        `<red>` are binary operations whose names are specified in the attribute
        and operands can be permuted and optionally forwarded through a chain of
        unary side effect-free operations.

  }], StructuredPredicate.extraDescription, [{

    #### Return modes

    Succeeds if the operation body satisfies the specified criteria, produces a
    silenceable failure otherwise. Produces a definite failure if the operand is
    not associated with a single payload op.
  }]);
  let arguments = (ins TransformHandleTypeInterface:$operand_handle,
                       OptionalAttr<I64Attr>:$reduction_position,
                       UnitAttr:$passthrough,
                       UnitAttr:$elementwise,
                       OptionalAttr<StrArrayAttr>:$contraction);
  let assemblyFormat = "$operand_handle attr-dict `:` type($operand_handle)";
  let extraClassDeclaration = SingleOpMatcher.extraDeclaration;
  let hasVerifier = 1;
}

def MatchStructuredClassifyContractionDimsOp
    : Op<Transform_Dialect, "match.structured.classify_contraction_dims", [
    SingleOpMatcher,
    StructuredPredicate,
    MatchOpInterface,
    MemoryEffectsOpInterface]> {
  let summary =
      "Checks if an operation has contraction-like dimensions and returns them";
  let description = !strconcat([{
    Checks if the structured payload op has contraction-like dimensions as
    follows:

      C(batch, m, n) += A(batch, m, k) * B(batch, k, n)

    That is:
    
      - 'batch' are parallel dimensions used in inputs and result;
      - 'm' are parallel dimensions used in the LHS and result;
      - 'n' are parallel dimensions used in rhe RHS and result;
      - 'k' are reduction dimensions present only in LHS and RHS.

    Note that this doesn't check the operation in the body.

  }], StructuredPredicate.extraDescription, [{

    #### Return modes

    Succeeds if the operation has the contraction-like dimensions, produces a
    silenceable failure otherwise.
  }]);

  let arguments = (ins TransformHandleTypeInterface:$operand_handle);
  let results = (outs TransformParamTypeInterface:$batch,
                      TransformParamTypeInterface:$m,
                      TransformParamTypeInterface:$n,
                      TransformParamTypeInterface:$k);
  let assemblyFormat =
    "$operand_handle attr-dict `:` functional-type(operands, results)";
  let extraClassDeclaration = SingleOpMatcher.extraDeclaration;
}

def MatchStructuredClassifyConvolutionDimsOp
    : Op<Transform_Dialect, "match.structured.classify_convolution_dims", [
    SingleOpMatcher,
    StructuredPredicate,
    MatchOpInterface,
    MemoryEffectsOpInterface]> {
  let summary =
      "Checks if an operation has convolution-like dimensions and returns them";
  let description = !strconcat([{
    Checks if the structured payload op has convolution-like dimensions as
    follows:

      C(batch, depth, oi, oc) += A(batch, depth, oi, ic) * B(fl, depth, ic, oc)

    That is:

      - 'batch' are parallel dimensions used in the input and result;
      - 'output_image' ('oi') are parallel dimensions used in the input and result;
      - 'output_channel' ('oc') are parallel dimensions used in the filter and result;
      - 'filter_loop' ('fl') are reduction dimensions representing the dimensions of the sliding window;
      - 'input_channel' ('ic') are reduction dimensions present only in the input and filter.
      - 'depth' ('ic') are parallel dimensions present in the input, filter, and output.

    Additionally this will match stride and dilation information for the convolution:
      - 'strides' are the static strides per convolution window dimension;
      - 'dilations' are the static dilations per convolution window dimension.

    Note that this doesn't check the operation in the body.

  }], StructuredPredicate.extraDescription, [{

    #### Return modes

    Succeeds if the operation has the convolution-like dimensions, produces a
    silenceable failure otherwise.
  }]);

  let arguments = (ins TransformHandleTypeInterface:$operand_handle);
  let results = (outs TransformParamTypeInterface:$batch,
                      TransformParamTypeInterface:$output_image,
                      TransformParamTypeInterface:$output_channel,
                      TransformParamTypeInterface:$filter_loop,
                      TransformParamTypeInterface:$input_channel,
                      TransformParamTypeInterface:$depth,
                      TransformParamTypeInterface:$strides,
                      TransformParamTypeInterface:$dilations);
  let assemblyFormat =
    "$operand_handle attr-dict `:` functional-type(operands, results)";
  let extraClassDeclaration = SingleOpMatcher.extraDeclaration;
}

class StructuredDimDescription<string kind> {
  string description = !strconcat([{
     The following }], kind ,[{ specifications are supported:

      * `all`: all }], kind ,[{s are checked and captured;
      * list of integers: the listed }], kind, [{s are checked and captured;
      * `except(` list of integers `)`: all }], kind, [{s except the
        specified ones are checked and captured.

    Negative indexes are interpreted by counting values from the last one
    (similarly to Python). For example, `-1` means the last }], kind, [{ and
    `except(-1)` means all }], kind, [{s but the last. Indexes must be unique,
    including after interpretation of negative ones.

    Produces a silenceable failure in case of index overflow, including backward
    counting.
  }]);
}

def MatchStructuredDimOp : Op<Transform_Dialect, "match.structured.dim", [
    StructuredPredicate,
    SingleOpMatcher,
    MatchOpInterface,
    MemoryEffectsOpInterface]> {
  let summary =
      "Checks if the dimensions of the structured op satisfy some criteria";
  let description = !strconcat([{
    Checks if the dimensions (loop ranges) of the structured payload op satisfy
    the criteria specified as attributes. May capture the numeric value of the
    dimension into a parameter that it returns.

    }], 
    StructuredDimDescription<"dimension">.description,
    [{

    The following mutually exclusive conditions are available as unit
    attributes:

      * `parallel`: the dimension corresponds to a parallel loop;
      * `reduction`: the dimension corresponds to a reduction loop.

    If the result type is specified, associates the parameter with the (static)
    values of dimensions in the same order as listed and preserving the natural
    order for `all` and `except`. Specifically, if `-1, -2` are specified, the
    parameter will be associated with the value of the second-to-last dimension
    followed by the last dimension. If the dimension is dynamic, the parameter
    will contain a negative value corresponding to kDynamic in C++.
    
    }], StructuredPredicate.extraDescription, [{

    #### Return modes

    Succeeds if the specified dimensions satisfy the specified criteria,
    produces a silenceable failure otherwise. Produces a definite failure if
    the operand is not associated with a single payload op.
  }]);

  let arguments = (ins TransformHandleTypeInterface:$operand_handle,
                       DenseI64ArrayAttr:$raw_dim_list,
                       UnitAttr:$is_inverted,
                       UnitAttr:$is_all,
                       UnitAttr:$parallel,
                       UnitAttr:$reduction);

  let results = (outs Optional<TransformParamTypeInterface>:$result);
  let assemblyFormat =
      "$operand_handle `[`"
      "custom<TransformMatchDims>($raw_dim_list, $is_inverted, $is_all)"
      "`]` attr-dict `:` "
      "custom<SemiFunctionType>(type($operand_handle), type($result))";

  let extraClassDeclaration = SingleOpMatcher.extraDeclaration # [{
    ::mlir::DiagnosedSilenceableFailure getDimensionsFor(
        ::mlir::linalg::LinalgOp op,
        ::llvm::SmallVectorImpl<int64_t> &dims);
  }];

  let hasVerifier = 1;
}

def MatchStructuredElementalBitwidthOp
  : Op<Transform_Dialect, "match.structured.elemental_bitwidth", [
    SingleValueMatcher,
    MatchOpInterface,
    MemoryEffectsOpInterface]> {
  let summary =
    "Captures the bitwidth of the value's elemental type as a parameter";
  let description = !strconcat([{
    Produces a transform dialect parameter associated with the bitwidth of the
    elemental type of the payload value passed as the operand.}],

    StructuredPredicate.extraDescription, [{

    #### Return modes

    Succeeds if the operand is associated with exactly one payload value of
    `ShapedType`. Produces a silenceable failure otherwise.
  }]);
  let arguments = (ins TransformValueHandleTypeInterface:$operand_handle);
  let results = (outs TransformParamTypeInterface:$result);
  let assemblyFormat =
    "$operand_handle attr-dict `:` functional-type(operands, results)";
  let extraClassDeclaration = SingleValueMatcher.extraDeclaration;
}

class MatchStructuredOperandOp<string opname> : Op<Transform_Dialect, opname, [
    StructuredPredicate,
    SingleOpMatcher,
    MatchOpInterface,
    MemoryEffectsOpInterface]> {
      
  // TODO: consider an attribute controlling whether to fail or succeed on
  // out-of-bounds accesses.
  let arguments = (ins TransformHandleTypeInterface:$operand_handle,
                       DenseI64ArrayAttr:$raw_position_list,
                       UnitAttr:$is_inverted,
                       UnitAttr:$is_all,
                       UnitAttr:$permutation,
                       UnitAttr:$projected_permutation);

  // TODO: allow this to bind multiple inputs simultaneously after checking that
  // `transform.foreach` works well in matches.
  let results =
      (outs Optional<AnyTypeOf<[TransformAnyHandle,Transform_AffineMapParamType]>>:$result);
  let assemblyFormat =
      "$operand_handle `[`"
      "custom<TransformMatchDims>($raw_position_list, $is_inverted, $is_all)"
      "`]` attr-dict "
      "`:` custom<SemiFunctionType>(type($operand_handle), type($result))";

  let extraClassDeclaration = SingleOpMatcher.extraDeclaration # [{
    ::mlir::DiagnosedSilenceableFailure getPositionsFor(
        ::mlir::linalg::LinalgOp op,
        ::llvm::SmallVectorImpl<int64_t> &positions);
  }];

  let hasVerifier = 1;
}

def MatchStructuredInputOp : MatchStructuredOperandOp<"match.structured.input"> {
  let summary =
    "Captures input operand(s) of a structured operation";
  let description = !strconcat([{
    Produces a transform dialect value depending on the result type:
    
      - If the result type is a value handle, it will be associated with the input
        operand(s) of the payload operation associated with the operand handle.
      - If the result type is an operation handle, it will be associated with the
        operation defining the input operand(s) of the payload operation associated
        with the operand handle.
      - If the result type is an affine map parameter type, it will be associated
        with the indexing map that corresponds to the input operand(s) of the
        payload operation associated with the operand handle.

    For example, given the following operation:

    ```mlir
    %arg1 = some.op
    linalg.matmul ins(%arg1, %arg2 : ...) outs(%arg3 : ...)
    ```

    in case of a successful match for operand 0 this operation will return, for
    each of the respective cases above:

      - A handle to `%arg1` if the result is a value handle.
      - A handle to `some.op` if the result is an operation handle.
      - A parameter containing the LHS map of the matrix multiplication, i.e.
        `affine_map<(d0, d1, d2) -> (d0, d2)>` if the result is an affine
        map parameter.

    The match succeeds if the conditions specified as attributes succeed.

    }], 
    StructuredDimDescription<"input">.description,
    [{

    }],
    StructuredPredicate.extraDescription,
    [{

    #### Return modes

    Succeeds if all input indexes are in bounds, produces a silenceable failure
    otherwise. Additionally, when the result is an operation handle, produces a
    silenceable failure if the input specification defines more than one input
    or if the operand is not an operation result.
  }]);
}

def MatchStructuredInitOp : MatchStructuredOperandOp<"match.structured.init"> {
  let summary =
    "Captures init operand(s) of a structured operation";
  let description = !strconcat([{
    Produces a transform dialect value depending on the result type:
      - If the result type is a value handle, it will be associated with the init
        operand(s) of the payload operation associated with the operand handle.
      - If the result type is an operation handle, it will be associated with the
        operation defining the init operand(s) of the payload operation associated
        with the operand handle.
      - If the result type is an affine map parameter type, it will be associated
        with the indexing map that corresponds to the init operand(s) of the
        payload operation associated with the operand handle.

    For example, given the following operation:

    ```mlir
    %arg3 = linalg.fill
    linalg.matmul ins(%arg1, %arg2 : ...) outs(%arg3 : ...)
    ```

    in case of a successful match for init operand 0 this operation will return,
    for each of the respective cases above:

      - A handle to `%arg3` if the result is a value handle.
      - A handle to `linalg.fill` if the result is an operation handle.
      - A parameter containing the result map of the matrix multiplication, i.e.
        `affine_map<(d0, d1, d2) -> (d0, d1)>` if the result is an affine
        map parameter.

    The match succeeds if the conditions specified as attributes succeed.

    }], 
    StructuredDimDescription<"init">.description,
    [{

    }],
    StructuredPredicate.extraDescription,
    [{

    #### Return modes

    Succeeds if all init(outs) indexes are in bounds, produces a silenceable
    failure otherwise. Additionally, when the result is an operation handle,
    produces a silenceable failure if the init(outs) specification defines
    more than one init(outs) or if the operand is not an operation result.
  }]);
}


def MatchStructuredNumInputsOp
  : Op<Transform_Dialect, "match.structured.num_inputs", [
    StructuredPredicate,
    SingleOpMatcher,
    MatchOpInterface,
    MemoryEffectsOpInterface]> {
  let summary = "Captures the number of input operands of a structured "
                "operation as parameter";
  let description = !strconcat([{
    Produces a transform dialect parameter value associated with an integer
    attribute containing the number of input operands of the payload operation
    associated with the operand handle.

    }], StructuredPredicate.extraDescription, [{

    #### Return modes

    Succeeds if the operand is associated with exactly one structured payload
    operation. Produces a silenceable failure otherwise.
  }]);

  let arguments = (ins TransformHandleTypeInterface:$operand_handle);
  let results = (outs TransformParamTypeInterface:$result);
  let assemblyFormat =
    "$operand_handle attr-dict `:` functional-type(operands, results)";  
  let extraClassDeclaration = SingleOpMatcher.extraDeclaration;
}

def MatchStructuredNumInitsOp
  : Op<Transform_Dialect, "match.structured.num_inits", [
    StructuredPredicate,
    SingleOpMatcher,
    MatchOpInterface,
    MemoryEffectsOpInterface]> {
  let summary = "Captures the number of init(outs) operands of a structured"
                "operation as parameter";
  let description = !strconcat([{
    Produces a transform dialect parameter value associated with an integer
    attribute containing the number of init(outs) operands of the payload
    operation associated with the operand handle.

    }], StructuredPredicate.extraDescription, [{

    #### Return modes

    Succeeds if the operand is associated with exactly one structured payload
    operation. Produces a silenceable failure otherwise.
  }]);

  let arguments = (ins TransformHandleTypeInterface:$operand_handle);
  let results = (outs TransformParamTypeInterface:$result);
  let assemblyFormat =
    "$operand_handle attr-dict `:` functional-type(operands, results)";  
  let extraClassDeclaration = SingleOpMatcher.extraDeclaration;
}

def MatchStructuredRankOp : Op<Transform_Dialect, "match.structured.rank", [
    StructuredPredicate,
    SingleOpMatcher,
    MatchOpInterface,
    MemoryEffectsOpInterface]> {
  let summary = "Captures the rank of a structured operation as parameter";
  let description = !strconcat([{
    Produces a transform dialect parameter value associated with an integer
    attribute containing the rank of the structured payload operation associated
    with the operand handle.

    }], StructuredPredicate.extraDescription, [{

    #### Return modes

    Succeeds if the operand is associated with exactly one structured payload
    operation. Produces a silenceable failure otherwise.
  }]);

  let arguments = (ins TransformHandleTypeInterface:$operand_handle);
  let results = (outs TransformParamTypeInterface:$rank);
  let assemblyFormat =
      "$operand_handle attr-dict `:`"
      "custom<SemiFunctionType>(type($operand_handle), type($rank))";

  let extraClassDeclaration = SingleOpMatcher.extraDeclaration;
}

def MatchStructuredResultOp : Op<Transform_Dialect, "match.structured.result", [
    StructuredPredicate,
    SingleOpMatcher,
    MatchOpInterface,
    MemoryEffectsOpInterface]> {
  let summary = "Captures the result of a structured payload operation in an "
                "op or value handle";
  let description = !strconcat([{
    Produces a transform dialect value handle associated with the payload value
    defined as a result of the payload operation associated with the operand
    handle, or an operation handle to an operation using the produced result
    with additional constraints specified by the attributes as follows.

      * If `any` is specified, binds the resulting handle to any operation using
        the result and succeeds.
      * If `single` is specified, binds the resulting handle to the only
        operation using the result or fails if there is more than one (or no)
        such operation.

    The number of the result is specified as `position` attribute. It may take
    positive and negative values. Negative values are interpreted as counting
    results from backwards, e.g., `-1` means the last result and `-2` means the
    second-to-last result. In any case, the position must be in bounds for the
    given payload operation. A silenceable failure is produced for out-of-bounds
    positions.

  }], StructuredPredicate.extraDescription, [{

    #### Return modes

    Succeeds if the position is in bounds and if the user operation could be
    found when requested. Produces a silenceable failure otherwise.
  }]);
  let arguments = (ins TransformHandleTypeInterface:$operand_handle,
                       I64Attr:$position,
                       UnitAttr:$any,
                       UnitAttr:$single);
  let results = (outs TransformAnyHandle:$result);
  let assemblyFormat = 
      "$operand_handle `[` $position `]` (`any` $any^)? (`single` $single^)?"
      "attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;

  let extraClassDeclaration = SingleOpMatcher.extraDeclaration # [{
    ::mlir::DiagnosedSilenceableFailure
    getPositionFor(::mlir::linalg::LinalgOp op, int64_t &position);
  }];
}

def MatchStructuredYieldOp : Op<Transform_Dialect, "match.structured.yield", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    Terminator]> {
  let summary = "Terminator for transform.match.structured blocks";
  let description = [{
    Forwards the payload association from the operands to the results of the
    parent op. Always succeeds.
  }];
  let builders = [
    OpBuilder<(ins)>
  ];

  let arguments = (ins Variadic<Transform_AnyHandleOrParamType>:$handles);
  let assemblyFormat = "$handles attr-dict (`:` type($handles)^)?";
}

#endif // LINALG_MATCH_OPS
