# Chapter 4: Matching Payload with Transform Operations

**Check the continuously-tested version of MLIR files under
[mlir/test/Examples/transform/Ch4](https://github.com/llvm/llvm-project/tree/main/mlir/test/Examples/transform/Ch4).**

Up until now, we were applying transform dialect scripts under the assumption
that specific payload operations are identified by the caller when the transform
dialect interpreter is invoked. This may be seen as contrary to the idea of
driving transformations from a dialect since the transformation targets must be
identified through mechanisms external to the transform dialect interpreter, for
example, when invoking the interpreter programmatically in C++ or through pass
arguments as seen in previous chapters. It also adds practical overhead due to
increased interaction with the interpreter in C++, and cognitive overhead of
manipulating two interfaces at once. To remedy this, Transform dialect proposes
a subset of operations for _matching_ payload operations that need to be
transformed.

_Match_ operations are simply transform operations with some additional
guarantees. In particular, they are not expected to modify the payload IR and
are expected to fail if their operands (typically payload operation handles) are
not associated with payload IR objects having desired properties, such as
operation names or kinds of arguments. Using simple combinator operations, it
becomes possible to set up a higher-level match and rewrite infrastructure
directly within the transform dialect.


## Simple match

Let us reconsider the “fully connected layer” example from [Chapter
1](Ch1.md#chaining-transformations-with-handles), reproduced below for
convenience.


```mlir
// Original function to optimize.
func.func @fc_relu(%lhs: tensor<512x512xf32>, %rhs: tensor<512x512xf32>,
                   %bias: tensor<512x512xf32>, %output: tensor<512x512xf32>)
                   -> tensor<512x512xf32> {
  // Matrix-matrix multiplication.
  %matmul = linalg.matmul
            ins(%lhs, %rhs: tensor<512x512xf32>, tensor<512x512xf32>)
            outs(%output: tensor<512x512xf32>) -> tensor<512x512xf32>

  // Elementwise addition.
  %biased = linalg.elemwise_binary { fun = #linalg.binary_fn<add> }
    ins(%matmul, %bias : tensor<512x512xf32>, tensor<512x512xf32>)
    outs(%output : tensor<512x512xf32>) -> tensor<512x512xf32>

  // Elementwise max with 0 (ReLU).
  %c0f = arith.constant 0.0 : f32
  %relued = linalg.elemwise_binary { fun = #linalg.binary_fn<max_signed> }
    ins(%biased, %c0f : tensor<512x512xf32>, f32)
    outs(%output : tensor<512x512xf32>) -> tensor<512x512xf32>
  func.return %relued : tensor<512x512xf32>
}

```


In Chapter 1, we were calling the test transform interpreter pass with
additional arguments, `bind-first-extra-to-ops=linalg.matmul
bind-second-extra-to-ops=linalg.elemwise_binary`, to provide initial
associations for operation handles. Instead, we can use match operations to
discover relevant operations in the payload IR. Match operations can be combined
with “regular” transform operations using, e.g., the
`transform.collect_matching` combinator operation that leverages the concept of
named sequences to organize matchers.


```mlir
// The module containing named sequences must have an attribute allowing them
// to enable verification.
module @transforms attributes { transform.with_named_sequence } {
  // Entry point. This takes as the only argument the root operation (typically
  // pass root) given to the transform interpreter.
  transform.named_sequence @__transform_main(
      %root: !transform.any_op {transform.readonly}) {
    // Collect operations that match the criteria specified in named sequence.
    // If the named sequence fails with a silenceable failure, silences it (the
    // message is forwarded to the debug stream). If the named sequence
    // succeeds, appends its results to the results of this operation.
    %elemwise = transform.collect_matching @match_elemwise in %root
      : (!transform.any_op) -> !transform.any_op
    %matmul = transform.collect_matching @match_matmul in %root
      : (!transform.any_op) -> !transform.any_op
    transform.include @print_elemwise failures(propagate)  (%elemwise)
      : (!transform.any_op) -> ()
    transform.include @print_matmul failures(propagate)  (%matmul)
      : (!transform.any_op) -> ()

    transform.yield
  }

  // This is a matcher sequence. It is given an operation to match and the
  // match is considered successful unless any nested operation produces a
  // failure. The values yielded by this operation will be forwarded to the
  // rewriter sequence on success.
  transform.named_sequence @match_elemwise(
      %entry: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %entry ["linalg.elemwise_binary"]
      : !transform.any_op
    transform.yield %entry : !transform.any_op
  }
  transform.named_sequence @match_matmul(
      %entry: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %entry ["linalg.matmul"] : !transform.any_op
    transform.yield %entry : !transform.any_op
  }

  // This is a rewriter sequence.
  transform.named_sequence @print_elemwise(
      %elemwise_binary: !transform.any_op {transform.readonly}) {
    transform.test_print_remark_at_operand
      %elemwise_binary, "elementwise binary" : !transform.any_op
    transform.yield
  }
  transform.named_sequence @print_matmul(
      %matmul: !transform.any_op {transform.readonly}) {
    transform.test_print_remark_at_operand %matmul, "matmul" : !transform.any_op
    transform.yield
  }
}

```


This script can be executed using the non-test interpreter pass running on the
root operation of the translation unit without additional flags: `mlir-opt
--transform-interpreter`. It will emit corresponding remarks at
`linalg.elemwise_binary` and `linalg.matmul` operations. In debug builds, the
infrastructure provides a convenient method to understand the matching process
by passing `-debug-only=transform-matcher` to `mlir-opt` or a derived tool. It
will print the silenceable failure messages produced by the match operations
into the debug stream, for example:


```
<...>
[transform-matcher] matching %0 = linalg.matmul ins(%arg0, %arg1 : tensor<512x512xf32>, tensor<512x512xf32>) outs(%arg3 : tensor<512x512xf32>) -> tensor<512x512xf32> @0x5622eee08410
[transform-matcher] matcher match_elemwise failed: wrong operation name
<...>
```


This is now sufficient to run the rest of the transform script from Chapter 1,
substituting `%arg1` with `%matmul` and `%arg2` with `%elemwise`.


## Matching Chains of Operations

The matcher above remains naive as it matches _all_ operations of the certain
kind under the payload root. These operations may or may not be related, and
may, for example, belong to different functions. Even if they are in a single
function, if there are multiple groups of such operations, we wouldn’t be able
to differentiate them with this approach. In reality, we want to match a
specific group of operations where a `matmul` operation produces a result that
is used by an elementwise operation, which in turn feeds another elementwise
operation in a similar way.

This can be achieved using the following matcher sequence.


```mlir
// This is also a matcher sequence. It is similarly given an operation to
// match and nested operations must succeed in order for a match to be deemed
// successful. It starts matching from the last operation in the use-def chain
// and goes back because each operand (use) has exactly one definition.
transform.named_sequence @match_matmul_elemwise(
    %last: !transform.any_op {transform.readonly})
    -> (!transform.any_op, !transform.any_op, !transform.any_op) {
  // The last operation must be an elementwise binary.
  transform.match.operation_name %last ["linalg.elemwise_binary"]
    : !transform.any_op
  // Its first operand must be defined by another operation, to which we
  // will get a handle here. We are guaranteed that the first operand exists
  // because we know the operation is binary, but even in absence of such a
  // guarantee, this operation would have produced a silenceable failure when
  // `%last` does not have enough operands.
  %middle = transform.get_producer_of_operand %last[0]
    : (!transform.any_op) -> !transform.any_op
  // The defining operation must itself be an elementwise binary.
  transform.match.operation_name %middle ["linalg.elemwise_binary"]
    : !transform.any_op
  // And the first operand of that operation must be defined by yet another
  // operation.
  %matmul = transform.get_producer_of_operand %middle[0]
    : (!transform.any_op) -> !transform.any_op
  // And that operation is a matmul.
  transform.match.operation_name %matmul ["linalg.matmul"] : !transform.any_op
  // We will yield the handles to the matmul and the two elementwise
  // operations separately.
  transform.yield %matmul, %middle, %last
    : !transform.any_op, !transform.any_op, !transform.any_op
}
```

This matcher is applicable in presence of other `elemwise` and `matmul`
operations and will return the triple of _related_ operations rather than
operations in the order in which they are found. It can be exercised similarly
to the previous incarnation, as follows.

```mlir
// Alternative entry point.
transform.named_sequence @__transform_main(
    %root: !transform.any_op {transform.readonly}) {
  // Collect groups of operations that match the criteria specified in the
  // named sequence.
  %matmul, %el1, %el2 = transform.collect_matching @match_matmul_elemwise in %root 
    : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
  %elemwise = transform.merge_handles %el1, %el2 : !transform.any_op

  transform.include @print_elemwise failures(propagate)  (%elemwise)
    : (!transform.any_op) -> ()
  transform.include @print_matmul failures(propagate)  (%matmul)
    : (!transform.any_op) -> ()

  transform.yield
}
```


## Defining Match Operations

The matcher of a chain of operations is correct in presence of other operations,
but is still insufficiently robust for many cases of interest. In particular,
using `transform.get_producer_of_operand %last[0]` requires that the _first_
operand of elementwise operations is produced by another operation. The same
transformation strategy may however apply regardless of the operand position:
many binary operations are associative. Let us use this opportunity to introduce
a new match operation. Specifically, we would like this operation to succeed if
_any_ of the operands satisfies certain conditions that can be expressed as
other match operations. We also want it to return some of the state and the
position of the matched operand in the operand list.

Match operations are defined similarly to other transform operations, with the
only difference of additionally implementing the `MatchOpInterface`. Note that
this interface has _no additional methods_ (though it may add some eventually)
and is only used as a verification contract that the operation is intended for
matching and will not attempt to transform the payload. The minimal definition
of our operation is as follows.


```tablegen
// Define the new operation. By convention, prefix its name with `match`
// followed by the name of the dialect extension.
def HasOperandSatisfyingOp : TransformDialectOp<"match.my.has_operand_satisfying",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     DeclareOpInterfaceMethods<TransformOpInterface>,
     // Indicate that the operation implements MatchOpInterface in addition to
     // the TransformOpInterface. This interface is only used as a tag at this
     // point and has no methods that are mandatory to implement.
     MatchOpInterface,
     SingleBlockImplicitTerminator<"::mlir::transform::YieldOp">]> {
  let summary = "Succeed if any of the operands matches all nested criteria";
  let arguments = (ins TransformHandleTypeInterface:$op);
  let results = (outs TransformParamTypeInterface:$position,
                      Variadic<Transform_AnyHandleOrParamType>:$results);

  // Match operations can be arbitrarily complex, e.g., containing regions.
  let regions = (region SizedRegion<1>:$body);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $op `:` functional-type($op, results) attr-dict-with-keyword $body
  }];
}
```


It takes as argument the handle associated with the payload operations whose
operands it will match, has an associated single-block region containing the
match criteria, and returns the position of the matched operand as well as any
other transform value yielded from the body on the successful match.

The matching logic is implemented in the `apply` method of the
`TransformOpInterface` and is easily composable with other transform operations.
All facilities for managing the interpreter state and recursively entering the
blocks are available in the same way as they are for “regular” transform
operations. Match operations are expected to return a silenceable failure to
indicate failure to match, and to immediately propagate definite failures. If
they have nested operations, they are expected to handle and, in most cases,
silence the silenceable failures produced when applying those operations. For
our operation, the matching is essentially a loop iterating over all operands of
the (single) payload operation and applying nested transform ops until they all
succeed for one of the operands.


```cpp
// Matcher ops implement `apply` similarly to other transform ops. They are not
// expected to modify payload, but use the tri-state result to signal failure or
// success to match, as well as potential irrecoverable errors.
mlir::DiagnosedSilenceableFailure
mlir::transform::HasOperandSatisfyingOp::apply(
    mlir::transform::TransformRewriter &rewriter,
    mlir::transform::TransformResults &results,
    mlir::transform::TransformState &state) {
  // For simplicity, only handle a single payload op. Actual implementations
  // can use `SingleOpMatcher` trait to simplify implementation and document
  // this expectation.
  auto payloadOps = state.getPayloadOps(getOp());
  if (!llvm::hasSingleElement(payloadOps))
    return emitSilenceableError() << "expected single payload";

  // Iterate over all operands of the payload op to see if they can be matched
  // using the body of this op.
  Operation *payload = *payloadOps.begin();
  for (OpOperand &operand : payload->getOpOperands()) {
    // Create a scope for transform values defined in the body. This corresponds
    // to the syntactic scope of the region attached to this op. Any values
    // associated with payloads from now on will be automatically dissociated
    // when this object is destroyed, i.e. at the end of the iteration.
    // Associate the block argument handle with the operand.
    auto matchScope = state.make_region_scope(getBody());
    if (failed(state.mapBlockArgument(getBody().getArgument(0),
                                      {operand.get()}))) {
      return DiagnosedSilenceableFailure::definiteFailure();
    }

    // Iterate over all nested matchers with the current mapping and see if they
    // succeed.
    bool matchSucceeded = true;
    for (Operation &matcher : getBody().front().without_terminator()) {
      // Matcher ops are applied similarly to any other transform op.
      DiagnosedSilenceableFailure diag =
          state.applyTransform(cast<TransformOpInterface>(matcher));

      // Definite failures are immediately propagated as they are irrecoverable.
      if (diag.isDefiniteFailure())
        return diag;

      // On success, keep checking the remaining conditions.
      if (diag.succeeded())
        continue;

      // Report failure-to-match for debugging purposes and stop matching this
      // operand.
      assert(diag.isSilenceableFailure());
      DEBUG_MATCHER(DBGS_MATCHER()
                    << "failed to match operand #" << operand.getOperandNumber()
                    << ": " << diag.getMessage());
      (void)diag.silence();
      matchSucceeded = false;
      break;
    }
    // If failed to match this operand, try other operands.
    if (!matchSucceeded)
      continue;

    // If we reached this point, the matching succeeded for the current operand.
    // Remap the values associated with terminator operands to be associated
    // with op results, and also map the parameter result to the operand's
    // position. Note that it is safe to do here despite the end of the scope
    // as `results` are integrated into `state` by the interpreter after `apply`
    // returns rather than immediately.
    SmallVector<SmallVector<MappedValue>> yieldedMappings;
    transform::detail::prepareValueMappings(
        yieldedMappings, getBody().front().getTerminator()->getOperands(),
        state);
    results.setParams(getPosition().cast<OpResult>(),
                      {rewriter.getI32IntegerAttr(operand.getOperandNumber())});
    for (auto &&[result, mapping] : llvm::zip(getResults(), yieldedMappings))
      results.setMappedValues(result, mapping);
    return DiagnosedSilenceableFailure::success();
  }

  // If we reached this point, none of the operands succeeded the match.
  return emitSilenceableError()
         << "none of the operands satisfied the conditions";
}

```


By convention, operations implementing `MatchOpInterface` must not modify
payload IR and must therefore specify that they only read operand handles and
payload as their effects.


```cpp
void transform::CollectMatchingOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getRoot(), effects);
  producesHandle(getResults(), effects);
  onlyReadsPayload(effects);
}
```


This operation can now be included in a transform dialect extension, loaded and
used in our matcher. Specifically, we will use it to indicate that either of the
operands of the “max” elementwise operation in our example can be produced by
the previous elementwise operation. The previous operation will still require
the matmul to produce the first operand for simplicity. The updated matcher
sequence looks as follows.


```mlir
transform.named_sequence @match_matmul_elemwise(
    %last: !transform.any_op {transform.readonly})
    -> (!transform.any_op, !transform.any_op, !transform.any_op,
        !transform.param<i32>) {
  // The last operation must be an elementwise binary.
  transform.match.operation_name %last ["linalg.elemwise_binary"]
    : !transform.any_op

  // One of its operands must be defined by another operation, to which we
  // will get a handle here. This is achieved thanks to a newly defined
  // operation that tries to match operands one by one using the match
  // operations nested in its region.
  %pos, %middle = transform.match.my.has_operand_satisfying %last
      : (!transform.any_op) -> (!transform.param<i32>, !transform.any_op) {
  ^bb0(%operand: !transform.any_value):
    // The operand must be defined by an operation.
    %def = transform.get_defining_op %operand
      : (!transform.any_value) -> !transform.any_op
    // The defining operation must itself be an elementwise binary.
    transform.match.operation_name %def ["linalg.elemwise_binary"]
      : !transform.any_op
    transform.yield %def : !transform.any_op
  }

  // And the first operand of that operation must be defined by yet another
  // operation.
  %matmul = transform.get_producer_of_operand %middle[0]
    : (!transform.any_op) -> !transform.any_op
  // And that operation is a matmul.
  transform.match.operation_name %matmul ["linalg.matmul"] : !transform.any_op
  // We will yield the handles to the matmul and the two elementwise
  // operations separately.
  transform.yield %matmul, %middle, %last, %pos
    : !transform.any_op, !transform.any_op, !transform.any_op,
      !transform.param<i32>
}
```


This achieves the desired effect and matches both `max(add(matmul(...), bias),
0)` and `max(0, add(matmul(...), bias))` in the same values. The `%pos` value is
a transform dialect _parameter_, which is used to store lists of entities known
to be constant throughout the transform application. Most often, parameters are
numeric values, but they can generally be any MLIR attributes.

In order to demonstrate that groups of operations are matched independently of
each other, let us use the `transform.foreach_match` operation that allows one
to implement a simple high-level pattern rewriting approach within the transform
dialect (for advanced or lower-level pattern rewriting, consider PDL(L) or C++
rewriting APIs). It maps a matcher named sequence to an action named sequence,
and the latter gets invoked whenever the former succeeds.


```mlir
// Traverses the payload IR associated with the operand handle, invoking
// @match_matmul_elemwise on each of the operations. If the named sequence
// succeeds, i.e., if none of the nested match (transform) operations
// produced a silenceable failure, invokes @print_matmul_elemwise and
// forwards the values yielded as arguments of the new invocation. If the
// named sequence fails with a silenceable failure, silences it (the message
// is forwarded to the debug stream). Definite failures are propagated
// immediately and unconditionally, as usual.
transform.foreach_match in %root
  @match_matmul_elemwise -> @print_matmul_elemwise
  : (!transform.any_op) -> !transform.any_op
```


The `@print_matmul_elemwise` named sequence, available in `multiple.mlir`, will
use the parameter with the position of the operand to differentiate the two
groups.


## Matchers for Inferred Features

The matcher sequences described above, although useful to drive transformations
from within the transform dialect interpreter, are rather basic since they
mostly rely on operation names and use-def chains. Alternative implementations
using APIs or various declarative rewrite rules are barely less expressive and
sometimes more concise. The real power of transform dialect matcher ops lies in
the possibility to define matchers of _inferred properties_ of payloads, i.e.,
properties that are not directly accessible as an attribute of an operation or
any straightforward relation between IR components.

The utility of such matchers can be easily demonstrated by slightly modifying
our original example. If matrix multiplication is expressed as a special case of
tensor contraction using `linalg.generic` instead of `linalg.matmul`, the
operation name-based matcher no longer applies. Yet such a representation is
very common and can appear both in the original input and during the course of
transformation, e.g., where a higher-dimensional contraction is decomposed into
loops around a matrix multiplication.

In order to be a (potentially transposed) matrix multiplication, the
`linalg.generic` operation must have the following features:



*   Total rank of 3.
*   Two inputs accessed as projected permutation of iteration dimensions.
*   One output accessed as projected permutation of iteration dimensions.
*   Iteration dimensions can be subdivided into LHS parallel, RHS parallel and reduction dimensions.
*   The body block consists of a multiplication and an addition.

Most of these features can be derived from the properties of the operation,
e.g., the total rank corresponds to the number of entries in the `iterators`
attribute, but almost none of them are immediately accessible in the IR or in
any declarative form, which is usually limited to checking the presence or the
exact match of an attribute or a type.  The transform dialect allows these
features to be implemented in the `apply` method of a matcher op and reused
across multiple matching cases. For structured linear algebra payload
operations, many such match operations are readily available in the `structured`
extension. They are sufficient to implement a matrix multiplication matcher
using the features listed above almost verbatim.


```mlir
transform.named_sequence @match_generic_matmul(
    %candidate: !transform.any_op {transform.readonly}) -> !transform.any_op {
  // Match a structured linear algebra operation.
  transform.match.structured %candidate : !transform.any_op {
  ^bb0(%c: !transform.any_op):
    // With a rank equal to 3.
    %rank = transform.match.structured.rank %c
      : (!transform.any_op) -> !transform.param<i64>
    %c3 = transform.param.constant 3 : i64 -> !transform.param<i64>
    transform.match.param.cmpi eq %rank, %c3 : !transform.param<i64>

    // With 2 inputs.
    %n_ins = transform.match.structured.num_inputs %c
      : (!transform.any_op) -> !transform.param<i64>
    %c2 = transform.param.constant 2 : i64 -> !transform.param<i64>
    transform.match.param.cmpi eq %n_ins, %c2 : !transform.param<i64>

    // With 1 output (note that structured ops in destination passing style
    // has as many inits as outputs).
    %n_inits = transform.match.structured.num_inits %c
      : (!transform.any_op) -> !transform.param<i64>
    %c1 = transform.param.constant 1 : i64 -> !transform.param<i64>
    transform.match.param.cmpi eq %n_inits, %c1 : !transform.param<i64>

    // All inputs and inits are accessed with a projected permutation.
    transform.match.structured.input %c[all] {projected_permutation}
      : !transform.any_op
    transform.match.structured.init %c[0] {projected_permutation}
      : !transform.any_op

    // The body is a mulf/addf contraction with appropriate dimensions.
    transform.match.structured.body %c
      { contraction = ["arith.mulf", "arith.addf"] } : !transform.any_op
    %batch, %lhs, %rhs, %reduction =
    transform.match.structured.classify_contraction_dims %c
      : (!transform.any_op)
      -> (!transform.param<i64>, !transform.param<i64>, !transform.param<i64>,
          !transform.param<i64>)


    // There is one of lhs, rhs and reduction dimensions and zero batch
    // dimensions.
    %n_batch = transform.num_associations %batch
      : (!transform.param<i64>) -> !transform.param<i64>
    %n_lhs = transform.num_associations %lhs
      : (!transform.param<i64>) -> !transform.param<i64>
    %n_rhs = transform.num_associations %rhs
      : (!transform.param<i64>) -> !transform.param<i64>
    %n_reduction = transform.num_associations %reduction
      : (!transform.param<i64>) -> !transform.param<i64>
    %c0 = transform.param.constant 0 : i64 -> !transform.param<i64>
    transform.match.param.cmpi eq %n_batch, %c0 : !transform.param<i64>
    transform.match.param.cmpi eq %n_lhs, %c1 : !transform.param<i64>
    transform.match.param.cmpi eq %n_rhs, %c1 : !transform.param<i64>
    transform.match.param.cmpi eq %n_reduction, %c1 : !transform.param<i64>
  }
  transform.yield %candidate : !transform.any_op
}
```


While this example leverages the contraction-specific matchers that have a
rather non-trivial C++ implementation, the transform dialect is sufficiently
flexible to implement this reasoning directly if desired. One could, for
example, obtain the access map of each input as a parameter and extract the
accessed dimensions as other parameters that can be compared with each other to
ensure the subscripts are `m,k` for LHS, `k,n` for RHS and `m,n` for the
init/result given the `m,n,k` notation for loops.

