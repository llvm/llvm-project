# Shape Inference

Shape inference as discussed here is considered a specific instance of type
inference for [ShapedType][ShapedType]. Type constraints are along (at least)
three axis: 1) elemental type, 2) rank (including static or dynamic), 3)
dimensions. While some operations have no compile time fixed shape (e.g., output
shape is dictated by data) we could still have some knowledge of
constraints/bounds in the system for that operation (e.g., the output of a
`tf.where` is at most the size of the input data). That is, there are additional
valuable constraints that could be captured even without full knowledge of the
shape.

Type inference is currently modelled executionally for operation creation using the
[`InferTypeOpInterface`][InferTypeOpInterface], while
`InferShapedTypeOpInterface` is used to implement the shape and element type
inference. The return type can often be deduced from the deduced return shape
and elemental type (queryable from `InferShapedTypeOpInterface`) and so type
inference for tensor types can be implemented with `InferShapedTypeOpInterface`.

## Shape functions

The C++ interfaces are the base mechanism whereby shape inference is queried and
executed, but not the intended way to specify shape constraints in general.

Initially the shape inference will be declaratively specified using:

*   Constraints on the operands of an operation directly. For example
    constraining the input type to be tensor/vector elements or that the
    elemental type be of a specific type (e.g., output of computing the size
    of a value is of elemental type `i1`) or class (e.g., float-like).
*   Constraints across operands and results of an operation.

    - For example, specifying equality constraints on type/constituents of a
      type (shape and elemental type) between operands and results (e.g., the
      output type of an add is the same as those of the input operands).

NOTE: The C++ shape functions are an intermediate step until the shape dialect
is more full-fledged, at which point the C++ functions should become the
exceptional case.

## Testing

Shape inference is currently tested alongside type inference by
`TestReturnTypeDriver` in the test dialect. This driver performs two checks:

1.  Verification that the return types specified matches the inferred types. This
    explicit check will be removed and made part of Op verification instead.
2.  Test the creation of Ops without specifying the return type explicitly in
    function `testCreateFunctions` by creating new binary Ops (Op classes
    specified in `TestReturnTypeDriver`) using 1) all operands to
    `testCreateFunctions` as both operands, and 2) using combinations of input
    operands of the function.

## Shape dialect

This section details the shape type inference dialect (`shape`). The initial
focus will be on shape functions that describe shape functions could be used in
runtime and compiler (for constructions of ops/refinement of shapes, reification
of dynamic allocations for dialect including TF, TFLite, XLA & tensor compute
dialect under discussion).

This will focus on the shape functions (e.g., determine the rank and dimensions
of the output shape). As shown in the shaped container type, shape will be one
of 3 components, the others being elemental type and attribute (which is
currently left open with the intention of supporting extensions such as layouts
or bounded shapes at a later point). This allows for decoupling of these:

*   Not all the information is needed for all analysis;
*   Not all shape functions need to provide all the information (e.g., one could
    define a base class function that only populates element type but composes
    with the others);
*   It allows reusing the constraints between, say, Tensor and Memref
    representation of an operation;
    
An argument could be made that these are metadata function instead of shape
functions, with some considering shape and elemental types different and some considering them both as
part of shape. But `shape function` is IMHO descriptive and metadata can span
too large a range of potential uses/values.

### Requirements

The requirements for the shape inference functions are determined by the
requirements of shape inference, but we believe the requirements below still
allow freedom to consider different shape inference approaches and so we do not
impose a particular shape inference approach here.

#### Shape inference functions

*   **Expressiveness** shape functions need to support programs where tensors
    have shapes that are not known statically (for example, `tensor<16x?xf32>`
    or `tensor<*xf32>*`);
*   **Shape error detection** Many operations will have constraints on their
    operands. If the constraints are not satisfied or cannot be determined if
    satisfied statically, then a runtime check/assertion could be generated.

    *   This also aligns with the requirement that the shape function description
        should be usable by both the compiler and runtime.
    *   Shape error functions should be easy to understand, at least what
        constraint of the operation is violated. This also requires that shape
        function error messages should be configurable by the author of the
        shape function (e.g., the author would be able to give the semantic
        constraint invalidated rather the low-level check that failed).
    *   The static analysis may be used to eliminate run-time checks that are
        guaranteed to pass.
        *   Ideally all would eventually (see section
            [Inlining shape checking](#inline)) be elided.
    *   Only reporting errors which are guaranteed to occur at runtime. If an error is only
        possible (rather than guaranteed) then we use a runtime assertion to fail and produce an error
        message with the invariant violated.

*   Shape functions usable by compiler and runtime.

    *   This does not mean the exact same C++ function, but rather the
        description should be consumable by either.
    *   Shape function description should not be constrained by either runtime
        or compiler's type system to handle types only used for analysis. That
        is, these two type systems differ and both should be supported, but the
        intersection of the two should not be required. As a particular example,
        if a compiler only wants to differentiate exact shapes vs dynamic
        shapes, then it need not consider a more generic shape lattice even
        though the shape description supports it.

*   Declarative (e.g., analyzable at compile time, possible to generate
    different versions for different use cases)

    *   This may not strictly be a requirement, but a way to handle the former:
        a declarative specification could be reused by both while avoiding a
        need to map to or from a 3rd representation given these two systems
        have/and will have different types.

*   Shape inference functions are expressible at runtime

    *   User can define a shape function for a new operation dynamically at runtime,
        this allows for vendors to describe an operation and shape function
        dynamically.

        This requirement is on the wishlist.

*   Doesn't require graph-wide shape information (e.g., only require local
    information)

    *   Shape functions should be cheap to invoke on each kernel launch.
    *   Shape function can be dictated by arguments (operands, attributes and regions)
        only (e.g., same operands as the corresponding operation could be
        constructed & invoked with).
    *   Shape information that needs higher-level/graph information should use
        richer types (e.g., `TensorList<F32>`);
    *   The function should be invocable before/while constructing an op (e.g.,
        can't rely on the op being constructed).

*   Shape functions should be pure functions.

*   Should support functions whose type is only known dynamically (e.g.,
    `read_from_file` op)

    *   Without needing to invoke the op (e.g., reading a file once for
        determining the shape & then post to be able to actually consume the
        output of the file).

*   The shape function operation dialect should be interoperable with non-shape function dialect operations.

    *   There may be a common set of operations that satisfy most uses (e.g., merge,
        equal_type, arithmetic expressions, slice, concat, pattern matching on
        attributes such as padding etc.) that will be discovered and could cover
        a large percentage of the use cases. Among these there will be some
        which carry extra semantic info that could be used for symbolic
        constraints (e.g., checking equality of two dimensions resulting in
        setting an equality constraint) and higher-order interpretation for
        constraint solving.

        It is therefore beneficial (but not required) to reuse operations, 
        especially as for statically known shapes, arbitrary arithmetic
        computations could still be performed. This means that the computations
        performed statically may or may not be supported by an arbitrary solver,
        but would still be allowed.

*   The shape function should be expandable such that symbolic equality and
    upper bound constraints (say) could be represented and may be propagated by
    shape inference.

    *   E.g., the shape functions may contain more information that is only
        useful when used from shape inference;

*   Shape functions are allowed to fail and report an error. The error reporting
    should report the location of the operation that failed with, where
    possible, a user actionable error message.

    *   These failures could become inlined and become runtime failures with
        runtime values and error messages.
    *   Reporting errors should be optional. E.g., The same function
        may be used as to query validity without reporting an error.

#### Non-goals

1.  The shape dialect is an IR representations and not a programming language;
    *   While the functions should be readable, it doesn't carry the
        conveniences of a programming language. Deciding how people write these
        things, e.g. a mini dsl, a C++ API that generates them, extracting them
        programmatically from `SetShapeFn` calls, etc., is still TBD.
1.  Describe the shape inference approach that will use the shape functions;
    *   The goal is that the shape functions and the constraints one could
        obtain from them are general enough that they would be useful for
        various analysis. But whether we follow very simple (e.g., only fully
        static information is used for shape output, unranked for everything
        else) to very advance (e.g., expression trees of symbolic constants) can
        be evaluated independently of this proposal and with concrete benefit
        analysis.
1.  Describe the approach whereby error messages will be generated;
    *   While the shape functions will be able to emit errors optionally, it
        will be possible to dictate when they emit an error. This enables
        deciding whether or which error to emit: there have been proposals in
        the literature that the iteration order for shape inference affect the
        quality of the error message produced, and the shape functions do not
        mandate that.
1.  Flow sensitive shape functions;
    *   To enable scalable/cheap shape inference, the shape functions do not
        intend to provide flow sensitive information. This facility could
        potentially be built as part of shome higher order analysis that reuse
        the shape functions/constraints due to the shape functions.
1.  All static functions are usable for dynamic/unknown shapes;
    *   More involved computations can be performed with statically known shapes
        than what can be sensibly analyzed with unknown/symbolic variables.

### Discussion

#### Inline shape inference checks {#inline}

Shape functions should be lowerable to runtime checks for validity. E.g. verify
as much as possible statically, but enable generating instructions to compute the
shape dynamically and or falling back to runtime checks for attributes not
verifiable at compile time. These checks inserted should ideally only check that
which could not have been verified statically.

These inlined calls could interfere with optimization patterns/passes (e.g.,
shape inference should not insert constructs that interfere with optimization
patterns) and so could be delayed until later (with another round of
optimizations, constant folding, CSE, etc., that should remove redundant runtime
operations).

### Possibly Asked Questions

#### What about ODS specifications of operations?

In ODS we have been recording the constraints for the operands & attributes of
an operation. Where these are sufficient to constrain the output shape (e.g.,
`SameOperandAndResultType` or broadcastable) we should generate the shape
function from those. Where not, an explicit shape function should be specified
(spelling TBD but currently considering using the MLIR textual form as
serialization approach).

#### Why not extract the shape function from reference implementation?

This could be done in future! The extracted shape function would use the shape
inference dialect, so we are starting there. Especially for operations described in a
structured way, one could autogenerate the shape function.

#### How/in what language will the shape functions be authored?

TBD. open to many approaches and suggestions, starting on the IR produced by
whatever language is the priority of this proposal.

#### What shape inference approach is being suggested here?

None. There are multiple different shape inference approaches that we could
layer on top of these. From the most basic (always return unranked), to more
useful (return fixed shape for constant inputs/arguments) to the more advanced
(create logical conjuctions of algebraic statements between symbolic named
values).

### Open points

1.  Should shape functions that produce dynamic outputs given all statically
    shaped inputs be marked specially? E.g., read from file.

TODO: Add examples here.

## WIP/Future considerations

Shape functions are determined by attributes and could be arbitrarily
complicated with a wide-range of specification possibilities. Equality
relationships are common (e.g., the elemental type of the output matches the
primitive type of the inputs, both inputs have exactly the same type [primitive
type and shape]) and so these should be easy to specify. Algebraic relationships
would also be common (e.g., a concat of `[n,m]` and `[n,m]` matrix along axis 0
is `[n+n, m]` matrix), while some ops only have defined shapes under certain
cases (e.g., matrix multiplication of `[a,b]` and `[c,d]` is only defined if `b
== c`).

Instead of specifying an additional mechanism to specify a shape transfer
function, the reference implementation of the operation will be used to derive
the shape function. The reference implementation is general and can support the
arbitrary computations needed to specify output shapes.

[InferTypeOpInterface]: https://github.com/llvm/llvm-project/tree/master/mlir/include/mlir/Interfaces/InferTypeOpInterface.td
[ShapedType]: https://github.com/llvm/llvm-project/tree/master/mlir/include/mlir/IR/BuiltinTypes.h
