# Shape inference

Shape inference as discussed here is considered a specific instance of type
inference for [ShapedType][ShapedType]. Type constraints are along (at least)
three axis: 1) elemental type, 2) rank (including static or dynamic), 3)
dimensions. While some operations have no compile time fixed shape (e.g., output
shape is dictated by data) we could still have some knowledge of
constraints/bounds in the system for that operation (e.g., the output of a
`tf.where` is at most the size of the input data). That is, there are additional
valuable constraints that could be captured even without full knowledge of the
shape.

Type inference is currently modelled executionally for op creation using the
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
    of a value is of elemental type `i1`) or class (e.g., float like).
*   Constraints across operands and results of an operation.

    - For example, specifying equality constraints on type/constituents of a
      type (shape and elemental type) between operands and results (e.g., the
      output type of an add is the same as those of the input operands).

NOTE: The C++ shape functions are an intermediate step until the shape dialect
is more full-fledged, at which point the C++ functions should become the
exceptional case.

## Testing

Shape inference is currently tested alongside type inference by
`TestReturnTypeDriver` in the test dialect. The driver performs two checks:

1.  Verification that the return types specified matches the infered types. This
    explicit check will be removed and made part of Op verification instead.
2.  Test the creation of Ops without specifying the return type explicitly in
    function `testCreateFunctions` by creating new binary Ops (Op classes
    specified in `TestReturnTypeDriver`) using 1) all operands to
    `testCreateFunctions` as both operands, and 2) using combinations of input
    operands of the function.

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

[InferTypeOpInterface]: https://github.com/llvm/llvm-project/tree/master/mlir/include/mlir/Analysis/InferTypeOpInterface.td
[ShapedType]: https://github.com/llvm/llvm-project/tree/master/mlir/include/mlir/IR/StandardTypes.h
