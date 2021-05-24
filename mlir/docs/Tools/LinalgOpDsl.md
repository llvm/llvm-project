# linalg_opdsl tool

Python based DSL for authoring Linalg op definitions and generating
`linalg.generic` IR based on them for samples.

The tool `linalg_opdsl` provides a high level DSL for constructing
structured op definitions in a way that can be exported to built-in, named
structured ops via the above YAML-based definitions or used interactively to
emit corresponding `linalg.generic` IR for the composition.

## Basic usage

The tool is bundled with the MLIR Python bindings. To use from the CMake build
tree, MLIR must be build with Python bindings enabled
(`-DMLIR_ENALBE_BINDINGS_PYTHON=ON`). Then add the `python` directory in the
build tree to your `PYTHONPATH` environment variable (i.e.
`export PYTHONPATH=$PWD/build/python`). Optionally, use an installed MLIR
package, if available, to avoid building.

```shell
# Dump the `core_named_ops.py` module as YAML.
python -m python -m mlir.tools.linalg_opdsl.dump_oplib .ops.core_named_ops
```

The tool is meant for use during both development and runtime, but not as
a build tool of the core compiler: in order to export static named op
definitions to be built as part of the compiler, the corresponding Linalg
dialect YAML file must be updated and reviewed. TODO: Develop a script to
automate op updates to these files.

## Language Guide

The language presented here is loosely inspired from the
[Tensor Comprehensions](https://arxiv.org/pdf/1802.04730.pdf) work, adapted to
represent linalg structured ops.

This tool is new and rapidly evolving. For language examples, refer to the
built-in ops in the `mlir.tools.linalg_opdsl.ops` package
(`lib/Bindings/Python/mlir/tools/linalg_opdsl/ops` in the repository).

Using a matmul as an example, we will decompose the language:

```python
T1 = TV.T1
T2 = TV.T2

@linalg_structured_op
def matmul(A=TensorDef(T1, S.M, S.K),
           B=TensorDef(T2, S.K, S.N),
           C=TensorDef(U, S.M, S.N, output=True)):
  """Performs a matrix multiplication of two 2D inputs.

  Numeric casting is performed on the operands to the inner multiply, promoting
  them to the same data type as the accumulator/output.
  """
  implements(ContractionOpInterface)
  C[D.m, D.n] += cast(U, A[D.m, D.k]) * cast(U, B[D.k, D.n])
```

Here we have a simple type polymorphic contraction that takes arguments `A`
and `B` and outputs `C`. Each is bound to a `TensorDef`, which specifies:

* The symbolic element type (`T1`, `T2`, `U` above).
* Symbolic shape expressions with symbols that are bound globally for the op (
note that in this simple example, the shape expressions are just symbol
references, but they are permitted to be a constrained set of affine
expressions).
* Usage (`output=True`).

The docstring will be transferred to the op definition verbatim.

Special identifying op interfaces can be declared for the op via
`implements(interface1[, interface2...])`.

## Parameters

Structured operations can take two types of parameters namely input/output
tensors and captures. Assignment expressions index the tensor parameters to
access the individual elements, while captures are scalars that can be
accessed directly.

The following example demonstrates the use of the two parameter types:

```python
@linalg_structured_op
def copy_and_scale(I=TensorDef(T, S.M, S.K),
                   O=TensorDef(T, S.M, S.K, output=True),
                   val=CaptureDef(T)):
  """Scale the input by the captured value and store the result"""
  O[D.m, D.n] = I[D.m, D.n] * val
```

The operation scales the input tensor `I` scales its elements by the value
`val` and writes the result to the output tensor `out`. The capture `val` is
bound to a `CaptureDef`, which specifies the type of the captured value. The
tensors are bound to a `TensorDef` as demonstrated by the matmul example. All
parameters appear in the parameter list of the operation:

```python
fill(in_tensor, outs=[out_tensor], captures=[captured_val])
```

## Assignments

The bulk of language consists of assignment expressions of the form above.
The iteration dimension order is determined lexically based on the order
encountered in the expression (following operator precedence if math operators
are used). TODO: Introduce a directive to fix the dimension bindings.

Reduction dimensions are inferred to be any dimensions on the RHS that are not
on the LHS.

A number of arithmetic primitive functions are supported:

* `PrimFn.add(a, b)` (also via overloading the binary `+` operator)
* `PrimFn.exp(a)`
* `PrimFn.log(a)`
* `PrimFn.mul(a, b)` (also via overloading the binary `*` operator)
* `PrimFn.max(a, b)`
* `PrimFn.sub(a, b)` (also via overloading the binary `-` operator)

Reduction functions can appear as the outer-most function on the RHS:

* `ReduceFn.add` (also overloading the inplace `+=` on a LHS)
* `ReduceFn.mul`
* `ReduceFn.max`

There are also special forms:

* `cast(TypeVar, operand)` casts the `operand` to the target type `TypeVar`.
* `const(TypeVar, value)` returns a constant value of type `TypeVar`.
* `index(dim)` returns the iteration index in the given dimension `dim`.

## Types

All types in assignment expressions are late bound based on actual input
and output types of constructed ops. An exception are predefined types such as
`I32`, `I64`, `F32`, and `F64`. These hardwired types enable intermediate
computations with a type that is independent of the input and output types.
For example, parts of floating point computation may require double precision
arithmetic despite all inputs and outputs being single precision values.
Assignment expressions with no `cast` calls will generally require uniform
types throughout and will fail to verify if violated. The presence of a
`cast` allows for a limited form of numeric type conversion between element
types that can be derived from inputs and outputs (and in the future,
attributes). `cast` calls with a `TypeVar` first argument are emitted as
`symbolic_cast` primitives in the YAML definition.

Casting will perform `int<->float` and `index->int` type conversions and will
perform any necessary extension or truncation within type family. Note that
presently, any integer type is assumed to be signed for the purpose of
determining how to extend or truncate. Supporting unsigned integer types is
left for future work.

Not all functions are applicable for all numeric types, and on mismatch, op
verification will fail.
