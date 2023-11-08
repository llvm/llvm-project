//===- TensorBase.td - Base definitions for tensor dialect -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TENSOR_BASE
#define TENSOR_BASE

include "mlir/IR/OpBase.td"

def Tensor_Dialect : Dialect {
  let name = "tensor";
  let cppNamespace = "::mlir::tensor";

  let description = [{
    The `tensor` dialect is intended to hold core tensor creation and
    manipulation ops, which are not strongly associated with any particular
    other dialect or domain abstraction. The primary inclusion criteria for ops
    in this dialect is that they make sense for any tensor element type. When
    this is not the case, the op is left to live in other dialects. Examples of
    element types that could be supported by the `tensor` dialect include:

    - representing large, dense aggregations of primitive types, suitable for
      high-performance numerical computing.
    - representing shapes in the `shape` dialect, which consist of small 1D
      tensors of `index` data type.
    - representing aggregations of strings or “variant” types.
    - representing large, sparse aggregations of primitive types, suitable for
      high-performance numerical computing.

    Because of this broad element type support, we prefer for now to keep the
    `tensor` dialect as small as possible. The expectation is that at some point
    in the future, the `tensor` dialect’s scope may be broadened through a
    careful discussion of the tradeoffs.

    One exception to the above is the `tensor` type itself, which is actually a
    builtin type (it lives in the builtin dialect), and does not live in this
    dialect.

    Finally, many ops in the the dialect use the `dest` operand. This is an
    operand that is used to encode information for bufferization via the
    `DestinationStyleOpInterface`, see the [Destination Passing Style](
    https://mlir.llvm.org/docs/Bufferization/#destination-passing-style)
    documentation for more information.
  }];

  let hasCanonicalizer = 1;
  let hasConstantMaterializer = 1;
  let dependentDialects = [
    "affine::AffineDialect",
    "arith::ArithDialect",
    "complex::ComplexDialect",
  ];
}

#endif // TENSOR_BASE
