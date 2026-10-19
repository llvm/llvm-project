//===- MathTransformOps.td - Math transform ops ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MATH_TRANSFORM_OPS
#define MATH_TRANSFORM_OPS

include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.td"
include "mlir/IR/OpBase.td"

def ApplyF32ExpansionPatternsOp : Op<Transform_Dialect,
    "apply_patterns.math.f32_expansion",
    [DeclareOpInterfaceMethods<PatternDescriptorOpInterface>]> {
  let description = [{
    Converts math functions to f32: operands are extended and the result is
    truncated back.
  }];

  let assemblyFormat = "attr-dict";
}

def ApplyPolynomialApproximationPatternsOp : Op<Transform_Dialect,
    "apply_patterns.math.polynomial_approximation",
    [DeclareOpInterfaceMethods<PatternDescriptorOpInterface>]> {
  let description = [{
    Enables polynomial approximations for math functions.
    Optionally, allows usage of x86 AVX2 intrinsics.
  }];

  let arguments = (ins UnitAttr:$enable_avx2);
  let assemblyFormat = "(`enable_avx2` $enable_avx2^)? attr-dict";
}

#endif // MATH_TRANSFORM_OPS
