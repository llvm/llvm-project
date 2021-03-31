//===- MathOps.td - Math op definitions --------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MATH_OPS
#define MATH_OPS

include "mlir/Dialect/Math/IR/MathBase.td"
include "mlir/Interfaces/VectorInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class MathOp<string mnemonic, list<OpTrait> traits = []>
    : Op<Math_Dialect, mnemonic, traits # [NoSideEffect]>;

class FloatUnaryOp<string mnemonic, list<OpTrait> traits = []> :
    MathOp<mnemonic, traits #
    [DeclareOpInterfaceMethods<VectorUnrollOpInterface>,
    SameOperandsAndResultType] # ElementwiseMappable.traits> {
  let arguments = (ins FloatLike:$operand);

  let results = (outs FloatLike:$result);

  let assemblyFormat = "$operand attr-dict `:` type($result)";
}

class FloatBinaryOp<string mnemonic, list<OpTrait> traits = []> :
    MathOp<mnemonic, traits # [
    DeclareOpInterfaceMethods<VectorUnrollOpInterface>,
    SameOperandsAndResultType] # ElementwiseMappable.traits> {
  let arguments = (ins FloatLike:$lhs, FloatLike:$rhs);
  let results = (outs FloatLike:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// AtanOp
//===----------------------------------------------------------------------===//

def AtanOp : FloatUnaryOp<"atan">{
  let summary = "arcus tangent of the given value";
  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `math.atan` ssa-use `:` type
    ```

    The `atan` operation computes the arcus tangent of a given value.  It takes
    one operand and returns one result of the same type.  This type may be a
    float scalar type, a vector whose element type is float, or a tensor of
    floats.  It has no standard attributes.

    Example:

    ```mlir
    // Arcus tangent of scalar value.
    %a = math.atan %b : f64

    // SIMD vector element-wise arcus tangent.
    %f = math.atan %g : vector<4xf32>

    // Tensor element-wise arcus tangent.
    %x = math.atan %y : tensor<4x?xf8>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// Atan2Op
//===----------------------------------------------------------------------===//

def Atan2Op : FloatBinaryOp<"atan2">{
  let summary = "2-argument arcus tangent of the given values";
  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `math.atan2` ssa-use `,` ssa-use `:` type
    ```

    The `atan2` operation takes two operands and returns one result, all of
    which must be of the same type.  This type may be a floating point scalar
    type, a vector whose element type is a floating point type, or a floating
    point tensor.

    The 2-argument arcus tangent `atan2(y, x)` returns the angle in the
    Euclidian plane between the positive x-axis and the ray through the point
    (x, y).  It is a generalization of the 1-argument arcus tangent which
    returns the angle on the basis of the ratio y/x.

    See also https://en.wikipedia.org/wiki/Atan2

    Example:

    ```mlir
    // Scalar variant.
    %a = math.atan2 %b, %c : f32

    // SIMD vector variant.
    %f = math.atan2 %g, %h : vector<4xf32>

    // Tensor variant.
    %x = math.atan2 %y, %z : tensor<4x?xf32>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// CosOp
//===----------------------------------------------------------------------===//

def CosOp : FloatUnaryOp<"cos"> {
  let summary = "cosine of the specified value";
  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `math.cos` ssa-use `:` type
    ```

    The `cos` operation computes the cosine of a given value. It takes one
    operand and returns one result of the same type. This type may be a float
    scalar type, a vector whose element type is float, or a tensor of floats.
    It has no standard attributes.

    Example:

    ```mlir
    // Scalar cosine value.
    %a = math.cos %b : f64

    // SIMD vector element-wise cosine value.
    %f = math.cos %g : vector<4xf32>

    // Tensor element-wise cosine value.
    %x = math.cos %y : tensor<4x?xf8>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// SinOp
//===----------------------------------------------------------------------===//

def SinOp : FloatUnaryOp<"sin"> {
  let summary = "sine of the specified value";
  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `math.sin` ssa-use `:` type
    ```

    The `sin` operation computes the sine of a given value. It takes one
    operand and returns one result of the same type. This type may be a float
    scalar type, a vector whose element type is float, or a tensor of floats.
    It has no standard attributes.

    Example:

    ```mlir
    // Scalar sine value.
    %a = math.sin %b : f64

    // SIMD vector element-wise sine value.
    %f = math.sin %g : vector<4xf32>

    // Tensor element-wise sine value.
    %x = math.sin %y : tensor<4x?xf8>
    ```
  }];
}


//===----------------------------------------------------------------------===//
// ExpOp
//===----------------------------------------------------------------------===//

def ExpOp : FloatUnaryOp<"exp"> {
  let summary = "base-e exponential of the specified value";
  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `math.exp` ssa-use `:` type
    ```

    The `exp` operation takes one operand and returns one result of the same
    type. This type may be a float scalar type, a vector whose element type is
    float, or a tensor of floats. It has no standard attributes.

    Example:

    ```mlir
    // Scalar natural exponential.
    %a = math.exp %b : f64

    // SIMD vector element-wise natural exponential.
    %f = math.exp %g : vector<4xf32>

    // Tensor element-wise natural exponential.
    %x = math.exp %y : tensor<4x?xf8>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// Exp2Op
//===----------------------------------------------------------------------===//

def Exp2Op : FloatUnaryOp<"exp2"> {
  let summary = "base-2 exponential of the specified value";

  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `math.exp2` ssa-use `:` type
    ```

    The `exp` operation takes one operand and returns one result of the same
    type. This type may be a float scalar type, a vector whose element type is
    float, or a tensor of floats. It has no standard attributes.

    Example:

    ```mlir
    // Scalar natural exponential.
    %a = math.exp2 %b : f64

    // SIMD vector element-wise natural exponential.
    %f = math.exp2 %g : vector<4xf32>

    // Tensor element-wise natural exponential.
    %x = math.exp2 %y : tensor<4x?xf8>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// ExpM1Op
//===----------------------------------------------------------------------===//

def ExpM1Op : FloatUnaryOp<"expm1"> {
  let summary = "base-e exponential of the specified value minus 1";
  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `math.expm1` ssa-use `:` type
    ```

    expm1(x) := exp(x) - 1

    The `expm1` operation takes one operand and returns one result of the same
    type. This type may be a float scalar type, a vector whose element type is
    float, or a tensor of floats. It has no standard attributes.

    Example:

    ```mlir
    // Scalar natural exponential minus 1.
    %a = math.expm1 %b : f64

    // SIMD vector element-wise natural exponential minus 1.
    %f = math.expm1 %g : vector<4xf32>

    // Tensor element-wise natural exponential minus 1.
    %x = math.expm1 %y : tensor<4x?xf8>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// LogOp
//===----------------------------------------------------------------------===//

def LogOp : FloatUnaryOp<"log"> {
  let summary = "base-e logarithm of the specified value";

  let description = [{
    Computes the base-e logarithm of the given value. It takes one operand and
    returns one result of the same type.

    Example:

    ```mlir
    %y = math.log %x : f64
    ```
  }];
}

//===----------------------------------------------------------------------===//
// Log10Op
//===----------------------------------------------------------------------===//

def Log10Op : FloatUnaryOp<"log10"> {
  let summary = "base-10 logarithm of the specified value";

  let description = [{
    Computes the base-10 logarithm of the given value. It takes one operand and
    returns one result of the same type.

    Example:

    ```mlir
    %y = math.log10 %x : f64
    ```
  }];
}

//===----------------------------------------------------------------------===//
// Log1pOp
//===----------------------------------------------------------------------===//

def Log1pOp : FloatUnaryOp<"log1p"> {
  let summary = "Computes the natural logarithm of one plus the given value";

  let description = [{
    Computes the base-e logarithm of one plus the given value. It takes one
    operand and returns one result of the same type.

    log1p(x) := log(1 + x)

    Example:

    ```mlir
    %y = math.log1p %x : f64
    ```
  }];
}

//===----------------------------------------------------------------------===//
// Log2Op
//===----------------------------------------------------------------------===//

def Log2Op : FloatUnaryOp<"log2"> {
  let summary = "base-2 logarithm of the specified value";

  let description = [{
    Computes the base-2 logarithm of the given value. It takes one operand and
    returns one result of the same type.

    Example:

    ```mlir
    %y = math.log2 %x : f64
    ```
  }];
}

//===----------------------------------------------------------------------===//
// PowFOp
//===----------------------------------------------------------------------===//

def PowFOp : FloatBinaryOp<"powf"> {
  let summary = "floating point raised to the power of operation";
  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `math.powf` ssa-use `,` ssa-use `:` type
    ```

    The `powf` operation takes two operands and returns one result, each of
    these is required to be the same type. This type may be a floating point
    scalar type, a vector whose element type is a floating point type, or a
    floating point tensor.

    Example:

    ```mlir
    // Scalar exponentiation.
    %a = math.powf %b, %c : f64

    // SIMD pointwise vector exponentiation
    %f = math.powf %g, %h : vector<4xf32>

    // Tensor pointwise exponentiation.
    %x = math.powf %y, %z : tensor<4x?xbf16>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// RsqrtOp
//===----------------------------------------------------------------------===//

def RsqrtOp : FloatUnaryOp<"rsqrt"> {
  let summary = "reciprocal of sqrt (1 / sqrt of the specified value)";
  let description = [{
    The `rsqrt` operation computes the reciprocal of the square root. It takes
    one operand and returns one result of the same type. This type may be a
    float scalar type, a vector whose element type is float, or a tensor of
    floats. It has no standard attributes.
  }];
}

//===----------------------------------------------------------------------===//
// SqrtOp
//===----------------------------------------------------------------------===//

def SqrtOp : FloatUnaryOp<"sqrt"> {
  let summary = "sqrt of the specified value";
  let description = [{
    The `sqrt` operation computes the square root. It takes one operand and
    returns one result of the same type. This type may be a float scalar type, a
    vector whose element type is float, or a tensor of floats. It has no standard
    attributes.

    Example:

    ```mlir
    // Scalar square root value.
    %a = math.sqrt %b : f64
    // SIMD vector element-wise square root value.
    %f = math.sqrt %g : vector<4xf32>
    // Tensor element-wise square root value.
    %x = math.sqrt %y : tensor<4x?xf32>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// TanhOp
//===----------------------------------------------------------------------===//

def TanhOp : FloatUnaryOp<"tanh"> {
  let summary = "hyperbolic tangent of the specified value";
  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `std.tanh` ssa-use `:` type
    ```

    The `tanh` operation computes the hyperbolic tangent. It takes one operand
    and returns one result of the same type. This type may be a float scalar
    type, a vector whose element type is float, or a tensor of floats. It has
    no standard attributes.

    Example:

    ```mlir
    // Scalar hyperbolic tangent value.
    %a = math.tanh %b : f64

    // SIMD vector element-wise hyperbolic tangent value.
    %f = math.tanh %g : vector<4xf32>

    // Tensor element-wise hyperbolic tangent value.
    %x = math.tanh %y : tensor<4x?xf8>
    ```
  }];
}

#endif // MATH_OPS
