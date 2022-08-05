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
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/VectorInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Base class for math dialect ops. Ops in this dialect have no side effects and
// can be applied element-wise to vectors and tensors.
class Math_Op<string mnemonic, list<Trait> traits = []> :
    Op<Math_Dialect, mnemonic, traits # [NoSideEffect,
    DeclareOpInterfaceMethods<VectorUnrollOpInterface>] #
    ElementwiseMappable.traits>;

// Base class for unary math operations on integer types. Require an operand
// and result of the same type. This type can be an integer type, vector or
// tensor thereof.
class Math_IntegerUnaryOp<string mnemonic, list<Trait> traits = []> :
    Math_Op<mnemonic, traits # [SameOperandsAndResultType]> {
  let arguments = (ins SignlessIntegerLike:$operand);
  let results = (outs SignlessIntegerLike:$result);

  let assemblyFormat = "$operand attr-dict `:` type($result)";
}

// Base class for unary math operations on floating point types. Require an
// operand and result of the same type. This type can be a floating point type,
// vector or tensor thereof.
class Math_FloatUnaryOp<string mnemonic, list<Trait> traits = []> :
    Math_Op<mnemonic, traits # [SameOperandsAndResultType]> {
  let arguments = (ins FloatLike:$operand);
  let results = (outs FloatLike:$result);

  let assemblyFormat = "$operand attr-dict `:` type($result)";
}

// Base class for binary math operations on floating point types. Require two
// operands and one result of the same type. This type can be a floating point
// type, vector or tensor thereof.
class Math_FloatBinaryOp<string mnemonic, list<Trait> traits = []> :
    Math_Op<mnemonic, traits # [SameOperandsAndResultType]> {
  let arguments = (ins FloatLike:$lhs, FloatLike:$rhs);
  let results = (outs FloatLike:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

// Base class for floating point ternary operations. Require three operands and
// one result of the same type. This type can be a floating point type, vector
// or tensor thereof.
class Math_FloatTernaryOp<string mnemonic, list<Trait> traits = []> :
    Math_Op<mnemonic, traits # [SameOperandsAndResultType]> {
  let arguments = (ins FloatLike:$a, FloatLike:$b, FloatLike:$c);
  let results = (outs FloatLike:$result);

  let assemblyFormat = "$a `,` $b `,` $c attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// AbsOp
//===----------------------------------------------------------------------===//

def Math_AbsOp : Math_FloatUnaryOp<"abs"> {
  let summary = "floating point absolute-value operation";
  let description = [{
    The `abs` operation computes the absolute value. It takes one operand of
    floating point type (i.e., scalar, tensor or vector) and returns one result of
    the same type.

    Example:

    ```mlir
    // Scalar absolute value.
    %a = math.abs %b : f64
    ```
  }];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// AtanOp
//===----------------------------------------------------------------------===//

def Math_AtanOp : Math_FloatUnaryOp<"atan">{
  let summary = "arcus tangent of the given value";
  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `math.atan` ssa-use `:` type
    ```

    The `atan` operation computes the arcus tangent of a given value.  It takes
    one operand of floating point type (i.e., scalar, tensor or vector) and returns
    one result of the same type. It has no standard attributes.

    Example:

    ```mlir
    // Arcus tangent of scalar value.
    %a = math.atan %b : f64
    ```
  }];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Atan2Op
//===----------------------------------------------------------------------===//

def Math_Atan2Op : Math_FloatBinaryOp<"atan2">{
  let summary = "2-argument arcus tangent of the given values";
  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `math.atan2` ssa-use `,` ssa-use `:` type
    ```

    The `atan2` operation takes two operands and returns one result, all of
    which must be of the same type.  The operands must be of floating point type
    (i.e., scalar, tensor or vector).

    The 2-argument arcus tangent `atan2(y, x)` returns the angle in the
    Euclidian plane between the positive x-axis and the ray through the point
    (x, y).  It is a generalization of the 1-argument arcus tangent which
    returns the angle on the basis of the ratio y/x.

    See also https://en.wikipedia.org/wiki/Atan2

    Example:

    ```mlir
    // Scalar variant.
    %a = math.atan2 %b, %c : f32
    ```
  }];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// CeilOp
//===----------------------------------------------------------------------===//

def Math_CeilOp : Math_FloatUnaryOp<"ceil"> {
  let summary = "ceiling of the specified value";
  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `math.ceil` ssa-use `:` type
    ```

    The `ceil` operation computes the ceiling of a given value. It takes one
    operand of floating point type (i.e., scalar, tensor or vector) and returns one
    result of the same type.  It has no standard attributes.

    Example:

    ```mlir
    // Scalar ceiling value.
    %a = math.ceil %b : f64
    ```
  }];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// CopySignOp
//===----------------------------------------------------------------------===//

def Math_CopySignOp : Math_FloatBinaryOp<"copysign"> {
  let summary = "A copysign operation";
  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `math.copysign` ssa-use `,` ssa-use `:` type
    ```

    The `copysign` returns a value with the magnitude of the first operand and
    the sign of the second operand. It takes two operands and returns one result of
    the same type. The operands must be of floating point type (i.e., scalar,
    tensor or vector). It has no standard attributes.

    Example:

    ```mlir
    // Scalar copysign value.
    %a = math.copysign %b, %c : f64
    ```
  }];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// CosOp
//===----------------------------------------------------------------------===//

def Math_CosOp : Math_FloatUnaryOp<"cos"> {
  let summary = "cosine of the specified value";
  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `math.cos` ssa-use `:` type
    ```

    The `cos` operation computes the cosine of a given value. It takes one
    operand of floating point type (i.e., scalar, tensor or vector) and returns one
    result of the same type.  It has no standard attributes.

    Example:

    ```mlir
    // Scalar cosine value.
    %a = math.cos %b : f64
    ```
  }];
}

//===----------------------------------------------------------------------===//
// SinOp
//===----------------------------------------------------------------------===//

def Math_SinOp : Math_FloatUnaryOp<"sin"> {
  let summary = "sine of the specified value";
  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `math.sin` ssa-use `:` type
    ```

    The `sin` operation computes the sine of a given value. It takes one
    operand of floating point type (i.e., scalar, tensor or vector) and returns one
    result of the same type.  It has no standard attributes.

    Example:

    ```mlir
    // Scalar sine value.
    %a = math.sin %b : f64
    ```
  }];
}

//===----------------------------------------------------------------------===//
// CountLeadingZerosOp
//===----------------------------------------------------------------------===//

def Math_CountLeadingZerosOp : Math_IntegerUnaryOp<"ctlz"> {
  let summary = "counts the leading zeros an integer value";
  let description = [{
    The `ctlz` operation computes the number of leading zeros of an integer value.
    It operates on scalar, tensor or vector.

    Example:

    ```mlir
    // Scalar ctlz function value.
    %a = math.ctlz %b : i32
    ```
  }];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// CountTrailingZerosOp
//===----------------------------------------------------------------------===//

def Math_CountTrailingZerosOp : Math_IntegerUnaryOp<"cttz"> {
  let summary = "counts the trailing zeros an integer value";
  let description = [{
    The `cttz` operation computes the number of trailing zeros of an integer value.
    It operates on scalar, tensor or vector.

    Example:

    ```mlir
    // Scalar cttz function value.
    %a = math.cttz %b : i32
    ```
  }];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// CtPopOp
//===----------------------------------------------------------------------===//

def Math_CtPopOp : Math_IntegerUnaryOp<"ctpop"> {
  let summary = "counts the number of set bits of an integer value";
  let description = [{
    The `ctpop` operation computes the number of set bits of an integer value.
    It operates on scalar, tensor or vector.

    Example:

    ```mlir
    // Scalar ctpop function value.
    %a = math.ctpop %b : i32
    ```
  }];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// ErfOp
//===----------------------------------------------------------------------===//

def Math_ErfOp : Math_FloatUnaryOp<"erf"> {
  let summary = "error function of the specified value";
  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `math.erf` ssa-use `:` type
    ```

    The `erf` operation computes the error function. It takes one operand of
    floating point type (i.e., scalar, tensor or vector) and returns one result of
    the same type. It has no standard attributes.

    Example:

    ```mlir
    // Scalar error function value.
    %a = math.erf %b : f64
    ```
  }];
}


//===----------------------------------------------------------------------===//
// ExpOp
//===----------------------------------------------------------------------===//

def Math_ExpOp : Math_FloatUnaryOp<"exp"> {
  let summary = "base-e exponential of the specified value";
  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `math.exp` ssa-use `:` type
    ```

    The `exp` operation takes one operand of floating point type (i.e., scalar,
    tensor or vector) and returns one result of the same type. It has no standard
    attributes.

    Example:

    ```mlir
    // Scalar natural exponential.
    %a = math.exp %b : f64
    ```
  }];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Exp2Op
//===----------------------------------------------------------------------===//

def Math_Exp2Op : Math_FloatUnaryOp<"exp2"> {
  let summary = "base-2 exponential of the specified value";

  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `math.exp2` ssa-use `:` type
    ```

    The `exp` operation takes one operand of floating point type (i.e., scalar,
    tensor or vector) and returns one result of the same type. It has no standard
    attributes.

    Example:

    ```mlir
    // Scalar natural exponential.
    %a = math.exp2 %b : f64
    ```
  }];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// ExpM1Op
//===----------------------------------------------------------------------===//

def Math_ExpM1Op : Math_FloatUnaryOp<"expm1"> {
  let summary = "base-e exponential of the specified value minus 1";
  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `math.expm1` ssa-use `:` type
    ```

    expm1(x) := exp(x) - 1

    The `expm1` operation takes one operand of floating point type (i.e.,
    scalar, tensor or vector) and returns one result of the same type. It has no
    standard attributes.

    Example:

    ```mlir
    // Scalar natural exponential minus 1.
    %a = math.expm1 %b : f64
    ```
  }];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// FloorOp
//===----------------------------------------------------------------------===//

def Math_FloorOp : Math_FloatUnaryOp<"floor"> {
  let summary = "floor of the specified value";
  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `math.floor` ssa-use `:` type
    ```

    The `floor` operation computes the floor of a given value. It takes one
    operand of floating point type (i.e., scalar, tensor or vector) and returns one
    result of the same type.  It has no standard attributes.

    Example:

    ```mlir
    // Scalar floor value.
    %a = math.floor %b : f64 
    ```
  }];
}

//===----------------------------------------------------------------------===//
// FmaOp
//===----------------------------------------------------------------------===//

def Math_FmaOp : Math_FloatTernaryOp<"fma"> {
  let summary = "floating point fused multipy-add operation";
  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `math.fma` ssa-use `,` ssa-use `,` ssa-use `:` type
    ```

    The `fma` operation takes three operands and returns one result, each of
    these is required to be the same type. Operands must be of floating point type
    (i.e., scalar, tensor or vector).

    Example:

    ```mlir
    // Scalar fused multiply-add: d = a*b + c
    %d = math.fma %a, %b, %c : f64
    ```

    The semantics of the operation correspond to those of the `llvm.fma`
    [intrinsic](https://llvm.org/docs/LangRef.html#llvm-fma-intrinsic). In the
    particular case of lowering to LLVM, this is guaranteed to lower
    to the `llvm.fma.*` intrinsic.
  }];
}

//===----------------------------------------------------------------------===//
// LogOp
//===----------------------------------------------------------------------===//

def Math_LogOp : Math_FloatUnaryOp<"log"> {
  let summary = "base-e logarithm of the specified value";

  let description = [{
    Computes the base-e logarithm of the given value. It takes one operand of
    floating point type (i.e., scalar, tensor or vector) and returns one result of
    the same type.

    Example:

    ```mlir
    // Scalar log operation.
    %y = math.log %x : f64
    ```
  }];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Log10Op
//===----------------------------------------------------------------------===//

def Math_Log10Op : Math_FloatUnaryOp<"log10"> {
  let summary = "base-10 logarithm of the specified value";

  let description = [{
    Computes the base-10 logarithm of the given value. It takes one operand of
    floating point type (i.e., scalar, tensor or vector) and returns one result of
    the same type.

    Example:

    ```mlir
    // Scalar log10 operation.
    %y = math.log10 %x : f64
    ```
  }];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Log1pOp
//===----------------------------------------------------------------------===//

def Math_Log1pOp : Math_FloatUnaryOp<"log1p"> {
  let summary = "Computes the natural logarithm of one plus the given value";

  let description = [{
    Computes the base-e logarithm of one plus the given value. It takes one
    operand of floating point type (i.e., scalar, tensor or vector) and returns one
    result of the same type.

    log1p(x) := log(1 + x)

    Example:

    ```mlir
    // Scalar log1p operation.
    %y = math.log1p %x : f64
    ```
  }];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Log2Op
//===----------------------------------------------------------------------===//

def Math_Log2Op : Math_FloatUnaryOp<"log2"> {
  let summary = "base-2 logarithm of the specified value";

  let description = [{
    Computes the base-2 logarithm of the given value. It takes one operand of
    floating point type (i.e., scalar, tensor or vector) and returns one result of
    the same type.

    Example:

    ```mlir
    // Scalar log2 operation.
    %y = math.log2 %x : f64
    ```
  }];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// PowFOp
//===----------------------------------------------------------------------===//

def Math_PowFOp : Math_FloatBinaryOp<"powf"> {
  let summary = "floating point raised to the power of operation";
  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `math.powf` ssa-use `,` ssa-use `:` type
    ```

    The `powf` operation takes two operands of floating point type (i.e.,
    scalar, tensor or vector) and returns one result of the same type. Operands
    must have the same type.

    Example:

    ```mlir
    // Scalar exponentiation.
    %a = math.powf %b, %c : f64
    ```
  }];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// RsqrtOp
//===----------------------------------------------------------------------===//

def Math_RsqrtOp : Math_FloatUnaryOp<"rsqrt"> {
  let summary = "reciprocal of sqrt (1 / sqrt of the specified value)";
  let description = [{
    The `rsqrt` operation computes the reciprocal of the square root. It takes
    one operand of floating point type (i.e., scalar, tensor or vector) and returns
    one result of the same type. It has no standard attributes.

    Example:
  
    ```mlir
    // Scalar reciprocal square root value.
    %a = math.rsqrt %b : f64
    ```
  }];
}

//===----------------------------------------------------------------------===//
// SqrtOp
//===----------------------------------------------------------------------===//

def Math_SqrtOp : Math_FloatUnaryOp<"sqrt"> {
  let summary = "sqrt of the specified value";
  let description = [{
    The `sqrt` operation computes the square root. It takes one operand of
    floating point type (i.e., scalar, tensor or vector) and returns one result of
    the same type. It has no standard attributes.

    Example:

    ```mlir
    // Scalar square root value.
    %a = math.sqrt %b : f64
    ```
  }];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// TanOp
//===----------------------------------------------------------------------===//

def Math_TanOp : Math_FloatUnaryOp<"tan"> {
  let summary = "tangent of the specified value";
  let description = [{
    The `tan` operation computes the tangent. It takes one operand
    of floating point type (i.e., scalar, tensor or vector) and returns one
    result of the same type. It has no standard attributes.

    Example:

    ```mlir
    // Scalar tangent value.
    %a = math.tan %b : f64
    ```
  }];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// TanhOp
//===----------------------------------------------------------------------===//

def Math_TanhOp : Math_FloatUnaryOp<"tanh"> {
  let summary = "hyperbolic tangent of the specified value";
  let description = [{
    The `tanh` operation computes the hyperbolic tangent. It takes one operand
    of floating point type (i.e., scalar, tensor or vector) and returns one
    result of the same type. It has no standard attributes.

    Example:

    ```mlir
    // Scalar hyperbolic tangent value.
    %a = math.tanh %b : f64
    ```
  }];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// RoundOp
//===----------------------------------------------------------------------===//

def Math_RoundOp : Math_FloatUnaryOp<"round"> {
  let summary = "round of the specified value";
  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `math.round` ssa-use `:` type
    ```

    The `round` operation returns the operand rounded to the nearest integer
    value in floating-point format. It takes one operand of floating point type
    (i.e., scalar, tensor or vector) and produces one result of the same type.  The
    operation rounds the argument to the nearest integer value in floating-point
    format, rounding halfway cases away from zero, regardless of the current
    rounding direction.

    Example:

    ```mlir
    // Scalar round operation.
    %a = math.round %b : f64
    ```
  }];
}

#endif // MATH_OPS
