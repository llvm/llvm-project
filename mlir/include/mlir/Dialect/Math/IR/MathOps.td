//===- MathOps.td - Math op definitions --------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MATH_OPS
#define MATH_OPS

include "mlir/Dialect/Arith/IR/ArithBase.td"
include "mlir/Dialect/Arith/IR/ArithOpsInterfaces.td"
include "mlir/Dialect/Math/IR/MathBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/VectorInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Base class for math dialect ops. Ops in this dialect have no side effects and
// can be applied element-wise to vectors and tensors.
class Math_Op<string mnemonic, list<Trait> traits = []> :
    Op<Math_Dialect, mnemonic, traits # [Pure,
    DeclareOpInterfaceMethods<VectorUnrollOpInterface>] #
    ElementwiseMappable.traits>;

// Base class for unary math operations on integer types. Require an operand
// and result of the same type. This type can be an integer type, vector or
// tensor thereof.
class Math_IntegerUnaryOp<string mnemonic, list<Trait> traits = []> :
    Math_Op<mnemonic, traits # [SameOperandsAndResultType]> {
  let arguments = (ins SignlessIntegerOrIndexLike:$operand);
  let results = (outs SignlessIntegerOrIndexLike:$result);

  let assemblyFormat = "$operand attr-dict `:` type($result)";
}

// Base class for floating point classification ops. Require an operand and
// result of the same shape, which can be a floating point scalar, a vector or a
// tensor thereof.
class Math_FloatClassificationOp<string mnemonic, list<Trait> traits = []> :
    Math_Op<mnemonic,
      traits # [DeclareOpInterfaceMethods<ArithFastMathInterface>,
                TypesMatchWith<
          "result type has i1 element type and same shape as operands",
          "operand", "result", "::getI1SameShape($_self)">]> {
  let arguments = (ins FloatLike:$operand,
      DefaultValuedAttr<Arith_FastMathAttr,
                        "::mlir::arith::FastMathFlags::none">:$fastmath);
  let results = (outs BoolLike:$result);

  let assemblyFormat = "$operand attr-dict `:` type($operand)";
}

// Base class for unary math operations on floating point types. Require an
// operand and result of the same type. This type can be a floating point type,
// vector or tensor thereof.
class Math_FloatUnaryOp<string mnemonic, list<Trait> traits = []> :
    Math_Op<mnemonic,
        traits # [SameOperandsAndResultType,
                  DeclareOpInterfaceMethods<ArithFastMathInterface>]> {
  let arguments = (ins FloatLike:$operand,
      DefaultValuedAttr<Arith_FastMathAttr,
                        "::mlir::arith::FastMathFlags::none">:$fastmath);
  let results = (outs FloatLike:$result);

  let assemblyFormat = [{ $operand (`fastmath` `` $fastmath^)?
                          attr-dict `:` type($result) }];
}

// Base class for binary math operations on integer types. Require two
// operands and one result of the same type. This type can be an integer
// type, vector or tensor thereof.
class Math_IntegerBinaryOp<string mnemonic, list<Trait> traits = []> :
    Math_Op<mnemonic, traits # [SameOperandsAndResultType]> {
  let arguments = (ins SignlessIntegerOrIndexLike:$lhs, SignlessIntegerOrIndexLike:$rhs);
  let results = (outs SignlessIntegerOrIndexLike:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

// Base class for binary math operations on floating point types. Require two
// operands and one result of the same type. This type can be a floating point
// type, vector or tensor thereof.
class Math_FloatBinaryOp<string mnemonic, list<Trait> traits = []> :
    Math_Op<mnemonic,
        traits # [SameOperandsAndResultType,
                  DeclareOpInterfaceMethods<ArithFastMathInterface>]> {
  let arguments = (ins FloatLike:$lhs, FloatLike:$rhs,
      DefaultValuedAttr<Arith_FastMathAttr,
                        "::mlir::arith::FastMathFlags::none">:$fastmath);
  let results = (outs FloatLike:$result);

  let assemblyFormat = [{ $lhs `,` $rhs (`fastmath` `` $fastmath^)?
                          attr-dict `:` type($result) }];
}

// Base class for floating point ternary operations. Require three operands and
// one result of the same type. This type can be a floating point type, vector
// or tensor thereof.
class Math_FloatTernaryOp<string mnemonic, list<Trait> traits = []> :
    Math_Op<mnemonic,
        traits # [SameOperandsAndResultType,
                  DeclareOpInterfaceMethods<ArithFastMathInterface>]> {
  let arguments = (ins FloatLike:$a, FloatLike:$b, FloatLike:$c,
      DefaultValuedAttr<Arith_FastMathAttr,
                        "::mlir::arith::FastMathFlags::none">:$fastmath);
  let results = (outs FloatLike:$result);

  let assemblyFormat = [{ $a `,` $b `,` $c (`fastmath` `` $fastmath^)?
                          attr-dict `:` type($result) }];
}

//===----------------------------------------------------------------------===//
// AbsFOp
//===----------------------------------------------------------------------===//

def Math_AbsFOp : Math_FloatUnaryOp<"absf"> {
  let summary = "floating point absolute-value operation";
  let description = [{
    The `absf` operation computes the absolute value. It takes one operand of
    floating point type (i.e., scalar, tensor or vector) and returns one result
    of the same type.

    Example:

    ```mlir
    // Scalar absolute value.
    %a = math.absf %b : f64
    ```
  }];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// AbsIOp
//===----------------------------------------------------------------------===//

def Math_AbsIOp : Math_IntegerUnaryOp<"absi"> {
  let summary = "integer absolute-value operation";
  let description = [{
    The `absi` operation computes the absolute value. It takes one operand of
    integer type (i.e., scalar, tensor or vector) and returns one result of the
    same type.

    Example:

    ```mlir
    // Scalar absolute value.
    %a = math.absi %b : i64
    ```
  }];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// AcoshOp
//===----------------------------------------------------------------------===//

def Math_AcoshOp : Math_FloatUnaryOp<"acosh">{
  let summary = "Hyperbolic arcus cosine of the given value";
  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `math.acosh` ssa-use `:` type
    ```

    The `acosh` operation computes the arcus cosine of a given value.  It takes
    one operand of floating point type (i.e., scalar, tensor or vector) and returns
    one result of the same type. It has no standard attributes.

    Example:

    ```mlir
    // Hyperbolic arcus cosine of scalar value.
    %a = math.acosh %b : f64
    ```
  }];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// AsinOp
//===----------------------------------------------------------------------===//

def Math_AsinOp : Math_FloatUnaryOp<"asin">{
  let summary = "arcus sine of the given value";
  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `math.asin` ssa-use `:` type
    ```

    The `asin` operation computes the arcus sine of a given value.  It takes
    one operand of floating point type (i.e., scalar, tensor or vector) and returns
    one result of the same type. It has no standard attributes.

    Example:

    ```mlir
    // Arcus sine of scalar value.
    %a = math.asin %b : f64
    ```
  }];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// AsinhOp
//===----------------------------------------------------------------------===//

def Math_AsinhOp : Math_FloatUnaryOp<"asinh">{
  let summary = "hyperbolic arcus sine of the given value";
  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `math.asinh` ssa-use `:` type
    ```

    The `asinh` operation computes the hyperbolic arcus sine of a given value.  It takes
    one operand of floating point type (i.e., scalar, tensor or vector) and returns
    one result of the same type. It has no standard attributes.

    Example:

    ```mlir
    // Hyperbolic arcus sine of scalar value.
    %a = math.asinh %b : f64
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
// AtanhOp
//===----------------------------------------------------------------------===//

def Math_AtanhOp : Math_FloatUnaryOp<"atanh">{
  let summary = "hyperbolic arcus tangent of the given value";
  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `math.atanh` ssa-use `:` type
    ```

    The `atanh` operation computes the hyperbolic arcus tangent of a given value.  It takes
    one operand of floating point type (i.e., scalar, tensor or vector) and returns
    one result of the same type. It has no standard attributes.

    Example:

    ```mlir
    // Hyperbolic arcus tangent of scalar value.
    %a = math.atanh %b : f64
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
// CbrtOp
//===----------------------------------------------------------------------===//

def Math_CbrtOp : Math_FloatUnaryOp<"cbrt"> {
  let summary = "cube root of the specified value";
  let description = [{
    The `cbrt` operation computes the cube root. It takes one operand of
    floating point type (i.e., scalar, tensor or vector) and returns one result
    of the same type. It has no standard attributes.

    Example:

    ```mlir
    // Scalar cube root value.
    %a = math.cbrt %b : f64
    ```

    Note: This op is not equivalent to powf(..., 1/3.0).
  }];
}

//===----------------------------------------------------------------------===//
// CeilOp
//===----------------------------------------------------------------------===//

def Math_CeilOp : Math_FloatUnaryOp<"ceil"> {
  let summary = "ceiling of the specified value";
  let description = [{
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
// ClampFOp
//===----------------------------------------------------------------------===//

def Math_ClampFOp : Math_FloatTernaryOp<"clampf"> {
  let summary = "floating point clamping operation";
  let description = [{
    The `clampf` operation takes three operands and returns one result, each of
    these is required to be the same type. Operands must be of floating point type
    (i.e., scalar, tensor or vector).

    The semantics of the operation are described by:
    ```
      clampf(value, min, max) = maxf(minf(value, min), max)
    ```

    Example:

    ```mlir
    %d = math.clampf %value to [%min, %max] : f64
    ```
  }];
  let arguments = (ins FloatLike:$value, FloatLike:$min, FloatLike:$max,
      DefaultValuedAttr<Arith_FastMathAttr,
                        "::mlir::arith::FastMathFlags::none">:$fastmath);
  let assemblyFormat = [{
    $value `to` ` ` `[` $min `,` $max `]` (`fastmath` `` $fastmath^)?
    attr-dict `:` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// CopySignOp
//===----------------------------------------------------------------------===//

def Math_CopySignOp : Math_FloatBinaryOp<"copysign"> {
  let summary = "A copysign operation";
  let description = [{
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
    The `cos` operation computes the cosine of a given value. It takes one
    operand of floating point type (i.e., scalar, tensor or vector) and returns one
    result of the same type.  It has no standard attributes.

    Example:

    ```mlir
    // Scalar cosine value.
    %a = math.cos %b : f64
    ```
  }];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// AcosOp
//===----------------------------------------------------------------------===//

def Math_AcosOp : Math_FloatUnaryOp<"acos"> {
  let summary = "arcus cosine of the specified value";
  let description = [{
    The `acos` operation computes the arcus cosine of a given value. It takes one
    operand of floating point type (i.e., scalar, tensor or vector) and returns one
    result of the same type.  It has no standard attributes.

    Example:

    ```mlir
    // Scalar arcus cosine value.
    %a = math.acos %b : f64
    ```
  }];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// CoshOp
//===----------------------------------------------------------------------===//

def Math_CoshOp : Math_FloatUnaryOp<"cosh"> {
  let summary = "hyperbolic cosine of the specified value";
  let description = [{
    The `cosh` operation computes the hyperbolic cosine. It takes one operand
    of floating point type (i.e., scalar, tensor or vector) and returns one
    result of the same type. It has no standard attributes.

    Example:

    ```mlir
    // Scalar hyperbolic cosine value.
    %a = math.cosh %b : f64
    ```
  }];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// SinOp
//===----------------------------------------------------------------------===//

def Math_SinOp : Math_FloatUnaryOp<"sin"> {
  let summary = "sine of the specified value";
  let description = [{
    The `sin` operation computes the sine of a given value. It takes one
    operand of floating point type (i.e., scalar, tensor or vector) and returns one
    result of the same type.  It has no standard attributes.

    Example:

    ```mlir
    // Scalar sine value.
    %a = math.sin %b : f64
    ```
  }];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// SinhOp
//===----------------------------------------------------------------------===//

def Math_SinhOp : Math_FloatUnaryOp<"sinh"> {
  let summary = "hyperbolic sine of the specified value";
  let description = [{
    The `sinh` operation computes the hyperbolic sine. It takes one operand
    of floating point type (i.e., scalar, tensor or vector) and returns one
    result of the same type. It has no standard attributes.

    Example:

    ```mlir
    // Scalar hyperbolic sine value.
    %a = math.sinh %b : f64
    ```
  }];
  let hasFolder = 1;
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
    The `erf` operation computes the error function. It takes one operand of
    floating point type (i.e., scalar, tensor or vector) and returns one result of
    the same type. It has no standard attributes.

    Example:

    ```mlir
    // Scalar error function value.
    %a = math.erf %b : f64
    ```
  }];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// ErfcOp
//===----------------------------------------------------------------------===//

def Math_ErfcOp : Math_FloatUnaryOp<"erfc"> {
  let summary = "complementary error function of the specified value";
  let description = [{

    The `erfc` operation computes the complementary error function, defined as
    1-erf(x). This function is part of libm and is needed for accuracy, since
    simply calculating 1-erf(x) when x is close to 1 will give inaccurate results.
    It takes one operand of floating point type (i.e., scalar,
    tensor or vector) and returns one result of the same type. It has no
    standard attributes.

    Example:

    ```mlir
    // Scalar error function value.
    %a = math.erfc %b : f64
    ```
  }];
  let hasFolder = 1;
}


//===----------------------------------------------------------------------===//
// ExpOp
//===----------------------------------------------------------------------===//

def Math_ExpOp : Math_FloatUnaryOp<"exp"> {
  let summary = "base-e exponential of the specified value";
  let description = [{
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
    The `floor` operation computes the floor of a given value. It takes one
    operand of floating point type (i.e., scalar, tensor or vector) and returns one
    result of the same type.  It has no standard attributes.

    Example:

    ```mlir
    // Scalar floor value.
    %a = math.floor %b : f64
    ```
  }];

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// FmaOp
//===----------------------------------------------------------------------===//

def Math_FmaOp : Math_FloatTernaryOp<"fma"> {
  let summary = "floating point fused multipy-add operation";
  let description = [{
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
// IPowIOp
//===----------------------------------------------------------------------===//

def Math_IPowIOp : Math_IntegerBinaryOp<"ipowi"> {
  let summary = "signed integer raised to the power of operation";
  let description = [{
    The `ipowi` operation takes two operands of integer type (i.e., scalar,
    tensor or vector) and returns one result of the same type. Operands
    must have the same type.

    Example:

    ```mlir
    // Scalar signed integer exponentiation.
    %a = math.ipowi %b, %c : i32
    ```
  }];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// IsFiniteOp
//===----------------------------------------------------------------------===//

def Math_IsFiniteOp : Math_FloatClassificationOp<"isfinite"> {
  let summary = "returns true if the operand classifies as finite";
  let description = [{
    Determines if the given floating-point number has finite value i.e. it
    is normal, subnormal or zero, but not infinite or NaN.

    Example:

    ```mlir
    %f = math.isfinite %a : f32
    ```
  }];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// IsInfOp
//===----------------------------------------------------------------------===//

def Math_IsInfOp : Math_FloatClassificationOp<"isinf"> {
  let summary = "returns true if the operand classifies as infinite";
  let description = [{
    Determines if the given floating-point number is positive or negative
    infinity.

    Example:

    ```mlir
    %f = math.isinf %a : f32
    ```
  }];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// IsNaNOp
//===----------------------------------------------------------------------===//

def Math_IsNaNOp : Math_FloatClassificationOp<"isnan"> {
  let summary = "returns true if the operand classifies as NaN";
  let description = [{
    Determines if the given floating-point number is a not-a-number (NaN)
    value.

    Example:

    ```mlir
    %f = math.isnan %a : f32
    ```
  }];
  let hasFolder = 1;
}


//===----------------------------------------------------------------------===//
// IsNormalOp
//===----------------------------------------------------------------------===//

def Math_IsNormalOp : Math_FloatClassificationOp<"isnormal"> {
  let summary = "returns true if the operand classifies as normal";
  let description = [{
    Determines if the given floating-point number is normal, i.e. is neither
    zero, subnormal, infinite, nor NaN.

    Example:

    ```mlir
    %f = math.isnormal %a : f32
    ```
  }];
  let hasFolder = 1;
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
// RoundEvenOp
//===----------------------------------------------------------------------===//

def Math_RoundEvenOp : Math_FloatUnaryOp<"roundeven"> {
  let summary = "round of the specified value with halfway cases to even";
  let description = [{
    The `roundeven` operation returns the operand rounded to the nearest integer
    value in floating-point format. It takes one operand of floating point type
    (i.e., scalar, tensor or vector) and produces one result of the same type.  The
    operation rounds the argument to the nearest integer value in floating-point
    format, rounding halfway cases to even, regardless of the current
    rounding direction.

    Example:

    ```mlir
    // Scalar round operation.
    %a = math.roundeven %b : f64
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
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// TruncOp
//===----------------------------------------------------------------------===//

def Math_TruncOp : Math_FloatUnaryOp<"trunc"> {
  let summary = "trunc of the specified value";
  let description = [{
    The `trunc` operation returns the operand rounded to the nearest integer
    value in floating-point format. It takes one operand of floating point type
    (i.e., scalar, tensor or vector) and produces one result of the same type.
    The operation always rounds to the nearest integer not larger in magnitude
    than the operand, regardless of the current rounding direction.

    Example:

    ```mlir
    // Scalar trunc operation.
    %a = math.trunc %b : f64
    ```
  }];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// FPowIOp
//===----------------------------------------------------------------------===//

def Math_FPowIOp : Math_Op<"fpowi",
    [SameOperandsAndResultShape, AllTypesMatch<["lhs", "result"]>,
     DeclareOpInterfaceMethods<ArithFastMathInterface>]> {
  let summary = "floating point raised to the signed integer power";
  let description = [{
    The `fpowi` operation takes a `base` operand of floating point type
    (i.e. scalar, tensor or vector) and a `power` operand of integer type
    (also scalar, tensor or vector) and returns one result of the same type
    as `base`. The result is `base` raised to the power of `power`.
    The operation is elementwise for non-scalars, e.g.:

    ```mlir
    %v = math.fpowi %base, %power : vector<2xf32>, vector<2xi32
    ```

    The result is a vector of:

    ```
    [<math.fpowi %base[0], %power[0]>, <math.fpowi %base[1], %power[1]>]
    ```

    Example:

    ```mlir
    // Scalar exponentiation.
    %a = math.fpowi %base, %power : f64, i32
    ```
  }];

  let arguments = (ins FloatLike:$lhs, SignlessIntegerOrIndexLike:$rhs,
      DefaultValuedAttr<Arith_FastMathAttr,
                        "::mlir::arith::FastMathFlags::none">:$fastmath);
  let results = (outs FloatLike:$result);
  let assemblyFormat = [{ $lhs `,` $rhs (`fastmath` `` $fastmath^)?
                          attr-dict `:` type($lhs) `,` type($rhs) }];

  // TODO: add a constant folder using pow[f] for cases, when
  //       the power argument is exactly representable in floating
  //       point type of the base.
}

#endif // MATH_OPS
