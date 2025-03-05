//===- SPIRVCLOps.td - OpenCL extended insts spec file ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the op definition spec of OpenCL extension ops.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SPIRV_IR_CL_OPS
#define MLIR_DIALECT_SPIRV_IR_CL_OPS

include "mlir/Dialect/SPIRV/IR/SPIRVBase.td"

//===----------------------------------------------------------------------===//
// SPIR-V OpenCL opcode specification.
//===----------------------------------------------------------------------===//

// Base class for all OpenCL ops.
class SPIRV_CLOp<string mnemonic, int opcode, list<Trait> traits = []> :
  SPIRV_ExtInstOp<mnemonic, "CL", "OpenCL.std", opcode, traits> {

  let availability = [
    MinVersion<SPIRV_V_1_0>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[]>,
    Capability<[SPIRV_C_Kernel]>
  ];
}

// Base class for OpenCL unary ops.
class SPIRV_CLUnaryOp<string mnemonic, int opcode, Type resultType,
                      Type operandType, list<Trait> traits = []> :
  SPIRV_CLOp<mnemonic, opcode, !listconcat([Pure], traits)> {

  let arguments = (ins
    SPIRV_ScalarOrVectorOf<operandType>:$operand
  );

  let results = (outs
    SPIRV_ScalarOrVectorOf<resultType>:$result
  );

  let assemblyFormat = "$operand `:` type($operand) attr-dict";

  let hasVerifier = 0;
}

// Base class for OpenCL Unary arithmetic ops where return type matches
// the operand type.
class SPIRV_CLUnaryArithmeticOp<string mnemonic, int opcode, Type type,
                                list<Trait> traits = []> :
  SPIRV_CLUnaryOp<mnemonic, opcode, type, type,
                 traits # [SameOperandsAndResultType]>;

// Base class for OpenCL binary ops.
class SPIRV_CLBinaryOp<string mnemonic, int opcode, Type resultType,
                       Type operandType, list<Trait> traits = []> :
  SPIRV_CLOp<mnemonic, opcode, !listconcat([Pure], traits)> {

  let arguments = (ins
    SPIRV_ScalarOrVectorOf<operandType>:$lhs,
    SPIRV_ScalarOrVectorOf<operandType>:$rhs
  );

  let results = (outs
    SPIRV_ScalarOrVectorOf<resultType>:$result
  );

  let hasVerifier = 0;
}

// Base class for OpenCL Binary arithmetic ops where operand types and
// return type matches.
class SPIRV_CLBinaryArithmeticOp<string mnemonic, int opcode, Type type,
                                 list<Trait> traits = []> :
  SPIRV_CLBinaryOp<mnemonic, opcode, type, type,
                  traits # [SameOperandsAndResultType]> {
  let assemblyFormat = "operands attr-dict `:` type($result)";
}

// Base class for OpenCL binary ops.
class SPIRV_CLTernaryOp<string mnemonic, int opcode, Type resultType,
                      Type operandType, list<Trait> traits = []> :
  SPIRV_CLOp<mnemonic, opcode, !listconcat([Pure], traits)> {

  let arguments = (ins
    SPIRV_ScalarOrVectorOf<operandType>:$x,
    SPIRV_ScalarOrVectorOf<operandType>:$y,
    SPIRV_ScalarOrVectorOf<operandType>:$z
  );

  let results = (outs
    SPIRV_ScalarOrVectorOf<resultType>:$result
  );

  let hasVerifier = 0;
}

// Base class for OpenCL Ternary arithmetic ops where operand types and
// return type matches.
class SPIRV_CLTernaryArithmeticOp<string mnemonic, int opcode, Type type,
                                  list<Trait> traits = []> :
  SPIRV_CLTernaryOp<mnemonic, opcode, type, type,
                  traits # [SameOperandsAndResultType]> {
  let assemblyFormat = "operands attr-dict `:` type($result)";
}



// -----

def SPIRV_CLAtan2Op : SPIRV_CLBinaryArithmeticOp<"atan2", 7, SPIRV_Float > {
  let summary = "Compute the arc tangent of y / x.";

  let description = [{
    Result is an angle in radians.

    Result Type, y and x must be floating-point or vector(2,3,4,8,16) of
    floating-point values.

    All of the operands, including the Result Type operand, must be of the
    same type.

    #### Example:

    ```mlir
    %2 = spirv.CL.atan2 %0, %1 : f32
    %3 = spirv.CL.atan2 %0, %1 : vector<4xf16>
    ```
  }];
}

// -----

def SPIRV_CLAcosOp : SPIRV_CLUnaryArithmeticOp<"acos", 0, SPIRV_Float > {
  let summary = "Compute the arc cosine of x.";

  let description = [{
    Result is an angle in radians.

    Result Type and x must be floating-point or vector(2,3,4,8,16) of
    floating-point values.

    All of the operands, including the Result Type operand, must be of the
    same type.

    #### Example:

    ```mlir
    %2 = spirv.CL.acos %0 : f32
    %3 = spirv.CL.acos %1 : vector<4xf16>
    ```
  }];
}

// -----

def SPIRV_CLAcoshOp : SPIRV_CLUnaryArithmeticOp<"acosh", 1, SPIRV_Float > {
  let summary = "Compute the inverse hyperbolic cosine of x .";

  let description = [{
    Result is an angle in radians.

    Result Type and x must be floating-point or vector(2,3,4,8,16) of
    floating-point values.

    All of the operands, including the Result Type operand, must be of the
    same type.

    #### Example:

    ```mlir
    %2 = spirv.CL.acosh %0 : f32
    %3 = spirv.CL.acosh %1 : vector<4xf16>
    ```
  }];
}

// -----

def SPIRV_CLAsinOp : SPIRV_CLUnaryArithmeticOp<"asin", 3, SPIRV_Float > {
  let summary = "Compute the arc sine of x.";

  let description = [{
    Result is an angle in radians.

    Result Type and x must be floating-point or vector(2,3,4,8,16) of
    floating-point values.

    All of the operands, including the Result Type operand, must be of the
    same type.

    #### Example:

    ```mlir
    %2 = spirv.CL.asin %0 : f32
    %3 = spirv.CL.asin %1 : vector<4xf16>
    ```
  }];
}

// -----

def SPIRV_CLAsinhOp : SPIRV_CLUnaryArithmeticOp<"asinh", 4, SPIRV_Float > {
  let summary = "Compute the inverse hyperbolic sine of x.";

  let description = [{
    Result is an angle in radians.

    Result Type and x must be floating-point or vector(2,3,4,8,16) of
    floating-point values.

    All of the operands, including the Result Type operand, must be of the
    same type.

    #### Example:

    ```mlir
    %2 = spirv.CL.asinh %0 : f32
    %3 = spirv.CL.asinh %1 : vector<4xf16>
    ```
  }];
}

// -----

def SPIRV_CLAtanOp : SPIRV_CLUnaryArithmeticOp<"atan", 6, SPIRV_Float > {
  let summary = "Compute the arc tangent of x.";

  let description = [{
    Result is an angle in radians.

    Result Type and x must be floating-point or vector(2,3,4,8,16) of
    floating-point values.

    All of the operands, including the Result Type operand, must be of the
    same type.

    #### Example:

    ```mlir
    %2 = spirv.CL.atan %0 : f32
    %3 = spirv.CL.atan %1 : vector<4xf16>
    ```
  }];
}

// -----

def SPIRV_CLAtanhOp : SPIRV_CLUnaryArithmeticOp<"atanh", 8, SPIRV_Float > {
  let summary = "Compute the hyperbolic arc tangent of x.";

  let description = [{
    Result is an angle in radians.

    Result Type and x must be floating-point or vector(2,3,4,8,16) of
    floating-point values.

    All of the operands, including the Result Type operand, must be of the
    same type.

    #### Example:

    ```mlir
    %2 = spirv.CL.atanh %0 : f32
    %3 = spirv.CL.atanh %1 : vector<4xf16>
    ```
  }];
}

// -----

def SPIRV_CLCeilOp : SPIRV_CLUnaryArithmeticOp<"ceil", 12, SPIRV_Float> {
  let summary = [{
    Round x to integral value using the round to positive infinity rounding
    mode.
  }];

  let description = [{
    Result Type and x must be floating-point or vector(2,3,4,8,16) of
    floating-point values.

    All of the operands, including the Result Type operand, must be of the
    same type.

    #### Example:

    ```mlir
    %2 = spirv.CL.ceil %0 : f32
    %3 = spirv.CL.ceil %1 : vector<3xf16>
    ```
  }];
}

// -----

def SPIRV_CLCosOp : SPIRV_CLUnaryArithmeticOp<"cos", 14, SPIRV_Float> {
  let summary = "Compute the cosine of x radians.";

  let description = [{
    Result Type and x must be floating-point or vector(2,3,4,8,16) of
    floating-point values.

    All of the operands, including the Result Type operand, must be of the
    same type.

    #### Example:

    ```mlir
    %2 = spirv.CL.cos %0 : f32
    %3 = spirv.CL.cos %1 : vector<3xf16>
    ```
  }];
}

// -----

def SPIRV_CLCoshOp : SPIRV_CLUnaryArithmeticOp<"cosh", 15, SPIRV_Float > {
  let summary = "Compute the hyperbolic cosine of x radians.";

  let description = [{
    Result Type and x must be floating-point or vector(2,3,4,8,16) of
    floating-point values.

    All of the operands, including the Result Type operand, must be of the
    same type.

    #### Example:

    ```mlir
    %2 = spirv.CL.cosh %0 : f32
    %3 = spirv.CL.cosh %1 : vector<4xf16>
    ```
  }];
}

// -----

def SPIRV_CLErfOp : SPIRV_CLUnaryArithmeticOp<"erf", 18, SPIRV_Float> {
  let summary = [{
    Error function of x encountered in integrating the normal distribution.
  }];

  let description = [{
    Result Type and x must be floating-point or vector(2,3,4,8,16) of
    floating-point values.

    All of the operands, including the Result Type operand, must be of the
    same type.

    #### Example:

    ```mlir
    %2 = spirv.CL.erf %0 : f32
    %3 = spirv.CL.erf %1 : vector<3xf16>
    ```
  }];
}

// -----

def SPIRV_CLExpOp : SPIRV_CLUnaryArithmeticOp<"exp", 19, SPIRV_Float> {
  let summary = "Exponentiation of Operand 1";

  let description = [{
    Compute the base-e exponential of x. (i.e. ex)

    Result Type and x must be floating-point or vector(2,3,4,8,16) of
    floating-point values.

    All of the operands, including the Result Type operand,
    must be of the same type.

    #### Example:

    ```mlir
    %2 = spirv.CL.exp %0 : f32
    %3 = spirv.CL.exp %1 : vector<3xf16>
    ```
  }];
}

// -----

def SPIRV_CLFAbsOp : SPIRV_CLUnaryArithmeticOp<"fabs", 23, SPIRV_Float> {
  let summary = "Absolute value of operand";

  let description = [{
    Compute the absolute value of x.

    Result Type and x must be floating-point or vector(2,3,4,8,16) of
    floating-point values.

    All of the operands, including the Result Type operand,
    must be of the same type.

    #### Example:

    ```mlir
    %2 = spirv.CL.fabs %0 : f32
    %3 = spirv.CL.fabs %1 : vector<3xf16>
    ```
  }];
}

// -----

def SPIRV_CLFMaxOp : SPIRV_CLBinaryArithmeticOp<"fmax", 27, SPIRV_Float> {
  let summary = "Return maximum of two floating-point operands";

  let description = [{
    Returns y if x < y, otherwise it returns x. If one argument is a NaN,
    Fmax returns the other argument. If both arguments are NaNs, Fmax returns a NaN.

    Result Type, x and y must be floating-point or vector(2,3,4,8,16)
    of floating-point values.

    All of the operands, including the Result Type operand,
    must be of the same type.

    #### Example:

    ```mlir
    %2 = spirv.CL.fmax %0, %1 : f32
    %3 = spirv.CL.fmax %0, %1 : vector<3xf16>
    ```
  }];
}

// -----

def SPIRV_CLFMinOp : SPIRV_CLBinaryArithmeticOp<"fmin", 28, SPIRV_Float> {
  let summary = "Return minimum of two floating-point operands";

  let description = [{
    Returns y if y < x, otherwise it returns x. If one argument is a NaN, Fmin returns the other argument.
    If both arguments are NaNs, Fmin returns a NaN.

    Result Type,x and y must be floating-point or vector(2,3,4,8,16) of floating-point values.

    All of the operands, including the Result Type operand, must be of the same type.

    #### Example:

    ```mlir
    %2 = spirv.CL.fmin %0, %1 : f32
    %3 = spirv.CL.fmin %0, %1 : vector<3xf16>
    ```
  }];
}

// -----

def SPIRV_CLFloorOp : SPIRV_CLUnaryArithmeticOp<"floor", 25, SPIRV_Float> {
  let summary = [{
    Round x to the integral value using the round to negative infinity
    rounding mode.
  }];

  let description = [{
    Result Type and x must be floating-point or vector(2,3,4,8,16) of
    floating-point values.

    All of the operands, including the Result Type operand, must be of the
    same type.

    #### Example:

    ```mlir
    %2 = spirv.CL.floor %0 : f32
    %3 = spirv.CL.floor %1 : vector<3xf16>
    ```
  }];
}

// -----

def SPIRV_CLFmaOp : SPIRV_CLTernaryArithmeticOp<"fma", 26, SPIRV_Float> {
  let summary = [{
    Compute the correctly rounded floating-point representation of the sum
    of c with the infinitely precise product of a and b. Rounding of
    intermediate products shall not occur. Edge case results are per the
    IEEE 754-2008 standard.
  }];

  let description = [{
    Result Type, a, b and c must be floating-point or vector(2,3,4,8,16) of
    floating-point values.

    All of the operands, including the Result Type operand, must be of the
    same type.

    #### Example:

    ```mlir
    %0 = spirv.CL.fma %a, %b, %c : f32
    %1 = spirv.CL.fma %a, %b, %c : vector<3xf16>
    ```
  }];
}

// -----

def SPIRV_CLLogOp : SPIRV_CLUnaryArithmeticOp<"log", 37, SPIRV_Float> {
  let summary = "Compute the natural logarithm of x.";

  let description = [{
    Result Type and x must be floating-point or vector(2,3,4,8,16) of
    floating-point values.

    All of the operands, including the Result Type operand, must be of the
    same type.

    #### Example:

    ```mlir
    %2 = spirv.CL.log %0 : f32
    %3 = spirv.CL.log %1 : vector<3xf16>
    ```
  }];
}

// -----

def SPIRV_CLMixOp : SPIRV_CLTernaryArithmeticOp<"mix", 99, SPIRV_Float> {
  let summary = "Returns the linear blend of x & y implemented as: x + (y - x) * a";

  let description = [{
    Result Type, x, y and a must be floating-point or vector(2,3,4,8,16) of
    floating-point values.

    All of the operands, including the Result Type operand, must be of the
    same type.

    Note: This instruction can be implemented using contractions such as mad
    or fma.

    #### Example:

    ```mlir
    %0 = spirv.CL.mix %a, %b, %c : f32
    %1 = spirv.CL.mix %a, %b, %c : vector<3xf16>
    ```
  }];
}

// -----

def SPIRV_CLPowOp : SPIRV_CLBinaryArithmeticOp<"pow", 48, SPIRV_Float> {
  let summary = "Compute x to the power y.";

  let description = [{
    Result Type, x and y must be floating-point or vector(2,3,4,8,16) of
    floating-point values.

    All of the operands, including the Result Type operand, must be of the
    same type.

    #### Example:

    ```mlir
    %2 = spirv.CL.pow %0, %1 : f32
    %3 = spirv.CL.pow %0, %1 : vector<3xf16>
    ```
  }];
}

// -----

def SPIRV_CLRintOp : SPIRV_CLUnaryArithmeticOp<"rint", 53, SPIRV_Float> {
  let summary = [{
    Round x to integral value (using round to nearest even rounding mode) in
    floating-point format.
  }];

  let description = [{
    Result Type and x must be floating-point or vector(2,3,4,8,16) of
    floating-point values.

    All of the operands, including the Result Type operand, must be of the
    same type.

    #### Example:

    ```mlir
    %0 = spirv.CL.rint %0 : f32
    %1 = spirv.CL.rint %1 : vector<3xf16>
    ```
  }];
}

// -----

def SPIRV_CLRoundOp : SPIRV_CLUnaryArithmeticOp<"round", 55, SPIRV_Float> {
  let summary = [{
    Return the integral value nearest to x rounding halfway cases away from
    zero, regardless of the current rounding direction.
  }];

  let description = [{
    Result Type and x must be floating-point or vector(2,3,4,8,16) of
    floating-point values.

    All of the operands, including the Result Type operand, must be of the
    same type.

    #### Example:

    ```mlir
    %2 = spirv.CL.round %0 : f32
    %3 = spirv.CL.round %0 : vector<3xf16>
    ```
  }];
}

// -----

def SPIRV_CLRsqrtOp : SPIRV_CLUnaryArithmeticOp<"rsqrt", 56, SPIRV_Float> {
  let summary = "Compute inverse square root of x.";

  let description = [{
    Result Type and x must be floating-point or vector(2,3,4,8,16) of
    floating-point values.

    All of the operands, including the Result Type operand, must be of the
    same type.

    #### Example:

    ```mlir
    %2 = spirv.CL.rsqrt %0 : f32
    %3 = spirv.CL.rsqrt %1 : vector<3xf16>
    ```
  }];
}

// -----

def SPIRV_CLSAbsOp : SPIRV_CLUnaryArithmeticOp<"s_abs", 141, SPIRV_Integer> {
  let summary = "Absolute value of operand";

  let description = [{
    Returns |x|, where x is treated as signed integer.

    Result Type and x must be integer or vector(2,3,4,8,16) of
    integer values.

    All of the operands, including the Result Type operand,
    must be of the same type.

    #### Example:

    ```mlir
    %2 = spirv.CL.s_abs %0 : i32
    %3 = spirv.CL.s_abs %1 : vector<3xi16>
    ```
  }];
}

// -----

def SPIRV_CLSMaxOp : SPIRV_CLBinaryArithmeticOp<"s_max", 156, SPIRV_Integer> {
  let summary = "Return maximum of two signed integer operands";

  let description = [{
    Returns y if x < y, otherwise it returns x, where x and y are treated as signed integers.

    Result Type,x and y must be integer or vector(2,3,4,8,16) of integer values.

    All of the operands, including the Result Type operand, must be of the same type.

    #### Example:

    ```mlir
    %2 = spirv.CL.s_max %0, %1 : i32
    %3 = spirv.CL.s_max %0, %1 : vector<3xi16>
    ```
  }];
}

// -----

def SPIRV_CLSMinOp : SPIRV_CLBinaryArithmeticOp<"s_min", 158, SPIRV_Integer> {
  let summary = "Return minimum of two signed integer operands";

  let description = [{
    Returns y if x < y, otherwise it returns x, where x and y are treated as signed integers.

    Result Type,x and y must be integer or vector(2,3,4,8,16) of integer values.

    All of the operands, including the Result Type operand, must be of the same type.

    #### Example:

    ```mlir
    %2 = spirv.CL.s_min %0, %1 : i32
    %3 = spirv.CL.s_min %0, %1 : vector<3xi16>
    ```
  }];
}

// -----

def SPIRV_CLSinOp : SPIRV_CLUnaryArithmeticOp<"sin", 57, SPIRV_Float> {
  let summary = "Compute sine of x radians.";

  let description = [{
    Result Type and x must be floating-point or vector(2,3,4,8,16) of
    floating-point values.

    All of the operands, including the Result Type operand, must be of the
    same type.

    #### Example:

    ```mlir
    %2 = spirv.CL.sin %0 : f32
    %3 = spirv.CL.sin %1 : vector<3xf16>
    ```
  }];
}

// -----

def SPIRV_CLSinhOp : SPIRV_CLUnaryArithmeticOp<"sinh", 59, SPIRV_Float > {
  let summary = "Compute hyperbolic sine of x radians.";

  let description = [{
    Result Type and x must be floating-point or vector(2,3,4,8,16) of
    floating-point values.

    All of the operands, including the Result Type operand, must be of the
    same type.

    #### Example:

    ```mlir
    %2 = spirv.CL.sinh %0 : f32
    %3 = spirv.CL.sinh %1 : vector<4xf16>
    ```
  }];
}

// -----

def SPIRV_CLSqrtOp : SPIRV_CLUnaryArithmeticOp<"sqrt", 61, SPIRV_Float> {
  let summary = "Compute square root of x.";

  let description = [{
    Result Type and x must be floating-point or vector(2,3,4,8,16) of
    floating-point values.

    All of the operands, including the Result Type operand, must be of the
    same type.

    #### Example:

    ```mlir
    %2 = spirv.CL.sqrt %0 : f32
    %3 = spirv.CL.sqrt %1 : vector<3xf16>
    ```
  }];
}

// -----

def SPIRV_CLTanOp : SPIRV_CLUnaryArithmeticOp<"tan", 62, SPIRV_Float > {
  let summary = "Compute tangent of x radians.";

  let description = [{
    Result Type and x must be floating-point or vector(2,3,4,8,16) of
    floating-point values.

    All of the operands, including the Result Type operand, must be of the
    same type.

    #### Example:

    ```mlir
    %2 = spirv.CL.tan %0 : f32
    %3 = spirv.CL.tan %1 : vector<4xf16>
    ```
  }];
}

// -----

def SPIRV_CLTanhOp : SPIRV_CLUnaryArithmeticOp<"tanh", 63, SPIRV_Float> {
  let summary = "Compute hyperbolic tangent of x radians.";

  let description = [{
    Result Type and x must be floating-point or vector(2,3,4,8,16) of
    floating-point values.

    All of the operands, including the Result Type operand, must be of the
    same type.

    #### Example:

    ```mlir
    %2 = spirv.CL.tanh %0 : f32
    %3 = spirv.CL.tanh %1 : vector<3xf16>
    ```
  }];
}

// -----

def SPIRV_CLUMaxOp : SPIRV_CLBinaryArithmeticOp<"u_max", 157, SPIRV_Integer> {
  let summary = "Return maximum of two unsigned integer operands";

  let description = [{
    Returns y if x < y, otherwise it returns x, where x and y are treated as unsigned integers.

    Result Type,x and y must be integer or vector(2,3,4,8,16) of integer values.

    All of the operands, including the Result Type operand, must be of the same type.

    #### Example:

    ```mlir
    %2 = spirv.CL.u_max %0, %1 : i32
    %3 = spirv.CL.u_max %0, %1 : vector<3xi16>
    ```
  }];
}

// -----

def SPIRV_CLUMinOp : SPIRV_CLBinaryArithmeticOp<"u_min", 159, SPIRV_Integer> {
  let summary = "Return minimum of two unsigned integer operands";

  let description = [{
    Returns y if x < y, otherwise it returns x, where x and y are treated as unsigned integers.

    Result Type,x and y must be integer or vector(2,3,4,8,16) of integer values.

    All of the operands, including the Result Type operand, must be of the same type.

    #### Example:

    ```mlir
    %2 = spirv.CL.u_min %0, %1 : i32
    %3 = spirv.CL.u_min %0, %1 : vector<3xi16>
    ```
  }];
}

// -----

def SPIRV_CLPrintfOp : SPIRV_CLOp<"printf", 184, []> {
  let summary = [{
    The printf extended instruction writes output to an implementation-
    defined stream such as stdout under control of the string pointed to by
    format that specifies how subsequent arguments are converted for output.
  }];

  let description = [{
    printf returns 0 if it was executed successfully and -1 otherwise.

    Result Type must be i32.

    Format must be a pointer(constant) to i8. If there are insufficient 
    arguments for the format, the behavior is undefined. If the format 
    is exhausted while arguments remain, the excess arguments are evaluated
    (as always) but are otherwise ignored. The printf instruction returns 
    when the end of the format string is encountered.

    <!-- End of AutoGen section -->

    #### Example:

    ```mlir
    %0 = spirv.CL.printf %fmt %1, %2  : !spirv.ptr<i8, UniformConstant>, i32, i32 -> i32
    ```
  }];

  let arguments = (ins
    SPIRV_AnyPtr:$format,
    Variadic<SPIRV_Type>:$arguments
  );
  
   let results = (outs
    SPIRV_Integer:$result
  );

  let assemblyFormat = [{
   $format ( $arguments^ )? attr-dict `:`  type($format) ( `,` type($arguments)^ )? `->` type($result)
  }];

  let hasVerifier = 0;
}
#endif // MLIR_DIALECT_SPIRV_IR_CL_OPS
