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

    <!-- End of AutoGen section -->

    ```
    float-scalar-vector-type ::= float-type |
                                 `vector<` integer-literal `x` float-type `>`
    ceil-op ::= ssa-id `=` `spirv.CL.ceil` ssa-use `:`
               float-scalar-vector-type
    ```mlir

    #### Example:

    ```
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

    <!-- End of AutoGen section -->

    ```
    float-scalar-vector-type ::= float-type |
                                 `vector<` integer-literal `x` float-type `>`
    cos-op ::= ssa-id `=` `spirv.CL.cos` ssa-use `:`
               float-scalar-vector-type
    ```mlir

    #### Example:

    ```
    %2 = spirv.CL.cos %0 : f32
    %3 = spirv.CL.cos %1 : vector<3xf16>
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

    <!-- End of AutoGen section -->

    ```
    float-scalar-vector-type ::= float-type |
                                 `vector<` integer-literal `x` float-type `>`
    erf-op ::= ssa-id `=` `spirv.CL.erf` ssa-use `:`
               float-scalar-vector-type
    ```mlir

    #### Example:

    ```
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

    <!-- End of AutoGen section -->
    ```
    float-scalar-vector-type ::= float-type |
                                 `vector<` integer-literal `x` float-type `>`
    exp-op ::= ssa-id `=` `spirv.CL.exp` ssa-use `:`
               float-scalar-vector-type
    ```
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

    <!-- End of AutoGen section -->
    ```
    float-scalar-vector-type ::= float-type |
                                 `vector<` integer-literal `x` float-type `>`
    abs-op ::= ssa-id `=` `spirv.CL.fabs` ssa-use `:`
               float-scalar-vector-type
    ```
    #### Example:

    ```mlir
    %2 = spirv.CL.fabs %0 : f32
    %3 = spirv.CL.fabs %1 : vector<3xf16>
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

    <!-- End of AutoGen section -->

    ```
    float-scalar-vector-type ::= float-type |
                                 `vector<` integer-literal `x` float-type `>`
    floor-op ::= ssa-id `=` `spirv.CL.floor` ssa-use `:`
               float-scalar-vector-type
    ```mlir

    #### Example:

    ```
    %2 = spirv.CL.floor %0 : f32
    %3 = spirv.CL.ceifloorl %1 : vector<3xf16>
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

    <!-- End of AutoGen section -->

    ```
    fma-op ::= ssa-id `=` `spirv.CL.fma` ssa-use, ssa-use, ssa-use `:`
               float-scalar-vector-type
    ```mlir

    ```
    %0 = spirv.CL.fma %a, %b, %c : f32
    %1 = spirv.CL.fma %a, %b, %c : vector<3xf16>
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

    <!-- End of AutoGen section -->
    ```
    float-scalar-vector-type ::= float-type |
                                 `vector<` integer-literal `x` float-type `>`
    fmax-op ::= ssa-id `=` `spirv.CL.fmax` ssa-use `:`
                float-scalar-vector-type
    ```
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


    <!-- End of AutoGen section -->
    ```
    float-scalar-vector-type ::= float-type |
                                 `vector<` integer-literal `x` float-type `>`
    fmin-op ::= ssa-id `=` `spirv.CL.fmin` ssa-use `:`
                float-scalar-vector-type
    ```
    #### Example:

    ```mlir
    %2 = spirv.CL.fmin %0, %1 : f32
    %3 = spirv.CL.fmin %0, %1 : vector<3xf16>
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

    <!-- End of AutoGen section -->

    ```
    float-scalar-vector-type ::= float-type |
                                 `vector<` integer-literal `x` float-type `>`
    log-op ::= ssa-id `=` `spirv.CL.log` ssa-use `:`
               float-scalar-vector-type
    ```mlir

    #### Example:

    ```
    %2 = spirv.CL.log %0 : f32
    %3 = spirv.CL.log %1 : vector<3xf16>
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

    <!-- End of AutoGen section -->

    ```
    restricted-float-scalar-type ::=  `f16` | `f32`
    restricted-float-scalar-vector-type ::=
      restricted-float-scalar-type |
      `vector<` integer-literal `x` restricted-float-scalar-type `>`
    pow-op ::= ssa-id `=` `spirv.CL.pow` ssa-use `:`
               restricted-float-scalar-vector-type
    ```
    #### Example:

    ```mlir
    %2 = spirv.CL.pow %0, %1 : f32
    %3 = spirv.CL.pow %0, %1 : vector<3xf16>
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

    <!-- End of AutoGen section -->

    ```
    float-scalar-vector-type ::= float-type |
                                 `vector<` integer-literal `x` float-type `>`
    round-op ::= ssa-id `=` `spirv.CL.round` ssa-use `:`
               float-scalar-vector-type
    ```
    #### Example:

    ```mlir
    %2 = spirv.CL.round %0 : f32
    %3 = spirv.CL.round %0 : vector<3xf16>
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

    <!-- End of AutoGen section -->

    ```
    float-scalar-vector-type ::= float-type |
                                 `vector<` integer-literal `x` float-type `>`
    rint-op ::= ssa-id `=` `spirv.CL.rint` ssa-use `:`
               float-scalar-vector-type
    ```

    #### Example:

    ```mlir
    %0 = spirv.CL.rint %0 : f32
    %1 = spirv.CL.rint %1 : vector<3xf16>
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

    <!-- End of AutoGen section -->

    ```
    float-scalar-vector-type ::= float-type |
                                 `vector<` integer-literal `x` float-type `>`
    rsqrt-op ::= ssa-id `=` `spirv.CL.rsqrt` ssa-use `:`
               float-scalar-vector-type
    ```mlir

    #### Example:

    ```
    %2 = spirv.CL.rsqrt %0 : f32
    %3 = spirv.CL.rsqrt %1 : vector<3xf16>
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

    <!-- End of AutoGen section -->

    ```
    float-scalar-vector-type ::= float-type |
                                 `vector<` integer-literal `x` float-type `>`
    sin-op ::= ssa-id `=` `spirv.CL.sin` ssa-use `:`
               float-scalar-vector-type
    ```mlir

    #### Example:

    ```
    %2 = spirv.CL.sin %0 : f32
    %3 = spirv.CL.sin %1 : vector<3xf16>
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

    <!-- End of AutoGen section -->

    ```
    float-scalar-vector-type ::= float-type |
                                 `vector<` integer-literal `x` float-type `>`
    sqrt-op ::= ssa-id `=` `spirv.CL.sqrt` ssa-use `:`
               float-scalar-vector-type
    ```mlir

    #### Example:

    ```
    %2 = spirv.CL.sqrt %0 : f32
    %3 = spirv.CL.sqrt %1 : vector<3xf16>
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

    <!-- End of AutoGen section -->

    ```
    float-scalar-vector-type ::= float-type |
                                 `vector<` integer-literal `x` float-type `>`
    tanh-op ::= ssa-id `=` `spirv.CL.tanh` ssa-use `:`
               float-scalar-vector-type
    ```mlir

    #### Example:

    ```
    %2 = spirv.CL.tanh %0 : f32
    %3 = spirv.CL.tanh %1 : vector<3xf16>
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

    <!-- End of AutoGen section -->
    ```
    integer-scalar-vector-type ::= integer-type |
                                   `vector<` integer-literal `x` integer-type `>`
    abs-op ::= ssa-id `=` `spirv.CL.s_abs` ssa-use `:`
               integer-scalar-vector-type
    ```
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

    <!-- End of AutoGen section -->
    ```
    integer-scalar-vector-type ::= integer-type |
                                   `vector<` integer-literal `x` integer-type `>`
    smax-op ::= ssa-id `=` `spirv.CL.s_max` ssa-use `:`
                integer-scalar-vector-type
    ```
    #### Example:
    ```mlir
    %2 = spirv.CL.s_max %0, %1 : i32
    %3 = spirv.CL.s_max %0, %1 : vector<3xi16>
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

    <!-- End of AutoGen section -->
    ```
    integer-scalar-vector-type ::= integer-type |
                                   `vector<` integer-literal `x` integer-type `>`
    umax-op ::= ssa-id `=` `spirv.CL.u_max` ssa-use `:`
                integer-scalar-vector-type
    ```
    #### Example:
    ```mlir
    %2 = spirv.CL.u_max %0, %1 : i32
    %3 = spirv.CL.u_max %0, %1 : vector<3xi16>
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

    <!-- End of AutoGen section -->
    ```
    integer-scalar-vector-type ::= integer-type |
                                   `vector<` integer-literal `x` integer-type `>`
    smin-op ::= ssa-id `=` `spirv.CL.s_min` ssa-use `:`
                integer-scalar-vector-type
    ```
    #### Example:
    ```mlir
    %2 = spirv.CL.s_min %0, %1 : i32
    %3 = spirv.CL.s_min %0, %1 : vector<3xi16>
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

    <!-- End of AutoGen section -->
    ```
    integer-scalar-vector-type ::= integer-type |
                                   `vector<` integer-literal `x` integer-type `>`
    umin-op ::= ssa-id `=` `spirv.CL.u_min` ssa-use `:`
                integer-scalar-vector-type
    ```
    #### Example:
    ```mlir
    %2 = spirv.CL.u_min %0, %1 : i32
    %3 = spirv.CL.u_min %0, %1 : vector<3xi16>
    ```
  }];
}

#endif // MLIR_DIALECT_SPIRV_IR_CL_OPS
