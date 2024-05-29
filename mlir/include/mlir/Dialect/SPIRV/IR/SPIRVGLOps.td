//===- SPIRVGLOps.td - GLSL extended insts spec file -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the op definition spec of GLSL extension ops.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SPIRV_IR_GL_OPS
#define MLIR_DIALECT_SPIRV_IR_GL_OPS

include "mlir/Dialect/SPIRV/IR/SPIRVBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// SPIR-V GLSL 4.50 opcode specification.
//===----------------------------------------------------------------------===//

// Base class for all OpenGL ops.
class SPIRV_GLOp<string mnemonic, int opcode, list<Trait> traits = []> :
  SPIRV_ExtInstOp<mnemonic, "GL", "GLSL.std.450", opcode, traits> {

  let availability = [
    MinVersion<SPIRV_V_1_0>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[]>,
    Capability<[SPIRV_C_Shader]>
  ];
}

// Base class for GL unary ops.
class SPIRV_GLUnaryOp<string mnemonic, int opcode, Type resultType,
                      Type operandType, list<Trait> traits = []> :
  SPIRV_GLOp<mnemonic, opcode, !listconcat([Pure], traits)> {

  let arguments = (ins
    SPIRV_ScalarOrVectorOf<operandType>:$operand
  );

  let results = (outs
    SPIRV_ScalarOrVectorOf<resultType>:$result
  );

  let hasVerifier = 0;
}

// Base class for GL Unary arithmetic ops where return type matches
// the operand type.
class SPIRV_GLUnaryArithmeticOp<string mnemonic, int opcode, Type type,
                                list<Trait> traits = []> :
  SPIRV_GLUnaryOp<mnemonic, opcode, type, type,
                  traits # [SameOperandsAndResultType]> {
  let assemblyFormat = "$operand `:` type($operand) attr-dict";
}

// Base class for GL binary ops.
class SPIRV_GLBinaryOp<string mnemonic, int opcode, Type resultType,
                        Type operandType, list<Trait> traits = []> :
  SPIRV_GLOp<mnemonic, opcode, !listconcat([Pure], traits)> {

  let arguments = (ins
    SPIRV_ScalarOrVectorOf<operandType>:$lhs,
    SPIRV_ScalarOrVectorOf<operandType>:$rhs
  );

  let results = (outs
    SPIRV_ScalarOrVectorOf<resultType>:$result
  );

  let hasVerifier = 0;
}

// Base class for GL Binary arithmetic ops where operand types and
// return type matches.
class SPIRV_GLBinaryArithmeticOp<string mnemonic, int opcode, Type type,
                                 list<Trait> traits = []> :
  SPIRV_GLBinaryOp<mnemonic, opcode, type, type,
                   traits # [SameOperandsAndResultType]> {
  let assemblyFormat = "operands attr-dict `:` type($result)";
}

// Base class for GL ternary ops.
class SPIRV_GLTernaryArithmeticOp<string mnemonic, int opcode, Type type,
                        list<Trait> traits = []> :
  SPIRV_GLOp<mnemonic, opcode, !listconcat([Pure], traits)> {

  let arguments = (ins
    SPIRV_ScalarOrVectorOf<type>:$x,
    SPIRV_ScalarOrVectorOf<type>:$y,
    SPIRV_ScalarOrVectorOf<type>:$z
  );

  let results = (outs
    SPIRV_ScalarOrVectorOf<type>:$result
  );

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 0;
}

// -----

def SPIRV_GLFAbsOp : SPIRV_GLUnaryArithmeticOp<"FAbs", 4, SPIRV_Float> {
  let summary = "Absolute value of operand";

  let description = [{
    Result is x if x >= 0; otherwise result is -x.

    The operand x must be a scalar or vector whose component type is
    floating-point.

    Result Type and the type of x must be the same type. Results are computed
    per component.

    #### Example:

    ```mlir
    %2 = spirv.GL.FAbs %0 : f32
    %3 = spirv.GL.FAbs %1 : vector<3xf16>
    ```
  }];
}

// -----

def SPIRV_GLSAbsOp : SPIRV_GLUnaryArithmeticOp<"SAbs", 5, SPIRV_Integer> {
  let summary = "Absolute value of operand";

  let description = [{
    Result is x if x ≥ 0; otherwise result is -x, where x is interpreted as a
    signed integer.

    Result Type and the type of x must both be integer scalar or integer vector
    types. Result Type and operand types must have the same number of components
    with the same component width. Results are computed per component.

    #### Example:

    ```mlir
    %2 = spirv.GL.SAbs %0 : i32
    %3 = spirv.GL.SAbs %1 : vector<3xi16>
    ```
  }];
}

// -----

def SPIRV_GLCeilOp : SPIRV_GLUnaryArithmeticOp<"Ceil", 9, SPIRV_Float> {
  let summary = "Rounds up to the next whole number";

  let description = [{
    Result is the value equal to the nearest whole number that is greater than
    or equal to x.

    The operand x must be a scalar or vector whose component type is
    floating-point.

    Result Type and the type of x must be the same type. Results are computed
    per component.

    #### Example:

    ```mlir
    %2 = spirv.GL.Ceil %0 : f32
    %3 = spirv.GL.Ceil %1 : vector<3xf16>
    ```
  }];
}

// -----

def SPIRV_GLCosOp : SPIRV_GLUnaryArithmeticOp<"Cos", 14, SPIRV_Float16or32> {
  let summary = "Cosine of operand in radians";

  let description = [{
    The standard trigonometric cosine of x radians.

    The operand x must be a scalar or vector whose component type is 16-bit or
    32-bit floating-point.

    Result Type and the type of x must be the same type. Results are computed
    per component.

    #### Example:

    ```mlir
    %2 = spirv.GL.Cos %0 : f32
    %3 = spirv.GL.Cos %1 : vector<3xf16>
    ```
  }];
}

// -----

def SPIRV_GLSinOp : SPIRV_GLUnaryArithmeticOp<"Sin", 13, SPIRV_Float16or32> {
  let summary = "Sine of operand in radians";

  let description = [{
    The standard trigonometric sine of x radians.

    The operand x must be a scalar or vector whose component type is 16-bit or
    32-bit floating-point.

    Result Type and the type of x must be the same type. Results are computed
    per component.

    #### Example:

    ```mlir
    %2 = spirv.GL.Sin %0 : f32
    %3 = spirv.GL.Sin %1 : vector<3xf16>
    ```
  }];
}

// -----

def SPIRV_GLTanOp : SPIRV_GLUnaryArithmeticOp<"Tan", 15, SPIRV_Float16or32> {
  let summary = "Tangent of operand in radians";

  let description = [{
    The standard trigonometric tangent of x radians.

    The operand x must be a scalar or vector whose component type is 16-bit or
    32-bit floating-point.

    Result Type and the type of x must be the same type. Results are computed
    per component.

    #### Example:

    ```mlir
    %2 = spirv.GL.Tan %0 : f32
    %3 = spirv.GL.Tan %1 : vector<3xf16>
    ```
  }];
}

// -----

def SPIRV_GLAsinOp : SPIRV_GLUnaryArithmeticOp<"Asin", 16, SPIRV_Float16or32> {
  let summary = "Arc Sine of operand in radians";

  let description = [{
    The standard trigonometric arc sine of x radians.

    Result is an angle, in radians, whose sine is x. The range of result values
    is [-π / 2, π / 2]. Result is undefined if abs x > 1.

    The operand x must be a scalar or vector whose component type is 16-bit or
    32-bit floating-point.

    Result Type and the type of x must be the same type. Results are computed
    per component.

    #### Example:

    ```mlir
    %2 = spirv.GL.Asin %0 : f32
    %3 = spirv.GL.Asin %1 : vector<3xf16>
    ```
  }];
}

// -----

def SPIRV_GLAcosOp : SPIRV_GLUnaryArithmeticOp<"Acos", 17, SPIRV_Float16or32> {
  let summary = "Arc Cosine of operand in radians";

  let description = [{
    The standard trigonometric arc cosine of x radians.

    Result is an angle, in radians, whose cosine is x. The range of result
    values is [0, π]. Result is undefined if abs x > 1.

    The operand x must be a scalar or vector whose component type is 16-bit or
    32-bit floating-point.

    Result Type and the type of x must be the same type. Results are computed
    per component.

    #### Example:

    ```mlir
    %2 = spirv.GL.Acos %0 : f32
    %3 = spirv.GL.Acos %1 : vector<3xf16>
    ```
  }];
}

// -----

def SPIRV_GLAtanOp : SPIRV_GLUnaryArithmeticOp<"Atan", 18, SPIRV_Float16or32> {
  let summary = "Arc Tangent of operand in radians";

  let description = [{
    The standard trigonometric arc tangent of x radians.

    Result is an angle, in radians, whose tangent is y_over_x. The range of
    result values is [-π / 2, π / 2].

    The operand x must be a scalar or vector whose component type is 16-bit or
    32-bit floating-point.

    Result Type and the type of x must be the same type. Results are computed
    per component.

    #### Example:

    ```mlir
    %2 = spirv.GL.Atan %0 : f32
    %3 = spirv.GL.Atan %1 : vector<3xf16>
    ```
  }];
}

// -----

def SPIRV_GLExpOp : SPIRV_GLUnaryArithmeticOp<"Exp", 27, SPIRV_Float16or32> {
  let summary = "Exponentiation of Operand 1";

  let description = [{
    Result is the natural exponentiation of x; e^x.

    The operand x must be a scalar or vector whose component type is
    16-bit or 32-bit floating-point.

    Result Type and the type of x must be the same type. Results are
    computed per component.";

    #### Example:

    ```mlir
    %2 = spirv.GL.Exp %0 : f32
    %3 = spirv.GL.Exp %1 : vector<3xf16>
    ```
  }];
}

// -----

def SPIRV_GLFloorOp : SPIRV_GLUnaryArithmeticOp<"Floor", 8, SPIRV_Float> {
  let summary = "Rounds down to the next whole number";

  let description = [{
    Result is the value equal to the nearest whole number that is less than or
    equal to x.

    The operand x must be a scalar or vector whose component type is
    floating-point.

    Result Type and the type of x must be the same type. Results are computed
    per component.

    #### Example:

    ```mlir
    %2 = spirv.GL.Floor %0 : f32
    %3 = spirv.GL.Floor %1 : vector<3xf16>
    ```
  }];
}

// -----

def SPIRV_GLRoundOp: SPIRV_GLUnaryArithmeticOp<"Round", 1, SPIRV_Float> {
  let summary = "Rounds to the nearest whole number";

  let description = [{
    Result is the value equal to the nearest whole number to x. The fraction
    0.5 will round in a direction chosen by the implementation, presumably
    the direction that is fastest. This includes the possibility that
    Round x is the same value as RoundEven x for all values of x.

    The operand x must be a scalar or vector whose component type is
    floating-point.

    Result Type and the type of x must be the same type. Results are computed
    per component.

    #### Example:

    ```mlir
    %2 = spirv.GL.Round %0 : f32
    %3 = spirv.GL.Round %1 : vector<3xf16>
    ```
  }];
}

// -----

def SPIRV_GLRoundEvenOp: SPIRV_GLUnaryArithmeticOp<"RoundEven", 2, SPIRV_Float> {
  let summary = "Rounds to the nearest even whole number";

  let description = [{
    Result is the value equal to the nearest whole number to x. A fractional
    part of 0.5 will round toward the nearest even whole number. (Both 3.5 and
    4.5 for x will be 4.0.)

    The operand x must be a scalar or vector whose component type is
    floating-point.

    Result Type and the type of x must be the same type. Results are computed
    per component.

    #### Example:

    ```mlir
    %2 = spirv.GL.RoundEven %0 : f32
    %3 = spirv.GL.RoundEven %1 : vector<3xf16>
    ```
  }];
}

// -----

def SPIRV_GLInverseSqrtOp : SPIRV_GLUnaryArithmeticOp<"InverseSqrt", 32, SPIRV_Float> {
  let summary = "Reciprocal of sqrt(operand)";

  let description = [{
    Result is the reciprocal of sqrt x. Result is undefined if x <= 0.

    The operand x must be a scalar or vector whose component type is
    floating-point.

    Result Type and the type of x must be the same type. Results are computed
    per component.

    #### Example:

    ```mlir
    %2 = spirv.GL.InverseSqrt %0 : f32
    %3 = spirv.GL.InverseSqrt %1 : vector<3xf16>
    ```
  }];
}

// -----

def SPIRV_GLLogOp : SPIRV_GLUnaryArithmeticOp<"Log", 28, SPIRV_Float16or32> {
  let summary = "Natural logarithm of the operand";

  let description = [{
    Result is the natural logarithm of x, i.e., the value y which satisfies the
    equation x = ey. Result is undefined if x <= 0.

    The operand x must be a scalar or vector whose component type is 16-bit or
    32-bit floating-point.

    Result Type and the type of x must be the same type. Results are computed
    per component.

    #### Example:

    ```mlir
    %2 = spirv.GL.Log %0 : f32
    %3 = spirv.GL.Log %1 : vector<3xf16>
    ```
  }];
}

// -----

def SPIRV_GLFMaxOp : SPIRV_GLBinaryArithmeticOp<"FMax", 40, SPIRV_Float> {
  let summary = "Return maximum of two floating-point operands";

  let description = [{
    Result is y if x < y; otherwise result is x. Which operand is the
    result is undefined if one of the operands is a NaN.

    The operands must all be a scalar or vector whose component type
    is floating-point.

    Result Type and the type of all operands must be the same
    type. Results are computed per component.

    #### Example:

    ```mlir
    %2 = spirv.GL.FMax %0, %1 : f32
    %3 = spirv.GL.FMax %0, %1 : vector<3xf16>
    ```
  }];
}

// -----

def SPIRV_GLUMaxOp : SPIRV_GLBinaryArithmeticOp<"UMax", 41, SPIRV_Integer> {
  let summary = "Return maximum of two unsigned integer operands";

  let description = [{
    Result is y if x < y; otherwise result is x, where x and y are interpreted
    as unsigned integers.

    Result Type and the type of x and y must both be integer scalar or integer
    vector types. Result Type and operand types must have the same number of
    components with the same component width. Results are computed per
    component.

    #### Example:

    ```mlir
    %2 = spirv.GL.UMax %0, %1 : i32
    %3 = spirv.GL.UMax %0, %1 : vector<3xi16>
    ```
  }];
}

// -----

def SPIRV_GLSMaxOp : SPIRV_GLBinaryArithmeticOp<"SMax", 42, SPIRV_Integer> {
  let summary = "Return maximum of two signed integer operands";

  let description = [{
    Result is y if x < y; otherwise result is x, where x and y are interpreted
    as signed integers.

    Result Type and the type of x and y must both be integer scalar or integer
    vector types. Result Type and operand types must have the same number of
    components with the same component width. Results are computed per
    component.

    #### Example:

    ```mlir
    %2 = spirv.GL.SMax %0, %1 : i32
    %3 = spirv.GL.SMax %0, %1 : vector<3xi16>
    ```
  }];
}

// -----

def SPIRV_GLFMinOp : SPIRV_GLBinaryArithmeticOp<"FMin", 37, SPIRV_Float> {
  let summary = "Return minimum of two floating-point operands";

  let description = [{
    Result is y if y < x; otherwise result is x. Which operand is the result is
    undefined if one of the operands is a NaN.

    The operands must all be a scalar or vector whose component type is
    floating-point.

    Result Type and the type of all operands must be the same type. Results are
    computed per component.

    #### Example:

    ```mlir
    %2 = spirv.GL.FMin %0, %1 : f32
    %3 = spirv.GL.FMin %0, %1 : vector<3xf16>
    ```
  }];
}

// -----

def SPIRV_GLUMinOp : SPIRV_GLBinaryArithmeticOp<"UMin", 38, SPIRV_Integer> {
  let summary = "Return minimum of two unsigned integer operands";

  let description = [{
    Result is y if y < x; otherwise result is x, where x and y are interpreted
    as unsigned integers.

    Result Type and the type of x and y must both be integer scalar or integer
    vector types. Result Type and operand types must have the same number of
    components with the same component width. Results are computed per
    component.

    #### Example:

    ```mlir
    %2 = spirv.GL.UMin %0, %1 : i32
    %3 = spirv.GL.UMin %0, %1 : vector<3xi16>
    ```
  }];
}

// -----

def SPIRV_GLSMinOp : SPIRV_GLBinaryArithmeticOp<"SMin", 39, SPIRV_Integer> {
  let summary = "Return minimum of two signed integer operands";

  let description = [{
    Result is y if y < x; otherwise result is x, where x and y are interpreted
    as signed integers.

    Result Type and the type of x and y must both be integer scalar or integer
    vector types. Result Type and operand types must have the same number of
    components with the same component width. Results are computed per
    component.

    #### Example:

    ```mlir
    %2 = spirv.GL.SMin %0, %1 : i32
    %3 = spirv.GL.SMin %0, %1 : vector<3xi16>
    ```
  }];
}

// -----

def SPIRV_GLPowOp : SPIRV_GLBinaryArithmeticOp<"Pow", 26, SPIRV_Float16or32> {
  let summary = "Return x raised to the y power of two operands";

  let description = [{
    Result is x raised to the y power; x^y.

    Result is undefined if x = 0 and y ≤ 0.

    The operand x and y must be a scalar or vector whose component type is
    16-bit or 32-bit floating-point.

    Result Type and the type of all operands must be the same type. Results are
    computed per component.

    #### Example:

    ```mlir
    %2 = spirv.GL.Pow %0, %1 : f32
    %3 = spirv.GL.Pow %0, %1 : vector<3xf16>
    ```
  }];
}

// -----

def SPIRV_GLFSignOp : SPIRV_GLUnaryArithmeticOp<"FSign", 6, SPIRV_Float> {
  let summary = "Returns the sign of the operand";

  let description = [{
    Result is 1.0 if x > 0, 0.0 if x = 0, or -1.0 if x < 0.

    The operand x must be a scalar or vector whose component type is
    floating-point.

    Result Type and the type of x must be the same type. Results are computed
    per component.

    #### Example:

    ```mlir
    %2 = spirv.GL.FSign %0 : f32
    %3 = spirv.GL.FSign %1 : vector<3xf16>
    ```
  }];
}

// -----

def SPIRV_GLSSignOp : SPIRV_GLUnaryArithmeticOp<"SSign", 7, SPIRV_Integer> {
  let summary = "Returns the sign of the operand";

  let description = [{
    Result is 1 if x > 0, 0 if x = 0, or -1 if x < 0, where x is interpreted as
    a signed integer.

    Result Type and the type of x must both be integer scalar or integer vector
    types. Result Type and operand types must have the same number of components
    with the same component width. Results are computed per component.

    #### Example:

    ```mlir
    %2 = spirv.GL.SSign %0 : i32
    %3 = spirv.GL.SSign %1 : vector<3xi16>
    ```
  }];
}

// -----

def SPIRV_GLSqrtOp : SPIRV_GLUnaryArithmeticOp<"Sqrt", 31, SPIRV_Float> {
  let summary = "Returns the square root of the operand";

  let description = [{
    Result is the square root of x. Result is undefined if x < 0.

    The operand x must be a scalar or vector whose component type is
    floating-point.

    Result Type and the type of x must be the same type. Results are computed
    per component.

    #### Example:

    ```mlir
    %2 = spirv.GL.Sqrt %0 : f32
    %3 = spirv.GL.Sqrt %1 : vector<3xf16>
    ```
  }];
}

// -----

def SPIRV_GLSinhOp : SPIRV_GLUnaryArithmeticOp<"Sinh", 19, SPIRV_Float16or32> {
  let summary = "Hyperbolic sine of operand in radians";

  let description = [{
    Hyperbolic sine of x radians.

    The operand x must be a scalar or vector whose component type is 16-bit or
    32-bit floating-point.

    Result Type and the type of x must be the same type. Results are computed
    per component.

    #### Example:

    ```mlir
    %2 = spirv.GL.Sinh %0 : f32
    %3 = spirv.GL.Sinh %1 : vector<3xf16>
    ```
  }];
}

// -----

def SPIRV_GLCoshOp : SPIRV_GLUnaryArithmeticOp<"Cosh", 20, SPIRV_Float16or32> {
  let summary = "Hyperbolic cosine of operand in radians";

  let description = [{
    Hyperbolic cosine of x radians.

    The operand x must be a scalar or vector whose component type is 16-bit or
    32-bit floating-point.

    Result Type and the type of x must be the same type. Results are computed
    per component.

    #### Example:

    ```mlir
    %2 = spirv.GL.Cosh %0 : f32
    %3 = spirv.GL.Cosh %1 : vector<3xf16>
    ```
  }];
}

// -----

def SPIRV_GLTanhOp : SPIRV_GLUnaryArithmeticOp<"Tanh", 21, SPIRV_Float16or32> {
  let summary = "Hyperbolic tangent of operand in radians";

  let description = [{
    Hyperbolic tangent of x radians.

    The operand x must be a scalar or vector whose component type is 16-bit or
    32-bit floating-point.

    Result Type and the type of x must be the same type. Results are computed
    per component.

    #### Example:

    ```mlir
    %2 = spirv.GL.Tanh %0 : f32
    %3 = spirv.GL.Tanh %1 : vector<3xf16>
    ```
  }];
}

// -----

def SPIRV_GLFClampOp : SPIRV_GLTernaryArithmeticOp<"FClamp", 43, SPIRV_Float> {
  let summary = "Clamp x between min and max values.";

  let description = [{
    Result is min(max(x, minVal), maxVal). The resulting value is undefined if
    minVal > maxVal. The semantics used by min() and max() are those of FMin and
    FMax.

    The operands must all be a scalar or vector whose component type is
    floating-point.

    Result Type and the type of all operands must be the same type. Results are
    computed per component.

    <!-- End of AutoGen section -->
    ```
    fclamp-op ::= ssa-id `=` `spirv.GL.FClamp` ssa-use, ssa-use, ssa-use `:`
               float-scalar-vector-type
    ```
    #### Example:

    ```mlir
    %2 = spirv.GL.FClamp %x, %min, %max : f32
    %3 = spirv.GL.FClamp %x, %min, %max : vector<3xf16>
    ```
  }];
}

// -----

def SPIRV_GLUClampOp : SPIRV_GLTernaryArithmeticOp<"UClamp", 44, SPIRV_Integer> {
  let summary = "Clamp x between min and max values.";

  let description = [{
    Result is min(max(x, minVal), maxVal), where x, minVal and maxVal are
    interpreted as unsigned integers. The resulting value is undefined if
    minVal > maxVal.

    Result Type and the type of the operands must both be integer scalar or
    integer vector types. Result Type and operand types must have the same number
    of components with the same component width. Results are computed per
    component.

    <!-- End of AutoGen section -->
    ```
    uclamp-op ::= ssa-id `=` `spirv.GL.UClamp` ssa-use, ssa-use, ssa-use `:`
               unsigned-signless-scalar-vector-type
    ```
    #### Example:

    ```mlir
    %2 = spirv.GL.UClamp %x, %min, %max : i32
    %3 = spirv.GL.UClamp %x, %min, %max : vector<3xui16>
    ```
  }];
}

// -----

def SPIRV_GLSClampOp : SPIRV_GLTernaryArithmeticOp<"SClamp", 45, SPIRV_Integer> {
  let summary = "Clamp x between min and max values.";

  let description = [{
    Result is min(max(x, minVal), maxVal), where x, minVal and maxVal are
    interpreted as signed integers. The resulting value is undefined if
    minVal > maxVal.

    Result Type and the type of the operands must both be integer scalar or
    integer vector types. Result Type and operand types must have the same number
    of components with the same component width. Results are computed per
    component.

    <!-- End of AutoGen section -->
    ```
    uclamp-op ::= ssa-id `=` `spirv.GL.UClamp` ssa-use, ssa-use, ssa-use `:`
               sgined-scalar-vector-type
    ```
    #### Example:

    ```mlir
    %2 = spirv.GL.SClamp %x, %min, %max : si32
    %3 = spirv.GL.SClamp %x, %min, %max : vector<3xsi16>
    ```
  }];
}

// -----

def SPIRV_GLFmaOp : SPIRV_GLTernaryArithmeticOp<"Fma", 50, SPIRV_Float> {
  let summary = "Computes a * b + c.";

  let description = [{
    In uses where this operation is decorated with NoContraction:

    - fma is considered a single operation, whereas the expression a * b + c
      is considered two operations.
    - The precision of fma can differ from the precision of the expression
      a * b + c.
    - fma will be computed with the same precision as any other fma decorated
      with NoContraction, giving invariant results for the same input values
      of a, b, and c.

    Otherwise, in the absence of a NoContraction decoration, there are no
    special constraints on the number of operations or difference in precision
    between fma and the expression a * b +c.

    The operands must all be a scalar or vector whose component type is
    floating-point.

    Result Type and the type of all operands must be the same type. Results
    are computed per component.

    <!-- End of AutoGen section -->
    ```
    fma-op ::= ssa-id `=` `spirv.GL.Fma` ssa-use, ssa-use, ssa-use `:`
               float-scalar-vector-type
    ```
    #### Example:

    ```mlir
    %0 = spirv.GL.Fma %a, %b, %c : f32
    %1 = spirv.GL.Fma %a, %b, %c : vector<3xf16>
    ```
  }];
}

// ----

def SPIRV_GLFrexpStructOp : SPIRV_GLOp<"FrexpStruct", 52, [Pure]> {
  let summary = "Splits x into two components such that x = significand * 2^exponent";

  let description = [{
    Result is a structure containing x split into a floating-point significand
    in the range (-1.0, 0.5] or [0.5, 1.0) and an integral exponent of 2, such that:

    x = significand * 2^exponent

    If x is a zero, the exponent is 0.0. If x is an infinity or a NaN, the
    exponent is undefined. If x is 0.0, the significand is 0.0. If x is -0.0,
    the significand is -0.0

    Result Type must be an OpTypeStruct with two members. Member 0 must have
    the same type as the type of x. Member 0 holds the significand. Member 1
    must be a scalar or vector with integer component type, with 32-bit
    component width. Member 1 holds the exponent. These two members and x must
    have the same number of components.

    The operand x must be a scalar or vector whose component type is
    floating-point.

    #### Example:

    ```mlir
    %2 = spirv.GL.FrexpStruct %0 : f32 -> !spirv.struct<f32, i32>
    %3 = spirv.GL.FrexpStruct %0 : vector<3xf32> -> !spirv.struct<vector<3xf32>, vector<3xi32>>
    ```
  }];

  let arguments = (ins
    SPIRV_ScalarOrVectorOf<SPIRV_Float>:$operand
  );

  let results = (outs
    SPIRV_AnyStruct:$result
  );

  let assemblyFormat = [{
    attr-dict $operand `:` type($operand) `->` type($result)
  }];
}

def SPIRV_GLLdexpOp :
  SPIRV_GLOp<"Ldexp", 53, [
      Pure, AllTypesMatch<["x", "y"]>]> {
  let summary = "Builds y such that y = significand * 2^exponent";

  let description = [{
    Builds a floating-point number from x and the corresponding
    integral exponent of two in exp:

    significand * 2^exponent

    If this product is too large to be represented in the floating-point
    type, the resulting value is undefined. If exp is greater than +128
    (single precision) or +1024 (double precision), the resulting value is
    undefined. If exp is less than -126 (single precision) or -1022 (double precision),
    the result may be flushed to zero. Additionally, splitting the value
    into a significand and exponent using frexp and then reconstructing a
    floating-point value using ldexp should yield the original input for
    zero and all finite non-denormalized values.

    The operand x must be a scalar or vector whose component type is floating-point.

    The exp operand must be a scalar or vector with integer component type.
    The number of components in x and exp must be the same.

    Result Type must be the same type as the type of x. Results are computed per
    component.

    <!-- End of AutoGen section -->

    #### Example:

    ```mlir
    %y = spirv.GL.Ldexp %x : f32, %exp : i32 -> f32
    %y = spirv.GL.Ldexp %x : vector<3xf32>, %exp : vector<3xi32> -> vector<3xf32>
    ```
  }];

  let arguments = (ins
    SPIRV_ScalarOrVectorOf<SPIRV_Float>:$x,
    SPIRV_ScalarOrVectorOf<SPIRV_Integer>:$exp
  );

  let results = (outs
    SPIRV_ScalarOrVectorOf<SPIRV_Float>:$y
  );

  let assemblyFormat = [{
    attr-dict $x `:` type($x) `,` $exp `:` type($exp) `->` type($y)
  }];
}

def SPIRV_GLFMixOp :
  SPIRV_GLOp<"FMix", 46, [
      Pure, AllTypesMatch<["x", "y", "a", "result"]>]> {
  let summary = "Builds the linear blend of x and y";

  let description = [{
    Result is the linear blend of x and y, i.e., x * (1 - a) + y * a.

    The operands must all be a scalar or vector whose component type is floating-point.

    Result Type and the type of all operands must be the same type. Results are computed per component.

    <!-- End of AutoGen section -->

    #### Example:

    ```mlir
    %0 = spirv.GL.FMix %x : f32, %y : f32, %a : f32 -> f32
    %0 = spirv.GL.FMix %x : vector<4xf32>, %y : vector<4xf32>, %a : vector<4xf32> -> vector<4xf32>
    ```
  }];

  let arguments = (ins
    SPIRV_ScalarOrVectorOf<SPIRV_Float>:$x,
    SPIRV_ScalarOrVectorOf<SPIRV_Float>:$y,
    SPIRV_ScalarOrVectorOf<SPIRV_Float>:$a
  );

  let results = (outs
    SPIRV_ScalarOrVectorOf<SPIRV_Float>:$result
  );

  let assemblyFormat = [{
    attr-dict $x `:` type($x) `,` $y `:` type($y) `,` $a `:` type($a) `->` type($result)
  }];

  let hasVerifier = 0;
}

def SPIRV_GLFindUMsbOp : SPIRV_GLUnaryArithmeticOp<"FindUMsb", 75, SPIRV_Int32> {
  let summary = "Unsigned-integer most-significant bit";

  let description = [{
    Results in the bit number of the most-significant 1-bit in the binary
    representation of Value. If Value is 0, the result is -1.

    Result Type and the type of Value must both be integer scalar or
    integer vector types. Result Type and operand types must have the
    same number of components with the same component width. Results are
    computed per component.

    This instruction is currently limited to 32-bit width components.
  }];
}

#endif // MLIR_DIALECT_SPIRV_IR_GL_OPS
