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

// Base class for all GL ops.
class SPV_GLOp<string mnemonic, int opcode, list<Trait> traits = []> :
  SPV_ExtInstOp<mnemonic, "GL", "GLSL.std.450", opcode, traits> {

  let availability = [
    MinVersion<SPV_V_1_0>,
    MaxVersion<SPV_V_1_5>,
    Extension<[]>,
    Capability<[SPV_C_Shader]>
  ];
}

// Base class for GL unary ops.
class SPV_GLUnaryOp<string mnemonic, int opcode, Type resultType,
                      Type operandType, list<Trait> traits = []> :
  SPV_GLOp<mnemonic, opcode, !listconcat([NoSideEffect], traits)> {

  let arguments = (ins
    SPV_ScalarOrVectorOf<operandType>:$operand
  );

  let results = (outs
    SPV_ScalarOrVectorOf<resultType>:$result
  );

  let hasVerifier = 0;
}

// Base class for GL Unary arithmetic ops where return type matches
// the operand type.
class SPV_GLUnaryArithmeticOp<string mnemonic, int opcode, Type type,
                                list<Trait> traits = []> :
  SPV_GLUnaryOp<mnemonic, opcode, type, type,
                  traits # [SameOperandsAndResultType]> {
  let assemblyFormat = "$operand `:` type($operand) attr-dict";
}

// Base class for GL binary ops.
class SPV_GLBinaryOp<string mnemonic, int opcode, Type resultType,
                        Type operandType, list<Trait> traits = []> :
  SPV_GLOp<mnemonic, opcode, !listconcat([NoSideEffect], traits)> {

  let arguments = (ins
    SPV_ScalarOrVectorOf<operandType>:$lhs,
    SPV_ScalarOrVectorOf<operandType>:$rhs
  );

  let results = (outs
    SPV_ScalarOrVectorOf<resultType>:$result
  );

  let hasVerifier = 0;
}

// Base class for GL Binary arithmetic ops where operand types and
// return type matches.
class SPV_GLBinaryArithmeticOp<string mnemonic, int opcode, Type type,
                                 list<Trait> traits = []> :
  SPV_GLBinaryOp<mnemonic, opcode, type, type,
                   traits # [SameOperandsAndResultType]> {
  let assemblyFormat = "operands attr-dict `:` type($result)";
}

// Base class for GL ternary ops.
class SPV_GLTernaryArithmeticOp<string mnemonic, int opcode, Type type,
                        list<Trait> traits = []> :
  SPV_GLOp<mnemonic, opcode, !listconcat([NoSideEffect], traits)> {

  let arguments = (ins
    SPV_ScalarOrVectorOf<type>:$x,
    SPV_ScalarOrVectorOf<type>:$y,
    SPV_ScalarOrVectorOf<type>:$z
  );

  let results = (outs
    SPV_ScalarOrVectorOf<type>:$result
  );

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 0;
}

// -----

def SPV_GLFAbsOp : SPV_GLUnaryArithmeticOp<"FAbs", 4, SPV_Float> {
  let summary = "Absolute value of operand";

  let description = [{
    Result is x if x >= 0; otherwise result is -x.

    The operand x must be a scalar or vector whose component type is
    floating-point.

    Result Type and the type of x must be the same type. Results are computed
    per component.

    <!-- End of AutoGen section -->
    ```
    float-scalar-vector-type ::= float-type |
                                 `vector<` integer-literal `x` float-type `>`
    abs-op ::= ssa-id `=` `spv.GL.FAbs` ssa-use `:`
               float-scalar-vector-type
    ```
    #### Example:

    ```mlir
    %2 = spv.GL.FAbs %0 : f32
    %3 = spv.GL.FAbs %1 : vector<3xf16>
    ```
  }];
}

// -----

def SPV_GLSAbsOp : SPV_GLUnaryArithmeticOp<"SAbs", 5, SPV_Integer> {
  let summary = "Absolute value of operand";

  let description = [{
    Result is x if x ≥ 0; otherwise result is -x, where x is interpreted as a
    signed integer.

    Result Type and the type of x must both be integer scalar or integer vector
    types. Result Type and operand types must have the same number of components
    with the same component width. Results are computed per component.

    <!-- End of AutoGen section -->
    ```
    integer-scalar-vector-type ::= integer-type |
                                   `vector<` integer-literal `x` integer-type `>`
    abs-op ::= ssa-id `=` `spv.GL.SAbs` ssa-use `:`
               integer-scalar-vector-type
    ```
    #### Example:

    ```mlir
    %2 = spv.GL.SAbs %0 : i32
    %3 = spv.GL.SAbs %1 : vector<3xi16>
    ```
  }];
}

// -----

def SPV_GLCeilOp : SPV_GLUnaryArithmeticOp<"Ceil", 9, SPV_Float> {
  let summary = "Rounds up to the next whole number";

  let description = [{
    Result is the value equal to the nearest whole number that is greater than
    or equal to x.

    The operand x must be a scalar or vector whose component type is
    floating-point.

    Result Type and the type of x must be the same type. Results are computed
    per component.

    <!-- End of AutoGen section -->
    ```
    float-scalar-vector-type ::= float-type |
                                 `vector<` integer-literal `x` float-type `>`
    ceil-op ::= ssa-id `=` `spv.GL.Ceil` ssa-use `:`
                float-scalar-vector-type
    ```
    #### Example:

    ```mlir
    %2 = spv.GL.Ceil %0 : f32
    %3 = spv.GL.Ceil %1 : vector<3xf16>
    ```
  }];
}

// -----

def SPV_GLCosOp : SPV_GLUnaryArithmeticOp<"Cos", 14, SPV_Float16or32> {
  let summary = "Cosine of operand in radians";

  let description = [{
    The standard trigonometric cosine of x radians.

    The operand x must be a scalar or vector whose component type is 16-bit or
    32-bit floating-point.

    Result Type and the type of x must be the same type. Results are computed
    per component.

    <!-- End of AutoGen section -->
    ```
    restricted-float-scalar-type ::=  `f16` | `f32`
    restricted-float-scalar-vector-type ::=
      restricted-float-scalar-type |
      `vector<` integer-literal `x` restricted-float-scalar-type `>`
    cos-op ::= ssa-id `=` `spv.GL.Cos` ssa-use `:`
               restricted-float-scalar-vector-type
    ```
    #### Example:

    ```mlir
    %2 = spv.GL.Cos %0 : f32
    %3 = spv.GL.Cos %1 : vector<3xf16>
    ```
  }];
}

// -----

def SPV_GLSinOp : SPV_GLUnaryArithmeticOp<"Sin", 13, SPV_Float16or32> {
  let summary = "Sine of operand in radians";

  let description = [{
    The standard trigonometric sine of x radians.

    The operand x must be a scalar or vector whose component type is 16-bit or
    32-bit floating-point.

    Result Type and the type of x must be the same type. Results are computed
    per component.

    <!-- End of AutoGen section -->
    ```
    restricted-float-scalar-type ::=  `f16` | `f32`
    restricted-float-scalar-vector-type ::=
      restricted-float-scalar-type |
      `vector<` integer-literal `x` restricted-float-scalar-type `>`
    sin-op ::= ssa-id `=` `spv.GL.Sin` ssa-use `:`
               restricted-float-scalar-vector-type
    ```
    #### Example:

    ```mlir
    %2 = spv.GL.Sin %0 : f32
    %3 = spv.GL.Sin %1 : vector<3xf16>
    ```
  }];
}

// -----

def SPV_GLTanOp : SPV_GLUnaryArithmeticOp<"Tan", 15, SPV_Float16or32> {
  let summary = "Tangent of operand in radians";

  let description = [{
    The standard trigonometric tangent of x radians.

    The operand x must be a scalar or vector whose component type is 16-bit or
    32-bit floating-point.

    Result Type and the type of x must be the same type. Results are computed
    per component.

    <!-- End of AutoGen section -->
    ```
    restricted-float-scalar-type ::=  `f16` | `f32`
    restricted-float-scalar-vector-type ::=
      restricted-float-scalar-type |
      `vector<` integer-literal `x` restricted-float-scalar-type `>`
    tan-op ::= ssa-id `=` `spv.GL.Tan` ssa-use `:`
               restricted-float-scalar-vector-type
    ```
    #### Example:

    ```mlir
    %2 = spv.GL.Tan %0 : f32
    %3 = spv.GL.Tan %1 : vector<3xf16>
    ```
  }];
}

// -----

def SPV_GLAsinOp : SPV_GLUnaryArithmeticOp<"Asin", 16, SPV_Float16or32> {
  let summary = "Arc Sine of operand in radians";

  let description = [{
    The standard trigonometric arc sine of x radians.

    Result is an angle, in radians, whose sine is x. The range of result values
    is [-π / 2, π / 2]. Result is undefined if abs x > 1.

    The operand x must be a scalar or vector whose component type is 16-bit or
    32-bit floating-point.

    Result Type and the type of x must be the same type. Results are computed
    per component.
    <!-- End of AutoGen section -->
    ```
    restricted-float-scalar-type ::=  `f16` | `f32`
    restricted-float-scalar-vector-type ::=
      restricted-float-scalar-type |
      `vector<` integer-literal `x` restricted-float-scalar-type `>`
    asin-op ::= ssa-id `=` `spv.GL.Asin` ssa-use `:`
                restricted-float-scalar-vector-type
    ```
    #### Example:

    ```mlir
    %2 = spv.GL.Asin %0 : f32
    %3 = spv.GL.Asin %1 : vector<3xf16>
    ```
  }];
}

// -----

def SPV_GLAcosOp : SPV_GLUnaryArithmeticOp<"Acos", 17, SPV_Float16or32> {
  let summary = "Arc Cosine of operand in radians";

  let description = [{
    The standard trigonometric arc cosine of x radians.

    Result is an angle, in radians, whose cosine is x. The range of result
    values is [0, π]. Result is undefined if abs x > 1.

    The operand x must be a scalar or vector whose component type is 16-bit or
    32-bit floating-point.

    Result Type and the type of x must be the same type. Results are computed
    per component.
    <!-- End of AutoGen section -->
    ```
    restricted-float-scalar-type ::=  `f16` | `f32`
    restricted-float-scalar-vector-type ::=
      restricted-float-scalar-type |
      `vector<` integer-literal `x` restricted-float-scalar-type `>`
    acos-op ::= ssa-id `=` `spv.GL.Acos` ssa-use `:`
                restricted-float-scalar-vector-type
    ```
    #### Example:

    ```mlir
    %2 = spv.GL.Acos %0 : f32
    %3 = spv.GL.Acos %1 : vector<3xf16>
    ```
  }];
}

// -----

def SPV_GLAtanOp : SPV_GLUnaryArithmeticOp<"Atan", 18, SPV_Float16or32> {
  let summary = "Arc Tangent of operand in radians";

  let description = [{
    The standard trigonometric arc tangent of x radians.

    Result is an angle, in radians, whose tangent is y_over_x. The range of
    result values is [-π / 2, π / 2].

    The operand x must be a scalar or vector whose component type is 16-bit or
    32-bit floating-point.

    Result Type and the type of x must be the same type. Results are computed
    per component.
    <!-- End of AutoGen section -->
    ```
    restricted-float-scalar-type ::=  `f16` | `f32`
    restricted-float-scalar-vector-type ::=
      restricted-float-scalar-type |
      `vector<` integer-literal `x` restricted-float-scalar-type `>`
    atan-op ::= ssa-id `=` `spv.GL.Atan` ssa-use `:`
                restricted-float-scalar-vector-type
    ```
    #### Example:

    ```mlir
    %2 = spv.GL.Atan %0 : f32
    %3 = spv.GL.Atan %1 : vector<3xf16>
    ```
  }];
}

// -----

def SPV_GLExpOp : SPV_GLUnaryArithmeticOp<"Exp", 27, SPV_Float16or32> {
  let summary = "Exponentiation of Operand 1";

  let description = [{
    Result is the natural exponentiation of x; e^x.

    The operand x must be a scalar or vector whose component type is
    16-bit or 32-bit floating-point.

    Result Type and the type of x must be the same type. Results are
    computed per component.";

    <!-- End of AutoGen section -->
    ```
    restricted-float-scalar-type ::=  `f16` | `f32`
    restricted-float-scalar-vector-type ::=
      restricted-float-scalar-type |
      `vector<` integer-literal `x` restricted-float-scalar-type `>`
    exp-op ::= ssa-id `=` `spv.GL.Exp` ssa-use `:`
               restricted-float-scalar-vector-type
    ```
    #### Example:

    ```mlir
    %2 = spv.GL.Exp %0 : f32
    %3 = spv.GL.Exp %1 : vector<3xf16>
    ```
  }];
}

// -----

def SPV_GLFloorOp : SPV_GLUnaryArithmeticOp<"Floor", 8, SPV_Float> {
  let summary = "Rounds down to the next whole number";

  let description = [{
    Result is the value equal to the nearest whole number that is less than or
    equal to x.

    The operand x must be a scalar or vector whose component type is
    floating-point.

    Result Type and the type of x must be the same type. Results are computed
    per component.

    <!-- End of AutoGen section -->
    ```
    float-scalar-vector-type ::= float-type |
                                 `vector<` integer-literal `x` float-type `>`
    floor-op ::= ssa-id `=` `spv.GL.Floor` ssa-use `:`
                float-scalar-vector-type
    ```
    #### Example:

    ```mlir
    %2 = spv.GL.Floor %0 : f32
    %3 = spv.GL.Floor %1 : vector<3xf16>
    ```
  }];
}

// -----

def SPV_GLRoundOp: SPV_GLUnaryArithmeticOp<"Round", 1, SPV_Float> {
  let summary = "Rounds to the whole number";

  let description = [{
    Result is the value equal to the nearest whole number.

    The operand x must be a scalar or vector whose component type is
    floating-point.

    Result Type and the type of x must be the same type. Results are computed
    per component.

    <!-- End of AutoGen section -->
    ```
    float-scalar-vector-type ::= float-type |
                                 `vector<` integer-literal `x` float-type `>`
    floor-op ::= ssa-id `=` `spv.GL.Round` ssa-use `:`
                float-scalar-vector-type
    ```
    #### Example:

    ```mlir
    %2 = spv.GL.Round %0 : f32
    %3 = spv.GL.Round %1 : vector<3xf16>
    ```
  }];
}

// -----

def SPV_GLInverseSqrtOp : SPV_GLUnaryArithmeticOp<"InverseSqrt", 32, SPV_Float> {
  let summary = "Reciprocal of sqrt(operand)";

  let description = [{
    Result is the reciprocal of sqrt x. Result is undefined if x <= 0.

    The operand x must be a scalar or vector whose component type is
    floating-point.

    Result Type and the type of x must be the same type. Results are computed
    per component.

    <!-- End of AutoGen section -->
    ```
    float-scalar-vector-type ::= float-type |
                                 `vector<` integer-literal `x` float-type `>`
    rsqrt-op ::= ssa-id `=` `spv.GL.InverseSqrt` ssa-use `:`
                 float-scalar-vector-type
    ```
    #### Example:

    ```mlir
    %2 = spv.GL.InverseSqrt %0 : f32
    %3 = spv.GL.InverseSqrt %1 : vector<3xf16>
    ```
  }];
}

// -----

def SPV_GLLogOp : SPV_GLUnaryArithmeticOp<"Log", 28, SPV_Float16or32> {
  let summary = "Natural logarithm of the operand";

  let description = [{
    Result is the natural logarithm of x, i.e., the value y which satisfies the
    equation x = ey. Result is undefined if x <= 0.

    The operand x must be a scalar or vector whose component type is 16-bit or
    32-bit floating-point.

    Result Type and the type of x must be the same type. Results are computed
    per component.

    <!-- End of AutoGen section -->
    ```
    restricted-float-scalar-type ::=  `f16` | `f32`
    restricted-float-scalar-vector-type ::=
      restricted-float-scalar-type |
      `vector<` integer-literal `x` restricted-float-scalar-type `>`
    log-op ::= ssa-id `=` `spv.GL.Log` ssa-use `:`
               restricted-float-scalar-vector-type
    ```
    #### Example:

    ```mlir
    %2 = spv.GL.Log %0 : f32
    %3 = spv.GL.Log %1 : vector<3xf16>
    ```
  }];
}

// -----

def SPV_GLFMaxOp : SPV_GLBinaryArithmeticOp<"FMax", 40, SPV_Float> {
  let summary = "Return maximum of two floating-point operands";

  let description = [{
    Result is y if x < y; otherwise result is x. Which operand is the
    result is undefined if one of the operands is a NaN.

    The operands must all be a scalar or vector whose component type
    is floating-point.

    Result Type and the type of all operands must be the same
    type. Results are computed per component.

    <!-- End of AutoGen section -->
    ```
    float-scalar-vector-type ::= float-type |
                                 `vector<` integer-literal `x` float-type `>`
    fmax-op ::= ssa-id `=` `spv.GL.FMax` ssa-use `:`
                float-scalar-vector-type
    ```
    #### Example:

    ```mlir
    %2 = spv.GL.FMax %0, %1 : f32
    %3 = spv.GL.FMax %0, %1 : vector<3xf16>
    ```
  }];
}

// -----

def SPV_GLUMaxOp : SPV_GLBinaryArithmeticOp<"UMax", 41, SPV_Integer> {
  let summary = "Return maximum of two unsigned integer operands";

  let description = [{
    Result is y if x < y; otherwise result is x, where x and y are interpreted
    as unsigned integers.

    Result Type and the type of x and y must both be integer scalar or integer
    vector types. Result Type and operand types must have the same number of
    components with the same component width. Results are computed per
    component.

    <!-- End of AutoGen section -->
    ```
    integer-scalar-vector-type ::= integer-type |
                                   `vector<` integer-literal `x` integer-type `>`
    smax-op ::= ssa-id `=` `spv.GL.UMax` ssa-use `:`
                integer-scalar-vector-type
    ```
    #### Example:

    ```mlir
    %2 = spv.GL.UMax %0, %1 : i32
    %3 = spv.GL.UMax %0, %1 : vector<3xi16>
    ```
  }];
}

// -----

def SPV_GLSMaxOp : SPV_GLBinaryArithmeticOp<"SMax", 42, SPV_Integer> {
  let summary = "Return maximum of two signed integer operands";

  let description = [{
    Result is y if x < y; otherwise result is x, where x and y are interpreted
    as signed integers.

    Result Type and the type of x and y must both be integer scalar or integer
    vector types. Result Type and operand types must have the same number of
    components with the same component width. Results are computed per
    component.

    <!-- End of AutoGen section -->
    ```
    integer-scalar-vector-type ::= integer-type |
                                   `vector<` integer-literal `x` integer-type `>`
    smax-op ::= ssa-id `=` `spv.GL.SMax` ssa-use `:`
                integer-scalar-vector-type
    ```
    #### Example:

    ```mlir
    %2 = spv.GL.SMax %0, %1 : i32
    %3 = spv.GL.SMax %0, %1 : vector<3xi16>
    ```
  }];
}

// -----

def SPV_GLFMinOp : SPV_GLBinaryArithmeticOp<"FMin", 37, SPV_Float> {
  let summary = "Return minimum of two floating-point operands";

  let description = [{
    Result is y if y < x; otherwise result is x. Which operand is the result is
    undefined if one of the operands is a NaN.

    The operands must all be a scalar or vector whose component type is
    floating-point.

    Result Type and the type of all operands must be the same type. Results are
    computed per component.

    <!-- End of AutoGen section -->
    ```
    float-scalar-vector-type ::= float-type |
                                 `vector<` integer-literal `x` float-type `>`
    fmin-op ::= ssa-id `=` `spv.GL.FMin` ssa-use `:`
                float-scalar-vector-type
    ```
    #### Example:

    ```mlir
    %2 = spv.GL.FMin %0, %1 : f32
    %3 = spv.GL.FMin %0, %1 : vector<3xf16>
    ```
  }];
}

// -----

def SPV_GLUMinOp : SPV_GLBinaryArithmeticOp<"UMin", 38, SPV_Integer> {
  let summary = "Return minimum of two unsigned integer operands";

  let description = [{
    Result is y if y < x; otherwise result is x, where x and y are interpreted
    as unsigned integers.

    Result Type and the type of x and y must both be integer scalar or integer
    vector types. Result Type and operand types must have the same number of
    components with the same component width. Results are computed per
    component.

    <!-- End of AutoGen section -->
    ```
    integer-scalar-vector-type ::= integer-type |
                                   `vector<` integer-literal `x` integer-type `>`
    smin-op ::= ssa-id `=` `spv.GL.UMin` ssa-use `:`
                integer-scalar-vector-type
    ```
    #### Example:

    ```mlir
    %2 = spv.GL.UMin %0, %1 : i32
    %3 = spv.GL.UMin %0, %1 : vector<3xi16>
    ```
  }];
}

// -----

def SPV_GLSMinOp : SPV_GLBinaryArithmeticOp<"SMin", 39, SPV_Integer> {
  let summary = "Return minimum of two signed integer operands";

  let description = [{
    Result is y if y < x; otherwise result is x, where x and y are interpreted
    as signed integers.

    Result Type and the type of x and y must both be integer scalar or integer
    vector types. Result Type and operand types must have the same number of
    components with the same component width. Results are computed per
    component.

    <!-- End of AutoGen section -->
    ```
    integer-scalar-vector-type ::= integer-type |
                                   `vector<` integer-literal `x` integer-type `>`
    smin-op ::= ssa-id `=` `spv.GL.SMin` ssa-use `:`
                integer-scalar-vector-type
    ```
    #### Example:

    ```mlir
    %2 = spv.GL.SMin %0, %1 : i32
    %3 = spv.GL.SMin %0, %1 : vector<3xi16>
    ```
  }];
}

// -----

def SPV_GLPowOp : SPV_GLBinaryArithmeticOp<"Pow", 26, SPV_Float16or32> {
  let summary = "Return x raised to the y power of two operands";

  let description = [{
    Result is x raised to the y power; x^y.

    Result is undefined if x = 0 and y ≤ 0.

    The operand x and y must be a scalar or vector whose component type is
    16-bit or 32-bit floating-point.

    Result Type and the type of all operands must be the same type. Results are
    computed per component.

    <!-- End of AutoGen section -->
    ```
    restricted-float-scalar-type ::=  `f16` | `f32`
    restricted-float-scalar-vector-type ::=
      restricted-float-scalar-type |
      `vector<` integer-literal `x` restricted-float-scalar-type `>`
    pow-op ::= ssa-id `=` `spv.GL.Pow` ssa-use `:`
               restricted-float-scalar-vector-type
    ```
    #### Example:

    ```mlir
    %2 = spv.GL.Pow %0, %1 : f32
    %3 = spv.GL.Pow %0, %1 : vector<3xf16>
    ```
  }];
}

// -----

def SPV_GLFSignOp : SPV_GLUnaryArithmeticOp<"FSign", 6, SPV_Float> {
  let summary = "Returns the sign of the operand";

  let description = [{
    Result is 1.0 if x > 0, 0.0 if x = 0, or -1.0 if x < 0.

    The operand x must be a scalar or vector whose component type is
    floating-point.

    Result Type and the type of x must be the same type. Results are computed
    per component.

    <!-- End of AutoGen section -->
    ```
    float-scalar-vector-type ::= float-type |
                                 `vector<` integer-literal `x` float-type `>`
    sign-op ::= ssa-id `=` `spv.GL.FSign` ssa-use `:`
                float-scalar-vector-type
    ```
    #### Example:

    ```mlir
    %2 = spv.GL.FSign %0 : f32
    %3 = spv.GL.FSign %1 : vector<3xf16>
    ```
  }];
}

// -----

def SPV_GLSSignOp : SPV_GLUnaryArithmeticOp<"SSign", 7, SPV_Integer> {
  let summary = "Returns the sign of the operand";

  let description = [{
    Result is 1 if x > 0, 0 if x = 0, or -1 if x < 0, where x is interpreted as
    a signed integer.

    Result Type and the type of x must both be integer scalar or integer vector
    types. Result Type and operand types must have the same number of components
    with the same component width. Results are computed per component.

    <!-- End of AutoGen section -->
    ```
    integer-scalar-vector-type ::= integer-type |
                                   `vector<` integer-literal `x` integer-type `>`
    sign-op ::= ssa-id `=` `spv.GL.SSign` ssa-use `:`
                integer-scalar-vector-type
    ```
    #### Example:

    ```mlir
    %2 = spv.GL.SSign %0 : i32
    %3 = spv.GL.SSign %1 : vector<3xi16>
    ```
  }];
}

// -----

def SPV_GLSqrtOp : SPV_GLUnaryArithmeticOp<"Sqrt", 31, SPV_Float> {
  let summary = "Returns the square root of the operand";

  let description = [{
    Result is the square root of x. Result is undefined if x < 0.

    The operand x must be a scalar or vector whose component type is
    floating-point.

    Result Type and the type of x must be the same type. Results are computed
    per component.

    <!-- End of AutoGen section -->
    ```
    float-scalar-vector-type ::= float-type |
                                 `vector<` integer-literal `x` float-type `>`
    sqrt-op ::= ssa-id `=` `spv.GL.Sqrt` ssa-use `:`
                float-scalar-vector-type
    ```
    #### Example:

    ```mlir
    %2 = spv.GL.Sqrt %0 : f32
    %3 = spv.GL.Sqrt %1 : vector<3xf16>
    ```
  }];
}

// -----

def SPV_GLSinhOp : SPV_GLUnaryArithmeticOp<"Sinh", 19, SPV_Float16or32> {
  let summary = "Hyperbolic sine of operand in radians";

  let description = [{
    Hyperbolic sine of x radians.

    The operand x must be a scalar or vector whose component type is 16-bit or
    32-bit floating-point.

    Result Type and the type of x must be the same type. Results are computed
    per component.

    <!-- End of AutoGen section -->
    ```
    restricted-float-scalar-type ::=  `f16` | `f32`
    restricted-float-scalar-vector-type ::=
      restricted-float-scalar-type |
      `vector<` integer-literal `x` restricted-float-scalar-type `>`
    sinh-op ::= ssa-id `=` `spv.GL.Sinh` ssa-use `:`
                restricted-float-scalar-vector-type
    ```
    #### Example:

    ```mlir
    %2 = spv.GL.Sinh %0 : f32
    %3 = spv.GL.Sinh %1 : vector<3xf16>
    ```
  }];
}

// -----

def SPV_GLCoshOp : SPV_GLUnaryArithmeticOp<"Cosh", 20, SPV_Float16or32> {
  let summary = "Hyperbolic cosine of operand in radians";

  let description = [{
    Hyperbolic cosine of x radians.

    The operand x must be a scalar or vector whose component type is 16-bit or
    32-bit floating-point.

    Result Type and the type of x must be the same type. Results are computed
    per component.

    <!-- End of AutoGen section -->
    ```
    restricted-float-scalar-type ::=  `f16` | `f32`
    restricted-float-scalar-vector-type ::=
      restricted-float-scalar-type |
      `vector<` integer-literal `x` restricted-float-scalar-type `>`
    cosh-op ::= ssa-id `=` `spv.GL.Cosh` ssa-use `:`
                restricted-float-scalar-vector-type
    ```
    #### Example:

    ```mlir
    %2 = spv.GL.Cosh %0 : f32
    %3 = spv.GL.Cosh %1 : vector<3xf16>
    ```
  }];
}

// -----

def SPV_GLTanhOp : SPV_GLUnaryArithmeticOp<"Tanh", 21, SPV_Float16or32> {
  let summary = "Hyperbolic tangent of operand in radians";

  let description = [{
    Hyperbolic tangent of x radians.

    The operand x must be a scalar or vector whose component type is 16-bit or
    32-bit floating-point.

    Result Type and the type of x must be the same type. Results are computed
    per component.

    <!-- End of AutoGen section -->
    ```
    restricted-float-scalar-type ::=  `f16` | `f32`
    restricted-float-scalar-vector-type ::=
      restricted-float-scalar-type |
      `vector<` integer-literal `x` restricted-float-scalar-type `>`
    tanh-op ::= ssa-id `=` `spv.GL.Tanh` ssa-use `:`
                restricted-float-scalar-vector-type
    ```
    #### Example:

    ```mlir
    %2 = spv.GL.Tanh %0 : f32
    %3 = spv.GL.Tanh %1 : vector<3xf16>
    ```
  }];
}

// -----

def SPV_GLFClampOp : SPV_GLTernaryArithmeticOp<"FClamp", 43, SPV_Float> {
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
    fclamp-op ::= ssa-id `=` `spv.GL.FClamp` ssa-use, ssa-use, ssa-use `:`
               float-scalar-vector-type
    ```
    #### Example:

    ```mlir
    %2 = spv.GL.FClamp %x, %min, %max : f32
    %3 = spv.GL.FClamp %x, %min, %max : vector<3xf16>
    ```
  }];
}

// -----

def SPV_GLUClampOp : SPV_GLTernaryArithmeticOp<"UClamp", 44, SPV_Integer> {
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
    uclamp-op ::= ssa-id `=` `spv.GL.UClamp` ssa-use, ssa-use, ssa-use `:`
               unsigned-signless-scalar-vector-type
    ```
    #### Example:

    ```mlir
    %2 = spv.GL.UClamp %x, %min, %max : i32
    %3 = spv.GL.UClamp %x, %min, %max : vector<3xui16>
    ```
  }];
}

// -----

def SPV_GLSClampOp : SPV_GLTernaryArithmeticOp<"SClamp", 45, SPV_Integer> {
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
    uclamp-op ::= ssa-id `=` `spv.GL.UClamp` ssa-use, ssa-use, ssa-use `:`
               sgined-scalar-vector-type
    ```
    #### Example:

    ```mlir
    %2 = spv.GL.SClamp %x, %min, %max : si32
    %3 = spv.GL.SClamp %x, %min, %max : vector<3xsi16>
    ```
  }];
}

// -----

def SPV_GLFmaOp : SPV_GLTernaryArithmeticOp<"Fma", 50, SPV_Float> {
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
    fma-op ::= ssa-id `=` `spv.GL.Fma` ssa-use, ssa-use, ssa-use `:`
               float-scalar-vector-type
    ```
    #### Example:

    ```mlir
    %0 = spv.GL.Fma %a, %b, %c : f32
    %1 = spv.GL.Fma %a, %b, %c : vector<3xf16>
    ```
  }];
}

// ----

def SPV_GLFrexpStructOp : SPV_GLOp<"FrexpStruct", 52, [NoSideEffect]> {
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

    <!-- End of AutoGen section -->
    ```
    float-scalar-vector-type ::= float-type |
                                 `vector<` integer-literal `x` float-type `>`
    integer-scalar-vector-type ::= integer-type |
                                 `vector<` integer-literal `x` integer-type `>`
    frexpstruct-op ::= ssa-id `=` `spv.GL.FrexpStruct` ssa-use `:`
                                  `!spv.struct<` float-scalar-vector-type `,`
                                                  integer-scalar-vector-type `>`
    ```
    #### Example:

    ```mlir
    %2 = spv.GL.FrexpStruct %0 : f32 -> !spv.struct<f32, i32>
    %3 = spv.GL.FrexpStruct %0 : vector<3xf32> -> !spv.struct<vector<3xf32>, vector<3xi32>>
    ```
  }];

  let arguments = (ins
    SPV_ScalarOrVectorOf<SPV_Float>:$operand
  );

  let results = (outs
    SPV_AnyStruct:$result
  );

  let assemblyFormat = [{
    attr-dict $operand `:` type($operand) `->` type($result)
  }];
}

def SPV_GLLdexpOp :
  SPV_GLOp<"Ldexp", 53, [
      NoSideEffect, AllTypesMatch<["x", "y"]>]> {
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
    %y = spv.GL.Ldexp %x : f32, %exp : i32 -> f32
    %y = spv.GL.Ldexp %x : vector<3xf32>, %exp : vector<3xi32> -> vector<3xf32>
    ```
  }];

  let arguments = (ins
    SPV_ScalarOrVectorOf<SPV_Float>:$x,
    SPV_ScalarOrVectorOf<SPV_Integer>:$exp
  );

  let results = (outs
    SPV_ScalarOrVectorOf<SPV_Float>:$y
  );

  let assemblyFormat = [{
    attr-dict $x `:` type($x) `,` $exp `:` type($exp) `->` type($y)
  }];
}

def SPV_GLFMixOp :
  SPV_GLOp<"FMix", 46, [
      NoSideEffect, AllTypesMatch<["x", "y", "a", "result"]>]> {
  let summary = "Builds the linear blend of x and y";

  let description = [{
    Result is the linear blend of x and y, i.e., x * (1 - a) + y * a.

    The operands must all be a scalar or vector whose component type is floating-point.

    Result Type and the type of all operands must be the same type. Results are computed per component.

    <!-- End of AutoGen section -->

    #### Example:

    ```mlir
    %0 = spv.GL.FMix %x : f32, %y : f32, %a : f32 -> f32
    %0 = spv.GL.FMix %x : vector<4xf32>, %y : vector<4xf32>, %a : vector<4xf32> -> vector<4xf32>
    ```
  }];

  let arguments = (ins
    SPV_ScalarOrVectorOf<SPV_Float>:$x,
    SPV_ScalarOrVectorOf<SPV_Float>:$y,
    SPV_ScalarOrVectorOf<SPV_Float>:$a
  );

  let results = (outs
    SPV_ScalarOrVectorOf<SPV_Float>:$result
  );

  let assemblyFormat = [{
    attr-dict $x `:` type($x) `,` $y `:` type($y) `,` $a `:` type($a) `->` type($result)
  }];

  let hasVerifier = 0;
}

def SPV_GLFindUMsbOp : SPV_GLUnaryArithmeticOp<"FindUMsb", 75, SPV_Int32> {
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
