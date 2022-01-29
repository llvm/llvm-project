//===- SPIRVOCLOps.td - OpenCL extended insts spec file ----*- tablegen -*-===//
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

#ifndef MLIR_DIALECT_SPIRV_IR_OCL_OPS
#define MLIR_DIALECT_SPIRV_IR_OCL_OPS

include "mlir/Dialect/SPIRV/IR/SPIRVBase.td"

//===----------------------------------------------------------------------===//
// SPIR-V OpenCL opcode specification.
//===----------------------------------------------------------------------===//

// Base class for all OpenCL ops.
class SPV_OCLOp<string mnemonic, int opcode, list<Trait> traits = []> :
  SPV_ExtInstOp<mnemonic, "OCL", "OpenCL.std", opcode, traits> {

  let availability = [
    MinVersion<SPV_V_1_0>,
    MaxVersion<SPV_V_1_5>,
    Extension<[]>,
    Capability<[SPV_C_Kernel]>
  ];
}

// Base class for OpenCL unary ops.
class SPV_OCLUnaryOp<string mnemonic, int opcode, Type resultType,
                     Type operandType, list<Trait> traits = []> :
  SPV_OCLOp<mnemonic, opcode, !listconcat([NoSideEffect], traits)> {

  let arguments = (ins
    SPV_ScalarOrVectorOf<operandType>:$operand
  );

  let results = (outs
    SPV_ScalarOrVectorOf<resultType>:$result
  );

  let parser = [{ return parseUnaryOp(parser, result); }];

  let printer = [{ return printUnaryOp(getOperation(), p); }];

  let verifier = [{ return success(); }];
}

// Base class for OpenCL Unary arithmetic ops where return type matches
// the operand type.
class SPV_OCLUnaryArithmeticOp<string mnemonic, int opcode, Type type,
                               list<Trait> traits = []> :
  SPV_OCLUnaryOp<mnemonic, opcode, type, type, traits>;

// Base class for OpenCL binary ops.
class SPV_OCLBinaryOp<string mnemonic, int opcode, Type resultType,
                      Type operandType, list<Trait> traits = []> :
  SPV_OCLOp<mnemonic, opcode, !listconcat([NoSideEffect], traits)> {

  let arguments = (ins
    SPV_ScalarOrVectorOf<operandType>:$lhs,
    SPV_ScalarOrVectorOf<operandType>:$rhs
  );

  let results = (outs
    SPV_ScalarOrVectorOf<resultType>:$result
  );

  let parser = [{ return impl::parseOneResultSameOperandTypeOp(parser, result); }];

  let printer = [{ return impl::printOneResultOp(getOperation(), p); }];

  let verifier = [{ return success(); }];
}

// Base class for OpenCL Binary arithmetic ops where operand types and
// return type matches.
class SPV_OCLBinaryArithmeticOp<string mnemonic, int opcode, Type type,
                                list<Trait> traits = []> :
  SPV_OCLBinaryOp<mnemonic, opcode, type, type, traits>;

// -----

def SPV_OCLErfOp : SPV_OCLUnaryArithmeticOp<"erf", 18, SPV_Float> {
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
    erf-op ::= ssa-id `=` `spv.OCL.erf` ssa-use `:`
               float-scalar-vector-type
    ```mlir

    #### Example:

    ```
    %2 = spv.OCL.erf %0 : f32
    %3 = spv.OCL.erf %1 : vector<3xf16>
    ```
  }];
}

// -----

def SPV_OCLCeilOp : SPV_OCLUnaryArithmeticOp<"ceil", 12, SPV_Float> {
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
    ceil-op ::= ssa-id `=` `spv.OCL.ceil` ssa-use `:`
               float-scalar-vector-type
    ```mlir

    #### Example:

    ```
    %2 = spv.OCL.ceil %0 : f32
    %3 = spv.OCL.ceil %1 : vector<3xf16>
    ```
  }];
}

// -----

def SPV_OCLCosOp : SPV_OCLUnaryArithmeticOp<"cos", 14, SPV_Float> {
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
    cos-op ::= ssa-id `=` `spv.OCL.cos` ssa-use `:`
               float-scalar-vector-type
    ```mlir

    #### Example:

    ```
    %2 = spv.OCL.cos %0 : f32
    %3 = spv.OCL.cos %1 : vector<3xf16>
    ```
  }];
}

// -----

def SPV_OCLExpOp : SPV_OCLUnaryArithmeticOp<"exp", 19, SPV_Float> {
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
    exp-op ::= ssa-id `=` `spv.OCL.exp` ssa-use `:`
               float-scalar-vector-type
    ```
    #### Example:

    ```mlir
    %2 = spv.OCL.exp %0 : f32
    %3 = spv.OCL.exp %1 : vector<3xf16>
    ```
  }];
}

// -----

def SPV_OCLFAbsOp : SPV_OCLUnaryArithmeticOp<"fabs", 23, SPV_Float> {
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
    abs-op ::= ssa-id `=` `spv.OCL.fabs` ssa-use `:`
               float-scalar-vector-type
    ```
    #### Example:

    ```mlir
    %2 = spv.OCL.fabs %0 : f32
    %3 = spv.OCL.fabs %1 : vector<3xf16>
    ```
  }];
}

// -----

def SPV_OCLFloorOp : SPV_OCLUnaryArithmeticOp<"floor", 25, SPV_Float> {
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
    floor-op ::= ssa-id `=` `spv.OCL.floor` ssa-use `:`
               float-scalar-vector-type
    ```mlir

    #### Example:

    ```
    %2 = spv.OCL.floor %0 : f32
    %3 = spv.OCL.ceifloorl %1 : vector<3xf16>
    ```
  }];
}

// -----

def SPV_OCLLogOp : SPV_OCLUnaryArithmeticOp<"log", 37, SPV_Float> {
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
    log-op ::= ssa-id `=` `spv.OCL.log` ssa-use `:`
               float-scalar-vector-type
    ```mlir

    #### Example:

    ```
    %2 = spv.OCL.log %0 : f32
    %3 = spv.OCL.log %1 : vector<3xf16>
    ```
  }];
}

// -----

def SPV_OCLPowOp : SPV_OCLBinaryArithmeticOp<"pow", 48, SPV_Float> {
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
    pow-op ::= ssa-id `=` `spv.OCL.pow` ssa-use `:`
               restricted-float-scalar-vector-type
    ```
    #### Example:

    ```mlir
    %2 = spv.OCL.pow %0, %1 : f32
    %3 = spv.OCL.pow %0, %1 : vector<3xf16>
    ```
  }];
}

// -----

def SPV_OCLRsqrtOp : SPV_OCLUnaryArithmeticOp<"rsqrt", 56, SPV_Float> {
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
    rsqrt-op ::= ssa-id `=` `spv.OCL.rsqrt` ssa-use `:`
               float-scalar-vector-type
    ```mlir

    #### Example:

    ```
    %2 = spv.OCL.rsqrt %0 : f32
    %3 = spv.OCL.rsqrt %1 : vector<3xf16>
    ```
  }];
}

// -----

def SPV_OCLSinOp : SPV_OCLUnaryArithmeticOp<"sin", 57, SPV_Float> {
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
    sin-op ::= ssa-id `=` `spv.OCL.sin` ssa-use `:`
               float-scalar-vector-type
    ```mlir

    #### Example:

    ```
    %2 = spv.OCL.sin %0 : f32
    %3 = spv.OCL.sin %1 : vector<3xf16>
    ```
  }];
}

// -----

def SPV_OCLSqrtOp : SPV_OCLUnaryArithmeticOp<"sqrt", 61, SPV_Float> {
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
    sqrt-op ::= ssa-id `=` `spv.OCL.sqrt` ssa-use `:`
               float-scalar-vector-type
    ```mlir

    #### Example:

    ```
    %2 = spv.OCL.sqrt %0 : f32
    %3 = spv.OCL.sqrt %1 : vector<3xf16>
    ```
  }];
}

// -----

def SPV_OCLTanhOp : SPV_OCLUnaryArithmeticOp<"tanh", 63, SPV_Float> {
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
    tanh-op ::= ssa-id `=` `spv.OCL.tanh` ssa-use `:`
               float-scalar-vector-type
    ```mlir

    #### Example:

    ```
    %2 = spv.OCL.tanh %0 : f32
    %3 = spv.OCL.tanh %1 : vector<3xf16>
    ```
  }];
}

// -----

def SPV_OCLSAbsOp : SPV_OCLUnaryArithmeticOp<"s_abs", 141, SPV_Integer> {
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
    abs-op ::= ssa-id `=` `spv.OCL.s_abs` ssa-use `:`
               integer-scalar-vector-type
    ```
    #### Example:

    ```mlir
    %2 = spv.OCL.s_abs %0 : i32
    %3 = spv.OCL.s_abs %1 : vector<3xi16>
    ```
  }];
}

#endif // MLIR_DIALECT_SPIRV_IR_OCL_OPS
