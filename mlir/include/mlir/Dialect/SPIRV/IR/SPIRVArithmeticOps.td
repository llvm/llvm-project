//===-- SPIRVArithmeticOps.td - MLIR SPIR-V Arithmetic Ops -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains arithmetic ops for the SPIR-V dialect. It corresponds
// to "3.32.13. Arithmetic Instructions" of the SPIR-V specification.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SPIRV_IR_ARITHMETIC_OPS
#define MLIR_DIALECT_SPIRV_IR_ARITHMETIC_OPS

include "mlir/Dialect/SPIRV/IR/SPIRVBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class SPV_ArithmeticBinaryOp<string mnemonic, Type type,
                       list<Trait> traits = []> :
      // Operands type same as result type.
      SPV_BinaryOp<mnemonic, type, type,
                   !listconcat(traits,
                               [NoSideEffect, SameOperandsAndResultType])> {
  // In addition to normal types arithmetic instructions can support cooperative
  // matrix.
  let arguments = (ins
    SPV_ScalarOrVectorOrCoopMatrixOf<type>:$operand1,
    SPV_ScalarOrVectorOrCoopMatrixOf<type>:$operand2
  );

  let results = (outs
    SPV_ScalarOrVectorOrCoopMatrixOf<type>:$result
  );
  let assemblyFormat = "operands attr-dict `:` type($result)";
}

class SPV_ArithmeticUnaryOp<string mnemonic, Type type,
                            list<Trait> traits = []> :
      // Operand type same as result type.
      SPV_UnaryOp<mnemonic, type, type,
                   !listconcat(traits,
                               [NoSideEffect, SameOperandsAndResultType])>;

// -----

def SPV_FAddOp : SPV_ArithmeticBinaryOp<"FAdd", SPV_Float, [Commutative]> {
  let summary = "Floating-point addition of Operand 1 and Operand 2.";

  let description = [{
    Result Type must be a scalar or vector of floating-point type.

     The types of Operand 1 and Operand 2 both must be the same as Result
    Type.

     Results are computed per component.

    <!-- End of AutoGen section -->
    ```
    float-scalar-vector-type ::= float-type |
                                 `vector<` integer-literal `x` float-type `>`
    fadd-op ::= ssa-id `=` `spv.FAdd` ssa-use, ssa-use
                          `:` float-scalar-vector-type
    ```
    #### Example:

    ```mlir
    %4 = spv.FAdd %0, %1 : f32
    %5 = spv.FAdd %2, %3 : vector<4xf32>
    ```
  }];
}

// -----

def SPV_FDivOp : SPV_ArithmeticBinaryOp<"FDiv", SPV_Float, []> {
  let summary = "Floating-point division of Operand 1 divided by Operand 2.";

  let description = [{
    Result Type must be a scalar or vector of floating-point type.

     The types of Operand 1 and Operand 2 both must be the same as Result
    Type.

     Results are computed per component.  The resulting value is undefined
    if Operand 2 is 0.

    <!-- End of AutoGen section -->
    ```
    float-scalar-vector-type ::= float-type |
                                 `vector<` integer-literal `x` float-type `>`
    fdiv-op ::= ssa-id `=` `spv.FDiv` ssa-use, ssa-use
                          `:` float-scalar-vector-type
    ```

    #### Example:

    ```mlir
    %4 = spv.FDiv %0, %1 : f32
    %5 = spv.FDiv %2, %3 : vector<4xf32>
    ```
  }];
}

// -----

def SPV_FModOp : SPV_ArithmeticBinaryOp<"FMod", SPV_Float, []> {
  let summary = [{
    The floating-point remainder whose sign matches the sign of Operand 2.
  }];

  let description = [{
    Result Type must be a scalar or vector of floating-point type.

     The types of Operand 1 and Operand 2 both must be the same as Result
    Type.

     Results are computed per component.  The resulting value is undefined
    if Operand 2 is 0.  Otherwise, the result is the remainder r of Operand
    1 divided by Operand 2 where if r ≠ 0, the sign of r is the same as the
    sign of Operand 2.

    <!-- End of AutoGen section -->
    ```
    float-scalar-vector-type ::= float-type |
                                 `vector<` integer-literal `x` float-type `>`
    fmod-op ::= ssa-id `=` `spv.FMod` ssa-use, ssa-use
                          `:` float-scalar-vector-type
    ```
    #### Example:

    ```mlir
    %4 = spv.FMod %0, %1 : f32
    %5 = spv.FMod %2, %3 : vector<4xf32>
    ```
  }];
}

// -----

def SPV_FMulOp : SPV_ArithmeticBinaryOp<"FMul", SPV_Float, [Commutative]> {
  let summary = "Floating-point multiplication of Operand 1 and Operand 2.";

  let description = [{
    Result Type must be a scalar or vector of floating-point type.

     The types of Operand 1 and Operand 2 both must be the same as Result
    Type.

     Results are computed per component.

    <!-- End of AutoGen section -->

    ```
    float-scalar-vector-type ::= float-type |
                                 `vector<` integer-literal `x` float-type `>`
    fmul-op ::= `spv.FMul` ssa-use, ssa-use
                          `:` float-scalar-vector-type
    ```

    #### Example:

    ```mlir
    %4 = spv.FMul %0, %1 : f32
    %5 = spv.FMul %2, %3 : vector<4xf32>
    ```
  }];
}

// -----

def SPV_FNegateOp : SPV_ArithmeticUnaryOp<"FNegate", SPV_Float, []> {
  let summary = [{
    Inverts the sign bit of Operand. (Note, however, that OpFNegate is still
    considered a floating-point instruction, and so is subject to the
    general floating-point rules regarding, for example, subnormals and NaN
    propagation).
  }];

  let description = [{
    Result Type must be a scalar or vector of floating-point type.

     The type of Operand must be the same as Result Type.

     Results are computed per component.

    <!-- End of AutoGen section -->

    ```
    float-scalar-vector-type ::= float-type |
                                 `vector<` integer-literal `x` float-type `>`
    fmul-op ::= `spv.FNegate` ssa-use `:` float-scalar-vector-type
    ```

    #### Example:

    ```mlir
    %1 = spv.FNegate %0 : f32
    %3 = spv.FNegate %2 : vector<4xf32>
    ```
  }];
}

// -----

def SPV_FRemOp : SPV_ArithmeticBinaryOp<"FRem", SPV_Float, []> {
  let summary = [{
    The floating-point remainder whose sign matches the sign of Operand 1.
  }];

  let description = [{
    Result Type must be a scalar or vector of floating-point type.

     The types of Operand 1 and Operand 2 both must be the same as Result
    Type.

     Results are computed per component.  The resulting value is undefined
    if Operand 2 is 0.  Otherwise, the result is the remainder r of Operand
    1 divided by Operand 2 where if r ≠ 0, the sign of r is the same as the
    sign of Operand 1.

    <!-- End of AutoGen section -->
    ```
    float-scalar-vector-type ::= float-type |
                                 `vector<` integer-literal `x` float-type `>`
    frem-op ::= ssa-id `=` `spv.FRemOp` ssa-use, ssa-use
                          `:` float-scalar-vector-type
    ```

    #### Example:

    ```mlir
    %4 = spv.FRemOp %0, %1 : f32
    %5 = spv.FRemOp %2, %3 : vector<4xf32>
    ```
  }];
}

// -----

def SPV_FSubOp : SPV_ArithmeticBinaryOp<"FSub", SPV_Float, []> {
  let summary = "Floating-point subtraction of Operand 2 from Operand 1.";

  let description = [{
    Result Type must be a scalar or vector of floating-point type.

     The types of Operand 1 and Operand 2 both must be the same as Result
    Type.

     Results are computed per component.

    <!-- End of AutoGen section -->
    ```
    float-scalar-vector-type ::= float-type |
                                 `vector<` integer-literal `x` float-type `>`
    fsub-op ::= ssa-id `=` `spv.FRemOp` ssa-use, ssa-use
                          `:` float-scalar-vector-type
    ```

    #### Example:

    ```mlir
    %4 = spv.FRemOp %0, %1 : f32
    %5 = spv.FRemOp %2, %3 : vector<4xf32>
    ```
  }];
}

// -----

def SPV_IAddOp : SPV_ArithmeticBinaryOp<"IAdd",
                                        SPV_Integer,
                                        [Commutative, UsableInSpecConstantOp]> {
  let summary = "Integer addition of Operand 1 and Operand 2.";

  let description = [{
    Result Type must be a scalar or vector of integer type.

     The type of Operand 1 and Operand 2  must be a scalar or vector of
    integer type.  They must have the same number of components as Result
    Type. They must have the same component width as Result Type.

    The resulting value will equal the low-order N bits of the correct
    result R, where N is the component width and R is computed with enough
    precision to avoid overflow and underflow.

     Results are computed per component.

    <!-- End of AutoGen section -->
    ```
    integer-scalar-vector-type ::= integer-type |
                                 `vector<` integer-literal `x` integer-type `>`
    iadd-op ::= ssa-id `=` `spv.IAdd` ssa-use, ssa-use
                          `:` integer-scalar-vector-type
    ```

    #### Example:

    ```mlir
    %4 = spv.IAdd %0, %1 : i32
    %5 = spv.IAdd %2, %3 : vector<4xi32>

    ```
  }];

  let hasFolder = 1;
}

// -----

def SPV_IMulOp : SPV_ArithmeticBinaryOp<"IMul",
                                        SPV_Integer,
                                        [Commutative, UsableInSpecConstantOp]> {
  let summary = "Integer multiplication of Operand 1 and Operand 2.";

  let description = [{
    Result Type must be a scalar or vector of integer type.

     The type of Operand 1 and Operand 2  must be a scalar or vector of
    integer type.  They must have the same number of components as Result
    Type. They must have the same component width as Result Type.

    The resulting value will equal the low-order N bits of the correct
    result R, where N is the component width and R is computed with enough
    precision to avoid overflow and underflow.

     Results are computed per component.

    <!-- End of AutoGen section -->
    ```
    integer-scalar-vector-type ::= integer-type |
                                 `vector<` integer-literal `x` integer-type `>`
    imul-op ::= ssa-id `=` `spv.IMul` ssa-use, ssa-use
                          `:` integer-scalar-vector-type
    ```

    #### Example:

    ```mlir
    %4 = spv.IMul %0, %1 : i32
    %5 = spv.IMul %2, %3 : vector<4xi32>

    ```
  }];

  let hasFolder = 1;
}

// -----

def SPV_ISubOp : SPV_ArithmeticBinaryOp<"ISub",
                                        SPV_Integer,
                                        [UsableInSpecConstantOp]> {
  let summary = "Integer subtraction of Operand 2 from Operand 1.";

  let description = [{
    Result Type must be a scalar or vector of integer type.

     The type of Operand 1 and Operand 2  must be a scalar or vector of
    integer type.  They must have the same number of components as Result
    Type. They must have the same component width as Result Type.

    The resulting value will equal the low-order N bits of the correct
    result R, where N is the component width and R is computed with enough
    precision to avoid overflow and underflow.

     Results are computed per component.

    <!-- End of AutoGen section -->
    ```
    integer-scalar-vector-type ::= integer-type |
                                 `vector<` integer-literal `x` integer-type `>`
    isub-op ::= `spv.ISub` ssa-use, ssa-use
                          `:` integer-scalar-vector-type
    ```

    #### Example:

    ```mlir
    %4 = spv.ISub %0, %1 : i32
    %5 = spv.ISub %2, %3 : vector<4xi32>

    ```
  }];

  let hasFolder = 1;
}

// -----

def SPV_SDivOp : SPV_ArithmeticBinaryOp<"SDiv",
                                        SPV_Integer,
                                        [UsableInSpecConstantOp]> {
  let summary = "Signed-integer division of Operand 1 divided by Operand 2.";

  let description = [{
    Result Type must be a scalar or vector of integer type.

     The type of Operand 1 and Operand 2  must be a scalar or vector of
    integer type.  They must have the same number of components as Result
    Type. They must have the same component width as Result Type.

     Results are computed per component.  The resulting value is undefined
    if Operand 2 is 0.

    <!-- End of AutoGen section -->
    ```
    integer-scalar-vector-type ::= integer-type |
                                 `vector<` integer-literal `x` integer-type `>`
    sdiv-op ::= ssa-id `=` `spv.SDiv` ssa-use, ssa-use
                           `:` integer-scalar-vector-type
    ```

    #### Example:

    ```mlir
    %4 = spv.SDiv %0, %1 : i32
    %5 = spv.SDiv %2, %3 : vector<4xi32>

    ```
  }];
}

// -----

def SPV_SModOp : SPV_ArithmeticBinaryOp<"SMod",
                                        SPV_Integer,
                                        [UsableInSpecConstantOp]> {
  let summary = [{
    Signed remainder operation for the remainder whose sign matches the sign
    of Operand 2.
  }];

  let description = [{
    Result Type must be a scalar or vector of integer type.

     The type of Operand 1 and Operand 2  must be a scalar or vector of
    integer type.  They must have the same number of components as Result
    Type. They must have the same component width as Result Type.

     Results are computed per component.  The resulting value is undefined
    if Operand 2 is 0.  Otherwise, the result is the remainder r of Operand
    1 divided by Operand 2 where if r ≠ 0, the sign of r is the same as the
    sign of Operand 2.

    <!-- End of AutoGen section -->
    ```
    integer-scalar-vector-type ::= integer-type |
                                 `vector<` integer-literal `x` integer-type `>`
    smod-op ::= ssa-id `=` `spv.SMod` ssa-use, ssa-use
                           `:` integer-scalar-vector-type
    ```
    #### Example:

    ```mlir
    %4 = spv.SMod %0, %1 : i32
    %5 = spv.SMod %2, %3 : vector<4xi32>

    ```
  }];
}

// -----

def SPV_SNegateOp : SPV_ArithmeticUnaryOp<"SNegate",
                                          SPV_Integer,
                                          [UsableInSpecConstantOp]> {
  let summary = "Signed-integer subtract of Operand from zero.";

  let description = [{
    Result Type must be a scalar or vector of integer type.

    Operand’s type  must be a scalar or vector of integer type.  It must
    have the same number of components as Result Type.  The component width
    must equal the component width in Result Type.

     Results are computed per component.

    <!-- End of AutoGen section -->

    #### Example:

    ```mlir
    %1 = spv.SNegate %0 : i32
    %3 = spv.SNegate %2 : vector<4xi32>
    ```
  }];
}

// -----

def SPV_SRemOp : SPV_ArithmeticBinaryOp<"SRem",
                                        SPV_Integer,
                                        [UsableInSpecConstantOp]> {
  let summary = [{
    Signed remainder operation for the remainder whose sign matches the sign
    of Operand 1.
  }];

  let description = [{
    Result Type must be a scalar or vector of integer type.

     The type of Operand 1 and Operand 2  must be a scalar or vector of
    integer type.  They must have the same number of components as Result
    Type. They must have the same component width as Result Type.

     Results are computed per component.  The resulting value is undefined
    if Operand 2 is 0.  Otherwise, the result is the remainder r of Operand
    1 divided by Operand 2 where if r ≠ 0, the sign of r is the same as the
    sign of Operand 1.

    <!-- End of AutoGen section -->
    ```
    integer-scalar-vector-type ::= integer-type |
                                 `vector<` integer-literal `x` integer-type `>`
    srem-op ::= ssa-id `=` `spv.SRem` ssa-use, ssa-use
                           `:` integer-scalar-vector-type
    ```
    #### Example:

    ```mlir
    %4 = spv.SRem %0, %1 : i32
    %5 = spv.SRem %2, %3 : vector<4xi32>

    ```
  }];
}

// -----

def SPV_UDivOp : SPV_ArithmeticBinaryOp<"UDiv",
                                        SPV_Integer,
                                        [UnsignedOp, UsableInSpecConstantOp]> {
  let summary = "Unsigned-integer division of Operand 1 divided by Operand 2.";

  let description = [{
    Result Type must be a scalar or vector of integer type, whose Signedness
    operand is 0.

     The types of Operand 1 and Operand 2 both must be the same as Result
    Type.

     Results are computed per component.  The resulting value is undefined
    if Operand 2 is 0.

    <!-- End of AutoGen section -->
    ```
    integer-scalar-vector-type ::= integer-type |
                                 `vector<` integer-literal `x` integer-type `>`
    udiv-op ::= ssa-id `=` `spv.UDiv` ssa-use, ssa-use
                           `:` integer-scalar-vector-type
    ```
    #### Example:

    ```mlir
    %4 = spv.UDiv %0, %1 : i32
    %5 = spv.UDiv %2, %3 : vector<4xi32>

    ```
  }];
}

// -----

def SPV_VectorTimesScalarOp : SPV_Op<"VectorTimesScalar", [NoSideEffect]> {
  let summary = "Scale a floating-point vector.";

  let description = [{
    Result Type must be a vector of floating-point type.

     The type of Vector must be the same as Result Type. Each component of
    Vector is multiplied by Scalar.

    Scalar must have the same type as the Component Type in Result Type.

    <!-- End of AutoGen section -->

    #### Example:

    ```mlir
    %0 = spv.VectorTimesScalar %vector, %scalar : vector<4xf32>
    ```
  }];

  let arguments = (ins
    VectorOfLengthAndType<[2, 3, 4], [SPV_Float]>:$vector,
    SPV_Float:$scalar
  );

  let results = (outs
    VectorOfLengthAndType<[2, 3, 4], [SPV_Float]>:$result
  );

  let assemblyFormat = "operands attr-dict `:` `(` type(operands) `)` `->` type($result)";
}

// -----

def SPV_UModOp : SPV_ArithmeticBinaryOp<"UMod",
                                        SPV_Integer,
                                        [UnsignedOp, UsableInSpecConstantOp]> {
  let summary = "Unsigned modulo operation of Operand 1 modulo Operand 2.";

  let description = [{
    Result Type must be a scalar or vector of integer type, whose Signedness
    operand is 0.

     The types of Operand 1 and Operand 2 both must be the same as Result
    Type.

     Results are computed per component.  The resulting value is undefined
    if Operand 2 is 0.

    <!-- End of AutoGen section -->
    ```
    integer-scalar-vector-type ::= integer-type |
                                 `vector<` integer-literal `x` integer-type `>`
    umod-op ::= ssa-id `=` `spv.UMod` ssa-use, ssa-use
                           `:` integer-scalar-vector-type
    ```
    #### Example:

    ```mlir
    %4 = spv.UMod %0, %1 : i32
    %5 = spv.UMod %2, %3 : vector<4xi32>

    ```
  }];
}

#endif // MLIR_DIALECT_SPIRV_IR_ARITHMETIC_OPS
