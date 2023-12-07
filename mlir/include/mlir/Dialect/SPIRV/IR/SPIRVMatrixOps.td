//===-- SPIRVMatrixOps.td - MLIR SPIR-V Matrix Ops ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains matrix operations for the SPIR-V dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SPIRV_IR_MATRIX_OPS
#define MLIR_DIALECT_SPIRV_IR_MATRIX_OPS
include "mlir/Interfaces/SideEffectInterfaces.td"

// -----

def SPIRV_MatrixTimesMatrixOp : SPIRV_Op<"MatrixTimesMatrix", [Pure]> {
  let summary = "Linear-algebraic multiply of LeftMatrix X RightMatrix.";

  let description = [{
    Result Type must be an OpTypeMatrix whose Column Type is a vector of
    floating-point type.

    LeftMatrix must be a matrix whose Column Type is the same as the Column
    Type in Result Type.

    RightMatrix must be a matrix with the same Component Type as the
    Component Type in Result Type. Its number of columns must equal the
    number of columns in Result Type. Its columns must have the same number
    of components as the number of columns in LeftMatrix.

    <!-- End of AutoGen section -->

    ```
    matrix-times-matrix-op ::= ssa-id `=` `spirv.MatrixTimesMatrix` ssa-use,
    ssa-use `:` matrix-type `,` matrix-type `->` matrix-type
    ```

    #### Example:

    ```mlir
    %0 = spirv.MatrixTimesMatrix %matrix_1, %matrix_2 :
        !spirv.matrix<4 x vector<3xf32>>, !spirv.matrix<3 x vector<4xf32>> ->
        !spirv.matrix<4 x vector<4xf32>>
    ```
  }];

  let availability = [
    MinVersion<SPIRV_V_1_0>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[]>,
    Capability<[SPIRV_C_Matrix]>
  ];

  let arguments = (ins
    SPIRV_AnyMatrix:$leftmatrix,
    SPIRV_AnyMatrix:$rightmatrix
  );

  let results = (outs
    SPIRV_AnyMatrix:$result
  );

  let assemblyFormat = [{
    operands attr-dict `:` type($leftmatrix) `,` type($rightmatrix) `->` type($result)
  }];
}

// -----

def SPIRV_MatrixTimesScalarOp : SPIRV_Op<
    "MatrixTimesScalar", [Pure, AllTypesMatch<["matrix", "result"]>]> {
  let summary = "Scale a floating-point matrix.";

  let description = [{
    Result Type must be an OpTypeMatrix whose Column Type is a vector of
    floating-point type.

     The type of Matrix must be the same as Result Type. Each component in
    each column in Matrix is multiplied by Scalar.

    Scalar must have the same type as the Component Type in Result Type.

    <!-- End of AutoGen section -->

    ```
    matrix-times-scalar-op ::= ssa-id `=` `spirv.MatrixTimesScalar` ssa-use,
    ssa-use `:` matrix-type `,` float-type `->` matrix-type

    ```

    #### Example:

    ```mlir
    %0 = spirv.MatrixTimesScalar %matrix, %scalar :
    !spirv.matrix<3 x vector<3xf32>>, f32 -> !spirv.matrix<3 x vector<3xf32>>

    ```
  }];

  let availability = [
    MinVersion<SPIRV_V_1_0>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[]>,
    Capability<[SPIRV_C_Matrix]>
  ];

  let arguments = (ins
    SPIRV_MatrixOrCoopMatrixOf<SPIRV_Float>:$matrix,
    SPIRV_Float:$scalar
  );

  let results = (outs
    SPIRV_MatrixOrCoopMatrixOf<SPIRV_Float>:$result
  );

  let assemblyFormat = [{
    operands attr-dict `:` type($matrix) `,` type($scalar)
  }];

  let availability = [
    MinVersion<SPIRV_V_1_0>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[]>,
    Capability<[SPIRV_C_Matrix]>
  ];
}

// -----

def SPIRV_TransposeOp : SPIRV_Op<"Transpose", [Pure]> {
  let summary = "Transpose a matrix.";

  let description = [{
    Result Type must be an OpTypeMatrix.

    Matrix must be an object of type OpTypeMatrix. The number of columns and
    the column size of Matrix must be the reverse of those in Result Type.
    The types of the scalar components in Matrix and Result Type must be the
    same.

    Matrix must have of type of OpTypeMatrix.

    <!-- End of AutoGen section -->

    ```
    transpose-op ::= ssa-id `=` `spirv.Transpose` ssa-use `:` matrix-type `->`
    matrix-type
    ```


    #### Example:

    ```mlir
    %0 = spirv.Transpose %matrix: !spirv.matrix<2 x vector<3xf32>> ->
    !spirv.matrix<3 x vector<2xf32>>

    ```
  }];

  let availability = [
    MinVersion<SPIRV_V_1_0>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[]>,
    Capability<[SPIRV_C_Matrix]>
  ];

  let arguments = (ins
    SPIRV_AnyMatrix:$matrix
  );

  let results = (outs
    SPIRV_AnyMatrix:$result
  );

  let assemblyFormat = [{
    operands attr-dict `:` type($matrix) `->` type($result)
  }];
}

// -----

#endif // MLIR_DIALECT_SPIRV_IR_MATRIX_OPS
