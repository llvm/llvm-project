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

#ifndef SPIRV_MATRIX_OPS
#define SPIRV_MATRIX_OPS
include "mlir/Interfaces/SideEffectInterfaces.td"

// -----

def SPV_MatrixTimesMatrixOp : SPV_Op<"MatrixTimesMatrix", [NoSideEffect]> {
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
    matrix-times-matrix-op ::= ssa-id `=` `spv.MatrixTimesMatrix` ssa-use,
    ssa-use `:` matrix-type `,` matrix-type `->` matrix-type
    ```mlir

    #### Example:

    ```
    %0 = spv.MatrixTimesMatrix %matrix_1, %matrix_2 :
        !spv.matrix<4 x vector<3xf32>>, !spv.matrix<3 x vector<4xf32>> ->
        !spv.matrix<4 x vector<4xf32>>
    ```
  }];

  let availability = [
    MinVersion<SPV_V_1_0>,
    MaxVersion<SPV_V_1_5>,
    Extension<[]>,
    Capability<[SPV_C_Matrix]>
  ];

  let arguments = (ins
    SPV_AnyMatrix:$leftmatrix,
    SPV_AnyMatrix:$rightmatrix
  );

  let results = (outs
    SPV_AnyMatrix:$result
  );
  let assemblyFormat = [{
    operands attr-dict `:` type($leftmatrix) `,` type($rightmatrix) `->` type($result)
  }];
  let verifier = [{ return verifyMatrixTimesMatrix(*this); }];
}

// -----

def SPV_MatrixTimesScalarOp : SPV_Op<"MatrixTimesScalar", [NoSideEffect]> {
  let summary = "Scale a floating-point matrix.";

  let description = [{
    Result Type must be an OpTypeMatrix whose Column Type is a vector of
    floating-point type.

     The type of Matrix must be the same as Result Type. Each component in
    each column in Matrix is multiplied by Scalar.

    Scalar must have the same type as the Component Type in Result Type.

    <!-- End of AutoGen section -->

    ```
    matrix-times-scalar-op ::= ssa-id `=` `spv.MatrixTimesScalar` ssa-use,
    ssa-use `:` matrix-type `,` float-type `->` matrix-type

    ```

    #### Example:

    ```mlir

    %0 = spv.MatrixTimesScalar %matrix, %scalar :
    !spv.matrix<3 x vector<3xf32>>, f32 -> !spv.matrix<3 x vector<3xf32>>

    ```
  }];

  let availability = [
    MinVersion<SPV_V_1_0>,
    MaxVersion<SPV_V_1_5>,
    Extension<[]>,
    Capability<[SPV_C_Matrix]>
  ];

  let arguments = (ins
    SPV_AnyMatrix:$matrix,
    SPV_Float:$scalar
  );

  let results = (outs
    SPV_AnyMatrix:$result
  );

  // TODO: we need just one matrix type given that the input and result are the
  // same and the scalar's type can be deduced from it.
  let assemblyFormat = [{
    operands attr-dict `:` type($matrix) `,` type($scalar) `->` type($result)
  }];

  let availability = [
    MinVersion<SPV_V_1_0>,
    MaxVersion<SPV_V_1_5>,
    Extension<[]>,
    Capability<[SPV_C_Matrix]>
  ];

  let verifier = [{ return verifyMatrixTimesScalar(*this); }];
}

// -----

def SPV_TransposeOp : SPV_Op<"Transpose", [NoSideEffect]> {
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
    transpose-op ::= ssa-id `=` `spv.Transpose` ssa-use `:` matrix-type `->`
    matrix-type

    ```mlir

    #### Example:

    ```
    %0 = spv.Transpose %matrix: !spv.matrix<2 x vector<3xf32>> ->
    !spv.matrix<3 x vector<2xf32>>

    ```
  }];

  let availability = [
    MinVersion<SPV_V_1_0>,
    MaxVersion<SPV_V_1_5>,
    Extension<[]>,
    Capability<[SPV_C_Matrix]>
  ];

  let arguments = (ins
    SPV_AnyMatrix:$matrix
  );

  let results = (outs
    SPV_AnyMatrix:$result
  );

  let assemblyFormat = [{
    operands attr-dict `:` type($matrix) `->` type($result)
  }];

  let verifier = [{ return verifyTranspose(*this); }];
}

// -----

#endif // SPIRV_MATRIX_OPS
