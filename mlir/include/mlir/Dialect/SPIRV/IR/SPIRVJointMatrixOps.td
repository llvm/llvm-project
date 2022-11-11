//===- SPIRVJointMatrixOps.td - joint matmul ---------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the op definition spec of joint matrix multiply extension ops.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SPIRV_IR_JOINT_MATRIX_OPS
#define MLIR_DIALECT_SPIRV_IR_JOINT_MATRIX_OPS

// -----

def SPIRV_INTELJointMatrixWorkItemLengthOp : SPIRV_IntelVendorOp<"JointMatrixWorkItemLength",
  [Pure]> {
  let summary = "See extension SPV_INTEL_joint_matrix";

  let description = [{
    Return number of components owned by the current work-item in
    a joint matrix.

    Result Type must be an 32-bit unsigned integer type scalar.

    Type is a joint matrix type.

    ``` {.ebnf}
    joint-matrix-length-op ::= ssa-id `=` `spirv.INTEL.JointMatrixWorkItemLength
                                    ` : ` joint-matrix-type
    ```

    For example:

    ```
    %0 = spirv.INTEL.JointMatrixWorkItemLength : !spirv.jointmatrix<Subgroup, i32, 8, 16>
    ```
  }];

  let assemblyFormat = "attr-dict `:` $joint_matrix_type";

  let availability = [
    MinVersion<SPIRV_V_1_0>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[SPV_INTEL_joint_matrix]>,
    Capability<[SPIRV_C_JointMatrixINTEL]>
  ];

  let arguments = (ins
    TypeAttr:$joint_matrix_type
  );

  let results = (outs
    SPIRV_Int32:$result
  );
  let hasVerifier = 0;
}

// -----

def SPIRV_INTELJointMatrixLoadOp : SPIRV_IntelVendorOp<"JointMatrixLoad", []> {
  let summary = "See extension SPV_INTEL_joint_matrix";

  let description = [{
    Load a matrix through a pointer.

    Result Type is the type of the loaded matrix. It must be OpTypeJointMatrixINTEL.

    Pointer is the pointer to load through. It specifies start of memory region where
    elements of the matrix are stored and arranged according to Layout.

    Stride is the number of elements in memory between beginnings of successive rows,
    columns (or words) in the result. It must be a scalar integer type.

    Layout indicates how the values loaded from memory are arranged. It must be the
    result of a constant instruction.

    Scope is syncronization scope for operation on the matrix. It must be the result
    of a constant instruction with scalar integer type.

    If present, any Memory Operands must begin with a memory operand literal. If not
    present, it is the same as specifying the memory operand None.

    #### Example:
    ```mlir
    %0 = spirv.INTEL.JointMatrixLoad <Subgroup> <RowMajor> %ptr, %stride
         {memory_access = #spirv.memory_access<Volatile>} :
         (!spirv.ptr<i32, CrossWorkgroup>, i32) ->
         !spirv.jointmatrix<8x16xi32, ColumnMajor, Subgroup>
    ```
  }];

  let assemblyFormat = [{
    $scope $layout operands attr-dict `:` `(` type(operands) `)` `->` type($result)
  }];

  let availability = [
    MinVersion<SPIRV_V_1_0>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[SPV_INTEL_joint_matrix]>,
    Capability<[SPIRV_C_JointMatrixINTEL]>
  ];

  let arguments = (ins
    SPIRV_AnyPtr:$pointer,
    SPIRV_Integer:$stride,
    SPIRV_MatrixLayoutAttr:$layout,
    SPIRV_ScopeAttr:$scope,
    OptionalAttr<SPIRV_MemoryAccessAttr>:$memory_access,
    OptionalAttr<I32Attr>:$alignment
  );

  let results = (outs
    SPIRV_AnyJointMatrix:$result
  );
}

// -----

def SPIRV_INTELJointMatrixMadOp : SPIRV_IntelVendorOp<"JointMatrixMad",
  [Pure, AllTypesMatch<["c", "result"]>]> {
  let summary = "See extension SPV_INTEL_joint_matrix";

  let description = [{
    Multiply matrix A by matrix B and add matrix C to the result
    of the multiplication: A*B+C. Here A is a M x K matrix, B is
    a K x N matrix and C is a M x N matrix.

    Behavior is undefined if sizes of operands do not meet the
    conditions above. All operands and the Result Type must be
    OpTypeJointMatrixINTEL.

    A must be a OpTypeJointMatrixINTEL whose Component Type is a
    signed numerical type, Row Count equals to M and Column Count
    equals to K

    B must be a OpTypeJointMatrixINTEL whose Component Type is a
    signed numerical type, Row Count equals to K and Column Count
    equals to N

    C and Result Type must be a OpTypeJointMatrixINTEL with Row
    Count equals to M and Column Count equals to N

    Scope is syncronization scope for operation on the matrix.
    It must be the result of a constant instruction with scalar
    integer type.

    #### Example:
    ```mlir
    %r = spirv.INTEL.JointMatrixMad <Subgroup> %a, %b, %c :
         !spirv.jointmatrix<8x32xi8, RowMajor, Subgroup>,
         !spirv.jointmatrix<32x8xi8, ColumnMajor, Subgroup>
         -> !spirv.jointmatrix<8x8xi32,  RowMajor, Subgroup>
    ```

  }];

  let assemblyFormat = [{
    $scope operands attr-dict`:` type($a) `,` type($b) `->` type($c)
  }];

  let availability = [
    MinVersion<SPIRV_V_1_0>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[SPV_INTEL_joint_matrix]>,
    Capability<[SPIRV_C_JointMatrixINTEL]>
  ];

  let arguments = (ins
    SPIRV_AnyJointMatrix:$a,
    SPIRV_AnyJointMatrix:$b,
    SPIRV_AnyJointMatrix:$c,
    SPIRV_ScopeAttr:$scope
  );

  let results = (outs
    SPIRV_AnyJointMatrix:$result
  );
}

// -----

def SPIRV_INTELJointMatrixStoreOp : SPIRV_IntelVendorOp<"JointMatrixStore", []> {
  let summary = "See extension SPV_INTEL_joint_matrix";

  let description = [{
    Store a matrix through a pointer.

    Pointer is the pointer to store through. It specifies
    start of memory region where elements of the matrix must
    be stored and arranged according to Layout.

    Object is the matrix to store. It must be
    OpTypeJointMatrixINTEL.

    Stride is the number of elements in memory between beginnings
    of successive rows, columns (or words) of the Object. It must
    be a scalar integer type.

    Layout indicates how the values stored to memory are arranged.
    It must be the result of a constant instruction.

    Scope is syncronization scope for operation on the matrix.
    It must be the result of a constant instruction with scalar
    integer type.

    If present, any Memory Operands must begin with a memory operand
    literal. If not present, it is the same as specifying the memory
    operand None.

    #### Example:
    ```mlir
    spirv.INTEL.JointMatrixStore <Subgroup> <ColumnMajor> %ptr, %m, %stride
    {memory_access = #spirv.memory_access<Volatile>} : (!spirv.ptr<i32, Workgroup>,
    !spirv.jointmatrix<8x16xi32, RowMajor, Subgroup>, i32)
    ```

  }];

   let assemblyFormat = [{
    $scope $layout operands attr-dict `:` `(` type(operands) `)`
  }];

  let availability = [
    MinVersion<SPIRV_V_1_0>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[SPV_INTEL_joint_matrix]>,
    Capability<[SPIRV_C_JointMatrixINTEL]>
  ];

  let arguments = (ins
    SPIRV_AnyPtr:$pointer,
    SPIRV_AnyJointMatrix:$object,
    SPIRV_Integer:$stride,
    SPIRV_MatrixLayoutAttr:$layout,
    SPIRV_ScopeAttr:$scope,
    OptionalAttr<SPIRV_MemoryAccessAttr>:$memory_access,
    OptionalAttr<I32Attr>:$alignment
  );

  let results = (outs);
}

// -----

#endif // MLIR_DIALECT_SPIRV_IR_JOINT_MATRIX_OPS
