//===- SPIRVCooperativeMatrixOps.td - cooperative matmul ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the op definition spec of cooperative matrix multiply extension ops.
// We support both cooperative matrix extensions:
//  - SPV_NV_cooperative_matrix
//  - SPV_KHR_cooperative_matrix
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SPIRV_IR_COOPERATIVE_MATRIX_OPS
#define MLIR_DIALECT_SPIRV_IR_COOPERATIVE_MATRIX_OPS

//===----------------------------------------------------------------------===//
// SPV_KHR_cooperative_matrix extension ops.
//===----------------------------------------------------------------------===//

// -----

def SPIRV_KHRCooperativeMatrixLengthOp :
      SPIRV_KhrVendorOp<"CooperativeMatrixLength", [Pure]> {
  let summary = "Queries the number of cooperative matrix components";

  let description = [{
    Number of components of a cooperative matrix type accessible to each
    invocation when treated as a composite.

    The type attribute must be a cooperative matrix type.

    ``` {.ebnf}
    cooperative-matrix-length-op ::= ssa-id `=` `spirv.KHR.CooperativeMatrixLength
                                    ` : ` cooperative-matrix-type
    ```

    #### Example:

    ```
    %0 = spirv.KHR.CooperativeMatrixLength :
           !spirv.coopmatrix<8x16xi32, Subgroup, MatrixA>
    ```
  }];

  let assemblyFormat = "attr-dict `:` $cooperative_matrix_type";

  let availability = [
    MinVersion<SPIRV_V_1_6>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[SPV_KHR_cooperative_matrix]>,
    Capability<[SPIRV_C_CooperativeMatrixKHR]>
  ];

  let arguments = (ins
    TypeAttr:$cooperative_matrix_type
  );

  let results = (outs
    SPIRV_Int32:$result
  );
}

// -----

def SPIRV_KHRCooperativeMatrixLoadOp : SPIRV_KhrVendorOp<"CooperativeMatrixLoad", []> {
  let summary = "Loads a cooperative matrix through a pointer";

  let description = [{
    Load a cooperative matrix through a pointer.

    Result Type is the type of the loaded object. It must be a cooperative
    matrix type.

    Pointer is a pointer. Its type must be an OpTypePointer whose Type operand is
    a scalar or vector type. If the Shader capability was declared, Pointer must
    point into an array and any ArrayStride decoration on Pointer is ignored.

    MemoryLayout specifies how matrix elements are laid out in memory. It must
    come from a 32-bit integer constant instruction whose value corresponds to a
    Cooperative Matrix Layout. See the Cooperative Matrix Layout table for a
    description of the layouts and detailed layout-specific rules.

    Stride further qualifies how matrix elements are laid out in memory. It must
    be a scalar integer type and its exact semantics depend on MemoryLayout.

    Memory Operand must be a Memory Operand literal. If not present, it is the
    same as specifying None.

    NOTE: In earlier versions of the SPIR-V spec, 'Memory Operand' was known
    as 'Memory Access'.

    For a given dynamic instance of this instruction, all operands of this
    instruction must be the same for all invocations in a given scope instance
    (where the scope is the scope the cooperative matrix type was created with).
    All invocations in a given scope instance must be active or all must be
    inactive.

    ``` {.ebnf}
    cooperative-matrix-load-op ::= ssa-id `=` `spirv.KHR.CooperativeMatrixLoad`
                              ssa-use `,` ssa-use `,`
                              `<` cooperative-matrix-layout `>`
                              (`,` `<` memory-operand `>`)? `:`
                                pointer-type `,` stride-type `->` cooperative-matrix-type
    ```

    TODO: In the SPIR-V spec, `stride` is an optional argument. We should also
    support this optionality in the SPIR-V dialect.

    #### Example:

    ```
    %0 = spirv.KHR.CooperativeMatrixLoad %ptr, %stride, <RowMajor>
         : !spirv.ptr<i32, StorageBuffer>, i32
             -> !spirv.KHR.coopmatrix<16x8xi32, Workgroup, MatrixA>

    %1 = spirv.KHR.CooperativeMatrixLoad %ptr, %stride, <ColumnMajor>, <Volatile>
         : !spirv.ptr<f32, StorageBuffer>, i64
             -> !spirv.KHR.coopmatrix<8x8xf32, Subgroup, MatrixAcc>
    ```
  }];

  let assemblyFormat = [{
    $pointer `,` $stride `,` $matrix_layout ( `,` $memory_operand^ )? attr-dict `:`
      type(operands) `->` type($result)
  }];

  let availability = [
    MinVersion<SPIRV_V_1_6>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[SPV_KHR_cooperative_matrix]>,
    Capability<[SPIRV_C_CooperativeMatrixKHR]>
  ];

  let arguments = (ins
    SPIRV_AnyPtr:$pointer,
    SPIRV_KHR_CooperativeMatrixLayoutAttr:$matrix_layout,
    SPIRV_Integer:$stride,
    OptionalAttr<SPIRV_MemoryAccessAttr>:$memory_operand
  );

  let results = (outs
    SPIRV_AnyCooperativeMatrix:$result
  );
}

// -----

def SPIRV_KHRCooperativeMatrixStoreOp : SPIRV_KhrVendorOp<"CooperativeMatrixStore", []> {
  let summary = "Stores a cooperative matrix through a pointer";

  let description = [{
    Store a cooperative matrix through a pointer.
    Pointer is a pointer. Its type must be an OpTypePointer whose Type operand
    is a scalar or vector type. If the Shader capability was declared, Pointer
    must point into an array and any ArrayStride decoration on Pointer is
    ignored.

    Object is the object to store. Its type must be an
    OpTypeCooperativeMatrixKHR.

    MemoryLayout specifies how matrix elements are laid out in memory. It must
    come from a 32-bit integer constant instruction whose value corresponds to a
    Cooperative Matrix Layout. See the Cooperative Matrix Layout table for a
    description of the layouts and detailed layout-specific rules.

    Stride further qualifies how matrix elements are laid out in memory. It must
    be a scalar integer type and its exact semantics depend on MemoryLayout.

    Memory Operand must be a Memory Operand literal. If not present, it is the
    same as specifying None.

    NOTE: In earlier versions of the SPIR-V spec, 'Memory Operand' was known
    as 'Memory Access'.

    For a given dynamic instance of this instruction, all operands of this
    instruction must be the same for all invocations in a given scope instance
    (where the scope is the scope the cooperative matrix type was created with).
    All invocations in a given scope instance must be active or all must be
    inactive.

    ``` {.ebnf}
     coop-matrix-store-op ::= `spirv.KHR.CooperativeMatrixStore`
                              ssa-use `,` ssa-use `,`
                              ssa-use `,` `<` cooperative-matrix-layout `>`
                              (`,` `<` memory-operand `>`)? `:`
                              pointer-type `,` coop-matrix-type `,` stride-type
    ```

    TODO: In the SPIR-V spec, `stride` is an optional argument. We should also
    support this optionality in the SPIR-V dialect.

    #### Example:

    ```
      spirv.KHR.CooperativeMatrixStore %ptr, %obj, %stride, <RowMajor> :
        !spirv.ptr<i32, StorageBuffer>, !spirv.coopmatrix<16x8xi32, Workgroup, MatrixA>, i32

      spirv.KHR.CooperativeMatrixStore %ptr, %obj, %stride, <ColumnMajor>, <Volatile> :
        !spirv.ptr<f32, StorageBuffer>, !spirv.coopmatrix<8x8xf32, Subgroup, MatrixAcc>, i64
    ```
  }];

  let assemblyFormat = [{
    $pointer `,` $object `,` $stride `,` $matrix_layout ( `,` $memory_operand^ )? attr-dict `:`
      type(operands)
  }];

  let availability = [
    MinVersion<SPIRV_V_1_6>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[SPV_KHR_cooperative_matrix]>,
    Capability<[SPIRV_C_CooperativeMatrixKHR]>
  ];

  let arguments = (ins
    SPIRV_AnyPtr:$pointer,
    SPIRV_AnyCooperativeMatrix:$object,
    SPIRV_KHR_CooperativeMatrixLayoutAttr:$matrix_layout,
    SPIRV_Integer:$stride,
    OptionalAttr<SPIRV_MemoryAccessAttr>:$memory_operand
  );

  let results = (outs);
}

// -----

def SPIRV_KHRCooperativeMatrixMulAddOp : SPIRV_KhrVendorOp<"CooperativeMatrixMulAdd",
  [Pure, AllTypesMatch<["c", "result"]>]> {
  let summary = "Returns the result of `(A x B) + C` of matrices A, B, and C";

  let description = [{
    Linear-algebraic matrix multiply of A by B and then component-wise add C.
    The order of the operations is implementation-dependent. The internal
    precision of floating-point operations is defined by the client API. Integer
    operations used in the multiplication of A by B are performed at the
    precision of the Result Type and the resulting value will equal the
    low-order N bits of the correct result R, where N is the result width and R
    is computed with enough precision to avoid overflow and underflow if the
    SaturatingAccumulation Cooperative Matrix Operand is not present. If the
    SaturatingAccumulation Cooperative Matrix Operand is present and overflow or
    underflow occurs as part of calculating that intermediate result, the result
    of the instruction is undefined. Integer additions of the elements of that
    intermediate result with those of C are performed at the precision of Result
    Type, are exact, and are saturating if the SaturatingAccumulation
    Cooperative Matrix Operand is present, with the signedness of the saturation
    being that of the components of Result Type. If the SaturatingAccumulation
    Cooperative Matrix Operand is not present then the resulting value will
    equal the low-order N bits of the correct result R, where N is the result
    width and R is computed with enough precision to avoid overflow and
    underflow.

    Result Type must be a cooperative matrix type with M rows and N columns
    whose Use must be MatrixAccumulatorKHR.

    A is a cooperative matrix with M rows and K columns whose Use must be
    MatrixAKHR.

    B is a cooperative matrix with K rows and N columns whose Use must be
    MatrixBKHR.

    C is a cooperative matrix with M rows and N columns whose Use must be
    MatrixAccumulatorKHR.

    The values of M, N, and K must be consistent across the result and operands.
    This is referred to as an MxNxK matrix multiply.

    A, B, C, and Result Type must have the same scope, and this defines the
    scope of the operation. A, B, C, and Result Type need not necessarily have
    the same component type, this is defined by the client API.

    If the Component Type of any matrix operand is an integer type, then its
    components are treated as signed if the Matrix{A,B,C,Result}SignedComponents
    Cooperative Matrix Operand is present and are treated as unsigned otherwise.

    Cooperative Matrix Operands is an optional Cooperative Matrix Operand
    literal. If not present, it is the same as specifying the Cooperative Matrix
    Operand None.

    For a given dynamic instance of this instruction, all invocations in a given
    scope instance must be active or all must be inactive (where the scope is
    the scope of the operation).

    ``` {.ebnf}
    cooperative-matrixmuladd-op ::= ssa-id `=` `spirv.KHR.CooperativeMatrixMulAdd`
                              ssa-use `,` ssa-use `,` ssa-use
                              (`<` matrix-operands `>`)? `:`
                              a-cooperative-matrix-type `,`
                              b-cooperative-matrix-type `->`
                                result-cooperative-matrix-type
    ```

    #### Example:

    ```
    %0 = spirv.KHR.CooperativeMatrixMulAdd %matA, %matB, %matC :
      !spirv.coopmatrix<4x4xf32, Subgroup, MatrixA>,
      !spirv.coopmatrix<4x4xf32, Subgroup, MatrixB> ->
        !spirv.coopmatrix<4x4xf32, Subgroup, MatrixAcc>

    %1 = spirv.KHR.CooperativeMatrixMulAdd %matA, %matB, %matC, <ASigned | AccSat> :
      !spirv.coopmatrix<8x16xi32, Subgroup, MatrixA>,
      !spirv.coopmatrix<16x4xi32, Subgroup, MatrixB> ->
        !spirv.coopmatrix<8x4xi32, Subgroup, MatrixAcc>
    ```
  }];

  let assemblyFormat = [{
    $a `,` $b `,` $c ( `,` $matrix_operands^ )? attr-dict `:`
      type($a) `,` type($b) `->` type($c)
  }];

  let availability = [
    MinVersion<SPIRV_V_1_6>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[SPV_KHR_cooperative_matrix]>,
    Capability<[SPIRV_C_CooperativeMatrixKHR]>
  ];

  let arguments = (ins
    SPIRV_AnyCooperativeMatrix:$a,
    SPIRV_AnyCooperativeMatrix:$b,
    SPIRV_AnyCooperativeMatrix:$c,
    OptionalAttr<SPIRV_KHR_CooperativeMatrixOperandsAttr>:$matrix_operands
  );

  let results = (outs
    SPIRV_AnyCooperativeMatrix:$result
  );
}

//===----------------------------------------------------------------------===//
// SPV_NV_cooperative_matrix extension ops.
//===----------------------------------------------------------------------===//

// -----

def SPIRV_NVCooperativeMatrixLengthOp : SPIRV_NvVendorOp<"CooperativeMatrixLength",
  [Pure]> {
  let summary = "See extension SPV_NV_cooperative_matrix";

  let description = [{
    Number of components of a cooperative matrix type accessible to each
    invocation when treated as a composite.

    Result Type must be an OpTypeInt with 32-bit Width and 0 Signedness.

    Type is a cooperative matrix type.

    ``` {.ebnf}
    cooperative-matrix-length-op ::= ssa-id `=` `spirv.NV.CooperativeMatrixLength
                                    ` : ` cooperative-matrix-type
    ```

    For example:

    ```
    %0 = spirv.NV.CooperativeMatrixLength : !spirv.NV.coopmatrix<8x16xi32, Subgroup>
    ```
  }];

  let assemblyFormat = "attr-dict `:` $cooperative_matrix_type";

  let availability = [
    MinVersion<SPIRV_V_1_0>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[SPV_NV_cooperative_matrix]>,
    Capability<[SPIRV_C_CooperativeMatrixNV]>
  ];

  let arguments = (ins
    TypeAttr:$cooperative_matrix_type
  );

  let results = (outs
    SPIRV_Int32:$result
  );
}

// -----

def SPIRV_NVCooperativeMatrixLoadOp : SPIRV_NvVendorOp<"CooperativeMatrixLoad", []> {
  let summary = "See extension SPV_NV_cooperative_matrix";

  let description = [{
    Load a cooperative matrix through a pointer.

    Result Type is the type of the loaded object. It must be a cooperative
    matrix type.

    Pointer is a pointer into an array. Its type must be an OpTypePointer whose
    Type operand is a scalar or vector type. The storage class of Pointer must
    be Workgroup, StorageBuffer, or (if SPV_EXT_physical_storage_buffer is
    supported) PhysicalStorageBufferEXT.

    Stride is the number of elements in the array in memory between the first
    component of consecutive rows (or columns) in the result. It must be a
    scalar integer type.

    ColumnMajor indicates whether the values loaded from memory are arranged in
    column-major or row-major order. It must be a boolean constant instruction,
    with false indicating row major and true indicating column major.

    Memory Access must be a Memory Access literal. If not present, it is the
    same as specifying None.

    If ColumnMajor is false, then elements (row,*) of the result are taken in
    order from contiguous locations starting at Pointer[row*Stride]. If
    ColumnMajor is true, then elements (*,col) of the result are taken in order
    from contiguous locations starting from Pointer[col*Stride]. Any ArrayStride
    decoration on Pointer is ignored.

    For a given dynamic instance of this instruction, all operands of this
    instruction must be the same for all invocations in a given scope instance
    (where the scope is the scope the cooperative matrix type was created with).
    All invocations in a given scope instance must be active or all must be
    inactive.

    ### Custom assembly form

    ``` {.ebnf}
    cooperative-matrixload-op ::= ssa-id `=` `spirv.NV.CooperativeMatrixLoad`
                              ssa-use `,` ssa-use `,` ssa-use
                              (`[` memory-access `]`)? ` : `
                              pointer-type `as`
                              cooperative-matrix-type
    ```

    For example:

    ```
    %0 = spirv.NV.CooperativeMatrixLoad %ptr, %stride, %colMajor
         : !spirv.ptr<i32, StorageBuffer> as !spirv.NV.coopmatrix<16x8xi32, Workgroup>
    ```
  }];

  let availability = [
    MinVersion<SPIRV_V_1_0>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[SPV_NV_cooperative_matrix]>,
    Capability<[SPIRV_C_CooperativeMatrixNV]>
  ];

  let arguments = (ins
    SPIRV_AnyPtr:$pointer,
    SPIRV_Integer:$stride,
    SPIRV_Bool:$columnmajor,
    OptionalAttr<SPIRV_MemoryAccessAttr>:$memory_access
  );

  let results = (outs
    SPIRV_AnyCooperativeMatrixNV:$result
  );
}

// -----

def SPIRV_NVCooperativeMatrixMulAddOp : SPIRV_NvVendorOp<"CooperativeMatrixMulAdd",
  [Pure, AllTypesMatch<["c", "result"]>]> {
  let summary = "See extension SPV_NV_cooperative_matrix";

  let description = [{
    Linear-algebraic matrix multiply of A by B and then component-wise add C.
    The order of the operations is implementation-dependent. The internal
    precision of floating-point operations is defined by the client API.
    Integer operations are performed at the precision of the Result Type and are
    exact unless there is overflow or underflow, in which case the result is
    undefined.

    Result Type must be a cooperative matrix type with M rows and N columns.

    A is a cooperative matrix with M rows and K columns.

    B is a cooperative matrix with K rows and N columns.

    C is a cooperative matrix with M rows and N columns.

    The values of M, N, and K must be consistent across the result and operands.
    This is referred to as an MxNxK matrix multiply.

    A, B, C, and Result Type must have the same scope, and this defines the
    scope of the operation. A, B, C, and Result Type need not necessarily have
    the same component type, this is defined by the client API.

    If the Component Type of any matrix operand is an integer type, then its
    components are treated as signed if its Component Type has Signedness of 1
    and are treated as unsigned otherwise.

    For a given dynamic instance of this instruction, all invocations in a given
    scope instance must be active or all must be inactive (where the scope is
    the scope of the operation).

    ``` {.ebnf}
    cooperative-matrixmuladd-op ::= ssa-id `=` `spirv.NV.CooperativeMatrixMulAdd`
                              ssa-use `,` ssa-use `,` ssa-use ` : `
                              a-cooperative-matrix-type,
                              b-cooperative-matrix-type ->
                              result-cooperative-matrix-type
    ```
    For example:

    ```
    %0 = spirv.NV.CooperativeMatrixMulAdd %arg0, %arg1, %arg2,  :
      !spirv.NV.coopmatrix<8x16xi32, Subgroup>
    ```
  }];

  let assemblyFormat = [{
    operands attr-dict `:` type($a) `,` type($b) `->` type($c)
  }];

  let availability = [
    MinVersion<SPIRV_V_1_0>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[SPV_NV_cooperative_matrix]>,
    Capability<[SPIRV_C_CooperativeMatrixNV]>
  ];

  let arguments = (ins
    SPIRV_AnyCooperativeMatrixNV:$a,
    SPIRV_AnyCooperativeMatrixNV:$b,
    SPIRV_AnyCooperativeMatrixNV:$c
  );

  let results = (outs
    SPIRV_AnyCooperativeMatrixNV:$result
  );
}

// -----

def SPIRV_NVCooperativeMatrixStoreOp : SPIRV_NvVendorOp<"CooperativeMatrixStore", []> {
  let summary = "See extension SPV_NV_cooperative_matrix";

  let description = [{
    Store a cooperative matrix through a pointer.

    Pointer is a pointer into an array. Its type must be an OpTypePointer whose
    Type operand is a scalar or vector type. The storage class of Pointer must
    be Workgroup, StorageBuffer, or (if SPV_EXT_physical_storage_buffer is
    supported) PhysicalStorageBufferEXT.

    Object is the object to store. Its type must be an
    OpTypeCooperativeMatrixNV.

    Stride is the number of elements in the array in memory between the first
    component of consecutive rows (or columns) in the result. It must be a
    scalar integer type.

    ColumnMajor indicates whether the values stored to memory are arranged in
    column-major or row-major order. It must be a boolean constant instruction,
    with false indicating row major and true indicating column major.

    Memory Access must be a Memory Access literal. If not present, it is the
    same as specifying None.

    ``` {.ebnf}
    coop-matrix-store-op ::= `spirv.NV.CooperativeMatrixStore `
                              ssa-use `, ` ssa-use `, `
                              ssa-use `, ` ssa-use `, `
                              (`[` memory-access `]`)? `:`
                              pointer-type `,` coop-matrix-type
    ```

    For example:

    ```
      spirv.NV.CooperativeMatrixStore %arg0, %arg2, %arg1, %arg3 :
        !spirv.ptr<i32, StorageBuffer>, !spirv.NV.coopmatrix<16x8xi32, Workgroup>
    ```
  }];

  let availability = [
    MinVersion<SPIRV_V_1_0>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[SPV_NV_cooperative_matrix]>,
    Capability<[SPIRV_C_CooperativeMatrixNV]>
  ];

  let arguments = (ins
    SPIRV_AnyPtr:$pointer,
    SPIRV_AnyCooperativeMatrixNV:$object,
    SPIRV_Integer:$stride,
    SPIRV_Bool:$columnmajor,
    OptionalAttr<SPIRV_MemoryAccessAttr>:$memory_access
  );

  let results = (outs);
}

// -----

#endif // MLIR_DIALECT_SPIRV_IR_COOPERATIVE_MATRIX_OPS
