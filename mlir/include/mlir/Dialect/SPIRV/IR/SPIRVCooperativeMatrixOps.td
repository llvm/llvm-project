//===- SPIRVCooperativeMatrixOps.td - cooperative matmul ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the op definition spec of cooperative matrix multiply extension ops.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SPIRV_IR_COOPERATIVE_MATRIX_OPS
#define MLIR_DIALECT_SPIRV_IR_COOPERATIVE_MATRIX_OPS

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
    %0 = spirv.NV.CooperativeMatrixLength : !spirv.coopmatrix<Subgroup, i32, 8, 16>
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
  let hasVerifier = 0;
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
         : !spirv.ptr<i32, StorageBuffer> as !spirv.coopmatrix<i32, Workgroup, 16, 8>
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
    SPIRV_AnyCooperativeMatrix:$result
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
      !spirv.coopmatrix<Subgroup, i32, 8, 16>
    ```
  }];

  let assemblyFormat = [{
    operands attr-dict`:` type($a) `,` type($b) `->` type($c)
  }];

  let availability = [
    MinVersion<SPIRV_V_1_0>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[SPV_NV_cooperative_matrix]>,
    Capability<[SPIRV_C_CooperativeMatrixNV]>
  ];

  let arguments = (ins
    SPIRV_AnyCooperativeMatrix:$a,
    SPIRV_AnyCooperativeMatrix:$b,
    SPIRV_AnyCooperativeMatrix:$c
  );

  let results = (outs
    SPIRV_AnyCooperativeMatrix:$result
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
                              pointer-type `,` spirv-element-type
    ```

    For example:

    ```
      spirv.NV.CooperativeMatrixStore %arg0, %arg2, %arg1, %arg3 :
        !spirv.ptr<i32, StorageBuffer>, !spirv.coopmatrix<Workgroup, i32, 16, 8>
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
    SPIRV_AnyCooperativeMatrix:$object,
    SPIRV_Integer:$stride,
    SPIRV_Bool:$columnmajor,
    OptionalAttr<SPIRV_MemoryAccessAttr>:$memory_access
  );

  let results = (outs);
}

// -----

#endif // MLIR_DIALECT_SPIRV_IR_COOPERATIVE_MATRIX_OPS
