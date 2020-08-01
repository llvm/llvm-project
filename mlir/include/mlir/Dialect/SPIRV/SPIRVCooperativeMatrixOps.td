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

#ifndef SPIRV_COOPERATIVE_MATRIX_OPS
#define SPIRV_COOPERATIVE_MATRIX_OPS

// -----

def SPV_CooperativeMatrixLengthNVOp : SPV_Op<"CooperativeMatrixLengthNV",
  [NoSideEffect]> {
  let summary = "See extension SPV_NV_cooperative_matrix";

  let description = [{
    Number of components of a cooperative matrix type accessible to each
    invocation when treated as a composite.

    Result Type must be an OpTypeInt with 32-bit Width and 0 Signedness.

    Type is a cooperative matrix type.

    ``` {.ebnf}
    cooperative-matrix-length-op ::= ssa-id `=` `spv.CooperativeMatrixLengthNV
                                    ` : ` cooperative-matrix-type
    ```

    For example:

    ```
    %0 = spv.CooperativeMatrixLengthNV : !spv.coopmatrix<Subgroup, i32, 8, 16>
    ```
  }];

  let assemblyFormat = "attr-dict `:` $type";

  let availability = [
    MinVersion<SPV_V_1_0>,
    MaxVersion<SPV_V_1_5>,
    Extension<[SPV_NV_cooperative_matrix]>,
    Capability<[SPV_C_CooperativeMatrixNV]>
  ];

  let arguments = (ins
    TypeAttr:$type
  );

  let results = (outs
    SPV_Int32:$result
  );
  let verifier = [{ return success(); }];
}

// -----

def SPV_CooperativeMatrixLoadNVOp : SPV_Op<"CooperativeMatrixLoadNV", []> {
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
    cooperative-matrixload-op ::= ssa-id `=` `spv.CooperativeMatrixLoadNV`
                              ssa-use `,` ssa-use `,` ssa-use
                              (`[` memory-access `]`)? ` : `
                              pointer-type `as`
                              cooperative-matrix-type
    ```

    For example:

    ```
    %0 = spv.CooperativeMatrixLoadNV %ptr, %stride, %colMajor
         : !spv.ptr<i32, StorageBuffer> as !spv.coopmatrix<i32, Workgroup, 16, 8>
    ```
  }];

  let availability = [
    MinVersion<SPV_V_1_0>,
    MaxVersion<SPV_V_1_5>,
    Extension<[SPV_NV_cooperative_matrix]>,
    Capability<[SPV_C_CooperativeMatrixNV]>
  ];

  let arguments = (ins
    SPV_AnyPtr:$pointer,
    SPV_Integer:$stride,
    SPV_Bool:$columnmajor,
    OptionalAttr<SPV_MemoryAccessAttr>:$memory_access
  );

  let results = (outs
    SPV_AnyCooperativeMatrix:$result
  );

  let verifier = [{
    return verifyPointerAndCoopMatrixType(*this, pointer().getType(),
                                          result().getType());
  }];
}

// -----

def SPV_CooperativeMatrixMulAddNVOp : SPV_Op<"CooperativeMatrixMulAddNV",
  [NoSideEffect, AllTypesMatch<["c", "result"]>]> {
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
    cooperative-matrixmuladd-op ::= ssa-id `=` `spv.CooperativeMatrixMulAddNV`
                              ssa-use `,` ssa-use `,` ssa-use ` : `
                              a-cooperative-matrix-type,
                              b-cooperative-matrix-type ->
                              result-cooperative-matrix-type
    ```
    For example:

    ```
    %0 = spv.CooperativeMatrixMulAddNV %arg0, %arg1, %arg2,  :
      !spv.coopmatrix<Subgroup, i32, 8, 16>
    ```
  }];

  let assemblyFormat = [{
    operands attr-dict`:` type($a) `,` type($b) `->` type($c)
  }];

  let availability = [
    MinVersion<SPV_V_1_0>,
    MaxVersion<SPV_V_1_5>,
    Extension<[SPV_NV_cooperative_matrix]>,
    Capability<[SPV_C_CooperativeMatrixNV]>
  ];

  let arguments = (ins
    SPV_AnyCooperativeMatrix:$a,
    SPV_AnyCooperativeMatrix:$b,
    SPV_AnyCooperativeMatrix:$c
  );

  let results = (outs
    SPV_AnyCooperativeMatrix:$result
  );

  let verifier = [{ return verifyCoopMatrixMulAdd(*this); }];
}

// -----

def SPV_CooperativeMatrixStoreNVOp : SPV_Op<"CooperativeMatrixStoreNV", []> {
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
    coop-matrix-store-op ::= `spv.CooperativeMatrixStoreNV `
                              ssa-use `, ` ssa-use `, `
                              ssa-use `, ` ssa-use `, `
                              (`[` memory-access `]`)? `:`
                              pointer-type `,` spirv-element-type
    ```

    For example:

    ```
      spv.CooperativeMatrixStoreNV %arg0, %arg2, %arg1, %arg3 :
        !spv.ptr<i32, StorageBuffer>, !spv.coopmatrix<Workgroup, i32, 16, 8>
    ```
  }];

  let availability = [
    MinVersion<SPV_V_1_0>,
    MaxVersion<SPV_V_1_5>,
    Extension<[SPV_NV_cooperative_matrix]>,
    Capability<[SPV_C_CooperativeMatrixNV]>
  ];

  let arguments = (ins
    SPV_AnyPtr:$pointer,
    SPV_AnyCooperativeMatrix:$object,
    SPV_Integer:$stride,
    SPV_Bool:$columnmajor,
    OptionalAttr<SPV_MemoryAccessAttr>:$memory_access
  );

  let results = (outs);

  let verifier = [{
    return verifyPointerAndCoopMatrixType(*this, pointer().getType(),
                                          object().getType());
  }];
}

// -----

#endif // SPIRV_COOPERATIVE_MATRIX_OPS
