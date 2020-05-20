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
    cooperative-matrix-op ::= ssa-id `=` `spv.CooperativeMatrixLoadNV`
                              storage-class ssa-use (`[` memory-access `]`)? `
                              : ` cooperative-matrix-type
    ```

    For example:

    ```
    %0 = spv.CooperativeMatrixLoadNV "StorageBuffer" %ptr, %stride, %colMajor
         : !spv.coopmatrix<i32, Workgroup, 16, 8>
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

  let verifier = [{ return success(); }];
}

// -----

#endif // SPIRV_COOPERATIVE_MATRIX_OPS
