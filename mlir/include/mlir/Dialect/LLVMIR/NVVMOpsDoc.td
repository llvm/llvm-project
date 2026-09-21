//===-- NVVMOpsDoc.td - NVVM op documentation ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Descriptions for NVVM ops. Each op that uses this file defines a record
// named `<DefName>_Doc` deriving from `NVVM_OpDoc`; the `NVVM_Op` class
// in NVVMOps.td uses its `text` field as the op description.
//
//===----------------------------------------------------------------------===//

#ifndef NVVMIR_OPS_DOC
#define NVVMIR_OPS_DOC

class NVVM_OpDoc<string docText> {
  string text = docText;
}

def NVVM_SinOp_Doc : NVVM_OpDoc<[{
    Computes a fast approximation of the sine of the input value (in radians).
    The `ftz` attribute, when set, flushes subnormal inputs and results to
    sign-preserving zero.

    For more information, see PTX ISA:
    [sin](https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-sin)
  }]>;

def NVVM_MmaOp_Doc : NVVM_OpDoc<[{
    The `nvvm.mma.sync` operation collectively performs the operation
    `D = matmul(A, B) + C` using all threads in a warp.

    All the threads in the warp must execute the same `mma.sync` operation.

    For each possible multiplicand PTX data type, there are one or more possible
    instruction shapes given as "mMnNkK". The below table describes the posssibilities
    as well as the types required for the operands. Note that the data type for
    C (the accumulator) and D (the result) can vary independently when there are
    multiple possibilities in the "C/D Type" column.

    When an optional attribute cannot be immediately inferred from the types of
    the operands and the result during parsing or validation, an error will be
    raised.

    `b1Op` is only relevant when the binary (b1) type is given to
    `multiplicandDataType`. It specifies how the multiply-and-acumulate is
    performed and is either `xor_popc` or `and_poc`. The default is `xor_popc`.

    `intOverflowBehavior` is only relevant when the `multiplicandType` attribute
    is one of `u8, s8, u4, s4`, this attribute describes how overflow is handled
    in the accumulator. When the attribute is `satfinite`, the accumulator values
    are clamped in the int32 range on overflow. Alternatively, accumulator
    behavior `wrapped` can be specified (this is the default), in which case
    overflow wraps from one end of the range to the other.

    `layoutA` and `layoutB` are required and should generally be set to
    `#nvvm.mma_layout<row>` and `#nvvm.mma_layout<col>` respectively, but other
    combinations are possible for certain layouts according to the table below.

    ```
    | A/B Type | Shape     | ALayout | BLayout | A Type   | B Type   | C/D Type          |
    |----------|-----------|---------|---------|----------|----------|-------------------|
    | f64      | .m8n8k4   | row     | col     | 1x f64   | 1x f64   | 2x f64            |
    | f16      | .m8n8k4   | row/col | row/col | 2x f16x2 | 2x f16x2 | 4x f16x2 or 8xf32 |
    |          | .m16n8k8  | row     | col     | 2x f16x2 | 1x f16x2 | 2x f16x2 or 4 f32 |
    |          | .m16n8k16 | row     | col     | 4x f16x2 | 2x f16x2 | 2x f16x2 or 4 f32 |
    | bf16     | .m16n8k8  | row     | col     | 2x i32   | 1x i32   | 4x f32            |
    |          | .m16n8k16 | row     | col     | 4x i32   | 2x i32   | 4x f32            |
    | tf32     | .m16n8k4  | row     | col     | 2x i32   | 1x i32   | 4x f32            |
    |          | .m16n8k8  | row     | col     | 4x i32   | 2x i32   | 2x f16x2 or 4 f32 |
    | u8/s8    | .m8n8k16  | row     | col     | 1x i32   | 1x i32   | 2x i32            |
    |          | .m16n8k16 | row     | col     | 2x i32   | 1x i32   | 4x i32            |
    |          | .m16n8k32 | row     | col     | 4x i32   | 2x i32   | 4x i32            |
    | u4/s4    | .m8n8k32  | row     | col     | 1x i32   | 1x i32   | 2x i32            |
    |          | m16n8k32  | row     | col     | 2x i32   | 1x i32   | 4x i32            |
    |          | m16n8k64  | row     | col     | 4x i32   | 2x i32   | 4x i32            |
    | b1       | m8n8k128  | row     | col     | 1x i32   | 1x i32   | 2x i32            |
    |          | m16n8k128 | row     | col     | 2x i32   | 1x i32   | 4x i32            |
    ```


    Example:
    ```mlir

    %128 = nvvm.mma.sync A[%120, %121, %122, %123]
                         B[%124, %125]
                         C[%126, %127]
                         shape = <m = 16, n = 8, k = 16>,
                         layout_a = row, layout_b = col
        : (vector<2xf16>, vector<2xf16>, vector<2xf16>)
           -> !llvm.struct<(vector<2xf16>, vector<2xf16>)>
    ```
  }]>;

#endif // NVVMIR_OPS_DOC
