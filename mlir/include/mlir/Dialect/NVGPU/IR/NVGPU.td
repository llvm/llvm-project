//===-- NVGPU.td - NVGPU dialect operation definitions *- tablegen -*------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the basic operations for the NVGPU dialect.
//
// This NVGPU provides a bridge between the target agnostic GPU and Vector
// dialects and lower level NVVM dialect. This allow representing PTX specific
// operations while using MLIR high level concepts like memref and 2-D vector.
//
// Ops semantic are going to be based on vendor specific PTX defintion:
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
//
//===----------------------------------------------------------------------===//

#ifndef NVGPU
#define NVGPU

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

def NVGPU_Dialect : Dialect {
  let name = "nvgpu";
  let cppNamespace = "::mlir::nvgpu";
  let description = [{
    The `NVGPU` dialect provides a bridge between higher-level target-agnostic 
    dialects (GPU and Vector) and the lower-level target-specific dialect 
    (LLVM IR based NVVM dialect) for NVIDIA GPUs. This allow representing PTX 
    specific operations while using MLIR high level dialects such as Memref 
    and Vector for memory and target-specific register operands, respectively.
  }];

  let useDefaultTypePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// NVGPU Type Definitions
//===----------------------------------------------------------------------===//

class NVGPU_Type<string name, string typeMnemonic,
        list<Trait> traits = []> : TypeDef<NVGPU_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def NVGPU_DeviceAsyncToken : NVGPU_Type<"DeviceAsyncToken",  
                                        "device.async.token", []> {
  let summary = "device async token type";
  let description = [{
    `nvgpu.device.async.token` is a type returned by an asynchronous operation
    that runs on the GPU (device). It is used to establish an SSA-based link
    between the async operation (e.g. DeviceAsyncCopy) and operations that
    group or synchronize the async operations (e.g. DeviceAsyncCreateGroupOp,
    DeviceAsyncWaitOp).
  }];
}

//===----------------------------------------------------------------------===//
// NVGPU Op Definitions
//===----------------------------------------------------------------------===//

class NVGPU_Op<string mnemonic, list<Trait> traits = []> :
  Op<NVGPU_Dialect, mnemonic, traits> {}

def NVGPU_LdMatrixOp : NVGPU_Op<"ldmatrix", [
                                MemoryEffects<[MemRead]>,
                                PredOpTrait<"srcMemref and res have same element type", 
                                            TCresVTEtIsSameAsOp<0, 0>>]> {
  let description = [{
    The `nvgpu.ldmatrix` op represents loading a matrix fragment from
    memory to registers. The source and result type must be compatible 
    with lowering to the `nvvm.ldmatrix` instruction. This op represents
    the distributed version of a `vector.transfer_read` as an intermediate
    step between lowering from `vector.transfer_read` to `nvvm.ldmatrix`.

    This operation is meant to follow the semantic of described here:
    https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-ldmatrix

    Example:
    ```mlir
    %0 = nvgpu.ldmatrix %sm[%c0, %c0] {numTiles = 4 : i32, transpose = false} :
      memref<?x?xf16, 3> -> vector<4x2xf16>
    ```
  }];

  let arguments = (ins Arg<AnyMemRef, "", [MemRead]>:$srcMemref,
                           Variadic<Index>:$indices, BoolAttr:$transpose,
                           I32Attr:$numTiles);
  let results = (outs AnyVector:$res);
  let assemblyFormat = [{
    $srcMemref`[` $indices `]` attr-dict `:` type($srcMemref) `->` type($res)
  }];

  let hasVerifier = 1;
}

class NVGPU_MmaSyncOp<string mnemonic> :
        NVGPU_Op<mnemonic,  [Pure,
                             PredOpTrait<"matrixA and matrixB have same element type", 
                                         TCopVTEtIsSameAs<0, 1>>]> {
  code extraBaseClassDeclaration = [{
    std::array<int64_t, 3> getMmaShapeAsArray() {      
      ArrayAttr mmaShape = this->getMmaShape();
      assert(mmaShape.size() == 3 && "mmaShape should be three integers");
      return {mmaShape[0].cast<IntegerAttr>().getInt(),
              mmaShape[1].cast<IntegerAttr>().getInt(),
              mmaShape[2].cast<IntegerAttr>().getInt()};
    }
  }];

  let hasVerifier = 1;
}

def NVGPU_MmaSyncOp : NVGPU_MmaSyncOp<"mma.sync"> {
  let description = [{
    The `nvgpu.mma.sync` op represents the warp-level matrix-multiply-and-
    accumulate (mma) operation that is compatible with `nvvm.mma.sync`. 
    The operands and results vector sizes are thread-level onwership to 
    the warp-level mma operation shape. `mmaShape` attribute holds the 
    warp-level matrix-multiply shape.
    
    The `nvgpu.mma.sync` op serves as an intermediate point between lowering from 
    `vector.contract` to `nvvm.mma.sync`.

    This operation is meant to follow the semantic of described here:
      https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma

    Example:

    ```mlir
    %res = nvgpu.mma.sync (%matrixA, %matrixB, %matrixC) {mmaShape = [16, 8, 16]} : 
        (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
    ```
  }];
  let arguments = (ins AnyVector:$matrixA, 
                       AnyVector:$matrixB,
                       AnyVector:$matrixC, 
                       I64ArrayAttr:$mmaShape,
                       OptionalAttr<UnitAttr>:$tf32Enabled
                       );

  let results = (outs AnyVector:$res);

  let builders = [
    OpBuilder<(ins "Value":$matrixA, 
                   "Value":$matrixB, 
                   "Value":$matrixC, 
                   "ArrayAttr":$mmaShape)>
  ];

  let assemblyFormat = [{
    `(` $matrixA`,` $matrixB`,` $matrixC `)` attr-dict
    `:` `(` type($matrixA) `,` type($matrixB) `,` type($matrixC) `)` `->` type($res)
  }];

  let extraClassDeclaration = extraBaseClassDeclaration;
}

def NVGPU_MmaSparseSyncMetadataType : FixedVectorOfLengthAndType<[2], [I16]>,
                        BuildableType<"::mlir::VectorType::get("
                          "{2},$_builder.getI16Type())">;

def NVGPU_MmaSparseSyncOp : NVGPU_MmaSyncOp<"mma.sp.sync"> {
  let description = [{
  The `nvgu.mma.sp.sync` operation performs a warp-distributed MMA operation
  where operand A is "structured sparse". In this case, the `matrixA` operand
  represents the (warp-distributed) non-zero values of operand A, and the
  `sparse_metadata` operand provides the indices.

  The full description of the sparsity storage format and distribution scheme is
  described in the PTX docs. This operation is meant to follow the semantic
  described in the PTX documentation here:
  https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-for-sparse-mma

  The way the indices are distributed among the threads in a warp is controlled
  by the optional `sparsity_selector` operand, which is `0` by default. For
  more information, please consult the PTX documentation linked above.

  Example (targetingthe f16 16x8x32 `mma.sp` PTX instruction):

  ```mlir  
  nvgpu.mma.sp.sync (%a, %b, %c) metadata (%meta) {mmaShape = [16, 8, 32]} :
    (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>    
  ```  
  }];

  let arguments = (ins AnyVector:$matrixA,                       
                       AnyVector:$matrixB,
                       AnyVector:$matrixC,
                       NVGPU_MmaSparseSyncMetadataType:$sparseMetadata,
                       I64ArrayAttr:$mmaShape,
                       DefaultValuedAttr<I32Attr, "0">:$sparsitySelector,
                       OptionalAttr<UnitAttr>:$tf32Enabled
                       );

  let results = (outs AnyVector:$res);

  let builders = [
    OpBuilder<(ins "Value":$matrixA,
                   "Value":$matrixB,
                   "Value":$matrixC,
                   "Value":$sparseMetadata,
                   "ArrayRef<int64_t>":$mmaShape)>
  ];

  let assemblyFormat = [{
    `(` $matrixA`,` $matrixB`,` $matrixC `)` `metadata` `(` $sparseMetadata `)` attr-dict
    `:` `(` type($matrixA) `,` type($matrixB) `,` type($matrixC) `)` `->` type($res)
  }];

  let extraClassDeclaration = extraBaseClassDeclaration;
}

def NVGPU_DeviceAsyncCopyOp : NVGPU_Op<"device_async_copy", [
                                       AttrSizedOperandSegments]> {
  let summary = "device-side asynchronous copy";
  let description = [{
    The `nvgpu.device_async_copy` op initiates an asynchronous copy operation of
    elements from source (global memory) to the destination (shared memory) 
    without blocking the thread. The async copy is added to a group.

    This op is meant to be used with `nvgpu.device_async_create_group` and
    `nvgpu.device_async_wait` to synchronize copies as explained in those ops
    descriptions. 
    
    `bypassL1` attribute is hint to the hardware to bypass the L1 cache during 
    async copy, this hint may be ignored by the hardware. 
    
    `dstElements` attribute is the total number of elements written to 
    destination (shared memory).

    `srcElements` argument is the total number of elements read from 
    source (global memory).
    
    `srcElements` is an optional argument and when present the op only reads 
    `srcElements` number of elements from the source (global memory) and zero fills 
    the rest of the elements in the destination (shared memory).

    In order to do a copy and wait for the result we need the following
    combination:
    ```
    // copy 1.
    %cp1 = nvgpu.device_async_copy %A[%c0], %B[%c0], 4 :memref<16xf32> to memref<16xf32, 3>
    // copy 2.
    %cp2 = nvgpu.device_async_copy %C[%c0], %D[%c0], 4 : memref<16xf32> to memref<16xf32, 3>
    // group 1 contains copy 1 and copy 2.
    %token1 = nvgpu.device_async_create_group %cp1, %cp2
    // copy 3.
    %cp3 = nvgpu.device_async_copy %E[%c0], %F[%c0], 4 : memref<16xf32> to memref<16xf32, 3>
    // group 2 contains copy 3.
    %token2 = nvgpu.device_async_create_group %cp3
    // after the wait copy 1 and copy 2 are complete.
    nvgpu.device_async_wait %token1
    // after the wait copy 3 is complete.
    nvgpu.device_async_wait %token2
    ```

    Example:

    ```mlir
    %0 = nvgpu.device_async_copy %src[%c0, %c0], %dst[%c0, %c0, %c0], 4 :
      memref<4x5xf32> to memref<2x7x5xf32, 3>
    ```
  }];
  let results = (outs NVGPU_DeviceAsyncToken:$asyncToken);
  let arguments = (ins Arg<AnyMemRef, "", [MemWrite]>:$dst,
                       Variadic<Index>:$dstIndices,
                       Arg<AnyMemRef, "", [MemRead]>:$src,
                       Variadic<Index>:$srcIndices,
                       IndexAttr:$dstElements,
                       Optional<Index>:$srcElements,
                       OptionalAttr<UnitAttr>:$bypassL1);
  let assemblyFormat = [{
    $src `[` $srcIndices `]` `,` $dst `[` $dstIndices `]` `,` $dstElements (`,` $srcElements^)?
      attr-dict `:` type($src) `to` type($dst)
  }];
  let hasVerifier = 1;
}

def NVGPU_DeviceAsyncCreateGroupOp : NVGPU_Op<"device_async_create_group", []> {
  let summary = "device side asynchronous create group operation";
  let description = [{
    The `nvgpu.device_async_create_group` op creates a group of memory accesses
    containing all the pending `device_async_copy` operations associated with
    argument tokens. Each token can only be part of one group.

    It returns a token that can be use to wait until the group fully completes.

    This is meant to be used with `nvgpu.device_async_wait` to synchronize copies
    as explained in those ops descriptions.

    Groups are executed in the order they are created.

    Example:

    ```mlir
    %0 = nvgpu.device_async_create_group
  ```
  }];
  let results = (outs NVGPU_DeviceAsyncToken:$asyncToken);
  let arguments = (ins Variadic<NVGPU_DeviceAsyncToken>:$inputTokens);
  let assemblyFormat = [{
    $inputTokens attr-dict
  }];
}

def NVGPU_DeviceAsyncWaitOp : NVGPU_Op<"device_async_wait", []> {
  let summary = "Wait for async gpu ops to complete.";
  let description = [{
    The `nvgpu.device_async_wait` op will block the execution thread until the group
    associated with the source token is fully completed.

    The optional `$numGroup` attribute gives a lower bound of the number of
    groups uncompleted when the wait can unblock the thread.

    Example:

    ```mlir
    nvgpu.device_async_wait %0
    ```
  }];
  let arguments = (ins NVGPU_DeviceAsyncToken:$asyncDependencies,
                       OptionalAttr<I32Attr>:$numGroups);
  let assemblyFormat = [{
    $asyncDependencies attr-dict
  }];
}

#endif // NVGPU
