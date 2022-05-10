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
include "mlir/IR/OpBase.td"

def NVGPU_Dialect : Dialect {
  let name = "nvgpu";
  let cppNamespace = "::mlir::nvgpu";
  let description = [{
    This `NVGPU` dialect provides a bridge between the target agnostic GPU and
    Vector dialects and the lower level LLVM IR based NVVM dialect. This allow
    representing PTX specific operations while using MLIR high level concepts
    like memref and 2-D vector.
  }];
  let useDefaultAttributePrinterParser = 1;
}

/// Device-side synchronization token.
def NVGPU_DeviceAsyncToken : DialectType<
  NVGPU_Dialect, CPred<"$_self.isa<::mlir::nvgpu::DeviceAsyncTokenType>()">,
   "device async token type">,
   BuildableType<
     "mlir::nvgpu::DeviceAsyncTokenType::get($_builder.getContext())">;


//===----------------------------------------------------------------------===//
// NVGPU Op definitions
//===----------------------------------------------------------------------===//

class NVGPU_Op<string mnemonic, list<Trait> traits = []> :
  Op<NVGPU_Dialect, mnemonic, traits> {}

def NVGPU_LdMatrixOp : NVGPU_Op<"ldmatrix",
                                [MemoryEffects<[MemRead]>]> {
  let description = [{
  The `nvgpu.ldmatrix` op represents loading a matrix fragment from
  memory. The load source and result type must be compatible with lowering
  to the `nvvm.ldmatrix` instruction. This op is meant to represent
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
}

def NVGPU_MmaSyncOp : NVGPU_Op<"mma.sync", [NoSideEffect]> {
  let description = [{
  The `nvgpu.mma.sync` op represents the distributed form of a collective
  matrix-multiply-and-accumulate (mma) operation that is compatible with
  `nvvm.mma.sync`. The operands and results are fragments of the full matrix
  operands. The full shape of the distributed mma operation is given by the
  `mmaShape` attribute in the form of a list of dimensions `[m, n, k]`.

  This operation is meant to be lowered to the `nvvm.mma.sync` instruction, and
  is an intermediate point between lowering from `vector.contract` to
  `nvvm.mma.sync`.

  This operation is meant to follow the semantic of described here:
    https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma

  Example:

  ```mlir
  nvgpu.mma.sync (%a, %b, %c) :
    (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
  ```
  }];
  let arguments = (ins AnyVector:$matrixA, AnyVector:$matrixB,
                       AnyVector:$matrixC, I64ArrayAttr:$mmaShape);

  let results = (outs AnyVector:$res);

  let assemblyFormat = [{
    `(` $matrixA`,` $matrixB`,` $matrixC `)` attr-dict
    `:` `(` type($matrixA) `,` type($matrixB) `,` type($matrixC) `)` `->` type($res)
  }];
}


def NVGPU_DeviceAsyncCopyOp : NVGPU_Op<"device_async_copy",
  [AttrSizedOperandSegments]> {
  let summary = "device-side asynchronous copy";
  let description = [{
    The `gpu.device_async_copy` op initiates an asynchronous copy operation of
    `$size` elements from source to the destination without blocking the thread.
    The destination has to be in shared memory.

    This is memory access will be pending to be added to a group.

    This op is meant to be used with `gpu.device_async_create_group` and
    `gpu.device_async_wait` to synchronize copies as explained in those ops
    descriptions. 
    `bypassL1` attribute is hint to the backend and hardware that
    the copy should by pass the L1 cache, this may be dropped by the backend or
    hardware. 

    In order to do a copy and wait for the result we need the following
    combination:
    ```
    // copy 1.
    %cp1 = gpu.device_async_copy %A[%c0], %B[%c0], 4 :memref<16xf32> to memref<16xf32, 3>
    // copy 2.
    %cp2 = gpu.device_async_copy %C[%c0], %D[%c0], 4 : memref<16xf32> to memref<16xf32, 3>
    // group 1 contains copy 1 and copy 2.
    %token1 = gpu.device_async_create_group %cp1, %cp2
    // copy 3.
    %cp3 = gpu.device_async_copy %E[%c0], %F[%c0], 4 : memref<16xf32> to memref<16xf32, 3>
    // group 2 contains copy 3.
    %token2 = gpu.device_async_create_group %cp3
    // after the wait copy 1 and copy 2 are complete.
    gpu.device_async_wait %token1
    // after the wait copy 3 is complete.
    gpu.device_async_wait %token2
    ```

    Example:

    ```mlir
    %0 = gpu.device_async_copy %src[%c0, %c0], %dst[%c0, %c0, %c0], 4 :
      memref<4x5xf32> to memref<2x7x5xf32, 3>
    ```
  }];
  let results = (outs NVGPU_DeviceAsyncToken:$asyncToken);
  let arguments = (ins Arg<AnyMemRef, "", [MemWrite]>:$dst,
                       Variadic<Index>:$dstIndices,
                       Arg<AnyMemRef, "", [MemRead]>:$src,
                       Variadic<Index>:$srcIndices,
                       IndexAttr:$numElements,
                       OptionalAttr<UnitAttr>:$bypassL1);
  let assemblyFormat = [{
    $src `[` $srcIndices `]` `,` $dst `[` $dstIndices `]` `,` $numElements
      attr-dict `:` type($src) `to` type($dst)
  }];
  let hasVerifier = 1;
}

def NVGPU_DeviceAsyncCreateGroupOp : NVGPU_Op<"device_async_create_group", []> {
  let summary = "device side asynchronous create group operation";
  let description = [{
  The `gpu.device_async_create_group` op creates a group of memory accesses
  containing all the pending `device_async_copy` operations associated with
  argument tokens. Each token can only be part of one group.

  It returns a token that can be use to wait until the group fully completes.

  This is meant to be used with `gpu.device_async_wait` to synchronize copies
  as explained in those ops descriptions.

  Groups are executed in the order they are created.

  Example:

  ```mlir
  %0 = gpu.device_async_create_group
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
  The `gpu.device_async_wait` op will block the execution thread until the group
  associated with the source token is fully completed.

    The optional `$numGroup` attribute gives a lower bound of the number of
    groups uncompleted when the wait can unblock the thread.
  Example:

  ```mlir
  gpu.device_async_wait %0
  ```
  }];
  let arguments = (ins NVGPU_DeviceAsyncToken:$asyncDependencies,
                       OptionalAttr<I32Attr>:$numGroups);
  let assemblyFormat = [{
    $asyncDependencies attr-dict
  }];
}

#endif // NVGPU
