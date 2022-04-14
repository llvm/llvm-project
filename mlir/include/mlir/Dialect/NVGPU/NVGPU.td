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
}

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

#endif // NVGPU
