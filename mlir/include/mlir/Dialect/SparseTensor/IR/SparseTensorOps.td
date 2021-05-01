//===- SparseTensorOps.td - Sparse tensor dialect ops ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SPARSETENSOR_OPS
#define SPARSETENSOR_OPS

include "mlir/Dialect/SparseTensor/IR/SparseTensorAttrDefs.td"
include "mlir/Dialect/SparseTensor/IR/SparseTensorBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Base class.
class SparseTensor_Op<string mnemonic, list<OpTrait> traits = []>
  : Op<SparseTensor_Dialect, mnemonic, traits> {
  let printer = [{ return ::print(p, *this); }];
  let verifier = ?;
  let parser = [{ return ::parse$cppClass(parser, result); }];
}

// TODO: remove me
def SparseTensor_FromPointerOp : SparseTensor_Op<"fromPtr">,
    Arguments<(ins AnyType:$ptr)>,
    Results<(outs AnyTensor:$result)> {
  let summary = "Views an opaque sparse tensor pointer as a tensor";
  let description = [{
     Lacking a first class citizen type for sparse tensors, this operation
     forms the glue between a sparse storage scheme (behind an opaque
     pointer) and the (dense) tensors used in the kernel definitions.
     This operation merely provides a way to assign a proper tensor
     type and shape to the incoming opaque pointer. It disappears
     completely during lowering.

     Example:

    ```mlir
     !SparseTensor = type !llvm.ptr<i8>

     %0 = sparse_tensor.fromPtr %arg0 : !SparseTensor to tensor<64x64xf64>
    ```
  }];
  let assemblyFormat = "$ptr attr-dict `:` type($ptr) `to` type($result)";
}

def SparseTensor_ToPointersOp : SparseTensor_Op<"pointers", [NoSideEffect]>,
    Arguments<(ins AnyTensor:$tensor, Index:$dim)>,
    Results<(outs AnyStridedMemRefOfRank<1>:$result)> {
  let summary = "Extract pointers array at given dimension from a tensor";
  let description = [{
     Returns the pointers array of the sparse storage scheme at the
     given dimension for the given tensor. This is similar to the
     `buffer_cast` operation in the sense that it provides a bridge
     between a tensor world view and a bufferized world view. Unlike the
     `buffer_cast` operation, however, this sparse operation actually
     lowers into a call into a support library to obtain access to the
     pointers array.

     Example:

    ```mlir
    %1 = sparse_tensor.pointers %0, %c1 : tensor<64x64xf64> to memref<?xindex>
    ```
  }];
  let assemblyFormat = "$tensor `,` $dim attr-dict `:` type($tensor)"
      " `to` type($result)";
}

def SparseTensor_ToIndicesOp : SparseTensor_Op<"indices", [NoSideEffect]>,
    Arguments<(ins AnyTensor:$tensor, Index:$dim)>,
    Results<(outs AnyStridedMemRefOfRank<1>:$result)> {
  let summary = "Extract indices array at given dimension from a tensor";
  let description = [{
     Returns the indices array of the sparse storage scheme at the
     given dimension for the given tensor. This is similar to the
     `buffer_cast` operation in the sense that it provides a bridge
     between a tensor world view and a bufferized world view. Unlike the
     `buffer_cast` operation, however, this sparse operation actually
     lowers into a call into a support library to obtain access to the
     indices array.

     Example:

    ```mlir
    %1 = sparse_tensor.indices %0, %c1 : tensor<64x64xf64> to memref<?xindex>
    ```
  }];
  let assemblyFormat = "$tensor `,` $dim attr-dict `:` type($tensor)"
      " `to` type($result)";
}

def SparseTensor_ToValuesOp : SparseTensor_Op<"values", [NoSideEffect]>,
    Arguments<(ins AnyTensor:$tensor)>,
    Results<(outs AnyStridedMemRefOfRank<1>:$result)> {
  let summary = "Extract numerical values array from a tensor";
  let description = [{
     Returns the values array of the sparse storage scheme for the given
     tensor, independent of the actual dimension. This is similar to the
     `buffer_cast` operation in the sense that it provides a bridge
     between a tensor world view and a bufferized world view. Unlike the
     `buffer_cast` operation, however, this sparse operation actually
     lowers into a call into a support library to obtain access to the
     values array.

     Example:

    ```mlir
    %1 = sparse_tensor.values %0 : tensor<64x64xf64> to memref<?xf64>
    ```
  }];
  let assemblyFormat = "$tensor attr-dict `:` type($tensor) `to` type($result)";
}

#endif // SPARSETENSOR_OPS
