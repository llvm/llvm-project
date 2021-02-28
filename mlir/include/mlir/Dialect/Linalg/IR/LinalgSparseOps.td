//===- LinalgSparseOps.td - Linalg dialect sparse ops ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The following operations bootstrap working with sparse tensors solely
// within the Linalg dialect. They provide temporary bridges between a
// future SparseTensorType (now an opaque pointer), the actual TensorType,
// and MemRef arrays underlying an actual sparse storage scheme in memory.
//
// Lacking a proper sparse tensor type, the 'sparse_tensor' operation
// provides a bridge between an opaque pointer and a regular tensor type
// just to simplify feeding the value into a Linalg op. The operation
// simply disappears during lowering.
//
// The other operations form the bridge between the opaque pointer and
// the actual storage of pointers, indices, and values. These operations
// resemble 'tensor_to_memref' in the sense that they map tensors to
// their bufferized memrefs, but they lower into actual calls since
// sparse storage does not bufferize into a single memrefs, as dense
// tensors do, but into a hierarchical storage scheme where pointers
// access memrefs with indices and eventually into values.
//
// TODO: introduce SparseTensorType as first class citizen in MLIR
//
//===----------------------------------------------------------------------===//

#ifndef LINALG_SPARSE_OPS
#define LINALG_SPARSE_OPS

include "mlir/Dialect/Linalg/IR/LinalgBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Base class.
class Linalg_SparseOp<string mnemonic, list<OpTrait> traits = []>
  : Op<Linalg_Dialect, mnemonic, traits> {
  let printer = [{ return ::print(p, *this); }];
  let verifier = ?;
  let parser = [{ return ::parse$cppClass(parser, result); }];
}

def Linalg_SparseTensorFromPointerOp :
    Linalg_SparseOp<"sparse_tensor">,
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

     %0 = linalg.sparse_tensor %arg0 : !SparseTensor to tensor<64x64xf64>
    ```
  }];
  let assemblyFormat = "$ptr attr-dict `:` type($ptr) `to` type($result)";
}

def Linalg_SparseTensorToPointersMemRefOp :
    Linalg_SparseOp<"sparse_pointers", [NoSideEffect]>,
    Arguments<(ins AnyTensor:$tensor, Index:$dim)>,
    Results<(outs AnyStridedMemRefOfRank<1>:$result)> {
  let summary = "Extract pointers array at given dimension from a tensor";
  let description = [{
     Returns the pointers array of the sparse storage scheme at the
     given dimension for the given tensor. This is similar to the
     `tensor_to_memref` operation in the sense that it provides a bridge
     between a tensor world view and a bufferized world view. Unlike the
     `tensor_to_memref` operation, however, this sparse operation actually
     lowers into a call into a support library to obtain access to the
     pointers array.

     Example:

    ```mlir
    %1 = linalg.sparse_pointers %0, %c1 : tensor<64x64xf64> to memref<?xindex>
    ```
  }];
  let assemblyFormat = "$tensor `,` $dim attr-dict `:` type($tensor)"
      " `to` type($result)";
}

def Linalg_SparseTensorToIndicesMemRefOp :
    Linalg_SparseOp<"sparse_indices", [NoSideEffect]>,
    Arguments<(ins AnyTensor:$tensor, Index:$dim)>,
    Results<(outs AnyStridedMemRefOfRank<1>:$result)> {
  let summary = "Extract indices array at given dimension from a tensor";
  let description = [{
     Returns the indices array of the sparse storage scheme at the
     given dimension for the given tensor. This is similar to the
     `tensor_to_memref` operation in the sense that it provides a bridge
     between a tensor world view and a bufferized world view. Unlike the
     `tensor_to_memref` operation, however, this sparse operation actually
     lowers into a call into a support library to obtain access to the
     indices array.

     Example:

    ```mlir
    %1 = linalg.sparse_indices %0, %c1 : tensor<64x64xf64> to memref<?xindex>
    ```
  }];
  let assemblyFormat = "$tensor `,` $dim attr-dict `:` type($tensor)"
      " `to` type($result)";
}

def Linalg_SparseTensorToValuesMemRefOp :
    Linalg_SparseOp<"sparse_values", [NoSideEffect]>,
    Arguments<(ins AnyTensor:$tensor)>,
    Results<(outs AnyStridedMemRefOfRank<1>:$result)> {
  let summary = "Extract numerical values array from a tensor";
  let description = [{
     Returns the values array of the sparse storage scheme for the given
     tensor, independent of the actual dimension. This is similar to the
     `tensor_to_memref` operation in the sense that it provides a bridge
     between a tensor world view and a bufferized world view. Unlike the
     `tensor_to_memref` operation, however, this sparse operation actually
     lowers into a call into a support library to obtain access to the
     values array.

     Example:

    ```mlir
    %1 = linalg.sparse_values %0 : tensor<64x64xf64> to memref<?xf64>
    ```
  }];
  let assemblyFormat = "$tensor attr-dict `:` type($tensor) `to` type($result)";
}

#endif // LINALG_SPARSE_OPS
