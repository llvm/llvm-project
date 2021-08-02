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

//===----------------------------------------------------------------------===//
// Base class.
//===----------------------------------------------------------------------===//

class SparseTensor_Op<string mnemonic, list<OpTrait> traits = []>
  : Op<SparseTensor_Dialect, mnemonic, traits> {
  let printer = [{ return ::print(p, *this); }];
  let verifier = [{ return ::verify(*this); }];
  let parser = [{ return ::parse$cppClass(parser, result); }];
}

//===----------------------------------------------------------------------===//
// Operations.
//===----------------------------------------------------------------------===//

def SparseTensor_NewOp : SparseTensor_Op<"new", []>,
    Arguments<(ins AnyType:$source)>,
    Results<(outs TensorOf<[AnyType]>:$result)> {
  string summary = "Constructs a new sparse tensor";
  string description = [{
    Constructs a sparse tensor value with contents taken from an opaque
    pointer provided by `source`. For targets that have access to a file
    system, for example, this pointer may be a filename (or file) of a sparse
    tensor in a particular external storage format. The form of the operation
    is kept deliberately very general to allow for alternative implementations
    in the future, such as pointers to buffers or runnable initialization
    code. The operation is provided as an anchor that materializes a fully
    typed sparse tensor values into a computation.

    Example:

    ```mlir
    sparse_tensor.new %source : !Source to tensor<1024x1024xf64, #CSR>
    ```
  }];
  let assemblyFormat = "$source attr-dict `:` type($source) `to` type($result)";
}

def SparseTensor_ConvertOp : SparseTensor_Op<"convert", [SameOperandsAndResultType]>,
    Arguments<(ins AnyTensor:$source)>,
    Results<(outs AnyTensor:$dest)> {
  string summary = "Converts between different tensor types";
  string description = [{
     Converts one sparse or dense tensor type to another tensor type. The rank
     and dimensions of the source and destination types must match exactly,
     only the sparse encoding of these types may be different. The name `convert`
     was preferred over `cast`, since the operation may incur a non-trivial cost.

     When converting between two different sparse tensor types, only explicitly
     stored values are moved from one underlying sparse storage format to
     the other. When converting from an unannotated dense tensor type to a
     sparse tensor type, an explicit test for nonzero values is used. When
     converting to an unannotated dense tensor type, implicit zeroes in the
     sparse storage format are made explicit. Note that the conversions can have
     non-trivial costs associated with them, since they may involve elaborate
     data structure transformations. Also, conversions from sparse tensor types
     into dense tensor types may be infeasible in terms of storage requirements.

    Examples:

    ```mlir
    %0 = sparse_tensor.convert %1 : tensor<32x32xf32> to tensor<32x32xf32, #CSR>

    %2 = sparse_tensor.convert %3 : tensor<8x8xi32, #CSC> to tensor<8x8xi32, #CSR>
    ```

  }];
  let assemblyFormat = "$source attr-dict `:` type($source) `to` type($dest)";
}

def SparseTensor_ToPointersOp : SparseTensor_Op<"pointers", [NoSideEffect]>,
    Arguments<(ins AnyTensor:$tensor, Index:$dim)>,
    Results<(outs AnyStridedMemRefOfRank<1>:$result)> {
  let summary = "Extract pointers array at given dimension from a tensor";
  let description = [{
     Returns the pointers array of the sparse storage scheme at the
     given dimension for the given sparse tensor. This is similar to the
     `memref.buffer_cast` operation in the sense that it provides a bridge
     between a tensor world view and a bufferized world view. Unlike the
     `memref.buffer_cast` operation, however, this sparse operation actually
     lowers into a call into a support library to obtain access to the
     pointers array.

     Example:

    ```mlir
    %1 = sparse_tensor.pointers %0, %c1
       : tensor<64x64xf64, #CSR> to memref<?xindex>
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
     given dimension for the given sparse tensor. This is similar to the
     `memref.buffer_cast` operation in the sense that it provides a bridge
     between a tensor world view and a bufferized world view. Unlike the
     `memref.buffer_cast` operation, however, this sparse operation actually
     lowers into a call into a support library to obtain access to the
     indices array.

     Example:

    ```mlir
    %1 = sparse_tensor.indices %0, %c1
       : tensor<64x64xf64, #CSR> to memref<?xindex>
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
     sparse tensor, independent of the actual dimension. This is similar to
     the `memref.buffer_cast` operation in the sense that it provides a bridge
     between a tensor world view and a bufferized world view. Unlike the
     `memref.buffer_cast` operation, however, this sparse operation actually
     lowers into a call into a support library to obtain access to the
     values array.

     Example:

    ```mlir
    %1 = sparse_tensor.values %0 : tensor<64x64xf64, #CSR> to memref<?xf64>
    ```
  }];
  let assemblyFormat = "$tensor attr-dict `:` type($tensor) `to` type($result)";
}

def SparseTensor_ToTensorOp : SparseTensor_Op<"tensor", [NoSideEffect]>,
    Arguments<(ins Variadic<AnyStridedMemRefOfRank<1>>:$memrefs)>,
    Results<(outs AnyTensor:$result)> {
  let summary = "Reconstructs tensor from arrays(s)";
  let description = [{
    Reconstructs the sparse tensor from the sparse storage scheme array(s).
    This is similar to the `memref.load` operation in the sense that it
    provides a bridge between a bufferized world view and a tensor world
    view. Unlike the `memref.load` operation, however, this sparse operation
    is used only temporarily to maintain a correctly typed intermediate
    representation during progressive bufferization. Eventually the operation
    is folded away.

    The input arrays are defined unambigously by the sparsity annotations
    (pointers and indices for overhead storage in every compressed dimension,
    followed by one final values array).

    Examples:

    ```mlir
    %1 = sparse_tensor.tensor %0 : memref<?xf64> to tensor<64x64xf64, #Dense>

    %3 = sparse_tensor.tensor %0, %1, %2 :
       memref<?xindex>, memref<?xindex>, memref<?xf32> to tensor<10x10xf32, #CSR>
    ```
  }];
  let assemblyFormat = "$memrefs attr-dict `:` type($memrefs) `to` type($result)";
}

#endif // SPARSETENSOR_OPS
