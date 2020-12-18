//===- TensorOps.td - Tensor op definitions ----------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TENSOR_OPS
#define TENSOR_OPS

include "mlir/Dialect/Tensor/IR/TensorBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class Tensor_Op<string mnemonic, list<OpTrait> traits = []>
    : Op<Tensor_Dialect, mnemonic, traits> {
  let printer = [{ return ::print(p, *this); }];
  let verifier = [{ return ::verify(*this); }];
  let parser = [{ return ::parse$cppClass(parser, result); }];
}

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//

def Tensor_CastOp : Tensor_Op<"cast", [NoSideEffect]> {
  let summary = "tensor cast operation";
  let description = [{
    Convert a tensor from one type to an equivalent type without changing any
    data elements. The source and destination types must both be tensor types
    with the same element type. If both are ranked, then the rank should be the
    same and static dimensions should match. The operation is invalid if
    converting to a mismatching constant dimension.

    Example:

    ```mlir
    // Convert from unknown rank to rank 2 with unknown dimension sizes.
    %2 = tensor.cast %1 : tensor<*xf32> to tensor<?x?xf32>

    // Convert to a type with more known dimensions.
    %3 = tensor.cast %2 : tensor<?x?xf32> to tensor<4x?xf32>

    // Discard static dimension and rank information.
    %4 = tensor.cast %3 : tensor<4x?xf32> to tensor<?x?xf32>
    %5 = tensor.cast %4 : tensor<?x?xf32> to tensor<*xf32>
    ```
  }];

  let arguments = (ins AnyTensor:$source);
  let results = (outs AnyTensor:$dest);
  let assemblyFormat = "$source attr-dict `:` type($source) `to` type($dest)";
  let verifier = "return impl::verifyCastOp(*this, areCastCompatible);";

  let extraClassDeclaration = [{
    /// Return true if `a` and `b` are valid operand and result pairs for
    /// the operation.
    static bool areCastCompatible(Type a, Type b);

    /// The result of a tensor.cast is always a tensor.
    TensorType getType() { return getResult().getType().cast<TensorType>(); }
  }];

  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// ExtractOp
//===----------------------------------------------------------------------===//

def Tensor_ExtractOp : Tensor_Op<"extract",
    [NoSideEffect,
     TypesMatchWith<"result type matches element type of tensor",
                    "tensor", "result",
                    "$_self.cast<ShapedType>().getElementType()">]> {
  let summary = "element extraction operation";
  let description = [{
    The `tensor.extract` op reads a tensor and returns one
    element from it specified by an index list. The output of the op is a
    new value with the same type as the elements of the tensor. The
    arity of indices must match the rank of the accessed value (i.e., if a
    tensor is of rank 3, then 3 indices are required for the extract. The
    indices should all be of `index` type.

    Example:

    ```mlir
    %4 = tensor.extract %t[%1, %2] : tensor<4x4xi32>
    %5 = tensor.extract %rt[%1, %2] : tensor<?x?xi32>
    %6 = tensor.extract %ut[%1, %2] : tensor<*xi32>
    ```
  }];

  let arguments = (ins AnyTensor:$tensor, Variadic<Index>:$indices);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$tensor `[` $indices `]` attr-dict `:` type($tensor)";

  let builders = [
    OpBuilderDAG<(ins "Value":$tensor, CArg<"ValueRange", "{}">:$indices), [{
      auto resType = tensor.getType().cast<ShapedType>().getElementType();
      build($_builder, $_state, resType, tensor, indices);
    }]>];

  let hasFolder = 1;
}

#endif // TENSOR_OPS
