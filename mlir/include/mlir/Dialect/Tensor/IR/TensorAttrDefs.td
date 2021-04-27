//===-- TensorAttrDefs.td - Tensor Attributes Definitions --*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TENSOR_ATTRDEFS
#define TENSOR_ATTRDEFS

include "mlir/Dialect/Tensor/IR/TensorBase.td"
include "mlir/IR/TensorEncoding.td"

// All of the Tensor attributes will extend this class.
class Tensor_Attr<string name,
                  list<Trait> traits = []> : AttrDef<Tensor_Dialect, name, traits>;

// Sparse tensor encoding attribute.
def SparseTensorEncodingAttr : Tensor_Attr<"SparseTensorEncoding",
         [ DeclareAttrInterfaceMethods<VerifiableTensorEncoding> ] > {
  let mnemonic = "sparse";

  let description = [{
    An attribute to encode "TACO"-style information (see tensor-compiler.org)
    on the sparsity of tensors. The semantics are defined by means of the
    methods getDimLevelType(), getDimOrdering(), getPointerType(), and
    getIndexType(), documented below. The encoding is eventually used by
    a `sparse compiler` pass to generate sparse code fully automatically
    for all tensor expressions that involve tensors with a sparse encoding.
    Compiler passes that run before this sparse compiler pass need to be
    aware of the semantics of tensor types with such an encoding.
  }];

  // All data is stored in a dictionary, interpreted by the methods below.
  let parameters = (
    ins
    "DictionaryAttr":$dict
  );

  let extraClassDeclaration = [{
    // Dimension level types that define sparse tensors:
    //   Dense      - dimension is dense, every entry is stored
    //   Compressed - dimension is sparse, only nonzeros are stored
    //   Singleton  - dimension contains single coordinate, no siblings
    enum class DimLevelType {
      Dense, Compressed, Singleton
    };

    // Returns the dimension level type in the given dimension `dim`
    // of this tensor type. The choices, defined by the `DimLevelType`
    // enum, are `dense` (the dimension should be stored in its entirety),
    // `compressed` (only non-zero regions or elements should be stored),
    // or `singleton` (no sibling elements for parent).
    DimLevelType getDimLevelType(unsigned dim) const;

    // Returns the dimension order of this tensor type as an AffineMap.
    // Unlike dense storage, most sparse storage schemes do not provide
    // fast random access. This affine map specifies the order of
    // dimensions that should be support by the sparse storage scheme
    // (e.g. (i,j) -> (i,j) requests 2-d row-wise and (i,j) -> (j,i)
    // requests 2-d column-wise storage).
    // TODO: block structure with higher-dim inputs
    AffineMap getDimOrdering() const;

    // Returns the required bit width for pointer storage. A narrow width
    // reduces the memory footprint of overhead storage, as long as the
    // width suffices to define the total required range (viz. the maximum
    // number of stored entries over all indirection dimensions). The choices
    // are `8`, `16`, `32`, `64`, or `0` for a native width.
    unsigned getPointerBitWidth() const;

    // Returns the required bit width for index storage. A narrow width
    // reduces the memory footprint of overhead storage, as long as the
    // width suffices to define the total required range (viz. the maximum
    // value of each tensor index over all dimensions). The choices are `8`,
    // `16`, `32`, `64`, or `0` for a native width.
    unsigned getIndexBitWidth() const;
  }];
}

#endif // LLVMIR_ATTRDEFS
