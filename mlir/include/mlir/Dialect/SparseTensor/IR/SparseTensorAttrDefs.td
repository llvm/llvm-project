//===-- SparseTensorAttrDefs.td - attributes definitions ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SPARSETENSOR_ATTRDEFS
#define SPARSETENSOR_ATTRDEFS

include "mlir/Dialect/SparseTensor/IR/SparseTensorBase.td"
include "mlir/IR/TensorEncoding.td"

// All of the Tensor attributes will extend this class.
class SparseTensor_Attr<string name,
                        list<Trait> traits = []>
	: AttrDef<SparseTensor_Dialect, name, traits>;

// Sparse tensor encoding attribute.
def SparseTensorEncodingAttr : SparseTensor_Attr<"SparseTensorEncoding",
         [ DeclareAttrInterfaceMethods<VerifiableTensorEncoding> ] > {
  let mnemonic = "encoding";

  let description = [{
    An attribute to encode TACO-style information on sparsity properties
    of tensors. The encoding is eventually used by a **sparse compiler**
    pass to generate sparse code fully automatically for all tensor
    expressions that involve tensors with a sparse encoding. Compiler
    passes that run before this sparse compiler pass need to be
    aware of the semantics of tensor types with such an encoding.

    Example:

    ```mlir

    #DCSC = #sparse_tensor.encoding<{
      dimLevelType = [ "compressed", "compressed" ],
      dimOrdering = affine_map<(i,j) -> (j,i)>,
      pointerBitWidth = 32,
      indexBitWidth = 8
    }>


    ... tensor<8x8xf64, #DCSR> ...

    ```
  }];

  // Data in sparse tensor encoding.
  let parameters = (
    ins
    // A dimension level type for each dimension of a tensor type.
    // The choices are `dense` (dimension should be stored in its entirety),
    // `compressed` (only non-zero regions or elements should be stored),
    // or `singleton` (no sibling elements for parent).
    ArrayRefParameter<
      "SparseTensorEncodingAttr::DimLevelType",
      "Per-dimension level type"
      >: $dimLevelType,
    // A dimension order on the indices of this tensor type.
    // Unlike dense storage, most sparse storage schemes do not provide
    // fast random access. This affine map specifies the order of
    // dimensions that should be support by the sparse storage scheme
    // (e.g. (i,j) -> (i,j) requests 2-d row-wise and (i,j) -> (j,i)
    // requests 2-d column-wise storage).
    // TODO: block structure with higher-dim inputs
    "AffineMap":$dimOrdering,
    // The required bit width for pointer storage. A narrow width reduces
    // the memory footprint of overhead storage, as long as the width
    // suffices to define the total required range (viz. the maximum
    // number of stored entries over all indirection dimensions). The choices
    // are `8`, `16`, `32`, `64`, or `0` for a native width.
    "unsigned":$pointerBitWidth,
    // The required bit width for index storage. A narrow width reduces
    // the memory footprint of overhead storage, as long as the width
    // suffices to define the total required range (viz. the maximum
    // value of each tensor index over all dimensions). The choices are `8`,
    // `16`, `32`, `64`, or `0` for a native width.
    "unsigned":$indexBitWidth
  );

  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    // Dimension level types that define sparse tensors:
    //   Dense      - dimension is dense, every entry is stored
    //   Compressed - dimension is sparse, only nonzeros are stored
    //   Singleton  - dimension contains single coordinate, no siblings
    enum class DimLevelType {
      Dense, Compressed, Singleton
    };
  }];
}

#endif // SPARSETENSOR_ATTRDEFS
