//===-- SparseTensorAttrDefs.td - attributes definitions ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SPARSETENSOR_ATTRDEFS
#define SPARSETENSOR_ATTRDEFS

include "mlir/IR/AttrTypeBase.td"
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

    The attribute consists of the following fields.
    - Dimension level type for each dimension of a tensor type:
        - **dense** : dimension is dense, all entries along this dimension
          are stored
        - **compressed** : dimension is sparse, only nonzeros along this dimensions
          are stored
        - **singleton** : dimension stores individual indices with no siblings
      By default, each dimension level types has the property of being unique
      (no duplicates at that level) and ordered (indices appear sorted at that
      level). The following two suffixes can be used to make the last two
      dimension level types not-unique (duplicates may appear) and not-ordered
      (indices may appear unsorted).
        - **-nu** : not unique
        - **-no** : not ordered
      Currently, these suffixes, is present, should appear in this order.
      In the future, we may introduce many more dimension level types and
      properties, and separate specifying the two completely rather than
      using this suffix mechanism.
    - An optional dimension ordering on the indices of this tensor type. Unlike
      dense storage, most sparse storage schemes do not provide fast random access.
      This affine map specifies the order of dimensions that should be supported
      by the sparse storage scheme. For example, for a 2-d tensor, "(i,j) -> (i,j)"
      requests row-wise storage and "(i,j) -> (j,i)" requests column-wise storage.
      By default, an identify mapping is used, which implies that the original
      indices directly correspond to stored indices (viz. "(i,j,...) -> (i,j,...)").
    - The required bit width for "pointer" storage (integral offsets into
      the sparse storage scheme). A narrow width reduces the memory footprint
      of overhead storage, as long as the width suffices to define the total
      required range (viz. the maximum number of stored entries over all indirection
      dimensions). The choices are `8`, `16`, `32`, `64`, or, the default, `0` to
      indicate the native bit width.
    - The required bit width for "index" storage (elements of the coordinates of
      stored entries). A narrow width reduces the memory footprint of overhead
      storage, as long as the width suffices to define the total required range
      (viz. the maximum value of each tensor index over all dimensions). The
      choices are `8`, `16`, `32`, `64`, or, the default, `0` to indicate the
      native bit width.

    Examples:

    ```mlir
    // Sparse vector.
    #SparseVector = #sparse_tensor.encoding<{
      dimLevelType = [ "compressed" ]
    }>
    ... tensor<?xf32, #SparseVector> ...

    // Sorted Coordinate Scheme.
    #SortedCOO = #sparse_tensor.encoding<{
      dimLevelType = [ "compressed-nu", "singleton" ]
    }>
    ... tensor<?x?xf64, #SortedCOO> ...

    // Doubly compressed sparse column storage with specific bitwidths.
    #DCSC = #sparse_tensor.encoding<{
      dimLevelType = [ "compressed", "compressed" ],
      dimOrdering = affine_map<(i,j) -> (j,i)>,
      pointerBitWidth = 32,
      indexBitWidth = 8
    }>
    ... tensor<8x8xf64, #DCSC> ...
    ```
  }];

  // Data in sparse tensor encoding.
  let parameters = (
    ins
    // A dimension level type for each dimension of the tensor type.
    ArrayRefParameter<
      "SparseTensorEncodingAttr::DimLevelType",
      "per dimension level type"
      >: $dimLevelType,
    // A dimension order on the indices of this tensor type.
    // TODO: block structure with higher-dim inputs
    "AffineMap":$dimOrdering,
    // The required bit width for pointer storage.
    "unsigned":$pointerBitWidth,
    // The required bit width for index storage.
    "unsigned":$indexBitWidth
  );

  let genVerifyDecl = 1;
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    // Dimension level types. By default, each type has the unique and
    // ordered properties. Alternatives properties are indicated by
    // Nu (not-unique) and No (not-ordered).
    //
    // TODO: separate type and property in encoding
    //
    enum class DimLevelType {
      Dense,
      Compressed, CompressedNu, CompressedNo, CompressedNuNo,
      Singleton, SingletonNu, SingletonNo, SingletonNuNo,
    };
  }];
}

def IsSparseTensorPred
  : CPred<"!!::mlir::sparse_tensor::getSparseTensorEncoding($_self)">;

// The following four follow the same idiom as `TensorOf`, `AnyTensor`,
// `RankedTensorOf`, `AnyRankedTensor`.

class SparseTensorOf<list<Type> allowedTypes>
  : TensorOf<allowedTypes, [IsSparseTensorPred], "sparse tensor">;

def AnySparseTensor : SparseTensorOf<[AnyType]>;

class RankedSparseTensorOf<list<Type> allowedTypes>
  : RankedTensorOf<allowedTypes, [IsSparseTensorPred], "ranked sparse tensor">;

def AnyRankedSparseTensor : RankedSparseTensorOf<[AnyType]>;

#endif // SPARSETENSOR_ATTRDEFS
