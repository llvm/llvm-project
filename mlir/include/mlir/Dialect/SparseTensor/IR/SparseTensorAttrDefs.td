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
include "mlir/IR/EnumAttr.td"
include "mlir/Dialect/SparseTensor/IR/SparseTensorBase.td"
include "mlir/IR/TensorEncoding.td"

// All of the Tensor attributes will extend this class.
class SparseTensor_Attr<string name,
                        list<Trait> traits = []>
    : AttrDef<SparseTensor_Dialect, name, traits>;

//===----------------------------------------------------------------------===//
// Sparse Tensor Dimension Slice Attribute.
//===----------------------------------------------------------------------===//

def SparseTensorDimSliceAttr : SparseTensor_Attr<"SparseTensorDimSlice", []> {
  let mnemonic = "slice";

  let description = [{
    An attribute to encode slice information of a sparse tensor on a particular
    dimension (a tuple of offset, size, stride).
  }];

  let parameters = (
    ins
    "int64_t" : $offset,
    "int64_t" : $size,
    "int64_t" : $stride
  );

  let extraClassDeclaration = [{
    /// Special value for dynamic offset/size/stride.
    static constexpr int64_t kDynamic = -1;

    static bool isDynamic(int64_t v) {
      return v == kDynamic;
    }

    std::optional<uint64_t> getStaticOffset() const {
      if (isDynamic(getOffset()))
        return std::nullopt;
      return static_cast<uint64_t>(getOffset());
    };

    std::optional<uint64_t> getStaticStride() const {
      if (isDynamic(getStride()))
        return std::nullopt;
      return static_cast<uint64_t>(getStride());
    }

    std::optional<uint64_t> getStaticSize() const {
      if (isDynamic(getSize()))
        return std::nullopt;
      return static_cast<uint64_t>(getSize());
    }

    bool isCompletelyDynamic() const {
      return isDynamic(getOffset()) && isDynamic(getStride()) && isDynamic(getSize());
    };
  }];

  let genVerifyDecl = 1;
  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// Sparse Tensor Type Encoding Attribute.
//===----------------------------------------------------------------------===//

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
      dense storage, most sparse storage schemes do not provide fast random
      access.  This affine map specifies the order of dimensions that should be
      supported by the sparse storage scheme. For example, for a 2-d tensor,
      `(i, j) -> (i, j)` requests row-wise storage and `(i, j) -> (j, i)`
      requests column-wise storage.  By default, an identify mapping is used,
      which implies that the original indices directly correspond to stored
      indices.

    - An optional higher-ordering mapping from the original index space of
      the tensor to a higher-order index space, used to define block-sparse
      storage or ELL (jagged diagonal) storage. For example, for a 2-d tensor,
      the mapping `(i, j) -> (i floordiv 2, j floordiv 3, i mod 2, j mod 3)`
      imposes an higher-order partitioning into 2x3 blocks along the matrix
      layout. A dimension ordering can be used to define a desired ordering
      on this higher-order index space. Likewise, the dimension level types
      define dense or compressed storage along this higher-order index space.
      For block-sparse, blocks are typically stored with compression while
      dense storage is used within each block (although hybrid schemes are
      possible as well). The higher-order mapping also provides a notion of
      "counting a dimension", where every stored element with the same index
      is mapped to a new slice. For instance, ELL storage of a 2-d tensor can
      be defined with the mapping `(i, j) -> (#i, i, j)` using the notation
      of [Chou20]. Lacking the `#` symbol in MLIR's affine mapping, we use
      a free symbol `c` to define such counting, together with a constant
      that denotes the number of resulting slices. For example, the mapping
      `(i, j)[c] -> (c * 3 * i, i, j)` with the first two higher-order indices
      stored dense and the innermost compressed denotes ELL storage with
      three jagged diagonals that count the dimension `i`.

      TODO: introduce a real counting symbol to MLIR's mapping, since an
            expression like 3*c*i has no direct interpretation?

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
      choices are `8`, `16`, `32`, `64`, or, the default, `0` to indicate a
      native bit width.

    - An optional array of SparseTensorDimSliceAttr, which specifies how the sparse
      tensor is partitioned on each level.

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
      dimOrdering = affine_map<(i, j) -> (j, i)>,
      pointerBitWidth = 32,
      indexBitWidth = 8
    }>
    ... tensor<8x8xf64, #DCSC> ...

    // Block sparse row storage (2x3 blocks).
    #BCSR = #sparse_tensor.encoding<{
      dimLevelType = [ "compressed", "compressed", "dense", "dense" ],
      dimOrdering  = affine_map<(ii, jj, i, j) -> (ii, jj, i, j)>,
      higherOrdering = affine_map<(i, j) -> (i floordiv 2, j floordiv 3, i mod 2, j mod 3)>
    }>
    ... tensor<20x30xf32, #BCSR> ...

    // ELL storage (4 jagged diagonals, i.e., at most 4 nonzeros per row).
    #ELL = #sparse_tensor.encoding<{
      dimLevelType = [ "dense", "dense", "compressed" ],
      dimOrdering  = affine_map<(ii, i, j) -> (ii, i, j)>,
      higherOrdering = affine_map<(i, j)[c] -> (c * 4 * i, i, j)>
    }>
    ... tensor<?x?xf64, #ELL> ...

    // CSR slice (offset = 0, size = 4, stride = 1 on the first dimension;
    // offset = 0, size = 8, and a dynamic stride on the second dimension).
    #CSR_SLICE = #sparse_tensor.encoding<{
      dimLevelType = [ "dense", "compressed" ],
      slice = [ (0, 4, 1), (0, 8, ?) ]
    }>
    ... tensor<?x?xf64, #CSC_SLICE> ...

    ```
  }];

  // Data in sparse tensor encoding.
  let parameters = (
    ins
    // A dimension level type for each dimension of the tensor type.
    ArrayRefParameter<
      "::mlir::sparse_tensor::DimLevelType",
      "per dimension level type"
      >: $dimLevelType,
    // A dimension order on the indices of this tensor type.
    "AffineMap":$dimOrdering,
    // A mapping between the original and higher-ordering index space.
    "AffineMap":$higherOrdering,
    // The required bit width for pointer storage.
    "unsigned":$pointerBitWidth,
    // The required bit width for index storage.
    "unsigned":$indexBitWidth,
    // A dimension level type for each dimension of the tensor type.
    ArrayRefParameter<
      "::mlir::sparse_tensor::SparseTensorDimSliceAttr",
      "per dimension slice metadata"
      >: $dimSlices
  );

  let builders = [
    AttrBuilder<(ins "ArrayRef<::mlir::sparse_tensor::DimLevelType>":$dimLevelType,
                     "AffineMap":$dimOrdering,
                     "AffineMap":$higherOrdering,
                     "unsigned":$pointerBitWidth,
                     "unsigned":$indexBitWidth), [{
      return $_get($_ctxt, dimLevelType,
                         dimOrdering,
                         higherOrdering,
                         pointerBitWidth,
                         indexBitWidth,
                         ArrayRef<::mlir::sparse_tensor::SparseTensorDimSliceAttr>{});
    }]>
  ];

  let extraClassDeclaration = [{
    /// Returns the type for pointer storage based on pointerBitWidth
    Type getPointerType() const;

    /// Returns the type for index storage based on indexBitWidth
    Type getIndexType() const;

    /// Constructs a new encoding with the dimOrdering and higherOrdering
    /// reset to the default/identity.
    SparseTensorEncodingAttr withoutOrdering() const;

    /// Return true if every level is dense in the encoding.
    bool isAllDense() const;

    /// Return true if the encoding has an identity dimension ordering.
    bool hasIdDimOrdering() const;

    bool isSlice() const {
      return !getDimSlices().empty();
    }

    std::optional<uint64_t> getStaticDimSliceOffset(unsigned dim) const;
    std::optional<uint64_t> getStaticDimSliceSize(unsigned dim) const;
    std::optional<uint64_t> getStaticDimSliceStride(unsigned dim) const;
    std::optional<uint64_t> getStaticLvlSliceOffset(unsigned lvl) const;
    std::optional<uint64_t> getStaticLvlSliceSize(unsigned lvl) const;
    std::optional<uint64_t> getStaticLvlSliceStride(unsigned lvl) const;
  }];

  let genVerifyDecl = 1;
  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// Sparse Tensor Storage Specifier Enum Attribute.
//===----------------------------------------------------------------------===//

// The C++ enum for Storage Specifier kind.
def SparseTensorStorageSpecifierKindEnum
    : I32EnumAttr<"StorageSpecifierKind", "sparse tensor storage specifier kind", [
        I32EnumAttrCase<"DimSize",    0, "dim_sz">,
        I32EnumAttrCase<"PtrMemSize", 1, "ptr_mem_sz">,
        I32EnumAttrCase<"IdxMemSize", 2, "idx_mem_sz">,
        I32EnumAttrCase<"ValMemSize", 3, "val_mem_sz">,
      ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = SparseTensor_Dialect.cppNamespace;
}

// Define the enum StorageSpecifier kind attribute.
def SparseTensorStorageSpecifierKindAttr
    : EnumAttr<SparseTensor_Dialect, SparseTensorStorageSpecifierKindEnum,
               "SparseTensorStorageSpecifierKind"> {
   let mnemonic = "kind";
}

//===----------------------------------------------------------------------===//
// Sparse Tensor Traits.
//===----------------------------------------------------------------------===//

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
