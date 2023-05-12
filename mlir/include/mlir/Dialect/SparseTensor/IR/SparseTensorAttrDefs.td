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
// Type aliases.
//
// These attributes are just like `IndexAttr` (include/mlir/IR/OpBase.td),
// except that:
// (1) the `summary` is more specific (i.e., the fourth parameter to
//     `TypedAttrBase`), which helps tablegen provide better error messages.
// (2) tablegen-generated getters will have the given `returnType`, in
//     lieu of the `APInt` that `IndexAttr` uses.  This avoids the boilerplate
//     of needing to say `get{FOO}().getZExtValue()`, as well as using
//     C++ types which better document intent.
//===----------------------------------------------------------------------===//

def DimensionAttr :
    TypedAttrBase<
      Index, "IntegerAttr",
      And<[CPred<"::llvm::isa<::mlir::IntegerAttr>($_self)">,
           CPred<"::llvm::isa<::mlir::IndexType>("
                     "::llvm::cast<::mlir::IntegerAttr>($_self).getType())">]>,
      "dimension attribute"> {
  let returnType = [{::mlir::sparse_tensor::Dimension}];
  let convertFromStorage = [{$_self.getValue().getZExtValue()}];
}

def LevelAttr :
    TypedAttrBase<
      Index, "IntegerAttr",
      And<[CPred<"::llvm::isa<::mlir::IntegerAttr>($_self)">,
           CPred<"::llvm::isa<::mlir::IndexType>("
                     "::llvm::cast<::mlir::IntegerAttr>($_self).getType())">]>,
      "level attribute"> {
  let returnType = [{::mlir::sparse_tensor::Level}];
  let convertFromStorage = [{$_self.getValue().getZExtValue()}];
}

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

    - Level-type for each level of a tensor type:
        - **dense** : all entries along this level are stored.
        - **compressed** : only nonzeros along this level are stored.
        - **singleton** : a variant of the compressed level-format,
          for when coordinates are guaranteed to have no siblings at this level.
      By default, each level-type has the property of being unique (no
      duplicates at that level) and ordered (coordinates appear sorted
      at that level).  The following two suffixes can be used to specify
      that the level should instead be non-unique (duplicates may appear)
      and/or non-ordered (coordinates may appear unsorted).
        - **-nu** : not unique
        - **-no** : not ordered
      Currently, these suffixes (if present) must appear in this order.
      In the future, we may introduce additional level-types and
      properties, and split up how the level-format and properties are
      specified rather than using this suffix mechanism.

      TODO: This field is called "dimLevelType" for historical reasons,
      even though the types are per-level rather than per-dimension.
      (This will be corrected in an upcoming change that completely
      overhauls the syntax of this attribute.)

    - An optional permutation which maps (higher-ordering)-coordinates
      to level-coordinates; defaulting to the identity permutation.
      For example, given a 2-d tensor with the default higher-ordering,
      `(i, j) -> (i, j)` specifies row-wise storage and `(i, j) ->
      (j, i)` specifies column-wise storage.

      TODO: this field is called "dimOrdering" for historical reasons,
      even though it actually operates on level-coordinates rather than
      dimension-coordinates.
      (This will be corrected in an upcoming change that completely
      overhauls the syntax of this attribute.)

    - An optional higher-order mapping from dimension-coordinates to
      a higher-order coordinate space; defaulting to the identity map.
      This is applied before the `dimOrdering`, thus we have the composite:
      dimCoords --higherOrdering--> hoCoords --dimOrdering--> lvlCoords.
      The higher-order mapping is used to define block-sparse storage,
      jagged-diagonal (JDS/ELL/ITPACK) storage, etc.

      For example, given a 2-d tensor, the mapping
      `(i, j) -> (i floordiv 2, j floordiv 3, i mod 2, j mod 3)`
      imposes an higher-order partitioning into 2x3 blocks along the
      matrix layout.  For block-sparsity, blocks are typically stored
      with compression while dense storage is used within each block
      (although hybrid schemes are possible as well).

      TODO: the following example is out-of-date and will be implemented
      in a different manner than described here.
      (This will be corrected in an upcoming change that completely
      overhauls the syntax of this attribute.)

      The higher-order mapping also provides a notion of "counting a
      dimension", where every stored element with the same coordinate
      is mapped to a new slice.  For instance, ELL storage of a 2-d
      tensor can be defined with the mapping `(i, j) -> (#i, i, j)`
      using the notation of [Chou20].  Lacking the `#` symbol in MLIR's
      affine mapping, we use a free symbol `c` to define such counting,
      together with a constant that denotes the number of resulting
      slices.  For example, the mapping `(i, j)[c] -> (c * 3 * i, i, j)`
      with the level-types `["dense", "dense", "compressed"]` denotes ELL
      storage with three jagged diagonals that count the dimension `i`.

    - The required bitwidth for "position" storage (integral offsets
      into the sparse storage scheme).  A narrow width reduces the memory
      footprint of overhead storage, as long as the width suffices to
      define the total required range (viz. the maximum number of stored
      entries over all indirection levels).  The choices are `8`, `16`,
      `32`, `64`, or, the default, `0` to indicate the native bitwidth.

    - The required bitwidth for "coordinate" storage (the coordinates
      of stored entries).  A narrow width reduces the memory footprint
      of overhead storage, as long as the width suffices to define
      the total required range (viz. the maximum value of each tensor
      coordinate over all levels).  The choices are `8`, `16`, `32`,
      `64`, or, the default, `0` to indicate a native bitwidth.

    - An optional array of `SparseTensorDimSliceAttr`, which specifies
      how the sparse tensor is partitioned on each dimension.

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
      posWidth = 32,
      crdWidth = 8
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
    // A level-type for each level of the sparse storage.
    ArrayRefParameter<
      "::mlir::sparse_tensor::DimLevelType",
      "level-types"
      >: $dimLevelType,
    // A permutation from (higher-ordering)-coordinates to level-coordinates.
    "AffineMap":$dimOrdering,
    // A mapping from dimension-coordinates to (higher-ordering)-coordinates.
    "AffineMap":$higherOrdering,
    // The required bitwidth for position storage.
    "unsigned":$posWidth,
    // The required bitwidth for coordinate storage.
    "unsigned":$crdWidth,
    // A slice attribute for each dimension of the tensor type.
    ArrayRefParameter<
      "::mlir::sparse_tensor::SparseTensorDimSliceAttr",
      "per dimension slice metadata"
      >: $dimSlices
  );

  let builders = [
    AttrBuilder<(ins "ArrayRef<::mlir::sparse_tensor::DimLevelType>":$dimLevelType,
                     "AffineMap":$dimOrdering,
                     "AffineMap":$higherOrdering,
                     "unsigned":$posWidth,
                     "unsigned":$crdWidth), [{
      return $_get($_ctxt, dimLevelType,
                         dimOrdering,
                         higherOrdering,
                         posWidth,
                         crdWidth,
                         ArrayRef<::mlir::sparse_tensor::SparseTensorDimSliceAttr>{});
    }]>
  ];

  let extraClassDeclaration = [{
    /// Returns the type for position storage based on posWidth
    Type getPosType() const;

    /// Returns the type for coordinate storage based on crdWidth
    Type getCrdType() const;

    /// Constructs a new encoding with the dimOrdering and higherOrdering
    /// reset to the default/identity.
    SparseTensorEncodingAttr withoutOrdering() const;

    /// Constructs a new encoding with the pointer and index bitwidth
    /// reset to the default.
    SparseTensorEncodingAttr withoutBitWidths() const;

    /// Returns true if every level is dense.  Also returns true for
    /// the null encoding (since dense-tensors are always all-dense).
    bool isAllDense() const;

    /// Returns true if every level is ordered.  Also returns true for
    /// the null encoding (since dense-tensors are always all-ordered).
    bool isAllOrdered() const;

    /// Returns true if the encoding has an identity dimension ordering.
    /// Also returns true for the null encoding (since dense-tensors
    /// always have the identity ordering).
    bool hasIdDimOrdering() const;

    /// Returns the number of storage levels.  Asserts that the encoding
    /// is non-null (since there is no fixed result that's valid for
    /// every dense-tensor).
    ::mlir::sparse_tensor::Level getLvlRank() const;

    /// Safely looks up the level-type for the requested level.  (Returns
    /// `DimLevelType::Dense` for the null encoding, since dense-tensors
    /// are always all-dense.)
    ::mlir::sparse_tensor::DimLevelType getLvlType(::mlir::sparse_tensor::Level l) const;

    bool isDenseLvl(::mlir::sparse_tensor::Level l) const { return isDenseDLT(getLvlType(l)); }
    bool isCompressedLvl(::mlir::sparse_tensor::Level l) const { return isCompressedDLT(getLvlType(l)); }
    bool isCompressedWithHiLvl(::mlir::sparse_tensor::Level l) const { return isCompressedWithHiDLT(getLvlType(l)); }
    bool isSingletonLvl(::mlir::sparse_tensor::Level l) const { return isSingletonDLT(getLvlType(l)); }
    bool isOrderedLvl(::mlir::sparse_tensor::Level l) const { return isOrderedDLT(getLvlType(l)); }
    bool isUniqueLvl(::mlir::sparse_tensor::Level l) const { return isUniqueDLT(getLvlType(l)); }

    bool isSlice() const {
      return !getDimSlices().empty();
    }

    std::optional<uint64_t> getStaticDimSliceOffset(::mlir::sparse_tensor::Dimension dim) const;
    std::optional<uint64_t> getStaticDimSliceSize(::mlir::sparse_tensor::Dimension dim) const;
    std::optional<uint64_t> getStaticDimSliceStride(::mlir::sparse_tensor::Dimension dim) const;
    std::optional<uint64_t> getStaticLvlSliceOffset(::mlir::sparse_tensor::Level lvl) const;
    std::optional<uint64_t> getStaticLvlSliceSize(::mlir::sparse_tensor::Level lvl) const;
    std::optional<uint64_t> getStaticLvlSliceStride(::mlir::sparse_tensor::Level lvl) const;
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
        I32EnumAttrCase<"LvlSize",    0, "lvl_sz">,
        I32EnumAttrCase<"PosMemSize", 1, "pos_mem_sz">,
        I32EnumAttrCase<"CrdMemSize", 2, "crd_mem_sz">,
        I32EnumAttrCase<"ValMemSize", 3, "val_mem_sz">,
        I32EnumAttrCase<"DimOffset",  4, "dim_offset">,
        I32EnumAttrCase<"DimStride",  5, "dim_stride">,
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

def IsSparseTensorSlicePred
  : CPred<"!!::mlir::sparse_tensor::getSparseTensorEncoding($_self) && "
          "  ::mlir::sparse_tensor::getSparseTensorEncoding($_self).isSlice()">;

// The following four follow the same idiom as `TensorOf`, `AnyTensor`,
// `RankedTensorOf`, `AnyRankedTensor`.

class SparseTensorOf<list<Type> allowedTypes>
  : TensorOf<allowedTypes, [IsSparseTensorPred], "sparse tensor">;

class SparseTensorSliceOf<list<Type> allowedTypes>
  : TensorOf<allowedTypes, [IsSparseTensorSlicePred], "sparse tensor slice">;

def AnySparseTensor : SparseTensorOf<[AnyType]>;
def AnySparseTensorSlice : SparseTensorSliceOf<[AnyType]>;

class RankedSparseTensorOf<list<Type> allowedTypes>
  : RankedTensorOf<allowedTypes, [IsSparseTensorPred], "ranked sparse tensor">;

def AnyRankedSparseTensor : RankedSparseTensorOf<[AnyType]>;

//===----------------------------------------------------------------------===//
// Sparse Tensor Sorting Algorithm Attribute.
//===----------------------------------------------------------------------===//

// TODO: Currently, we only provide four implementations, and expose the
// implementations via attribute algorithm. In the future, if we will need
// to support both stable and non-stable quick sort, we may add
// quick_sort_nonstable enum to the attribute. Alternative, we may use two
// attributes, (stable|nonstable, algorithm), to specify a sorting
// implementation.
//
// --------------------------------------------------------------------------
// |           | hybrid_qsort| insertion_sort | qsort       | heap_sort.    |
// |non-stable | Impl        | X              |  Impl       | Impl          |
// |stable     | X           | Impl           |  Not Impl   | X             |
// --------------------------------------------------------------------------

// The C++ enum for sparse tensor sort kind.
def SparseTensorSortKindEnum
    : I32EnumAttr<"SparseTensorSortKind", "sparse tensor sort algorithm", [
        I32EnumAttrCase<"HybridQuickSort",    0, "hybrid_quick_sort">,
        I32EnumAttrCase<"InsertionSortStable", 1, "insertion_sort_stable">,
        I32EnumAttrCase<"QuickSort", 2, "quick_sort">,
        I32EnumAttrCase<"HeapSort", 3, "heap_sort">,
      ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = SparseTensor_Dialect.cppNamespace;
}

// Define the enum sparse tensor sort kind attribute.
def SparseTensorSortKindAttr
    : EnumAttr<SparseTensor_Dialect, SparseTensorSortKindEnum,
               "SparseTensorSortAlgorithm"> {
}

#endif // SPARSETENSOR_ATTRDEFS
