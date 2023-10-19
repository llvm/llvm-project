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
include "mlir/Dialect/SparseTensor/IR/SparseTensorTypes.td"
include "mlir/Dialect/SparseTensor/IR/SparseTensorInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Base class.
//===----------------------------------------------------------------------===//

class SparseTensor_Op<string mnemonic, list<Trait> traits = []>
  : Op<SparseTensor_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Sparse Tensor Operations.
//===----------------------------------------------------------------------===//

def SparseTensor_NewOp : SparseTensor_Op<"new", [Pure]>,
    Arguments<(ins AnyType:$source)>,
    Results<(outs AnySparseTensor:$result)> {
  string summary = "Materializes a new sparse tensor from given source";
  string description = [{
    Materializes a sparse tensor with contents taken from an opaque pointer
    provided by `source`. For targets that have access to a file system,
    for example, this pointer may be a filename (or file) of a sparse
    tensor in a particular external storage format. The form of the operation
    is kept deliberately very general to allow for alternative implementations
    in the future, such as pointers to buffers or runnable initialization
    code. The operation is provided as an anchor that materializes a properly
    typed sparse tensor with inital contents into a computation.

    Reading in a symmetric matrix will result in just the lower/upper triangular
    part of the matrix (so that only relevant information is stored). Proper
    symmetry support for operating on symmetric matrices is still TBD.

    Example:

    ```mlir
    sparse_tensor.new %source : !Source to tensor<1024x1024xf64, #CSR>
    ```
  }];
  let assemblyFormat = "$source attr-dict `:` type($source) `to` type($result)";
}

def SparseTensor_AssembleOp : SparseTensor_Op<"assemble", [Pure]>,
    Arguments<(ins TensorOf<[AnyType]>:$values,
                   Variadic<TensorOf<[AnySignlessIntegerOrIndex]>>:$levels)>,
    Results<(outs AnySparseTensor: $result)> {
  let summary = "Returns a sparse tensor from the given values, levels";

  let description = [{
    Assembles the values and per-level coordinate or postion arrays into a sparse tensor.
    The order and types of provided levels must be consistent with the actual storage
    layout of the returned sparse tensor described below.

    - `values : tensor<? x V>`
      supplies the value for each stored element in the sparse tensor.
    - `levels: [tensor<? x iType>, ...]`
      each supplies the sparse tensor coordinates scheme in the sparse tensor for
      the corresponding level as specifed by `sparse_tensor::StorageLayout`.

    This operation can be used to materialize a sparse tensor from external
    sources; e.g., when passing two numpy arrays from Python.

    Disclaimer: This is the user's responsibility to provide input that can be
    correctly interpreted by the sparsifier, which does not perform
    any sanity test during runtime to verify data integrity.

    TODO: The returned tensor is allowed (in principle) to have non-identity
    dimOrdering/higherOrdering mappings.  However, the current implementation
    does not yet support them.


    Example:

    ```mlir
    %values      = arith.constant dense<[ 1.1,   2.2,   3.3 ]> : tensor<3xf64>
    %coordinates = arith.constant dense<[[0,0], [1,2], [1,3]]> : tensor<3x2xindex>
    %st = sparse_tensor.assemble %values, %coordinates
        : tensor<3xf64>, tensor<3x2xindex> to tensor<3x4xf64, #COO>
    // yields COO format |1.1, 0.0, 0.0, 0.0|
    //     of 3x4 matrix |0.0, 0.0, 2.2, 3.3|
    //                   |0.0, 0.0, 0.0, 0.0|
    ```
  }];

  let assemblyFormat =
    "$values `,` $levels attr-dict"
    "`:` type($values) `,` type($levels) `to` type($result)";

  let hasVerifier = 1;
}

def SparseTensor_DisassembleOp : SparseTensor_Op<"disassemble", [Pure, SameVariadicResultSize]>,
    Arguments<(ins AnySparseTensor:$tensor,
                   TensorOf<[AnyType]>:$out_values,
                   Variadic<TensorOf<[AnySignlessIntegerOrIndex]>>:$out_levels)>,
    Results<(outs TensorOf<[AnyType]>:$ret_values,
                  Variadic<TensorOf<[AnySignlessIntegerOrIndex]>>:$ret_levels,
                  ScalarLikeOf<[AnySignlessIntegerOrIndex]>:$val_len,
                  Variadic<ScalarLikeOf<[AnySignlessIntegerOrIndex]>>:$lvl_lens)> {
  let summary = "Returns the (values, coordinates) pair unpacked from the input tensor";

  let description = [{
    The disassemble operation is the inverse of `sparse_tensor::assemble`.  It returns
    the values and per-level position and coordinate array to the user
    from the sparse tensor along with the actual length of the memory used in
    each returned buffer. This operation can be used for returning an
    unpacked MLIR sparse tensor to frontend; e.g., returning two numpy arrays to Python.

    Disclaimer: This is the user's responsibility to allocate large enough buffers
    to hold the sparse tensor. The sparsifier simply copies each fields
    of the sparse tensor into the user-supplied buffer without bound checking.

    TODO: the current implementation does not yet support non-identity mappings.

    Example:

    ```mlir
    // input COO format |1.1, 0.0, 0.0, 0.0|
    //    of 3x4 matrix |0.0, 0.0, 2.2, 3.3|
    //                  |0.0, 0.0, 0.0, 0.0|
    %v, %p, %c, %v_len, %p_len, %c_len =
        sparse_tensor.disassemble %sp : tensor<3x4xf64, #COO>
          outs(%od, %op, %oi : tensor<3xf64>, tensor<2xindex>, tensor<3x2xindex>)
                            -> tensor<3xf64>, (tensor<2xindex>, tensor<3x2xindex>), index, (index, index)
    // %v = arith.constant dense<[ 1.1,   2.2,   3.3 ]> : tensor<3xf64>
    // %p = arith.constant dense<[ 0,              3 ]> : tensor<2xindex>
    // %c = arith.constant dense<[[0,0], [1,2], [1,3]]> : tensor<3x2xindex>
    // %v_len = 3
    // %p_len = 2
    // %c_len = 6 (3x2)
    ```
  }];

  let assemblyFormat =
    "$tensor `:` type($tensor) "
    "`outs` `(` $out_values `,` $out_levels `:` type($out_values) `,` type($out_levels) `)` attr-dict"
    "`->` type($ret_values) `,` `(` type($ret_levels) `)` `,` type($val_len) `,` `(` type($lvl_lens) `)`";

  let hasVerifier = 1;
}

def SparseTensor_ConvertOp : SparseTensor_Op<"convert",
  [Pure, StageWithSortSparseOpInterface]>,
    Arguments<(ins AnyTensor:$source)>,
    Results<(outs AnyTensor:$dest)> {
  string summary = "Converts between different tensor types";
  string description = [{
    Converts one sparse or dense tensor type to another tensor type. The rank
    of the source and destination types must match exactly, and the dimension
    sizes must either match exactly or relax from a static to a dynamic size.
    The sparse encoding of the two types can obviously be completely different.
    The name `convert` was preferred over `cast`, since the operation may incur
    a non-trivial cost.

    When converting between two different sparse tensor types, only explicitly
    stored values are moved from one underlying sparse storage format to
    the other. When converting from an unannotated dense tensor type to a
    sparse tensor type, an explicit test for nonzero values is used. When
    converting to an unannotated dense tensor type, implicit zeroes in the
    sparse storage format are made explicit. Note that the conversions can have
    non-trivial costs associated with them, since they may involve elaborate
    data structure transformations. Also, conversions from sparse tensor types
    into dense tensor types may be infeasible in terms of storage requirements.

    Trivial dense-to-dense convert will be removed by canonicalization while
    trivial sparse-to-sparse convert will be removed by the sparse codegen. This
    is because we use trivial sparse-to-sparse convert to tell bufferization
    that the sparse codegen will expand the tensor buffer into sparse tensor
    storage.

    Examples:

    ```mlir
    %0 = sparse_tensor.convert %a : tensor<32x32xf32> to tensor<32x32xf32, #CSR>
    %1 = sparse_tensor.convert %a : tensor<32x32xf32> to tensor<?x?xf32, #CSR>
    %2 = sparse_tensor.convert %b : tensor<8x8xi32, #CSC> to tensor<8x8xi32, #CSR>
    %3 = sparse_tensor.convert %c : tensor<4x8xf64, #CSR> to tensor<4x?xf64, #CSC>

    // The following conversion is not allowed (since it would require a
    // runtime assertion that the source's dimension size is actually 100).
    %4 = sparse_tensor.convert %d : tensor<?xf64> to tensor<100xf64, #SV>
    ```

  }];

  let extraClassDeclaration = [{
     // Whether the convert can be done by a single step or it would require
     // an extra sort. Inherited from StageWithSortSparseOpInterface.
     bool needsExtraSort();
  }];

  let assemblyFormat = "$source attr-dict `:` type($source) `to` type($dest)";
  let hasFolder = 1;
  let hasVerifier = 1;
}

def SparseTensor_ToPositionsOp : SparseTensor_Op<"positions", [Pure]>,
    Arguments<(ins AnySparseTensor:$tensor, LevelAttr:$level)>,
    Results<(outs AnyStridedMemRefOfRank<1>:$result)> {
  let summary = "Extracts the `level`-th positions array of the `tensor`";
  let description = [{
    Returns the positions array of the tensor's storage at the given
    level.  This is similar to the `bufferization.to_memref` operation
    in the sense that it provides a bridge between a tensor world view
    and a bufferized world view.  Unlike the `bufferization.to_memref`
    operation, however, this sparse operation actually lowers into code
    that extracts the positions array from the sparse storage itself
    (either by calling a support library or through direct code).

    Writing into the result of this operation is undefined behavior.

    Example:

    ```mlir
    %1 = sparse_tensor.positions %0 { level = 1 : index }
       : tensor<64x64xf64, #CSR> to memref<?xindex>
    ```
  }];
  let assemblyFormat = "$tensor attr-dict `:` type($tensor) `to` type($result)";
  let hasVerifier = 1;
}

def SparseTensor_ToCoordinatesOp : SparseTensor_Op<"coordinates", [Pure]>,
    Arguments<(ins AnySparseTensor:$tensor, LevelAttr:$level)>,
    Results<(outs AnyStridedMemRefOfRank<1>:$result)> {
  let summary = "Extracts the `level`-th coordinates array of the `tensor`";
  let description = [{
    Returns the coordinates array of the tensor's storage at the given
    level.  This is similar to the `bufferization.to_memref` operation
    in the sense that it provides a bridge between a tensor world view
    and a bufferized world view.  Unlike the `bufferization.to_memref`
    operation, however, this sparse operation actually lowers into code
    that extracts the coordinates array from the sparse storage itself
    (either by calling a support library or through direct code).

    Writing into the result of this operation is undefined behavior.

    Example:

    ```mlir
    %1 = sparse_tensor.coordinates %0 { level = 1 : index }
       : tensor<64x64xf64, #CSR> to memref<?xindex>
    ```
  }];
  let assemblyFormat = "$tensor attr-dict `:` type($tensor) `to` type($result)";
  let hasVerifier = 1;
}

def SparseTensor_ToCoordinatesBufferOp : SparseTensor_Op<"coordinates_buffer", [Pure]>,
    Arguments<(ins AnySparseTensor:$tensor)>,
    Results<(outs AnyStridedMemRefOfRank<1>:$result)> {
  let summary = "Extracts the linear coordinates array from a tensor";
  let description = [{
    Returns the linear coordinates array for a sparse tensor with
    a trailing COO region with at least two levels.  It is an error
    if the tensor doesn't contain such a COO region.  This is similar
    to the `bufferization.to_memref` operation in the sense that it
    provides a bridge between a tensor world view and a bufferized
    world view.  Unlike the `bufferization.to_memref` operation,
    however, this operation actually lowers into code that extracts
    the linear coordinates array from the sparse storage scheme that
    stores the coordinates for the COO region as an array of structures.
    For example, a 2D COO sparse tensor with two non-zero elements at
    coordinates (1, 3) and (4, 6) are stored in a linear buffer as
    (1, 4, 3, 6) instead of two buffer as (1, 4) and (3, 6).

    Writing into the result of this operation is undefined behavior.

    Example:

    ```mlir
    %1 = sparse_tensor.coordinates_buffer %0
       : tensor<64x64xf64, #COO> to memref<?xindex>
    ```
  }];
  let assemblyFormat = "$tensor attr-dict `:` type($tensor) `to` type($result)";
  let hasVerifier = 1;
}

def SparseTensor_ToValuesOp : SparseTensor_Op<"values", [Pure]>,
    Arguments<(ins AnySparseTensor:$tensor)>,
    Results<(outs AnyStridedMemRefOfRank<1>:$result)> {
  let summary = "Extracts numerical values array from a tensor";
  let description = [{
    Returns the values array of the sparse storage format for the given
    sparse tensor, independent of the actual dimension. This is similar to
    the `bufferization.to_memref` operation in the sense that it provides a bridge
    between a tensor world view and a bufferized world view. Unlike the
    `bufferization.to_memref` operation, however, this sparse operation actually
    lowers into code that extracts the values array from the sparse storage
    scheme (either by calling a support library or through direct code).

    Writing into the result of this operation is undefined behavior.

    Example:

    ```mlir
    %1 = sparse_tensor.values %0 : tensor<64x64xf64, #CSR> to memref<?xf64>
    ```
  }];
  let assemblyFormat = "$tensor attr-dict `:` type($tensor) `to` type($result)";
  let hasVerifier = 1;
}

def SparseTensor_NumberOfEntriesOp : SparseTensor_Op<"number_of_entries", [Pure]>,
    Arguments<(ins AnySparseTensor:$tensor)>,
    Results<(outs Index:$result)> {
  let summary = "Returns the number of entries that are stored in the tensor.";
  let description = [{
    Returns the number of entries that are stored in the given sparse tensor.
    Note that this is typically the number of nonzero elements in the tensor,
    but since explicit zeros may appear in the storage formats, the more
    accurate nomenclature is used.

    Example:

    ```mlir
    %noe = sparse_tensor.number_of_entries %tensor : tensor<64x64xf64, #CSR>
    ```
  }];
  let assemblyFormat = "$tensor attr-dict `:` type($tensor)";
}

def SparseTensor_ConcatenateOp : SparseTensor_Op<"concatenate",
                                 [Pure, StageWithSortSparseOpInterface]>,
    Arguments<(ins Variadic<AnyRankedTensor>:$inputs, DimensionAttr:$dimension)>,
    Results<(outs AnyRankedTensor:$result)> {

  let summary = "Concatenates a list of tensors into a single tensor.";
  let description = [{
     Concatenates a list input tensors and the output tensor with the same
     dimension-rank.  The concatenation happens on the specified `dimension`
     (0 <= dimension < dimRank).  The resulting `dimension` size is the
     sum of all the input sizes for that dimension, while all the other
     dimensions should have the same size in the input and output tensors.

     Only statically-sized input tensors are accepted, while the output tensor
     can be dynamically-sized.

     Example:

     ```mlir
     %0 = sparse_tensor.concatenate %1, %2 { dimension = 0 : index }
       : tensor<64x64xf64, #CSR>, tensor<64x64xf64, #CSR> to tensor<128x64xf64, #CSR>
     ```
   }];

  let extraClassDeclaration = [{
     // Whether the concatenate can be done by a single step or it would require
     // an extra sort. Inherited from StageWithSortSparseOpInterface.
     bool needsExtraSort();
  }];

  let assemblyFormat = "$inputs attr-dict `:` type($inputs) `to` type($result)";
  let hasVerifier = 1;
}

def SparseTensor_ToSliceOffsetOp : SparseTensor_Op<"slice.offset", [Pure]>,
    Arguments<(ins AnySparseTensorSlice:$slice, IndexAttr:$dim)>,
    Results<(outs Index:$offset)> {
  let summary = "Extracts the offset of the sparse tensor slice at the given dimension";
  let description = [{
    Extracts the offset of the sparse tensor slice at the given dimension.

    Currently, sparse tensor slices are still a work in progress, and only
    works when runtime library is disabled (i.e., running the sparsifier
    with `enable-runtime-library=false`).

    Example:

    ```mlir
    %0 = tensor.extract_slice %s[%v1, %v2][64, 64][1, 1] : tensor<128x128xf64, #DCSR>
                                                        to tensor<64x64xf64, #Slice>

    %1 = sparse_tensor.slice.offset %0 at 0 : tensor<64x64xf64, #Slice>
    %2 = sparse_tensor.slice.offset %0 at 1 : tensor<64x64xf64, #Slice>
    // %1 = %v1
    // %2 = %v2
    ```
  }];
  let assemblyFormat = "$slice `at` $dim attr-dict `:` type($slice)";
  let hasVerifier = 1;
}

def SparseTensor_ToSliceStrideOp : SparseTensor_Op<"slice.stride", [Pure]>,
    Arguments<(ins AnySparseTensorSlice:$slice, IndexAttr:$dim)>,
    Results<(outs Index:$stride)> {
  let summary = "Extracts the stride of the sparse tensor slice at the given dimension";
  let description = [{
    Extracts the stride of the sparse tensor slice at the given dimension.

    Currently, sparse tensor slices are still a work in progress, and only
    works when runtime library is disabled (i.e., running the sparsifier
    with `enable-runtime-library=false`).

    Example:

    ```mlir
    %0 = tensor.extract_slice %s[%v1, %v2][64, 64][%s1, %s2] : tensor<128x128xf64, #DCSR>
                                                            to tensor<64x64xf64, #Slice>

    %1 = sparse_tensor.slice.stride %0 at 0 : tensor<64x64xf64, #Slice>
    %2 = sparse_tensor.slice.stride %0 at 1 : tensor<64x64xf64, #Slice>
    // %1 = %s1
    // %2 = %s2

    ```
  }];
  let assemblyFormat = "$slice `at` $dim attr-dict `:` type($slice)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Sparse Tensor Storage Specifier Operations.
//===----------------------------------------------------------------------===//

def SparseTensor_StorageSpecifierInitOp : SparseTensor_Op<"storage_specifier.init",
  [Pure]>,
    Arguments<(ins Optional<SparseTensorStorageSpecifier>:$source)>,
    Results<(outs SparseTensorStorageSpecifier:$result)> {
  let summary = "";
  let description = [{
    Returns an initial storage specifier value.  A storage specifier
    value holds the level-sizes, position arrays, coordinate arrays,
    and the value array.
    If this is a specifier for slices, it also holds the extra strides/offsets
    for each tensor dimension.

    TODO: The sparse tensor slice support is currently in a unstable state, and
    is subject to change in the future.

    Example:

    ```mlir
    #CSR = #sparse_tensor.encoding<{
      map = (i, j) -> (i : dense, j : compressed)
    }>
    #CSR_SLICE = #sparse_tensor.encoding<{
      map = (d0 : #sparse_tensor<slice(1, 4, 1)>,
             d1 : #sparse_tensor<slice(1, 4, 2)>) ->
            (d0 : dense, d1 : compressed)
    }>

    %0 = sparse_tensor.storage_specifier.init :  !sparse_tensor.storage_specifier<#CSR>
    %1 = sparse_tensor.storage_specifier.init with %src
         : !sparse_tensor.storage_specifier<#CSR> to
           !sparse_tensor.storage_specifier<#CSR_SLICE>
    ```
  }];

  let builders = [
    OpBuilder<(ins "Type":$result),
    [{
      build($_builder, $_state, result, Value());
    }]>
  ];

  let assemblyFormat = "attr-dict (`with` $source^)? `:` (`from` qualified(type($source))^ `to`)?"
                                                        " qualified(type($result))";
}

def SparseTensor_GetStorageSpecifierOp : SparseTensor_Op<"storage_specifier.get", [Pure]>,
    Arguments<(ins SparseTensorStorageSpecifier:$specifier,
                   SparseTensorStorageSpecifierKindAttr:$specifierKind,
                   OptionalAttr<LevelAttr>:$level)>,
    Results<(outs Index:$result)> {
  let summary = "";
  let description = [{
    Returns the requested field of the given storage_specifier.

    Example of querying the size of the coordinates array for level 0:

    ```mlir
    %0 = sparse_tensor.storage_specifier.get %arg0 crd_mem_sz at 0
         : !sparse_tensor.storage_specifier<#COO>
    ```
  }];

  let assemblyFormat = "$specifier $specifierKind (`at` $level^)? attr-dict"
                       "`:` qualified(type($specifier))";
  let hasVerifier = 1;
  let hasFolder = 1;
}

def SparseTensor_SetStorageSpecifierOp : SparseTensor_Op<"storage_specifier.set",
    [Pure, AllTypesMatch<["result", "specifier"]>]>,
    Arguments<(ins SparseTensorStorageSpecifier:$specifier,
                   SparseTensorStorageSpecifierKindAttr:$specifierKind,
                   OptionalAttr<LevelAttr>:$level,
                   Index:$value)>,
    Results<(outs SparseTensorStorageSpecifier:$result)> {
  let summary = "";
  let description = [{
    Set the field of the storage specifier to the given input value. Returns
    the updated storage_specifier as a new SSA value.

    Example of updating the sizes of the coordinates array for level 0:

    ```mlir
    %0 = sparse_tensor.storage_specifier.set %arg0 crd_mem_sz at 0 with %new_sz
       : !sparse_tensor.storage_specifier<#COO>
    ```
  }];
  let assemblyFormat = "$specifier $specifierKind (`at` $level^)? `with` $value"
                       " attr-dict `:` qualified(type($result))";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Sparse Tensor Coordinate Translation Operation.
//===----------------------------------------------------------------------===//

def SparseTensor_CrdTranslateOp : SparseTensor_Op<"crd_translate", [Pure]>,
    Arguments<(ins Variadic<Index>:$in_crds,
                   SparseTensorCrdTransDirectionAttr:$direction,
                   SparseTensorEncodingAttr:$oracle)>,
    Results<(outs Variadic<Index>:$out_crds)> {
  string summary = "Performs coordinate translation between level and dimension coordinate space.";
  string description = [{
    Performs coordinate translation between level and dimension coordinate space according
    to the provided affine maps.
  }];
  let assemblyFormat = "$direction `[` $in_crds `]` `as` $oracle attr-dict `:` type($out_crds)";
  let hasVerifier = 1;
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Sparse Tensor Management Operations. These operations are "impure" in the
// sense that some behavior is defined by side-effects. These operations provide
// a bridge between "sparsification" on one hand and a support library or actual
// code generation on the other hand. The semantics of these operations may be
// refined over time as our sparse abstractions evolve.
//===----------------------------------------------------------------------===//

def SparseTensor_InsertOp : SparseTensor_Op<"insert",
    [TypesMatchWith<"value type matches element type of tensor",
                    "tensor", "value",
                    "::llvm::cast<ShapedType>($_self).getElementType()">,
     AllTypesMatch<["tensor", "result"]>]>,
    Arguments<(ins AnyType:$value,
               AnySparseTensor:$tensor,
               Variadic<Index>:$lvlCoords)>,
    Results<(outs AnySparseTensor:$result)> {
  string summary = "Inserts a value into the sparse tensor";
  string description = [{
    Inserts the value into the underlying storage of the tensor at the
    given level-coordinates.  The arity of `lvlCoords` must match the
    level-rank of the tensor.  This operation can only be applied when
    the tensor materializes unintialized from a `bufferization.alloc_tensor`
    operation and the final tensor is constructed with a `load` operation
    which has the `hasInserts` attribute set.

    The level-properties of the sparse tensor type fully describe what
    kind of insertion order is allowed.  When all levels have "unique"
    and "ordered" properties, for example, insertions should occur in
    strict lexicographical level-coordinate order.  Other properties
    define different insertion regimens.  Inserting in a way contrary
    to these properties results in undefined behavior.

    Note that this operation is "impure" in the sense that even though
    the result is modeled through an SSA value, the insertion is eventually
    done "in place", and referencing the old SSA value is undefined behavior.
    This operation is scheduled to be unified with the dense counterpart
    `tensor.insert` that has pure SSA semantics.

    Example:

    ```mlir
    %result = sparse_tensor.insert %val into %tensor[%i,%j] : tensor<1024x1024xf64, #CSR>
    ```
  }];
  let assemblyFormat = "$value `into` $tensor `[` $lvlCoords `]` attr-dict"
                       "`:` type($tensor)";
  let hasVerifier = 1;
}

def SparseTensor_PushBackOp : SparseTensor_Op<"push_back",
    [TypesMatchWith<"value type matches element type of inBuffer",
                    "inBuffer", "value",
                    "::llvm::cast<ShapedType>($_self).getElementType()">,
     AllTypesMatch<["inBuffer", "outBuffer"]>]>,
    Arguments<(ins Index:$curSize,
               StridedMemRefRankOf<[AnyType], [1]>:$inBuffer,
               AnyType:$value, Optional<Index>:$n,
               UnitAttr:$inbounds)>,
    Results<(outs StridedMemRefRankOf<[AnyType], [1]>:$outBuffer,
             Index:$newSize)>  {
  string summary = "Pushes a value to the back of a given buffer";
  string description = [{
    Pushes `value` to the end of the given sparse tensor storage buffer
    `inBuffer` as indicated by the value of `curSize` and returns the
    new size of the buffer in `newSize` (`newSize = curSize + n`).
    The capacity of the buffer is recorded in the memref type of `inBuffer`.
    If the current buffer is full, then `inBuffer.realloc` is called before
    pushing the data to the buffer. This is similar to std::vector push_back.

    The optional input `n` specifies the number of times to repeately push
    the value to the back of the tensor. When `n` is a compile-time constant,
    its value can't be less than 1. If `n` is a runtime value that is less
    than 1, the behavior is undefined. Although using input `n` is semantically
    equivalent to calling push_back n times, it gives compiler more chances to
    to optimize the memory reallocation and the filling of the memory with the
    same value.

    The `inbounds` attribute tells the compiler that the insertion won't go
    beyond the current storage buffer. This allows the compiler to not generate
    the code for capacity check and reallocation. The typical usage will be for
    "dynamic" sparse tensors for which a capacity can be set beforehand.

    Note that this operation is "impure" in the sense that even though
    the result is modeled through an SSA value, referencing the memref
    through the old SSA value after this operation is undefined behavior.

    Example:

    ```mlir
    %buf, %newSize = sparse_tensor.push_back %curSize, %buffer, %val
       : index, memref<?xf64>, f64
    ```

    ```mlir
    %buf, %newSize = sparse_tensor.push_back inbounds %curSize, %buffer, %val
       : xindex, memref<?xf64>, f64
    ```

    ```mlir
    %buf, %newSize = sparse_tensor.push_back inbounds %curSize, %buffer, %val, %n
       : xindex, memref<?xf64>, f64
    ```
  }];
  let assemblyFormat = "(`inbounds` $inbounds^)? $curSize `,` $inBuffer"
                       " `,` $value (`,` $n^ )?  attr-dict `:`"
                       " type($curSize) `,` type($inBuffer) `,`"
                       " type($value) (`,` type($n)^ )?";

  let builders = [
    // Build an op (reusing type from curSize and inBuffer) without input `n`
    OpBuilder<(ins "Value":$curSize, "Value":$inBuffer, "Value":$value)>
  ];

  let hasVerifier = 1;
}

def SparseTensor_ExpandOp : SparseTensor_Op<"expand", []>,
    Arguments<(ins AnySparseTensor:$tensor)>,
    Results<(outs AnyStridedMemRefOfRank<1>:$values,
                  StridedMemRefRankOf<[I1],[1]>:$filled,
                  StridedMemRefRankOf<[Index],[1]>:$added,
                  Index:$count)> {
  string summary = "Expands an access pattern for insertion";
  string description = [{
    Performs an access pattern expansion for the innermost levels of the
    given tensor. This operation is useful to implement kernels in which a
    sparse tensor appears as output. This technique is known under several
    different names and using several alternative implementations,
    for example, phase counter [Gustavson72], expanded or switch array
    [Pissanetzky84], in phase scan [Duff90], access pattern expansion [Bik96],
    and workspaces [Kjolstad19].

    The `values` and `filled` arrays must have lengths equal to the
    level-size of the innermost level (i.e., as if the innermost level
    were *dense*).  The `added` array and `count` are used to store new
    level-coordinates when a false value is encountered in the `filled`
    array.  All arrays should be allocated before the loop (possibly even
    shared between loops in a future optimization) so that their *dense*
    initialization can be amortized over many iterations.  Setting and
    resetting the dense arrays in the loop nest itself is kept *sparse*
    by only iterating over set elements through an indirection using
    the added array, so that the operations are kept proportional to
    the number of nonzeros.

    Note that this operation is "impure" in the sense that even though the
    results are modeled through SSA values, the operation relies on a proper
    side-effecting context that sets and resets the expanded arrays.

    Example:

    ```mlir
    %values, %filled, %added, %count = sparse_tensor.expand %tensor
      : tensor<4x4xf64, #CSR> to memref<?xf64>, memref<?xi1>, memref<?xindex>
    ```
  }];
  let assemblyFormat = "$tensor attr-dict `:` type($tensor) `to` type($values)"
                       " `,` type($filled) `,` type($added)";
}

def SparseTensor_CompressOp : SparseTensor_Op<"compress",
    [AllTypesMatch<["tensor", "result"]>]>,
    Arguments<(ins AnyStridedMemRefOfRank<1>:$values,
                   StridedMemRefRankOf<[I1],[1]>:$filled,
                   StridedMemRefRankOf<[Index],[1]>:$added,
                   Index:$count,
                   AnySparseTensor:$tensor,
                   Variadic<Index>:$lvlCoords)>,
    Results<(outs AnySparseTensor:$result)> {
  string summary = "Compressed an access pattern for insertion";
  string description = [{
    Finishes a single access pattern expansion by moving inserted elements
    into the sparse storage scheme of the given tensor with the given
    level-coordinates.  The arity of `lvlCoords` is one less than the
    level-rank of the tensor, with the coordinate of the innermost
    level defined through the `added` array.  The `values` and `filled`
    arrays are reset in a *sparse* fashion by only iterating over set
    elements through an indirection using the `added` array, so that
    the operations are kept proportional to the number of nonzeros.
    See the `sparse_tensor.expand` operation for more details.

    Note that this operation is "impure" in the sense that even though
    the result is modeled through an SSA value, the insertion is eventually
    done "in place", and referencing the old SSA value is undefined behavior.

    Example:

    ```mlir
    %result = sparse_tensor.compress %values, %filled, %added, %count into %tensor[%i]
      : memref<?xf64>, memref<?xi1>, memref<?xindex>, tensor<4x4xf64, #CSR>
    ```
  }];
  let assemblyFormat = "$values `,` $filled `,` $added `,` $count"
                       " `into` $tensor `[` $lvlCoords `]` attr-dict"
                       " `:` type($values) `,` type($filled) `,` type($added)"
                       " `,` type($tensor)";
  let hasVerifier = 1;
}

def SparseTensor_LoadOp : SparseTensor_Op<"load", [SameOperandsAndResultType]>,
    Arguments<(ins AnySparseTensor:$tensor, UnitAttr:$hasInserts)>,
    Results<(outs AnyTensor:$result)> {
  let summary =
    "Rematerializes tensor from underlying sparse storage format";
  let description = [{
    Rematerializes a tensor from the underlying sparse storage format of the
    given tensor. This is similar to the `bufferization.to_tensor` operation
    in the sense that it provides a bridge between a bufferized world view
    and a tensor world view. Unlike the `bufferization.to_tensor` operation,
    however, this sparse operation is used only temporarily to maintain a
    correctly typed intermediate representation during progressive
    bufferization.

    The `hasInserts` attribute denote whether insertions to the underlying
    sparse storage format may have occurred, in which case the underlying
    sparse storage format needs to be finalized. Otherwise, the operation
    simply folds away.

    Note that this operation is "impure" in the sense that even though
    the result is modeled through an SSA value, the operation relies on
    a proper context of materializing and inserting the tensor value.

    Examples:

    ```mlir
    %result = sparse_tensor.load %tensor : tensor<8xf64, #SV>

    %1 = sparse_tensor.load %0 hasInserts : tensor<16x32xf32, #CSR>
    ```
  }];
  let assemblyFormat = "$tensor (`hasInserts` $hasInserts^)? attr-dict `:` type($tensor)";
}

def SparseTensor_OutOp : SparseTensor_Op<"out", []>,
    Arguments<(ins AnySparseTensor:$tensor, AnyType:$dest)> {
  string summary = "Outputs a sparse tensor to the given destination";
  string description = [{
    Outputs the contents of a sparse tensor to the destination defined by an
    opaque pointer provided by `dest`. For targets that have access to a file
    system, for example, this pointer may specify a filename (or file) for output.
    The form of the operation is kept deliberately very general to allow for
    alternative implementations in the future, such as sending the contents to
    a buffer defined by a pointer.

    Note that this operation is "impure" in the sense that its behavior
    is solely defined by side-effects and not SSA values.

    Example:

    ```mlir
    sparse_tensor.out %t, %dest : tensor<1024x1024xf64, #CSR>, !Dest
    ```
  }];
  let assemblyFormat = "$tensor `,` $dest attr-dict `:` type($tensor) `,` type($dest)";
}

//===----------------------------------------------------------------------===//
// Sparse Tensor Sorting/Ordering Operations.
//===----------------------------------------------------------------------===//

def SparseTensor_SortOp : SparseTensor_Op<"sort">,
    Arguments<(ins Index:$n, StridedMemRefRankOf<[AnyInteger, Index], [1]>:$xy,
               Variadic<StridedMemRefRankOf<[AnyType], [1]>>:$ys,
               AffineMapAttr:$perm_map, OptionalAttr<IndexAttr>:$ny,
               SparseTensorSortKindAttr:$algorithm)>  {
  let summary = "Sorts the arrays in xs and ys lexicographically on the "
                "integral values found in the xs list";
  let description = [{
    Sparse_tensor.sort sort the `xs` values along with some `ys` values
    that are put in a single linear buffer `xy`.
    The affine map attribute `perm_map` specifies the permutation to be applied on
    the `xs` before comparison, the rank of the permutation map
    also specifies the number of `xs` values in `xy`.
    The optional index attribute `ny` provides the number of `ys` values in `xy`.
    When `ny` is not explicitly specified, its value is 0.
    This instruction supports a more efficient way to store the COO definition
    in sparse tensor type.

    The buffer xy should have a dimension not less than n * (rank(perm_map) + ny) while the
    buffers in `ys` should have a dimension not less than `n`. The behavior of
    the operator is undefined if this condition is not met.

    Example:

    ```mlir
    sparse_tensor.sort insertion_sort_stable %n, %x { perm_map = affine_map<(i,j) -> (j,i)> }
      : memref<?xindex>
    ```
  }];

  let assemblyFormat = "$algorithm $n"
                       "`,`$xy (`jointly` $ys^)? attr-dict"
                       "`:` type($xy) (`jointly` type($ys)^)?";
  let hasVerifier = 1;
}

def SparseTensor_ReorderCOOOp : SparseTensor_Op<"reorder_coo", [Pure]>,
    Arguments<(ins AnyCOOSparseTensor: $input_coo,
                   SparseTensorSortKindAttr:$algorithm)>,
    Results<(outs AnyCOOSparseTensor: $result_coo)> {
  let summary = "Reorder the input COO such that it has the the same order as "
                "the output COO";
  let description = [{
    sparse_tensor.reorder_coo reorder input COO to the same order as specified by
    the output format. E.g., reorder an unordered COO into an ordered one.

    The input and result COO tensor must have the same element type, position type and
    coordinate type. At the moment, the operation also only supports ordering
    input and result COO with the same dim2lvl map.

    Example:

    ```mlir
    %res = sparse_tensor.reorder_coo quick_sort %coo : tensor<?x?xf64 : #Unordered_COO> to
                                                       tensor<?x?xf64 : #Ordered_COO>

    ```
  }];

  let assemblyFormat = "$algorithm $input_coo attr-dict"
                       "`:` type($input_coo) `to` type($result_coo)";

  let hasFolder = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Sparse Tensor Syntax Operations.
//===----------------------------------------------------------------------===//

def SparseTensor_BinaryOp : SparseTensor_Op<"binary", [Pure]>,
    Arguments<(ins AnyType:$x, AnyType:$y, UnitAttr:$left_identity, UnitAttr:$right_identity)>,
    Results<(outs AnyType:$output)> {
  let summary = "Binary set operation utilized within linalg.generic";
  let description = [{
      Defines a computation within a `linalg.generic` operation that takes two
      operands and executes one of the regions depending on whether both operands
      or either operand is nonzero (i.e. stored explicitly in the sparse storage
      format).

      Three regions are defined for the operation and must appear in this order:
      - overlap (elements present in both sparse tensors)
      - left (elements only present in the left sparse tensor)
      - right (element only present in the right sparse tensor)

      Each region contains a single block describing the computation and result.
      Every non-empty block must end with a sparse_tensor.yield and the return
      type must match the type of `output`. The primary region's block has two
      arguments, while the left and right region's block has only one argument.

      A region may also be declared empty (i.e. `left={}`), indicating that the
      region does not contribute to the output. For example, setting both
      `left={}` and `right={}` is equivalent to the intersection of the two
      inputs as only the overlap region will contribute values to the output.

      As a convenience, there is also a special token `identity` which can be
      used in place of the left or right region. This token indicates that
      the return value is the input value (i.e. func(%x) => return %x).
      As a practical example, setting `left=identity` and `right=identity`
      would be equivalent to a union operation where non-overlapping values
      in the inputs are copied to the output unchanged.

      Due to the possibility of empty regions, i.e. lack of a value for certain
      cases, the result of this operation may only feed directly into the output
      of the `linalg.generic` operation or into into a custom reduction
      `sparse_tensor.reduce` operation that follows in the same region.

      Example of isEqual applied to intersecting elements only:

      ```mlir
      %C = bufferization.alloc_tensor...
      %0 = linalg.generic #trait
        ins(%A: tensor<?xf64, #SparseVector>,
            %B: tensor<?xf64, #SparseVector>)
        outs(%C: tensor<?xi8, #SparseVector>) {
        ^bb0(%a: f64, %b: f64, %c: i8) :
          %result = sparse_tensor.binary %a, %b : f64, f64 to i8
            overlap={
              ^bb0(%arg0: f64, %arg1: f64):
                %cmp = arith.cmpf "oeq", %arg0, %arg1 : f64
                %ret_i8 = arith.extui %cmp : i1 to i8
                sparse_tensor.yield %ret_i8 : i8
            }
            left={}
            right={}
          linalg.yield %result : i8
      } -> tensor<?xi8, #SparseVector>
      ```

      Example of A+B in upper triangle, A-B in lower triangle:

      ```mlir
      %C = bufferization.alloc_tensor...
      %1 = linalg.generic #trait
        ins(%A: tensor<?x?xf64, #CSR>, %B: tensor<?x?xf64, #CSR>
        outs(%C: tensor<?x?xf64, #CSR> {
        ^bb0(%a: f64, %b: f64, %c: f64) :
          %row = linalg.index 0 : index
          %col = linalg.index 1 : index
          %result = sparse_tensor.binary %a, %b : f64, f64 to f64
            overlap={
              ^bb0(%x: f64, %y: f64):
                %cmp = arith.cmpi "uge", %col, %row : index
                %upperTriangleResult = arith.addf %x, %y : f64
                %lowerTriangleResult = arith.subf %x, %y : f64
                %ret = arith.select %cmp, %upperTriangleResult, %lowerTriangleResult : f64
                sparse_tensor.yield %ret : f64
            }
            left=identity
            right={
              ^bb0(%y: f64):
                %cmp = arith.cmpi "uge", %col, %row : index
                %lowerTriangleResult = arith.negf %y : f64
                %ret = arith.select %cmp, %y, %lowerTriangleResult : f64
                sparse_tensor.yield %ret : f64
            }
          linalg.yield %result : f64
      } -> tensor<?x?xf64, #CSR>
      ```

      Example of set difference. Returns a copy of A where its sparse structure
      is *not* overlapped by B. The element type of B can be different than A
      because we never use its values, only its sparse structure:

      ```mlir
      %C = bufferization.alloc_tensor...
      %2 = linalg.generic #trait
        ins(%A: tensor<?x?xf64, #CSR>, %B: tensor<?x?xi32, #CSR>
        outs(%C: tensor<?x?xf64, #CSR> {
        ^bb0(%a: f64, %b: i32, %c: f64) :
          %result = sparse_tensor.binary %a, %b : f64, i32 to f64
            overlap={}
            left=identity
            right={}
          linalg.yield %result : f64
      } -> tensor<?x?xf64, #CSR>
      ```
  }];

  let regions = (region AnyRegion:$overlapRegion, AnyRegion:$leftRegion, AnyRegion:$rightRegion);
  let assemblyFormat = [{
        $x `,` $y `:` attr-dict type($x) `,` type($y) `to` type($output) `\n`
        `overlap` `=` $overlapRegion `\n`
        `left` `=` (`identity` $left_identity^):($leftRegion)? `\n`
        `right` `=` (`identity` $right_identity^):($rightRegion)?
  }];
  let hasVerifier = 1;
}

def SparseTensor_UnaryOp : SparseTensor_Op<"unary", [Pure]>,
    Arguments<(ins AnyType:$x)>,
    Results<(outs AnyType:$output)> {
  let summary = "Unary set operation utilized within linalg.generic";
  let description = [{
      Defines a computation with a `linalg.generic` operation that takes a single
      operand and executes one of two regions depending on whether the operand is
      nonzero (i.e. stored explicitly in the sparse storage format).

      Two regions are defined for the operation must appear in this order:
      - present (elements present in the sparse tensor)
      - absent (elements not present in the sparse tensor)

      Each region contains a single block describing the computation and result.
      A non-empty block must end with a sparse_tensor.yield and the return type
      must match the type of `output`. The primary region's block has one
      argument, while the missing region's block has zero arguments.

      A region may also be declared empty (i.e. `absent={}`), indicating that the
      region does not contribute to the output.

      Due to the possibility of empty regions, i.e. lack of a value for certain
      cases, the result of this operation may only feed directly into the output
      of the `linalg.generic` operation or into into a custom reduction
      `sparse_tensor.reduce` operation that follows in the same region.

      Example of A+1, restricted to existing elements:

      ```mlir
      %C = bufferization.alloc_tensor...
      %0 = linalg.generic #trait
         ins(%A: tensor<?xf64, #SparseVector>)
        outs(%C: tensor<?xf64, #SparseVector>) {
        ^bb0(%a: f64, %c: f64) :
          %result = sparse_tensor.unary %a : f64 to f64
            present={
              ^bb0(%arg0: f64):
                %cf1 = arith.constant 1.0 : f64
                %ret = arith.addf %arg0, %cf1 : f64
                sparse_tensor.yield %ret : f64
            }
            absent={}
          linalg.yield %result : f64
      } -> tensor<?xf64, #SparseVector>
      ```

      Example returning +1 for existing values and -1 for missing values:

      ```mlir
      %C = bufferization.alloc_tensor...
      %1 = linalg.generic #trait
         ins(%A: tensor<?xf64, #SparseVector>)
        outs(%C: tensor<?xf64, #SparseVector>) {
        ^bb0(%a: f64, %c: f64) :
          %result = sparse_tensor.unary %a : f64 to i32
            present={
            ^bb0(%x: f64):
              %ret = arith.constant 1 : i32
              sparse_tensor.yield %ret : i32
          }
          absent={
            %ret = arith.constant -1 : i32
            sparse_tensor.yield %ret : i32
          }
          linalg.yield %result : f64
      } -> tensor<?xf64, #SparseVector>
      ```

      Example showing a structural inversion (existing values become missing in
      the output, while missing values are filled with 1):

      ```mlir
      %C = bufferization.alloc_tensor...
      %2 = linalg.generic #trait
          ins(%A: tensor<?xf64, #SparseVector>)
          outs(%C: tensor<?xf64, #SparseVector>) {
            %result = sparse_tensor.unary %a : f64 to i64
              present={}
              absent={
                %ret = arith.constant 1 : i64
                sparse_tensor.yield %ret : i64
              }
          linalg.yield %result : f64
      } -> tensor<?xf64, #SparseVector>
      ```
  }];

  let regions = (region AnyRegion:$presentRegion, AnyRegion:$absentRegion);
  let assemblyFormat = [{
        $x attr-dict `:` type($x) `to` type($output) `\n`
        `present` `=` $presentRegion `\n`
        `absent` `=` $absentRegion
  }];
  let hasVerifier = 1;
}

def SparseTensor_ReduceOp : SparseTensor_Op<"reduce", [Pure, SameOperandsAndResultType]>,
    Arguments<(ins AnyType:$x, AnyType:$y, AnyType:$identity)>,
    Results<(outs AnyType:$output)> {
  let summary = "Custom reduction operation utilized within linalg.generic";
  let description = [{
      Defines a computation with a `linalg.generic` operation that takes two
      operands and an identity value and reduces all stored values down to a
      single result based on the computation in the region.

      The region must contain exactly one block taking two arguments. The block
      must end with a sparse_tensor.yield and the output must match the input
      argument types.

      Note that this operation is only required for custom reductions beyond
      the standard reduction operations (add, sub, or, xor) that can be
      sparsified by merely reducing the stored values. More elaborate reduction
      operations (mul, and, min, max, etc.) would need to account for implicit
      zeros as well. They can still be handled using this custom reduction
      operation. The `linalg.generic` `iterator_types` defines which indices
      are being reduced. When the associated operands are used in an operation,
      a reduction will occur. The use of this explicit `reduce` operation
      is not required in most cases.

      Example of Matrix->Vector reduction using max(product(x_i), 100):

      ```mlir
      %cf1 = arith.constant 1.0 : f64
      %cf100 = arith.constant 100.0 : f64
      %C = bufferization.alloc_tensor...
      %0 = linalg.generic #trait
         ins(%A: tensor<?x?xf64, #SparseMatrix>)
        outs(%C: tensor<?xf64, #SparseVector>) {
        ^bb0(%a: f64, %c: f64) :
          %result = sparse_tensor.reduce %c, %a, %cf1 : f64 {
              ^bb0(%arg0: f64, %arg1: f64):
                %0 = arith.mulf %arg0, %arg1 : f64
                %cmp = arith.cmpf "ogt", %0, %cf100 : f64
                %ret = arith.select %cmp, %cf100, %0 : f64
                sparse_tensor.yield %ret : f64
            }
          linalg.yield %result : f64
      } -> tensor<?xf64, #SparseVector>
      ```
  }];

  let regions = (region SizedRegion<1>:$region);

  let assemblyFormat = [{
         $x `,` $y `,` $identity attr-dict `:` type($output) $region
  }];
  let hasVerifier = 1;
}

def SparseTensor_SelectOp : SparseTensor_Op<"select", [Pure, SameOperandsAndResultType]>,
    Arguments<(ins AnyType:$x)>,
    Results<(outs AnyType:$output)> {
  let summary = "Select operation utilized within linalg.generic";
  let description = [{
      Defines an evaluation within a `linalg.generic` operation that takes a single
      operand and decides whether or not to keep that operand in the output.

      A single region must contain exactly one block taking one argument. The block
      must end with a sparse_tensor.yield and the output type must be boolean.

      Value threshold is an obvious usage of the select operation. However, by using
      `linalg.index`, other useful selection can be achieved, such as selecting the
      upper triangle of a matrix.

      Example of selecting A >= 4.0:

      ```mlir
      %C = bufferization.alloc_tensor...
      %0 = linalg.generic #trait
         ins(%A: tensor<?xf64, #SparseVector>)
        outs(%C: tensor<?xf64, #SparseVector>) {
        ^bb0(%a: f64, %c: f64) :
          %result = sparse_tensor.select %a : f64 {
              ^bb0(%arg0: f64):
                %cf4 = arith.constant 4.0 : f64
                %keep = arith.cmpf "uge", %arg0, %cf4 : f64
                sparse_tensor.yield %keep : i1
            }
          linalg.yield %result : f64
      } -> tensor<?xf64, #SparseVector>
      ```

      Example of selecting lower triangle of a matrix:

      ```mlir
      %C = bufferization.alloc_tensor...
      %1 = linalg.generic #trait
         ins(%A: tensor<?x?xf64, #CSR>)
        outs(%C: tensor<?x?xf64, #CSR>) {
        ^bb0(%a: f64, %c: f64) :
          %row = linalg.index 0 : index
          %col = linalg.index 1 : index
          %result = sparse_tensor.select %a : f64 {
              ^bb0(%arg0: f64):
                %keep = arith.cmpf "olt", %col, %row : f64
                sparse_tensor.yield %keep : i1
            }
          linalg.yield %result : f64
      } -> tensor<?x?xf64, #CSR>
      ```
  }];

  let regions = (region SizedRegion<1>:$region);
  let assemblyFormat = [{
         $x attr-dict `:` type($x) $region
  }];
  let hasVerifier = 1;
}

def SparseTensor_YieldOp : SparseTensor_Op<"yield", [Pure, Terminator]>,
    Arguments<(ins Optional<AnyType>:$result)> {
  let summary = "Yield from sparse_tensor set-like operations";
  let description = [{
      Yields a value from within a `binary`, `unary`, `reduce`,
      `select` or `foreach` block.

      Example:

      ```mlir
      %0 = sparse_tensor.unary %a : i64 to i64 {
        present={
          ^bb0(%arg0: i64):
            %cst = arith.constant 1 : i64
            %ret = arith.addi %arg0, %cst : i64
            sparse_tensor.yield %ret : i64
        }
      }
      ```
  }];

  let builders = [
    OpBuilder<(ins),
    [{
      build($_builder, $_state, Value());
    }]>
  ];

  let assemblyFormat = [{
        $result attr-dict `:` type($result)
  }];
  let hasVerifier = 1;
}

def SparseTensor_ForeachOp : SparseTensor_Op<"foreach",
    [SingleBlockImplicitTerminator<"YieldOp">]>,
     Arguments<(ins AnyTensor:$tensor,
                    Variadic<AnyType>:$initArgs,
                    OptionalAttr<AffineMapAttr>:$order)>,
     Results<(outs Variadic<AnyType>:$results)> {
  let summary = "Iterates over elements in a tensor";
  let description = [{
     Iterates over stored elements in a tensor (which are typically, but not always,
     non-zero for sparse tensors) and executes the block.

     `tensor`: the input tensor to iterate over.
     `initArgs`: the initial loop argument to carry and update during each iteration.
     `order`: an optional permutation affine map that specifies the order in which
     the dimensions are visited (e.g., row first or column first). This is only
     applicable when the input tensor is a non-annotated dense tensor.

     For an input tensor with dim-rank `n`, the block must take `n + 1`
     arguments (plus additional loop-carried variables as described below).
     The first `n` arguments provide the dimension-coordinates of the element
     being visited, and must all have `index` type.  The `(n+1)`-th argument
     provides the element's value, and must have the tensor's element type.

     `sparse_tensor.foreach` can also operate on loop-carried variables and returns
     the final values after loop termination. The initial values of the variables are
     passed as additional SSA operands to the "sparse_tensor.foreach" following the n + 1
     SSA values mentioned above (n coordinates and 1 value).

     The region must terminate with a "sparse_tensor.yield" that passes the current
     values of all loop-carried variables to the next iteration, or to the
     result, if at the last iteration. The number and static types of loop-carried
     variables may not change with iterations.

     For example:
     ```mlir
     %c0 = arith.constant 0 : i32
     %ret = sparse_tensor.foreach in %0 init(%c0): tensor<?x?xi32, #DCSR>, i32 -> i32 do {
      ^bb0(%arg1: index, %arg2: index, %arg3: i32, %iter: i32):
        %sum = arith.add %iter, %arg3
        sparse_tensor.yield %sum
     }
     ```

     It is important to note that the generated loop iterates over
     elements in their storage order.  However, regardless of the
     storage scheme used by the tensor, the block is always given
     the dimension-coordinates.

     For example:
     ```mlir
     #COL_MAJOR = #sparse_tensor.encoding<{
       map = (d0, d1) -> (d1 : compressed, d0 : compressed)
     }>

     // foreach on a column-major sparse tensor
     sparse_tensor.foreach in %0 : tensor<2x3xf64, #COL_MAJOR> do {
      ^bb0(%row: index, %col: index, %arg3: f64):
         // [%row, %col] -> [0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]
     }

     #ROW_MAJOR = #sparse_tensor.encoding<{
       map = (d0, d1) -> (d0 : compressed, d1 : compressed)
     }>

     // foreach on a row-major sparse tensor
     sparse_tensor.foreach in %0 : tensor<2x3xf64, #ROW_MAJOR> do {
      ^bb0(%row: index, %col: index, %arg3: f64):
         // [%row, %col] -> [0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]
     }

     // foreach on a row-major dense tensor but visit column first
     sparse_tensor.foreach in %0 {order=affine_map<(i,j)->(j,i)>}: tensor<2x3xf64> do {
      ^bb0(%row: index, %col: index, %arg3: f64):
         // [%row, %col] -> [0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]
     }

     ```
  }];

  let builders = [
    OpBuilder<(ins "Value":$tensor, "ValueRange":$iterArgs, "AffineMapAttr":$order,
      "function_ref<void(OpBuilder &, Location, ValueRange, Value, ValueRange)>")>,
    OpBuilder<(ins "Value":$tensor, "AffineMapAttr":$order,
      "function_ref<void(OpBuilder &, Location, ValueRange, Value, ValueRange)>":$bodyBuilder),
    [{
      build($_builder, $_state, tensor, ValueRange(), order, bodyBuilder);
    }]>,
    OpBuilder<(ins "Value":$tensor,
      "function_ref<void(OpBuilder &, Location, ValueRange, Value, ValueRange)>":$bodyBuilder),
    [{
      build($_builder, $_state, tensor, ValueRange(), nullptr, bodyBuilder);
    }]>,
    OpBuilder<(ins "Value":$tensor, "ValueRange":$iterArgs,
      "function_ref<void(OpBuilder &, Location, ValueRange, Value, ValueRange)>":$bodyBuilder),
    [{
      build($_builder, $_state, tensor, iterArgs, nullptr, bodyBuilder);
    }]>
  ];

  let regions = (region SizedRegion<1>:$region);
  let assemblyFormat = "`in` $tensor (`init``(`$initArgs^`)`)? attr-dict"
                       "    `:` type($tensor) (`,` type($initArgs)^)?"
                       "  (`->` type($results)^)?  `do` $region";
  let hasVerifier = 1;
}

#endif // SPARSETENSOR_OPS
