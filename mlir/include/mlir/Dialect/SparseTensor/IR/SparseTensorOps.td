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
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/LoopLikeInterface.td"

//===----------------------------------------------------------------------===//
// Base class.
//===----------------------------------------------------------------------===//

class SparseTensor_Op<string mnemonic, list<Trait> traits = []>
  : Op<SparseTensor_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Sparse Tensor Operations.
//===----------------------------------------------------------------------===//

def SparseTensor_NewOp : SparseTensor_Op<"new", [Pure]> {
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

  let arguments = (ins AnyType:$source);
  let results = (outs AnySparseTensor:$result);
  let assemblyFormat = "$source attr-dict `:` type($source) `to` type($result)";
}

def SparseTensor_AssembleOp : SparseTensor_Op<"assemble", [Pure]> {
  let summary = "Returns a sparse tensor assembled from the given levels and values";
  let description = [{
    Assembles the per-level position and coordinate arrays together with
    the values arrays into a sparse tensor. The order and types of the
    provided levels must be consistent with the actual storage layout of
    the returned sparse tensor described below.

    - `levels: [tensor<? x iType>, ...]`
      supplies the sparse tensor position and coordinate arrays
      of the sparse tensor for the corresponding level as specifed by
      `sparse_tensor::StorageLayout`.
    - `values : tensor<? x V>`
      supplies the values array for the stored elements in the sparse tensor.

    This operation can be used to assemble a sparse tensor from an
    external source; e.g., by passing numpy arrays from Python. It
    is the user's responsibility to provide input that can be correctly
    interpreted by the sparsifier, which does not perform any sanity
    test to verify data integrity.

    Example:

    ```mlir
    %pos    = arith.constant dense<[0, 3]>                : tensor<2xindex>
    %index  = arith.constant dense<[[0,0], [1,2], [1,3]]> : tensor<3x2xindex>
    %values = arith.constant dense<[ 1.1,   2.2,   3.3 ]> : tensor<3xf64>
    %s = sparse_tensor.assemble (%pos, %index), %values
       : (tensor<2xindex>, tensor<3x2xindex>), tensor<3xf64> to tensor<3x4xf64, #COO>
    // yields COO format |1.1, 0.0, 0.0, 0.0|
    //     of 3x4 matrix |0.0, 0.0, 2.2, 3.3|
    //                   |0.0, 0.0, 0.0, 0.0|
    ```
  }];

  let arguments = (ins Variadic<TensorOf<[AnySignlessIntegerOrIndex]>>:$levels,
                   TensorOf<[AnyType]>:$values);
  let results = (outs AnySparseTensor: $result);
  let assemblyFormat =
    "` ` `(` $levels       `)` `,` $values attr-dict `:`"
    "    `(` type($levels) `)` `,` type($values) `to` type($result)";

  let hasVerifier = 1;
}

def SparseTensor_DisassembleOp : SparseTensor_Op<"disassemble", [Pure, SameVariadicResultSize]> {
  let summary = "Copies the levels and values of the given sparse tensor";
  let description = [{
    The disassemble operation is the inverse of `sparse_tensor::assemble`.
    It copies the per-level position and coordinate arrays together with
    the values array of the given sparse tensor into the user-supplied buffers
    along with the actual length of the memory used in each returned buffer.

    This operation can be used for returning a disassembled MLIR sparse tensor;
    e.g., copying the sparse tensor contents into pre-allocated numpy arrays
    back to Python. It is the user's responsibility to allocate large enough
    buffers of the appropriate types to hold the sparse tensor contents.
    The sparsifier simply copies all fields of the sparse tensor into the
    user-supplied buffers without any sanity test to verify data integrity.

    Example:

    ```mlir
    // input COO format |1.1, 0.0, 0.0, 0.0|
    //    of 3x4 matrix |0.0, 0.0, 2.2, 3.3|
    //                  |0.0, 0.0, 0.0, 0.0|
    %p, %c, %v, %p_len, %c_len, %v_len =
      sparse_tensor.disassemble %s : tensor<3x4xf64, #COO>
         out_lvls(%op, %oi : tensor<2xindex>, tensor<3x2xindex>)
         out_vals(%od : tensor<3xf64>) ->
           (tensor<2xindex>, tensor<3x2xindex>), tensor<3xf64>, (index, index), index
    // %p = arith.constant dense<[ 0,              3 ]> : tensor<2xindex>
    // %c = arith.constant dense<[[0,0], [1,2], [1,3]]> : tensor<3x2xindex>
    // %v = arith.constant dense<[ 1.1,   2.2,   3.3 ]> : tensor<3xf64>
    // %p_len = 2
    // %c_len = 6 (3x2)
    // %v_len = 3
    ```
  }];

  let arguments = (ins AnySparseTensor:$tensor,
                   Variadic<TensorOf<[AnySignlessIntegerOrIndex]>>:$out_levels,
                   TensorOf<[AnyType]>:$out_values);
  let results = (outs Variadic<TensorOf<[AnySignlessIntegerOrIndex]>>:$ret_levels,
                  TensorOf<[AnyType]>:$ret_values,
                  Variadic<AnyIndexingScalarLike>:$lvl_lens,
                  AnyIndexingScalarLike:$val_len);
  let assemblyFormat =
    "$tensor attr-dict `:` type($tensor)"
    "`out_lvls` `(` $out_levels `:` type($out_levels) `)` "
    "`out_vals` `(` $out_values `:` type($out_values) `)` `->`"
    "`(` type($ret_levels) `)` `,` type($ret_values) `,` "
    "`(` type($lvl_lens)   `)` `,` type($val_len)";

  let hasVerifier = 1;
}

def SparseTensor_ConvertOp : SparseTensor_Op<"convert",
  [Pure, StageWithSortSparseOpInterface]> {
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

  let arguments = (ins AnyTensor:$source);
  let results = (outs AnyTensor:$dest);
  let assemblyFormat = "$source attr-dict `:` type($source) `to` type($dest)";

  let extraClassDeclaration = [{
     // Whether the convert can be done by a single step or it would require
     // an extra sort. Inherited from StageWithSortSparseOpInterface.
     bool needsExtraSort();
  }];

  let hasFolder = 1;
  let hasVerifier = 1;
}

def SparseTensor_ReinterpretMapOp : SparseTensor_Op<"reinterpret_map",
    [NoMemoryEffect]> {
  let summary = "Reinterprets the dimension/level maps of the source tensor";
  let description = [{
    Reinterprets the dimension-to-level and level-to-dimension map specified in
    `source` according to the type of `dest`.
    `reinterpret_map` is a no-op and is introduced merely to resolve type conflicts.
    It does not make any modification to the source tensor and source/dest tensors
    are considered to be aliases.

    `source` and `dest` tensors are "reinterpretable" if and only if they have
    the exactly same storage at a low level.
    That is, both `source` and `dest` has the same number of levels and level types,
    and their shape is consistent before and after `reinterpret_map`.

    Example:
    ```mlir
    #CSC = #sparse_tensor.encoding<{
      map = (d0, d1) -> (d1: dense, d0: compressed)
    }>
    #CSR = #sparse_tensor.encoding<{
      map = (d0, d1) -> (d0: dense, d1: compressed)
    }>
    %t1 = sparse_tensor.reinterpret_map %t0 : tensor<3x4xi32, #CSC> to tensor<4x3xi32, #CSR>

    #BSR = #sparse_tensor.encoding<{
      map = ( i, j ) -> ( i floordiv 2 : dense,
                          j floordiv 3 : compressed,
                          i mod 2      : dense,
                          j mod 3      : dense
      )
    }>
    #DSDD = #sparse_tensor.encoding<{
      map = (i, j, k, l) -> (i: dense, j: compressed, k: dense, l: dense)
    }>
    %t1 = sparse_tensor.reinterpret_map %t0 : tensor<6x12xi32, #BSR> to tensor<3x4x2x3xi32, #DSDD>
    ```
    }];

  let arguments = (ins AnySparseTensor:$source);
  let results = (outs AnySparseTensor:$dest);
  let assemblyFormat = "$source attr-dict `:` type($source) `to` type($dest)";

  let builders = [
    OpBuilder<(ins "SparseTensorEncodingAttr":$dstEnc, "Value":$source)>
  ];

  let hasFolder = 1;
  let hasVerifier = 1;
}

def SparseTensor_ToPositionsOp : SparseTensor_Op<"positions",
      [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
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

  let arguments = (ins AnySparseTensor:$tensor, LevelAttr:$level);
  let results = (outs AnyNon0RankedMemRef:$result);
  let assemblyFormat = "$tensor attr-dict `:` type($tensor) `to` type($result)";

  let hasVerifier = 1;
}

def SparseTensor_ToCoordinatesOp : SparseTensor_Op<"coordinates",
      [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
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

  let arguments = (ins AnySparseTensor:$tensor, LevelAttr:$level);
  let results = (outs AnyNon0RankedMemRef:$result);
  let assemblyFormat = "$tensor attr-dict `:` type($tensor) `to` type($result)";

  let hasVerifier = 1;
}

def SparseTensor_ToCoordinatesBufferOp : SparseTensor_Op<"coordinates_buffer",
      [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
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

  let arguments = (ins AnySparseTensor:$tensor);
  let results = (outs AnyNon0RankedMemRef:$result);
  let assemblyFormat = "$tensor attr-dict `:` type($tensor) `to` type($result)";

  let hasVerifier = 1;
}

def SparseTensor_ToValuesOp : SparseTensor_Op<"values",
      [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
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

  let arguments = (ins AnySparseTensor:$tensor);
  let results = (outs AnyNon0RankedMemRef:$result);
  let assemblyFormat = "$tensor attr-dict `:` type($tensor) `to` type($result)";

  let hasVerifier = 1;
}

def SparseTensor_NumberOfEntriesOp : SparseTensor_Op<"number_of_entries", [Pure]> {
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

  let arguments = (ins AnySparseTensor:$tensor);
  let results = (outs Index:$result);
  let assemblyFormat = "$tensor attr-dict `:` type($tensor)";
}

def SparseTensor_ConcatenateOp : SparseTensor_Op<"concatenate",
      [Pure, StageWithSortSparseOpInterface]> {
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

  let arguments = (ins Variadic<AnyRankedTensor>:$inputs, DimensionAttr:$dimension);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$inputs attr-dict `:` type($inputs) `to` type($result)";

  let hasVerifier = 1;
}

def SparseTensor_ToSliceOffsetOp : SparseTensor_Op<"slice.offset", [Pure]> {
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

  let arguments = (ins AnySparseTensorSlice:$slice, IndexAttr:$dim);
  let results = (outs Index:$offset);
  let assemblyFormat = "$slice `at` $dim attr-dict `:` type($slice)";

  let hasVerifier = 1;
}

def SparseTensor_ToSliceStrideOp : SparseTensor_Op<"slice.stride", [Pure]> {
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

  let arguments = (ins AnySparseTensorSlice:$slice, IndexAttr:$dim);
  let results = (outs Index:$stride);
  let assemblyFormat = "$slice `at` $dim attr-dict `:` type($slice)";

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Sparse Tensor Storage Specifier Operations.
//===----------------------------------------------------------------------===//

def SparseTensor_StorageSpecifierInitOp : SparseTensor_Op<"storage_specifier.init",
      [Pure]> {
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

  let arguments = (ins Optional<SparseTensorStorageSpecifier>:$source);
  let results = (outs SparseTensorStorageSpecifier:$result);
  let assemblyFormat = "attr-dict (`with` $source^)? `:` (`from` qualified(type($source))^ `to`)?"
                                                        " qualified(type($result))";
  let builders = [
    OpBuilder<(ins "Type":$result),
    [{
      build($_builder, $_state, result, Value());
    }]>
  ];


}

def SparseTensor_GetStorageSpecifierOp : SparseTensor_Op<"storage_specifier.get", [Pure]> {
  let summary = "";
  let description = [{
    Returns the requested field of the given storage_specifier.

    Example of querying the size of the coordinates array for level 0:

    ```mlir
    %0 = sparse_tensor.storage_specifier.get %arg0 crd_mem_sz at 0
         : !sparse_tensor.storage_specifier<#COO>
    ```
  }];

  let arguments = (ins SparseTensorStorageSpecifier:$specifier,
                   SparseTensorStorageSpecifierKindAttr:$specifierKind,
                   OptionalAttr<LevelAttr>:$level);
  let results = (outs Index:$result);
  let assemblyFormat = "$specifier $specifierKind (`at` $level^)? attr-dict"
                       "`:` qualified(type($specifier))";

  let hasVerifier = 1;
  let hasFolder = 1;
}

def SparseTensor_SetStorageSpecifierOp : SparseTensor_Op<"storage_specifier.set",
    [Pure, AllTypesMatch<["result", "specifier"]>]> {
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

  let arguments = (ins SparseTensorStorageSpecifier:$specifier,
                   SparseTensorStorageSpecifierKindAttr:$specifierKind,
                   OptionalAttr<LevelAttr>:$level,
                   Index:$value);
  let results = (outs SparseTensorStorageSpecifier:$result);
  let assemblyFormat = "$specifier $specifierKind (`at` $level^)? `with` $value"
                       " attr-dict `:` qualified(type($result))";

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Sparse Tensor Coordinate Operations.
//===----------------------------------------------------------------------===//

def SparseTensor_LvlOp : SparseTensor_Op<"lvl", [ConditionallySpeculatable, NoMemoryEffect]> {
  let summary = "level index operation";
  let description = [{
    The `sparse_tensor.lvl` behaves similar to `tensor.dim` operation.
    It takes a sparse tensor and a level operand of type `index` and returns
    the size of the requested level of the given sparse tensor.
    If the sparse tensor has an identity dimension to level mapping, it returns
    the same result as `tensor.dim`.
    If the level index is out of bounds, the behavior is undefined.

    Example:

    ```mlir
    #BSR = #sparse_tensor.encoding<{
      map = ( i, j ) ->
        ( i floordiv 2 : dense,
          j floordiv 3 : compressed,
          i mod 2      : dense,
          j mod 3      : dense
        )
    }>

    // Always returns 2 (4 floordiv 2), can be constant folded:
    %c0 = arith.constant 0 : index
    %x = sparse_tensor.lvl %A, %c0 : tensor<4x?xf32, #BSR>

    // Return the dynamic dimension of %A computed by %j mod 3.
    %c1 = arith.constant 1 : index
    %y = sparse_tensor.lvl %A, %c1 : tensor<4x?xf32, #BSR>

    // Always return 3 (since j mod 3 < 3), can be constant fold
    %c3 = arith.constant 3 : index
    %y = sparse_tensor.lvl %A, %c3 : tensor<4x?xf32, #BSR>
    ```
  }];

  let arguments = (ins AnySparseTensor:$source, Index:$index);
  let results = (outs Index:$result);
  let assemblyFormat = "attr-dict $source `,` $index `:` type($source) ";

  let builders = [
    OpBuilder<(ins "Value":$source, "int64_t":$index)>
  ];

  let extraClassDeclaration = [{
    /// Helper function to get the index as a simple integer if it is constant.
    std::optional<uint64_t> getConstantLvlIndex();

    /// Interface method for ConditionallySpeculatable.
    Speculation::Speculatability getSpeculatability();
  }];

  let hasVerifier = 1;
  let hasFolder = 1;
}

def SparseTensor_CrdTranslateOp : SparseTensor_Op<"crd_translate", [Pure]> {
  string summary = "Performs coordinate translation between level and dimension coordinate space.";
  string description = [{
    Performs coordinate translation between level and dimension coordinate space according
    to the affine maps defined by $encoder.

    Example:

    ```mlir
    %l0, %l1, %l2, %l3 = sparse_tensor.crd_translate dim_to_lvl [%d0, %d1] as #BSR
                       : index, index, index, index
    ```
  }];

  let arguments = (ins Variadic<Index>:$in_crds,
                   SparseTensorCrdTransDirectionAttr:$direction,
                   SparseTensorEncodingAttr:$encoder);
  let results = (outs Variadic<Index>:$out_crds);
  let assemblyFormat = "$direction `[` $in_crds `]` `as` $encoder attr-dict `:` type($out_crds)";

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

def SparseTensor_PushBackOp : SparseTensor_Op<"push_back",
    [TypesMatchWith<"value type matches element type of inBuffer",
                    "inBuffer", "value",
                    "::llvm::cast<ShapedType>($_self).getElementType()">,
     AllTypesMatch<["inBuffer", "outBuffer"]>]> {
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

  let arguments = (ins Index:$curSize,
                       StridedMemRefRankOf<[AnyType], [1]>:$inBuffer,
                       AnyType:$value, Optional<Index>:$n,
                       UnitAttr:$inbounds);
  let results = (outs StridedMemRefRankOf<[AnyType], [1]>:$outBuffer,
                      Index:$newSize);
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

def SparseTensor_ExpandOp : SparseTensor_Op<"expand", []> {
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


  let arguments = (ins AnySparseTensor:$tensor);
  let results = (outs AnyStridedMemRefOfRank<1>:$values,
                      StridedMemRefRankOf<[I1],[1]>:$filled,
                      StridedMemRefRankOf<[Index],[1]>:$added,
                      Index:$count);
  let assemblyFormat = "$tensor attr-dict `:` type($tensor) `to` type($values)"
                       " `,` type($filled) `,` type($added)";
}

def SparseTensor_CompressOp : SparseTensor_Op<"compress",
    [AllTypesMatch<["tensor", "result"]>]> {
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

  let arguments = (ins AnyStridedMemRefOfRank<1>:$values,
                   StridedMemRefRankOf<[I1],[1]>:$filled,
                   StridedMemRefRankOf<[Index],[1]>:$added,
                   Index:$count,
                   AnySparseTensor:$tensor,
                   Variadic<Index>:$lvlCoords);
  let results = (outs AnySparseTensor:$result);
  let assemblyFormat = "$values `,` $filled `,` $added `,` $count"
                       " `into` $tensor `[` $lvlCoords `]` attr-dict"
                       " `:` type($values) `,` type($filled) `,` type($added)"
                       " `,` type($tensor)";
  let hasVerifier = 1;
}

def SparseTensor_LoadOp : SparseTensor_Op<"load", [SameOperandsAndResultType]> {
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

  let arguments = (ins AnySparseTensor:$tensor, UnitAttr:$hasInserts);
  let results = (outs AnyTensor:$result);
  let assemblyFormat = "$tensor (`hasInserts` $hasInserts^)? attr-dict `:` type($tensor)";
}

def SparseTensor_OutOp : SparseTensor_Op<"out", []> {
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

  let arguments = (ins AnySparseTensor:$tensor, AnyType:$dest);
  let assemblyFormat = "$tensor `,` $dest attr-dict `:` type($tensor) `,` type($dest)";
}

//===----------------------------------------------------------------------===//
// Sparse Tensor Sorting/Ordering Operations.
//===----------------------------------------------------------------------===//

def SparseTensor_SortOp : SparseTensor_Op<"sort"> {
  let summary = "Sorts the arrays in xs and ys lexicographically on the "
                "integral values found in the xs list";
  let description = [{
    Sorts the `xs` values along with some `ys` values that are put in a single linear
    buffer `xy`.  The affine map attribute `perm_map` specifies the permutation to be
    applied on the `xs` before comparison, the rank of the permutation map
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

  let arguments = (ins Index:$n,
                       StridedMemRefRankOf<[AnyInteger, Index], [1]>:$xy,
                       Variadic<StridedMemRefRankOf<[AnyType], [1]>>:$ys,
                       AffineMapAttr:$perm_map, OptionalAttr<IndexAttr>:$ny,
                       SparseTensorSortKindAttr:$algorithm);
  let assemblyFormat = "$algorithm $n"
                       "`,`$xy (`jointly` $ys^)? attr-dict"
                       "`:` type($xy) (`jointly` type($ys)^)?";
  let hasVerifier = 1;
}

def SparseTensor_ReorderCOOOp : SparseTensor_Op<"reorder_coo", [Pure]> {
  let summary = "Reorder the input COO such that it has the the same order as "
                "the output COO";
  let description = [{
    Reorders the input COO to the same order as specified by the output format.
    E.g., reorder an unordered COO into an ordered one.

    The input and result COO tensor must have the same element type, position type and
    coordinate type. At the moment, the operation also only supports ordering
    input and result COO with the same dim2lvl map.

    Example:

    ```mlir
    %res = sparse_tensor.reorder_coo quick_sort %coo : tensor<?x?xf64 : #Unordered_COO> to
                                                       tensor<?x?xf64 : #Ordered_COO>

    ```
  }];

  let arguments = (ins AnySparseTensor: $input_coo,
                       SparseTensorSortKindAttr:$algorithm);
  let results = (outs AnySparseTensor: $result_coo);
  let assemblyFormat = "$algorithm $input_coo attr-dict"
                       "`:` type($input_coo) `to` type($result_coo)";

  let hasFolder = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Sparse Tensor Syntax Operations.
//===----------------------------------------------------------------------===//

def SparseTensor_BinaryOp : SparseTensor_Op<"binary", [Pure]> {
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
      %C = tensor.empty(...)
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
      %C = tensor.empty(...)
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
      %C = tensor.empty(...)
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
  let arguments = (ins AnyType:$x, AnyType:$y, UnitAttr:$left_identity, UnitAttr:$right_identity);
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
        $x `,` $y `:` attr-dict type($x) `,` type($y) `to` type($output) `\n`
        `overlap` `=` $overlapRegion `\n`
        `left` `=` (`identity` $left_identity^):($leftRegion)? `\n`
        `right` `=` (`identity` $right_identity^):($rightRegion)?
  }];

  let hasVerifier = 1;
}

def SparseTensor_UnaryOp : SparseTensor_Op<"unary", [Pure]> {

  let arguments = (ins AnyType:$x);

  let results = (outs AnyType:$output);

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
      argument, while the missing region's block has zero arguments. The
      absent region may only generate constants or values already computed
      on entry of the `linalg.generic` operation.

      A region may also be declared empty (i.e. `absent={}`), indicating that the
      region does not contribute to the output.

      Due to the possibility of empty regions, i.e. lack of a value for certain
      cases, the result of this operation may only feed directly into the output
      of the `linalg.generic` operation or into into a custom reduction
      `sparse_tensor.reduce` operation that follows in the same region.

      Example of A+1, restricted to existing elements:

      ```mlir
      %C = tensor.empty(...) : tensor<?xf64, #SparseVector>
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
      %p1 = arith.constant  1 : i32
      %m1 = arith.constant -1 : i32
      %C = tensor.empty(...) : tensor<?xi32, #SparseVector>
      %1 = linalg.generic #trait
         ins(%A: tensor<?xf64, #SparseVector>)
        outs(%C: tensor<?xi32, #SparseVector>) {
        ^bb0(%a: f64, %c: i32) :
          %result = sparse_tensor.unary %a : f64 to i32
            present={
            ^bb0(%x: f64):
              sparse_tensor.yield %p1 : i32
            }
            absent={
              sparse_tensor.yield %m1 : i32
            }
          linalg.yield %result : i32
      } -> tensor<?xi32, #SparseVector>
      ```

      Example showing a structural inversion (existing values become missing in
      the output, while missing values are filled with 1):

      ```mlir
      %c1 = arith.constant 1 : i64
      %C = tensor.empty(...) : tensor<?xi64, #SparseVector>
      %2 = linalg.generic #trait
         ins(%A: tensor<?xf64, #SparseVector>)
        outs(%C: tensor<?xi64, #SparseVector>) {
        ^bb0(%a: f64, %c: i64) :
          %result = sparse_tensor.unary %a : f64 to i64
            present={}
            absent={
              sparse_tensor.yield %c1 : i64
            }
          linalg.yield %result : i64
      } -> tensor<?xi64, #SparseVector>
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

def SparseTensor_ReduceOp : SparseTensor_Op<"reduce", [Pure, SameOperandsAndResultType]> {
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
      %C = tensor.empty(...)
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
  let arguments = (ins AnyType:$x, AnyType:$y, AnyType:$identity);
  let results = (outs AnyType:$output);
  let assemblyFormat = "$x `,` $y `,` $identity attr-dict `:` type($output) $region";

  let hasVerifier = 1;
}

def SparseTensor_SelectOp : SparseTensor_Op<"select", [Pure, SameOperandsAndResultType]> {
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
      %C = tensor.empty(...)
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
      %C = tensor.empty(...)
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
  let arguments = (ins AnyType:$x);
  let results = (outs AnyType:$output);
  let assemblyFormat = "$x attr-dict `:` type($x) $region";

  let hasVerifier = 1;
}

def SparseTensor_YieldOp : SparseTensor_Op<"yield", [Pure, Terminator,
    ParentOneOf<["BinaryOp", "UnaryOp", "ReduceOp", "SelectOp",
                 "ForeachOp", "IterateOp", "CoIterateOp"]>]> {
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
      build($_builder, $_state, ValueRange());
    }]>,
    OpBuilder<(ins "Value":$yieldVal),
    [{
      build($_builder, $_state, ValueRange(yieldVal));
    }]>
  ];

  let extraClassDeclaration = [{
     Value getSingleResult() {
        assert(hasSingleResult());
        return getResults().front();
     }
     bool hasSingleResult() {
        return getResults().size() == 1;
     }
  }];

  let arguments = (ins Variadic<AnyType>:$results);
  let assemblyFormat = "$results attr-dict `:` type($results)";
}

def SparseTensor_ForeachOp : SparseTensor_Op<"foreach",
    [SingleBlockImplicitTerminator<"YieldOp">]> {
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
  let arguments = (ins AnyTensor:$tensor,
                       Variadic<AnyType>:$initArgs,
                       OptionalAttr<AffineMapAttr>:$order);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "`in` $tensor (`init``(`$initArgs^`)`)? attr-dict"
                       "    `:` type($tensor) (`,` type($initArgs)^)?"
                       "  (`->` type($results)^)?  `do` $region";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Sparse Tensor Iteration Operations.
//===----------------------------------------------------------------------===//

def ExtractIterSpaceOp : SparseTensor_Op<"extract_iteration_space",
    [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Extracts an iteration space from a sparse tensor between certain levels";
  let description = [{
      Extracts a `!sparse_tensor.iter_space` from a sparse tensor between
      certain (consecutive) levels. For sparse levels, it is usually done by
      loading a postion range from the underlying sparse tensor storage.
      E.g., for a compressed level, the iteration space is extracted by
      [pos[i], pos[i+1]) supposing the the parent iterator points at `i`.

      `tensor`: the input sparse tensor that defines the iteration space.
      `parentIter`: the iterator for the previous level, at which the iteration space
      at the current levels will be extracted.
      `loLvl`, `hiLvl`: the level range between [loLvl, hiLvl) in the input tensor that
      the returned iteration space covers. `hiLvl - loLvl` defines the dimension of the
      iteration space.

      The type of returned the value is must be
      `!sparse_tensor.iter_space<#INPUT_ENCODING, lvls = $loLvl to $hiLvl>`.
      The returned iteration space can then be iterated over by
      `sparse_tensor.iterate` operations to visit every stored element
      (usually nonzeros) in the input sparse tensor.

      Example:
      ```mlir
      // Extracts a 1-D iteration space from a COO tensor at level 1.
      %space = sparse_tensor.iteration.extract_space %sp at %it1 lvls = 1
        : tensor<4x8xf32, #COO>, !sparse_tensor.iterator<#COO, lvls = 0>
       ->!sparse_tensor.iter_space<#COO, lvls = 1>
      ```
  }];

  let arguments = (ins AnySparseTensor:$tensor,
                       Optional<AnySparseIterator>:$parentIter,
                       LevelAttr:$loLvl, LevelAttr:$hiLvl);
  let results = (outs AnySparseIterSpace:$extractedSpace);

  let extraClassDeclaration = [{
    std::pair<Level, Level> getLvlRange() {
      return std::make_pair(getLoLvl(), getHiLvl());
    }
    unsigned getSpaceDim() {
      return getHiLvl() - getLoLvl();
    }
    ArrayRef<::mlir::sparse_tensor::LevelType> getSpaceLvlTypes() {
      return getExtractedSpace().getType().getLvlTypes();
    }
  }];

  let builders = [
    // Construct a 1-D iteration space.
    OpBuilder<(ins "Value":$tensor, "Value":$parentIter,
                   "sparse_tensor::Level":$loLvl),
    [{
      build($_builder, $_state, tensor, parentIter, loLvl, loLvl + 1);
    }]>,
    // Construct a 1-D root iteration space
    OpBuilder<(ins "Value":$tensor),
    [{
      build($_builder, $_state, tensor, nullptr, 0);
    }]>
  ];

  let assemblyFormat = "$tensor (`at` $parentIter^)? `lvls` `=` custom<LevelRange>($loLvl, $hiLvl) "
                       " attr-dict `:` type($tensor) (`,` type($parentIter)^)? "
                       "`->` qualified(type($extractedSpace))";

  let hasVerifier = 1;
}

def ExtractValOp : SparseTensor_Op<"extract_value", [
    Pure,
    TypesMatchWith<"result type matches element type of tensor",
                   "tensor", "result",
                   "::llvm::cast<TensorType>($_self).getElementType()">]> {
  let summary = "Extracts a value from a sparse tensor using an iterator.";
  let description = [{
      The `sparse_tensor.extract_value` operation extracts the value
      pointed to by a sparse iterator from a sparse tensor.

      Example:

      ```mlir
      %val = sparse_tensor.extract_value %sp at %it
           : tensor<?x?xf32, #CSR>, !sparse_tensor.iterator<#CSR, lvl = 1>
      ```
  }];

  let arguments = (ins AnySparseTensor:$tensor, AnySparseIterator:$iterator);
  let results = (outs AnyType:$result);

  let assemblyFormat = "$tensor `at` $iterator attr-dict `:` type($tensor)`,` qualified(type($iterator))";
  let hasVerifier = 1;
}

def IterateOp : SparseTensor_Op<"iterate",
    [RecursiveMemoryEffects, RecursivelySpeculatable,
     DeclareOpInterfaceMethods<LoopLikeOpInterface,
      ["getInitsMutable", "getLoopResults", "getRegionIterArgs",
       "getYieldedValuesMutable"]>,
     DeclareOpInterfaceMethods<RegionBranchOpInterface,
      ["getEntrySuccessorOperands"]>,
     SingleBlockImplicitTerminator<"sparse_tensor::YieldOp">]> {

  let summary = "Iterates over a sparse iteration space";
  let description = [{
      The `sparse_tensor.iterate` operation represents a loop (nest) over
      the provided iteration space extracted from a specific sparse tensor.
      The operation defines an SSA value for a sparse iterator that points
      to the current stored element in the sparse tensor and SSA values
      for coordinates of the stored element. The coordinates are always
      converted to `index` type despite of the underlying sparse tensor
      storage. When coordinates are not used, the SSA values can be skipped
      by `_` symbols, which usually leads to simpler generated code after
      sparsification. For example:

      ```mlir
      // The coordinate for level 0 is not used when iterating over a 2-D
      // iteration space.
      %sparse_tensor.iterate %iterator in %space at(_, %crd_1)
        : !sparse_tensor.iter_space<#CSR, lvls = 0 to 2>
      ```

      `sparse_tensor.iterate` can also operate on loop-carried variables.
      It returns the final values after loop termination.
      The initial values of the variables are passed as additional SSA operands
      to the iterator SSA value and used coordinate SSA values mentioned
      above. The operation region has an argument for the iterator, variadic
      arguments for specified (used) coordiates and followed by one argument
      for each loop-carried variable, representing the value of the variable
      at the current iteration.
      The body region must contain exactly one block that terminates with
      `sparse_tensor.yield`.

      The results of an `sparse_tensor.iterate` hold the final values after
      the last iteration. If the `sparse_tensor.iterate` defines any values,
      a yield must be explicitly present.
      The number and types of the `sparse_tensor.iterate` results must match
      the initial values in the iter_args binding and the yield operands.


      A nested `sparse_tensor.iterate` example that prints all the coordinates
      stored in the sparse input:

      ```mlir
      func.func @nested_iterate(%sp : tensor<4x8xf32, #COO>) {
        // Iterates over the first level of %sp
        %l1 = sparse_tensor.extract_iteration_space %sp lvls = 0
            : tensor<4x8xf32, #COO> -> !sparse_tensor.iter_space<#COO, lvls = 0 to 1>
        %r1 = sparse_tensor.iterate %it1 in %l1 at (%coord0)
            : !sparse_tensor.iter_space<#COO, lvls = 0 to 1>  {
          // Iterates over the second level of %sp
          %l2 = sparse_tensor.extract_iteration_space %sp at %it1 lvls = 1
              : tensor<4x8xf32, #COO>, !sparse_tensor.iterator<#COO, lvls = 0 to 1>
             -> !sparse_tensor.iter_space<#COO, lvls = 1 to 2>
          %r2 = sparse_tensor.iterate %it2 in %l2 at (coord1)
              : !sparse_tensor.iter_space<#COO, lvls = 1 to 2>  {
             vector.print %coord0 : index
             vector.print %coord1 : index
          }
        }
      }

      ```
  }];

  let arguments = (ins AnySparseIterSpace:$iterSpace,
                       Variadic<AnyType>:$initArgs,
                       I64BitSetAttr:$crdUsedLvls);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$region);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "Value":$iterSpace, "ValueRange":$initArgs)>,
    OpBuilder<(ins "Value":$iterSpace, "ValueRange":$initArgs, "I64BitSet" :$crdUsedLvls)>
  ];

  let extraClassDeclaration = [{
    unsigned getSpaceDim() {
      return getIterSpace().getType().getSpaceDim();
    }
    BlockArgument getIterator() {
      return getRegion().getArguments().back();
    }
    std::optional<BlockArgument> getLvlCrd(Level lvl) {
      if (getCrdUsedLvls()[lvl]) {
        uint64_t mask = (static_cast<uint64_t>(0x01u) << lvl) - 1;
        return getCrds()[llvm::popcount(mask & getCrdUsedLvls())];
      }
      return std::nullopt;
    }
    Block::BlockArgListType getCrds() {
      // User-provided iteration arguments -> coords -> iterator.
      return getRegion().getArguments().slice(getNumRegionIterArgs(), getCrdUsedLvls().count());
    }
    unsigned getNumRegionIterArgs() {
      return getRegion().getArguments().size() - 1 - getCrdUsedLvls().count();
    }
  }];

  let hasVerifier = 1;
  let hasRegionVerifier = 1;
  let hasCanonicalizer = 1;
  let hasCustomAssemblyFormat = 1;
}

def SparseTensor_CoIterateOp : SparseTensor_Op<"coiterate",
    [AttrSizedOperandSegments,
     SingleBlockImplicitTerminator<"sparse_tensor::YieldOp">,
     RecursiveMemoryEffects]> {
  let summary = "Co-iterates over a set of sparse iteration spaces";
  let description = [{
      The `sparse_tensor.coiterate` operation represents a loop (nest) over
      a set of iteration spaces. The operation can have multiple regions,
      with each of them defining a case to compute a result at the current iterations.
      The case condition is defined solely based on the pattern of specified iterators.
      For example:
      ```mlir
      %ret = sparse_tensor.coiterate (%sp1, %sp2) at(%coord) iter_args(%arg = %init)
           : (!sparse_tensor.iter_space<#CSR, lvls = 0>,
              !sparse_tensor.iter_space<#COO, lvls = 0>)
           -> index
      case %it1, _ {
        // %coord is specifed in space %sp1 but *NOT* specified in space %sp2.
      }
      case %it1, %it2 {
        // %coord is specifed in *BOTH* spaces %sp1 and %sp2.
      }
      ```

      `sparse_tensor.coiterate` can also operate on loop-carried variables.
      It returns the final value for each loop-carried variable after loop termination.
      The initial values of the variables are passed as additional SSA operands
      to the iterator SSA value and used coordinate SSA values.
      Each operation region has variadic arguments for specified (used), one argument
      for each loop-carried variable, representing the value of the variable
      at the current iteration, followed by a list of arguments for iterators.
      The body region must contain exactly one block that terminates with
      `sparse_tensor.yield`.

      The results of an `sparse_tensor.coiterate` hold the final values after
      the last iteration. If the `sparse_tensor.coiterate` defines any values,
      a yield must be explicitly present in every region defined in the operation.
      The number and types of the `sparse_tensor.coiterate` results must match
      the initial values in the iter_args binding and the yield operands.


      A `sparse_tensor.coiterate` example that does elementwise addition between two
      sparse vectors.


      ```mlir
      %ret = sparse_tensor.coiterate (%sp1, %sp2) at(%coord) iter_args(%arg = %init)
           : (!sparse_tensor.iter_space<#CSR, lvls = 0>,
              !sparse_tensor.iter_space<#CSR, lvls = 0>)
           -> tensor<?xindex, #CSR>
      case %it1, _ {
         // v = v1 + 0 = v1
         %v1 = sparse_tensor.extract_value %t1 at %it1 : index
         %yield = sparse_tensor.insert %v1 into %arg[%coord]
         sparse_tensor.yield %yield
      }
      case _, %it2 {
         // v = v2 + 0 = v2
         %v2 = sparse_tensor.extract_value %t2 at %it2 : index
         %yield = sparse_tensor.insert %v1 into %arg[%coord]
         sparse_tensor.yield %yield
      }
      case %it1, %it2 {
         // v = v1 + v2
         %v1 = sparse_tensor.extract_value %t1 at %it1 : index
         %v2 = sparse_tensor.extract_value %t2 at %it2 : index
         %v = arith.addi %v1, %v2 : index
         %yield = sparse_tensor.insert %v into %arg[%coord]
         sparse_tensor.yield %yield
      }
      ```
  }];

  let arguments = (ins Variadic<AnySparseIterSpace>:$iterSpaces,
                       Variadic<AnyType>:$initArgs,
                       I64BitSetAttr:$crdUsedLvls,
                       I64BitSetArrayAttr:$cases);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region VariadicRegion<SizedRegion<1>>:$caseRegions);

  let builders = [
    OpBuilder<(ins "ValueRange":$iterSpace, "ValueRange":$initArgs, "unsigned":$numCases)>,
  ];

  let extraClassDeclaration = [{
    unsigned getSpaceDim() {
      return llvm::cast<::mlir::sparse_tensor::IterSpaceType>(
                 getIterSpaces().front().getType())
          .getSpaceDim();
    }
    I64BitSet getRegionDefinedSpace(unsigned regionIdx) {
      return I64BitSet(llvm::cast<IntegerAttr>(getCases()[regionIdx])
                           .getValue().getZExtValue());
    }
    auto getRegionDefinedSpaces() {
      return llvm::map_range(getCases().getValue(), [](Attribute attr) {
        return I64BitSet(llvm::cast<IntegerAttr>(attr).getValue().getZExtValue());
      });
    }

    // The block arguments starts with user-provided iteration arguments,
    // follows by referenced coordinates and ends with iterators.
    Block::BlockArgListType getCrds(unsigned regionIdx) {
      return getRegion(regionIdx).getArguments()
          .slice(getNumRegionIterArgs(), getCrdUsedLvls().count());
    }
    unsigned getNumRegionIterArgs() {
      return getInitArgs().size();
    }
    Block::BlockArgListType getRegionIterArgs(unsigned regionIdx) {
      return getRegion(regionIdx).getArguments()
          .take_front(getNumRegionIterArgs());
    }
    Block::BlockArgListType getRegionIterators(unsigned regionIdx) {
      return getRegion(regionIdx).getArguments()
          .take_back(getRegionDefinedSpace(regionIdx).count());
    }
    ValueRange getYieldedValues(unsigned regionIdx);

    // Returns a vector of regions that are the `sub-cases` of the given case region.
    // E.g., `case %it1, _, %it3` is a subcase of `case %it1, %it2, %it3`.
    SmallVector<Region *> getSubCasesOf(unsigned regionIdx);
  }];

  let hasVerifier = 1;
  let hasRegionVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// Sparse Tensor Debugging and Test-Only Operations.
//===----------------------------------------------------------------------===//

def SparseTensor_PrintOp : SparseTensor_Op<"print"> {
  string summary = "Prints a sparse tensor (for testing and debugging)";
  string description = [{
    Prints the individual components of a sparse tensors (the positions,
    coordinates, and values components) to stdout for testing and debugging
    purposes. This operation lowers to just a few primitives in a light-weight
    runtime support to simplify supporting this operation on new platforms.

    Example:

    ```mlir
    sparse_tensor.print %tensor : tensor<1024x1024xf64, #CSR>
    ```
  }];

  let arguments = (ins AnySparseTensor:$tensor);
  let assemblyFormat = "$tensor attr-dict `:` type($tensor)";
}

def SparseTensor_HasRuntimeLibraryOp
    : SparseTensor_Op<"has_runtime_library", []>, Results<(outs I1:$result)> {
  string summary = "Indicates whether running in runtime/codegen mode";
  string description = [{
    Returns a boolean value that indicates whether the sparsifier runs in
    runtime library mode or not. For testing only! This operation is useful
    for writing test cases that require different code depending on
    runtime/codegen mode.

    Example:

    ```mlir
    %has_runtime = sparse_tensor.has_runtime_library
    scf.if %has_runtime {
      ...
    }
    ```
  }];
  let assemblyFormat = "attr-dict";
}

#endif // SPARSETENSOR_OPS
