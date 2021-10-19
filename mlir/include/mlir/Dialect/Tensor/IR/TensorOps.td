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
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"

class Tensor_Op<string mnemonic, list<OpTrait> traits = []>
    : Op<Tensor_Dialect, mnemonic, traits> {
  let printer = [{ return ::print(p, *this); }];
  let verifier = [{ return ::verify(*this); }];
  let parser = [{ return ::parse$cppClass(parser, result); }];
}

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//

def Tensor_CastOp : Tensor_Op<"cast", [
    DeclareOpInterfaceMethods<CastOpInterface>, NoSideEffect
  ]> {
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

  let hasCanonicalizer = 1;
  let verifier = ?;
}

//===----------------------------------------------------------------------===//
// DimOp
//===----------------------------------------------------------------------===//

def Tensor_DimOp : Tensor_Op<"dim", [NoSideEffect]> {
  let summary = "dimension index operation";
  let description = [{
    The `dim` operation takes a tensor and a dimension operand of type `index`.
    It returns the size of the requested dimension of the given tensor.
    If the dimension index is out of bounds, the behavior is undefined.

    The specified tensor type is that of the first operand.

    Example:

    ```mlir
    // Always returns 4, can be constant folded:
    %c0 = arith.constant 0 : index
    %x = tensor.dim %A, %c0 : tensor<4x?xf32>

    // Returns the dynamic dimension of %A.
    %c1 = arith.constant 1 : index
    %y = tensor.dim %A, %c1 : memref<4x?xf32>

    // Equivalent generic form:
    %x = "tensor.dim"(%A, %c0) : (memref<4x?xf32>, index) -> index
    %y = "tensor.dim"(%A, %c1) : (memref<4x?xf32>, index) -> index
    ```
  }];

  let arguments = (ins AnyTensor:$source,
                       Index:$index);
  let results = (outs Index:$result);

  let assemblyFormat = [{
    attr-dict $source `,` $index `:` type($source)
  }];

  let builders = [
    OpBuilder<(ins "Value":$source, "int64_t":$index)>
  ];

  let extraClassDeclaration = [{
    /// Helper function to get the index as a simple integer if it is constant.
    Optional<int64_t> getConstantIndex();
  }];

  let hasCanonicalizer = 1;
  let hasFolder = 1;
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
    OpBuilder<(ins "Value":$tensor, CArg<"ValueRange", "{}">:$indices), [{
      auto resType = tensor.getType().cast<ShapedType>().getElementType();
      build($_builder, $_state, resType, tensor, indices);
    }]>];

  let hasFolder = 1;
}


//===----------------------------------------------------------------------===//
// ExtractSliceOp
//===----------------------------------------------------------------------===//

def Tensor_ExtractSliceOp : BaseOpWithOffsetSizesAndStrides<
    Tensor_Dialect, "extract_slice",
    [NoSideEffect, AttrSizedOperandSegments,
     DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>,
     OffsetSizeAndStrideOpInterface]> {
  let summary = "extract slice operation";
  let description = [{
    The "extract_slice" operation extract a tensor from another tensor as
    specified by the operation's offsets, sizes and strides arguments.

    The extract_slice operation supports the following arguments:

    * source: the "base" tensor from which to extract a slice.
    * offsets: tensor-rank number of offsets into the "base" tensor from which
               to extract the slice.
    * sizes: tensor-rank number of sizes which specify the sizes of the result
             tensor type.
    * strides: tensor-rank number of strides specifying subsampling in each
               dimension.

    The representation based on offsets, sizes and strides support a
    partially-static specification via attributes specified through the
    `static_offsets`, `static_sizes` and `static_strides` arguments. A special
    sentinel value ShapedType::kDynamicSize and
    ShapedType::kDynamicStrideOrOffset encodes that the corresponding entry has
    a dynamic value.

    After buffer-allocation, the "extract_slice" op is expected to lower into a
    "subview" op.

    An extract_slice operation may additionally reduce the rank of the resulting
    tensor by removing dimensions that are statically known to be of size 1.

    Example:

    ```
    // Rank-reducing extract_slice.
    %1 = tensor.extract_slice %0[0, 0, 0][1, 16, 4][1, 1, 1] :
      tensor<8x16x4xf32> to tensor<16x4xf32>
    %3 = tensor.extract_slice %2[3, 4, 2][1, 6, 3][1, 1, 1] :
      tensor<8x16x4xf32> to tensor<6x3xf32>
    ```
  }];

  let arguments = (ins
    AnyRankedTensor:$source,
    Variadic<Index>:$offsets,
    Variadic<Index>:$sizes,
    Variadic<Index>:$strides,
    I64ArrayAttr:$static_offsets,
    I64ArrayAttr:$static_sizes,
    I64ArrayAttr:$static_strides
  );
  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $source ``
    custom<OperandsOrIntegersOffsetsOrStridesList>($offsets, $static_offsets)
    custom<OperandsOrIntegersSizesList>($sizes, $static_sizes)
    custom<OperandsOrIntegersOffsetsOrStridesList>($strides, $static_strides)
    attr-dict `:` type($source) `to` type($result)
  }];

  let builders = [
    // Build an ExtractSliceOp with mixed static and dynamic entries and
    // inferred result type.
    OpBuilder<(ins "Value":$source, "ArrayRef<OpFoldResult>":$offsets,
      "ArrayRef<OpFoldResult>":$sizes, "ArrayRef<OpFoldResult>":$strides,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
    // Build an ExtractSliceOp with mixed static and dynamic entries and custom
    // result type. If the type passed is nullptr, it is inferred.
    OpBuilder<(ins "RankedTensorType":$resultType, "Value":$source,
      "ArrayRef<OpFoldResult>":$offsets, "ArrayRef<OpFoldResult>":$sizes,
      "ArrayRef<OpFoldResult>":$strides,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
    // Build an ExtractSliceOp with dynamic entries and custom result type. If
    // the type passed is nullptr, it is inferred.
    OpBuilder<(ins "Value":$source, "ValueRange":$offsets,
      "ValueRange":$sizes, "ValueRange":$strides,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
    // Build an ExtractSliceOp with dynamic entries and inferred result type.
    OpBuilder<(ins "RankedTensorType":$resultType, "Value":$source,
      "ValueRange":$offsets, "ValueRange":$sizes, "ValueRange":$strides,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>
  ];

  let extraClassDeclaration = extraBaseClassDeclaration # [{
    /// Returns the type of the base tensor operand.
    RankedTensorType getSourceType() {
      return source().getType().cast<RankedTensorType>();
    }

    /// The result of an extract_slice is always a tensor.
    RankedTensorType getType() {
      return getResult().getType().cast<RankedTensorType>();
    }

    /// An extract_slice result type can be fully inferred from the source type
    /// and the static representation of offsets, sizes and strides. Special
    /// sentinels encode the dynamic case.
    static Type inferResultType(RankedTensorType sourceRankedTensorType,
                                ArrayRef<int64_t> staticOffsets,
                                ArrayRef<int64_t> staticSizes,
                                ArrayRef<int64_t> staticStrides);
    static Type inferResultType(RankedTensorType sourceRankedTensorType,
                                ArrayRef<OpFoldResult> staticOffsets,
                                ArrayRef<OpFoldResult> staticSizes,
                                ArrayRef<OpFoldResult> staticStrides);
    static Type inferRankReducedResultType(unsigned resultRank,
                                           RankedTensorType sourceRankedTensorType,
                                           ArrayRef<int64_t> staticOffsets,
                                           ArrayRef<int64_t> staticSizes,
                                           ArrayRef<int64_t> staticStrides);
    static Type inferRankReducedResultType(unsigned resultRank,
                                           RankedTensorType sourceRankedTensorType,
                                           ArrayRef<OpFoldResult> staticOffsets,
                                           ArrayRef<OpFoldResult> staticSizes,
                                           ArrayRef<OpFoldResult> staticStrides);

    /// Return the expected rank of each of the`static_offsets`, `static_sizes`
    /// and `static_strides` attributes.
    std::array<unsigned, 3> getArrayAttrMaxRanks() {
      unsigned rank = getSourceType().getRank();
      return {rank, rank, rank};
    }

    /// Return the number of leading operands before the `offsets`, `sizes` and
    /// and `strides` operands.
    static unsigned getOffsetSizeAndStrideStartOperandIndex() { return 1; }

    /// Return the dimensions of the source that are dropped in the
    /// result when the result is rank-reduced.
    llvm::SmallDenseSet<unsigned> getDroppedDims();

  }];

  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// FromElementsOp
//===----------------------------------------------------------------------===//

def Tensor_FromElementsOp : Tensor_Op<"from_elements", [
    NoSideEffect,
    TypesMatchWith<"operand types match result element type",
                   "result", "elements", "SmallVector<Type, 2>("
                   "$_self.cast<ShapedType>().getDimSize(0), "
                   "$_self.cast<ShapedType>().getElementType())">
  ]> {
  string summary = "tensor from elements operation.";
  string description = [{
    Create a 1D tensor from a range of same-type arguments.

    Example:

    ```mlir
    tensor.from_elements i_1, ..., i_N :  tensor<Nxindex>
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$elements);
  let results = (outs 1DTensorOf<[AnyType]>:$result);

  let assemblyFormat = "$elements attr-dict `:` type($result)";

  // This op is fully verified by its traits.
  let verifier = ?;

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "Type":$elementType, "ValueRange":$elements)>,
    // Special case builder for when `elements` has size >=1.
    OpBuilder<(ins "ValueRange":$elements)>
  ];

  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// GenerateOp
//===----------------------------------------------------------------------===//

def Tensor_GenerateOp : Tensor_Op<"generate",
    [RecursiveSideEffects,
     SingleBlockImplicitTerminator<"mlir::tensor::YieldOp">]> {
  string summary = "Creates a dynamically sized tensor from elements";
  string description = [{
    This operation creates a dynamically sized tensor with elements of any type.
    It expects one index operand per dynamic extent of the result tensor.

    The body region defines the tensor's elements. It takes index operands as
    its region arguments that span the index space. The element at the given
    position is yielded with the `yield` operation (see `YieldOp`). There is
    no defined ordering to the invocations of the body. It is conceptually
    a "parallel map" operation.

    Example:

    ```mlir
      %tnsr = tensor.generate %m, %n {
      ^bb0(%i : index, %j : index, %k : index):
        ...
        yield %elem : f32
      } : tensor<?x3x?f32>
    ```
  }];

  let arguments = (ins Variadic<Index>:$dynamicExtents);
  let results = (outs AnyRankedTensor:$result);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$dynamicExtents $body attr-dict `:` type($result)";

  let builders = [
    // Build op and populate its body per callback function.
    OpBuilder<(ins "Type":$resultTy, "ValueRange":$dynamicExtents,
      "function_ref<void(OpBuilder &, Location, ValueRange)>")>,
  ];

  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// InsertOp
//===----------------------------------------------------------------------===//

def Tensor_InsertOp : Tensor_Op<"insert",
    [NoSideEffect,
     TypesMatchWith<"result type matches type of dest",
                    "dest", "result",
                    "$_self.cast<ShapedType>()">,
     TypesMatchWith<"scalar type matches element type of dest",
                    "dest", "scalar",
                    "$_self.cast<ShapedType>().getElementType()">]> {
  let summary = "element insertion operation";
  let description = [{
    The `tensor.insert` op writes a tensor into a tensor `dest`as specified by
    the operation's indices.

    It returns a copy of `dest` with the proper slice updated with the value
    of `scalar`.

    The arity of indices must match the rank of the tensor `dest` (i.e., if a
    tensor is of rank 3, then 3 indices are required for the extract. The
    indices should all be of `index` type.

    Example:

    ```mlir
    %4 = tensor.insert %t into %dest[%1, %2] : tensor<4x4xi32>
    %5 = tensor.insert %rt into %dest[%1, %2] : tensor<?x?xi32>
    %6 = tensor.insert %ut into %dest[%1, %2] : tensor<*xi32>
    ```
  }];

  let arguments = (ins AnyType:$scalar,
                       AnyTensor:$dest,
                       Variadic<Index>:$indices);
  let results = (outs AnyTensor:$result);
  let assemblyFormat = [{
    $scalar `into` $dest `[` $indices `]` attr-dict `:` type($dest)
  }];

  let builders = [
    OpBuilder<(ins "Value":$scalar, "Value":$dest,
      CArg<"ValueRange", "{}">:$indices), [{
      auto resType = dest.getType();
      build($_builder, $_state, resType, scalar, dest, indices);
    }]>];

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// InsertSliceOp
//===----------------------------------------------------------------------===//

def Tensor_InsertSliceOp : BaseOpWithOffsetSizesAndStrides<
    Tensor_Dialect, "insert_slice",
    [NoSideEffect, AttrSizedOperandSegments, OffsetSizeAndStrideOpInterface,
     DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>,
     TypesMatchWith<"expected result type to match dest type",
                    "dest", "result", "$_self">]> {
  let summary = "insert_slice operation";
  let description = [{
    The "insert_slice" operation insert a tensor `source` into another
    tensor `dest` as specified by the operation's offsets, sizes and strides
    arguments.

    It returns a copy of `dest` with the proper slice updated with the value
    of `source`.

    The insert_slice operation supports the following arguments:

    * source: the tensor that is inserted.
    * dest: the tensor into which the source tensor is inserted.
    * offsets: tensor-rank number of offsets into the `dest` tensor into which
               the slice is inserted.
    * sizes: tensor-rank number of sizes which specify the sizes of the result
             tensor type.
    * strides: tensor-rank number of strides that specify subsampling in each
               dimension.

    The representation based on offsets, sizes and strides support a
    partially-static specification via attributes specified through the
    `static_offsets`, `static_sizes` and `static_strides` arguments. A special
    sentinel value ShapedType::kDynamicSize and
    ShapedType::kDynamicStrideOrOffset encodes that the corresponding entry has
    a dynamic value.

    After buffer-allocation, the "insert_slice" op is expected to become an
    in-place buffer update.
  }];

  let arguments = (ins
    AnyRankedTensor:$source,
    AnyRankedTensor:$dest,
    Variadic<Index>:$offsets,
    Variadic<Index>:$sizes,
    Variadic<Index>:$strides,
    I64ArrayAttr:$static_offsets,
    I64ArrayAttr:$static_sizes,
    I64ArrayAttr:$static_strides
  );
  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $source `into` $dest ``
    custom<OperandsOrIntegersOffsetsOrStridesList>($offsets, $static_offsets)
    custom<OperandsOrIntegersSizesList>($sizes, $static_sizes)
    custom<OperandsOrIntegersOffsetsOrStridesList>($strides, $static_strides)
    attr-dict `:` type($source) `into` type($dest)
  }];

  let verifier = ?;

  let builders = [
    // Build a InsertSliceOp with mixed static and dynamic entries.
    OpBuilder<(ins "Value":$source, "Value":$dest,
      "ArrayRef<OpFoldResult>":$offsets, "ArrayRef<OpFoldResult>":$sizes,
      "ArrayRef<OpFoldResult>":$strides,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
    // Build a InsertSliceOp with dynamic entries.
    OpBuilder<(ins "Value":$source, "Value":$dest,
      "ValueRange":$offsets, "ValueRange":$sizes, "ValueRange":$strides,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>
  ];

  let extraClassDeclaration = extraBaseClassDeclaration # [{
    /// Returns the type of the base tensor operand.
    RankedTensorType getSourceType() {
      return source().getType().cast<RankedTensorType>();
    }

    /// The result of a insert_slice is always a tensor.
    RankedTensorType getType() {
      return getResult().getType().cast<RankedTensorType>();
    }

    /// Return the expected rank of each of the`static_offsets`, `static_sizes`
    /// and `static_strides` attributes.
    std::array<unsigned, 3> getArrayAttrMaxRanks() {
      unsigned rank = getType().getRank();
      return {rank, rank, rank};
    }

    /// Return the number of leading operands before the `offsets`, `sizes` and
    /// and `strides` operands.
    static unsigned getOffsetSizeAndStrideStartOperandIndex() { return 2; }
  }];

  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// ReshapeOp
//===----------------------------------------------------------------------===//

def Tensor_ReshapeOp: Tensor_Op<"reshape", [NoSideEffect]>  {
  let summary = "tensor reshape operation";
  let description = [{
    The `reshape` operation converts a tensor from one type to an equivalent
    type with a provided shape. The source and destination types are compatible
    if both have the same element type, same number of elements. The following
    combinations are possible:

    a. Source type is ranked or unranked. Shape argument has static size.
    Result type is ranked.

    ```mlir
    // Reshape statically-shaped tensor.
    %dst = tensor.reshape %src(%shape)
             : (tensor<4x1xf32>, tensor<1xi32>) -> tensor<4xf32>
    %dst0 = tensor.reshape %src(%shape0)
             : (tensor<4x1xf32>, tensor<2xi32>) -> tensor<2x2xf32>
    // Flatten unranked tensor.
    %dst = tensor.reshape %src(%shape)
             : (tensor<*xf32>, tensor<1xi32>) -> tensor<?xf32>
    ```

    b. Source type is ranked or unranked. Shape argument has dynamic size.
    Result type is unranked.

    ```mlir
    // Reshape dynamically-shaped 1D tensor.
    %dst = tensor.reshape %src(%shape)
             : (tensor<?xf32>, tensor<?xi32>) -> tensor<*xf32>
    // Reshape unranked tensor.
    %dst = tensor.reshape %src(%shape)
             : (tensor<*xf32>, tensor<?xi32>) -> tensor<*xf32>
    ```
  }];

  let arguments = (ins
    AnyTensor:$source,
    TensorRankOf<[AnySignlessInteger, Index], [1]>:$shape
  );
  let results = (outs AnyTensor:$result);

  let builders = [OpBuilder<
     (ins "TensorType":$resultType, "Value":$operand, "Value":$shape), [{
       $_state.addOperands(operand);
       $_state.addOperands(shape);
       $_state.addTypes(resultType);
     }]>];

  let extraClassDeclaration = [{
    TensorType getResultType() { return getResult().getType().cast<TensorType>(); }
  }];

  let assemblyFormat = [{
    $source `(` $shape `)` attr-dict `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

def Tensor_YieldOp : Tensor_Op<"yield",
    [NoSideEffect, ReturnLike, Terminator,
     HasParent<"::mlir::tensor::GenerateOp">]> {
  let summary = "Yield a value from a region";
  let description = [{
     This operation is used to yield a single value from a within a region. It
     is used to create dynamically sized tensors
     (see `tensor.generate` op).
  }];

  let arguments = (ins AnyType:$value);
  let assemblyFormat = "$value attr-dict `:` type($value)";
  // Dummy builder to appease code in templated ensureTerminator that
  // GenerateOp's auto-generated parser calls.
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];
  let verifier = ?;
}

#endif // TENSOR_OPS
