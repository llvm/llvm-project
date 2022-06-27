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
include "mlir/Interfaces/TilingInterface.td"
include "mlir/Interfaces/ViewLikeInterface.td"

class Tensor_Op<string mnemonic, list<Trait> traits = []>
    : Op<Tensor_Dialect, mnemonic, traits>;

// Base class for ops with static/dynamic offset, sizes and strides
// attributes/arguments.
class Tensor_OpWithOffsetSizesAndStrides<string mnemonic,
                                         list<Trait> traits = []>
    : Tensor_Op<mnemonic, traits> {
  code extraBaseClassDeclaration = [{
    /// Returns the dynamic sizes for this subview operation if specified.
    ::mlir::Operation::operand_range getDynamicSizes() { return getSizes(); }

    /// Return the list of Range (i.e. offset, size, stride). Each
    /// Range entry contains either the dynamic value or a ConstantIndexOp
    /// constructed with `b` at location `loc`.
    ::mlir::SmallVector<::mlir::Range, 8> getOrCreateRanges(
        ::mlir::OpBuilder &b, ::mlir::Location loc) {
      return ::mlir::getOrCreateRanges(*this, b, loc);
    }
  }];
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
}

//===----------------------------------------------------------------------===//
// DimOp
//===----------------------------------------------------------------------===//

def Tensor_DimOp : Tensor_Op<"dim", [NoSideEffect]> {
  let summary = "dimension index operation";
  let description = [{
    The `tensor.dim` operation takes a tensor and a dimension operand of type
    `index`. It returns the size of the requested dimension of the given
    tensor. If the dimension index is out of bounds, the behavior is undefined.

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
  let hasVerifier = 1;
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
  let hasVerifier = 1;
}


//===----------------------------------------------------------------------===//
// ExtractSliceOp
//===----------------------------------------------------------------------===//

def Tensor_ExtractSliceOp : Tensor_OpWithOffsetSizesAndStrides<"extract_slice", [
    NoSideEffect, AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>,
    OffsetSizeAndStrideOpInterface
  ]> {
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

    After buffer allocation, the "extract_slice" op is expected to lower into a
    memref.subview op.

    An extract_slice operation may additionally reduce the rank of the resulting
    tensor by removing dimensions that are statically known to be of size 1.
    This rank-reduction behavior is not required by the op semantics: this
    flexibility allows to progressively drop unit dimensions while lowering
    between different flavors of ops on that operate on tensors.

    Example:

    ```
    // Rank-reducing extract_slice.
    %1 = tensor.extract_slice %0[0, 0, 0][1, 16, 4][1, 1, 1] :
      tensor<8x16x4xf32> to tensor<16x4xf32>
    %3 = tensor.extract_slice %2[%o0, 4, %o2][1, %sz1, 1][1, %st1, 1] :
      tensor<8x16x4xf32> to tensor<1x?xf32>
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
      return getSource().getType().cast<RankedTensorType>();
    }

    /// The result of an extract_slice is always a tensor.
    RankedTensorType getType() {
      return getResult().getType().cast<RankedTensorType>();
    }

    /// An extract_slice result type can be fully inferred from the source type
    /// and the static representation of offsets, sizes and strides. Special
    /// sentinels encode the dynamic case.
    static RankedTensorType inferResultType(
      RankedTensorType sourceRankedTensorType,
      ArrayRef<int64_t> staticOffsets,
      ArrayRef<int64_t> staticSizes,
      ArrayRef<int64_t> staticStrides);
    static RankedTensorType inferResultType(
      RankedTensorType sourceRankedTensorType,
      ArrayRef<OpFoldResult> staticOffsets,
      ArrayRef<OpFoldResult> staticSizes,
      ArrayRef<OpFoldResult> staticStrides);
    static RankedTensorType inferRankReducedResultType(
      unsigned resultRank,
      RankedTensorType sourceRankedTensorType,
      ArrayRef<int64_t> staticOffsets,
      ArrayRef<int64_t> staticSizes,
      ArrayRef<int64_t> staticStrides);
    static RankedTensorType inferRankReducedResultType(
      unsigned resultRank,
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
    llvm::SmallBitVector getDroppedDims();
  }];

  let hasCanonicalizer = 1;
  let hasFolder = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// FromElementsOp
//===----------------------------------------------------------------------===//

def Tensor_FromElementsOp : Tensor_Op<"from_elements", [
    NoSideEffect,
    TypesMatchWith<"operand types match result element type",
                   "result", "elements", "SmallVector<Type, 2>("
                   "$_self.cast<ShapedType>().getNumElements(), "
                   "$_self.cast<ShapedType>().getElementType())">
  ]> {
  string summary = "tensor from elements operation.";
  string description = [{
    Create a N-D tensor from a range of same-type arguments. The number of
    provided `elements` should equal to the number of the elements in the
    result type. The `elements` correspond to a flattened tensor.

    Example:

    ```mlir
    tensor.from_elements %a, %b, %c, %d, %e, %f :  tensor<2x3xindex>
    ```

    will result in a tensor

    [[%a, %b, %c]
     [%d, %e, %f]]
  }];

  let arguments = (ins Variadic<AnyType>:$elements);
  let results = (outs AnyStaticShapeTensor:$result);

  let assemblyFormat = "$elements attr-dict `:` type($result)";

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "Type":$resultType, "ValueRange":$elements)>,
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
     DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>,
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
  let hasVerifier = 1;
  let hasRegionVerifier = 1;
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
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// InsertSliceOp
//===----------------------------------------------------------------------===//

def Tensor_InsertSliceOp : Tensor_OpWithOffsetSizesAndStrides<"insert_slice", [
    NoSideEffect, AttrSizedOperandSegments, OffsetSizeAndStrideOpInterface,
    DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>,
    TypesMatchWith<"expected result type to match dest type",
                   "dest", "result", "$_self">
  ]> {
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
    * sizes: tensor-rank number of sizes which specify the sizes of the source
             tensor type.
    * strides: tensor-rank number of strides that specify subsampling in each
               dimension.

    The representation based on offsets, sizes and strides support a
    partially-static specification via attributes specified through the
    `static_offsets`, `static_sizes` and `static_strides` arguments. A special
    sentinel value ShapedType::kDynamicSize and
    ShapedType::kDynamicStrideOrOffset encodes that the corresponding entry has
    a dynamic value.

    After buffer allocation, the "insert_slice" op is expected to lower into a
    memref.subview op.

    An insert_slice operation may additionally specify insertion into a tensor
    of higher rank than the source tensor, along dimensions that are statically
    known to be of size 1.
    This rank-altering behavior is not required by the op semantics: this
    flexibility allows to progressively drop unit dimensions while lowering
    between different flavors of ops on that operate on tensors.
    The rank-altering behavior of tensor.insert_slice matches the rank-reducing
    behavior of tensor.extract_slice.

    Example:

    ```
    // Rank-altering insert_slice.
    %1 = tensor.insert_slice %t into %0[0, 0, 0][1, 16, 4][1, 1, 1] :
      tensor<16x4xf32> into tensor<8x16x4xf32>
    %3 = tensor.insert_slice %tt into %2[%o0, 4, %o2][1, %sz1, 1][1, %st1, 1] :
      tensor<1x?xf32> into tensor<8x16x4xf32>
    ```
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
      return getSource().getType().cast<RankedTensorType>();
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
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// RankOp
//===----------------------------------------------------------------------===//

def Tensor_RankOp : Tensor_Op<"rank", [NoSideEffect]> {
  let summary = "rank operation";
  let description = [{
    The `tensor.rank` operation takes a tensor operand and returns its rank.

    Example:

    ```mlir
    %0 = tensor.rank %arg0 : tensor<*xf32>
    %1 = tensor.rank %arg1 : tensor<?x?xf32>
    ```
  }];

  let arguments = (ins AnyTensor:$tensor);
  let results = (outs Index);

  let hasFolder = 1;
  let assemblyFormat = "$tensor attr-dict `:` type($tensor)";
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
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ExpandShapeOp / CollapseShapeOp
//===----------------------------------------------------------------------===//

class Tensor_ReassociativeReshapeOp<string mnemonic, list<Trait> traits = []> :
    Tensor_Op<mnemonic, !listconcat(traits, [NoSideEffect])>,
    Arguments<(ins AnyTensor:$src, IndexListArrayAttr:$reassociation)>,
    Results<(outs AnyTensor:$result)> {

  code commonExtraClassDeclaration = [{
    static StringRef getReassociationAttrStrName() { return "reassociation"; }
    SmallVector<AffineMap, 4> getReassociationMaps();
    SmallVector<ReassociationExprs, 4> getReassociationExprs();
    SmallVector<ReassociationIndices, 4> getReassociationIndices() {
      SmallVector<ReassociationIndices, 4> reassociationIndices;
      for (auto attr : getReassociation())
        reassociationIndices.push_back(llvm::to_vector<2>(
            llvm::map_range(attr.cast<ArrayAttr>(), [&](Attribute indexAttr) {
              return indexAttr.cast<IntegerAttr>().getInt();
            })));
      return reassociationIndices;
    };
    RankedTensorType getSrcType() {
      return getSrc().getType().cast<RankedTensorType>();
    }
    RankedTensorType getResultType() {
      return getResult().getType().cast<RankedTensorType>();
    }
  }];

  let assemblyFormat = [{
    $src $reassociation attr-dict `:` type($src) `into` type($result)
  }];

  let hasFolder = 1;
  let hasCanonicalizer = 1;
  let hasVerifier = 1;
}

def Tensor_ExpandShapeOp : Tensor_ReassociativeReshapeOp<"expand_shape"> {
  let summary = "operation to produce a tensor with a higher rank";
  let description = [{
    The `tensor.expand_shape` op produces a new tensor with a higher
    rank whose sizes are a reassociation of the original `src`.

    A reassociation is defined as a continuous grouping of dimensions and is
    represented with an array of I64ArrayAttr attribute.

    The verification rule is that the reassociation maps are applied to the
    result tensor with the higher rank to obtain the operand tensor with the
    smaller rank.

    The operand tensor type of a reshape can be zero-ranked if the result
    tensor type is statically shaped with all dimensions being unit extent. In
    such cases the reassociation map is empty.

    Examples:

    ```mlir
    // Dimension expansion i -> (i', j') and (k) -> (k')
    %b = tensor.expand_shape %a [[0, 1], [2]]
        : tensor<?x?xf32> into tensor<?x?x?xf32>
    ```
  }];
  let builders = [
    // Builders using ReassociationIndices.
    OpBuilder<(ins "Type":$resultType, "Value":$src,
      "ArrayRef<ReassociationIndices>":$reassociation,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs),
    [{
      build($_builder, $_state, resultType, src, attrs);
      $_state.addAttribute("reassociation",
          getReassociationIndicesAttribute($_builder, reassociation));
    }]>,
    OpBuilder<(ins "Type":$resultType, "Value":$src,
      "ArrayRef<ReassociationExprs>":$reassociation,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs),
    [{
      auto reassociationMaps =
          convertReassociationMapsToIndices($_builder, reassociation);
      build($_builder, $_state, resultType, src, reassociationMaps, attrs);
    }]>
  ];

  let extraClassDeclaration = commonExtraClassDeclaration;
  let hasVerifier = 1;
}

def Tensor_CollapseShapeOp : Tensor_ReassociativeReshapeOp<"collapse_shape"> {
  let summary = "operation to produce a tensor with a smaller rank";
  let description = [{
    The `tensor.collapse_shape` op produces a new tensor with a smaller
    rank whose sizes are a reassociation of the original `src`.

    A reassociation is defined as a continuous grouping of dimensions and is
    represented with an array of I64ArrayAttr attribute.

    The verification rule is that the reassociation maps are applied to the
    operand tensor with the higher rank to obtain the result tensor with the
    smaller rank.

    The result tensor type of a reshape can be zero-ranked if the operand
    tensor type is statically shaped with all dimensions being unit extent. In
    such case the reassociation map is empty.

    Examples:

    ```mlir
    // Dimension collapse (i, j) -> i' and k -> k'
    %b = tensor.collapse_shape %a [[0, 1], [2]]
        : tensor<?x?x?xf32> into tensor<?x?xf32>
    ```
  }];
  let builders = [
    // Builders for a contracting reshape whose result type is computed from
    // `src` and `reassociation`.
    OpBuilder<(ins "Value":$src,
      "ArrayRef<ReassociationIndices>":$reassociation,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
    OpBuilder<(ins "Value":$src,
      "ArrayRef<ReassociationExprs>":$reassociation,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs),
    [{
      auto reassociationMaps =
          convertReassociationMapsToIndices($_builder, reassociation);
      build($_builder, $_state, src, reassociationMaps, attrs);
    }]>,

    // Builders for a reshape whose result type is passed explicitly.
    OpBuilder<(ins "Type":$resultType, "Value":$src,
      "ArrayRef<ReassociationIndices>":$reassociation,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs),
    [{
      build($_builder, $_state, resultType, src, attrs);
      $_state.addAttribute("reassociation",
          getReassociationIndicesAttribute($_builder, reassociation));
    }]>,
    OpBuilder<(ins "Type":$resultType, "Value":$src,
      "ArrayRef<ReassociationExprs>":$reassociation,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs),
    [{
      auto reassociationMaps =
          convertReassociationMapsToIndices($_builder, reassociation);
      build($_builder, $_state, resultType, src, reassociationMaps, attrs);
    }]>
  ];

  let extraClassDeclaration = commonExtraClassDeclaration;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// PadOp
//===----------------------------------------------------------------------===//

def Tensor_PadOp : Tensor_Op<"pad", [AttrSizedOperandSegments, NoSideEffect,
    SingleBlockImplicitTerminator<"mlir::tensor::YieldOp">]> {
  let summary = "tensor pad operation";
  let description = [{
    `tensor.pad` is an operation that pads the `source` tensor
    with given `low` and `high` padding config.

    The PadOp operation supports the following arguments:

    * source: the "base" tensor on which to pad.
    * low: A list contains the padding along the start of each
           dimension, i.e `low`.
    * high: A list contains the padding along the end of each
            dimension, i.e. `high`.
    * nofold: indicates that the operation should not be folded when source and
              result types are equal.

    The result tensor dimensions are `low` + `dim` + `high` along that
    dimension. The number of elements of `low` and `high` must match
    the rank of the input tensor. They can be either a constant or a
    dynamic value.

    The region of the `tensor.pad` operation returns the value to use
    for the padding. The arguments of the region represent the index
    of the source being accessed. There should be as many arguments as
    the rank of the `source` tensor. The value `yield`-ed by the
    region is used as the value of the view at the given position.

    If `nofold` is set, the padding operation will not be folded away even
    if the source type and the padded type have the same static shape. This can
    be used, e.g., for packing or promotion to faster memory.

    Example 1:

    ```mlir
      %pad_value = ... : f32
      %0 = tensor.pad %0 low[1, 2] high[2, 3] {
      ^bb0(%arg0 : index, %arg1 : index):
        tensor.yield %pad_value : f32
      } : tensor<?x?xf32> to tensor<?x?xf32>
    ```

    Example 2:

    ```mlir
      %pad_value = ... : f32
      %0 = tensor.pad %arg0 low[2, %arg1, 3, 3] high[3, 3, %arg1, 2] {
      ^bb0(%arg2: index, %arg3: index, %arg4: index, %arg5: index):
          tensor.yield %pad_value : f32
      } : tensor<1x2x2x?xf32> to tensor<6x?x?x?xf32>
    ```

    Example 3:

    ```mlir
      %pad_value = ... : f32
      %0 = tensor.pad %arg0 low[0, 0] high[%ub0, %ub1] {
      ^bb0(%arg1: index, %arg2: index):
        tensor.yield %pad_value : f32
      } : tensor<2x3xf32> to tensor<?x?xf32>
    ```

    Example 4:

    ```mlir
      // Force a padded value to be always exist with `nofold`.
      %pad_value = ... : f32
      %0 = tensor.pad %arg0 nofold low[0, 0] high[0, 0] {
      ^bb0(%arg1: index, %arg2: index):
        tensor.yield %pad_value : f32
      } : tensor<2x3xf32> to tensor<2x3xf32>
    ```
  }];

  let arguments = (ins
    AnyTensor:$source,
    Variadic<Index>:$low,
    Variadic<Index>:$high,
    I64ArrayAttr:$static_low,
    I64ArrayAttr:$static_high,
    UnitAttr:$nofold);

  let regions = (region SizedRegion<1>:$region);

  let results = (outs AnyTensor:$result);

  // TODO: Remove custom<InferType> when AllTypesMatch supports opt. operands.
  let assemblyFormat = [{
    $source
    (`nofold` $nofold^)?
    `low` `` custom<OperandsOrIntegersSizesList>($low, $static_low)
    `high` `` custom<OperandsOrIntegersSizesList>($high, $static_high)
    $region attr-dict `:` type($source) `to` type($result)
  }];

  let extraClassDeclaration = [{
    static StringRef getStaticLowAttrStrName() {
      return "static_low";
    }

    static StringRef getStaticHighAttrStrName() {
      return "static_high";
    }

    RankedTensorType getSourceType() {
      return getSource().getType().cast<RankedTensorType>();
    }
    RankedTensorType getResultType() {
      return getResult().getType().cast<RankedTensorType>();
    }

    // Infer the shape of the result tensor given the type of the source tensor
    // and paddings. Known result dimensions that cannot necessarily be inferred
    // from low/high padding sizes can be optionally specified. Those will be
    // considered when computing the result type.
    static RankedTensorType inferResultType(
                                RankedTensorType sourceType,
                                ArrayRef<int64_t> staticLow,
                                ArrayRef<int64_t> staticHigh,
                                ArrayRef<int64_t> resultShape = {});

    // Return the pad value if it is a constant. Return null value otherwise.
    Value getConstantPaddingValue();

    // Return a vector of all the static or dynamic values (low/high padding) of
    // the op.
    inline SmallVector<OpFoldResult> getMixedPadImpl(ArrayAttr staticAttrs,
                                                     ValueRange values) {
      SmallVector<OpFoldResult> res;
      unsigned numDynamic = 0;
      unsigned count = staticAttrs.size();
      for (unsigned idx = 0; idx < count; ++idx) {
        if (ShapedType::isDynamic(staticAttrs[idx].cast<IntegerAttr>().getInt()))
          res.push_back(values[numDynamic++]);
        else
          res.push_back(staticAttrs[idx]);
      }
      return res;
    }
    SmallVector<OpFoldResult> getMixedLowPad() {
      return getMixedPadImpl(getStaticLow(), getLow());
    }
    SmallVector<OpFoldResult> getMixedHighPad() {
      return getMixedPadImpl(getStaticHigh(), getHigh());
    }
    // Return true if low padding is guaranteed to be 0.
    bool hasZeroLowPad() {
      return llvm::all_of(getMixedLowPad(), [](OpFoldResult ofr) {
        return getConstantIntValue(ofr) == static_cast<int64_t>(0);
      });
    }
    // Return true if high padding is guaranteed to be 0.
    bool hasZeroHighPad() {
      return llvm::all_of(getMixedHighPad(), [](OpFoldResult ofr) {
        return getConstantIntValue(ofr) == static_cast<int64_t>(0);
      });
    }
    /// Return the dimensions with a non-zero low or high padding.
    llvm::SmallBitVector getPaddedDims();
  }];

  let builders = [
    // Build a PadOp with mixed static and dynamic entries.
    OpBuilder<(ins "Value":$source, "ArrayRef<int64_t>":$staticLow,
      "ArrayRef<int64_t>":$staticHigh, "ValueRange":$low, "ValueRange":$high,
      CArg<"bool", "false">:$nofold,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
    // Build a PadOp with all dynamic entries.
    OpBuilder<(ins "Value":$source, "ValueRange":$low, "ValueRange":$high,
      CArg<"bool", "false">:$nofold,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
    // Build a PadOp with mixed static and dynamic entries and custom
    // result type. If the type passed is nullptr, it is inferred.
    OpBuilder<(ins "Type":$resultType, "Value":$source,
      "ArrayRef<OpFoldResult>":$low, "ArrayRef<OpFoldResult>":$high,
      CArg<"bool", "false">:$nofold,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
  ];

  let hasCanonicalizer = 1;
  let hasFolder = 1;
  let hasVerifier = 1;
  let hasRegionVerifier = 1;
}

//===----------------------------------------------------------------------===//
// SplatOp
//===----------------------------------------------------------------------===//

def Tensor_SplatOp : Tensor_Op<"splat", [
    NoSideEffect,
    TypesMatchWith<"operand type matches element type of result",
                   "aggregate", "input",
                   "$_self.cast<TensorType>().getElementType()">
  ]> {
  let summary = "tensor splat or broadcast operation";
  let description = [{
    Broadcast the operand to all elements of the result tensor. The operand is
    required to be of integer/index/float type, and the result tensor must be
    statically shaped.

    Example:

    ```mlir
    %s = arith.constant 10.1 : f32
    %t = tensor.splat %s : tensor<8x16xi32>
    ```

    TODO: This operation is easy to extend to broadcast to dynamically shaped
          tensors:

    ```mlir
    // Broadcasts %s to a 2-d dynamically shaped tensor, with %m, %n binding
    // to the sizes of the two dynamic dimensions.
    %m = "foo"() : () -> (index)
    %n = "bar"() : () -> (index)
    %t = tensor.splat %s [%m, %n] : tensor<?x?xi32>
    ```
  }];

  let arguments = (ins AnyTypeOf<[AnySignlessInteger, Index, AnyFloat],
                                 "integer/index/float type">:$input);
  let results = (outs AnyStaticShapeTensor:$aggregate);

  let builders = [
    OpBuilder<(ins "Value":$element, "Type":$aggregateType),
    [{ build($_builder, $_state, aggregateType, element); }]>];
  let assemblyFormat = "$input attr-dict `:` type($aggregate)";

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

def Tensor_YieldOp : Tensor_Op<"yield",
    [NoSideEffect, ReturnLike, Terminator,
     HasParent<"::mlir::tensor::GenerateOp, ::mlir::tensor::PadOp">]> {
  let summary = "Yield a value from a region";
  let description = [{
     This operation is used to yield a single value from a within a region. It
     is used to create dynamically sized tensors
     (see `tensor.generate` and `tensor.pad` ops).
  }];

  let arguments = (ins AnyType:$value);
  let assemblyFormat = "$value attr-dict `:` type($value)";

  // Dummy builder to appease code in templated ensureTerminator that
  // GenerateOp's auto-generated parser calls.
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];
}

#endif // TENSOR_OPS
