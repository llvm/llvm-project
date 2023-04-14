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
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/ParallelCombiningOpInterface.td"
include "mlir/Interfaces/ShapedOpInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/TilingInterface.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "mlir/IR/OpAsmInterface.td"

class Tensor_Op<string mnemonic, list<Trait> traits = []>
    : Op<Tensor_Dialect, mnemonic, traits>;

// Base class for ops with static/dynamic offset, sizes and strides
// attributes/arguments.
class Tensor_OpWithOffsetSizesAndStrides<string mnemonic,
                                         list<Trait> traits = []>
    : Tensor_Op<mnemonic, traits> {
  code extraBaseClassDeclaration = [{
    /// Return the type of the base tensor operand.
    ::mlir::RankedTensorType getSourceType() { 
      return getSource().getType().cast<RankedTensorType>();
    }

    /// Return the type of the result tensor.
    ::mlir::RankedTensorType getResultType() { 
      return getResult().getType().cast<RankedTensorType>();
    }

    /// Return the dynamic sizes for this subview operation if specified.
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
    DeclareOpInterfaceMethods<CastOpInterface>,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
    Pure
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

def Tensor_DimOp : Tensor_Op<"dim", [
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
    ConditionallySpeculatable, NoMemoryEffect,
    ShapedDimOpInterface]> {
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

    // Return the dynamic dimension of %A.
    %c1 = arith.constant 1 : index
    %y = tensor.dim %A, %c1 : memref<4x?xf32>

    // Equivalent generic form:
    %x = "tensor.dim"(%A, %c0) : (memref<4x?xf32>, index) -> index
    %y = "tensor.dim"(%A, %c1) : (memref<4x?xf32>, index) -> index
    ```
  }];

  let arguments = (ins AnyNon0RankedOrUnrankedTensor:$source,
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
    std::optional<int64_t> getConstantIndex();

    /// Interface method of ShapedDimOpInterface: Return the source tensor.
    Value getShapedValue() { return getSource(); }

    /// Interface method of ShapedDimOpInterface: Return the dimension.
    OpFoldResult getDimension() { return getIndex(); }

    /// Interface method for ConditionallySpeculatable.
    Speculation::Speculatability getSpeculatability();
  }];

  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// EmptyOp
//===----------------------------------------------------------------------===//

def Tensor_EmptyOp : Tensor_Op<"empty",
    [Pure,
     DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>]> {
  let summary = "empty tensor operation";

  let description = [{
    `tensor.empty` is an operation that defines a tensor of a particular shape.
    The shape could be dynamic or static. The contents of the tensor are
    unspecified and the only purpose of the op result is to materialize the
    specified shape in IR and make it available to other transformations.

    `tensor.empty` is useful in transformations that expect destination style
    ops. I.e., ops that implement `DestinationStyleOpInterface`. Ops that are
    not in destination style can be made compatible with such transformations
    with a `tensor.empty` destination.

    Note: This op can be lowered to a `bufferization.alloc_tensor`, at which
    point it turns into an explicit buffer allocation.
  }];

  let arguments = (ins Variadic<Index>:$dynamicSizes);

  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = "`(`$dynamicSizes`)` attr-dict `:` type($result)";

  let extraClassDeclaration = [{
    RankedTensorType getType() {
      return getResult().getType().cast<RankedTensorType>();
    }

    // Return both static and dynamic sizes as a list of `OpFoldResult`.
    SmallVector<OpFoldResult> getMixedSizes();

    // Return the Value of the dynamic size of the tensor at dimension `idx`.
    // Asserts that the shape is dynamic at that `idx`.
    Value getDynamicSize(unsigned idx);
  }];

  let builders = [
    // Build with fully static sizes.
    OpBuilder<(ins "ArrayRef<int64_t>":$staticShape, "Type":$elementType,
                   CArg<"Attribute", "{}">:$encoding)>,

    // Build with mixed static/dynamic sizes.
    OpBuilder<(ins "ArrayRef<int64_t>":$staticShape, "Type":$elementType,
                   "ValueRange":$dynamicSizes,
                   CArg<"Attribute", "{}">:$encoding)>,

    // Build with mixed static/dynamic sizes.
    OpBuilder<(ins "ArrayRef<OpFoldResult>":$sizes, "Type":$elementType,
                   CArg<"Attribute", "{}">:$encoding)>
  ];

  let hasCanonicalizer = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ExtractOp
//===----------------------------------------------------------------------===//

def Tensor_ExtractOp : Tensor_Op<"extract", [
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
    Pure,
    TypesMatchWith<"result type matches element type of tensor",
                   "tensor", "result",
                   "$_self.cast<ShapedType>().getElementType()">]> {
  let summary = "element extraction operation";
  let description = [{
    The `tensor.extract` op reads a ranked tensor and returns one element as
    specified by the given indices. The result of the op is a value with the
    same type as the elements of the tensor. The arity of indices must match
    the rank of the accessed value. All indices should all be of `index` type.

    Example:

    ```mlir
    %4 = tensor.extract %t[%1, %2] : tensor<4x4xi32>
    %5 = tensor.extract %rt[%1, %2] : tensor<?x?xi32>
    ```
  }];

  let arguments = (ins AnyRankedTensor:$tensor, Variadic<Index>:$indices);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$tensor `[` $indices `]` attr-dict `:` type($tensor)";

  let hasCanonicalizer = 1;
  let hasFolder = 1;
  let hasVerifier = 1;
}


//===----------------------------------------------------------------------===//
// ExtractSliceOp
//===----------------------------------------------------------------------===//

def Tensor_ExtractSliceOp : Tensor_OpWithOffsetSizesAndStrides<"extract_slice", [
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
    DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>,
    AttrSizedOperandSegments,
    Pure,
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
    sentinel value ShapedType::kDynamic and
    ShapedType::kDynamic encodes that the corresponding entry has
    a dynamic value.

    After buffer allocation, the "extract_slice" op is expected to lower into a
    memref.subview op.

    An extract_slice operation may additionally reduce the rank of the resulting
    tensor by removing dimensions that are statically known to be of size 1.
    This rank-reduction behavior is not required by the op semantics: this
    flexibility allows to progressively drop unit dimensions while lowering
    between different flavors of ops on that operate on tensors.

    #### Verification vs Inference in the rank-reduced case

    Note that there may be multiple ways to infer a resulting rank-reduced type.
      e.g. 1x6x1 could potentially rank-reduce to either 1x6 or 6x1 2-D shapes.

    To disambiguate, the inference helpers `inferCanonicalRankReducedResultType`
    only drop the first unit dimensions, in order:
      e.g. 1x6x1 rank-reduced to 2-D will infer the 6x1 2-D shape, but not 1x6.

    Verification however has access to result type and does not need to infer.
    The verifier calls `isRankReducedType(getSource(), getResult())` to
    determine whether the result type is rank-reduced from the source type.
    This computes a so-called rank-reduction mask, consisting of dropped unit
    dims, to map the rank-reduced type to the source type by dropping ones:
      e.g. 1x6 is a rank-reduced version of 1x6x1 by mask {2}
           6x1 is a rank-reduced version of 1x6x1 by mask {0}
           1x2x1x4 is a rank-reduced version of 1x1x2x1x1x4x1 by mask {1, 4, 6}
             (remaining common 1 dimensions are matched eagerly)

    Example:

    ```mlir
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
    DenseI64ArrayAttr:$static_offsets,
    DenseI64ArrayAttr:$static_sizes,
    DenseI64ArrayAttr:$static_strides
  );
  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $source ``
    custom<DynamicIndexList>($offsets, $static_offsets)
    custom<DynamicIndexList>($sizes, $static_sizes)
    custom<DynamicIndexList>($strides, $static_strides)
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
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
    // Build an ExtractSliceOp with mixed static and dynamic entries packed in
    // a Range vector.
    OpBuilder<(ins "Value":$source, "ArrayRef<Range>":$ranges,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
  ];

  let extraClassDeclaration = extraBaseClassDeclaration # [{
    /// The result of an extract_slice is always a tensor.
    // TODO: deprecate
    RankedTensorType getType() {
      return getResultType();
    }

    /// Compute the rank-reduction mask that can be applied to map the source
    /// tensor type to the result tensor type by dropping unit dims.
    std::optional<llvm::SmallDenseSet<unsigned>>
    computeRankReductionMask() {
      return ::mlir::computeRankReductionMask(getSourceType().getShape(),
                                              getType().getShape());
    };

    /// An extract_slice result type can be inferred, when it is not
    /// rank-reduced, from the source type and the static representation of
    /// offsets, sizes and strides. Special sentinels encode the dynamic case.
    static RankedTensorType inferResultType(
      ShapedType sourceShapedTensorType,
      ArrayRef<int64_t> staticOffsets,
      ArrayRef<int64_t> staticSizes,
      ArrayRef<int64_t> staticStrides);
    static RankedTensorType inferResultType(
      ShapedType sourceShapedTensorType,
      ArrayRef<OpFoldResult> staticOffsets,
      ArrayRef<OpFoldResult> staticSizes,
      ArrayRef<OpFoldResult> staticStrides);

    /// If the rank is reduced (i.e. the desiredResultRank is smaller than the
    /// number of sizes), drop as many size 1 as needed to produce an inferred type
    /// with the desired rank.
    ///
    /// Note that there may be multiple ways to compute this rank-reduced type:
    ///   e.g. 1x6x1 can rank-reduce to either 1x6 or 6x1 2-D tensors.
    ///
    /// To disambiguate, this function always drops the first 1 sizes occurrences.
    static RankedTensorType inferCanonicalRankReducedResultType(
      unsigned resultRank,
      RankedTensorType sourceRankedTensorType,
      ArrayRef<int64_t> staticOffsets,
      ArrayRef<int64_t> staticSizes,
      ArrayRef<int64_t> staticStrides);
    static RankedTensorType inferCanonicalRankReducedResultType(
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

    /// Given a `value`, asserted to be of RankedTensorType, build an
    /// ExtractSliceOp that results in a rank-reducing extract to the desired
    /// tensor shape and return the new value created.
    /// If the shape of `value` is already the `desiredShape`, just return
    /// `value`.
    /// If the shape of `value` cannot be rank-reduced to `desiredShape`, fail.
    static FailureOr<Value> rankReduceIfNeeded(
      OpBuilder &b, Location loc, Value value, ArrayRef<int64_t> desiredShape);
  }];

  let hasCanonicalizer = 1;
  let hasFolder = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// FromElementsOp
//===----------------------------------------------------------------------===//

def Tensor_FromElementsOp : Tensor_Op<"from_elements", [
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
    Pure,
    TypesMatchWith<"operand types match result element type",
                   "result", "elements", "SmallVector<Type, 2>("
                   "$_self.cast<ShapedType>().getNumElements(), "
                   "$_self.cast<ShapedType>().getElementType())">
  ]> {
  let summary = "tensor from elements operation.";
  let description = [{
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
// GatherOp
//===----------------------------------------------------------------------===//

def Tensor_GatherOp : Tensor_Op<"gather", [
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
    Pure
  ]> {
  let summary = "gather a subset of a tensor at specified indices";
  let description = [{
    The `gather` operation extracts a subset of the elements from a `source`
    tensor at the given indices.

    In its most general form, the tensor of indices specifies all the coordinates
    of every element to extract (i.e. COO format, without the payload).
    The indices are expected to be confined to coordinate values that fit the
    range of the `source` tensor, otherwise the behavior is undefined.

    The leading dimensions of the index tensor give the result tensor its leading
    dimensions. The trailing dimensions of the result tensor are obtained from
    the source tensor by omitting the dimensions specified in `gather_dims`
    (rank-reducing semantics) or setting them to `1` (rank-preserving semantics)
    (see examples).
    The trailing dimension of the index tensor contains the coordinates and is
    expected to have its size equal to the number of dimensions being gathered.
    This convention allows an idiomatic specification and lowering of "gathering
    multiple N-D slices from the source tensor".

    Note: in the examples below, we separate out the indexing part of the tensor
    type by a whitespace for readability purposes.

    Example:

    ```mlir
        // For each 1x2 triple of coordinates in %indices, extract the
        // element (i.e. 0-D subset) at the coordinates triple in %source.
        //
        %out = tensor.gather %source[%indices] gather_dims([0, 1, 2]) :
          (tensor<4x4x4xf32>, tensor<1x2x 3xindex>) -> tensor<1x2x 1x1x1xf32>

        // Note: result type may be further rank-reduced to tensor<1x2x f32>.
    ```

    A slice variant is provided to allow specifying whole slices of the source
    tensor.

    Example:

    ```mlir
        // For each 5x6 singleton of coordinates in %indices, extract the 2-D
        // slice %source[*, %indices[...]:%indices[...] + 1, *] with the indices
        // corresponding to the `gather_dims` attribute specified by %indices.
        //
        %out = tensor.gather %source[%indices] gather_dims([1]) :
          (tensor<3x4x5xf32>, tensor<6x7x 1xindex>) -> tensor<6x7x 3x1x5xf32>

        // Note: result type may be further rank-reduced to tensor<6x7x 3x5xf32>.
    ```

    The dimensions specified in the gather_dims attribute are ones for which the
    result tensor has size `1`.
    I.e. if the source type is `axbxcxd` and the coordinates are [1, 3], then
    the shape suffix is `ax1xcx1`.
    Gather also allows rank-reducing semantics where the shape `ax1xcx1` can be
    further simplified to `axc`.

    The elemental type of the indices tensor can be any integer type.
    In the absence of target-specific or problem specific information the default
    type one should use is `index`.

    This operation does not support unranked tensors.

    An optional `unique` unit attribute may be specified to indicate that the
    coordinates in `indices` are statically guaranteed to be unique at runtime.
    Incorrectly setting the `unique` attribute when the coordinates are not truly
    unique is undefined behavior.

    Only full slices are meant to be supported by this op, if one desires
    partial slices (e.g. strided windows) one should compose this op with other
    tensor ops (e.g. tensor.extract_slice). This is to avoid a slippery slope of
    complexity that would make the op unusable in practice.

    At the tensor-level, the index tensor is specified in an AoS form (i.e.
    coordinate tuple is the most minor). It is the responsibility of further
    lowerings and bufferiation to implement various concrete layouts.

    Note: As currently specified, the operation must lower to an abstraction that
    performs copies to the output tensor. This is because the buffer type system
    is currently not rich enough to allow multiple non-contiguous views in the
    same type. This is visible more clearly in a notional buffer version of the
    op:

    ```mlir
        // memref<?x4x1xf32> is a contiguous buffer of ?x4x1 elements.
        // gather from random source slices must copy to the contiguous output.
        %out = memref.gather %source[%indices] gather_dims([1]) :
          (memref<4x4xf32>, memref<?x 1xindex>) -> memref<?x 4x1xf32>

        // Nested buffer support would allow gather to directly index into the
        // source buffer (i.e. represent a jagged view into the source).
        %out = memref.gather %source[%indices] gather_dims([1]) :
          (memref<4x4xf32>, memref<?x 1xindex>) -> memref<? x memref<4x1xf32>>
    ```
  }];

  let arguments = (ins AnyRankedTensor:$source,
                       RankedTensorOf<[AnySignlessIntegerOrIndex]>:$indices,
                       DenseI64ArrayAttr:$gather_dims,
                       UnitAttr:$unique);
  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $source `[` $indices `]`
      `gather_dims` `(` $gather_dims `)`
      (`unique` $unique^)?
      attr-dict
    `:` functional-type(operands, results)
  }];

  let extraClassDeclaration = [{
    // TODO: InferTypeOpInterface once enough confidence is built with
    // tensor<tensor> and its lwoering to memref<memref>.
    static RankedTensorType inferResultType(RankedTensorType sourceType,
                                            RankedTensorType indicesType,
                                            ArrayRef<int64_t> gatherDims,
                                            bool rankReduced);
    RankedTensorType getIndicesType() {
      return getIndices().getType().cast<RankedTensorType>();
    }
    RankedTensorType getSourceType() {
      return getSource().getType().cast<RankedTensorType>();
    }
    RankedTensorType getResultType() {
      return getResult().getType().cast<RankedTensorType>();
    }
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// GenerateOp
//===----------------------------------------------------------------------===//

def Tensor_GenerateOp : Tensor_Op<"generate", [
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
    RecursiveMemoryEffects,
    DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>,
    SingleBlockImplicitTerminator<"mlir::tensor::YieldOp">]> {
  let summary = "Creates a dynamically sized tensor from elements";
  let description = [{
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

def Tensor_InsertOp : Tensor_Op<"insert", [
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
    DestinationStyleOpInterface,
    Pure,
    TypesMatchWith<"result type matches type of dest",
                   "dest", "result",
                   "$_self">,
    TypesMatchWith<"scalar type matches element type of dest",
                   "dest", "scalar",
                   "$_self.cast<ShapedType>().getElementType()">]> {
  let summary = "element insertion operation";
  let description = [{
    The `tensor.insert` op inserts a scalar into a ranked tensor `dest` as
    specified by the operation's indices.

    It returns a copy of `dest` with the indexed position updated to the value
    of `scalar`.

    The arity of `indices `must match the rank of the tensor `dest`. All
    indices should be of `index` type.

    Example:

    ```mlir
    %4 = tensor.insert %t into %dest[%1, %2] : tensor<4x4xi32>
    %5 = tensor.insert %rt into %dest[%1, %2] : tensor<?x?xi32>
    ```
  }];

  let arguments = (ins AnyType:$scalar,
                       AnyRankedTensor:$dest,
                       Variadic<Index>:$indices);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = [{
    $scalar `into` $dest `[` $indices `]` attr-dict `:` type($dest)
  }];

  let extraClassDeclaration = [{
    std::pair<int64_t, int64_t> getDpsInitsPositionRange() {
      return {1, 2};  // `dest` operand
    }
  }];

  let hasFolder = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// InsertSliceOp
//===----------------------------------------------------------------------===//

def Tensor_InsertSliceOp : Tensor_OpWithOffsetSizesAndStrides<"insert_slice", [
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
    DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>,
    AttrSizedOperandSegments,
    DestinationStyleOpInterface,
    Pure,
    OffsetSizeAndStrideOpInterface,
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
    sentinel value ShapedType::kDynamic and
    ShapedType::kDynamic encodes that the corresponding entry has
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

    #### Verification in the rank-reduced case

    The same verification discussion and mechanisms apply as for ExtractSliceOp.
    Unlike ExtractSliceOp however, there is no need for a specific inference.

    Example:

    ```mlir
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
    DenseI64ArrayAttr:$static_offsets,
    DenseI64ArrayAttr:$static_sizes,
    DenseI64ArrayAttr:$static_strides
  );
  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $source `into` $dest ``
    custom<DynamicIndexList>($offsets, $static_offsets)
    custom<DynamicIndexList>($sizes, $static_sizes)
    custom<DynamicIndexList>($strides, $static_strides)
    attr-dict `:` type($source) `into` type($dest)
  }];

  let builders = [
    // Build a InsertSliceOp with mixed static and dynamic entries and inferred
    // result type.
    OpBuilder<(ins "Value":$source, "Value":$dest,
      "ArrayRef<OpFoldResult>":$offsets, "ArrayRef<OpFoldResult>":$sizes,
      "ArrayRef<OpFoldResult>":$strides,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
    // Build a InsertSliceOp with dynamic entries and inferred result type.
    OpBuilder<(ins "Value":$source, "Value":$dest,
      "ValueRange":$offsets, "ValueRange":$sizes, "ValueRange":$strides,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
    // Build an InsertSliceOp with mixed static and dynamic entries packed in
    // a Range vector and inferred result type.
    OpBuilder<(ins "Value":$source, "Value":$dest,
      "ArrayRef<Range>":$ranges,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>
  ];

  let extraClassDeclaration = extraBaseClassDeclaration # [{
    /// The result of a insert_slice is always a tensor.
    // TODO: Deprecate this method.
    RankedTensorType getType() {
      return getResultType();
    }

    /// The `dest` type is the same as the result type.
    RankedTensorType getDestType() {
      return getResultType();
    }

    /// Return the expected rank of each of the`static_offsets`, `static_sizes`
    /// and `static_strides` attributes.
    std::array<unsigned, 3> getArrayAttrMaxRanks() {
      unsigned rank = getResultType().getRank();
      return {rank, rank, rank};
    }

    /// Return the dimensions of the dest that are omitted to insert a source
    /// when the result is rank-extended.
    llvm::SmallBitVector getDroppedDims();

    /// Return the number of leading operands before the `offsets`, `sizes` and
    /// and `strides` operands.
    static unsigned getOffsetSizeAndStrideStartOperandIndex() { return 2; }

    std::pair<int64_t, int64_t> getDpsInitsPositionRange() {
      return {1, 2};  // `dest` operand
    }
  }];

  let hasCanonicalizer = 1;
  let hasFolder = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// RankOp
//===----------------------------------------------------------------------===//

def Tensor_RankOp : Tensor_Op<"rank", [
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
    Pure]> {
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

def Tensor_ReshapeOp: Tensor_Op<"reshape", [
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
    Pure]>  {
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
    Tensor_Op<mnemonic, !listconcat(traits, [
      DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
      Pure])>,
    Arguments<(ins AnyRankedTensor:$src, IndexListArrayAttr:$reassociation)>,
    Results<(outs AnyRankedTensor:$result)> {

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
    }
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
    represented with an array of DenseI64ArrayAttr attribute.

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

  let extraClassDeclaration = commonExtraClassDeclaration # [{
    int64_t getCorrespondingSourceDim(int64_t resultDim);
  }];

  let hasVerifier = 1;
}

def Tensor_CollapseShapeOp : Tensor_ReassociativeReshapeOp<"collapse_shape"> {
  let summary = "operation to produce a tensor with a smaller rank";
  let description = [{
    The `tensor.collapse_shape` op produces a new tensor with a smaller
    rank whose sizes are a reassociation of the original `src`.

    A reassociation is defined as a continuous grouping of dimensions and is
    represented with an array of DenseI64ArrayAttr attribute.

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

  let extraClassDeclaration = commonExtraClassDeclaration # [{  
    static RankedTensorType
    inferCollapsedType(RankedTensorType type, ArrayRef<AffineMap> reassociation);
    static RankedTensorType
    inferCollapsedType(RankedTensorType type, 
                       SmallVector<ReassociationIndices> reassociation);
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// PadOp
//===----------------------------------------------------------------------===//

def Tensor_PadOp : Tensor_Op<"pad", [
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
    AttrSizedOperandSegments,
    Pure,
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
    AnyRankedTensor:$source,
    Variadic<Index>:$low,
    Variadic<Index>:$high,
    DenseI64ArrayAttr:$static_low,
    DenseI64ArrayAttr:$static_high,
    UnitAttr:$nofold);

  let regions = (region SizedRegion<1>:$region);

  let results = (outs AnyRankedTensor:$result);

  // TODO: Remove custom<InferType> when AllTypesMatch supports opt. operands.
  let assemblyFormat = [{
    $source
    (`nofold` $nofold^)?
    `low` `` custom<DynamicIndexList>($low, $static_low)
    `high` `` custom<DynamicIndexList>($high, $static_high)
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
    inline SmallVector<OpFoldResult> getMixedPadImpl(ArrayRef<int64_t> staticAttrs,
                                                     ValueRange values) {
      Builder builder(*this);
      SmallVector<OpFoldResult> res;
      unsigned numDynamic = 0;
      unsigned count = staticAttrs.size();
      for (unsigned idx = 0; idx < count; ++idx) {
        if (ShapedType::isDynamic(staticAttrs[idx]))
          res.push_back(values[numDynamic++]);
        else
          res.push_back(builder.getI64IntegerAttr(staticAttrs[idx]));
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
    OpBuilder<(ins "Type":$resultType, "Value":$source,
      "ArrayRef<int64_t>":$staticLow, "ArrayRef<int64_t>":$staticHigh,
      "ValueRange":$low, "ValueRange":$high, CArg<"bool", "false">:$nofold,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
    // Build a PadOp with all dynamic entries.
    OpBuilder<(ins "Type":$resultType, "Value":$source, "ValueRange":$low,
      "ValueRange":$high, CArg<"bool", "false">:$nofold,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
    // Build a PadOp with mixed static and dynamic entries and custom
    // result type. If the type passed is nullptr, it is inferred.
    OpBuilder<(ins "Type":$resultType, "Value":$source,
      "ArrayRef<OpFoldResult>":$low, "ArrayRef<OpFoldResult>":$high,
      CArg<"bool", "false">:$nofold,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
    // Build a PadOp with constant padding,  mixed static and dynamic entries
    // and custom result type. If the type passed is nullptr, it is inferred.
    OpBuilder<(ins "Type":$resultType, "Value":$source,
      "ArrayRef<OpFoldResult>":$low, "ArrayRef<OpFoldResult>":$high,
      "Value":$constantPadValue, CArg<"bool", "false">:$nofold,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>
  ];

  let hasCanonicalizer = 1;
  let hasFolder = 1;
  let hasVerifier = 1;
  let hasRegionVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ParallelInsertSliceOp
//===----------------------------------------------------------------------===//

// TODO: Implement InParallelOpInterface.
def Tensor_ParallelInsertSliceOp : Tensor_Op<"parallel_insert_slice", [
       AttrSizedOperandSegments,
       OffsetSizeAndStrideOpInterface,
       // TODO: Cannot use an interface here atm, verify this manually for now.
       // HasParent<"ParallelCombiningOpInterface">
  ]> {
  let summary = [{
    Specify the tensor slice update of a single thread of a parent
    ParallelCombiningOpInterface op.
  }];
  let description = [{
    The `parallel_insert_slice` yields a subset tensor value to its parent
    ParallelCombiningOpInterface. These subset tensor values are aggregated to
    in some unspecified order into a full tensor value returned by the parent
    parallel iterating op.
    The `parallel_insert_slice` is one such op allowed in the
    ParallelCombiningOpInterface op.

    Conflicting writes result in undefined semantics, in that the indices written
    to by multiple parallel updates might contain data from any of the updates,
    or even a malformed bit pattern.

    If an index is updated exactly once, the value contained at that index
    in the resulting tensor will be equal to the value at a corresponding index
    of a slice that was used for the updated. If an index is not updated at all,
    its value will be equal to the one in the original tensor.

    This op does not create a new value, which allows maintaining a clean
    separation between the subset and full tensor.

    Note that we cannot mark this operation as pure (Pures), even
    though it has no side effects, because it will get DCEd during
    canonicalization.

    The parallel_insert_slice operation supports the following arguments:

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
    sentinel value ShapedType::kDynamic and
    ShapedType::kDynamic encodes that the corresponding entry has
    a dynamic value.

    After buffer allocation, the "parallel_insert_slice" op is expected to lower
    into a memref.subview op.

    A parallel_insert_slice operation may additionally specify insertion into a
    tensor of higher rank than the source tensor, along dimensions that are
    statically known to be of size 1.
    This rank-altering behavior is not required by the op semantics: this
    flexibility allows to progressively drop unit dimensions while lowering
    between different flavors of ops on that operate on tensors.
    The rank-altering behavior of tensor.parallel_insert_slice matches the
    rank-reducing behavior of tensor.insert_slice and tensor.extract_slice.

    #### Verification in the rank-reduced case

    The same verification discussion and mechanisms apply as for ExtractSliceOp.
    Unlike ExtractSliceOp however, there is no need for a specific inference.
  }];

  let arguments = (ins
    AnyRankedTensor:$source,
    AnyRankedTensor:$dest,
    Variadic<Index>:$offsets,
    Variadic<Index>:$sizes,
    Variadic<Index>:$strides,
    DenseI64ArrayAttr:$static_offsets,
    DenseI64ArrayAttr:$static_sizes,
    DenseI64ArrayAttr:$static_strides
  );
  let assemblyFormat = [{
    $source `into` $dest ``
    custom<DynamicIndexList>($offsets, $static_offsets)
    custom<DynamicIndexList>($sizes, $static_sizes)
    custom<DynamicIndexList>($strides, $static_strides)
    attr-dict `:` type($source) `into` type($dest)
  }];

  let extraClassDeclaration = [{
    Type yieldedType() { return getDest().getType(); }

    RankedTensorType getSourceType() {
      return getSource().getType().cast<RankedTensorType>();
    }

    RankedTensorType getDestType() {
      return getDest().getType().cast<RankedTensorType>();
    }

    ParallelCombiningOpInterface getParallelCombiningParent() {
      return dyn_cast<ParallelCombiningOpInterface>(
        getOperation()->getParentOp());
    }

    /// Return the expected rank of each of the `static_offsets`, `static_sizes`
    /// and `static_strides` attributes.
    std::array<unsigned, 3> getArrayAttrMaxRanks() {
      unsigned rank = getDestType().getRank();
      return {rank, rank, rank};
    }

    /// Return the number of leading operands before `offsets`, `sizes` and
    /// `strides` operands.
    static unsigned getOffsetSizeAndStrideStartOperandIndex() { return 1; }

    /// Return the OpResult of the enclosing ForallOp that is
    /// corresponding to this ParallelInsertSliceOp.
    OpResult getTiedOpResult();

    /// Return the dimensions of the dest that are omitted to insert a source
    /// when the result is rank-extended.
    llvm::SmallBitVector getDroppedDims();
  }];

  let builders = [
    // Build a ParallelInsertSliceOp with mixed static and dynamic entries.
    OpBuilder<(ins "Value":$source, "Value":$dest,
      "ArrayRef<OpFoldResult>":$offsets, "ArrayRef<OpFoldResult>":$sizes,
      "ArrayRef<OpFoldResult>":$strides,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
    // Build a ParallelInsertSliceOp with mixed static and dynamic entries
    // packed into a Range vector.
    OpBuilder<(ins "Value":$source, "Value":$dest,
      "ArrayRef<Range>":$ranges,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
    // Build a ParallelInsertSliceOp with dynamic entries.
    OpBuilder<(ins "Value":$source, "Value":$dest,
      "ValueRange":$offsets, "ValueRange":$sizes, "ValueRange":$strides,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>
  ];

  let hasCanonicalizer = 1;
  let hasVerifier = 1;
}


//===----------------------------------------------------------------------===//
// ScatterOp
//===----------------------------------------------------------------------===//

def Tensor_ScatterOp : Tensor_Op<"scatter", [
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
    Pure
  ]> {
  let summary =
    "scatter a tensor into a destination tensor at specified indices";
  let description = [{
    The `scatter` operation inserts a `source` tensor into a `dest` tensor at
    the given indices.

    In its most general form, the tensor of indices specifies all the coordinates
    of every element to insert (i.e. COO format, without the payload).
    The indices are expected to be confined to coordinate values that fit the
    range of the `dest` tensor, otherwise the behavior is undefined.

    The leading dimensions of the index tensor must match that of the dest
    tensor. The trailing dimensions of the dest tensor must match those of the
    source tensor by omitting the dimensions specified in scatter_dims
    (rank-reducing semantics) or setting them to `1` (rank-preserving semantics)
    (see examples).
    This convention allows an idiomatic specification and lowering of
    "scattering multiple N-D slices into the dest tensor".
    The result type must match the type of the dest tensor.

    Note: in the examples below, we separate out the indexing part of the tensor
    type by a whitespace for readability purposes.

    Example:

    ```mlir
        // For each 1x2 triple of coordinates in %indices, insert the
        // element (i.e. 0-D subset) at the coordinates triple in %dest.
        //
        %out = tensor.scatter %source into %dest[%indices]
            scatter_dims([0, 1, 2]) unique :
          (tensor<1x2x 1x1x1xf32>, tensor<4x4x4xf32>, tensor<1x2x 3xindex>)
            -> tensor<4x4x4xf32>

        // Note: source type may be further rank-reduced to tensor<1x2x f32>.
    ```

    A slice variant is provided to allow specifying insertion of whole tensor
    slices into the `dest` tensor.

    Example:

    ```mlir
        // For each 3 singleton of coordinates in %indices, insert the 2-D
        // slice into %dest[*, %indices[...]:%indices[...] + 1, *] with the
        // indices corresponding to the scatter_dims attribute specified by
        // %indices.
        //
        %out = tensor.scatter %source into %dest[%indices] scatter_dims([1]) unique :
          (tensor<3x 4x1x6xf32>, tensor<4x5x6xf32>, tensor<3x 1xindex>)
            -> tensor<4x5x6xf32>
    ```

    The dimensions specified in the scatter_dims attribute are ones for which the
    source tensor has size `1`.
    I.e. if the dest type is `axbxcxd` and the coordinates are [1, 3], then
    the source type suffix is `ax1xcx1`.
    Sactter also allows rank-reducing semantics where the shape `ax1xcx1` can be
    further simplified to `axc`.

    The elemental type of the indices tensor can be any integer type.
    In the absence of target-specific or problem specific information the default
    type one should use is `index`.

    This operation does not support unranked tensors.

    A `unique` unit attribute must be be specified to indicate that the
    coordinates are statically guaranteed to be unique at runtime. If coordinates
    are not truly unique at runtime, the behavior is undefined.

    Only full slices are meant to be supported by this op, if one desires
    partial slices (e.g. strided windows) one should compose this op with other
    tensor ops (e.g. tensor.insert_slice). This is to avoid a slippery slope of
    complexity that would make the op unusable in practice.

    At the tensor-level, the index tensor is specified in an AoS form (i.e.
    coordinate tuple is the most minor). It is the responsibility of further
    lowerings and bufferiation to implement various concrete layouts.

    Note: As currently specified, the operation must lower to an abstraction that
    performs copies to the output tensor. This is because the buffer type system
    is currently not rich enough to allow multiple non-contiguous views in the
    same type. This is visible more clearly in a notional buffer version of the
    op:

    ```mlir
        // memref<?x 4xf32> is a contiguous buffer of ?x4 elements, scatter into
        // random dest slices must copy to the contiguous dest.
        //
        some_side_effecting_op_writing_into %source, ...: memref<3x 4xf32>
        memref.scatter %source into %dest[%indices] scatter_dims([1]) unique :
          (memref<3x 4xf32>, memref<?x 4xf32>, memref<?x 1xindex>)

        // Nested buffer support in the producing op would allow writing directly
        // into the dest buffer.
        %v = some_nested_buffer_view_op %dest[%indices] scatter_dims([1]) unique :
          memref<? x memref<4xf32>>
        some_side_effecting_op_writing_into %v, ...: memref<? x memref<4xf32>>
    ```
  }];

  let arguments = (ins AnyRankedTensor:$source,
                       AnyRankedTensor:$dest,
                       RankedTensorOf<[AnySignlessIntegerOrIndex]>:$indices,
                       DenseI64ArrayAttr:$scatter_dims,
                       UnitAttr:$unique);
  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $source `into` $dest `[` $indices `]`
      `scatter_dims` `(` $scatter_dims `)`
      (`unique` $unique^)?
      attr-dict
    `:` functional-type(operands, results)
  }];

  let extraClassDeclaration = [{
    RankedTensorType getDestType() {
      return getDest().getType().cast<RankedTensorType>();
    }
    RankedTensorType getIndicesType() {
      return getIndices().getType().cast<RankedTensorType>();
    }
    RankedTensorType getSourceType() {
      return getSource().getType().cast<RankedTensorType>();
    }
    RankedTensorType getResultType() {
      return getResult().getType().cast<RankedTensorType>();
    }
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// SplatOp
//===----------------------------------------------------------------------===//

def Tensor_SplatOp : Tensor_Op<"splat", [
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
    Pure,
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
    %t = tensor.splat %s : tensor<8x16xf32>
    ```

    TODO: This operation is easy to extend to broadcast to dynamically shaped
          tensors:

    ```mlir
    // Broadcasts %s to a 2-d dynamically shaped tensor, with %m, %n binding
    // to the sizes of the two dynamic dimensions.
    %m = "foo"() : () -> (index)
    %n = "bar"() : () -> (index)
    %t = tensor.splat %s [%m, %n] : tensor<?x?xf32>
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
// PackOp
//===----------------------------------------------------------------------===//

class Tensor_RelayoutOp<string mnemonic, list<Trait> traits = []> :
      Tensor_Op<mnemonic, !listconcat(traits, [
        DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
        DestinationStyleOpInterface,
        ConditionallySpeculatable, NoMemoryEffect,
        DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>,
        TypesMatchWith<"result type matches type of dest",
                   "dest", "result",
                   "$_self">])> {

  code commonExtraClassDeclaration = [{
    size_t getSourceRank() { return getSourceType().getRank(); };
    size_t getDestRank() { return getDestType().getRank(); };
    RankedTensorType getSourceType() {
      return getSource().getType().cast<RankedTensorType>(); };
    RankedTensorType getDestType() {
      return getDest().getType().cast<RankedTensorType>(); };

    /// Return position for init operand. Init operand is `dest`.
    std::pair<int64_t, int64_t> getDpsInitsPositionRange() {
      return {1, 2}; // `dest` operand
    }

    /// Interface method for ConditionallySpeculatable.
    Speculation::Speculatability getSpeculatability();

    /// Return a mapping from positions `inner_dims_pos` to their
    /// tile factors.
    DenseMap<int64_t, OpFoldResult> getDimAndTileMapping();

    /// Return the tile sizes as OpFoldResult.
    SmallVector<OpFoldResult> getMixedTiles();

    /// Return the tile sizes as `int64_t`. If a tile size is dynamic
    /// a sentinel `kDynamic` is introduced at that position in
    /// the returned vector.
    SmallVector<int64_t> getStaticTiles();
  }];

  let hasVerifier = 1;
}

def Tensor_PackOp : Tensor_RelayoutOp<"pack", [
    AttrSizedOperandSegments]> {
  let summary = "tensor pack operation";
  let description = [{
    The pack operation converts an input tensor to a higher-dimensional tensor
    with a tiled and packed layout. The mandatory `inner_dims_pos` attribute
    specifies a permutation for the original dimensions, while `inner_tiles` is the
    tiling factor for each dimension. The optional attribute `outer_dims_perm`
    specifies the order for the tiled data dimension, while the attribute
    `padding_value` specifies a padding value at the boundary on non-perfectly
    divisible dimensions. Padding is optional:
    - If absent, it is UB if the tile does not perfectly divide the dimension.
    - If present, it will pad along high dimensions (high-padding) to make the
      tile complete.

    Example NC_to_NCnc:

    ```mlir
    %0 = tensor.pack %source inner_dims_pos = [0, 1]
      inner_tiles = [8, 32] into %dest : tensor<128x256xf32> -> tensor<16x8x8x32xf32>
    ```
    Example CK to KCck

    ```mlir
    %0 = tensor.pack %source outer_dims_perm = [1, 0] inner_dims_pos = [0, 1]
      inner_tiles = [8, 32] into %dest : tensor<128x256xf32> -> tensor<8x16x8x32xf32>
    ```

    In all cases, dimension at position 0 in the input tensor (128) is tiled
    with a factor of 8, while dimension at position 1 (256) is tiled with a factor
    of 32. In the second example, the outer data dimensions are interchanged
    according to `outer_dims_perm`.

    Example NC_to_NCnc with padding:

    ```mlir
    %0 = tensor.pack %arg padding_value(%pad : f32) inner_dims_pos = [0, 1]
      inner_tiles = [8, 2] into %arg1 : tensor<13x15xf32> -> tensor<2x8x8x2xf32>
    ```

  }];
  let arguments = (ins AnyRankedTensor:$source,
                       AnyRankedTensor:$dest,
                       Optional<AnyType>:$padding_value,
                       DefaultValuedOptionalAttr<DenseI64ArrayAttr, "{}">:$outer_dims_perm,
                       DenseI64ArrayAttr:$inner_dims_pos,
                       Variadic<Index>:$inner_tiles,
                       DenseI64ArrayAttr:$static_inner_tiles);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = [{
    $source
    (`padding_value` `(` $padding_value^ `:` type($padding_value) `)`)?
    (`outer_dims_perm` `=` $outer_dims_perm^)?
    `inner_dims_pos` `=` $inner_dims_pos
    `inner_tiles` `=`
    custom<DynamicIndexList>($inner_tiles, $static_inner_tiles)
    `into` $dest attr-dict `:` type($source) `->` type($dest)
  }];

  let builders = [
    OpBuilder<(ins "Value":$source, "Value":$dest,
      "ArrayRef<int64_t>":$innerDimsPos,
      "ArrayRef<OpFoldResult>":$innerTiles,
      CArg<"std::optional<Value>", "std::nullopt">:$paddingValue,
      CArg<"ArrayRef<int64_t>", "{}">:$outerDimsPerm)>
  ];

  let extraClassDeclaration = commonExtraClassDeclaration # [{
    // Method to get the shape of the result as `SmallVector<OpFoldResult>`.
    // This is a static method to allow getting the shape of the destination
    // expected while creating a `pack` op.
    static SmallVector<OpFoldResult> getResultShape(OpBuilder &builder,
        Location loc, ArrayRef<OpFoldResult> sourceDims,
        ArrayRef<OpFoldResult> innerTileDims, ArrayRef<int64_t> innerDimsPos,
        ArrayRef<int64_t> outerDimsPerm = {});

    // Method to get the `ShapedType` of the result based on the inner tiles,
    // position of the inner tiles (innerDimsPos)  and interchange vector of
    // outer loops (outerDimsPerm).
    static ShapedType inferPackedType(ShapedType sourceType,
        ArrayRef<int64_t> innerTileSizes, ArrayRef<int64_t> innerDimsPos,
        ArrayRef<int64_t> outerDimsPerm = {});

    // Returns true if we have enough static information to catch undefined
    // behavior when the tile size does not divide perfectly the dimension of
    // the input tensor.
    static bool requirePaddingValue(ArrayRef<int64_t> inputShape,
                                    ArrayRef<int64_t> innerDimsPos,
                                    ArrayRef<OpFoldResult> innerTiles);

    static Value createDestinationTensor(OpBuilder &b, Location loc,
        Value source, ArrayRef<OpFoldResult> innerTileSizes,
        ArrayRef<int64_t> innerDimsPos, ArrayRef<int64_t> outerDimsPerm);

    /// Build and return a new PackOp that is a clone of the current PackOp with
    /// (innerDimsPos, innerTiles) (resp. outerDimsPerm) are permuted by 
    /// innerPermutation (resp. outerPermutation).
    /// A new `tensor.empty` of the proper shape is built in the process.
    /// Asserts that:
    ///   - At least one of innerPermutation or outerPermutation is non-empty.
    ///   - If not empty, innerPermutation is a valid permutation of size
    ///     matching innerDimPos.
    ///   - If not empty, outerPermutation is a valid permutation of size 
    ///     matching outerDimsPerm.
    PackOp createTransposedClone(OpBuilder &b,
                                 Location loc,
                                 ArrayRef<int64_t> innerPermutation,
                                 ArrayRef<int64_t> outerPermutation);

    /// Check if this PackOp is like a simple pad operation.
    /// In other words, this operation:
    /// 1. adds useless dimensions (dimension of size 1),
    /// 2. pads the other ones, and
    /// 3. doesn't shuffle the dimensions
    bool isLikePad();
  }];

  let hasCanonicalizeMethod = 1;
}

//===----------------------------------------------------------------------===//
// UnPackOp
//===----------------------------------------------------------------------===//

def Tensor_UnPackOp : Tensor_RelayoutOp<"unpack"> {
  let summary = "tensor unpack operation";
  let description = [{
    The unpack operation converts a tensor with a tiled and packed layout to a
    lower-dimensional tensor. Similar to `pack`,  the mandatory attributes
    `inner_dims_pos` specifies a permutation for the inner data dimensions, while
    `inner_tiles` is the tiling factor. The attribute `outer_dims_perm` has the
    exact behavior as the one described in `pack`. In `unpack`, it is UB if the
    tile does not perfectly divide the dimension.

    Example NCnc_to_NC:

    ```mlir
    %0 = tensor.unpack %source inner_dims_pos = [0, 1]
      inner_tiles = [8, 32] into %dest : tensor<16x8x8x32xf32> -> tensor<128x256xf32>
    ```

    Example CK to KCck:

    ```mlir
    %0 = tensor.unapck %source outer_dims_perm = [1, 0] inner_dims_pos = [0, 1]
      inner_tiles = [8, 32] into %dest : tensor<8x16x8x32xf32> -> tensor<128x256xf32>
    ```
  }];
  let arguments = (ins AnyRankedTensor:$source,
                       AnyRankedTensor:$dest,
                       DefaultValuedOptionalAttr<DenseI64ArrayAttr, "{}">:$outer_dims_perm,
                       DenseI64ArrayAttr:$inner_dims_pos,
                       Variadic<Index>:$inner_tiles,
                       DenseI64ArrayAttr:$static_inner_tiles);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = [{
    $source
    (`outer_dims_perm` `=` $outer_dims_perm^)?
    `inner_dims_pos` `=` $inner_dims_pos
    `inner_tiles` `=`
    custom<DynamicIndexList>($inner_tiles, $static_inner_tiles)
    `into` $dest attr-dict `:` type($source) `->` type($dest)
  }];

  let builders = [
    OpBuilder<(ins "Value":$source, "Value":$dest,
    "ArrayRef<int64_t>":$innerDimsPos,
    "ArrayRef<OpFoldResult>":$innerTiles,
    CArg<"ArrayRef<int64_t>", "{}">:$outerDimsPerm)>
  ];

  let extraClassDeclaration = commonExtraClassDeclaration # [{
    static Value createDestinationTensor(OpBuilder &b, Location loc,
        Value source, ArrayRef<OpFoldResult> innerTileSizes,
        ArrayRef<int64_t> innerDimsPos, ArrayRef<int64_t> outerDimsPerm);

    /// Build and return a new UnPackOp that is a clone of the current UnPackOp
    /// with (innerDimsPos, innerTiles) (resp. outerDimsPerm) are permuted by 
    /// innerPermutation (resp. outerPermutation).
    /// Asserts that:
    ///   - At least one of innerPermutation or outerPermutation is non-empty.
    ///   - If not empty, innerPermutation is a valid permutation of size
    ///     matching innerDimPos.
    ///   - If not empty, outerPermutation is a valid permutation of size 
    ///     matching outerDimsPerm.
    UnPackOp createTransposedClone(OpBuilder &b,
                                   Location loc,
                                   Value transposedSource,
                                   ArrayRef<int64_t> innerPermutation,
                                   ArrayRef<int64_t> outerPermutation);

    /// Check if this UnPackOp is like a simple unpad operation.
    /// In other words, this operation:
    /// 1. drops useless dimensions (dimension of size 1), and
    /// 2. reduces dimensions in place (i.e., no tranpose.)
    bool isLikeUnPad();
  }];

  let hasCanonicalizeMethod = 1;
}

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

def Tensor_YieldOp : Tensor_Op<"yield",
    [Pure, ReturnLike, Terminator,
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
