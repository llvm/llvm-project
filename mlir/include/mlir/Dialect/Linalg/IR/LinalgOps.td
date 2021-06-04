//===- LinalgOps.td - Linalg dialect ops -------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the operation definition file for linear algebra operations.
//
//===----------------------------------------------------------------------===//

#ifndef LINALG_OPS
#define LINALG_OPS

include "mlir/Dialect/Linalg/IR/LinalgBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"

// Base class for Linalg dialect ops that do not correspond to library calls.
class Linalg_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<Linalg_Dialect, mnemonic, traits> {
  // For every linalg op, there needs to be a:
  //   * void print(OpAsmPrinter &p, ${C++ class of Op} op)
  //   * LogicalResult verify(${C++ class of Op} op)
  //   * ParseResult parse${C++ class of Op}(OpAsmParser &parser,
  //                                         OperationState &result)
  // functions.
  let printer = [{ return ::print(p, *this); }];
  let verifier = [{ return ::verify(*this); }];
  let parser = [{ return ::parse$cppClass(parser, result); }];
}

def Linalg_InitTensorOp : Linalg_Op<"init_tensor",
    [NoSideEffect,
     DeclareOpInterfaceMethods<InferShapedTypeOpInterface,
         ["reifyReturnTypeShapesPerResultDim"]>]> {
  let summary = "operation to define a tensor of particular value";

  let description = [{
    `linalg.init_tensor` is an operation that materializes a tensor of
    a given shape. The shape could be dynamic or static.
  }];

  let arguments =
    (ins Variadic<Index>:$sizes, I64ArrayAttr:$static_sizes);

  let results = (outs AnyTensor:$result);

  let assemblyFormat = [{
    custom<OperandsOrIntegersSizesList>($sizes, $static_sizes) attr-dict
    `:` type($result)
  }];

  let verifier = [{ return ::verify(*this); }];

  let extraClassDeclaration = [{
    static StringRef getStaticSizesAttrName() {
      return "static_sizes";
    }

    RankedTensorType getType() {
      return getResult().getType().cast<RankedTensorType>(); }

    // Infer the shape of the result tensor given the static shapes
    // and element type of the result tensor.
    static Type inferResultType(ArrayRef<int64_t> staticSizes, Type elementType);

    // Return true if the size of the tensor is dynamic at `idx`
    bool isDynamicSize(unsigned idx) {
      APInt v = *(static_sizes().getAsValueRange<IntegerAttr>().begin() + idx);
      return ShapedType::isDynamic(v.getSExtValue());
    }

    // Assert that the size of the result tensor is static at `idx`
    // and return the shape.
    int64_t getStaticSize(unsigned idx) {
      assert(!isDynamicSize(idx) && "expected static size");
      APInt v = *(static_sizes().
          template getAsValueRange<IntegerAttr>().begin() + idx);
        return v.getSExtValue();
    }

    // Return the argument position that contains the dynamic size of
    // the tensor at dimension `idx`. Asserts that the shape is
    // dynamic at that `idx`.
    unsigned getIndexOfDynamicSize(unsigned idx) {
      assert(isDynamicSize(idx) && "expected dynamic size");
      return std::count_if(
          static_sizes().getValue().begin(),
          static_sizes().getValue().begin() + idx,
          [&](Attribute attr) {
            return ShapedType::isDynamic(attr.cast<IntegerAttr>().getInt());
          });
    }

    // Return the Value of the dynamic size of the tensor at dimension
    // `idx`. Asserts that the shape is dynamic at that `idx.
    Value getDynamicSize(unsigned idx) {
      return getOperand(getIndexOfDynamicSize(idx));
    }
  }];

  let builders = [
    OpBuilder<(ins "ValueRange":$shape,
                  "ArrayRef<int64_t>":$staticShape, "Type":$elementType),
    [{
      build($_builder, $_state,
            InitTensorOp::inferResultType(staticShape, elementType),
            shape, $_builder.getI64ArrayAttr(staticShape));
    }]>,
    OpBuilder<(ins "ValueRange":$shape, "Type":$elementType),
    [{
      SmallVector<int64_t, 4> staticShape(
        shape.size(), ShapedType::kDynamicSize);
      build($_builder, $_state, shape, staticShape, elementType);
    }]>,
    OpBuilder<(ins "ArrayRef<int64_t>":$staticShape, "Type":$elementType),
    [{
      build($_builder, $_state, ValueRange{}, staticShape, elementType);
    }]>,
    OpBuilder<(ins "ArrayRef<OpFoldResult>":$sizes, "Type":$elementType,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>
  ];

  let hasCanonicalizer = 1;
}

def Linalg_PadTensorOp : Linalg_Op<"pad_tensor",
    [AttrSizedOperandSegments,
     DeclareOpInterfaceMethods<InferShapedTypeOpInterface,
         ["reifyReturnTypeShapesPerResultDim"]>,
     NoSideEffect]> {
  let summary = "tensor pad operation";
  let description = [{
    `linalg.pad_tensor` is an operation that pads the `source` tensor
    with given `low` and `high` padding config.

    The PadTensor operation supports the following arguments:

    * source: the "base" tensor on which to pad.
    * low: A list contains the padding along the start of each
           dimension, i.e `low`.
    * high: A list contains the padding along the end of each
           dimension, i.e. `high`.

    The result tensor dimensions are `low` + `dim` + `high` along that
    dimension. The number of elements of `low` and `high` must match
    the rank of the input tensor (which is also the rank of the output
    tensor). They can be either a constant or a dynamic value.

    The region of the `pad_tensor` operation returns the value to use
    for the padding. The arguments of the region represent the index
    of the source being accessed. There should be as many arguments as
    the rank of the `source` tensor. The value `yield`-ed by the
    region is used as the value of the view at the given position.

    Example 1:

    ```mlir
      %pad_value = ... : f32
      %0 = linalg.pad_tensor %0 low[1, 2] high[2, 3] {
      ^bb0(%arg0 : index, %arg1 : index):
        linalg.yield %pad_value : f32
      } : tensor<?x?xf32> to tensor<?x?xf32>
    ```

    Example 2:

    ```mlir
      %pad_value = ... : f32
      %0 = linalg.pad_tensor %arg0 low[2, %arg1, 3, 3] high[3, 3, %arg1, 2] {
      ^bb0(%arg2: index, %arg3: index, %arg4: index, %arg5: index):
          linalg.yield %pad_value : f32
      } : tensor<1x2x2x?xf32> to tensor<6x?x?x?xf32>
    ```

    Example 3:

    ```mlir
      %pad_value = ... : f32
      %0 = linalg.pad_tensor %arg0 low[0, 0] high[%ub0, %ub1] {
      ^bb0(%arg1: index, %arg2: index):
        linalg.yield %pad_value : f32
      } : tensor<2x3xf32> to tensor<?x?xf32>
    ```
  }];

  let arguments = (ins
    AnyTensor:$source,
    Variadic<Index>:$low,
    Variadic<Index>:$high,
    I64ArrayAttr:$static_low,
    I64ArrayAttr:$static_high);

  let regions = (region SizedRegion<1>:$region);

  let results = (outs AnyTensor:$result);

  let assemblyFormat = [{
    $source `low` `` custom<OperandsOrIntegersSizesList>($low, $static_low)
    `high` `` custom<OperandsOrIntegersSizesList>($high, $static_high)
    $region attr-dict `:` type($source) `to` type($result)
  }];

  let extraClassDeclaration = [{
    static StringRef getStaticLowAttrName() {
      return "static_low";
    }

    static StringRef getStaticHighAttrName() {
      return "static_high";
    }

    RankedTensorType getSourceType() {
      return source().getType().cast<RankedTensorType>();
    }
    RankedTensorType getResultType() {
      return getResult().getType().cast<RankedTensorType>();
    }

    // Infer the shape of the result tensor given the static shapes
    // and element type of the result tensor.
    static RankedTensorType inferResultType(RankedTensorType sourceType,
                                ArrayRef<int64_t> staticLow,
                                ArrayRef<int64_t> staticHigh);

    // Return a PadTensorOp that pads `source` to `type` size where the static
    // sizes are assumed to be greater than the dynamic sizes. The op performs
    // "high" padding (i.e. it adds trailing padding values until the desired
    // size is met).
    static linalg::PadTensorOp createPadHighOp(
        Type type, Value source, Value pad, Location loc, OpBuilder & builder);

    // Return a PadTensorOp that pads `source to `type` size with `pad` value.
    // I.e., a block will be created and the `pad` value will be yielded
    // directly. If the type passed is nullptr, it is inferred.
    static linalg::PadTensorOp createPadScalarOp(
        Type type, Value source, Value pad, ArrayRef<OpFoldResult> low,
        ArrayRef<OpFoldResult> high, Location loc, OpBuilder & builder);

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
      return getMixedPadImpl(static_low(), low());
    }
    SmallVector<OpFoldResult> getMixedHighPad() {
      return getMixedPadImpl(static_high(), high());
    }
  }];

  let builders = [
    // Build a PadTensorOp with mixed static and dynamic entries.
    OpBuilder<(ins "Value":$source, "ArrayRef<int64_t>":$staticLow,
      "ArrayRef<int64_t>":$staticHigh, "ValueRange":$low, "ValueRange":$high,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
    // Build a PadTensorOp with all dynamic entries.
    OpBuilder<(ins "Value":$source, "ValueRange":$low, "ValueRange":$high,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
    // Build a PadTensorOp with mixed static and dynamic entries and custom
    // result type. If the type passed is nullptr, it is inferred.
    OpBuilder<(ins "Type":$resultType, "Value":$source,
      "ArrayRef<OpFoldResult>":$low, "ArrayRef<OpFoldResult>":$high,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>
  ];
}

def Linalg_RangeOp :
    Linalg_Op<"range", [NoSideEffect]>,
    Arguments<(ins Index:$min, Index:$max, Index:$step)>,
    Results<(outs Range)> {
  let summary = "Create a `range` type value, used to create `view`s";
  let description = [{
    The `linalg.range` op creates a `!linalg.range` from 3 values of type
    `index` that represent the min, max and step values of the `range`. This
    type does not pass function boundaries at the moment.

    Example:

    ```mlir
    %3 = linalg.range %0:%1:%2 : !linalg.range
    ````
  }];
  let builders = [
    OpBuilder<(ins "Value":$min, "Value":$max, "Value":$step),
    [{
      auto rangeType = RangeType::get($_builder.getContext());
      build($_builder, $_state, rangeType, min, max, step);
    }]>];

  // Fully specified by traits.
  let verifier = ?;
  let assemblyFormat = "$min `:` $max `:` $step attr-dict `:` type(results)";
}

class Linalg_ReshapeLikeOp<string mnemonic, list<OpTrait> traits = []> :
    Linalg_Op<mnemonic, !listconcat(traits, [NoSideEffect])> {
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

    // Builders for a reshape whose result type is passed explicitly. This may
    // be either a contracting or expanding reshape.
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

  code commonExtraClassDeclaration = [{
    static StringRef getReassociationAttrName() { return "reassociation"; }
    SmallVector<AffineMap, 4> getReassociationMaps();
    SmallVector<ReassociationExprs, 4> getReassociationExprs();
    SmallVector<ReassociationIndices, 4> getReassociationIndices() {
      SmallVector<ReassociationIndices, 4> reassociationIndices;
      for (auto attr : reassociation())
        reassociationIndices.push_back(llvm::to_vector<2>(
            llvm::map_range(attr.cast<ArrayAttr>(), [&](Attribute indexAttr) {
              return indexAttr.cast<IntegerAttr>().getInt();
            })));
      return reassociationIndices;
    };
  }];

  let parser = [{ return ::parseReshapeLikeOp(parser, result); }];
}

def IndexListArrayAttr :
  TypedArrayAttrBase<I64ArrayAttr, "Array of 64-bit integer array attributes">;

class Linalg_ReshapeOp<string mnemonic> : Linalg_ReshapeLikeOp<mnemonic,
    [DeclareOpInterfaceMethods<ViewLikeOpInterface>]>,
    Arguments<(ins AnyStridedMemRef:$src, IndexListArrayAttr:$reassociation)>,
    Results<(outs AnyStridedMemRef:$result)> {
  let extraClassDeclaration = commonExtraClassDeclaration # [{
    MemRefType getSrcType() { return src().getType().cast<MemRefType>(); }
    MemRefType getResultType() { return result().getType().cast<MemRefType>(); }
  }];
  let hasFolder = 1;
  let hasCanonicalizer = 1;
  let printer = [{ return ::print(p, *this); }];
}

def Linalg_ExpandShapeOp : Linalg_ReshapeOp<"expand_shape"> {
  let summary = "operation to produce a memref with a higher rank.";
  let description = [{
    The `linalg.expand_shape` op produces a new view with a higher rank whose
    sizes are a reassociation of the original `view`. Depending on whether or
    not the reassociated MemRefType is contiguous, the resulting memref may
    require explicit alloc and copies.

    A reassociation is defined as a continuous grouping of dimensions and is
    represented with an array of I64ArrayAttr attribute.

    For now, it is assumed that either:
      1. a reassociation produces and consumes contiguous MemRefType or,
      2. the reshape op will be folded into its consumers (by changing the shape
         of the computations).
    All other cases are undefined behavior and a reshape op may not lower to
    LLVM if it cannot be proven statically that it does not require alloc+copy.

    The operand memref type when dimensions can be zero-ranked if the result
    memref type is statically shaped with all dimensions being unit extent. In
    such case the reassociation map is empty.

    The verification rule is that the reassociation maps are applied to the
    result memref with the larger rank to obtain the operand memref with the
    smaller rank.

    Example:

    ```mlir
    // Dimension expansion i -> (i', j') and (k) -> (k')
    %1 = linalg.expand_shape %0 [[0, 1], [2]] :
      memref<?x?xf32, stride_spec> into memref<?x?x?xf32, stride_spec_2>
    ```
  }];
}

def Linalg_CollapseShapeOp : Linalg_ReshapeOp<"collapse_shape"> {
  let summary = "operation to produce a memref with a smaller rank.";
  let description = [{
    The `linalg.collapse_shape` op produces a new view with a smaller rank
    whose sizes are a reassociation of the original `view`. Depending on
    whether or not the reassociated MemRefType is contiguous, the resulting
    memref may require explicit alloc and copies.

    A reassociation is defined as a continuous grouping of dimensions and is
    represented with an array of I64ArrayAttr attribute.

    For now, it is assumed that either:
      1. a reassociation produces and consumes contiguous MemRefType or,
      2. the reshape op will be folded into its consumers (by changing the shape
         of the computations).
    All other cases are undefined behavior and a reshape op may not lower to
    LLVM if it cannot be proven statically that it does not require alloc+copy.

    The result memref type of a reshape can be zero-ranked if the operand
    memref type is statically shaped with all dimensions being unit extent. In
    such case the reassociation map is empty.

    The verification rule is that the reassociation maps are applied to the
    operand memref with the larger rank to obtain the result memref with the
    smaller rank.

    Examples:

    ```mlir
    // Dimension collapse (i, j) -> i' and k -> k'
    %1 = linalg.collapse_shape %0 [[0, 1], [2]] :
      memref<?x?x?xf32, stride_spec> into memref<?x?xf32, stride_spec_2>
    ```
  }];
}

class Linalg_TensorReshapeOp<string mnemonic> : Linalg_ReshapeLikeOp<
    mnemonic,
    [DeclareOpInterfaceMethods<InferShapedTypeOpInterface,
      ["reifyReturnTypeShapesPerResultDim"]>]>,
    Arguments<(ins AnyTensor:$src,
                   IndexListArrayAttr:$reassociation)>,
    Results<(outs AnyTensor:$result)> {
  let extraClassDeclaration = commonExtraClassDeclaration # [{
    RankedTensorType getSrcType() {
      return src().getType().cast<RankedTensorType>();
    }
    RankedTensorType getResultType() {
      return result().getType().cast<RankedTensorType>();
    }
  }];
  let hasFolder = 1;
  let hasCanonicalizer = 1;
  let printer = [{ return ::print(p, *this); }];
  let parser = [{ return ::parseReshapeLikeOp(parser, result); }];
}

def Linalg_TensorExpandShapeOp : Linalg_TensorReshapeOp<"tensor_expand_shape"> {
  let summary = "operation to produce a tensor with a higher rank";
  let description = [{
    The `linalg.tensor_expand_shape` op produces a new tensor with a higher
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
    %b = linalg.tensor_expand_shape %a [[0, 1], [2]]
        : tensor<?x?xf32> into tensor<?x?x?xf32>
    ```
  }];
}

def Linalg_TensorCollapseShapeOp : Linalg_TensorReshapeOp<"tensor_collapse_shape"> {
  let summary = "operation to produce a tensor with a smaller rank";
  let description = [{
    The `linalg.tensor_collapse_shape` op produces a new tensor with a smaller
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
    %b = linalg.tensor_collapse_shape %a [[0, 1], [2]]
        : tensor<?x?x?xf32> into tensor<?x?xf32>
    ```
  }];
}

def Linalg_YieldOp : Linalg_Op<"yield", [NoSideEffect, ReturnLike, Terminator]>,
    Arguments<(ins Variadic<AnyType>:$values)> {
  let summary = "Linalg yield operation";
  let description = [{
    `linalg.yield` is a special terminator operation for blocks inside regions
    in `linalg` generic ops. It returns values to the immediately enclosing
    `linalg` generic op.

    Example:

    ```mlir
    linalg.yield %f0, %f1 : f32, f32
    ```
  }];
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];
}

def Linalg_TiledLoopOp : Linalg_Op<"tiled_loop", [
     AttrSizedOperandSegments,
     DeclareOpInterfaceMethods<LoopLikeOpInterface>,
     RecursiveSideEffects,
     SingleBlockImplicitTerminator<"linalg::YieldOp">
    ]> {
  let summary = "Linalg tiled loop operation";
  let description = [{
    This is a loop-like operation with additional properties. The arguments
    also include the input and the output tensors or memrefs and the attributes
    to specify the iterator types.

    Parsing TiledLoopOp will set all elements of the `iterator_types` attribute
    to "parallel" type, when it is absent from the custom format.

    Tensor-based version:

    The body region of the loop contains `subtensor` operations applied to
    every tensor argument of TiledLoopOp.

    The body region must contain exactly one block that terminates with
    `linalg.yield` with the operands resulting from `subtensor_insert`
    operations.

    Example:

    ```mlir
    %0 = linalg.tiled_loop (%i) = (%c0) to (%c24) step (%c4)
        ins(%lhs, %rhs : tensor<24x64xi8>, tensor<24x64xi8>)
        outs(%out : tensor<24x64xi8>)
        iterators("parallel")
        distribution("block_x") {
      %lhs_sub = subtensor %lhs[%i, 0] [%c4, %c64] [1, 1]
          : tensor<24x64xi8> to tensor<?x?xi8>
      %rhs_sub = subtensor %rhs[%i, 0] [%c4, %c64] [1, 1]
          : tensor<24x64xi8> to tensor<?x?xi8>
      %out_sub = subtensor %out[%i, 0] [%c4, %c64] [1, 1]
          : tensor<24x64xi8> to tensor<?x?xi8>

      %result_sub = linalg.generic ...

      %result = subtensor_insert %result_sub into %out[%i, 0][%c4, %c64][1, 1]
        : tensor<?x?xi8> into tensor<24x64xi8>
      linalg.yield %result : tensor<24x64xi8>
    }
    ```

    MemRef-based version:

    The body region of the loop contains `subview` operations applied to
    every memref argument of TiledLoopOp.

    The body region must contain exactly one block that terminates with
    `linalg.yield` with no operands.

    Example:

    ```mlir
    linalg.tiled_loop (%i) = (%c0) to (%c24) step (%c4)
        ins(%lhs, %rhs : memref<24x64xi8>, memref<24x64xi8>)
        outs(%out : memref<24x64xi8>)
        iterators("parallel")
        distribution("block_x") {
      %lhs_sub = subview %lhs[%i, 0] [%c4, %c64] [1, 1]
          : memref<24x64xi8> to memref<?x?xi8>
      %rhs_sub = subview %rhs[%i, 0] [%c4, %c64] [1, 1]
          : memref<24x64xi8> to memref<?x?xi8>
      %out_sub = subview %out[%i, 0] [%c4, %c64] [1, 1]
          : memref<24x64xi8> to memref<?x?xi8>

      %result_sub = linalg.generic ...
      linalg.yield
    }
    ```
  }];

  let arguments = (ins Variadic<Index>:$lowerBound,
                       Variadic<Index>:$upperBound,
                       Variadic<Index>:$step,
                       Variadic<LinalgOperand>:$inputs,
                       Variadic<LinalgOperand>:$outputs,
                       ArrayAttr:$iterator_types,
                       OptionalAttr<ArrayAttr>:$distribution_types);
  let results = (outs Variadic<AnyRankedTensor>:$results);
  let regions = (region SizedRegion<1>:$region);

  let builders = [
    OpBuilder<(ins "ValueRange":$lowerBounds, "ValueRange":$upperBounds,
      "ValueRange":$steps, "ValueRange":$inputs, "ValueRange":$outputs,
      "ArrayAttr":$iteratorTypes, "Optional<ArrayAttr>":$distributionTypes,
      CArg<"function_ref<void (OpBuilder &, Location, /*ivs=*/ValueRange,"
        "/*inputs=*/ValueRange, /*outputs=*/ValueRange)>",
        "nullptr">:$bodyBuilderFn)>,
    OpBuilder<(ins "ValueRange":$lowerBounds, "ValueRange":$upperBounds,
      "ValueRange":$steps, "ValueRange":$inputs, "ValueRange":$outputs,
      "ArrayAttr":$iteratorTypes,
      CArg<"function_ref<void (OpBuilder &, Location, /*ivs=*/ValueRange,"
        "/*inputs=*/ValueRange, /*outputs=*/ValueRange)>",
        "nullptr">:$bodyBuilderFn)>,
  ];

  let extraClassDeclaration = [{
    /// Number of loops
    unsigned getNumLoops() { return step().size(); }

    /// Number of input operands
    unsigned getNumInputs() { return inputs().size(); }

    /// Number of output operands
    unsigned getNumOutputs() { return outputs().size(); }

    /// Number of operands controlling the loop: lbs, ubs, steps
    unsigned getNumControlOperands() { return 3 * getNumLoops(); }

    ValueRange getInductionVars() {
      return getBody()->getArguments().take_front(getNumLoops());
    }
    ValueRange getRegionInputArgs() {
      return getBody()->getArguments().slice(getNumLoops(), inputs().size());
    }
    ValueRange getRegionOutputArgs() {
      return getBody()->getArguments().take_back(outputs().size());
    }

    void setDistributionTypes(Builder& b, ArrayRef<StringRef> types) {
      assert(types.size() == getNumLoops() &&
             "expected distribution type for every dimension");
      distribution_typesAttr(b.getStrArrayAttr(types));
    }

    void setLowerBounds(ValueRange lowerBounds) {
      unsigned numLoops = getNumLoops();
      assert(lowerBounds.size() == numLoops &&
             "expected lower bounds for every loop dimension");
      for (unsigned i = 0; i < numLoops; ++i)
        setOperand(i, lowerBounds[i]);
    }

    void setUpperBounds(ValueRange upperBounds) {
      unsigned numLoops = getNumLoops();
      assert(upperBounds.size() == numLoops &&
             "expected upper bounds for every loop dimension");
      for (unsigned i = 0, pos = numLoops; i < numLoops; ++i, ++pos)
        setOperand(pos, upperBounds[i]);
    }

    void setSteps(ValueRange steps) {
      unsigned numLoops = getNumLoops();
      assert(steps.size() == numLoops &&
             "expected upper bounds for every loop dimension");
      for (unsigned i = 0, pos = 2 * numLoops; i < numLoops; ++i, ++pos)
        setOperand(pos, steps[i]);
    }

    /// Block argument that corresponds to the `input` or `output` operand.
    BlockArgument getTiedBlockArgument(OpOperand& operand) {
      auto operandIndex = operand.getOperandNumber();
      assert(
          operandIndex >= getNumControlOperands() &&
          operandIndex < getNumOperands() &&
          "tied block arg is defined only for `input` and `output` arguments");
      return getBody()->getArgument(operandIndex - 2 * getNumLoops());
    }

   /// Result that corresponds to the `outputs` argument of tensor type.
   OpResult getTiedOpResult(OpOperand& opOperand) {
      // No result can correspond to a memref argument.
      if (opOperand.get().getType().isa<MemRefType>()) return OpResult();

      // Check whether the operand index is in bounds of `outputs()` arg.
      int operandIndex = opOperand.getOperandNumber();
      int outputIndexStart =
          getNumControlOperands() + inputs().size();
      int outputIndexEnd = outputIndexStart + outputs().size();
      if (operandIndex < outputIndexStart || operandIndex >= outputIndexEnd)
        return OpResult();

      // Count tensor arguments in `outputs` to compute the result index.
      int tensorId = -1;
      for (int i = outputIndexStart; i <= operandIndex; ++i)
        tensorId += getOperand(i).getType().isa<RankedTensorType>();
      return getOperation()->getResult(tensorId);
    }

    /// Append `operand` to the `input` arguments.
    OpOperand& appendInputOperand(OpBuilder& builder, Value operand) {
      int numLoops = getNumLoops();
      int numInputs = getNumInputs();
      int numOutputs = getNumOutputs();

      getOperation()->insertOperands(getNumControlOperands() + numInputs,
                                     operand);
      getBody()->insertArgument(numLoops + numInputs, operand.getType());
      getOperation()->setAttr(
          TiledLoopOp::getOperandSegmentSizeAttr(),
          builder.getI32VectorAttr(
              {numLoops, numLoops, numLoops, numInputs + 1, numOutputs}));
      return getOperation()->getOpOperand(getNumControlOperands() + numInputs);
    }

    /// Append `operand` to the `output` arguments.
    OpOperand& appendOutputOperand(OpBuilder& builder, Value operand) {
      int numLoops = getNumLoops();
      int numInputs = getNumInputs();
      int numOutputs = getNumOutputs();

      getOperation()->insertOperands(
          getNumControlOperands() + numInputs + numOutputs, operand);
      getBody()->insertArgument(numLoops + numInputs + numOutputs,
                                operand.getType());
      getOperation()->setAttr(
          TiledLoopOp::getOperandSegmentSizeAttr(),
          builder.getI32VectorAttr(
              {numLoops, numLoops, numLoops, numInputs, numOutputs + 1}));
      return getOperation()->getOpOperand(getNumControlOperands() + numInputs +
                                          numOutputs);
    }

    /// Erase `operand` from the `input` or `output` arguments.
    void eraseOperand(OpBuilder& builder, OpOperand& operand) {
      int numInputs = getNumInputs();
      int numLoops = getNumLoops();
      int numOutputs = getNumOutputs();
      int numControlOperands = getNumControlOperands();

      int operandIndex = operand.getOperandNumber();
      assert(operandIndex >= numControlOperands &&
             operandIndex < static_cast<int>(getNumOperands()) &&
             "Can erase only `input` or `output` operand");

      if (operandIndex >= numControlOperands + numInputs)
        --numOutputs;
      else
        --numInputs;

      getOperation()->eraseOperand(operandIndex);
      getBody()->eraseArgument(operandIndex - 2 * numLoops);
      getOperation()->setAttr(
          TiledLoopOp::getOperandSegmentSizeAttr(),
          builder.getI32VectorAttr(
              {numLoops, numLoops, numLoops, numInputs, numOutputs}));
    }

    OpOperand* findInputOperand(Value value) {
      OperandRange::iterator it = llvm::find(inputs(), value);
      if (it == inputs().end()) return nullptr;
      return it.getBase();
    }

    OpOperand* findOutputOperand(Value value) {
      OperandRange::iterator it = llvm::find(outputs(), value);
      if (it == outputs().end()) return nullptr;
      return it.getBase();
    }
  }];

  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def Linalg_IndexOp : Linalg_Op<"index", [NoSideEffect]>,
    Arguments<(ins Confined<I64Attr, [IntMinValue<0>]>:$dim)>,
    Results<(outs Index:$result)> {
  let summary = "linalg index operation";
  let description = [{
    The `linalg.index` operation returns the iteration index of the immediately
    enclosing linalg structured operation for the iteration dimension `dim`. The
    `dim` attribute specifies the position of the accessed dimension in the
    indexing map domain.

    Example:

    ```mlir
    #map = affine_map<(i, j) -> (i, j)>
    linalg.generic {indexing_maps = [#map, #map],
                    iterator_types = ["parallel", "parallel"]}
      outs(%I, %J : memref<?x?xindex>, memref<?x?xindex>) {
      ^bb0(%arg0 : index, %arg1 : index):
      // Access the outer iteration dimension i
      %i = linalg.index 0 : index
      // Access the inner iteration dimension j
      %j = linalg.index 1 : index
      linalg.yield %i, %j : index, index
    }
    ```

    This may lower to IR resembling:

    ```mlir
    %0 = dim %I, %c0 : memref<?x?xindex>
    %1 = dim %I, %c1 : memref<?x?xindex>
    scf.for %i = %c0 to %0 step %c1 {
      scf.for %j = %c0 to %1 step %c1 {
        store %i, %I[%i, %j] : memref<?x?xindex>
        store %j, %J[%i, %j] : memref<?x?xindex>
      }
    }
    ```
  }];

  let assemblyFormat = [{ $dim attr-dict `:` type($result) }];
}

#endif // LINALG_OPS
