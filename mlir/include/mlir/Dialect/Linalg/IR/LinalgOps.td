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

def Linalg_InitTensorOp : Linalg_Op<"init_tensor", [NoSideEffect]> {
  let summary = "operation to define a tensor of particular value";

  let description = [{
    `linalg.init_tensor` is an operation that materializes a tensor of
    a given shape. The shape could be dynamic or static.
  }];

  let arguments =
    (ins Variadic<Index>:$sizes, I64ArrayAttr:$static_sizes);

  let results = (outs AnyTensor:$result);

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
    OpBuilderDAG<(ins "ValueRange":$shape,
                  "ArrayRef<int64_t>":$staticShape, "Type":$elementType),
    [{
      build($_builder, $_state,
            InitTensorOp::inferResultType(staticShape, elementType),
            shape, $_builder.getI64ArrayAttr(staticShape));
    }]>,
    OpBuilderDAG<(ins "ValueRange":$shape, "Type":$elementType),
    [{
      SmallVector<int64_t, 4> staticShape(
        shape.size(), ShapedType::kDynamicSize);
      build($_builder, $_state, shape, staticShape, elementType);
    }]>,
    OpBuilderDAG<(ins "ArrayRef<int64_t>":$staticShape, "Type":$elementType),
    [{
      build($_builder, $_state, ValueRange{}, staticShape, elementType);
    }]>
  ];

  let hasCanonicalizer = 1;
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
    OpBuilderDAG<(ins "Value":$min, "Value":$max, "Value":$step),
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
    OpBuilderDAG<(ins "Value":$src,
      "ArrayRef<ReassociationExprs>":$reassociation,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
    OpBuilderDAG<(ins "Value":$src,
      "ArrayRef<ReassociationIndices>":$reassociation,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs),
    [{
      auto reassociationMaps =
          convertReassociationIndicesToMaps($_builder, reassociation);
      build($_builder, $_state, src, reassociationMaps, attrs);
    }]>,

    // Builders for a reshape whose result type is passed explicitly. This may
    // be either a contracting or expanding reshape.
    OpBuilderDAG<(ins "Type":$resultType, "Value":$src,
      "ArrayRef<ReassociationExprs>":$reassociation,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
    OpBuilderDAG<(ins "Type":$resultType, "Value":$src,
      "ArrayRef<ReassociationIndices>":$reassociation,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs),
    [{
      auto reassociationMaps =
          convertReassociationIndicesToMaps($_builder, reassociation);
      build($_builder, $_state, resultType, src, reassociationMaps, attrs);
    }]>
  ];

  code commonExtraClassDeclaration = [{
    static StringRef getReassociationAttrName() { return "reassociation"; }
    SmallVector<AffineMap, 4> getReassociationMaps() {
      return llvm::to_vector<4>(llvm::map_range(reassociation(), [
      ](Attribute a) { return a.cast<AffineMapAttr>().getValue(); }));
    }
    SmallVector<ReassociationExprs, 4> getReassociationExprs() {
      return
        llvm::to_vector<4>(llvm::map_range(reassociation(),
	  [](Attribute a) {
	    return llvm::to_vector<2>(
	      a.cast<AffineMapAttr>().getValue().getResults());
	  }));
    }
  }];
  let assemblyFormat = [{
    $src $reassociation attr-dict `:` type($src) `into` type(results)
  }];
}

def Linalg_ReshapeOp : Linalg_ReshapeLikeOp<"reshape",
    [DeclareOpInterfaceMethods<ViewLikeOpInterface>]>,
    Arguments<(ins AnyStridedMemRef:$src, AffineMapArrayAttr:$reassociation)>,
    Results<(outs AnyStridedMemRef:$result)> {
  let summary = "linalg.reshape produces a new view into the operand view";
  let description = [{
    The `linalg.reshape` op produces a new view whose sizes are a reassociation
    of the original `view`. Depending on whether or not the reassociated
    MemRefType is contiguous, the resulting memref may require explicit alloc
    and copies.

    A reassociation is defined as a continuous grouping of dimensions and is
    represented with an affine map array attribute. In the future,
    non-continuous groupings may be allowed (i.e. permutations, reindexings
    etc).

    For now, it is assumed that either:
      1. a reassociation produces and consumes contiguous MemRefType or,
      2. the reshape op will be folded into its consumers (by changing the shape
         of the computations).
    All other cases are undefined behavior and a reshape op may not lower to
    LLVM if it cannot be proven statically that it does not require alloc+copy.

    A reshape may either collapse or expand dimensions, depending on the
    relationship between source and target memref ranks. The verification rule
    is that the reassociation maps are applied to the memref with the larger
    rank to obtain the memref with the smaller rank. In the case of a dimension
    expansion, the reassociation maps can be interpreted as inverse maps.

    The result memref type of a reshape when dimensions are collapsed
    (operand memref type when dimensions are expanded) can be
    zero-ranked if the operand memref type (or the result memref type
    when dimensions are expanded) is statically shaped with all
    dimensions being unit extent. In such cases the reassociation map
    is empty.

    Examples:

    ```mlir
    // Dimension collapse (i, j) -> i' and k -> k'
    %1 = linalg.reshape %0 [(i, j, k) -> (i, j), (i, j, k) -> (k)] :
      memref<?x?x?xf32, stride_spec> into memref<?x?xf32, stride_spec_2>
    ```

    ```mlir
    // Dimension expansion i -> (i', j') and (k) -> (k')
    %1 = linalg.reshape %0 [(i, j, k) -> (i, j), (i, j, k) -> (k)] :
      memref<?x?xf32, stride_spec> into memref<?x?x?xf32, stride_spec_2>
    ```
  }];
  let extraClassDeclaration = commonExtraClassDeclaration # [{
    MemRefType getSrcType() { return src().getType().cast<MemRefType>(); }
    MemRefType getResultType() { return result().getType().cast<MemRefType>(); }
  }];
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

def Linalg_TensorReshapeOp : Linalg_ReshapeLikeOp<"tensor_reshape">,
    Arguments<(ins AnyTensor:$src,
                   AffineMapArrayAttr:$reassociation)>,
    Results<(outs AnyTensor:$result)> {
  let summary = "linalg.tensor_reshape produces a new reshaped tensor.";
  let description = [{
    The `linalg.reshape` op produces a new tensor whose sizes are a
    reassociation of the original `src`.

    A reassociation is defined as a continuous grouping of dimensions and is
    represented with an affine map array attribute. In the future,
    non-continuous groupings may be allowed (i.e. permutations, reindexings
    etc).

    A reshape may either collapse or expand dimensions, depending on the
    relationship between source and target tensor ranks. The verification rule
    is that the reassociation maps are applied to the tensor with the larger
    rank to obtain the tensor with the smaller rank. In the case of a dimension
    expansion, the reassociation maps can be interpreted as inverse maps.

    The result tensor type of a reshape when dimensions are collapsed
    (operand tensor type when dimensions are expanded) can be
    zero-ranked if the operand tensor type (or the result tensor type
    when dimensions are expanded) is statically shaped with all
    dimensions being unit extent. In such cases the reassociation map
    is empty.

    Examples:

    ```mlir
    // Dimension collapse (i, j) -> i' and k -> k'
    %b = linalg.tensor_reshape %a [(i, j, k) -> (i, j), (i, j, k) -> (k)] :
      tensor<?x?x?xf32> into tensor<?x?xf32>
    ```

    ```mlir
    // Dimension expansion i -> (i', j') and (k) -> (k')
    %b = linalg.tensor_reshape %a [(i, j, k) -> (i, j), (i, j, k) -> (k)] :
      tensor<?x?xf32> into tensor<?x?x?xf32>
    ```
  }];
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
}

def Linalg_SliceOp : Linalg_Op<"slice", [
      DeclareOpInterfaceMethods<ViewLikeOpInterface>, NoSideEffect]>,
    Arguments<(ins AnyStridedMemRef:$view,
                   Variadic<AnyTypeOf<[Range, Index]>>:$indexings)>,
    Results<(outs AnyStridedMemRef)> {
  let summary = "Produce a rank-reduced `subview` of a base `view`.";
  let description = [{
    The `linalg.slice` op allows defining a subregion of a smaller rank than the
    operand `view` within the underlying buffer.

    A `linalg.slice` op takes a view and a variadic number of indexings and
    produces a `view` of the same elemental type. An indexing is either:
      1. a `linalg.range`, in which case it does not reduce the rank of the
         parent `view` along the corresponding dimension.
      2. an `index`, in which case it reduces the rank of the parent view by
         one.

    If an indexing extends past the size of the `view`, this is undefined
    behavior. Ideally the `linalg.slice` operation would automatically truncate
    it to be within bounds but there are tradeoffs involved now that `std.view`
    is a standard op.

    Examples:

      1. rank-preserving `slice`:

      ```mlir
      %4 = linalg.slice %0[%1, %2] : memref<?x?xf32, stride_spec>,
        !linalg.range, !linalg.range, memref<?x?xf32, stride_spec>
      ```

      2. rank-reducing `slice` (from 2-D to 1-D):

      ```mlir
      %4 = linalg.slice %0[%1, %2] : memref<?x?xf32, stride_spec>,
        index, !linalg.range, memref<?x?xf32, stride_spec>
      ```

      3. rank-reducing `slice` (from 2-D to 0-D):

      ```mlir
      %4 = linalg.slice %0[%1, %2] : memref<?x?xf32, stride_spec>,
        index, index, memref<?x?xf32, stride_spec>
      ```
  }];

  let builders = [OpBuilderDAG<(ins "Value":$base, "ValueRange":$indexings)>];

  let extraClassDeclaration = [{
    enum { FirstIndexingOperand = 1 };
    unsigned getRank() { return getShapedType().getRank(); }
    Type getElementType() { return getShapedType().getElementType(); }
    ShapedType getShapedType() { return getType().cast<ShapedType>(); }
    unsigned getBaseViewRank() { return getBaseViewType().getRank(); }
    ShapedType getBaseViewType() { return view().getType().cast<ShapedType>();}

    // Get the underlying indexing at a given rank.
    Value indexing(unsigned rank) { return *(indexings().begin() + rank); }

    // Get the subset of indexings that are of RangeType.
    SmallVector<Value, 8> getRanges() {
      SmallVector<Value, 8> res;
      for (auto operand : indexings())
        if (!operand.getType().isa<IndexType>())
          res.push_back(operand);
      return res;
    }
  }];

  let hasFolder = 1;
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
}

#endif // LINALG_OPS
