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
include "mlir/Dialect/Linalg/IR/LinalgInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/TilingInterface.td"
include "mlir/Interfaces/ViewLikeInterface.td"

// Base class for Linalg dialect ops that do not correspond to library calls.
class Linalg_Op<string mnemonic, list<Trait> traits = []> :
    Op<Linalg_Dialect, mnemonic, traits>;

def Linalg_YieldOp : Linalg_Op<"yield", [Pure, ReturnLike, Terminator]>,
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
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def Linalg_IndexOp : Linalg_Op<"index", [Pure]>,
    Arguments<(ins ConfinedAttr<I64Attr, [IntMinValue<0>]>:$dim)>,
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
  let hasVerifier = 1;
}

def Linalg_SoftmaxOp : Linalg_Op<"softmax",
    [DestinationStyleOpInterface,
     PredOpTrait<"input and output have same element type", TCopVTEtIsSameAs<0, 1>>,
     DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>,
     DeclareOpInterfaceMethods<AggregatedOpInterface, ["decomposeOperation"]>,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     DeclareOpInterfaceMethods<TilingInterface,
      ["getIterationDomain",
       "getLoopIteratorTypes",
       "getResultTilePosition",
       "getTiledImplementation"]>]> {
  let summary = "Softmax operator";
  let description = [{
    linalg.softmax computes a numerically stable version of softmax.

    For a given input tensor and a specified dimension `d`, compute:
      1. the max `m` along that dimension `d`
      2. f(x) = exp(x - m)
      3. sum f(x) along dimension d to get l(x).
      4. compute the final result f(x) / l(x).

    This is an aggregate linalg operation that further reduces to a small DAG of
    structured operations.

    Warning: Regarding the tiling capabilities, the implementation doesn't
    check that the provided dimensions make sense. This is the responsability
    of the transformation calling the tiling to ensure that the provided
    sizes for each dimension make sense with respect to the semantic of
    softmax.
  }];

  let arguments = (ins AnyShaped:$input,
                       AnyShaped:$output,
                       I64Attr:$dimension
  );

  let results = (outs Variadic<AnyRankedTensor>:$result);
  let hasFolder = 1;
  let assemblyFormat = [{
    attr-dict
    `dimension` `(` $dimension `)`
    `ins` `(` $input `:` type($input) `)`
    `outs` `(` $output `:` type($output) `)`
    (`->` type($result)^)?
  }];

  let extraClassDeclaration = [{
    ShapedType getInputOperandType() {
      return cast<ShapedType>(getInput().getType());
    }
    ShapedType getOutputOperandType() {
      return cast<ShapedType>(getOutput().getType());
    }
    int64_t getInputOperandRank() {
      return getInputOperandType().getRank();
    }
    int64_t getOutputOperandRank() {
      return getOutputOperandType().getRank();
    }
    MutableOperandRange getDpsInitsMutable() { return getOutputMutable(); }
  }];
  let hasVerifier = 1;
}

def Linalg_WinogradFilterTransformOp : Linalg_Op<"winograd_filter_transform",
    [AllElementTypesMatch<["filter", "output"]>,
     DeclareOpInterfaceMethods<TilingInterface,
      ["getIterationDomain",
       "getLoopIteratorTypes",
       "getResultTilePosition",
       "getTiledImplementation"]>]> {
  let summary = "Winograd filter transform operator";
  let description = [{
    Winograd Conv2D algorithm will convert linalg Conv2D operator into batched
    matrix multiply. Before the matrix multiply, it will convert filter and
    input into a format suitable for batched matrix multiply. After the matrix
    multiply, it will convert output to the final result tensor.

    The algorithm F(m x m, r x r) is

    Y = A^T x [(G x g x G^T) @ (B^T x d x B)] x A

    The size of output Y is m x m. The size of filter g is r x r. The size of
    input d is (m + r - 1) x (m + r - 1). A^T, A, G^T, G, B^T, and B are
    transformation matrices.

    This operator is defined to represent the high level concept of filter
    transformation (G x g x G^T) in the Winograd Conv2D algorithm.
  }];

  let arguments = (ins TensorRankOf<[AnyType], [4]>:$filter,
                       TensorRankOf<[AnyType], [4]>:$output,
                       I64Attr:$m,
                       I64Attr:$r
  );

  let results = (outs TensorRankOf<[AnyType], [4]>:$result);
  let assemblyFormat = [{
    attr-dict
    `m` `(` $m `)`
    `r` `(` $r `)`
    `ins` `(` $filter `:` type($filter) `)`
    `outs` `(` $output `:` type($output) `)`
    `->` type($result)
  }];
  let extraClassDeclaration = [{
    ShapedType getFilterOperandType() {
      return cast<ShapedType>(getFilter().getType());
    }
    ShapedType getOutputOperandType() {
      return cast<ShapedType>(getOutput().getType());
    }
    int64_t getFilterOperandRank() {
      return getFilterOperandType().getRank();
    }
    int64_t getOutputOperandRank() {
      return getOutputOperandType().getRank();
    }
    int64_t getFilterFDim() {
      return 0;
    }
    int64_t getFilterHDim() {
      return 1;
    }
    int64_t getFilterWDim() {
      return 2;
    }
    int64_t getFilterCDim() {
      return 3;
    }
  }];
  let hasVerifier = 1;
}

def Linalg_WinogradInputTransformOp : Linalg_Op<"winograd_input_transform",
    [AllElementTypesMatch<["input", "output"]>,
     DeclareOpInterfaceMethods<TilingInterface,
      ["getIterationDomain",
       "getLoopIteratorTypes",
       "getResultTilePosition",
       "getTiledImplementation"]>]> {
  let summary = "Winograd input transform operator";
  let description = [{
    Winograd Conv2D algorithm will convert linalg Conv2D operator into batched
    matrix multiply. Before the matrix multiply, it will convert filter and
    input into a format suitable for batched matrix multiply. After the matrix
    multiply, it will convert output to the final result tensor.

    The algorithm F(m x m, r x r) is

    Y = A^T x [(G x g x G^T) @ (B^T x d x B)] x A

    The size of output Y is m x m. The size of filter g is r x r. The size of
    input d is (m + r - 1) x (m + r - 1). A^T, A, G^T, G, B^T, and B are
    transformation matrices.

    This operator is defined to represent the high level concept of input
    transformation (B^T x d x B) in the Winograd Conv2D algorithm.
  }];

  let arguments = (ins TensorRankOf<[AnyType], [4]>:$input,
                       TensorRankOf<[AnyType], [6]>:$output,
                       I64Attr:$m,
                       I64Attr:$r
  );

  let results = (outs TensorRankOf<[AnyType], [6]>:$result);
  let assemblyFormat = [{
    attr-dict
    `m` `(` $m `)`
    `r` `(` $r `)`
    `ins` `(` $input `:` type($input) `)`
    `outs` `(` $output `:` type($output) `)`
    `->` type($result)
  }];
  let extraClassDeclaration = [{
    ShapedType getInputOperandType() {
      return cast<ShapedType>(getInput().getType());
    }
    ShapedType getOutputOperandType() {
      return cast<ShapedType>(getOutput().getType());
    }
    int64_t getInputOperandRank() {
      return getInputOperandType().getRank();
    }
    int64_t getOutputOperandRank() {
      return getOutputOperandType().getRank();
    }
    int64_t getInputNDim() {
      return 0;
    }
    int64_t getInputHDim() {
      return 1;
    }
    int64_t getInputWDim() {
      return 2;
    }
    int64_t getInputCDim() {
      return 3;
    }
    int64_t getOutputAlphaHDim() {
      return 0;
    }
    int64_t getOutputAlphaWDim() {
      return 1;
    }
    int64_t getOutputTileHDim() {
      return 2;
    }
    int64_t getOutputTileWDim() {
      return 3;
    }
    int64_t getOutputNDim() {
      return 4;
    }
    int64_t getOutputCDim() {
      return 5;
    }
  }];
  let hasVerifier = 1;
}

def Linalg_WinogradOutputTransformOp : Linalg_Op<"winograd_output_transform",
    [AllElementTypesMatch<["value", "output"]>, DestinationStyleOpInterface,
     DeclareOpInterfaceMethods<TilingInterface,
      ["getIterationDomain",
       "getLoopIteratorTypes",
       "getResultTilePosition",
       "getTiledImplementation"]>]> {
  let summary = "Winograd output transform operator";
  let description = [{
    Winograd Conv2D algorithm will convert linalg Conv2D operator into batched
    matrix multiply. Before the matrix multiply, it will convert filter and
    input into a format suitable for batched matrix multiply. After the matrix
    multiply, it will convert output to the final result tensor.

    The algorithm F(m x m, r x r) is

    Y = A^T x [(G x g x G^T) @ (B^T x d x B)] x A

    The size of output Y is m x m. The size of filter g is r x r. The size of
    input d is (m + r - 1) x (m + r - 1). A^T, A, G^T, G, B^T, and B are
    transformation matrices.

    This operator is defined to represent the high level concept of output
    transformation (A^T x y x A) in the Winograd Conv2D algorithm.
  }];

  let arguments = (ins TensorRankOf<[AnyType], [6]>:$value,
                       TensorRankOf<[AnyType], [4]>:$output,
                       I64Attr:$m,
                       I64Attr:$r
  );

  let results = (outs TensorRankOf<[AnyType], [4]>:$result);
  let assemblyFormat = [{
    attr-dict
    `m` `(` $m `)`
    `r` `(` $r `)`
    `ins` `(` $value `:` type($value) `)`
    `outs` `(` $output `:` type($output) `)`
    `->` type($result)
  }];
  let extraClassDeclaration = [{
    ShapedType getValueOperandType() {
      return cast<ShapedType>(getValue().getType());
    }
    ShapedType getOutputOperandType() {
      return cast<ShapedType>(getOutput().getType());
    }
    int64_t getValueOperandRank() {
      return getValueOperandType().getRank();
    }
    int64_t getOutputOperandRank() {
      return getOutputOperandType().getRank();
    }
    int64_t getValueAlphaHDim() {
      return 0;
    }
    int64_t getValueAlphaWDim() {
      return 1;
    }
    int64_t getValueTileHDim() {
      return 2;
    }
    int64_t getValueTileWDim() {
      return 3;
    }
    int64_t getValueNDim() {
      return 4;
    }
    int64_t getValueFDim() {
      return 5;
    }
    int64_t getOutputNDim() {
      return 0;
    }
    int64_t getOutputHDim() {
      return 1;
    }
    int64_t getOutputWDim() {
      return 2;
    }
    int64_t getOutputFDim() {
      return 3;
    }
    MutableOperandRange getDpsInitsMutable() { return getOutputMutable(); }
  }];
  let hasVerifier = 1;
}

#endif // LINALG_OPS
