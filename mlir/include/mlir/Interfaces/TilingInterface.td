//===- TilingInterface.td - Interface for tiling operations *- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains an interface to allow operations to generate a tiled
// implementation of themselves.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TILINGINTERFACE
#define MLIR_TILINGINTERFACE

include "mlir/IR/OpBase.td"

def TilingInterface : OpInterface<"TilingInterface"> {
  let description = [{
    Interface for allowing operations to expose information needed to
    tile them (similar to LinalgOp, but without having access to
    indexing maps)
  }];
  let cppNamespace = "::mlir";
  let methods = [
      InterfaceMethod<
        /*desc=*/[{
          Returns a list of iterator types that describe the number of loops.
        }],
        /*retType=*/"SmallVector<utils::IteratorType>",
        /*methodName=*/"getLoopIteratorTypes",
        /*args=*/(ins),
        /*methodBody=*/"",
        /*defaultImplementation=*/"return {};"
      >,
      InterfaceMethod<
        /*desc=*/[{
          Returns a list of ranges that describe the loop bounds and
          step for the loops of the operation.
        }],
        /*retTy=*/"SmallVector<Range>",
        /*methodName=*/"getIterationDomain",
        /*args=*/(ins "OpBuilder &":$b),
        /*methodBody=*/"",
        /*defaultImplementation=*/"return {};"
      >,
      InterfaceMethod<
        /*desc=*/[{
          Method to generate the tiled implementation of an operation.

          The iteration space of the operation is returned by
          `getIterationDomain`. The caller provides the information of the
          tile within this iteration space whose implementation the
          caller needs.
          - `offsets` provides the offset of the tile in the coordinate system
            of the original iteration space, i.e., if an iteration space
            dimension had non-zero offset, it must be included in the offset
            provided here (as opposed to zero-based offset "relative" to the
            iteration space).
          - `sizes` provides the size of the tile.

          The method returns the operation that is the tiled
          implementation.
        }],
        /*retType=*/"FailureOr<TilingResult>",
        /*methodName=*/"getTiledImplementation",
        /*args=*/(ins
            "OpBuilder &":$b,
            "ArrayRef<OpFoldResult> ":$offsets,
            "ArrayRef<OpFoldResult> ":$sizes),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          return {};
        }]
      >,
      InterfaceMethod<
        /*desc=*/[{
          Method to return the position of the result tile computed by the tiled operation.

          Specifies what tile of the result of the original tensor is computed
          by the tiled implementation. Expects the same `offsets` and `sizes` as
          used to obtain the tiled implementation of the operation.
        }],
        /*retType=*/"LogicalResult",
        /*methodName=*/"getResultTilePosition",
        /*args=*/(ins
          "OpBuilder &":$b,
          "unsigned":$resultNumber,
          "ArrayRef<OpFoldResult> ":$offsets,
          "ArrayRef<OpFoldResult> ":$sizes,
          "SmallVector<OpFoldResult> &":$resultOffsets,
          "SmallVector<OpFoldResult> &":$resultSizes),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          return failure();
        }]
      >,
      InterfaceMethod<
        /*desc=*/[{
          Method to generate the code that produces a tile of the result.

          Generates the IR that computes the tile of a result of the
          operation.  The `offsets` and `sizes` describe the tile of
          the output required. This is different from
          `getTiledImplementation` which generates the tiled
          implementation of the operation given a tile of the
          iteration space. This method generates a tiled
          implementation of the operation based on the tile of the
          result required. This method enables fusion by using tile
          and fuse. The method returns failure if the operation can't be
          tiled to generate the result tile. In practical terms this
          implies it cannot be tiled and fused with its consumers.

          - `offsets` provides the offset of the tile in the coordinate system
            of the original iteration space, i.e., if an iteration space
            dimension had non-zero offset, it must be included in the offset
            provided here (as opposed to zero-based offset "relative" to the
            iteration space).
          - `sizes` provides the size of the tile.
        }],
        /*retType=*/"FailureOr<TilingResult>",
        /*methodName=*/"generateResultTileValue",
        /*args=*/(ins
          "OpBuilder &":$b,
          "unsigned":$resultNumber,
          "ArrayRef<OpFoldResult>":$offsets,
          "ArrayRef<OpFoldResult>":$sizes),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          return failure();
        }]
      >,
      InterfaceMethod<
        /*desc=*/[{
          Generates the scalar implementation of the operation.

          Given the list `ivs` that represent points in the iteration space
          (as specified by `getIterationDomain()`) returns the scalar operations
          that represent the computation at that point in the iteration space.
          This method is typically used as the "exit path", i.e. once all
          transformations are done, this method can be used to lower to scalar
          code that can then be lowered to LLVM or SPIR-V dialects.
        }],
        /*retType=*/"LogicalResult",
        /*methodName=*/"generateScalarImplementation",
        /*args=*/(ins
            "OpBuilder &":$b,
            "Location ":$loc,
            "ValueRange ":$ivs),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          return failure();
        }]
      >
  ];
}

def PartialReductionOpInterface : OpInterface<"PartialReductionOpInterface"> {
  let description = [{
    Interface for allowing operations to expose information needed to
    tile reductions using partial reduction followed by merge. This is
    complementary to TilingInterface to tile reductions.
  }];
  let cppNamespace = "::mlir";
  let methods = [
      InterfaceMethod<
        /*desc=*/[{
          Method to generate a tensor initalized with the identity value of the
          operation reduction. The tensor shape is equal to operation result
          shape with new dimension for each non zero tile size.
        }],
        /*retType=*/"FailureOr<SmallVector<Value>>",
        /*methodName=*/"generateInitialTensorForPartialReduction",
        /*args=*/(ins
            "OpBuilder &":$b,
            "Location":$loc,
            "ArrayRef<OpFoldResult>":$sizes,
            "ArrayRef<int>":$reductionDim),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          return failure();
        }]
      >,
      InterfaceMethod<
        /*desc=*/[{
          Method to generate a tiled version of the operation where the tiled
          reduction dimension are converted to parallel dimensions with a size
          less or equal to the tile size. This is meant to be used with
          `mergeReductions` method which will combine the partial reductions.
        }],
        /*retType=*/"Operation*",
        /*methodName=*/"tileToPartialReduction",
        /*args=*/(ins
            "OpBuilder &":$b,
            "Location ":$loc,
            "ValueRange":$init,
            "ArrayRef<OpFoldResult>":$offsets,
            "ArrayRef<OpFoldResult>":$sizes,
            "ArrayRef<int>":$reductionDims),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          return nullptr;
        }]
      >,
      InterfaceMethod<
        /*desc=*/[{
          Method to merge partial reductions for an operation that has been
          tiled along the reduction dimensions. This will only apply the
          reduction the operation.
        }],
        /*retType=*/"Operation*",
        /*methodName=*/"mergeReductions",
        /*args=*/(ins
            "OpBuilder &":$b,
            "Location ":$loc,
            "ValueRange":$partialReduce,
            "ArrayRef<int>":$reductionDim),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          return nullptr;
        }]
      >
  ];
}
#endif // MLIR_TILINGINTERFACE
