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
          Returns a list of operands into which the result of the
          tiled implementation is written into. With `tensor`
          operands, this will be used as the initial tensor into which
          the tiled results are inserted into. With `memref` operands,
          this will be the operand into which the result of the tiled
          operation is written into.
        }],
        /*retType=*/"SmallVector<Value>",
        /*methodName=*/"getDestinationOperands",
        /*args=*/(ins "OpBuilder &":$b),
        /*methodBody=*/"",
        /*defaultImplementation=*/"return ValueRange{};"
      >,
      InterfaceMethod<
        /*desc=*/[{
          Returns a list of `StringRef`s that describe the number of
          loops and the iterator types of the operation. The list is
          expected to use
          `getParallelIteratorTypeName()`/`getReductionIteratorTypeName()`
          from MLIR Structured Op Utils.
        }],
        /*retType=*/"SmallVector<StringRef>",
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
          - `dest` are the Value into which the result of the tiled
            operation is to be inserted into. The type of the `dest`
            Values is same as the types returned by
            `getDestinationOperands` method.
          - `offsets` provides the offset of the tile within the
            iteration space
          - `sizes` provides the size of the tile.
          - `tileDestOperands` specifies whether to also tile `dest` operands
            or not. Avoiding tiling `dest` operands can be useful for 
            composition with various looping container ops.

          The method returns the operation that is the tiled
          implementation.
        }],
        /*retType=*/"SmallVector<Operation *>",
        /*methodName=*/"getTiledImplementation",
        /*args=*/(ins
            "OpBuilder &":$b,
            "ValueRange ":$dest,
            "ArrayRef<OpFoldResult> ":$offsets,
            "ArrayRef<OpFoldResult> ":$sizes,
            "bool ":$tileDestOperands),
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
      >
  ];
}
#endif // MLIR_TILINGINTERFACE
