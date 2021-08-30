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
        /*methodName=*/"getLoopIteratorTypes"
      >,
      InterfaceMethod<
        /*desc=*/[{
          Returns a list of ranges that describe the loop bounds and
          step for the loops of the operation.
        }],
        /*retTy=*/"SmallVector<Range>",
        /*methodName=*/"getLoopBounds",
        /*args=*/(ins "OpBuilder &":$b)
      >,
      InterfaceMethod<
        /*desc=*/[{
          Method to generate the tiled implementation of an operation.

          The iteration space of the operation is returned by
          `getLoopBounds`. The caller provides the information of the
          tile within this iteration space whose implementation the
          caller needs.
          - `dest` are the Value into which the result of the tiled
            operation is to be inserted into. The type of the `dest`
            Values is same as the types returned by
            `getDestinationOperands` method.
          - `offsets` provides the offset of the tile within the
            iteration space
          - `sizes` provides the size of the tile.

          The method returns the operation that is the tiled
          implementation.
        }],
        /*retType=*/"Operation *",
        /*methodName=*/"getTiledImplementation",
        /*args=*/(ins
            "OpBuilder &":$b,
            "ValueRange ":$dest,
            "ArrayRef<OpFoldResult> ":$offsets,
            "ArrayRef<OpFoldResult> ":$sizes),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          return nullptr;
        }]
      >
  ];
}
#endif // MLIR_TILINGINTERFACE
