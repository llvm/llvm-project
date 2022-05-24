//===- LinalgTransformOps.td - Linalg transform ops --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LINALG_TRANSFORM_OPS
#define LINALG_TRANSFORM_OPS

include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/IR/TransformEffects.td"
include "mlir/Dialect/Transform/IR/TransformInterfaces.td"
include "mlir/Dialect/PDL/IR/PDLTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

def InterchangeOp : Op<Transform_Dialect, "structured.interchange",
    [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
    TransformOpInterface, TransformEachOpTrait]> {
  let description = [{
    Interchanges the iterators of the operations pointed to by the target handle
    using the iterator interchange attribute.
  }];

  let arguments =
    (ins PDL_Operation:$target,
         DefaultValuedAttr<I64ArrayAttr, "{}">:$iterator_interchange);
  let results = (outs PDL_Operation:$transformed);

  let assemblyFormat = "$target attr-dict";
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ::mlir::FailureOr<::mlir::linalg::LinalgOp> applyToOne(
        ::mlir::linalg::LinalgOp target);
  }];
}

def PadOp : Op<Transform_Dialect, "structured.pad",
    [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
     TransformOpInterface, TransformEachOpTrait]> {
  let description = [{
    Pads the operations pointed to by the target handle using the options
    provides as operation attributes.
  }];

  let arguments =
    (ins PDL_Operation:$target,
         DefaultValuedAttr<ArrayAttr, "{}">:$padding_values,
         DefaultValuedAttr<I64ArrayAttr, "{}">:$padding_dimensions,
         DefaultValuedAttr<I64ArrayAttr, "{}">:$pack_paddings,
         DefaultValuedAttr<I64ArrayAttr, "{}">:$hoist_paddings,
         DefaultValuedAttr<
          TypedArrayAttrBase<I64ArrayAttr, "array of arrays of i64">,
          "{}">:$transpose_paddings);
  let results = (outs PDL_Operation:$transformed);

  let assemblyFormat = "$target attr-dict";
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ::mlir::FailureOr<::mlir::linalg::LinalgOp> applyToOne(
        ::mlir::linalg::LinalgOp target);
  }];
}

def ScalarizeOp : Op<Transform_Dialect, "structured.scalarize",
    [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
     TransformOpInterface, TransformEachOpTrait]> {
  let description = [{
    Indicates that ops of a specific kind in the given function should be
    scalarized (i.e. their dynamic dimensions tiled by 1).

    This operation returns the tiled op but not the loops.

    We make this design choice because it is hard to know ahead of time the
    number of loops that will be produced (it depends on the number of dynamic
    dimensions after multiple transformations have been applied).
  }];

  let arguments = (ins PDL_Operation:$target);
  let results = (outs PDL_Operation:$result);

  let assemblyFormat = "$target attr-dict";

  let extraClassDeclaration = [{
    ::mlir::FailureOr<::mlir::linalg::LinalgOp> applyToOne(
        ::mlir::linalg::LinalgOp target);
  }];
}

def TileOp : Op<Transform_Dialect, "structured.tile",
       [DeclareOpInterfaceMethods<TransformOpInterface>,
        DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let description = [{
    Indicates that the given `target` op should be tiled with the options
    provided as attributes. This transform generates a loop nest with a smaller
    ("tiled") target operation in its body. Currently limited to LinalgOps.

    `sizes` are the tile sizes. A tile size of `0` indicates that the
    respective dimension should not be tiled. No loop will be generated for such
    dimensions. If all tile sizes are `0`, this transform is effectively a
    no-op.

    This op returns handles to the tiled op (in the generated loop nest) and the
    generated loops. The number of loops is the number of non-zero tile sizes.
  }];

  let arguments = (ins PDL_Operation:$target,
                   DefaultValuedAttr<I64ArrayAttr, "{}">:$sizes,
                   DefaultValuedAttr<I64ArrayAttr, "{}">:$interchange);
  let results = (outs PDL_Operation:$tiled_linalg_op,
                      Variadic<PDL_Operation>:$loops);

  let hasCustomAssemblyFormat = 1;
}

#endif // LINALG_TRANSFORM_OPS
