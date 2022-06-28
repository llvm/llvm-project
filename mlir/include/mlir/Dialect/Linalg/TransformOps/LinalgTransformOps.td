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

def DecomposeOp : Op<Transform_Dialect, "structured.decompose",
    [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
     TransformOpInterface, TransformEachOpTrait]> {
  let description = [{
    Decomposes named complex operations, such as higher-dimensional
    (depthwise) convolutions, into combinations of lower-dimensional equivalents
    when possible. The operand handle must point to a list of such operations.
    The returning handle points to the main produced computational operation,
    such as the lower-dimensional convolution.
  }];

  let arguments = (ins PDL_Operation:$target);
  let results = (outs PDL_Operation:$transformed);
  let assemblyFormat = "$target attr-dict";

  let extraClassDeclaration = [{
    ::mlir::FailureOr<::mlir::linalg::LinalgOp> applyToOne(
        ::mlir::linalg::LinalgOp target, TransformState &state);
  }];
}

def FuseOp : Op<Transform_Dialect, "structured.fuse",
    [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
     DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let description = [{
    Tiles the operations pointed to by the target handle and fuses their
    producers greedily using the options provided as attributes.
  }];

  let arguments =
    (ins PDL_Operation:$target,
         DefaultValuedAttr<I64ArrayAttr, "{}">:$tile_sizes,
         DefaultValuedAttr<I64ArrayAttr, "{}">:$tile_interchange);
  let results = (outs PDL_Operation:$transformed,
                      Variadic<PDL_Operation>:$loops);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def GeneralizeOp : Op<Transform_Dialect, "structured.generalize",
    [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
     TransformOpInterface, TransformEachOpTrait]> {
  let description = [{
    Transforms a named structued operation into the generic form with the
    explicit attached region. The operand handle must point to a list of
    structured operations, it is consumed by the transformation and is not
    expected to be used afterwards. The resulting handle points to the list
    of equivalent generic operations, in the same order as the original named
    operations.
  }];

  let arguments = (ins PDL_Operation:$target);
  let results = (outs PDL_Operation:$transformed);
  let assemblyFormat = "$target attr-dict";

  let extraClassDeclaration = [{
    ::mlir::FailureOr<::mlir::linalg::LinalgOp> applyToOne(
        ::mlir::linalg::LinalgOp target, TransformState &state);
  }];
}

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
        ::mlir::linalg::LinalgOp target, TransformState &state);
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
        ::mlir::linalg::LinalgOp target, TransformState &state);
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
        ::mlir::linalg::LinalgOp target, TransformState &state);
  }];
}

def SplitReductionOp : Op<Transform_Dialect, "structured.split_reduction",
       [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
        TransformEachOpTrait, TransformOpInterface]> {
  let description = [{
    Indicates that the given `target` op should be transformed with the 
    `splitReduction` transformation and split factor provided as attribute.

    The `splitReduction` transformation splits the first single linalg op 
    reduction into a parallel and reduction dimension. 
    A new `linalg.generic` op is created to perform the rest of the reduction. 
    
    The transformation supports different configurations attributes:
      - split_factor: the factor by which to split (i.e. the size of the 
        remaining reduction after splitting).
      - insert_split_dimension: the dimension in the temporary tensor into 
        which the new parallel dimension is inserted.
      - use_scaling_algorithm: whether to use a scaling based formulation that 
        does not create an ExpandShapeOp (default: do not use scaling)
      - use_alloc: whether to use an alloc op to allocate the temporary 
        tensor (default: do not use alloc op)

    This op returns 4 handles to:
      - the init op (or tensor_alloc op if use_alloc = true), 
      - the fill op used to initialize the neutral element, 
      - the split op and 
      - the result-combining op.

    Example (default: use_scaling_algorithm = false, use_alloc = false):
    ====================================================================
    ```
      %r = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>,
                                            affine_map<(d0) -> ()>],
            iterator_types = ["reduction"]}
      ins(%in : tensor<32xf32>)
      outs(%out : tensor<f32>) {
      ^bb0(%arg1: f32, %arg2: f32):
        %y = arith.addf %arg1, %arg2 : f32
        linalg.yield %y : f32
      } -> tensor<f32>
    ```
    
    is split into:
    
    ```
      %cst = arith.constant 0.000000e+00 : f32
      %0 = tensor.expand_shape %in [[0, 1]] : tensor<32xf32> into tensor<4x8xf32>
      %1 = linalg.init_tensor [4] : tensor<4xf32>
      %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<4xf32>) -> tensor<4xf32>
      %3 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                            affine_map<(d0, d1) -> (d0)>],
        iterator_types = ["parallel", "reduction"]}
        ins(%0 : tensor<4x8xf32>) outs(%2 : tensor<4xf32>) {
        ^bb0(%arg3: f32, %arg5: f32):
        %5 = arith.addf %arg3, %arg4 : f32
        linalg.yield %5 : f32
      } -> tensor<4xf32>
      %r = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>,
                                            affine_map<(d0) -> ()>],
        iterator_types = ["reduction"]}
        ins(%3 : tensor<4xf32>) outs(%out : tensor<f32>) {
        ^bb0(%arg3: f32, %arg4: f32):
        %5 = arith.addf %arg3, %arg4 : f32
        linalg.yield %5 : f32
      } -> tensor<f32>
    ```

    Example (use_scaling_algorithm = true, use_alloc = true):
    =========================================================
    Instead of introducing an ExpandShapeOp, this scaling-based implementation 
    rewrites a reduction dimension `k` into `k * split_factor + kk`.
    The dimension `kk` is added as an extra parallel dimension to the 
    intermediate output tensor at position `insert_split_dimension`.

    Consider a minimal example where `k` is reduced: 
        O(i, j) += I(i, j, k)
    Assume i=3, j=5, k=128, split_factor=16 and insert_split_dimension=0.
    The compute is rewritten as: 
      a. O_i(kk, i, j) += I(i, j, 16 * k + kk)
      b. O(i, j) += O_i(kk, i, j)
    The intermediate tensor O_i is of shape (128/16)x3x5 == 8x3x5.

    Example:

    ```
     %0 = linalg.matmul ins(%A, %B: tensor<16x256xf32>, tensor<256x32xf32>)
       outs(%C: tensor<16x32xf32>) -> tensor<16x32xf32>
    ```

    Is transformed to:

    ```
     #map0 = affine_map<(d0, d1, d2, d3) -> (d0, d2 * 4 + d3)>
     #map1 = affine_map<(d0, d1, d2, d3) -> (d2 * 4 + d3, d1)>
     #map2 = affine_map<(d0, d1, d2, d3) -> (d2, d3)>
     #map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
     #map4 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
     #map5 = affine_map<(d0, d1, d2) -> (d0, d1)>
     %0 = linalg.init_tensor [16, 32, 64] : tensor<16x32x64xf32>
     %cst = arith.constant 0.000000e+00 : f32
     %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<16x32x64xf32>) ->
        tensor<16x32x64xf32>
     %2 = linalg.init_tensor [64, 4] : tensor<64x4xi1>

     %3 = linalg.generic {indexing_maps = [#map0, #map1, #map2, #map3],
       iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
       ins(%A, %B, %2 : tensor<16x256xf32>, tensor<256x32xf32>, tensor<64x4xi1>)
       outs(%1 : tensor<16x32x64xf32>) {
         ^bb0(%arg3: f32, %arg4: f32, %arg5: i1, %arg6: f32):
           %5 = arith.mulf %arg3, %arg4 : f32
           %6 = arith.addf %arg6, %5 : f32
           linalg.yield %6 : f32
     } -> tensor<16x32x64xf32>

     %4 = linalg.generic {indexing_maps = [#map4, #map5],
       iterator_types = ["parallel", "parallel", "reduction"]}
       ins(%3 : tensor<16x32x64xf32>)
       outs(%C : tensor<16x32xf32>) {
         ^bb0(%arg3: f32, %arg4: f32):
           %5 = arith.addf %arg3, %arg4 : f32
           linalg.yield %5 : f32
     } -> tensor<16x32xf32>

     return %4 : tensor<16x32xf32>
    ```
  }];

  let arguments = (ins PDL_Operation:$target,
                   DefaultValuedAttr<I64Attr, "{}">:$split_factor,
                   DefaultValuedAttr<I64Attr, "{}">:$insert_split_dimension,
                   UnitAttr:$use_scaling_algorithm,
                   UnitAttr:$use_alloc);
  let results = (outs PDL_Operation:$fill_op,
                      PDL_Operation:$split_linalg_op,
                      PDL_Operation:$combining_linalg_op);

  let assemblyFormat = "$target attr-dict";

  let extraClassDeclaration = [{
    ::mlir::FailureOr<::llvm::SmallVector<::mlir::Operation *>> applyToOne(
        ::mlir::linalg::LinalgOp target, TransformState &state);
  }];
}

def TileOp : Op<Transform_Dialect, "structured.tile",
       [DeclareOpInterfaceMethods<TransformOpInterface>,
        FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface]> {
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

def VectorizeOp : Op<Transform_Dialect, "structured.vectorize",
    [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
     TransformEachOpTrait, TransformOpInterface]> {
  let description = [{
    Indicates that the given `target` op all the ops it contains should be
    vectorized with the configuration specified by the attributes of this op.
    This vectorization only handles structured ops that operate on shaped types
    and does not vectorize loops or straight-line. Internally, it applies a
    set of rewrite patterns, some of which enable vectorization and some of
    which clean up the results. Therefore, it can only be applied to an op with
    the "isolated from above property". If finer granularity is required, it can
    be achieved by outlining the target part of the payload IR into, e.g., a
    function, performing the transformation, and inlining it back. This
    transformation only fails if the entire pattern rewriting failed, i.e., it
    does **not** fail when no ops were vectorized.

    Note that this transformation is invalidating the handles to any payload IR
    operation that is contained inside the vectorization target.
  }];

  let arguments = (ins PDL_Operation:$target,
                   DefaultValuedAttr<BoolAttr, "false">:$vectorize_padding);
  let results = (outs PDL_Operation:$transformed);

  let assemblyFormat = "$target attr-dict";

  let extraClassDeclaration = [{
    ::mlir::FailureOr<Operation *> applyToOne(
      ::mlir::Operation *target, TransformState &state);
  }];
}

#endif // LINALG_TRANSFORM_OPS
