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
include "mlir/Dialect/SCF/IR/DeviceMappingInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

def DecomposeOp : Op<Transform_Dialect, "structured.decompose",
    [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
     TransformOpInterface, TransformEachOpTrait]> {
  let description = [{
    Decomposes named complex operations, such as higher-dimensional
    (depthwise) convolutions, into combinations of lower-dimensional equivalents
    when possible.

    #### Return modes
    
    This operation ignores non-Linalg ops and drops them in the return.
    If all the operations referred to by the `target` PDLOperation decompose
    properly, the transform succeeds. Otherwise the transform silently fails.
    The return handle points to only the subset of successfully produced 
    computational operations, which can be empty.
  }];

  let arguments = (ins PDL_Operation:$target);
  let results = (outs PDL_Operation:$transformed);
  let assemblyFormat = "$target attr-dict";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::linalg::LinalgOp target, 
        ::llvm::SmallVectorImpl<::mlir::Operation *> &results, 
        ::mlir::transform::TransformState &state);
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

def FuseIntoContainingOp :
    Op<Transform_Dialect, "structured.fuse_into_containing_op",
      [DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let description = [{Fuse a producer into a containing operation.}];

  let summary = [{
    Fuses the `producer_op` into the `containing_op`.
    Returns a handle to the fused ops.

    The producer is typically a slice of a tileable op (i.e., implements
    TilingInterface). In that case, this transform computes the accessed
    producer slice inside of the containing op ("tile and fuse"). Otherwise,
    the entire producer is cloned inside the containing op ("clone and fuse").

    The containing op handle must be associated with exactly one payload op. The
    producer op handle may be associated with multiple payload ops. This
    transform fuses producers one-by-one, always picking an unspecified producer
    that has at least one use inside the containing op among the 
    producers.

    Note: If a producer has multiple uses inside the containing op, it is
    currently tiled and/or cloned multiple times into the containing op.
    TODO: Reuse already fused OpResults instead of tiling/cloning a second time
    when possible. Fuse producers according to a topological sorting to achieve
    the largest amount of reuse.

    #### Return modes

    If at least one producer could not be fused, this operation fails silently.
    This is the case when tiling fails or when no producer op could be found
    among the remaining producers that has at least one use within the
    containing op. I.e., "producers" that are not consumed within the containing
    op are rejected by this operation.

    This operation reads and frees the producer handle.
    This operation reads the containing op handle.
  }];

  let arguments = (ins Arg<PDL_Operation, "",
                           [TransformMappingRead,
                            TransformMappingFree]>:$producer_op,
                       Arg<PDL_Operation, "",
                           [TransformMappingRead]>:$containing_op);
  let results = (outs Res<PDL_Operation, "",
                          [TransformMappingAlloc,
                           TransformMappingWrite]>:$fused_op);
  let assemblyFormat = "$producer_op `into` $containing_op attr-dict";

  let builders = [
    OpBuilder<(ins "Value":$producerOp, "Value":$containingOp)>
  ];
}

def GeneralizeOp : Op<Transform_Dialect, "structured.generalize",
    [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
     TransformOpInterface, TransformEachOpTrait]> {
  let description = [{
    Transforms a named structured operation into the generic form with the
    explicit attached region. 
    
    #### Return modes
    
    This operation ignores non-Linalg ops and drops them in the return.
    If all the operations referred to by the `target` PDLOperation generalize
    properly, the transform succeeds. Otherwise the transform silently fails.
    The return handle points to only the subset of successfully produced 
    equivalent generic operations, which can be empty or contain the original
    ops if they were already in generic form.
  }];

  let arguments = (ins PDL_Operation:$target);
  let results = (outs PDL_Operation:$transformed);
  let assemblyFormat = "$target attr-dict";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::linalg::LinalgOp target, 
        ::llvm::SmallVectorImpl<::mlir::Operation *> &results, 
        ::mlir::transform::TransformState &state);
  }];
}

def InterchangeOp : Op<Transform_Dialect, "structured.interchange",
    [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
    TransformOpInterface, TransformEachOpTrait]> {
  let description = [{
    Interchanges the iterators of the operations pointed to by the target handle
    using the iterator interchange attribute.

    #### Return modes
    
    This operation ignores non-linalg::Generic ops and drops them in the return.
    This operation fails if the interchange attribute is invalid.
    If all the operations referred to by the `target` PDLOperation interchange
    properly, the transform succeeds. 
    If any interchange fails, the transform definitely fails.
    The return handle points to only the subset of successfully produced 
    interchanged operations, which can be empty.
  }];

  let arguments =
    (ins PDL_Operation:$target,
         DefaultValuedAttr<I64ArrayAttr, "{}">:$iterator_interchange);
  let results = (outs PDL_Operation:$transformed);

  let assemblyFormat = "$target attr-dict";
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::linalg::GenericOp target, 
        ::llvm::SmallVectorImpl<::mlir::Operation *> &results, 
        ::mlir::transform::TransformState &state);
  }];
}

def MatchInterfaceEnum : I32EnumAttr<"MatchInterfaceEnum", "An interface to match",
    [
      I32EnumAttrCase<"LinalgOp", 0>,
      I32EnumAttrCase<"TilingInterface", 1>
    ]>{
  let cppNamespace = "mlir::transform";
}

def MatchOp : Op<Transform_Dialect, "structured.match",
    [MemoryEffectsOpInterface,
     NavigationTransformOpTrait,
     DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let description = [{
    Match op with the specified constraints, within the target op.

    The following constraints are supported:
      - interface: an optional MatchInterfaceEnum specifying an enum
        representation for an interface to target.
      - ops: an optional StrArrayAttr specifying the concrete name of an op.
        Multiple names can be specified. Matched ops must have one of specified
        names.
      - attribute: the matched op must have all specified attributes (with their
        specified values).
      - filter_result_type: the matched op must return exactly this one type.
      
    Note: Only ops that satisfy all specified constraints are matched.

    TODO: Extend with regions to allow a limited form of constraints.

    #### Return modes

    This op traverses the ops nested under `target` and returns the handles to
    all the operations that match the requirements.

    This op fails if the target is not a handle to exactly one operation. 
    Otherwise it succeeds.

    This operation does not consume the target handle and produces new handles:
    it is a navigation op.
  }];

  let arguments = (ins PDL_Operation:$target,
                       OptionalAttr<StrArrayAttr>:$ops,
                       OptionalAttr<MatchInterfaceEnum>:$interface,
                       OptionalAttr<DictionaryAttr>:$op_attrs,
                       OptionalAttr<TypeAttr>:$filter_result_type);
  // TODO: variadic results when needed.
  let results = (outs PDL_Operation:$results);

  let builders = [
    OpBuilder<(ins "Value":$target, "ArrayRef<StringRef>":$opNames)>
  ];

  let assemblyFormat = [{
    (`ops` `{` $ops^ `}`)?
    (`interface` `{` $interface^ `}`)?
    (`attributes` $op_attrs^)?
    (`filter_result_type` `=` $filter_result_type^)?
    `in` $target attr-dict
  }];
}

def MultiTileSizesOp : Op<Transform_Dialect, "structured.multitile_sizes",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     TransformOpInterface, TransformEachOpTrait]> {
  let description = [{
    Emits the IR computing the tile sizes `s1` and `s2` such that:

      - there exists a combination of `n` tiles of size `s1` and `m` tiles of
        size `s2` that covers the entirety of the iteration space `dimension` of
        the target structured op;
      - `s1`, `s2` is less than or equal to `target_size`;
      - `s1` and `s2` are divisible by `divisor.

    For example, for a dimension of size 54 with target size 12 and divisor 2,
    this can emit the IR computing the tile size 10, used for 3 tiles, and 12,
    used for 2 tiles, totally 10*3 + 12*2 = 54. Note that when the divisor does
    not divide the original dimension size, it is impossible to compute such
    tile sizes. An assertion is emitted to guard against this in the dynamic
    case.

    Expects the target size and the divisor to be strictly positive. Folds the
    IR as much as possible, normally obtaining constant sizes and numbers of
    tiles for a statically known dimension.

    This does *not* consume the target handle and produces three handles each
    pointing to single-result index-typed operations (which may be arithmetic
    constant operations) defining the two respective tile sizes and the product
    of the first tile size with the number of tiles of that size (useful for
    splitting the iteration space).

    This operation composes with the regular tiling when applied per-dimension:

    ```mlir
    %sz1, %sz2, %split = structured.multitile_sizes %target
                         { target_size = 10, dimension = 1 }
    %low, %high = structured.split %target after %split { dimension = 1 }
    %tiled_low = structured.tile %low [0, %sz1]
    %tiled_high = structured.tile %high [0, %sz2]
    %common = merge_handles %tiled_low, %tiled_high

    %sz3, %sz4, %split = structured.multitile_size %target
                         { target_size = 42, dimension = 0 }
    %sz3r, %sz4r, %splitr = replicate num(%common) %sz3, %sz4, %splitr
    structured.split %common after %splitr { dimension = 0 }
    // ...
    ```
  }];

  let arguments = (ins PDL_Operation:$target,
                       I64Attr:$dimension,
                       I64Attr:$target_size,
                       DefaultValuedAttr<I64Attr, "1">:$divisor);
  let results = (outs PDL_Operation:$low_size,
                      PDL_Operation:$high_size,
                      PDL_Operation:$split_point);
  let assemblyFormat = "$target attr-dict";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::linalg::LinalgOp target, 
        ::llvm::SmallVector<::mlir::Operation *> &results,
        TransformState &state);
  }];
}

def PadOp : Op<Transform_Dialect, "structured.pad",
    [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
     TransformOpInterface, TransformEachOpTrait]> {
  let description = [{
    Pads the operations pointed to by the target handle using the options
    provides as operation attributes.

    #### Return modes
    
    This operation ignores non-Linalg ops and drops them in the return.
    This operation may produce a definiteFailure if the padding fails for any
    reason.
    If all the operations referred to by the `target` PDLOperation pad
    properly, the transform succeeds. Otherwise the transform silently fails.
    The return handle points to only the subset of successfully produced 
    padded operations, which can be empty.
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
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::linalg::LinalgOp target, 
        ::llvm::SmallVectorImpl<::mlir::Operation *> &results, 
        ::mlir::transform::TransformState &state);
  }];
}

def PromoteOp : Op<Transform_Dialect, "structured.promote",
    [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
    TransformOpInterface, TransformEachOpTrait]> {
  let description = [{
    Promotes the specified operands of the target into a separate memory buffer.

    At this point, this transform does not allow customizing alloc/dealloc 
    functions nor the behavior on copy in/out operations.

    #### Return modes
    
    This operation applies to a single Linalg op that satisfies the 
    `promoteSubviewsPrecondition`, otherwise it fails.
    
    If the operations referred to by the `target` PDLOperation promote
    properly, the transform succeeds. 

    When successful, the return handle points to the $target operation that 
    was modified inplace.
  }];

  let arguments = (ins PDL_Operation:$target,
                       DefaultValuedAttr<I64ArrayAttr, "{}">:$operands_to_promote,
                       DefaultValuedAttr<BoolArrayAttr, "{}">:$use_full_tile_buffers,
                       UnitAttr:$use_full_tiles_by_default,
                       UnitAttr:$use_alloca,
                       OptionalAttr<I64Attr>:$alignment);
  let results = (outs PDL_Operation:$transformed);

  let assemblyFormat = "$target attr-dict";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::linalg::LinalgOp target, 
        ::llvm::SmallVectorImpl<::mlir::Operation *> &results, 
        ::mlir::transform::TransformState &state);
  }];
}

def ScalarizeOp : Op<Transform_Dialect, "structured.scalarize",
    [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
     TransformOpInterface, TransformEachOpTrait]> {
  let description = [{
    Indicates that ops of a specific kind in the given function should be
    scalarized (i.e. their dynamic dimensions tiled by 1).

    #### Return modes:
    
    This operation ignores non-Linalg ops and drops them in the return.
    This operation produces `definiteFailure` if the scalarization fails for any
    reason.
    If all the operations referred to by the `target` PDLOperation scalarize
    properly, the transform succeeds. Otherwise the transform silently fails.

    The return handle points to only the subset of successfully produced 
    tiled-by-1 operations, which can be empty.

    This operation does not return handles to the tiled loop.
    We make this design choice because it is hard to know ahead of time the
    number of loops that will be produced (it depends on the number of dynamic
    dimensions after multiple transformations have been applied).
    Loops can always be recovered by navigating from the tiled operations if
    needed.
  }];

  let arguments = (ins PDL_Operation:$target);
  let results = (outs PDL_Operation:$result);

  let assemblyFormat = "$target attr-dict";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::linalg::LinalgOp target, 
        ::llvm::SmallVectorImpl<::mlir::Operation *> &results, 
        ::mlir::transform::TransformState &state);
  }];
}

def SplitOp : Op<Transform_Dialect, "structured.split",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let description = [{
    Indicates that the given `target` op should be split into two complementary
    parts, which combined cover the entire iteration domain of the original op.
    The split is performed along the iteration space dimension provided as
    attribute. In case of dimension overflow, the transformation fails. The
    split is performed at the dimension iterator value specified as either the
    static split point attribute when it is known at transform IR construction
    time or as the handle to an operation producing a single index-typed value
    when it is computed by payload IR. In the latter case, the static split
    point must be set to `ShapedType::kDynamicSize` and the dynamic size handle
    must point to as many value-producing operations as there are structured
    operations pointed to by the target handle.

    The operation consumes the target handle, but preserves the split point
    handle if provided. It produces two new handles pointing to the two parts
    of the structured op after splitting, in the same order as the target
    operand, with the first handle corresponding to the part with lower
    iteration space indices.
  }];

  let arguments = (ins PDL_Operation:$target,
                       I64Attr:$dimension,
                       Optional<PDL_Operation>:$dynamic_split_point,
                       I64Attr:$static_split_point);
  let results = (outs PDL_Operation:$first, PDL_Operation:$second);
  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
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
      - inner_parallel: specifies whether the parallel dimension is before or
        after the reduction dimension in the splitting op.
      - use_scaling_algorithm: whether to use a scaling based formulation that 
        does not create an ExpandShapeOp (default: do not use scaling)
      - use_alloc: whether to use an alloc op to allocate the temporary 
        tensor (default: do not use alloc op)

    #### Return modes
    
    This operation ignores non-Linalg ops and drops them in the return.
    This operation produces `definiteFailure` if the splitting fails for any
    reason.

    If all the operations referred to by the `target` PDLOperation split
    properly, the transform succeeds. Otherwise the transform silently fails.
    The 4 returned handles points to only the subset of successfully produced 
    computational operations, which can all be empty.
    This 4 returned handles point to:
      - the init op (or tensor_alloc op if use_alloc = true), 
      - the fill op used to initialize the neutral element, 
      - the split op and 
      - the result-combining op.

    #### Example (default: `use_scaling_algorithm = false, use_alloc = false`):
    
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
      %1 = tensor.empty() : tensor<4xf32>
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

    #### Example (`use_scaling_algorithm = true, use_alloc = true`):
    
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

    #### Example:

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
     %0 = tensor.empty() : tensor<16x32x64xf32>
     %cst = arith.constant 0.000000e+00 : f32
     %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<16x32x64xf32>) ->
        tensor<16x32x64xf32>
     %2 = tensor.empty() : tensor<64x4xi1>

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
                   UnitAttr:$inner_parallel,
                   UnitAttr:$use_scaling_algorithm,
                   UnitAttr:$use_alloc);
  let results = (outs PDL_Operation:$init_or_alloc_op,
                      PDL_Operation:$fill_op,
                      PDL_Operation:$split_linalg_op,
                      PDL_Operation:$combining_linalg_op);

  let assemblyFormat = "$target attr-dict";

  let builders = [
    OpBuilder<(ins "Value":$target,
                   "int64_t":$splitFactor,
                   "int64_t":$insertSplitDimension,
                   CArg<"bool", "false">:$innerParallel,
                   CArg<"bool", "false">:$useScalingAlgorithm,
                   CArg<"bool", "false">:$useAlloc)>
  ];

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::linalg::LinalgOp target, 
        ::llvm::SmallVectorImpl<::mlir::Operation *> &results, 
        ::mlir::transform::TransformState &state);
  }];
}


def TileReductionUsingScfOp : Op<Transform_Dialect, "structured.tile_reduction_using_scf",
       [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
        TransformEachOpTrait, TransformOpInterface]> {
  let description = [{
    Indicates that the given `target` op should be transformed with the 
    `tileReduction` transformation with the tile size provided as attribute.
    
    This transformation tiles the `target` along the reduction dimensions. It
    creates a tensor initialized with the identity value. Then it creates nested
    loops with a parallel version of `target` op inside. The parallel op
    dimensions are less or equal to the tile size passed by user.
    After the loop a merge operation is created to do a final reduction with the
    partial reductions.
    The initial tensor always uses the tile size dimension. This may overallocate
    if the tile size is greater than the reduction dimension.

    #### Return modes

    This 3 returned handles point to:
      - the fill op used to initialize the neutral element, 
      - the parallel tiled op and 
      - the result-combining op.

    #### Example:
    
    ```
      %red = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                              affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]}
      ins(%arg0 : tensor<?x?xf32>)
      outs(%out : tensor<?xf32>) {
        ^bb0(%arg7: f32, %arg9: f32):
        %1 = arith.addf %arg7, %arg9 : f32
        linalg.yield %1 : f32
      } -> tensor<?xf32>
      return %red : tensor<?xf32>
    ```
  
    is transformed into:
    
    ```
      %0 = tensor.empty(%dim_1) : tensor<?x5xf32>
      %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<?x5xf32>) -> tensor<?x5xf32>
      %2 = scf.for %arg2 = %c0 to %dim_0 step %c5 iter_args(%arg3 = %1) -> (tensor<?x5xf32>) {
        %extracted_slice = tensor.extract_slice %1[0, 0] [%dim, 5] [1, 1] : tensor<?x5xf32> to tensor<?x5xf32>
        %extracted_slice_2 = tensor.extract_slice %arg0[0, %arg2] [%dim, 5] [1, 1] : tensor<?x?xf32> to tensor<?x5xf32>
        %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, 
                                              affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]}
        ins(%extracted_slice_2 : tensor<?x5xf32>)
        outs(%extracted_slice : tensor<?x5xf32>) {
        ^bb0(%in: f32, %out: f32):
          %5 = arith.addf %in, %out : f32
          linalg.yield %5 : f32
        } -> tensor<?x5xf32>
        %dim_3 = tensor.dim %1, %c0 : tensor<?x5xf32>
        %inserted_slice = tensor.insert_slice %4 into %arg3[0, 0] [%dim_3, 5] [1, 1] : tensor<?x5xf32> into tensor<?x5xf32>
        scf.yield %inserted_slice : tensor<?x5xf32>
      }
      %3 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                            affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]}
      ins(%2 : tensor<?x5xf32>)
      outs(%arg1 : tensor<?xf32>) {
      ^bb0(%in: f32, %out: f32):
        %4 = arith.addf %in, %out : f32
        linalg.yield %4 : f32
      } -> tensor<?xf32>
    ```
  }];

  let arguments = (ins PDL_Operation:$target,
                   DefaultValuedAttr<I64ArrayAttr, "{}">:$tile_sizes);
  let results = (outs PDL_Operation:$fill_op,
                      PDL_Operation:$split_linalg_op,
                      PDL_Operation:$combining_linalg_op);

  let assemblyFormat = "$target attr-dict";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::linalg::LinalgOp target, 
        ::llvm::SmallVectorImpl<::mlir::Operation *> &results, 
        ::mlir::transform::TransformState &state);
  }];
}

def TileOp : Op<Transform_Dialect, "structured.tile",
       [DeclareOpInterfaceMethods<TransformOpInterface>,
        DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let description = [{
    Indicates that the given `target` op should be tiled with the given sizes.
    This transform generates a loop nest with a smaller ("tiled") target
    operation in its body. Currently limited to LinalgOps.

    Tile sizes may be known at transformation time, in which case they are
    expected to be provided in the `static_size` attribute, or not, in which
    case the tile value must be computed by the payload IR and the handle to the
    operation computing it must be provided through `dynamic_sizes`. When the
    sizes are not known statically, the corresponding entry in the
    `static_sizes` attribute must be set to `ShapedType::kDynamicSize`. Only
    the dynamic sizes must be provided in `dynamic_sizes`, i.e., there should
    be as many handles as `ShapedType::kDynamicSize` values in the
    `static_sizes` attribute. A static size of `0` indicates that the dimension
    should not be tiled. No loop will be generated for such dimensions. If all
    tile sizes are `0`, this transform is effectively a no-op.

    This op returns handles to the tiled op (in the generated loop nest) and the
    generated loops. The number of loops is the number of tile sizes that are
    statically known to be non-zero.

    #### Return modes

    On success, the resulting handles are associated with co-indexed lists of
    tiled operations and loops around them.

    This operation only supports Linalg ops and produces a silenceable failure
    if the input contains any non-Linalg ops. The ops preceding it in the list
    associated with the `target` handle will have been tiled.

    This operation produces a silenceable failure if the `dynamic_sizes` handles
    are associated with lists of payload operations of a size different than
    that of the list associated with the `target` handle.

    If the internal implementation of tiling for any of the operations fails,
    produces a definite failure.
  }];

  let arguments = (ins PDL_Operation:$target,
                   Variadic<PDL_Operation>:$dynamic_sizes,
                   DefaultValuedAttr<I64ArrayAttr, "{}">:$static_sizes,
                   DefaultValuedAttr<I64ArrayAttr, "{}">:$interchange);
  let results = (outs PDL_Operation:$tiled_linalg_op,
                      Variadic<PDL_Operation>:$loops);

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    /// Returns the list of tile sizes, which may be static (Attribute) or
    /// dynamic (Value).
    SmallVector<OpFoldResult> getMixedSizes();
  }];
}

def TileToForeachThreadOp :
    Op<Transform_Dialect, "structured.tile_to_foreach_thread_op",
      [AttrSizedOperandSegments,
       DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
       TransformOpInterface]> {
  let description = [{
    Tile a TilingInterface op to a tiled `scf.foreach_thread`.

    Tiling is applied by either specifying `num_threads` or `tile_size`. If
    `num_threads` is specified, then the tile size for each dimension `i` is
    calculated dynamically via `ceilDiv(dimSize[i], num_threads[i])`.
    `num_threads` and `tile_size` can be either static index attributes or SSA
    values of PDL operation handle type (or a mix thereof). Operation handles
    must be mapped to exactly one op that has exactly one result of index type.

    Static zero tile sizes indicate that the dimension is not tiled and can be
    thought of as tiling by the full size of data.

    It is the user's responsibility to ensure that `num_threads/tile_sizes` is
    a valid tiling specification (i.e. that only tiles parallel dimensions, 
    e.g. in the Linalg case).

    If non-empty, the `mapping` is added as an attribute to the
    resulting `scf.foreach_thread`.

    #### Return modes

    This operation ignores ops that do not implement the TilingInterface and
    drops them in the return.

    If all the operations referred to by the `target` PDLOperation tile
    successfully, the transform succeeds.
    Otherwise the transform silently fails.
    
    The two returned handles point to only the subset of successfully produced
    tiled operations, which can all be empty.
    
    These two returned handles point to:
      - the new scf.foreach_thread op,
      - the tiled op that implements TilingInterface.

    #### Example using `num_threads`

    ```
    %0 = pdl_match @match_matmul in %arg1
    %3:2 = transform.structured.tile_to_foreach_thread_op %0 num_threads [10, 20]
    ```

    #### Example using `tile_sizes`

    ```
    %0 = pdl_match @match_matmul in %arg1
    %sz = pdl_match @match_size_op in %arg1
    %3:2 = transform.structured.tile_to_foreach_thread_op %0 tile_sizes [0, %sz, 20]
    ```
  }];

  let arguments = (ins PDL_Operation:$target,
                   Variadic<PDL_Operation>:$num_threads,
                   Variadic<PDL_Operation>:$tile_sizes,
                   DefaultValuedAttr<I64ArrayAttr, "{}">:$static_num_threads,
                   DefaultValuedAttr<I64ArrayAttr, "{}">:$static_tile_sizes,
                   OptionalAttr<DeviceMappingArrayAttr>:$mapping);
  let results = (outs PDL_Operation:$foreach_thread_op,
                      PDL_Operation:$tiled_op);

  let builders = [
    OpBuilder<(ins "Value":$target,
                   "ArrayRef<int64_t>":$staticTileSizes,
                   CArg<"::mlir::transform::TileSizesSpec", 
                        "::mlir::transform::TileSizesSpec()">,
                   CArg<"ArrayAttr", "{}">:$mapping)>,
    OpBuilder<(ins "Value":$target,
                   "ArrayRef<OpFoldResult>":$mixedTileSizes,
                   CArg<"::mlir::transform::TileSizesSpec", 
                        "::mlir::transform::TileSizesSpec()">,
                   CArg<"ArrayAttr", "{}">:$mapping)>,
    OpBuilder<(ins "Value":$target,
                   "ArrayRef<int64_t>":$staticNumThreads,
                   CArg<"::mlir::transform::NumThreadsSpec", 
                        "::mlir::transform::NumThreadsSpec()">,
                   CArg<"ArrayAttr", "{}">:$mapping)>,
    OpBuilder<(ins "Value":$target,
                   "ArrayRef<OpFoldResult>":$mixedNumThreads,
                   CArg<"::mlir::transform::NumThreadsSpec", 
                        "::mlir::transform::NumThreadsSpec()">,
                   CArg<"ArrayAttr", "{}">:$mapping)>,
  ];

  let assemblyFormat = [{
    $target oilist(
        `num_threads` custom<DynamicIndexList>($num_threads,
                                               $static_num_threads,
                                               "ShapedType::kDynamicSize") |
         `tile_sizes` custom<DynamicIndexList>($tile_sizes,
                                               $static_tile_sizes,
                                               "ShapedType::kDynamicSize"))
    (`(` `mapping` `=` $mapping^ `)`)? attr-dict
  }];
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure apply(
        ::mlir::transform::TransformResults &transformResults,
        ::mlir::transform::TransformState &state);

    ::llvm::SmallVector<::mlir::OpFoldResult> getMixedNumThreads();
    ::llvm::SmallVector<::mlir::OpFoldResult> getMixedTileSizes();
  }];
}

def TileToScfForOp : Op<Transform_Dialect, "structured.tile_to_scf_for",
       [DeclareOpInterfaceMethods<TransformOpInterface>,
        DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let description = [{
    Indicates that the given `target` op should be tiled with the given sizes.
    This transform generates a loop nest with a smaller ("tiled") target
    operation in its body. The target must implement TilingInterface.

    Tile sizes may be known at transformation time, in which case they are
    expected to be provided in the `static_size` attribute, or not, in which
    case the tile value must be computed by the payload IR and the handle to the
    operation computing it must be provided through `dynamic_sizes`. When the
    sizes are not known statically, the corresponding entry in the
    `static_sizes` attribute must be set to `ShapedType::kDynamicSize`. Only
    the dynamic sizes must be provided in `dynamic_sizes`, i.e., there should
    be as many handles as `ShapedType::kDynamicSize` values in the
    `static_sizes` attribute. A static size of `0` indicates that the dimension
    should not be tiled. No loop will be generated for such dimensions. If all
    tile sizes are `0`, this transform is effectively a no-op.

    This op returns handles to the tiled op (in the generated loop nest) and the
    generated loops. The number of loops is the number of tile sizes that are
    statically known to be non-zero.

    #### Return modes

    On success, the resulting handles are associated with co-indexed lists of
    tiled operations and loops around them.

    This operation only supports TilingInterface ops and produces a silenceable
    failure if the input contains any non-TilingInterface ops. The ops preceding
    it in the list associated with the `target` handle will have been tiled.

    This operation produces a silenceable failure if the `dynamic_sizes` handles
    are associated with lists of payload operations of a size different than
    that of the list associated with the `target` handle.

    If the internal implementation of tiling for any of the operations fails,
    produces a definite failure.
  }];

  let arguments = (ins PDL_Operation:$target,
                   Variadic<PDL_Operation>:$dynamic_sizes,
                   DefaultValuedAttr<I64ArrayAttr, "{}">:$static_sizes,
                   DefaultValuedAttr<I64ArrayAttr, "{}">:$interchange);
  let results = (outs PDL_Operation:$tiled_linalg_op,
                      Variadic<PDL_Operation>:$loops);

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    /// Returns the list of tile sizes, which may be static (Attribute) or
    /// dynamic (Value).
    SmallVector<OpFoldResult> getMixedSizes();
  }];
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

    This transformation supports the following attributes:
      - `vectorize_padding`: a UnitAttr to activate the vectorization of 
      `tensor.pad` ops. Different pipelines may prefer to lower such ops to 
      loops.
      - `disable_multi_reduction_to_contract_patterns`: a UnitAttr to deactivate
      the rewrite of `vector.multi_reduction` to `vector.contract`. This is 
      intended to be used in tests only.
      - `disable_transfer_permutation_map_lowering_patterns`: a UnitAttr to
      deactivate the rewrite of `vector.transfer` with permutation maps into
      explicit `vector.transpose` operations. This is intended to be used in
      tests only but may be promotoed to a first class attribute in the future.

    #### Return modes:
    
    This operation produces `definiteFailure` if vectorization fails for any
    reason.
    The operation always returns the handle to the target op that is expected 
    to be isolated from above.
  }];

  let arguments = (ins PDL_Operation:$target,
                   UnitAttr:$vectorize_padding,                   
                   UnitAttr:$disable_multi_reduction_to_contract_patterns,                   
                   UnitAttr:$disable_transfer_permutation_map_lowering_patterns);
  let results = (outs PDL_Operation:$transformed);

  let assemblyFormat = "$target attr-dict";

  let builders = [
    OpBuilder<(ins "Value":$target, CArg<"bool", "false">:$vectorizePadding)>
  ];
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::Operation *target, 
        ::llvm::SmallVectorImpl<::mlir::Operation *> &results, 
        ::mlir::transform::TransformState &state);
  }];
}

#endif // LINALG_TRANSFORM_OPS
