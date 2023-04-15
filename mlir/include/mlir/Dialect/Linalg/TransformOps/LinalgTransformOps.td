//===- LinalgTransformOps.td - Linalg transform ops --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LINALG_TRANSFORM_OPS
#define LINALG_TRANSFORM_OPS

include "mlir/Dialect/Linalg/TransformOps/LinalgTransformEnums.td"
include "mlir/Dialect/Transform/IR/TransformAttrs.td"
include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/IR/TransformInterfaces.td"
include "mlir/Dialect/Transform/IR/TransformTypes.td"
include "mlir/Dialect/PDL/IR/PDLTypes.td"
include "mlir/Dialect/SCF/IR/DeviceMappingInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/RegionKindInterface.td"

// This is roughly similar to OpFoldResult assuming the handle produces a single
// value in the payload IR.
def TransformParamTypeOrAnyHandle : Type<
    Or<[TransformHandleTypeInterface.predicate,
        Transform_ParamType.predicate]>,
    "transform 'param' type or any handle type">;

//===----------------------------------------------------------------------===//
// BufferizeToAllocationOp
//===----------------------------------------------------------------------===//

def BufferizeToAllocationOp : Op<Transform_Dialect,
    "structured.bufferize_to_allocation",
    [DeclareOpInterfaceMethods<TransformOpInterface>,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let description = [{
    This transform materializes an allocation for the targeted tensor value. It
    replaces all original uses of the target with the newly allocated buffer,
    wrapped in a `bufferization.to_tensor` op. It returns a handle to the result
    of the `to_tensor` op.

    Example:
    ```
    %0 = "some_op"() : () -> (tensor<10xf32>)
    "some_use"(%0) : (tensor<10xf32>) -> ()
    ```

    Is rewritten to:
    ```
    %0 = "some_op"() : () -> (tensor<10xf32>)
    %1 = memref.alloc() : memref<10xf32>
    memref.tensor_store %0, %1 : memref<10xf32>
    %2 = bufferization.to_tensor %1 restrict writable : memref<10xf32>
    "some_use"(%2) : (tensor<10xf32>) -> ()
    ```

    This transform has optimized lowerings for certain targets that are results
    of non-DPS ops. For such targets, not only a buffer allocation is emitted
    but also the defining op is bufferized. This is to avoid a second
    allocation for the missing destination of the non-DPS op (when subsequently
    running a bufferization pass/transform). Currently supported ops with
    optimized lowerings:
    - tensor.pad

    An optional memory space attribute can be specified for the materialized
    buffer allocation.

    #### Return modes

    This operation consumes the `target` handle and produces the `transformed`
    handle. It always succeeds.
  }];

  let arguments = (ins Transform_AnyValue:$target,
                       OptionalAttr<AnyAttr>:$memory_space);
  let results = (outs Transform_AnyValue:$transformed);
  let assemblyFormat = "$target attr-dict";
}

//===----------------------------------------------------------------------===//
// DecomposeOp
//===----------------------------------------------------------------------===//

def DecomposeOp : Op<Transform_Dialect, "structured.decompose",
    [FunctionalStyleTransformOpTrait, 
     MemoryEffectsOpInterface,
     TransformOpInterface, 
     TransformEachOpTrait]> {
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
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

//===----------------------------------------------------------------------===//
// FuseOp
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// FuseIntoContainingOp
//===----------------------------------------------------------------------===//

def FuseIntoContainingOp :
    Op<Transform_Dialect, "structured.fuse_into_containing_op",
      [DeclareOpInterfaceMethods<TransformOpInterface>,
       DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Fuse a producer into a containing operation.";

  let description = [{
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

  let arguments = (ins PDL_Operation:$producer_op,
                       PDL_Operation:$containing_op);
  let results = (outs PDL_Operation:$fused_op);
  let assemblyFormat = "$producer_op `into` $containing_op attr-dict";

  let builders = [
    OpBuilder<(ins "Value":$producerOp, "Value":$containingOp)>
  ];
}

//===----------------------------------------------------------------------===//
// GeneralizeOp
//===----------------------------------------------------------------------===//

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
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

//===----------------------------------------------------------------------===//
// InterchangeOp
//===----------------------------------------------------------------------===//

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
         ConfinedAttr<DefaultValuedOptionalAttr<DenseI64ArrayAttr, "{}">,
                      [DenseArrayNonNegative<DenseI64ArrayAttr>]>:$iterator_interchange);
  let results = (outs PDL_Operation:$transformed);

  let assemblyFormat = [{ 
    $target 
    (`iterator_interchange` `=` $iterator_interchange^)? attr-dict
  }];
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::linalg::GenericOp target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

//===----------------------------------------------------------------------===//
// LowerPackOp
//===----------------------------------------------------------------------===//
def LowerPackOp : Op<Transform_Dialect, "structured.lower_pack", [
                         FunctionalStyleTransformOpTrait,
                         MemoryEffectsOpInterface,
                         TransformEachOpTrait,
                         TransformOpInterface]> {
  let description = [{
    Rewrite a tensor.pack into tensor.pad + tensor.expand_shape + linalg.transpose.

    #### Return modes

    This operation ignores non-pack ops and drops them in the return.
    This operation produces a silenceableFailure if the rewrite fails for any
    reason.
    If all the operations referred to by the `target` are rewritten, the
    transform succeeds.
    Return handles to the newly produced pad, expand_shape and transpose ops.
  }];

  let arguments = (ins Transform_ConcreteOpType<"tensor.pack">:$target);
  let results = (outs Transform_ConcreteOpType<"tensor.pad">:$pad_op,
                      Transform_ConcreteOpType<"tensor.expand_shape">:$expand_shape_op,
                      Transform_ConcreteOpType<"linalg.transpose">:$transpose_op);
  let assemblyFormat = [{
    $target attr-dict `:` functional-type(operands, results)
  }];

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::tensor::PackOp target,
        ::mlir::transform::ApplyToEachResultList &transformResults,
        ::mlir::transform::TransformState &state);
  }];
}

//===----------------------------------------------------------------------===//
// LowerUnPackOp
//===----------------------------------------------------------------------===//
def LowerUnPackOp : Op<Transform_Dialect, "structured.lower_unpack", [
                         FunctionalStyleTransformOpTrait,
                         MemoryEffectsOpInterface,
                         TransformEachOpTrait,
                         TransformOpInterface]> {
  let description = [{
    Lower a tensor.unpack into empty + linalg.transpose + tensor.collapse_shape + 
    tensor.extract_slice.

    #### Return modes

    This operation ignores non-unpack ops and drops them in the return.
    This operation produces a silenceableFailure if the rewrite fails for any
    reason.
    If all the operations referred to by the `target` are rewritten, the
    transform succeeds.
    Return handles to the newly produced empty, transpose, collapse_shape and extract_slice ops.
  }];

  let arguments = (ins Transform_ConcreteOpType<"tensor.unpack">:$target);
  let results = (outs Transform_ConcreteOpType<"tensor.empty">:$empty_op,
                      Transform_ConcreteOpType<"linalg.transpose">:$transpose_op,
                      Transform_ConcreteOpType<"tensor.collapse_shape">:$collapse_shape_op,
                      Transform_ConcreteOpType<"tensor.extract_slice">:$extract_slice_op);
  let assemblyFormat = [{ 
    $target attr-dict `:` functional-type(operands, results)
  }];

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::tensor::UnPackOp target,
        ::mlir::transform::ApplyToEachResultList &transformResults,
        ::mlir::transform::TransformState &state);
  }];
}

//===----------------------------------------------------------------------===//
// MatchOp
//===----------------------------------------------------------------------===//

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

  let arguments = (ins TransformHandleTypeInterface:$target,
                       OptionalAttr<StrArrayAttr>:$ops,
                       OptionalAttr<MatchInterfaceEnum>:$interface,
                       OptionalAttr<DictionaryAttr>:$op_attrs,
                       OptionalAttr<TypeAttr>:$filter_result_type);
  // TODO: variadic results when needed.
  let results = (outs TransformHandleTypeInterface:$results);

  let builders = [
    OpBuilder<(ins "Value":$target, "ArrayRef<StringRef>":$opNames)>
  ];

  let assemblyFormat = [{
    (`ops` `{` $ops^ `}`)?
    (`interface` `{` $interface^ `}`)?
    (`attributes` $op_attrs^)?
    (`filter_result_type` `=` $filter_result_type^)?
    `in` $target attr-dict
    `:` functional-type($target, results)
  }];
}

//===----------------------------------------------------------------------===//
// MultiTileSizesOp
//===----------------------------------------------------------------------===//

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
                       : !transform.any_op, !transform.param<i64>,
                         !transform.param<i64>, !transform.param<i64>
    %low, %high = structured.split %target after %split { dimension = 1 }
                : !transform.any_op, !transform.param<i64>
    %tiled_low, %loop1 = structured.tile %low [0, %sz1]
                       : (!transform.any_op, !transform.param<i64>)
                      -> (!transform.any_op, !transform.any_op)
    %tiled_high, %loop2 = structured.tile %high [0, %sz2]
                        : (!transform.any_op, !transform.param<i64>)
                       -> (!transform.any_op, !transform.any_op)
    %common = merge_handles %tiled_low, %tiled_high : !transform.any_op

    %sz3, %sz4, %split = structured.multitile_size %target
                         { target_size = 42, dimension = 0 }
                       : !transform.any_op, !transform.any_op,
                         !transform.any_op, !transform.any_op
    %sz3r, %sz4r, %splitr = replicate num(%common) %sz3, %sz4, %splitr
             : !transform.any_op, !transform.any_op, !transform.any_op
    structured.split %common after %splitr { dimension = 0 }
             : !transform.any_op, !transform.any_op
    // ...
    ```
  }];

  let arguments = (ins TransformHandleTypeInterface:$target,
                       I64Attr:$dimension,
                       I64Attr:$target_size,
                       DefaultValuedAttr<I64Attr, "1">:$divisor);
  let results = (outs TransformParamTypeOrAnyHandle:$low_size,
                      TransformParamTypeOrAnyHandle:$high_size,
                      TransformParamTypeOrAnyHandle:$split_point);
  let hasVerifier = 1;
  let assemblyFormat =
    "$target attr-dict `:` custom<MultitileSizesTypes>("
    "type($target), type($low_size), type($high_size), type($split_point))";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::linalg::LinalgOp target,
        ::mlir::transform::ApplyToEachResultList &results,
        TransformState &state);
  }];
}

//===----------------------------------------------------------------------===//
// PackOp
//===----------------------------------------------------------------------===//

def PackOp : Op<Transform_Dialect, "structured.pack", [
                DeclareOpInterfaceMethods<TransformOpInterface>,
                DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let description = [{
    Pack a LinalgOp by applying a data tiling transformation on the op and
    packing the operands according to the `packed_sizes` specification.
    
    Iterator dimensions are tiled in their canonical order in the op spec.
    Operands are packed according to the same canonical order of the op iterator
    dimensions.

    Specifying a packed size of 0 for an iterator removes it from consideration
    for packing.

    `tensor.pack` (resp. `tensor.unpack`) operations are inserted for the operands
    (resp. results) that need to be packed (resp. unpacked) according to the
    `packed_sizes` specification.

    #### Example

    Consider a `linalg.matmul` with indexing maps:
    ```
      //              M   N   K       M   K
      // affine_map<(d0, d1, d2) -> (d0, d2)>
      //                              K   N
      // affine_map<(d0, d1, d2) -> (d2, d1)>
      //                              M   N
      // affine_map<(d0, d1, d2) -> (d0, d1)>
      %0 = linalg.matmul  ins(%A, %B: tensor<?x?xf32>, tensor<?x?xf32>)
                         outs(    %C: tensor<?x?xf32>)
    ```

    Specifying packed_sizes [2, 3, 4] results in tiling the iterator dimensions
    M, N and K, in this order, in both the op and its operands.
    ```
      //              M   N   K   m   n   k       M   K   m   k
      // affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
      //                                          K   N   n   k
      // affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d1, d4, d5)>
      //                                          M   N   m   n
      // affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
      %0 = linalg.generic_representing_some_higher_d_matmul  
            ins(%A, %B: tensor<?x?x2x4xf32>, tensor<?x?x4x3xf32>)
           outs(    %C: tensor<?x?x2x3xf32>)
    ```
    In particular, note that the second operand `B` has shape `KxNxnxk` (and not
    `KxNxkxn` as one could expect by looking **only** at the operand).

    Other layouts can be obtained unsurprisingly from this canonical 
    transformation by composing the resulting operation with a (future) 
    `transform.structured.pack_transpose` op.
    This composition allows separating concerns and composes better compared
    to adding additional permutation attributes to this transform op.

    #### Return modes

    This operation applies to a single Linalg op, otherwise it fails.
    This operation may produce a definiteFailure if the packing fails for any
    reason.

    The returned handle point to the packed LinalgOp.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target,
                   Variadic<PDL_Operation>:$packed_sizes,
                   DefaultValuedAttr<DenseI64ArrayAttr, "{}">:$static_packed_sizes);
  let results = (outs TransformHandleTypeInterface:$packed_op);
  let assemblyFormat = [{
    $target 
    `packed_sizes` `=` custom<DynamicIndexList>($packed_sizes,
                                                $static_packed_sizes)
    attr-dict
    `:` functional-type($target, results)
  }];

  let builders = [
    OpBuilder<(ins "Value":$target,
                   "ArrayRef<OpFoldResult>":$mixedPackedSizes)>
  ];

  let extraClassDeclaration = [{
    ::llvm::SmallVector<::mlir::OpFoldResult> getMixedPackedSizes();
  }];
}

//===----------------------------------------------------------------------===//
// PackGreedilyOp
//===----------------------------------------------------------------------===//
def PackGreedilyOp : Op<Transform_Dialect, "structured.pack_greedily", [
                        DeclareOpInterfaceMethods<TransformOpInterface>,
                        DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let description = [{
    Target a Linalg op and rewrite it into packed LinalgOp form by trying to
    infer whether a known suboperation is embedded

    Different packing strategies are applied in order, when one applies 
    successfully, the transform returns:
      1. Matmul packing: Try to infer a matmul operation embedded in the target op.
         Specifically, this looks for 2 parallel dimensions that participate in
         an outer-product and 1 reduction dimension.
         These dimensions are referred as (m, n, k) to match canonical matmul
         terminology.
         
         The packed sizes for (m, n, k) are specified by `matmul_packed_sizes`
         and the optional `matmul_padded_sizes_next_multiple_of`.
         When an entry `matmul_packed_sizes[i]` is non-0, the corresponding 
         dimension is packed by `matmul_packed_sizes[i]`.
         Otherwise, the dimension is merely padded to the next multiple of
         `matmul_padded_sizes_next_multiple_of[i]`.

         `matmul_padded_sizes_next_multiple_of` is optional and is expected to
         either be empty or of size `3`, matching the size of `matmul_packed_sizes`.
         For each individual element of `matmul_packed_sizes` and 
         `matmul_padded_sizes_next_multiple_of`, only one of them is allowed to
         be non-zero.
         
         The ordering of the packed dimensions (mm, nn, kk) is specified by the
         `matmul_inner_dims_order` attribute.

    Packing occurs as follows:
      1. Find the dimensions to pack according to the strategy.
      2. The target is converted to linalg.generic form.
      3. An interchange transform is applied to isolate the dimensions to pack as
         the most minor indexing dimensions of the linalg.generic. The most minor
         dimensions are themselves ordered according to `inner_dims_order`.
      4. An elementwise traversal of `matmul_packed_sizes` and
         `matmul_padded_sizes_next_multiple_of` is performed and for each 
         dimension `d`, either pack to `matmul_packed_sizes[d]` or pad to the
         `matmul_padded_sizes_next_multiple_of[d]`.
      5. Packing/padding is performed by the amounts determined in step 4. and
         following `inner_dims_order`.

    By normalizing the most minor dimensions to `inner_dims_order`, the transform
    guarantees that packing immediately generates inner dimensions in a desirable
    layout.

    Outer dimension layout permutations are not controlled by this transform op
    at the moment and can be obtained by composing with the pack_transpose
    transformation.

    #### Return modes

    This operation ignores non-Linalg ops and drops them in the return.
    It returns the list of packed Linalg ops or the original op when all available
    packing strategies failed to apply.
  }];

  // TODO: Transform_ConcreteOpType<linalg::LinalgOp> needs interface.
  let arguments = (ins TransformHandleTypeInterface:$target,
                   Variadic<PDL_Operation>:$matmul_packed_sizes,
                   ConfinedAttr<DefaultValuedAttr<DenseI64ArrayAttr, "{}">,
                                 [DenseArrayCount<3>]>:$static_matmul_packed_sizes,
                   ConfinedAttr<DefaultValuedAttr<DenseI64ArrayAttr, "{}">,
                                 [Attr<
                                    Or<[DenseArrayCount<0>.predicate, 
                                        DenseArrayCount<3>.predicate]>,
                                        "with 0 or 3 elements"
                                      >]>
                                 :$matmul_padded_sizes_next_multiple_of,
                   ConfinedAttr<DefaultValuedAttr<DenseI64ArrayAttr, "{}">,
                                 [DenseArrayCount<3>]>:$matmul_inner_dims_order);
  let results = (outs Transform_ConcreteOpType<"linalg.generic">:$packed_op);

  let builders = [
    OpBuilder<(ins "Value":$target,
                   "ArrayRef<OpFoldResult>":$mixedMatmulPackedSizes,
                   "ArrayRef<int64_t>":$matmulPaddededSizesNextMultipleOf,
                   CArg<"ArrayRef<int64_t>", "{}">:$matmulDimsInnerDimsOrder)>
  ];

  let assemblyFormat = [{
    $target
    oilist(
      `matmul_packed_sizes` `=` custom<DynamicIndexList>($matmul_packed_sizes,
                                                         $static_matmul_packed_sizes)
      (`matmul_padded_sizes_next_multiple_of` `=` 
        $matmul_padded_sizes_next_multiple_of^)?
      `matmul_inner_dims_order` `=` $matmul_inner_dims_order
    )
    attr-dict
    `:` functional-type($target, results)
  }];
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    /// Returns the list of tile sizes, which may be static (Attribute) or
    /// dynamic (Value).
    SmallVector<OpFoldResult> getMixedMatmulPackedSizes();
  }];
}

//===----------------------------------------------------------------------===//
// PackTransposeOp
//===----------------------------------------------------------------------===//
def PackTransposeOp : Op<Transform_Dialect, "structured.pack_transpose", [
                         FunctionalStyleTransformOpTrait,
                         MemoryEffectsOpInterface,
                         DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let description = [{
    Apply a transposition to a single `tensor.pack` (resp. `tensor.unpack`) and 
    update the `linalg.generic` op that consumes (resp. produces) the operation.

    This transform allows composing a simple `structured.pack` with additional
    transpositions to e.g. match the data format required by a specific library
    call or ISA instruction.

    The transpose spec must specify at least one of `outer_perm` or `inner_perm`
    attributes, which will act upon the `outer_dims_perm` or `inner_dims_pos` of
    the specified `tensor.pack` or `tensor.unpack` op.

    If the `target` of this op is a `tensor.pack` then a new `tensor.empty` will
    be created along with transposed versions of the `tensor.pack` and the 
    consuming `linalg.generic`, which is expected to be the sole consumer.

    If the `target` of this op is a `tensor.unpack` then the whole pack / compute
    / unpack chain will be transposed and transposed clones of `tensor.pack`,
    the consuming `linalg.generic` and the tail `tensor.pack` will be created.

    #### Return modes

    This operation targets a single `tensor.pack` / `tensor.unpack` op and a
    single matching `linalg.generic` that consumes / produces the op. Otherwise,
    it produces a silenceableFailure.

    This operation may produce a silenceableFailure if the transpose spec is
    ill-formed (i.e. `outer_perm` or `inner_perm` are not permutations of the
    proper rank) or if the tranposition of all involved operations fails for any
    reason.

    This operation returns 3 handles, one to the transformed LinalgOp, one to
    the transformed `tensor.pack` and one to the transformed `tensor.unpack`.
    The last handle for `tensor.unpack` is empty if `target_pack_or_unpack_op` 
    was not itself a `tensor.unpack`.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target_pack_or_un_pack_op,
                       TransformHandleTypeInterface:$target_linalg_op,
                       DefaultValuedOptionalAttr<DenseI64ArrayAttr, "{}">:$outer_perm,
                       DefaultValuedOptionalAttr<DenseI64ArrayAttr, "{}">:$inner_perm);
  let results = (outs TransformHandleTypeInterface:$packed_op,
                      TransformHandleTypeInterface:$pack_op,
                      TransformHandleTypeInterface:$un_pack_op);
  let assemblyFormat = [{
    $target_pack_or_un_pack_op
    `with_compute_op` `(` $target_linalg_op `)`
    (`outer_perm` `=` $outer_perm^ )?
    (`inner_perm` `=` $inner_perm^ )?
    attr-dict
    `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// PadOp
//===----------------------------------------------------------------------===//

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
         DefaultValuedAttr<
          TypedArrayAttrBase<I64ArrayAttr, "array of arrays of i64">,
          "{}">:$transpose_paddings);
  let results = (outs PDL_Operation:$transformed);

  let assemblyFormat = "$target attr-dict";
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::linalg::LinalgOp target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

//===----------------------------------------------------------------------===//
// HoistPadOp
//===----------------------------------------------------------------------===//

def HoistPadBuildPackingLoopNestOp :
    Op<Transform_Dialect,
       "structured.hoist_pad.build_packing_loop_nest",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let description = [{
    Helper transform used to hoist a tensor.pad target operation. This operation
    creates the packing loop nest required by the hoist_pad operation and makes
    that functionality available independently.

    TODO: In the future, we should consider rewriting as a tensor.pack after
    hoisting since this abstraction is now available.

    #### Return modes

    This operation ignores non-tensor.pad ops and drops them in the result.
    If any non-tensor.pad is passed, the transform emits a silenceable failure.

    The return handle points to only the subset of successfully created packing
    loop nests, which can be empty.
  }];

  // Also allow any !pdl.operation for simpler composition. Non-tensor.pad ops
  // will be dropped from the results.
  let arguments =
    (ins TransformHandleTypeInterface:$target,
         TransformHandleTypeInterface:$loop,
         DefaultValuedAttr<DenseI64ArrayAttr, "{}">:$transpose);
  let results = (outs TransformHandleTypeInterface:$packing_loop);

  let assemblyFormat = [{
    $target
    `above` $loop
    (`,` `transpose` `by` $transpose^)?
    attr-dict
    `:` functional-type(operands, results)
  }];
  let hasVerifier = 1;
}

def HoistPadOp : Op<Transform_Dialect, "structured.hoist_pad",
    [FunctionalStyleTransformOpTrait,
     MemoryEffectsOpInterface,
     TransformOpInterface,
     TransformEachOpTrait]> {
  let description = [{
    Hoist the tensor.pad target operation by at most the given number of loops.
    Optionally apply the transpose attribute to the inner dimensions.

    TODO: In the future, we should consider rewriting as a tensor.pack after 
    hoisting since this abstraction is now available.
    TODO: Maybe also return the linalg.generic transpose created at some point.

    #### Return modes

    This operation ignores non-tensor.pad ops and drops them in the result.
    If any non-tensor.pad is passed, the transform emits a silenceable failure.

    If all the operations referred to by the `target` handle padproperly, the
    transform succeeds. Otherwise the transform silently fails.

    The return handle points to only the subset of successfully hoisted 
    tensor.pad operations, which can be empty.
  }];

  // Also allow any !pdl.operation for simpler composition. Non-tensor.pad ops
  // will be dropped from the results.
  let arguments =
    (ins TransformHandleTypeInterface:$target,
         I64Attr:$num_loops,
         DefaultValuedAttr<DenseI64ArrayAttr, "{}">:$transpose);
  let results = (outs TransformHandleTypeInterface:$transformed);

  let assemblyFormat = [{
    $target 
    `by` $num_loops `loops` 
    (`,` `transpose` `by` $transpose^)? 
    attr-dict
    `:` functional-type(operands, results)
  }];
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::tensor::PadOp,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

//===----------------------------------------------------------------------===//
// PromoteOp
//===----------------------------------------------------------------------===//


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
                       OptionalAttr<DeviceMappingArrayAttr>:$mapping,
                       OptionalAttr<I64Attr>:$alignment);
  let results = (outs PDL_Operation:$transformed);

  let assemblyFormat = "$target attr-dict";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::linalg::LinalgOp target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

//===----------------------------------------------------------------------===//
// ReplaceOp
//===----------------------------------------------------------------------===//

def ReplaceOp : Op<Transform_Dialect, "structured.replace",
    [IsolatedFromAbove, DeclareOpInterfaceMethods<TransformOpInterface>,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>] # GraphRegionNoTerminator.traits> {
  let description = [{
    Replace all `target` payload ops with the single op that is contained in
    this op's region. All targets must have zero arguments and must be isolated
    from above.

    This op is for debugging/experiments only.

    #### Return modes

    This operation consumes the `target` handle.
  }];

  let arguments = (ins PDL_Operation:$target);
  let results = (outs PDL_Operation:$replacement);
  let regions = (region SizedRegion<1>:$bodyRegion);
  let assemblyFormat = "$target attr-dict-with-keyword regions";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ScalarizeOp
//===----------------------------------------------------------------------===//

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
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

//===----------------------------------------------------------------------===//
// RewriteInDestinationPassingStyleOp.
//===----------------------------------------------------------------------===//

def RewriteInDestinationPassingStyleOp : Op<
    Transform_Dialect, "structured.rewrite_in_destination_passing_style",
    [FunctionalStyleTransformOpTrait, 
     MemoryEffectsOpInterface,
     TransformOpInterface, 
     TransformEachOpTrait]> {
  let description = [{
    Rewrite a supported tensor operation that is not in destination-passing style
    into a form that is in destination-passing style.
    Currently supported operations are:
      - tensor.pad
      - tensor.generate
      - tensor.from_elements
    This dichotomy hints at a future interface, for now the implementation just 
    switches between different implementation.

    #### Return modes

    This operation ignores non-unsupported ops and drops them from the return.
    If all the operations referred to by the `target` PDLOperation generalize
    properly, the transform succeeds. Otherwise the transform silently fails.
    The return handle points to a subset of successfully produced operations:
      - tensor.pad case, the returned handle points to the tensor.insert_slice.
      - tensor.generate case, the returned handle points to the linalg.generic.
      - tensor.from_elements case, the returned handle points to the last 
        tensor.insert.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target);
  let results = (outs TransformHandleTypeInterface:$transformed);
  let assemblyFormat = [{
    $target attr-dict
    `:` functional-type($target, results)
  }];

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::Operation *target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

//===----------------------------------------------------------------------===//
// SplitOp
//===----------------------------------------------------------------------===//

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
    point must be set to `ShapedType::kDynamic` and the dynamic size handle
    must point to as many value-producing operations as there are structured
    operations pointed to by the target handle.

    The operation consumes the target handle, but preserves the split point
    handle if provided. It produces two new handles pointing to the two parts
    of the structured op after splitting, in the same order as the target
    operand, with the first handle corresponding to the part with lower
    iteration space indices.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target,
                       I64Attr:$dimension,
                       Optional<TransformParamTypeOrAnyHandle>:$dynamic_split_point,
                       I64Attr:$static_split_point);
  let results = (outs TransformHandleTypeInterface:$first,
                      TransformHandleTypeInterface:$second);
  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// SplitReductionOp
//===----------------------------------------------------------------------===//

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
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

//===----------------------------------------------------------------------===//
// TileReductionUsingScfOp
//===----------------------------------------------------------------------===//

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

    This 4 returned handles point to:
      - the parent for op,
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

  // TODO: support mixed static-dynamic (see TileToForallOp).
  let arguments = (ins PDL_Operation:$target,
                   DefaultValuedAttr<DenseI64ArrayAttr, "{}">:$tile_sizes);
  let results = (outs PDL_Operation:$for_op,
                      PDL_Operation:$fill_op,
                      PDL_Operation:$split_linalg_op,
                      PDL_Operation:$combining_linalg_op);

  let builders = [
    OpBuilder<(ins "Value":$target,
                   "ArrayRef<int64_t>":$staticTileSizes)>
  ];

  let assemblyFormat = [{
    $target
    `by` `tile_sizes` `=` $tile_sizes
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::linalg::LinalgOp target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

//===----------------------------------------------------------------------===//
// TileReductionUsingForallOp
//===----------------------------------------------------------------------===//

def TileReductionUsingForallOp :
  Op<Transform_Dialect, "structured.tile_reduction_using_forall",
       [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
        TransformEachOpTrait, TransformOpInterface]> {
  let description = [{
    Tile a PartialReductionOpInterface op to a tiled `scf.forall` doing
    partial reduction.

    This transformation tiles the `target` along the reduction dimensions. It
    creates a tensor initialized with the identity value. Then it creates a
    `scf.forall` loops with the number threads given by `num_threads`.
    The op is tiled op with a size equal to `floordiv(size, num_threads)`.
    All the partial reduction value is are parallel inserted to create a new
    tensor. After the loop a merge operation is created to do a final reduction
    with the partial reductions tensor.
    If an extra `tile_sizes` parameter is passed the tiles are cyclically
    distributed on the threads of the `scf.foralls` loop.

    #### Return modes

    This 4 returned handles point to:
      - the parent forall op,
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
      %2 = scf.forall (%arg2) in (%c5) shared_outs(%arg3 = %1) -> (tensor<?x5xf32>) {
        %4 = affine.min #map(%arg2)[%dim_0]
        %5 = affine.max #map1(%4)
        %extracted_slice = tensor.extract_slice %arg3[0, %arg2] [%dim, 1] [1, 1] : tensor<?x5xf32> to tensor<?xf32>
        %6 = affine.apply #map2(%arg2)[%dim_0]
        %extracted_slice_2 = tensor.extract_slice %arg0[0, %6] [%dim, %5] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
        %extracted_slice_3 = tensor.extract_slice %extracted_slice[0] [%dim] [1] : tensor<?xf32> to tensor<?xf32>
        %7 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_2 : tensor<?x?xf32>) outs(%extracted_slice_3 : tensor<?xf32>) {
        ^bb0(%in: f32, %out: f32):
          %9 = arith.addf %in, %out : f32
          linalg.yield %9 : f32
        } -> tensor<?xf32>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %7 into %arg3[0, %arg2] [%dim, 1] [1, 1] : tensor<?xf32> into tensor<?x5xf32>
        }
      } {mapping = []}
      %3 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%2 : tensor<?x5xf32>) outs(%arg1 : tensor<?xf32>) {
      ^bb0(%in: f32, %out: f32):
        %4 = arith.addf %in, %out : f32
        linalg.yield %4 : f32
      } -> tensor<?xf32>
    ```
  }];

  // TODO: support mixed static-dynamic (see TileToForallOp).
  let arguments = (ins PDL_Operation:$target,
                   DefaultValuedAttr<DenseI64ArrayAttr, "{}">:$num_threads,
                   DefaultValuedAttr<DenseI64ArrayAttr, "{}">:$tile_sizes,
                   OptionalAttr<DeviceMappingArrayAttr>:$mapping);
  let results = (outs PDL_Operation:$forall_op,
                      PDL_Operation:$fill_op,
                      PDL_Operation:$split_linalg_op,
                      PDL_Operation:$combining_linalg_op);

  let builders = [
    OpBuilder<(ins "Value":$target,
                   "ArrayRef<int64_t>":$staticNumThreads,
                   "ArrayRef<int64_t>":$staticTileSizes,
                   CArg<"ArrayAttr", "{}">:$mapping)>
  ];

  let assemblyFormat = [{
    $target
    `by`
    (`num_threads` `=` $num_threads^)?
    (`,` `tile_sizes` `=` $tile_sizes^)?
    (`,` `mapping` `=` $mapping^)?
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::linalg::LinalgOp target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];

}

//===----------------------------------------------------------------------===//
// TileOp
//===----------------------------------------------------------------------===//

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
    `static_sizes` attribute must be set to `ShapedType::kDynamic`. Only
    the dynamic sizes must be provided in `dynamic_sizes`, i.e., there should
    be as many handles as `ShapedType::kDynamic` values in the
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

  let arguments = (ins TransformHandleTypeInterface:$target,
                   Variadic<TransformParamTypeOrAnyHandle>:$dynamic_sizes,
                   DefaultValuedOptionalAttr<DenseI64ArrayAttr, "{}">:$static_sizes,
                   DefaultValuedOptionalAttr<DenseI64ArrayAttr, "{}">:$interchange);
  let results = (outs TransformHandleTypeInterface:$tiled_linalg_op,
                      Variadic<TransformHandleTypeInterface>:$loops);
  let builders = [
    OpBuilder<(ins "TypeRange":$loopTypes,
                   "Value":$target,
                   "ArrayRef<int64_t>":$staticTileSizes,
                   CArg<"ArrayRef<int64_t>", "{}">:$interchange)>,
    OpBuilder<(ins "TypeRange":$loopTypes,
                   "Value":$target,
                   "ArrayRef<OpFoldResult>":$mixedTileSizes,
                   CArg<"ArrayRef<int64_t>", "{}">:$interchange)>,
    OpBuilder<(ins "Value":$target,
                   "ArrayRef<int64_t>":$staticTileSizes,
                   CArg<"ArrayRef<int64_t>", "{}">:$interchange)>,
    OpBuilder<(ins "Value":$target,
                   "ArrayRef<OpFoldResult>":$mixedTileSizes,
                   CArg<"ArrayRef<int64_t>", "{}">:$interchange)>

  ];

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    /// Returns the list of tile sizes, which may be static (Attribute) or
    /// dynamic (Value).
    SmallVector<OpFoldResult> getMixedSizes();
  }];
}

//===----------------------------------------------------------------------===//
// TileToForallOp
//===----------------------------------------------------------------------===//

def TileToForallOp :
    Op<Transform_Dialect, "structured.tile_to_forall_op",
      [AttrSizedOperandSegments,
       DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
       TransformOpInterface]> {
  let description = [{
    Tile a TilingInterface op to a tiled `scf.forall`.

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
    resulting `scf.forall`.

    Note: `tile_sizes` and `num_threads` are variadic. Each tile size/number of
    threads can be an index attribute or a transform handle that is mapped to
    exactly one payload op with exactly one index result.

    #### Return modes

    This operation ignores ops that do not implement the TilingInterface and
    drops them in the return.

    If all the operations referred to by the `target` PDLOperation tile
    successfully, the transform succeeds.
    Otherwise the transform silently fails.

    The two returned handles point to only the subset of successfully produced
    tiled operations, which can all be empty.

    These two returned handles point to:
      - the new scf.forall op,
      - the tiled op that implements TilingInterface.

    #### Example using `num_threads`

    ```
    %0 = pdl_match @match_matmul in %arg1
    %3:2 = transform.structured.tile_to_forall_op %0 num_threads [10, 20]
    ```

    #### Example using `tile_sizes`

    ```
    %0 = pdl_match @match_matmul in %arg1
    %sz = pdl_match @match_size_op in %arg1
    %3:2 = transform.structured.tile_to_forall_op %0 tile_sizes [0, %sz, 20]
    ```
  }];

  let arguments = (ins PDL_Operation:$target,
                   Variadic<PDL_Operation>:$num_threads,
                   Variadic<PDL_Operation>:$tile_sizes,
                   Optional<PDL_Operation>:$packed_num_threads,
                   Optional<PDL_Operation>:$packed_tile_sizes,
                   DefaultValuedOptionalAttr<DenseI64ArrayAttr, "{}">:$static_num_threads,
                   DefaultValuedOptionalAttr<DenseI64ArrayAttr, "{}">:$static_tile_sizes,
                   OptionalAttr<DeviceMappingArrayAttr>:$mapping);
  let results = (outs PDL_Operation:$forall_op,
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
                   CArg<"ArrayAttr", "{}">:$mapping)>
  ];

  let assemblyFormat = [{
    $target oilist(
        `num_threads` custom<PackedOrDynamicIndexList>($packed_num_threads,
                                                       $num_threads,
                                                       $static_num_threads) |
         `tile_sizes` custom<PackedOrDynamicIndexList>($packed_tile_sizes,
                                                       $tile_sizes,
                                                       $static_tile_sizes))
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

//===----------------------------------------------------------------------===//
// TileToScfForOp
//===----------------------------------------------------------------------===//

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
    `static_sizes` attribute must be set to `ShapedType::kDynamic`. Only
    the dynamic sizes must be provided in `dynamic_sizes`, i.e., there should
    be as many handles as `ShapedType::kDynamic` values in the
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
                   DefaultValuedOptionalAttr<DenseI64ArrayAttr, "{}">:$static_sizes,
                   DefaultValuedOptionalAttr<DenseI64ArrayAttr, "{}">:$interchange);
  let results = (outs PDL_Operation:$tiled_linalg_op,
                      Variadic<PDL_Operation>:$loops);

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    /// Returns the list of tile sizes, which may be static (Attribute) or
    /// dynamic (Value).
    SmallVector<OpFoldResult> getMixedSizes();
  }];
}

//===----------------------------------------------------------------------===//
// VectorizeOp
//===----------------------------------------------------------------------===//

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
                   UnitAttr:$vectorize_nd_extract,
                   UnitAttr:$disable_multi_reduction_to_contract_patterns,
                   UnitAttr:$disable_transfer_permutation_map_lowering_patterns);
  let results = (outs PDL_Operation:$transformed);

  let assemblyFormat = "$target attr-dict";

  let builders = [
    OpBuilder<(ins "Value":$target,
               CArg<"bool", "false">:$vectorizePadding,
               CArg<"bool", "false">:$vectorizeNDExtract)>,
  ];
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::Operation *target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

def MaskedVectorizeOp : Op<Transform_Dialect, "structured.masked_vectorize",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     TransformOpInterface]> {
  let description = [{
    Vectorize the target ops, which must be Linalg ops, with masked vectors
    of the specified size.

    The vector sizes can be either static or dynamic (SSA values). In case of
    SSA values, the handle must be mapped to exactly one payload op with
    exactly one index-typed result.

    Note: The input vector sizes must be bigger than or equal to their
    counterpart iteration space sizes.

    Typically this operator should be applied to linalg operations that have
    already be tiled to the appropriate sizes.

    #### Return modes:

    This operation produces a definite failure if the dynamic vector sizes (SSA
    values) do not satify the constraints mentioned above. It produces a
    silenceable failure if at least one target op is not a Linalg op or fails to
    vectorize.
  }];

  let arguments = (ins PDL_Operation:$target,
                       Variadic<PDL_Operation>:$vector_sizes,
                       UnitAttr:$vectorize_nd_extract,
                       DefaultValuedOptionalAttr<DenseI64ArrayAttr, "{}">:
                          $static_vector_sizes);
  let results = (outs);
  let assemblyFormat = [{
      $target
      `vector_sizes` custom<DynamicIndexList>($vector_sizes,
                                              $static_vector_sizes)
      attr-dict
  }];

  let extraClassDeclaration = [{
    // TODO: applyToOne.
    ::mlir::DiagnosedSilenceableFailure apply(
        ::mlir::transform::TransformResults &transformResults,
        ::mlir::transform::TransformState &state);

    ::llvm::SmallVector<::mlir::OpFoldResult> getMixedVectorSizes();
  }];
}

//===----------------------------------------------------------------------===//
// HoistRedundantVectorTransfersOp
//===----------------------------------------------------------------------===//

def HoistRedundantVectorTransfersOp :
  Op<Transform_Dialect, "structured.hoist_redundant_vector_transfers",
    [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
     TransformEachOpTrait, TransformOpInterface]> {
  let description = [{
    Hoist vector.transfer_read / vector.transfer_write pairs out of immediately
    enclosing scf::ForOp iteratively, if the following conditions are true:
       1. The 2 ops access the same memref with the same indices.
       2. All operands are invariant under the enclosing scf::ForOp.
       3. No uses of the memref either dominate the transfer_read or are
       dominated by the transfer_write (i.e. no aliasing between the write and
       the read across the loop)

    WARNING: This hoisting does not model parallelism and is generally incorrect
    when used on distributed loops with memref semantics!
    TODO: obsolete and should be retired.

    #### Return modes:

    The operation always succeeds and returns a handle to the transformed
    function op.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target);
  let results = (outs TransformHandleTypeInterface:$transformed);

  let assemblyFormat = "$target attr-dict `:` functional-type(operands, results) ";

  let builders = [
    OpBuilder<(ins "Value":$target)>,
  ];
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::func::FuncOp target,
         ::mlir::transform::ApplyToEachResultList &results,
         ::mlir::transform::TransformState &state);
   }];
}

//===----------------------------------------------------------------------===//
// ConvertConv2DToImg2ColOp
//===----------------------------------------------------------------------===//

def ConvertConv2DToImg2ColOp : Op<Transform_Dialect,
    "structured.convert_conv2d_to_img2col",
    [FunctionalStyleTransformOpTrait,
     MemoryEffectsOpInterface,
     TransformOpInterface,
     TransformEachOpTrait]> {
  let description = [{
    Convert linalg.conv_2d_xxx into linalg.generic (for img2col packing)
    and linalg.matmul.

    A convolution operation can be written as a matrix-matrix multiplication by
    unfolding the cross-correlation between input and filter and explicitly copy
    overlapped sliding window inputs.

    Consider 2D input X with single channel input and output and 2x2 filter W:
    ```
    [x(0, 0)  , x(0, 1)  , ...,   x(0, n)  ]
    [x(1, 0)  , x(1, 1)  , ...,   x(1, n)  ]
    [.        ,  .       ,.   ,      .     ]            [w(0, 0), w(0, 1)]
    [.        ,  .       , .  ,      .     ]    (conv)  [w(1, 0), w(1, 1)]
    [.        ,  .       ,   .,      .     ]
    [x(n-1, 0), x(n-1, 1), ..., x(n-1, n-1)]
    ```

    The packed input data (img2col) is a matrix with |rows| = output spatial
    size, |columns| = filter spatial size. To compute the output Y(i, j) we need
    to calculate the dot product between filter window at input X(x, y)) and the
    filter which will look like the following where r.h.s is the img2col matrix
    and l.h.s is the flattned filter:
    ```
    [x(0,0), x(0,1), x(1,0), x(1,1)]
    [x(0,1), x(1,1), x(0,2), x(1,2)] (matmul) [w(0,0), w(0,1), w(1,0), w(1,1)]
    [x(0,1), x(1,1), x(0,2), x(1,2)]
    [   .  ,    .  ,    .  ,    .  ]
    ```

    In general for 2D case with (N, H, W, C) input and (Kh, Kw, C, D) filter
    and output (N, Ho, Wo, D) the convolution is the following matrix-matrix
    multiplication (Ho x Wo, Kh x Kw x C) * (Kh x Kw x C, D) for each input in
    the N input. For the case where N > 1 its a batched matrxi-matrix
    multplication.

    Returns two handles:
    - One on the operation that produces the img2col tensor.
    - One on the final operation of the sequence that replaces the original
      convolution.

    #### Return modes:

    Returns a definite failure if target is not isolated from above.
    Returns a silenceable failure if the pattern application failed.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target);
  let results = (outs TransformHandleTypeInterface:$img2col_tensor,
                      TransformHandleTypeInterface:$transformed);

  let assemblyFormat =
    "$target attr-dict `:` functional-type($target, results)";

  let builders = [
    OpBuilder<(ins "Value":$target)>
  ];

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::linalg::LinalgOp target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

//===----------------------------------------------------------------------===//
// HoistRedundantTensorSubsetsOp
//===----------------------------------------------------------------------===//

def HoistRedundantTensorSubsetsOp :
  Op<Transform_Dialect, "structured.hoist_redundant_tensor_subsets",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     TransformEachOpTrait,
     TransformOpInterface]> {
  let description = [{
    Hoists supported tensor subset extract/insert operation pairs out of 
    immediately enclosing loop iteratively, if the following conditions
    are true:
       1. The 2 ops access the same tensor subset.
       2. All operands are invariant under the enclosing loop.
    
    The supported subset extract/insert operation pairs currently comprise:
       - tensor.extract_slice / tensor.insert_slice
       - vector.transfer_read / vector.transfer_write on tensors
    
    Only scf.for loops are currently supported.

    When applied to:
       1. an scf.for loop, hoist out of this loop only.
       2. a non-loop op, apply hoisting to all the contained loop ops.

    #### Return modes:

    The operation always succeeds and returns nothing.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target);
  let results = (outs);

  let assemblyFormat = [{
    $target 
    attr-dict 
    `:` functional-type(operands, results)
  }];

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::Operation *target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

//===----------------------------------------------------------------------===//
// InsertSliceToCopyOp
//===----------------------------------------------------------------------===//

def InsertSliceToCopyOp :
  Op<Transform_Dialect, "structured.insert_slice_to_copy",
    [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
     TransformEachOpTrait, TransformOpInterface]> {
  let description = [{
    Targeted rewrite of an tensor.insert_slice to linalg.copy.
    This is useful to materialize copies explicitly before bufferization and 
    transform them, avoiding the need to rediscover them after bufferization.

    If the insert_slice source is already a linalg.copy, only return the source
    op (i.e. do not create an additional linalg.copy op).

    #### Return modes:

    The operation always succeeds and returns a handle to the relevant 
    linalg.copy op.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target);
  let results = (outs TransformHandleTypeInterface:$transformed);

  let assemblyFormat = "$target attr-dict `:` functional-type(operands, results) ";

  let builders = [
    OpBuilder<(ins "Value":$target)>,
  ];
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::Operation *target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

#endif // LINALG_TRANSFORM_OPS
