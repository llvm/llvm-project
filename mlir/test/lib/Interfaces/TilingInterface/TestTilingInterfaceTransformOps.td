//===- TestTilingInterfaceTransformOps.td -----------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_TILINGINTERFACE_TRANSFORM_OPS
#define TEST_TILINGINTERFACE_TRANSFORM_OPS

include "mlir/Dialect/SCF/IR/DeviceMappingInterface.td"
include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.td"
include "mlir/Dialect/Transform/IR/TransformTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

// Those operations in this file are meant for testing the tiling interface
// transformations using scf operations. Over time these testing options
// might be useful transformations in their own right. Move these over
// as transform ops in the main repo (also find a proper place for them)

def TestFuseAndYieldOp : Op<Transform_Dialect, "test.fuse_and_yield",
  [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
   DeclareOpInterfaceMethods<TransformOpInterface>,
   ReportTrackingListenerFailuresOpTrait]> {
  let description = [{
    Tiles the operations pointed to by the target handle, fuses their
    producers greedily using the options provided as attributes.
    It also yields some of the fused producers for testing.

    On success returns the tiled operations as well as generated loops. Emits
    a definite failure if tiling fails.
  }];

  let arguments =
    (ins TransformHandleTypeInterface:$target,
        DefaultValuedAttr<I64ArrayAttr, "{}">:$tile_sizes,
        DefaultValuedAttr<I64ArrayAttr, "{}">:$tile_interchange,
        DefaultValuedAttr<BoolAttr, "false">:$use_forall);
  let results = (outs TransformHandleTypeInterface:$transfomed,
      Variadic<TransformHandleTypeInterface>:$loops);

  let assemblyFormat = [{
    $target ($tile_sizes^)? (`interchange` $tile_interchange^)?
    (`use_forall` $use_forall^)? attr-dict 
    `:` functional-type(operands, results)
  }];
}

def TestFuseConsumerUsingSliceOp : Op<Transform_Dialect, "test.fuse_consumer_using_slice",
       [AttrSizedOperandSegments,
        DeclareOpInterfaceMethods<TransformOpInterface>,
        DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
        ReportTrackingListenerFailuresOpTrait]> {
  let description = [{
    For the `insert_slice`-like operations (that are typically generated through tiling),
    within the loop nests passed in as `loops` (that are typically generated through tiling),
    find the consumer that these slices map to (have to be the same consumer) and fuse
    the consumer into the loop.

    Returns a handle to the original consumer operation and the consumer operation after
    fusion.
  }];

  let arguments = (ins 
      Variadic<TransformHandleTypeInterface>:$targets,
      Variadic<TransformHandleTypeInterface>:$loops,
      DefaultValuedAttr<I32Attr, "1">:$num_consumer_to_fuse);
  let results = (outs TransformHandleTypeInterface:$consumer,
                      TransformHandleTypeInterface:$fused_consumer);

  let assemblyFormat = [{
    $targets `in` `(` $loops `)`
    (`num_consumer_to_fuse` `=` $num_consumer_to_fuse^)? 
    attr-dict `:` functional-type(operands, results)
  }];
}

def TestFuseConsumerOp : Op<Transform_Dialect, "test.fuse_consumer",
       [DeclareOpInterfaceMethods<TransformOpInterface>,
        DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
        ReportTrackingListenerFailuresOpTrait]> {
  let description = [{
    For the `consumer` that uses the result of the outer-most loop of a loop nest passed in 
    as `loops` (that are typically generated through tiling), fuse the consumer into the
    loop.

    Returns a handle to the consumer operation after fusion and the loops that might be
    modified.
  }];

  let arguments = (ins 
      TransformHandleTypeInterface:$consumer,
      Variadic<TransformHandleTypeInterface>:$loops);
  let results = (outs TransformHandleTypeInterface:$fused_consumer,
      Variadic<TransformHandleTypeInterface>:$result_loops);

  let assemblyFormat = [{
    $consumer `into` `(` $loops `)`
    attr-dict `:` functional-type(operands, results)
  }];
}


def TestTileUsingForallOp : Op<Transform_Dialect, "test.tile_using_forall",
       [DeclareOpInterfaceMethods<TransformOpInterface>,
        DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
        ReportTrackingListenerFailuresOpTrait]> {
  let description = [{
    Test operation use to test tiling using TilingInterface and scf.forall for
    the loop constructs. This is similar to
    `transform.structured.tile_using_for`. Use of this operation is an
    intermediate state and will be replaced in due course with either
    `transform.structured.tile_using_for` or
    `transform.structured.tile_using_forall`.

    On success returns the tiled operations as well as generated loops. Emits
    a definite failure if tiling fails.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target,
                   DefaultValuedAttr<I64ArrayAttr, "{}">:$tile_sizes,
                   DefaultValuedOptionalAttr<I64ArrayAttr, "{}">:$interchange,
                   OptionalAttr<DeviceMappingArrayAttr>:$mapping);
  let results = (outs TransformHandleTypeInterface:$tiled_op,
                      Variadic<TransformHandleTypeInterface>:$loops);

  let assemblyFormat = [{
    $target ($tile_sizes^)? (`interchange` `=` $interchange^)?
    (`mapping` `=` $mapping^)?
    attr-dict `:` functional-type(operands, results)
  }];
}

def TestFuseUsingForallOp : Op<Transform_Dialect, "test.fuse_using_forall",
    [DeclareOpInterfaceMethods<TransformOpInterface>,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     ReportTrackingListenerFailuresOpTrait]> {
  let description = [{
    Test operation to tile the operation pointed to by the target handle and
    fuses their producers greedily using the options provided as attributes.
    This operation uses scf.forall for the loop construct.
  }];
  let arguments = (ins TransformHandleTypeInterface:$root_op,
                   DefaultValuedAttr<I64ArrayAttr, "{}">:$tile_sizes,
                   DefaultValuedOptionalAttr<I64ArrayAttr, "{}">:$interchange,
                   OptionalAttr<DeviceMappingArrayAttr>:$mapping);
  let results = (outs TransformHandleTypeInterface:$tiled_ops,
                      Variadic<TransformHandleTypeInterface>:$loops);

  let assemblyFormat = [{
    $root_op ($tile_sizes^)? (`interchange` $interchange^)?
    (`mapping` `=` $mapping^)?
    attr-dict `:` functional-type(operands, results)
  }];
}

def TestTileAndFuseOuterParallelPartialReductionOp : Op<
    Transform_Dialect, "test.tile_and_fuse_outer_parallel_partial_reduction",
    [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
     DeclareOpInterfaceMethods<TransformOpInterface>,
     ReportTrackingListenerFailuresOpTrait]> {
  let description = [{
    Test operation to tile an operation using partial reduction with
    outer parallel strategy, and to fuse its producers.
  }];

  let arguments = (ins TransformHandleTypeInterface:$root_op,
                   DefaultValuedAttr<I64ArrayAttr, "{}">:$reduction_dims,
                   DefaultValuedAttr<I64ArrayAttr, "{}">:$tile_sizes,
                   OptionalAttr<DeviceMappingArrayAttr>:$mapping);

  let results = (outs TransformHandleTypeInterface:$tiled_ops,
                 Variadic<TransformHandleTypeInterface>:$loops);
  let assemblyFormat = [{
    $root_op (`reduction_dims` `=` $reduction_dims^)?
    (`tile_sizes` `=` $tile_sizes^)? (`mapping` `=` $mapping^)?
    attr-dict `:` functional-type(operands, results)
  }];
}

def TestTileUsingCustomLoopOp : Op<
    Transform_Dialect, "test.tile_using_custom_loop",
    [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
     DeclareOpInterfaceMethods<TransformOpInterface>,
     ReportTrackingListenerFailuresOpTrait]> {
  let description = [{
    Test Transform op to tile an operation using custom loops.

    The test just folds all the loops and into a single loop and then
    delinearizes the indices.
  }];

  let arguments = (ins TransformHandleTypeInterface:$root_op,
                   DefaultValuedAttr<I64ArrayAttr, "{}">:$tile_sizes);
  let results  = (outs TransformHandleTypeInterface:$tiled_ops,
                  Variadic<TransformHandleTypeInterface>:$loops);

  let assemblyFormat = [{
    $root_op `tile_sizes` `=` $tile_sizes
    attr-dict `:` functional-type(operands, results)
  }];
}

def TestQueryProducerFusability : Op<
    Transform_Dialect, "test.query_producer_fusability",
    [DeclareOpInterfaceMethods<TransformOpInterface>,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let description = [{
    Test operation for the producer fusability query method in the
    TilingInterface.

    For each operation in the target handle, this looks for tensor.insert_slice
    ops that produce operands to the tilable op. The offset/sizes from those
    inserts is used as the arguments to `isOpFusableWithProducerSlices` and
    emits a remark with the result of the query.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target);
  let results = (outs);

  let assemblyFormat = [{
    $target attr-dict `:` type($target)
  }];
}

def TestQueryConsumerFusability
    : Op<Transform_Dialect, "test.query_consumer_fusability",
         [DeclareOpInterfaceMethods<TransformOpInterface>,
          DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let description = [{
    Test operation for the consumer fusability query method in the
    TilingInterface.

    For each operation in the target handle, this looks for tensor.extract_slice
    ops that consume results of the tilable op. The offset/sizes from those
    extracts is used as the arguments to `isOpFusableWithConsumerSlice` and
    emits a remark with the result of the query.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target);
  let results = (outs);

  let assemblyFormat = [{
    $target attr-dict `:` type($target)
  }];
}

#endif // TEST_TILINGINTERFACE_TRANSFORM_OPS
