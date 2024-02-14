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
include "mlir/Dialect/Transform/IR/TransformInterfaces.td"
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

#endif // TEST_TILINGINTERFACE_TRANSFORM_OPS
