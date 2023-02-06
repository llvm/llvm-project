//===- BufferizationTransformOps.td - Buff. transf. ops ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BUFFERIZATION_TRANSFORM_OPS
#define BUFFERIZATION_TRANSFORM_OPS

include "mlir/Dialect/Bufferization/IR/BufferizationEnums.td"
include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/IR/TransformInterfaces.td"
include "mlir/Dialect/Transform/IR/TransformTypes.td"
include "mlir/Dialect/PDL/IR/PDLTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

def OneShotBufferizeOp
    : Op<Transform_Dialect, "bufferization.one_shot_bufferize",
        [DeclareOpInterfaceMethods<TransformOpInterface>,
         DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let description = [{
    Indicates that the given `target` op should be bufferized with One-Shot
    Bufferize. The bufferization can be configured with various attributes that
    corresponding to options in `BufferizationOptions` and the
    `one-shot-bufferize` pass. More information can be found in the pass
    documentation.

    If `target_is_module` is set, `target` must be a module. In that case the
    `target` handle can be reused by other transform ops. When bufferizing other
    ops, the `target` handled is freed after bufferization and can no longer be
    used.

    Note: Only ops that implement `BufferizableOpInterface` are bufferized. All
    other ops are ignored if `allow_unknown_ops`. If `allow_unknown_ops` is
    unset, this transform fails when an unknown/non-bufferizable op is found.
    Many ops implement `BufferizableOpInterface` via an external model. These
    external models must be registered when applying this transform op;
    otherwise, said ops would be considered non-bufferizable.
  }];

  let arguments = (
      ins PDL_Operation:$target,
      OptionalAttr<LayoutMapOption>:$function_boundary_type_conversion,
      DefaultValuedAttr<BoolAttr, "false">:$allow_return_allocs,
      DefaultValuedAttr<BoolAttr, "false">:$allow_unknown_ops,
      DefaultValuedAttr<BoolAttr, "false">:$bufferize_function_boundaries,
      DefaultValuedAttr<BoolAttr, "true">:$create_deallocs,
      DefaultValuedAttr<BoolAttr, "true">:$target_is_module,
      DefaultValuedAttr<BoolAttr, "false">:$test_analysis_only,
      DefaultValuedAttr<BoolAttr, "false">:$print_conflicts);

  let results = (outs);

  let assemblyFormat = [{
    (`layout` `{` $function_boundary_type_conversion^ `}`)?
    $target attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// EmptyTensorToAllocTensorOp
//===----------------------------------------------------------------------===//


def Transform_EmptyOp : Transform_ConcreteOpType<"tensor.empty">;
def Transform_AllocTensorOp : Transform_ConcreteOpType<"bufferization.alloc_tensor">;

def EmptyTensorToAllocTensorOp
    : Op<Transform_Dialect, "bufferization.empty_tensor_to_alloc_tensor",
        [FunctionalStyleTransformOpTrait,
         MemoryEffectsOpInterface,
         TransformOpInterface,
         TransformEachOpTrait]> {
  let description = [{
    Replace a tensor.empty with a bufferization.tensor_alloc.

    #### Return modes

    This operation consumes the `target` handle and produces the `transformed`
    handle. `target` is expected to be a `tensor.empty` operation. The transform
    always succeeds.
  }];

  let arguments = (ins Transform_EmptyOp:$target);
  let results = (outs Transform_AllocTensorOp:$transformed);

  let assemblyFormat = "$target attr-dict `:` functional-type(operands, results)";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::tensor::EmptyOp target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

#endif // BUFFERIZATION_TRANSFORM_OPS
