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
include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.td"
include "mlir/Dialect/Transform/IR/TransformTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

def Transform_EmptyOp : Transform_ConcreteOpType<"tensor.empty">;
def Transform_AllocTensorOp : Transform_ConcreteOpType<"bufferization.alloc_tensor">;

//===----------------------------------------------------------------------===//
// BufferLoopHoistingOp
//===----------------------------------------------------------------------===//

def BufferLoopHoistingOp
    : Op<Transform_Dialect, "bufferization.buffer_loop_hoisting",
        [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
         TransformEachOpTrait, TransformOpInterface]> {
  let description = [{
    Hoist buffer allocations ("memref.alloc" and "memref.alloca") from loops
    within the targeted op. This transform assumes that there are no buffer
    deallocation ops in the IR.

    This transform reads the `target` handle and modifies the payload.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target);
  let results = (outs);
  let assemblyFormat = "$target attr-dict `:` type($target)";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::Operation *target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

//===----------------------------------------------------------------------===//
// OneShotBufferizeOp
//===----------------------------------------------------------------------===//

def OneShotBufferizeOp
    : Op<Transform_Dialect, "bufferization.one_shot_bufferize",
        [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
         DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let description = [{
    Indicates that the given `target` op should be bufferized with One-Shot
    Bufferize. The bufferization can be configured with various attributes that
    corresponding to options in `BufferizationOptions` and the
    `one-shot-bufferize` pass. More information can be found in the pass
    documentation.

    The targeted ops must be modules or functions. This is because there is
    always a single, bufferized replacement op for such targets.

    Note: Only ops that implement `BufferizableOpInterface` are bufferized. All
    other ops are ignored if `allow_unknown_ops`. If `allow_unknown_ops` is
    unset, this transform fails when an unknown/non-bufferizable op is found.
    Many ops implement `BufferizableOpInterface` via an external model. These
    external models must be registered when applying this transform op;
    otherwise, said ops would be considered non-bufferizable.

    #### Return modes

    This operation consumes the `target` handle and produces the `transformed`
    handle.
  }];

  let arguments = (
      ins TransformHandleTypeInterface:$target,
      OptionalAttr<LayoutMapOption>:$function_boundary_type_conversion,
      DefaultValuedAttr<BoolAttr, "false">:$allow_return_allocs_from_loops,
      DefaultValuedAttr<BoolAttr, "false">:$allow_unknown_ops,
      DefaultValuedAttr<BoolAttr, "false">:$bufferize_function_boundaries,
      DefaultValuedAttr<BoolAttr, "false">:$dump_alias_sets,
      DefaultValuedAttr<BoolAttr, "false">:$test_analysis_only,
      DefaultValuedAttr<BoolAttr, "false">:$print_conflicts,
      DefaultValuedAttr<BoolAttr, "true">:$check_parallel_regions,
      DefaultValuedAttr<StrAttr, "\"memref.copy\"">:$memcpy_op);

  let results = (outs TransformHandleTypeInterface:$transformed);

  let hasVerifier = 1;
  let assemblyFormat = [{
    (`layout` `{` $function_boundary_type_conversion^ `}`)?
    $target attr-dict `:` functional-type($target, results)
  }];
}

//===----------------------------------------------------------------------===//
// EliminateEmptyTensorsOp
//===----------------------------------------------------------------------===//

def EliminateEmptyTensorsOp
    : Op<Transform_Dialect, "bufferization.eliminate_empty_tensors",
        [DeclareOpInterfaceMethods<TransformOpInterface>,
         DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let description = [{
    Try to eliminate all `tensor.empty` ops within the targeted op by replacing
    them with another destination tensor.

    "tensor.empty" ops cannot be bufferized. They can either be converted to
    "bufferization.alloc_tensor" or replaced with another tensor (via this
    transform). "tensor.empty" does not specify the contents of the returned
    tensor so their results can be replaced with arbitrary tensor values as long
    as the dimensions match.

    This transformation looks for subset ops that insert a tensor that
    originates from a "tensor.empty" (as per the reverse use-def chain). Such
    "tensor.empty" ops are replaced with the destination subset.

    Example:

    ```
    %0 = tensor.empty() : tensor<5xf32>
    %1 = linalg.fill ... outs(%0)
    %2 = tensor.insert_slice %1 into %t[1][5][1]
    ```

    Is rewritten with:
    ```
    %0 = tensor.extract_slice %t[1][5][1]
    %1 = linalg.fill ... outs(%0)
    %2 = tensor.insert_slice %1 into %t[1][5][1]
    ```

    In the above example, the subset op is "tensor.insert_slice". When tracing
    back the reverse use-def chain of a the source, we end up at a
    "tensor.empty" op.

    The above example can bufferize without an allocation (in the absence of
    other conflicts) because there is no longer a `tensor.empty` op.

    See `-eliminate-empty-tensors` for more details.

    #### Return modes

    This transform reads the target handle and modifies the payload. It does
    not produce any handle.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target);

  let results = (outs);

  let assemblyFormat = "$target attr-dict `:` type($target)";
}

//===----------------------------------------------------------------------===//
// EmptyTensorToAllocTensorOp
//===----------------------------------------------------------------------===//

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
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::tensor::EmptyOp target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

#endif // BUFFERIZATION_TRANSFORM_OPS
