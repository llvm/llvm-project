//===- TensorTransformOps.td - Tensor transformation ops ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TENSOR_TRANSFORM_OPS
#define TENSOR_TRANSFORM_OPS

include "mlir/Dialect/PDL/IR/PDLTypes.td"
include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/IR/TransformInterfaces.td"
include "mlir/Dialect/Transform/IR/TransformTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

def Transform_TensorPadOp : Transform_ConcreteOpType<"tensor.pad">;

def MakeLoopIndependentOp
    : Op<Transform_Dialect, "tensor.make_loop_independent",
         [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
          TransformOpInterface, TransformEachOpTrait]> {
  let description = [{
    Rewrite the targeted ops such that their index-typed operands no longer
    depend on any loop induction variable of the `num_loop` enclosing `scf.for`
    loops. I.e., compute an upper bound that is independent of any such loop IV
    for every tensor dimension. The transformed op could then be hoisted from
    the `num_loop` enclosing loops. To preserve the original semantics, place a
    `tensor.extract_slice` inside the loop.

    Currently supported operations are:
    - tensor.empty: Replaced with a new tensor.empty with upper bound sizes,
      followed by a tensor.extract_slice.
    - tensor.pad: Replaced by an upper bound padding, followed by a
      tensor.extract_slice.

    #### Return modes

    This operation fails if at least one induction variable could not be
    eliminated. In case the targeted op is already independent of induction
    variables, this transform succeeds and returns the unmodified target op.

    Otherwise, the returned handle points to a subset of the produced ops:
    - tensor.empty: The returned handle points to the tensor.extract_slice op.
    - tensor.pad: The returned handle points to the tensor.extract_slice op.

    This transform op consumes the target handle and produces a result handle.
  }];

  let arguments = (ins PDL_Operation:$target, I64Attr:$num_loops);
  let results = (outs PDL_Operation:$transformed);
  let assemblyFormat = "$target attr-dict";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::Operation *target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

#endif // TENSOR_TRANSFORM_OPS
