//===- AffineTransformOps.td - Affine transformation ops ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AFFINE_TRANSFORM_OPS
#define AFFINE_TRANSFORM_OPS

include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/IR/TransformEffects.td"
include "mlir/Dialect/Transform/IR/TransformInterfaces.td"
include "mlir/Dialect/Transform/IR/TransformTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

def Transform_AffineForOp : Transform_ConcreteOpType<"affine.for">;

def AffineGetParentForOp : Op<Transform_Dialect, "affine.get_parent_for", [
  NavigationTransformOpTrait, MemoryEffectsOpInterface,
  DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let summary =
      "Gets a handle to the parent 'affine.for' loop of the given operation";
  let description = [{
    Produces a handle to the n-th (default 1) parent `affine.for` loop for each
    Payload IR operation associated with the operand. Fails if such a loop
    cannot be found. The list of operations associated with the handle contains
    parent operations in the same order as the list associated with the operand,
    except for operations that are parents to more than one input which are only
    present once.
  }];

  let arguments =
    (ins TransformTypeInterface:$target,
         DefaultValuedAttr<ConfinedAttr<I64Attr, [IntPositive]>,
                           "1">:$num_loops);
  let results = (outs TransformTypeInterface:$parent);

  let assemblyFormat =
      "$target attr-dict `:` functional-type(operands, results)";
}


def AffineLoopUnrollOp : Op<Transform_Dialect, "affine.unroll", [
  FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
  TransformOpInterface, TransformEachOpTrait]> {
  let summary = "Unrolls the given loop with the given unroll factor";
  let description = [{
    Unrolls each loop associated with the given handle to have up to the given
    number of loop body copies per iteration. If the unroll factor is larger
    than the loop trip count, the latter is used as the unroll factor instead.

    #### Return modes

    This operation ignores non-affine::For ops and drops them in the return.
    If all the operations referred to by the `target` PDLOperation unroll
    properly, the transform succeeds. Otherwise the transform silently fails.

    Does not return handles as the operation may result in the loop being
    removed after a full unrolling.
  }];

  let arguments = (ins Transform_AffineForOp:$target,
                       ConfinedAttr<I64Attr, [IntPositive]>:$factor);

  let assemblyFormat = "$target attr-dict `:` type($target)";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::AffineForOp target,
        ::llvm::SmallVector<::mlir::Operation *> & results,
        ::mlir::transform::TransformState & state);
  }];
}

#endif // Affine_TRANSFORM_OPS
