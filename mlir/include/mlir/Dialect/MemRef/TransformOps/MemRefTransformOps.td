//===- MemRefTransformOps.td - MemRef transformation ops --*- tablegen -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MEMREF_TRANSFORM_OPS
#define MEMREF_TRANSFORM_OPS

include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/IR/TransformInterfaces.td"
include "mlir/Dialect/Transform/IR/TransformTypes.td"
include "mlir/Dialect/PDL/IR/PDLTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

def Transform_MemRefAllocOp : Transform_ConcreteOpType<"memref.alloc">;

def MemRefMultiBufferOp : Op<Transform_Dialect, "memref.multibuffer",
    [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
     DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let summary = "Multibuffers an allocation";
  let description = [{
     Transformation to do multi-buffering/array expansion to remove
     dependencies on the temporary allocation between consecutive loop
     iterations. This transform expands the size of an allocation by
     a given multiplicative factor and fixes up any users of the
     multibuffered allocation.
     If skip analysis is not set the transformation will only apply
     if it can prove that there is no data being carried across loop
     iterations.

     #### Return modes

     This operation returns the new allocation if multi-buffering
     succeeds, and failure otherwise.
  }];

  let arguments =
      (ins Transform_MemRefAllocOp:$target,
           ConfinedAttr<I64Attr, [IntPositive]>:$factor,
           UnitAttr:$skip_analysis);

  let results = (outs PDL_Operation:$transformed);

  let assemblyFormat =
    "$target attr-dict `:` functional-type(operands, results)";
}

def MemRefExtractAddressComputationsOp :
    Op<Transform_Dialect, "memref.extract_address_computations",
       [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
        TransformEachOpTrait, TransformOpInterface]> {
  let summary = "Extract address computations from memory accesses";
  let description = [{
    Transformation that extracts address computations from instructions
    with memory accesses such that these memory accesses use only a base
    pointer.

    For instance,
    ```mlir
    memref.load %base[%off0, ...]
    ```

    Will be rewritten in:
    ```mlir
    %new_base = memref.subview %base[%off0,...][1,...][1,...]
    memref.load %new_base[%c0,...]
    ```

    Note: The current implementation requires that the input operation
    is "isolated from above".

    #### Return modes

    This operation produces `definiteFailure` if the extraction fails for any
    reason.
    The operation always returns the handle to the target op that is expected
    to be isolated from above.
  }];

  let arguments = (ins PDL_Operation:$target);
  let results = (outs PDL_Operation:$transformed);

  let assemblyFormat = "$target attr-dict `:` functional-type(operands, results)";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::Operation *target,
        ::mlir::transform::ApplyToEachResultList &transformResults,
        ::mlir::transform::TransformState &state);
  }];
}

def MemRefMakeLoopIndependentOp
    : Op<Transform_Dialect, "memref.make_loop_independent",
         [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
          TransformOpInterface, TransformEachOpTrait]> {
  let description = [{
    Rewrite the targeted ops such that their index-typed operands no longer
    depend on any loop induction variable of the `num_loop` enclosing `scf.for`
    loops. I.e., compute an upper bound that is independent of any such loop IV
    for every tensor dimension. The transformed op could then be hoisted from
    the `num_loop` enclosing loops. To preserve the original semantics, place a
    `memref.subview` inside the loop.

    Currently supported operations are:
    - memref.alloca: Replaced with a new memref.alloca with upper bound sizes,
      followed by a memref.subview.

    #### Return modes

    This operation fails if at least one induction variable could not be
    eliminated. In case the targeted op is already independent of induction
    variables, this transform succeeds and returns the unmodified target op.

    Otherwise, the returned handle points to a subset of the produced ops:
    - memref.alloca: The returned handle points to the memref.subview op.

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

#endif // MEMREF_TRANSFORM_OPS
