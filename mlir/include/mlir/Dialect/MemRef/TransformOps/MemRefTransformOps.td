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
include "mlir/Dialect/Transform/IR/TransformEffects.td"
include "mlir/Dialect/Transform/IR/TransformInterfaces.td"
include "mlir/Dialect/PDL/IR/PDLTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

def MemRefMultiBufferOp : Op<Transform_Dialect, "memref.multibuffer",
    [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
     TransformOpInterface, TransformEachOpTrait]> {
  let summary = "Multibuffers an allocation";
  let description = [{
     Transformation to do multi-buffering/array expansion to remove
     dependencies on the temporary allocation between consecutive loop
     iterations. This transform expands the size of an allocation by
     a given multiplicative factor and fixes up any users of the
     multibuffered allocation.

     #### Return modes

     This operation returns the new allocation if multi-buffering
     succeeds, and failure otherwise.
  }];

  let arguments =
      (ins PDL_Operation:$target,
           ConfinedAttr<I64Attr, [IntPositive]>:$factor);

  let results = (outs PDL_Operation:$transformed);

  let assemblyFormat = "$target attr-dict";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        memref::AllocOp target,
        ::llvm::SmallVector<::mlir::Operation *> &results,
        ::mlir::transform::TransformState &state);
  }];
}

#endif // MEMREF_TRANSFORM_OPS
