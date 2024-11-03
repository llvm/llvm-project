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

#endif // MEMREF_TRANSFORM_OPS
