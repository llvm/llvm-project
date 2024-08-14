//===- DLTITransformOps.td - DLTI transform ops ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DLTI_TRANSFORM_OPS
#define DLTI_TRANSFORM_OPS

include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.td"
include "mlir/Dialect/Transform/IR/TransformTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

def QueryOp : Op<Transform_Dialect, "dlti.query", [
  TransformOpInterface, TransformEachOpTrait,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Return attribute (as param) associated to key via DTLI";
  let description = [{
    This op queries data layout and target information associated to payload
    IR by way of the DLTI dialect. A lookup is performed for the given `key`
    at the `target` op, with the DLTI dialect determining which interfaces and
    attributes are consulted - first checking `target` and then its ancestors.

    When only `key` is provided, the lookup occurs with respect to the data
    layout specification of DLTI. When `device` is provided, the lookup occurs
    with respect to DLTI's target device specifications associated to a DLTI
    system device specification.

    #### Return modes

    When succesful, the result, `associated_attr`, associates one attribute as a
    param for each op in `target`'s payload.

    If the lookup fails - as DLTI specifications or entries with the right
    names are missing (i.e. the values of `device` and `key`) - a definite
    failure is returned.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target,
                       OptionalAttr<StrAttr>:$device,
                       StrAttr:$key);
  let results = (outs TransformParamTypeInterface:$associated_attr);
  let assemblyFormat =
      "(`:``:` $device^ `:``:`)? $key `at` $target attr-dict `:`"
      "functional-type(operands, results)";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::Operation *target,
        ::mlir::transform::ApplyToEachResultList &results,
        TransformState &state);
  }];
}

#endif // DLTI_TRANSFORM_OPS
