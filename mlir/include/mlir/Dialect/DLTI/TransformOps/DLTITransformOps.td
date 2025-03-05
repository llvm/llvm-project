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
    IR by way of the DLTI dialect.

    A lookup is performed for the given `keys` at `target` op - or its closest
    interface-implementing ancestor - by way of the `DLTIQueryInterface`, which
    returns an attribute for a key. Each key should be either a (quoted) string
    or a type. If more than one key is provided, the lookup continues
    recursively, now on the returned attributes, with the condition that these
    implement the above interface. For example if the payload IR is

    ```
    module attributes {#dlti.map = #dlti.map<#dlti.dl_entry<"A",
                                     #dlti.map<#dlti.dl_entry<"B", 42: int>>>} {
      func.func private @f()
    }
    ```
    and we have that `%func` is a Tranform handle to op `@f`, then
    `transform.dlti.query ["A", "B"] at %func` returns 42 as a param and
    `transform.dlti.query ["A"] at %func` returns the `#dlti.map` attribute
    containing just the key "B" and its value. Using `["B"]` or `["A","C"]` as
    `keys` will yield an error.

    #### Return modes

    When successful, the result, `associated_attr`, associates one attribute as
    a param for each op in `target`'s payload.

    If the lookup fails - as no DLTI attributes/interfaces are found or entries
    with the right names are missing - a silenceable failure is returned.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target,
                       ArrayAttr:$keys);
  let results = (outs TransformParamTypeInterface:$associated_attr);
  let assemblyFormat =
      "$keys `at` $target attr-dict `:` functional-type(operands, results)";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::Operation *target,
        ::mlir::transform::ApplyToEachResultList &results,
        TransformState &state);
  }];
}

#endif // DLTI_TRANSFORM_OPS
