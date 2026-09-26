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
  let summary = "Return attribute (as param) associated to key via DLTI";
  let description = [{
    This op queries data layout and target information associated to payload
    IR by way of the DLTI dialect.

    A lookup of the first key is performed at `target` or its ancestors by way
    of `DLTIQueryOpInterface`. Ancestors that do not implement the interface or
    cannot answer that key are skipped. Each key should be either a (quoted)
    string or a type. If more than one key is provided, the lookup continues
    recursively on the returned attributes by way of `DLTIQueryInterface`.
    Operation ancestor lookup is not restarted for these later keys. For
    example if the payload IR is

    ```
    module attributes {dlti = #dlti.map<"A" = #dlti.map<"B" = 42 : i32>>} {
      func.func private @f()
    }
    ```
    and we have that `%func` is a Transform handle to op `@f`, then
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
