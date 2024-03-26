//===- DebugExtensionOps.td - Transform Debug extension ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines operations of the transform dialect extension for debugging transform
// scripts.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_TRANSFORM_DEBUGEXTENSION_DBEUGEXTENSIONOPS
#define MLIR_DIALECT_TRANSFORM_DEBUGEXTENSION_DBEUGEXTENSIONOPS

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/Dialect/Transform/IR/MatchInterfaces.td"
include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.td"
include "mlir/Dialect/Transform/IR/TransformDialect.td"

def DebugEmitRemarkAtOp : TransformDialectOp<"debug.emit_remark_at",
  [MatchOpInterface,
   DeclareOpInterfaceMethods<TransformOpInterface>,
   MemoryEffectsOpInterface, NavigationTransformOpTrait]> {
  let summary = "Print a message as diagnostic remark attached to payload";
  let description = [{
    This operation emits a diagnostic remark with the given message at the
    location of each payload object associated with the argument. The argument
    may be an operation or a value handle.

    This operation always succeeds.
  }];

  let arguments = (ins
    Transform_AnyHandleType:$at,
    StrAttr:$message);
  let assemblyFormat = "$at `,` $message attr-dict `:` type($at)";
}

def DebugEmitParamAsRemarkOp 
  : TransformDialectOp<"debug.emit_param_as_remark",
    [MatchOpInterface,
     DeclareOpInterfaceMethods<TransformOpInterface>,
     MemoryEffectsOpInterface, NavigationTransformOpTrait]> {
  let summary = "Prints the parameter as a diagnostic remark";
  let description = [{
    This operation emits a diagnostic remark containing the string form of the
    attributes associated with the parameter provided as attribute. It takes
    as optional arguments:
      - an additional message text to prepend;
      - a handle pointing to operations the location of which will be used to
        emit the diagnostic; if multiple operations are associated, the
        diagnostic is emitted for all of their respective locations.

    This operation always succeeds.
  }];

  let arguments = (ins TransformParamTypeInterface:$param,
                       Optional<TransformHandleTypeInterface>:$anchor,
                       OptionalAttr<StrAttr>:$message);
  let assemblyFormat = "$param (`,` $message^)?  (`at` $anchor^)?"
                       "attr-dict `:` type($param) (`,` type($anchor)^)?";
}

#endif // MLIR_DIALECT_TRANSFORM_DEBUGEXTENSION_DBEUGEXTENSIONOPS
