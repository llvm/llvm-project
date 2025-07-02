//===- TuneExtensionOps.td - Transform dialect operations --*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_TRANSFORM_TUNEEXTENSION_TUNEEXTENSIONOPS
#define MLIR_DIALECT_TRANSFORM_TUNEEXTENSION_TUNEEXTENSIONOPS

include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/CommonAttrConstraints.td"

def SelectOp : Op<Transform_Dialect, "tune.select", [
  DeclareOpInterfaceMethods<TransformOpInterface>,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
]> {
  let summary = "Non-deterministically select a value from a set of values";
  let description = [{
    TODO
  }];
  let cppNamespace = [{ mlir::transform::tune }];
  let hasVerifier = 1;

  let arguments = (ins SymbolRefAttr:$name,
                       AnyAttr:$options,
                       OptionalAttr<AnyAttr>:$selected);
  let results = (outs TransformParamTypeInterface:$result);
  let assemblyFormat =
      "$name (`=` $selected^ `selected`)? `from` $options attr-dict `->` type(results)";
}

#endif // MLIR_DIALECT_TRANSFORM_TUNEEXTENSION_TUNEEXTENSIONOPS
