//===- IRDLExtensionOps.td - Transform dialect extension ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_TRANSFORM_IRDLEXTENSION_IRDLEXTENSIONOPS
#define MLIR_DIALECT_TRANSFORM_IRDLEXTENSION_IRDLEXTENSIONOPS

include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

def IRDLCollectMatchingOp : TransformDialectOp<"irdl.collect_matching",
    [DeclareOpInterfaceMethods<TransformOpInterface>,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     SymbolTable,
     NoTerminator]> {
  let summary = 
    "Finds ops that match the IRDL definition without registering them.";

  let arguments = (ins TransformHandleTypeInterface:$root);
  let regions = (region SizedRegion<1>:$body);
  let results = (outs TransformHandleTypeInterface:$matched);

  let assemblyFormat =
    "`in` $root `:` functional-type(operands, results) attr-dict-with-keyword "
    "regions";

  let hasVerifier = 1;
}

#endif // MLIR_DIALECT_TRANSFORM_IRDLEXTENSION_IRDLEXTENSIONOPS
