//===- EmitCAttributes.td - EmitC attributes ---------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the MLIR EmitC attributes.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_EMITC_IR_EMITCATTRIBUTES
#define MLIR_DIALECT_EMITC_IR_EMITCATTRIBUTES

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/Dialect/EmitC/IR/EmitCBase.td"

//===----------------------------------------------------------------------===//
// EmitC attribute definitions
//===----------------------------------------------------------------------===//

class EmitC_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<EmitC_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

def EmitC_OpaqueAttr : EmitC_Attr<"Opaque", "opaque", [TypedAttrInterface]> {
  let summary = "An opaque attribute";

  let description = [{
    An opaque attribute of which the value gets emitted as is.

    Example:

    ```mlir
    #emitc.opaque<"">
    #emitc.opaque<"NULL">
    #emitc.opaque<"nullptr">
    ```
  }];

  let parameters = (ins "Type":$type,
                        StringRefParameter<"the opaque value">:$value);

  let hasCustomAssemblyFormat = 1;
}

#endif // MLIR_DIALECT_EMITC_IR_EMITCATTRIBUTES
