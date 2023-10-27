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
include "mlir/IR/EnumAttr.td"
include "mlir/Dialect/EmitC/IR/EmitCBase.td"

//===----------------------------------------------------------------------===//
// EmitC attribute definitions
//===----------------------------------------------------------------------===//

class EmitC_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<EmitC_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

def EmitC_CmpPredicateAttr : I64EnumAttr<
    "CmpPredicate", "",
    [
      I64EnumAttrCase<"eq", 0>,
      I64EnumAttrCase<"ne", 1>,
      I64EnumAttrCase<"lt", 2>,
      I64EnumAttrCase<"le", 3>,
      I64EnumAttrCase<"gt", 4>,
      I64EnumAttrCase<"ge", 5>,
      I64EnumAttrCase<"three_way", 6>,
    ]> {
  let cppNamespace = "::mlir::emitc";
}

def EmitC_OpaqueAttr : EmitC_Attr<"Opaque", "opaque"> {
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

  let parameters = (ins StringRefParameter<"the opaque value">:$value);

  let hasCustomAssemblyFormat = 1;
}

def EmitC_OpaqueOrTypedAttr : AnyAttrOf<[EmitC_OpaqueAttr, TypedAttrInterface]>;

#endif // MLIR_DIALECT_EMITC_IR_EMITCATTRIBUTES
