//===- CIRAttrs.td - CIR dialect types ---------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the CIR dialect attributes.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CIR_DIALECT_CIR_ATTRS
#define MLIR_CIR_DIALECT_CIR_ATTRS

include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/Dialect/CIR/IR/CIRDialect.td"
include "mlir/Dialect/CIR/IR/CIRTypes.td"

//===----------------------------------------------------------------------===//
// CIR Attrs
//===----------------------------------------------------------------------===//

class CIR_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<CIR_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

def NullAttr : CIR_Attr<"Null", "null", [TypedAttrInterface]> {
  let summary = "A simple attr to represent nullptr";
  let description = [{
    The NullAttr represents the value of nullptr within cir.
  }];

  let parameters = (ins AttributeSelfTypeParameter<"">:$type);

  let assemblyFormat = [{}];
}

#endif // MLIR_CIR_DIALECT_CIR_ATTRS
