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

def CstArrayAttr : CIR_Attr<"CstArray", "cst_array", [TypedAttrInterface]> {
  let summary = "An Attribute containing a mlir::ArrayAttr";
  let description = [{
    An CIR array attribute is an array of literals of the specified attr types.
  }];

  // `$type` is the `self` type of the attribute (i.e. the type of the
  // Attribute itself).
  //
  // `arrayAttr` is the actual attribute array with elements for this constant
  // array, there's yet no need to own these elements.
  let parameters = (ins AttributeSelfTypeParameter<"">:$type,
                        "ArrayAttr":$value);

  // Define a custom builder for the type; that removes the need to pass
  // in an MLIRContext instance, as it can be infered from the `type`.
  let builders = [
    AttrBuilderWithInferredContext<(ins "mlir::cir::ArrayType":$type,
                                        "ArrayAttr":$value), [{
      return $_get(type.getContext(), type, value);
    }]>
  ];

  // Generate parser and printer logic, example:
  //
  // #cir.cst_array<[0 : i8, -23 : i8, 33 : i8] : !cir.array<i8 x 3>>
  //
  let assemblyFormat = "`<` $value `>`";
}

#endif // MLIR_CIR_DIALECT_CIR_ATTRS
