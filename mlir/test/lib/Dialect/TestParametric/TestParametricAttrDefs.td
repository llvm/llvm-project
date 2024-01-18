//===-- TestAttrDefs.td - Test dialect attr definitions ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// TableGen data attribute definitions for Test dialect.
//
//===----------------------------------------------------------------------===//

#ifndef TESTPARAMETRIC_ATTRDEFS
#define TESTPARAMETRIC_ATTRDEFS

// To get the test dialect definition.
include "TestParametricDialect.td"
include "mlir/Dialect/Utils/StructuredOpsUtils.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpAsmInterface.td"

// All of the attributes will extend this class.
class TestParametric_Attr<string name, list<Trait> traits = []>
    : AttrDef<TestParametric_Dialect, name, traits>;

def TestParametric_ParamAttr : TestParametric_Attr<"Param"> {
  let mnemonic = "param";
  // List of type parameters.
  let parameters = (
    ins
    "::mlir::StringAttr":$ref
  );
  let assemblyFormat = "`<` $ref `>`";
}

#endif // TESTPARAMETRIC_ATTRDEFS
