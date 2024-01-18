//===-- TestTypeDefs.td - Test dialect type definitions ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// TableGen data type definitions for Test dialect.
//
//===----------------------------------------------------------------------===//

#ifndef TESTPARAMETRIC_TYPEDEFS
#define TESTPARAMETRIC_TYPEDEFS

// To get the test dialect def.
include "TestParametricDialect.td"
include "TestParametricAttrDefs.td"
include "TestParametricInterfaces.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/Interfaces/DataLayoutInterfaces.td"

// All of the types will extend this class.
class TestParametric_Type<string name, list<Trait> traits = []>
    : TypeDef<TestParametric_Dialect, name, traits>;

def TestParametric_ParamType : TestParametric_Type<"Param"> {
  let mnemonic = "param";
  // List of type parameters.
  let parameters = (
    ins
    "::mlir::StringAttr":$ref
  );
  let assemblyFormat = "`<` $ref `>`";
}

#endif // TESTPARAMETRIC_TYPEDEFS
