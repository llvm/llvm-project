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

#ifndef TEST_ATTRDEFS
#define TEST_ATTRDEFS

// To get the test dialect definition.
include "TestOps.td"

// All of the attributes will extend this class.
class Test_Attr<string name> : AttrDef<Test_Dialect, name>;

def SimpleAttrA : Test_Attr<"SimpleA"> {
  let mnemonic = "smpla";
}

// A more complex parameterized attribute.
def CompoundAttrA : Test_Attr<"CompoundA"> {
  let mnemonic = "cmpnd_a";

  // List of type parameters.
  let parameters = (
    ins
    "int":$widthOfSomething,
    "::mlir::Type":$oneType,
    // This is special syntax since ArrayRefs require allocation in the
    // constructor.
    ArrayRefParameter<
      "int", // The parameter C++ type.
      "An example of an array of ints" // Parameter description.
      >: $arrayOfInts
  );
}

// An attribute testing AttributeSelfTypeParameter.
def AttrWithSelfTypeParam : Test_Attr<"AttrWithSelfTypeParam"> {
  let mnemonic = "attr_with_self_type_param";
  let parameters = (ins AttributeSelfTypeParameter<"">:$type);
}

// An attribute testing AttributeSelfTypeParameter.
def AttrWithTypeBuilder : Test_Attr<"AttrWithTypeBuilder"> {
  let mnemonic = "attr_with_type_builder";
  let parameters = (ins "::mlir::IntegerAttr":$attr);
  let typeBuilder = "$_attr.getType()";
}

#endif // TEST_ATTRDEFS
