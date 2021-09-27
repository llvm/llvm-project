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
include "mlir/IR/BuiltinAttributeInterfaces.td"

// All of the attributes will extend this class.
class Test_Attr<string name, list<Trait> traits = []>
    : AttrDef<Test_Dialect, name, traits>;

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

def TestAttrTrait : NativeAttrTrait<"TestAttrTrait">;

// The definition of a singleton attribute that has a trait.
def AttrWithTrait : Test_Attr<"AttrWithTrait", [TestAttrTrait]> {
  let mnemonic = "attr_with_trait";
  let parameters = (ins );
}

// Test support for ElementsAttrInterface.
def TestI64ElementsAttr : Test_Attr<"TestI64Elements", [
    ElementsAttrInterface
  ]> {
  let mnemonic = "i64_elements";
  let parameters = (ins
    AttributeSelfTypeParameter<"", "::mlir::ShapedType">:$type,
    ArrayRefParameter<"uint64_t">:$elements
  );
  let extraClassDeclaration = [{
    /// The set of data types that can be iterated by this attribute.
    using ContiguousIterableTypesT = std::tuple<uint64_t>;
    using NonContiguousIterableTypesT = std::tuple<mlir::Attribute, llvm::APInt>;

    /// Provide begin iterators for the various iterable types.
    // * uint64_t
    auto value_begin_impl(OverloadToken<uint64_t>) const {
      return getElements().begin();
    }
    // * Attribute
    auto value_begin_impl(OverloadToken<mlir::Attribute>) const {
      mlir::Type elementType = getType().getElementType();
      return llvm::map_range(getElements(), [=](uint64_t value) {
        return mlir::IntegerAttr::get(elementType,
                                      llvm::APInt(/*numBits=*/64, value));
      }).begin();
    }
    // * APInt
    auto value_begin_impl(OverloadToken<llvm::APInt>) const {
      return llvm::map_range(getElements(), [=](uint64_t value) {
        return llvm::APInt(/*numBits=*/64, value);
      }).begin();
    }
  }];
  let genVerifyDecl = 1;
}

#endif // TEST_ATTRDEFS
