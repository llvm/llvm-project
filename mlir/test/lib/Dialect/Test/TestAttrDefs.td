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
include "TestDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SubElementInterfaces.td"

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
  let hasCustomAssemblyFormat = 1;
}
def CompoundAttrNested : Test_Attr<"CompoundAttrNested"> {
  let mnemonic = "cmpnd_nested";
  let parameters = (ins CompoundAttrA : $nested );
  let assemblyFormat = "`<` `nested` `=` $nested `>`";
}

// An attribute testing AttributeSelfTypeParameter.
def AttrWithSelfTypeParam : Test_Attr<"AttrWithSelfTypeParam"> {
  let mnemonic = "attr_with_self_type_param";
  let parameters = (ins AttributeSelfTypeParameter<"">:$type);
  let assemblyFormat = "";
}

// An attribute testing AttributeSelfTypeParameter.
def AttrWithTypeBuilder : Test_Attr<"AttrWithTypeBuilder"> {
  let mnemonic = "attr_with_type_builder";
  let parameters = (ins "::mlir::IntegerAttr":$attr);
  let typeBuilder = "$_attr.getType()";
  let hasCustomAssemblyFormat = 1;
}

def TestAttrTrait : NativeAttrTrait<"TestAttrTrait">;

// The definition of a singleton attribute that has a trait.
def AttrWithTrait : Test_Attr<"AttrWithTrait", [TestAttrTrait]> {
  let mnemonic = "attr_with_trait";
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
  let hasCustomAssemblyFormat = 1;
}

def TestSubElementsAccessAttr : Test_Attr<"TestSubElementsAccess", [
    DeclareAttrInterfaceMethods<SubElementAttrInterface,
        ["replaceImmediateSubAttribute"]>
  ]> {

  let mnemonic = "sub_elements_access";

  let parameters = (ins
    "::mlir::Attribute":$first,
    "::mlir::Attribute":$second,
    "::mlir::Attribute":$third
  );
  let hasCustomAssemblyFormat = 1;
}

// A more complex parameterized attribute with multiple level of nesting.
def CompoundNestedInner : Test_Attr<"CompoundNestedInner"> {
  let mnemonic = "cmpnd_nested_inner";
  // List of type parameters.
  let parameters = (
    ins
    "int":$some_int,
    CompoundAttrA:$cmpdA
  );
  let assemblyFormat = "`<` $some_int $cmpdA `>`";
}

def CompoundNestedOuter : Test_Attr<"CompoundNestedOuter"> {
  let mnemonic = "cmpnd_nested_outer";

  // List of type parameters.
  let parameters = (
    ins
    CompoundNestedInner:$inner
  );
  let assemblyFormat = "`<` `i`  $inner `>`";
}

def CompoundNestedOuterQual : Test_Attr<"CompoundNestedOuterQual"> {
  let mnemonic = "cmpnd_nested_outer_qual";

  // List of type parameters.
  let parameters = (ins CompoundNestedInner:$inner);
  let assemblyFormat = "`<` `i`  qualified($inner) `>`";
}

def TestParamOne : AttrParameter<"int64_t", ""> {}

def TestParamTwo : AttrParameter<"std::string", "", "llvm::StringRef"> {
  let printer = "$_printer << '\"' << $_self << '\"'";
}

def TestParamFour : ArrayRefParameter<"int", ""> {
  let cppStorageType = "llvm::SmallVector<int>";
  let parser = "::parseIntArray($_parser)";
  let printer = "::printIntArray($_printer, $_self)";
}

def TestAttrWithFormat : Test_Attr<"TestAttrWithFormat"> {
  let parameters = (
    ins
    TestParamOne:$one,
    TestParamTwo:$two,
    "::mlir::IntegerAttr":$three,
    TestParamFour:$four,
    // Array of another attribute.
    ArrayRefParameter<
      "AttrWithTypeBuilderAttr", // The parameter C++ type.
      "An example of an array of another Attribute" // Parameter description.
      >: $arrayOfAttrWithTypeBuilderAttr
  );

  let mnemonic = "attr_with_format";
  let assemblyFormat = [{
    `<` $one `:` struct($two, $four) `:` $three `,`
    `[` `` $arrayOfAttrWithTypeBuilderAttr `]` `>`
  }];
  let genVerifyDecl = 1;
}

def TestAttrUgly : Test_Attr<"TestAttrUgly"> {
  let parameters = (ins "::mlir::Attribute":$attr);

  let mnemonic = "attr_ugly";
  let assemblyFormat = "`begin` $attr `end`";
}

def TestAttrParams: Test_Attr<"TestAttrParams"> {
  let parameters = (ins "int":$v0, "int":$v1);

  let mnemonic = "attr_params";
  let assemblyFormat = "`<` params `>`";
}

// Test types can be parsed/printed.
def TestAttrWithTypeParam : Test_Attr<"TestAttrWithTypeParam"> {
  let parameters = (ins "::mlir::IntegerType":$int_type,
                        "::mlir::Type":$any_type);
  let mnemonic = "attr_with_type";
  let assemblyFormat = "`<` $int_type `,` $any_type `>`";
}

// Test self type parameter with assembly format.
def TestAttrSelfTypeParameterFormat
    : Test_Attr<"TestAttrSelfTypeParameterFormat"> {
  let parameters = (ins "int":$a, AttributeSelfTypeParameter<"">:$type);

  let mnemonic = "attr_self_type_format";
  let assemblyFormat = "`<` $a `>`";
}

// Test overridding attribute builders with a custom builder.
def TestOverrideBuilderAttr : Test_Attr<"TestOverrideBuilder"> {
  let mnemonic = "override_builder";
  let parameters = (ins "int":$a);
  let assemblyFormat = "`<` $a `>`";

  let skipDefaultBuilders = 1;
  let genVerifyDecl = 1;
  let builders = [AttrBuilder<(ins "int":$a), [{
    return ::mlir::IntegerAttr::get(::mlir::IndexType::get($_ctxt), a);
  }], "::mlir::Attribute">];
}

// Test simple extern 1D vector using ElementsAttrInterface.
def TestExtern1DI64ElementsAttr : Test_Attr<"TestExtern1DI64Elements", [
    ElementsAttrInterface
  ]> {
  let mnemonic = "e1di64_elements";
  let parameters = (ins
    AttributeSelfTypeParameter<"", "::mlir::ShapedType">:$type,
    ResourceHandleParameter<"TestExternalElementsDataHandle">:$handle
  );
  let extraClassDeclaration = [{
    /// Return the elements referenced by this attribute.
    llvm::ArrayRef<uint64_t> getElements() const;

    /// The set of data types that can be iterated by this attribute.
    using ContiguousIterableTypesT = std::tuple<uint64_t>;

    /// Provide begin iterators for the various iterable types.
    // * uint64_t
    auto value_begin_impl(OverloadToken<uint64_t>) const {
      return getElements().begin();
    }
  }];
  let assemblyFormat = "`<` $handle `>`";
}

#endif // TEST_ATTRDEFS
