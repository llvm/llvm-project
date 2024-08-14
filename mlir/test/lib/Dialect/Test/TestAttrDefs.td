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
include "TestEnumDefs.td"
include "mlir/Dialect/Polynomial/IR/PolynomialAttributes.td"
include "mlir/Dialect/Utils/StructuredOpsUtils.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpAsmInterface.td"

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
def AttrWithSelfTypeParam
    : Test_Attr<"AttrWithSelfTypeParam", [TypedAttrInterface]> {
  let mnemonic = "attr_with_self_type_param";
  let parameters = (ins AttributeSelfTypeParameter<"">:$type);
  let assemblyFormat = "";
}

// An attribute testing AttributeSelfTypeParameter.
def AttrWithTypeBuilder
    : Test_Attr<"AttrWithTypeBuilder", [TypedAttrInterface]> {
  let mnemonic = "attr_with_type_builder";
  let parameters = (ins
    "::mlir::IntegerAttr":$attr,
    AttributeSelfTypeParameter<"", "mlir::Type", "$attr.getType()">:$type
  );
  let assemblyFormat = "$attr";
}

def TestAttrTrait : NativeAttrTrait<"TestAttrTrait">;

// The definition of a singleton attribute that has a trait.
def AttrWithTrait : Test_Attr<"AttrWithTrait", [TestAttrTrait]> {
  let mnemonic = "attr_with_trait";
}

// An attribute of a list of decimal formatted integers in similar format to shapes.
def TestDecimalShapeAttr : Test_Attr<"TestDecimalShape"> {
  let mnemonic = "decimal_shape";

  let parameters = (ins ArrayRefParameter<"int64_t">:$shape);

  let hasCustomAssemblyFormat = 1;
}

// Test support for ElementsAttrInterface.
def TestI64ElementsAttr : Test_Attr<"TestI64Elements", [ElementsAttrInterface]> {
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
    mlir::FailureOr<const uint64_t *>
    try_value_begin_impl(OverloadToken<uint64_t>) const {
      return getElements().begin();
    }
    // * Attribute
    auto try_value_begin_impl(OverloadToken<mlir::Attribute>) const {
      mlir::Type elementType = getType().getElementType();
      return mlir::success(llvm::map_range(getElements(), [=](uint64_t value) {
        return mlir::IntegerAttr::get(elementType,
                                      llvm::APInt(/*numBits=*/64, value));
      }).begin());
    }
    // * APInt
    auto try_value_begin_impl(OverloadToken<llvm::APInt>) const {
      return mlir::success(llvm::map_range(getElements(), [=](uint64_t value) {
        return llvm::APInt(/*numBits=*/64, value);
      }).begin());
    }
  }];
  let genVerifyDecl = 1;
  let hasCustomAssemblyFormat = 1;
}

def TestSubElementsAccessAttr : Test_Attr<"TestSubElementsAccess"> {
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


def TestParamVector : ArrayRefParameter<"int", ""> {
  let cppStorageType = "std::vector<int>";
}

def TestParamUnsigned : AttrParameter<"uint64_t", ""> {}

def TestAttrWithFormat : Test_Attr<"TestAttrWithFormat"> {
  let parameters = (
    ins
    TestParamOne:$one,
    TestParamTwo:$two,
    "::mlir::IntegerAttr":$three,
    TestParamFour:$four,
    TestParamUnsigned:$five,
    TestParamVector:$six,
    // Array of another attribute.
    ArrayRefParameter<
      "AttrWithTypeBuilderAttr", // The parameter C++ type.
      "An example of an array of another Attribute" // Parameter description.
      >: $arrayOfAttrWithTypeBuilderAttr
  );

  let mnemonic = "attr_with_format";
  let assemblyFormat = [{
    `<` $one `:` struct($two, $four) `:` $three `:` $five `:` `[` $six `]` `,`
    `[` `` $arrayOfAttrWithTypeBuilderAttr `]` `>`
  }];
  let genVerifyDecl = 1;
}

def TestAttrWithOptionalSigned : Test_Attr<"TestAttrWithOptionalSigned"> {
  let parameters = (ins OptionalParameter<"std::optional<int64_t>">:$value);
  let assemblyFormat = "`<` $value `>`";
  let mnemonic = "attr_with_optional_signed";
}

def TestAttrWithOptionalUnsigned : Test_Attr<"TestAttrWithOptionalUnsigned"> {
  let parameters = (ins OptionalParameter<"std::optional<uint64_t>">:$value);
  let assemblyFormat = "`<` $value `>`";
  let mnemonic = "attr_with_optional_unsigned";
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
    : Test_Attr<"TestAttrSelfTypeParameterFormat", [TypedAttrInterface]> {
  let parameters = (ins "int":$a, AttributeSelfTypeParameter<"">:$type);

  let mnemonic = "attr_self_type_format";
  let assemblyFormat = "`<` $a `>`";
}

def TestAttrSelfTypeParameterStructFormat
    : Test_Attr<"TestAttrSelfTypeParameterStructFormat", [TypedAttrInterface]> {
  let parameters = (ins "int":$a, AttributeSelfTypeParameter<"">:$type);

  let mnemonic = "attr_self_type_struct_format";
  let assemblyFormat = "`<` struct(params) `>`";
}

// Test overridding attribute builders with a custom builder.
def TestOverrideBuilderAttr : Test_Attr<"TestOverrideBuilder"> {
  let mnemonic = "override_builder";
  let parameters = (ins "int":$a);
  let assemblyFormat = "`<` $a `>`";

  let skipDefaultBuilders = 1;
  let builders = [AttrBuilder<(ins "int":$a), [{
    return ::mlir::IntegerAttr::get(::mlir::IndexType::get($_ctxt), a);
  }], "::mlir::Attribute">];
}

// Test simple extern 1D vector using ElementsAttrInterface.
def TestExtern1DI64ElementsAttr : Test_Attr<"TestExtern1DI64Elements", [ElementsAttrInterface]> {
  let mnemonic = "e1di64_elements";
  let parameters = (ins
    AttributeSelfTypeParameter<"", "::mlir::ShapedType">:$type,
    ResourceHandleParameter<"TestDialectResourceBlobHandle">:$handle
  );
  let extraClassDeclaration = [{
    /// Return the elements referenced by this attribute.
    llvm::ArrayRef<uint64_t> getElements() const;

    /// The set of data types that can be iterated by this attribute.
    using ContiguousIterableTypesT = std::tuple<uint64_t>;

    /// Provide begin iterators for the various iterable types.
    // * uint64_t
    mlir::FailureOr<const uint64_t *>
    try_value_begin_impl(OverloadToken<uint64_t>) const {
      return getElements().begin();
    }
  }];
  let assemblyFormat = "`<` $handle `>`";
}

// An array of nested attributes.
def TestArrayOfUglyAttrs : ArrayOfAttr<Test_Dialect, "ArrayOfUglyAttrs",
    "array_of_ugly", "TestAttrUglyAttr"> {
  let assemblyFormat = "`[` (`]`) : ($value^ ` ` `]`)?";
}

// An array of integers.
def TestArrayOfInts : ArrayOfAttr<Test_Dialect, "ArrayOfInts",
    "array_of_ints", "int32_t">;

// An array of enum attributes.
def TestSimpleEnumAttr : EnumAttr<Test_Dialect, TestSimpleEnum, "simple_enum"> {
  let assemblyFormat = "`` $value";
}
def TestArrayOfEnums : ArrayOfAttr<Test_Dialect, "ArrayOfEnums",
    "array_of_enums", "SimpleEnumAttr">;

// Test custom directive as optional group anchor.
def TestCustomAnchor : Test_Attr<"TestCustomAnchor"> {
  let parameters = (ins "int":$a, OptionalParameter<"std::optional<int>">:$b);
  let mnemonic = "custom_anchor";
  let assemblyFormat = "`<` $a (`>`) : (`,` custom<TrueFalse>($b)^ `>`)?";
}

def Test_IteratorTypeEnum
    : EnumAttr<Test_Dialect, IteratorType, "iterator_type"> {
  let assemblyFormat = "`<` $value `>`";
}

def Test_IteratorTypeArrayAttr
    : TypedArrayAttrBase<Test_IteratorTypeEnum,
  "Iterator type should be an enum.">;

def TestParamCopyCount : AttrParameter<"CopyCount", "", "const CopyCount &"> {}

// Test overridding attribute builders with a custom builder.
def TestCopyCount : Test_Attr<"TestCopyCount"> {
  let mnemonic = "copy_count";
  let parameters = (ins TestParamCopyCount:$copy_count);
  let assemblyFormat = "`<` $copy_count `>`";
}

def TestConditionalAliasAttr : Test_Attr<"TestConditionalAlias"> {
  let mnemonic = "conditional_alias";
  let parameters = (ins "mlir::StringAttr":$value);
  let assemblyFormat = [{
    `<` custom<ConditionalAlias>($value) `>`
  }];
}

// Test AsmParser::parseFloat(const fltSemnatics&, APFloat&) API through the
// custom parser and printer.
def TestCustomFloatAttr : Test_Attr<"TestCustomFloat"> {
  let mnemonic = "custom_float";
  let parameters = (ins "mlir::StringAttr":$type_str, APFloatParameter<"">:$value);

  let assemblyFormat = [{
    `<` custom<CustomFloatAttr>($type_str, $value) `>`
  }];
}

def NestedPolynomialAttr : Test_Attr<"NestedPolynomialAttr"> {
  let mnemonic = "nested_polynomial";
  let parameters = (ins Polynomial_IntPolynomialAttr:$poly);
  let assemblyFormat = [{
    `<` struct(params) `>`
  }];
}

def NestedPolynomialAttr2 : Test_Attr<"NestedPolynomialAttr2"> {
  let mnemonic = "nested_polynomial2";
  let parameters = (ins OptionalParameter<"::mlir::polynomial::IntPolynomialAttr">:$poly);
  let assemblyFormat = [{
    `<` struct(params) `>`
  }];
}


#endif // TEST_ATTRDEFS
