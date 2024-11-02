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

#ifndef TEST_TYPEDEFS
#define TEST_TYPEDEFS

// To get the test dialect def.
include "TestDialect.td"
include "TestAttrDefs.td"
include "TestInterfaces.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/Interfaces/DataLayoutInterfaces.td"

// All of the types will extend this class.
class Test_Type<string name, list<Trait> traits = []>
    : TypeDef<Test_Dialect, name, traits>;

def SimpleTypeA : Test_Type<"SimpleA"> {
  let mnemonic = "smpla";
}

// A more complex parameterized type.
def CompoundTypeA : Test_Type<"CompoundA"> {
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
    >:$arrayOfInts
  );

  let extraClassDeclaration = [{
    struct SomeCppStruct {};
  }];
  let hasCustomAssemblyFormat = 1;
}

// A more complex and nested parameterized type.
def CompoundNestedInnerType : Test_Type<"CompoundNestedInner"> {
  let mnemonic = "cmpnd_inner";
  // List of type parameters.
  let parameters = (
    ins
    "int":$some_int,
    CompoundTypeA:$cmpdA
  );
  let assemblyFormat = "`<` $some_int $cmpdA `>`";
}

def CompoundNestedOuterType : Test_Type<"CompoundNestedOuter"> {
  let mnemonic = "cmpnd_nested_outer";

  // List of type parameters.
  let parameters = (ins CompoundNestedInnerType:$inner);
  let assemblyFormat = "`<` `i`  $inner `>`";
}

def CompoundNestedOuterTypeQual : Test_Type<"CompoundNestedOuterQual"> {
  let mnemonic = "cmpnd_nested_outer_qual";

  // List of type parameters.
  let parameters = (
    ins
    CompoundNestedInnerType:$inner
  );
  let assemblyFormat = "`<` `i`  qualified($inner) `>`";
}

// An example of how one could implement a standard integer.
def IntegerType : Test_Type<"TestInteger"> {
  let mnemonic = "int";
  let genVerifyDecl = 1;
  let parameters = (
    ins
    "unsigned":$width,
    // SignednessSemantics is defined below.
    "::test::TestIntegerType::SignednessSemantics":$signedness
  );

  // Indicate we use a custom format.
  let hasCustomAssemblyFormat = 1;

  // Define custom builder methods.
  let builders = [
    TypeBuilder<(ins "unsigned":$width,
                     CArg<"SignednessSemantics", "Signless">:$signedness), [{
      return $_get($_ctxt, width, signedness);
    }]>
  ];
  let skipDefaultBuilders = 1;

  // Any extra code one wants in the type's class declaration.
  let extraClassDeclaration = [{
    /// Signedness semantics.
    enum SignednessSemantics {
      Signless, /// No signedness semantics
      Signed,   /// Signed integer
      Unsigned, /// Unsigned integer
    };

    /// Return true if this is a signless integer type.
    bool isSignless() const { return getSignedness() == Signless; }
    /// Return true if this is a signed integer type.
    bool isSigned() const { return getSignedness() == Signed; }
    /// Return true if this is an unsigned integer type.
    bool isUnsigned() const { return getSignedness() == Unsigned; }
  }];
}

// A parent type for any type which is just a list of fields (e.g. structs,
// unions).
class FieldInfo_Type<string name> : Test_Type<name> {
  let parameters = (
    ins
    // An ArrayRef of something which requires allocation in the storage
    // constructor.
    ArrayRefOfSelfAllocationParameter<
      "::test::FieldInfo", // FieldInfo is defined/declared in TestTypes.h.
      "Models struct fields">: $fields
  );
  let hasCustomAssemblyFormat = 1;
}

def StructType : FieldInfo_Type<"Struct"> {
    let mnemonic = "struct";
}

def TestType : Test_Type<"Test", [
  DeclareTypeInterfaceMethods<TestTypeInterface>
]> {
  let mnemonic = "test_type";
}

def TestTypeWithLayoutType : Test_Type<"TestTypeWithLayout", [
  DeclareTypeInterfaceMethods<DataLayoutTypeInterface, ["areCompatible"]>
]> {
  let mnemonic = "test_type_with_layout";
  let parameters = (ins "unsigned":$key);
  let extraClassDeclaration = [{
    ::mlir::LogicalResult verifyEntries(::mlir::DataLayoutEntryListRef params,
                                ::mlir::Location loc) const;

  private:
    unsigned extractKind(::mlir::DataLayoutEntryListRef params,
                         ::llvm::StringRef expectedKind) const;

  public:
  }];
  let hasCustomAssemblyFormat = 1;
}

def TestMemRefElementType : Test_Type<"TestMemRefElementType",
                                      [MemRefElementTypeInterface]> {
  let mnemonic = "memref_element";
}

def TestTypeTrait : NativeTypeTrait<"TestTypeTrait">;

// The definition of a singleton type that has a trait.
def TestTypeWithTrait : Test_Type<"TestTypeWithTrait", [TestTypeTrait]> {
  let mnemonic = "test_type_with_trait";
}

// Type with assembly format.
def TestTypeWithFormat : Test_Type<"TestTypeWithFormat"> {
  let parameters = (
    ins
    TestParamOne:$one,
    TestParamTwo:$two,
    "::mlir::Attribute":$three
  );

  let mnemonic = "type_with_format";
  let assemblyFormat = "`<` $one `,` struct($three, $two) `>`";
}

// Test dispatch to parseField
def TestTypeNoParser : Test_Type<"TestTypeNoParser"> {
  let parameters = (
    ins
    "uint32_t":$one,
    ArrayRefParameter<"int64_t">:$two,
    StringRefParameter<>:$three,
    "::test::CustomParam":$four
  );

  let mnemonic = "no_parser";
  let assemblyFormat = "`<` $one `,` `[` $two `]` `,` $three `,` $four `>`";
}

def TestTypeStructCaptureAll : Test_Type<"TestStructTypeCaptureAll"> {
  let parameters = (
    ins
    "int":$v0,
    "int":$v1,
    "int":$v2,
    "int":$v3
  );

  let mnemonic = "struct_capture_all";
  let assemblyFormat = "`<` struct(params) `>`";
}

def TestTypeOptionalParam : Test_Type<"TestTypeOptionalParam"> {
  let parameters = (ins OptionalParameter<"mlir::Optional<int>">:$a, "int":$b);
  let mnemonic = "optional_param";
  let assemblyFormat = "`<` $a `,` $b `>`";
}

def TestTypeOptionalParams : Test_Type<"TestTypeOptionalParams"> {
  let parameters = (ins OptionalParameter<"mlir::Optional<int>">:$a,
                        StringRefParameter<>:$b);
  let mnemonic = "optional_params";
  let assemblyFormat = "`<` params `>`";
}

def TestTypeOptionalParamsAfterRequired
    : Test_Type<"TestTypeOptionalParamsAfterRequired"> {
  let parameters = (ins StringRefParameter<>:$a,
                        OptionalParameter<"mlir::Optional<int>">:$b);
  let mnemonic = "optional_params_after";
  let assemblyFormat = "`<` params `>`";
}

def TestTypeOptionalStruct : Test_Type<"TestTypeOptionalStruct"> {
  let parameters = (ins OptionalParameter<"mlir::Optional<int>">:$a,
                        StringRefParameter<>:$b);
  let mnemonic = "optional_struct";
  let assemblyFormat = "`<` struct(params) `>`";
}

def TestTypeAllOptionalParams : Test_Type<"TestTypeAllOptionalParams"> {
  let parameters = (ins OptionalParameter<"mlir::Optional<int>">:$a,
                        OptionalParameter<"mlir::Optional<int>">:$b);
  let mnemonic = "all_optional_params";
  let assemblyFormat = "`<` params `>`";
}

def TestTypeAllOptionalStruct : Test_Type<"TestTypeAllOptionalStruct"> {
  let parameters = (ins OptionalParameter<"mlir::Optional<int>">:$a,
                        OptionalParameter<"mlir::Optional<int>">:$b);
  let mnemonic = "all_optional_struct";
  let assemblyFormat = "`<` struct(params) `>`";
}

def TestTypeOptionalGroup : Test_Type<"TestTypeOptionalGroup"> {
  let parameters = (ins "int":$a, OptionalParameter<"mlir::Optional<int>">:$b);
  let mnemonic = "optional_group";
  let assemblyFormat = "`<` (`(` $b^ `)`) : (`x`)? $a `>`";
}

def TestTypeOptionalGroupParams : Test_Type<"TestTypeOptionalGroupParams"> {
  let parameters = (ins DefaultValuedParameter<"mlir::Optional<int>", "10">:$a,
                        OptionalParameter<"mlir::Optional<int>">:$b);
  let mnemonic = "optional_group_params";
  let assemblyFormat = "`<` (`(` params^ `)`) : (`x`)? `>`";
}

def TestTypeOptionalGroupStruct : Test_Type<"TestTypeOptionalGroupStruct"> {
  let parameters = (ins OptionalParameter<"mlir::Optional<int>">:$a,
                        OptionalParameter<"mlir::Optional<int>">:$b);
  let mnemonic = "optional_group_struct";
  let assemblyFormat = "`<` (`(` struct(params)^ `)`) : (`x`)? `>`";
}

def TestTypeSpaces : Test_Type<"TestTypeSpaceS"> {
  let parameters = (ins "int":$a, "int":$b);
  let mnemonic = "spaces";
  let assemblyFormat = "`<` ` ` $a `\\n` `(` `)` `` `(` `)` $b `>`";
}

class DefaultValuedAPFloat<string value>
    : DefaultValuedParameter<"llvm::APFloat", "llvm::APFloat(" # value # ")"> {
  let comparator = "$_lhs.bitwiseIsEqual($_rhs)";
  let parser = [{ [&]() -> mlir::FailureOr<llvm::APFloat> {
    mlir::FloatAttr attr;
    auto result = $_parser.parseOptionalAttribute(attr);
    if (result.has_value() && mlir::succeeded(*result))
      return attr.getValue();
    if (!result.has_value())
      return llvm::APFloat(}] # value # [{);
    return mlir::failure();
  }() }];
}

def TestTypeAPFloat : Test_Type<"TestTypeAPFloat"> {
  let parameters = (ins
    DefaultValuedAPFloat<"APFloat::getZero(APFloat::IEEEdouble())">:$a
  );
  let mnemonic = "ap_float";
  let assemblyFormat = "`<` $a `>`";
}

def TestTypeDefaultValuedType : Test_Type<"TestTypeDefaultValuedType"> {
  let parameters = (ins
    DefaultValuedParameter<"mlir::IntegerType",
                           "mlir::IntegerType::get($_ctxt, 32)">:$type
  );
  let mnemonic = "default_valued_type";
  let assemblyFormat = "`<` (`(` $type^ `)`)? `>`";
}

def TestTypeCustom : Test_Type<"TestTypeCustom"> {
  let parameters = (ins "int":$a, OptionalParameter<"mlir::Optional<int>">:$b);
  let mnemonic = "custom_type";
  let assemblyFormat = [{ `<` custom<CustomTypeA>($a)
                              custom<CustomTypeB>(ref($a), $b) `>` }];
}

def TestTypeCustomString : Test_Type<"TestTypeCustomString"> {
  let parameters = (ins StringRefParameter<>:$foo);
  let mnemonic = "custom_type_string";
  let assemblyFormat = [{ `<` custom<FooString>($foo)
                              custom<BarString>(ref($foo)) `>` }];
}

def TestTypeElseAnchor : Test_Type<"TestTypeElseAnchor"> {
  let parameters = (ins OptionalParameter<"mlir::Optional<int>">:$a);
  let mnemonic = "else_anchor";
  let assemblyFormat = "`<` (`?`) : ($a^)? `>`";
}

def TestTypeElseAnchorStruct : Test_Type<"TestTypeElseAnchorStruct"> {
  let parameters = (ins OptionalParameter<"mlir::Optional<int>">:$a,
                        OptionalParameter<"mlir::Optional<int>">:$b);
  let mnemonic = "else_anchor_struct";
  let assemblyFormat = "`<` (`?`) : (struct($a, $b)^)? `>`";
}

#endif // TEST_TYPEDEFS
