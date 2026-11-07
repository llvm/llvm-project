//===- TestOpsProperties.td - Test dialect properties ops --*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_OPS_PROPERTIES
#define TEST_OPS_PROPERTIES

include "TestOpBase.td"

//===----------------------------------------------------------------------===//
// Test Properties
//===----------------------------------------------------------------------===//

// Op with a properties struct defined inline.
def TestOpWithProperties : TEST_Op<"with_properties"> {
  let assemblyFormat = [{
    `a` `=` $a `,`
    `b` `=` $b `,`
    `c` `=` $c `,`
    `flag` `=` $flag `,`
    `array` `=` $array `,`
    `array32` `=` $array32 attr-dict}];
  let arguments = (ins
    I64Prop:$a,
    StrAttr:$b, // Attributes can directly be used here.
    StringProp:$c,
    BoolProp:$flag,
    IntArrayProp<I64Prop>:$array, // Example of an array.
    IntArrayProp<I32Prop>:$array32 // Example of an array.
  );
}

def TestOpWithPropertiesAndAttr
  : TEST_Op<"with_properties_and_attr"> {
  let assemblyFormat = "$lhs prop-dict attr-dict";

  let arguments = (ins I32Attr:$lhs, IntProp<"int64_t">:$rhs);
}

def TestOpWithCustomPropDict : TEST_Op<"with_custom_prop_dict"> {
  let assemblyFormat = "prop-dict attr-dict";

  let arguments = (ins
    I32Attr:$attr,
    I64Prop:$prop,
    DefaultValuedProp<I64Prop, "42">:$defaulted,
    OptionalAttr<StrAttr>:$optional,
    UnitProp:$unit,
    UnitAttr:$unit_attr
  );
}

def TestOpWithCustomAttrPropDict
    : TEST_Op<"with_custom_attr_prop_dict"> {
  let assemblyFormat = "prop-dict attr-dict";
  let arguments = (ins
    TestEnumAttr:$attr,
    OptionalAttr<TestEnumAttr>:$optionalAttr,
    I64Prop:$prop
  );
}

def TestOpWithCustomBitEnumAttrPropDict
    : TEST_Op<"with_custom_bit_enum_attr_prop_dict"> {
  let assemblyFormat = "prop-dict attr-dict";
  let arguments = (ins TestBitEnumAttr:$flags, I64Attr:$next);
}

def TestOpWithPropertiesAndInferredType
  : TEST_Op<"with_properties_and_inferred_type", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]> {
  let assemblyFormat = "$lhs prop-dict attr-dict";

  let arguments = (ins I32Attr:$lhs, IntProp<"int64_t">:$rhs, OptionalAttr<UnitAttr>: $packed);
  let results = (outs AnyType:$result);
}

// Demonstrate how to wrap an existing C++ class named MyPropStruct.
def MyStructProperty : Property<"MyPropStruct"> {
  let convertToAttribute = "return $_storage.asAttribute($_ctxt);";
  let convertFromAttribute = "return MyPropStruct::setFromAttr($_storage, $_attr, $_diag);";
  let hashProperty = "$_storage.hash();";
  // An optional parser does not imply that the default required FieldParser is
  // available. `prop-dict` must still use attribute conversion as its fallback.
  let optionalParser = "return std::nullopt;";
}

def TestOpWithWrappedProperties : TEST_Op<"with_wrapped_properties"> {
  let assemblyFormat = "prop-dict attr-dict";
  let arguments = (ins
    MyStructProperty:$prop
  );
}

def TestOpWithDefaultWrappedProperties
    : TEST_Op<"with_default_wrapped_properties"> {
  let assemblyFormat = "prop-dict attr-dict";
  let arguments = (ins
    DefaultValuedProp<MyStructProperty, "MyPropStruct{}">:$prop
  );
}

def MyStructArrayProperty
    : Property<"::llvm::SmallVector<MyPropStruct>"> {
  let hashProperty =
      "::llvm::hash_combine_range($_storage.begin(), $_storage.end())";
  let readFromMlirBytecode = [{
    uint64_t size;
    if (::mlir::failed($_reader.readVarInt(size)))
      return ::mlir::failure();
    $_storage.clear();
    while (size--) {
      MyPropStruct value;
      if (::mlir::failed(readFromMlirBytecode($_reader, value)))
        return ::mlir::failure();
      $_storage.push_back(std::move(value));
    }
  }];
  let writeToMlirBytecode = [{
    $_writer.writeVarInt($_storage.size());
    for (MyPropStruct &value : $_storage)
      writeToMlirBytecode($_writer, value);
  }];
  let convertToAttribute = [{
    ::llvm::SmallVector<::mlir::Attribute> attrs;
    for (const MyPropStruct &value : $_storage)
      attrs.push_back(value.asAttribute($_ctxt));
    return ::mlir::ArrayAttr::get($_ctxt, attrs);
  }];
  let convertFromAttribute = [{
    auto array = ::llvm::dyn_cast<::mlir::ArrayAttr>($_attr);
    if (!array)
      return ::mlir::failure();
    $_storage.clear();
    for (::mlir::Attribute attr : array) {
      MyPropStruct value;
      if (::mlir::failed(MyPropStruct::setFromAttr(value, attr, $_diag)))
        return ::mlir::failure();
      $_storage.push_back(std::move(value));
    }
    return ::mlir::success();
  }];
}

def TestOpWithWrappedArrayProperties
    : TEST_Op<"with_wrapped_array_properties"> {
  let assemblyFormat = "prop-dict attr-dict";
  let arguments = (ins MyStructArrayProperty:$prop);
}

// The default FieldParser specializations for optional integers and generic
// containers do not consume exactly one value in an outer comma-separated
// list. The generated prop-dict parser must use attribute conversion for
// these storage types instead.
def KeyValueListProperty : Property<"::llvm::SmallVector<int64_t>"> {
  let convertToAttribute =
      "return ::mlir::DenseI64ArrayAttr::get($_ctxt, $_storage);";
  let convertFromAttribute = [{
    auto array = ::llvm::dyn_cast<::mlir::DenseI64ArrayAttr>($_attr);
    if (!array)
      return $_diag() << "expected a dense i64 array";
    auto values = array.asArrayRef();
    $_storage.assign(values.begin(), values.end());
    return ::mlir::success();
  }];
  let hashProperty =
      "::llvm::hash_combine_range($_storage.begin(), $_storage.end())";
  let readFromMlirBytecode = readMlirBytecodeUsingConvertFromAttribute;
  let writeToMlirBytecode = writeMlirBytecodeWithConvertToAttribute;
}

def KeyValueOptionalProperty : Property<"std::optional<int64_t>"> {
  let convertToAttribute = [{
    if (!$_storage)
      return ::mlir::ArrayAttr::get($_ctxt, {});
    auto value = ::mlir::IntegerAttr::get(
        ::mlir::IntegerType::get($_ctxt, 64), *$_storage);
    return ::mlir::ArrayAttr::get($_ctxt, {value});
  }];
  let convertFromAttribute = [{
    auto array = ::llvm::dyn_cast<::mlir::ArrayAttr>($_attr);
    if (!array || array.size() > 1)
      return $_diag() << "expected a zero- or one-element array";
    if (array.empty()) {
      $_storage = std::nullopt;
      return ::mlir::success();
    }
    auto value = ::llvm::dyn_cast<::mlir::IntegerAttr>(array[0]);
    if (!value)
      return $_diag() << "expected an integer element";
    $_storage = value.getInt();
    return ::mlir::success();
  }];
  let hashProperty = "::llvm::hash_value($_storage.value_or(0))";
  let readFromMlirBytecode = [{
    bool isPresent;
    if (::mlir::failed($_reader.readBool(isPresent)))
      return ::mlir::failure();
    if (!isPresent) {
      $_storage = std::nullopt;
      return ::mlir::success();
    }
    int64_t value;
    if (::mlir::failed($_reader.readSignedVarInt(value)))
      return ::mlir::failure();
    $_storage = value;
  }];
  let writeToMlirBytecode = [{
    $_writer.writeOwnedBool($_storage.has_value());
    if ($_storage)
      $_writer.writeSignedVarInt(*$_storage);
  }];
}

def KeyValueOptionalEnumProperty
    : Property<"std::optional<test::TestEnum>"> {
  let convertToAttribute = [{
    if (!$_storage)
      return ::mlir::ArrayAttr::get($_ctxt, {});
    auto value = ::mlir::IntegerAttr::get(
        ::mlir::IntegerType::get($_ctxt, 32),
        static_cast<uint32_t>(*$_storage));
    return ::mlir::ArrayAttr::get($_ctxt, {value});
  }];
  let convertFromAttribute = [{
    auto array = ::llvm::dyn_cast<::mlir::ArrayAttr>($_attr);
    if (!array || array.size() > 1)
      return $_diag() << "expected a zero- or one-element array";
    if (array.empty()) {
      $_storage = std::nullopt;
      return ::mlir::success();
    }
    auto value = ::llvm::dyn_cast<::mlir::IntegerAttr>(array[0]);
    if (!value)
      return $_diag() << "expected an integer element";
    $_storage = static_cast<test::TestEnum>(value.getInt());
    return ::mlir::success();
  }];
  let hashProperty = [{
    ::llvm::hash_value(static_cast<uint32_t>(
        $_storage.value_or(test::TestEnum::First)))
  }];
  let readFromMlirBytecode = [{
    bool isPresent;
    if (::mlir::failed($_reader.readBool(isPresent)))
      return ::mlir::failure();
    if (!isPresent) {
      $_storage = std::nullopt;
      return ::mlir::success();
    }
    uint64_t value;
    if (::mlir::failed($_reader.readVarInt(value)))
      return ::mlir::failure();
    $_storage = static_cast<test::TestEnum>(value);
  }];
  let writeToMlirBytecode = [{
    $_writer.writeOwnedBool($_storage.has_value());
    if ($_storage)
      $_writer.writeVarInt(static_cast<uint64_t>(*$_storage));
  }];
}

def KeyValueSpecializedListProperty
    : Property<"test::KeyValueSpecializedList"> {
  let printer = [{
    [&]() {
      $_printer << "[";
      ::llvm::interleaveComma($_storage, $_printer);
      $_printer << "]";
    }()
  }];
  let convertToAttribute =
      "return ::mlir::DenseI32ArrayAttr::get($_ctxt, $_storage);";
  let convertFromAttribute = [{
    auto array = ::llvm::dyn_cast<::mlir::DenseI32ArrayAttr>($_attr);
    if (!array)
      return $_diag() << "expected a dense i32 array";
    auto values = array.asArrayRef();
    $_storage.assign(values.begin(), values.end());
    return ::mlir::success();
  }];
  let hashProperty =
      "::llvm::hash_combine_range($_storage.begin(), $_storage.end())";
  let readFromMlirBytecode = readMlirBytecodeUsingConvertFromAttribute;
  let writeToMlirBytecode = writeMlirBytecodeWithConvertToAttribute;
}

def KeyValueSpecializedOptionalProperty
    : Property<"std::optional<int16_t>"> {
  let printer = [{
    [&]() {
      if (!$_storage) {
        $_printer << "none";
        return;
      }
      $_printer << "some<" << *$_storage << ">";
    }()
  }];
  let convertToAttribute = [{
    if (!$_storage)
      return ::mlir::ArrayAttr::get($_ctxt, {});
    auto value = ::mlir::IntegerAttr::get(
        ::mlir::IntegerType::get($_ctxt, 16), *$_storage);
    return ::mlir::ArrayAttr::get($_ctxt, {value});
  }];
  let convertFromAttribute = [{
    auto array = ::llvm::dyn_cast<::mlir::ArrayAttr>($_attr);
    if (!array || array.size() > 1)
      return $_diag() << "expected a zero- or one-element array";
    if (array.empty()) {
      $_storage = std::nullopt;
      return ::mlir::success();
    }
    auto value = ::llvm::dyn_cast<::mlir::IntegerAttr>(array[0]);
    if (!value)
      return $_diag() << "expected an integer element";
    $_storage = static_cast<int16_t>(value.getInt());
    return ::mlir::success();
  }];
  let hashProperty = "::llvm::hash_value($_storage.value_or(0))";
  let readFromMlirBytecode = [{
    bool isPresent;
    if (::mlir::failed($_reader.readBool(isPresent)))
      return ::mlir::failure();
    if (!isPresent) {
      $_storage = std::nullopt;
      return ::mlir::success();
    }
    int64_t value;
    if (::mlir::failed($_reader.readSignedVarInt(value)))
      return ::mlir::failure();
    $_storage = static_cast<int16_t>(value);
  }];
  let writeToMlirBytecode = [{
    $_writer.writeOwnedBool($_storage.has_value());
    if ($_storage)
      $_writer.writeSignedVarInt(*$_storage);
  }];
}

def TestOpWithKeyValueParserBoundaries
    : TEST_Op<"with_key_value_parser_boundaries"> {
  let assemblyFormat = "prop-dict attr-dict";
  let arguments = (ins
    KeyValueListProperty:$values,
    KeyValueOptionalProperty:$maybe,
    KeyValueOptionalEnumProperty:$maybeEnum,
    KeyValueSpecializedListProperty:$specializedValues,
    KeyValueSpecializedOptionalProperty:$specializedMaybe,
    I64Prop:$next
  );
}

// Same as above, but without a custom `hashProperty` field, checking
// that ADL is correctly working.
def MyStructProperty2 : Property<"MyPropStruct"> {
  let convertToAttribute = "return $_storage.asAttribute($_ctxt);";
  let convertFromAttribute = "return MyPropStruct::setFromAttr($_storage, $_attr, $_diag);";
}

def TestOpWithWrappedProperties2 : TEST_Op<"with_wrapped_properties2"> {
  let assemblyFormat = "prop-dict attr-dict";
  let arguments = (ins
    MyStructProperty2:$prop
  );
}

def TestOpWithEmptyProperties : TEST_Op<"empty_properties"> {
  let assemblyFormat = "prop-dict attr-dict";
  let arguments = (ins);
}

def TestOpUsingPropertyInCustom : TEST_Op<"using_property_in_custom"> {
  let assemblyFormat = "custom<UsingPropertyInCustom>($prop) attr-dict";
  let arguments = (ins IntArrayProp<I64Prop>:$prop);
}

def TestOpUsingPropertyInCustomAndOther
  : TEST_Op<"using_property_in_custom_and_other"> {
  let assemblyFormat = "custom<UsingPropertyInCustom>($prop) prop-dict attr-dict";
  let arguments = (ins
    IntArrayProp<I64Prop>:$prop,
    I64Prop:$other
  );
}

def TestOpWithVariadicSegmentProperties : TEST_Op<"variadic_segment_prop",
    [AttrSizedOperandSegments, AttrSizedResultSegments]> {
  let arguments = (ins Variadic<I64>:$a1, Variadic<I64>:$a2);
  let results = (outs Variadic<I64>:$b1, Variadic<I64>:$b2);
  let assemblyFormat = [{
    $a1 `:` $a2 `:` type($b1) `:` type($b2) prop-dict attr-dict `end`
  }];
}

// Same as `TestOpWithVariadicSegmentProperties`, but the assembly format uses
// the bulk `type(operands)` / `type(results)` directives instead of spelling
// out the type of each individual variadic group. This prevents the printer
// from eliding `operandSegmentSizes` / `resultSegmentSizes` from the
// `<{...}>` properties dictionary (since there is no other way to recover the
// segment sizes on re-parse), so this op exercises the parser's ability to
// read those trait-injected properties back out of `prop-dict`.
def TestOpWithBulkTypesAndSegmentProperties : TEST_Op<"variadic_segment_prop_bulk_type",
    [AttrSizedOperandSegments, AttrSizedResultSegments]> {
  let arguments = (ins Variadic<I64>:$a1, Variadic<I64>:$a2);
  let results = (outs Variadic<I64>:$b1, Variadic<I64>:$b2);
  let assemblyFormat = [{
    `(` operands `)` `:` functional-type(operands, results) prop-dict attr-dict
  }];
}

def TestOpUsingPropertyRefInCustom : TEST_Op<"using_property_ref_in_custom"> {
  let assemblyFormat = "custom<IntProperty>($first) `+` custom<SumProperty>($second, ref($first)) attr-dict";
  let arguments = (ins IntProp<"int64_t">:$first, IntProp<"int64_t">:$second);
}

def IntPropertyWithWorseBytecode : Property<"int64_t"> {
  let writeToMlirBytecode = writeMlirBytecodeWithConvertToAttribute;

  let readFromMlirBytecode = readMlirBytecodeUsingConvertFromAttribute;
}

def TestOpUsingIntPropertyWithWorseBytecode
    : TEST_Op<"using_int_property_with_worse_bytecode"> {
  let arguments = (ins IntPropertyWithWorseBytecode:$value);
}

// Op with a properties struct defined out-of-line. The struct has custom
// printer/parser.
def PropertiesWithCustomPrint : Property<"PropertiesWithCustomPrint"> {
  let convertToAttribute = [{
    return getPropertiesAsAttribute($_ctxt, $_storage);
  }];
  let convertFromAttribute = [{
    return setPropertiesFromAttribute($_storage, $_attr, $_diag);
  }];
  let hashProperty = [{
    computeHash($_storage);
  }];
}

def TestOpWithNiceProperties : TEST_Op<"with_nice_properties"> {
  let hasCustomPropertiesPrinter = 1;
  let assemblyFormat = "prop-dict attr-dict";
  let arguments = (ins
     PropertiesWithCustomPrint:$prop
  );
  let extraClassDeclaration = [{
    void printProperties(::mlir::MLIRContext *ctx, ::mlir::OpAsmPrinter &p,
                         const Properties &prop,
                         ::mlir::ArrayRef<::llvm::StringRef> elidedProps);
    static ::mlir::ParseResult  parseProperties(::mlir::OpAsmParser &parser,
                                     ::mlir::OperationState &result);
    static ::llvm::LogicalResult readFromMlirBytecode(
        ::mlir::DialectBytecodeReader &,
        test::PropertiesWithCustomPrint &prop);
    static void writeToMlirBytecode(
        ::mlir::DialectBytecodeWriter &,
        const test::PropertiesWithCustomPrint &prop);
  }];
  let extraClassDefinition = [{
    ::llvm::LogicalResult TestOpWithNiceProperties::readFromMlirBytecode(
        ::mlir::DialectBytecodeReader &reader,
        test::PropertiesWithCustomPrint &prop) {
      StringRef label;
      uint64_t value;
      if (failed(reader.readString(label)) || failed(reader.readVarInt(value)))
        return failure();
      prop.label = std::make_shared<std::string>(label.str());
      prop.value = value;
      return success();
    }
    void TestOpWithNiceProperties::writeToMlirBytecode(
        ::mlir::DialectBytecodeWriter &writer,
        const test::PropertiesWithCustomPrint &prop) {
      writer.writeOwnedString(*prop.label);
      writer.writeVarInt(prop.value);
    }
    void TestOpWithNiceProperties::printProperties(::mlir::MLIRContext *ctx,
            ::mlir::OpAsmPrinter &p, const Properties &prop,
            ::mlir::ArrayRef<::llvm::StringRef> elidedProps) {
      customPrintProperties(p, prop.prop);
    }
    ::mlir::ParseResult TestOpWithNiceProperties::parseProperties(
        ::mlir::OpAsmParser &parser,
        ::mlir::OperationState &result) {
      Properties &prop = result.getOrAddProperties<Properties>();
      if (customParseProperties(parser, prop.prop))
        return failure();
      return success();
    }
  }];
}

def VersionedProperties : Property<"VersionedProperties"> {
  let convertToAttribute = [{
    return getPropertiesAsAttribute($_ctxt, $_storage);
  }];
  let convertFromAttribute = [{
    return setPropertiesFromAttribute($_storage, $_attr, $_diag);
  }];
  let hashProperty = [{
    computeHash($_storage);
  }];
}

def TestOpWithVersionedProperties : TEST_Op<"with_versioned_properties"> {
  let hasCustomPropertiesPrinter = 1;
  let assemblyFormat = "prop-dict attr-dict";
  let arguments = (ins
     VersionedProperties:$prop
  );
  let extraClassDeclaration = [{
    void printProperties(::mlir::MLIRContext *ctx, ::mlir::OpAsmPrinter &p,
                         const Properties &prop,
                         ::mlir::ArrayRef<::llvm::StringRef> elidedProps);
    static ::mlir::ParseResult  parseProperties(::mlir::OpAsmParser &parser,
                                     ::mlir::OperationState &result);
    static ::llvm::LogicalResult readFromMlirBytecode(
        ::mlir::DialectBytecodeReader &,
        test::VersionedProperties &prop);
    static void writeToMlirBytecode(
        ::mlir::DialectBytecodeWriter &,
        const test::VersionedProperties &prop);
  }];
  let extraClassDefinition = [{
    void TestOpWithVersionedProperties::printProperties(::mlir::MLIRContext *ctx,
            ::mlir::OpAsmPrinter &p, const Properties &prop,
            ::mlir::ArrayRef<::llvm::StringRef> elidedProps) {
      customPrintProperties(p, prop.prop);
    }
    ::mlir::ParseResult TestOpWithVersionedProperties::parseProperties(
        ::mlir::OpAsmParser &parser,
        ::mlir::OperationState &result) {
      Properties &prop = result.getOrAddProperties<Properties>();
      if (customParseProperties(parser, prop.prop))
        return failure();
      return success();
    }
  }];
}

def TestOpWithDefaultValuedProperties : TEST_Op<"with_default_valued_properties"> {
  let assemblyFormat = [{
    ($a^) : (`na`)?
    ($b^)?
    ($c^)?
    ($unit^)?
    attr-dict
  }];
  let arguments = (ins DefaultValuedAttr<I32Attr, "0">:$a,
    DefaultValuedProp<StringProp, "\"\"">:$b,
    DefaultValuedProp<IntProp<"int32_t">, "-1">:$c,
    UnitProp:$unit);
}

def TestOpWithOptionalProperties : TEST_Op<"with_optional_properties"> {
  let assemblyFormat = [{
    (`anAttr` `=` $anAttr^)?
    (`simple` `=` $simple^)?
    (`simplei8` `=` $simplei8^)?
    (`simpleui8` `=` $simpleui8^)?
    (`nonTrivialStorage` `=` $nonTrivialStorage^)?
    (`hasDefault` `=` $hasDefault^)?
    (`nested` `=` $nested^)?
    (`longSyntax` `=` $longSyntax^)?
    (`hasUnit` $hasUnit^)?
    (`maybeUnit` `=` $maybeUnit^)?
    attr-dict
  }];
  let arguments = (ins
    OptionalAttr<I32Attr>:$anAttr,
    OptionalProp<I64Prop>:$simple,
    OptionalProp<IntProp<"int8_t">>:$simplei8,
    OptionalProp<IntProp<"uint8_t">>:$simpleui8,
    OptionalProp<StringProp>:$nonTrivialStorage,
    // Confirm that properties with default values now default to nullopt and have
    // the long syntax.
    OptionalProp<DefaultValuedProp<I64Prop, "0">>:$hasDefault,
    OptionalProp<OptionalProp<I64Prop>>:$nested,
    OptionalProp<StringProp, 0>:$longSyntax,
    UnitProp:$hasUnit,
    OptionalProp<UnitProp>:$maybeUnit);
}

def TestOpWithArrayProperties : TEST_Op<"with_array_properties"> {
  let assemblyFormat = [{
    `ints` `=` $ints
    `strings` `=` $strings
    `nested` `=` $nested
    `opt` `=` $opt
    `explicitOptions` `=` $explicitOptions
    `explicitUnits` `=` $explicitUnits
    ($hasDefault^ `thats_has_default`)?
    attr-dict
  }];
  let arguments = (ins
    ArrayProp<I64Prop>:$ints,
    ArrayProp<StringProp>:$strings,
    ArrayProp<ArrayProp<I32Prop>>:$nested,
    OptionalProp<ArrayProp<I32Prop>>:$opt,
    ArrayProp<OptionalProp<I64Prop>>:$explicitOptions,
    ArrayProp<UnitProp>:$explicitUnits,
    DefaultValuedProp<ArrayProp<I64Prop>,
      "::llvm::ArrayRef<int64_t>{}", "::llvm::SmallVector<int64_t>{}">:$hasDefault
  );
}

def NonNegativeI64Prop : ConfinedProp<I64Prop,
  CPred<"$_self >= 0">, "non-negative int64_t">;

class NonEmptyArray<Property p> : ConfinedProp
    <ArrayProp<p>, Neg<CPred<"$_self.empty()">>,
    "non-empty array of " # p.summary>;

def OpWithPropertyPredicates : TEST_Op<"op_with_property_predicates"> {
  let arguments = (ins
    NonNegativeI64Prop:$scalar,
    OptionalProp<NonNegativeI64Prop>:$optional,
    DefaultValuedProp<NonNegativeI64Prop, "0">:$defaulted,
    ConfinedProp<NonNegativeI64Prop,
      CPred<"$_self <= 5">, "between 0 and 5">:$more_constrained,
    ArrayProp<NonNegativeI64Prop>:$array,
    NonEmptyArray<I64Prop>:$non_empty_unconstrained,
    NonEmptyArray<NonNegativeI64Prop>:$non_empty_constrained,
    // Test applying predicates when the fromStorage() on the optional<> isn't trivial.
    OptionalProp<NonEmptyArray<NonNegativeI64Prop>>:$non_empty_optional,
    I64Prop:$unconstrained
  );
  let assemblyFormat = "attr-dict prop-dict";
}

def TestPropPatternOp1 : TEST_Op<"prop_pattern_op_1"> {
  let arguments = (ins
    StringProp:$tag,
    I64Prop:$val,
    BoolProp:$cond
  );
  let results = (outs I32:$results);
  let assemblyFormat = "$tag $val $cond attr-dict";
}

def TestPropPatternOp2 : TEST_Op<"prop_pattern_op_2"> {
  let arguments = (ins
    I32:$input,
    StringProp:$tag
  );
  let assemblyFormat = "$input $tag attr-dict";
}

#endif // TEST_OPS_PROPERTIES
