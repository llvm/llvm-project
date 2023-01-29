// RUN: mlir-tblgen -gen-attrdef-defs -I %S/../../include %s | FileCheck %s --check-prefix=ATTR
// RUN: mlir-tblgen -gen-typedef-defs -I %S/../../include %s | FileCheck %s --check-prefix=TYPE

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

/// Test that attribute and type printers and parsers are correctly generated.
def Test_Dialect : Dialect {
  let name = "TestDialect";
  let cppNamespace = "::test";
  let useDefaultTypePrinterParser = 0;
}

class TestAttr<string name> : AttrDef<Test_Dialect, name>;
class TestType<string name> : TypeDef<Test_Dialect, name>;

def AttrParamA : AttrParameter<"TestParamA", "an attribute param A"> {
  let parser = "::parseAttrParamA($_parser, $_type)";
  let printer = "::printAttrParamA($_printer, $_self)";
}

def AttrParamB : AttrParameter<"TestParamB", "an attribute param B"> {
  let parser = "$_type ? ::parseAttrWithType($_parser, $_type) : ::parseAttrWithout($_parser)";
  let printer = "::printAttrB($_printer, $_self)";
}

def TypeParamA : TypeParameter<"TestParamC", "a type param C"> {
  let parser = "::parseTypeParamC($_parser)";
  let printer = "$_printer << $_self";
}

def TypeParamB : TypeParameter<"TestParamD", "a type param D"> {
  let parser = "someFcnCall()";
  let printer = "myPrinter($_self)";
}

/// Check simple attribute parser and printer are generated correctly.

// ATTR: ::mlir::Attribute TestAAttr::parse(::mlir::AsmParser &odsParser,
// ATTR:                                    ::mlir::Type odsType) {
// ATTR:   FailureOr<IntegerAttr> _result_value;
// ATTR:   FailureOr<TestParamA> _result_complex;
// ATTR:   if (odsParser.parseKeyword("hello"))
// ATTR:     return {};
// ATTR:   if (odsParser.parseEqual())
// ATTR:     return {};
// ATTR:   _result_value = ::mlir::FieldParser<IntegerAttr>::parse(odsParser);
// ATTR:   if (::mlir::failed(_result_value))
// ATTR:     return {};
// ATTR:   if (odsParser.parseComma())
// ATTR:     return {};
// ATTR:   _result_complex = ::parseAttrParamA(odsParser, odsType);
// ATTR:   if (::mlir::failed(_result_complex))
// ATTR:     return {};
// ATTR:   if (odsParser.parseRParen())
// ATTR:     return {};
// ATTR:   return TestAAttr::get(odsParser.getContext(),
// ATTR:                         IntegerAttr((*_result_value)),
// ATTR:                         TestParamA((*_result_complex)));
// ATTR: }

// ATTR: void TestAAttr::print(::mlir::AsmPrinter &odsPrinter) const {
// ATTR:   odsPrinter << ' ' << "hello";
// ATTR:   odsPrinter << ' ' << "=";
// ATTR:   odsPrinter << ' ';
// ATTR:   odsPrinter.printStrippedAttrOrType(getValue());
// ATTR:   odsPrinter << ",";
// ATTR:   odsPrinter << ' ';
// ATTR:   ::printAttrParamA(odsPrinter, getComplex());
// ATTR:   odsPrinter << ")";
// ATTR: }

def AttrA : TestAttr<"TestA"> {
  let parameters = (ins
      "IntegerAttr":$value,
      AttrParamA:$complex
  );

  let mnemonic = "attr_a";
  let assemblyFormat = "`hello` `=` $value `,` $complex `)`";
}

/// Test simple struct parser and printer are generated correctly.

// ATTR: ::mlir::Attribute TestBAttr::parse(::mlir::AsmParser &odsParser,
// ATTR:                                    ::mlir::Type odsType) {
// ATTR:   bool _seen_v0 = false;
// ATTR:   bool _seen_v1 = false;
// ATTR:   const auto _loop_body = [&](::llvm::StringRef _paramKey) -> bool {
// ATTR:     if (odsParser.parseEqual())
// ATTR:       return {};
// ATTR:     if (!_seen_v0 && _paramKey == "v0") {
// ATTR:       _seen_v0 = true;
// ATTR:       _result_v0 = ::parseAttrParamA(odsParser, odsType);
// ATTR:       if (::mlir::failed(_result_v0))
// ATTR:         return {};
// ATTR:     } else if (!_seen_v1 && _paramKey == "v1") {
// ATTR:       _seen_v1 = true;
// ATTR:       _result_v1 = odsType ? ::parseAttrWithType(odsParser, odsType) :
// ATTR-SAME:                         ::parseAttrWithout(odsParser);
// ATTR:       if (::mlir::failed(_result_v1))
// ATTR:         return {};
// ATTR:     } else {
// ATTR:       return {};
// ATTR:     }
// ATTR:     return true;
// ATTR:   }
// ATTR:   for (unsigned odsStructIndex = 0; odsStructIndex < 2; ++odsStructIndex) {
// ATTR:     StringRef _paramKey;
// ATTR:     if (odsParser.parseKeyword(&_paramKey))
// ATTR:       return {};
// ATTR:     if (!_loop_body(_paramKey)) return {};
// ATTR:     if ((odsStructIndex != 2 - 1) && odsParser.parseComma())
// ATTR:       return {};
// ATTR:   }
// ATTR:   return TestBAttr::get(odsParser.getContext(),
// ATTR:                         TestParamA((*_result_v0)),
// ATTR:                         TestParamB((*_result_v1)));
// ATTR: }

// ATTR: void TestBAttr::print(::mlir::AsmPrinter &odsPrinter) const {
// ATTR:   odsPrinter << "v0 = ";
// ATTR:   ::printAttrParamA(odsPrinter, getV0());
// ATTR:   odsPrinter << ", ";
// ATTR:   odsPrinter << "v1 = ";
// ATTR:   ::printAttrB(odsPrinter, getV1());
// ATTR: }

def AttrB : TestAttr<"TestB"> {
  let parameters = (ins
      AttrParamA:$v0,
      AttrParamB:$v1
  );

  let mnemonic = "attr_b";
  let assemblyFormat = "`{` struct($v0, $v1) `}`";
}

/// Test attribute with capture-all params has correct parser and printer.

// ATTR: ::mlir::Attribute TestFAttr::parse(::mlir::AsmParser &odsParser,
// ATTR:                                    ::mlir::Type odsType) {
// ATTR:   ::mlir::FailureOr<int> _result_v0;
// ATTR:   ::mlir::FailureOr<int> _result_v1;
// ATTR:   _result_v0 = ::mlir::FieldParser<int>::parse(odsParser);
// ATTR:   if (::mlir::failed(_result_v0))
// ATTR:     return {};
// ATTR:   if (odsParser.parseComma())
// ATTR:     return {};
// ATTR:   _result_v1 = ::mlir::FieldParser<int>::parse(odsParser);
// ATTR:   if (::mlir::failed(_result_v1))
// ATTR:     return {};
// ATTR:   return TestFAttr::get(odsParser.getContext(),
// ATTR:     int((*_result_v0)),
// ATTR:     int((*_result_v1)));
// ATTR: }

def AttrC : TestAttr<"TestF"> {
  let parameters = (ins "int":$v0, "int":$v1);

  let mnemonic = "attr_c";
  let assemblyFormat = "params";
}

/// Test attribute with self type parameter

// ATTR-LABEL: Attribute TestGAttr::parse
// ATTR: if (odsType)
// ATTR:   if (auto reqType = odsType.dyn_cast<::mlir::Type>())
// ATTR:     _result_type = reqType
// ATTR: TestGAttr::get
// ATTR-NEXT: *_result_a
// ATTR-NEXT: _result_type.value_or(::mlir::NoneType::get(
def AttrD : TestAttr<"TestG"> {
  let parameters = (ins "int":$a, AttributeSelfTypeParameter<"">:$type);
  let mnemonic = "attr_d";
  let assemblyFormat = "$a";
}

// Check that the self-type parameter can be referenced without being bound.

// ATTR-LABEL: Attribute TestHAttr::parse
// ATTR: _result_type = reqType
// ATTR: parseUseType(odsParser,
// ATTR-NEXT: *_result_type

// ATTR-LABEL: void TestHAttr::print
// ATTR: printUseType(odsPrinter,
// ATTR-NEXT: getType()

def AttrE : TestAttr<"TestH"> {
  let parameters = (ins AttributeSelfTypeParameter<"">:$type);
  let mnemonic = "attr_e";
  let assemblyFormat = "custom<UseType>(ref($type))";
}

def TestEnum : I32EnumAttr<"TestEnum", "TestEnumType", [
  I32EnumAttrCase<"first", 0>,
  I32EnumAttrCase<"second", 1>
]> {
  let genSpecializedAttr = 0;
}

// ATTR-LABEL: TestEnumAttr::parse
// ATTR: parseFoo(odsParser,
// ATTR-NEXT: _result_value
// ATTR-LABEL: TestEnumAttr::print
// ATTR: printFoo(odsPrinter,
// ATTR-NEXT: getValue()
def EnumAttrA : EnumAttr<Test_Dialect, TestEnum, "EnumAttrA"> {
  let assemblyFormat = "custom<Foo>($value)";
}

/// Test type parser and printer that mix variables and struct are generated
/// correctly.

// TYPE: ::mlir::Type TestCType::parse(::mlir::AsmParser &odsParser) {
// TYPE:  FailureOr<IntegerAttr> _result_value;
// TYPE:  FailureOr<TestParamC> _result_complex;
// TYPE:  if (odsParser.parseKeyword("foo"))
// TYPE:    return {};
// TYPE:  if (odsParser.parseComma())
// TYPE:    return {};
// TYPE:  if (odsParser.parseColon())
// TYPE:    return {};
// TYPE:  if (odsParser.parseKeyword("bob"))
// TYPE:    return {};
// TYPE:  if (odsParser.parseKeyword("bar"))
// TYPE:    return {};
// TYPE:  _result_value = ::mlir::FieldParser<IntegerAttr>::parse(odsParser);
// TYPE:  if (::mlir::failed(_result_value))
// TYPE:    return {};
// TYPE:  bool _seen_complex = false;
// TYPE:  const auto _loop_body = [&](::llvm::StringRef _paramKey) -> bool {
// TYPE:    if (!_seen_complex && _paramKey == "complex") {
// TYPE:      _seen_complex = true;
// TYPE:      _result_complex = ::parseTypeParamC(odsParser);
// TYPE:      if (::mlir::failed(_result_complex))
// TYPE:        return {};
// TYPE:    } else {
// TYPE:      return {};
// TYPE:    }
// TYPE:    return true;
// TYPE:  }
// TYPE:  for (unsigned odsStructIndex = 0; odsStructIndex < 1; ++odsStructIndex) {
// TYPE:    StringRef _paramKey;
// TYPE:    if (odsParser.parseKeyword(&_paramKey))
// TYPE:      return {};
// TYPE:    if (!_loop_body(_paramKey)) return {};
// TYPE:    if ((odsStructIndex != 1 - 1) && odsParser.parseComma())
// TYPE:      return {};
// TYPE:  }
// TYPE:  if (odsParser.parseRParen())
// TYPE:    return {};
// TYPE:  }

// TYPE: void TestCType::print(::mlir::AsmPrinter &odsPrinter) const {
// TYPE:   odsPrinter << ' ' << "foo";
// TYPE:   odsPrinter << ",";
// TYPE:   odsPrinter << ' ' << ":";
// TYPE:   odsPrinter << ' ' << "bob";
// TYPE:   odsPrinter << ' ' << "bar";
// TYPE:   odsPrinter << ' ';
// TYPE:   odsPrinter.printStrippedAttrOrType(getValue());
// TYPE:   odsPrinter << "complex = ";
// TYPE:   odsPrinter << getComplex();
// TYPE:   odsPrinter << ")";
// TYPE: }

def TypeA : TestType<"TestC"> {
  let parameters = (ins
      "IntegerAttr":$value,
      TypeParamA:$complex
  );

  let mnemonic = "type_c";
  let assemblyFormat = "`foo` `,` `:` `bob` `bar` $value struct($complex) `)`";
}

/// Test type parser and printer with mix of variables and struct are generated
/// correctly.

// TYPE: ::mlir::Type TestDType::parse(::mlir::AsmParser &odsParser) {
// TYPE:   _result_v0 = ::parseTypeParamC(odsParser);
// TYPE:   if (::mlir::failed(_result_v0))
// TYPE:     return {};
// TYPE:   bool _seen_v1 = false;
// TYPE:   bool _seen_v2 = false;
// TYPE:   const auto _loop_body = [&](::llvm::StringRef _paramKey) -> bool {
// TYPE:     if (odsParser.parseEqual())
// TYPE:       return {};
// TYPE:     if (!_seen_v1 && _paramKey == "v1") {
// TYPE:       _seen_v1 = true;
// TYPE:       _result_v1 = someFcnCall();
// TYPE:       if (::mlir::failed(_result_v1))
// TYPE:         return {};
// TYPE:     } else if (!_seen_v2 && _paramKey == "v2") {
// TYPE:       _seen_v2 = true;
// TYPE:       _result_v2 = ::parseTypeParamC(odsParser);
// TYPE:       if (::mlir::failed(_result_v2))
// TYPE:         return {};
// TYPE:     } else  {
// TYPE:       return {};
// TYPE:     }
// TYPE:     return true;
// TYPE:   }
// TYPE:   for (unsigned odsStructIndex = 0; odsStructIndex < 2; ++odsStructIndex) {
// TYPE:     StringRef _paramKey;
// TYPE:     if (odsParser.parseKeyword(&_paramKey))
// TYPE:       return {};
// TYPE:     if (!_loop_body(_paramKey)) return {};
// TYPE:     if ((odsStructIndex != 2 - 1) && odsParser.parseComma())
// TYPE:       return {};
// TYPE:   }
// TYPE:   _result_v3 = someFcnCall();
// TYPE:   if (::mlir::failed(_result_v3))
// TYPE:     return {};
// TYPE:   return TestDType::get(odsParser.getContext(),
// TYPE:                         TestParamC((*_result_v0)),
// TYPE:                         TestParamD((*_result_v1)),
// TYPE:                         TestParamC((*_result_v2)),
// TYPE:                         TestParamD((*_result_v3)));
// TYPE: }

// TYPE: void TestDType::print(::mlir::AsmPrinter &odsPrinter) const {
// TYPE:   odsPrinter << getV0();
// TYPE:   myPrinter(getV1());
// TYPE:   odsPrinter << "v2 = ";
// TYPE:   odsPrinter << getV2();
// TYPE:   myPrinter(getV3());
// TYPE: }

def TypeB : TestType<"TestD"> {
  let parameters = (ins
      TypeParamA:$v0,
      TypeParamB:$v1,
      TypeParamA:$v2,
      TypeParamB:$v3
  );

  let mnemonic = "type_d";
  let assemblyFormat = "`<` `foo` `:` $v0 `,` struct($v1, $v2) `,` $v3 `>`";
}

/// Type test with two struct directives has correctly generated parser and
/// printer.

// TYPE: ::mlir::Type TestEType::parse(::mlir::AsmParser &odsParser) {
// TYPE:   FailureOr<IntegerAttr> _result_v0;
// TYPE:   FailureOr<IntegerAttr> _result_v1;
// TYPE:   FailureOr<IntegerAttr> _result_v2;
// TYPE:   FailureOr<IntegerAttr> _result_v3;
// TYPE:   bool _seen_v0 = false;
// TYPE:   bool _seen_v2 = false;
// TYPE:   const auto _loop_body = [&](::llvm::StringRef _paramKey) -> bool {
// TYPE:     if (odsParser.parseEqual())
// TYPE:       return {};
// TYPE:     if (!_seen_v0 && _paramKey == "v0") {
// TYPE:       _seen_v0 = true;
// TYPE:       _result_v0 = ::mlir::FieldParser<IntegerAttr>::parse(odsParser);
// TYPE:       if (::mlir::failed(_result_v0))
// TYPE:         return {};
// TYPE:     } else if (!_seen_v2 && _paramKey == "v2") {
// TYPE:       _seen_v2 = true;
// TYPE:       _result_v2 = ::mlir::FieldParser<IntegerAttr>::parse(odsParser);
// TYPE:       if (::mlir::failed(_result_v2))
// TYPE:         return {};
// TYPE:     } else  {
// TYPE:       return {};
// TYPE:     }
// TYPE:     return true;
// TYPE:   }
// TYPE:   for (unsigned odsStructIndex = 0; odsStructIndex < 2; ++odsStructIndex) {
// TYPE:     StringRef _paramKey;
// TYPE:     if (odsParser.parseKeyword(&_paramKey))
// TYPE:       return {};
// TYPE:     if (!_loop_body(_paramKey)) return {};
// TYPE:     if ((odsStructIndex != 2 - 1) && odsParser.parseComma())
// TYPE:       return {};
// TYPE:   }
// TYPE:   bool _seen_v1 = false;
// TYPE:   bool _seen_v3 = false;
// TYPE:   const auto _loop_body = [&](::llvm::StringRef _paramKey) -> bool {
// TYPE:     if (odsParser.parseEqual())
// TYPE:       return {};
// TYPE:     if (!_seen_v1 && _paramKey == "v1") {
// TYPE:       _seen_v1 = true;
// TYPE:       _result_v1 = ::mlir::FieldParser<IntegerAttr>::parse(odsParser);
// TYPE:       if (::mlir::failed(_result_v1))
// TYPE:         return {};
// TYPE:     } else if (!_seen_v3 && _paramKey == "v3") {
// TYPE:       _seen_v3 = true;
// TYPE:       _result_v3 = ::mlir::FieldParser<IntegerAttr>::parse(odsParser);
// TYPE:       if (::mlir::failed(_result_v3))
// TYPE:         return {};
// TYPE:     } else  {
// TYPE:       return {};
// TYPE:     }
// TYPE:     return true;
// TYPE:   }
// TYPE:   for (unsigned odsStructIndex = 0; odsStructIndex < 2; ++odsStructIndex) {
// TYPE:     StringRef _paramKey;
// TYPE:     if (odsParser.parseKeyword(&_paramKey))
// TYPE:       return {};
// TYPE:     if (!_loop_body(_paramKey)) return {};
// TYPE:     if ((odsStructIndex != 2 - 1) && odsParser.parseComma())
// TYPE:       return {};
// TYPE:   }
// TYPE:   return TestEType::get(odsParser.getContext(),
// TYPE:     IntegerAttr((*_result_v0)),
// TYPE:     IntegerAttr((*_result_v1)),
// TYPE:     IntegerAttr((*_result_v2)),
// TYPE:     IntegerAttr((*_result_v3)));
// TYPE: }

// TYPE: void TestEType::print(::mlir::AsmPrinter &odsPrinter) const {
// TYPE:   odsPrinter << "v0 = ";
// TYPE:   odsPrinter.printStrippedAttrOrType(getV0());
// TYPE:   odsPrinter << ", ";
// TYPE:   odsPrinter << "v2 = ";
// TYPE:   odsPrinter.printStrippedAttrOrType(getV2());
// TYPE:   odsPrinter << ", ";
// TYPE:   odsPrinter << "v1 = ";
// TYPE:   odsPrinter.printStrippedAttrOrType(getV1());
// TYPE:   odsPrinter << ", ";
// TYPE:   odsPrinter << "v3 = ";
// TYPE:   odsPrinter.printStrippedAttrOrType(getV3());
// TYPE: }

def TypeC : TestType<"TestE"> {
  let parameters = (ins
      "IntegerAttr":$v0,
      "IntegerAttr":$v1,
      "IntegerAttr":$v2,
      "IntegerAttr":$v3
  );

  let mnemonic = "type_e";
  let assemblyFormat = "`{` struct($v0, $v2) `}` `{` struct($v1, $v3) `}`";
}

// TYPE: ::mlir::Type TestFType::parse(::mlir::AsmParser &odsParser) {
// TYPE:   _result_a.value_or(int())

// TYPE: void TestFType::print(::mlir::AsmPrinter &odsPrinter) const {
// TYPE if (getA()) {
// TYPE   odsPrinter << ' ';
// TYPE   odsPrinter.printStrippedAttrOrType(getA());
def TypeD : TestType<"TestF"> {
  let parameters = (ins OptionalParameter<"int">:$a);
  let mnemonic = "type_f";
  let assemblyFormat = "$a";
}

// TYPE: ::mlir::Type TestGType::parse(::mlir::AsmParser &odsParser) {
// TYPE:   if (::mlir::failed(_result_a))
// TYPE:     return {};
// TYPE:   if (::mlir::succeeded(_result_a) && !((*_result_a) == int()))
// TYPE:     if (odsParser.parseComma())
// TYPE:       return {};

// TYPE: if (!(getA() == int()))
// TYPE:   odsPrinter.printStrippedAttrOrType(getA());
// TYPE: odsPrinter << ", ";
// TYPE: odsPrinter.printStrippedAttrOrType(getB());

def TypeE : TestType<"TestG"> {
  let parameters = (ins OptionalParameter<"int">:$a, "int":$b);
  let mnemonic = "type_g";
  let assemblyFormat = "params";
}


// TYPE: ::mlir::Type TestHType::parse(::mlir::AsmParser &odsParser) {
// TYPE:   do {
// TYPE:     if (!_loop_body(_paramKey)) return {};
// TYPE:   } while(!odsParser.parseOptionalComma());
// TYPE:   if (!_seen_b)
// TYPE:     return {};

// TYPE: void TestHType::print(::mlir::AsmPrinter &odsPrinter) const {
// TYPE:   if (!(getA() == int())) {
// TYPE:     odsPrinter << "a = ";
// TYPE:     odsPrinter.printStrippedAttrOrType(getA());
// TYPE:     odsPrinter << ", ";
// TYPE:   }

def TypeF : TestType<"TestH"> {
  let parameters = (ins OptionalParameter<"int">:$a, "int":$b);
  let mnemonic = "type_h";
  let assemblyFormat = "struct(params)";
}


// TYPE: do {
// TYPE:   _result_a = ::mlir::FieldParser<int>::parse(odsParser);
// TYPE:   if (::mlir::failed(_result_a))
// TYPE:     return {};
// TYPE:   if (odsParser.parseOptionalComma()) break;
// TYPE:   _result_b = ::mlir::FieldParser<int>::parse(odsParser);
// TYPE:   if (::mlir::failed(_result_b))
// TYPE:     return {};
// TYPE: } while(false);

def TypeG : TestType<"TestI"> {
  let parameters = (ins "int":$a, OptionalParameter<"int">:$b);
  let mnemonic = "type_i";
  let assemblyFormat = "params";
}

// TYPE: ::mlir::Type TestJType::parse(::mlir::AsmParser &odsParser) {
// TYPE:   if (odsParser.parseOptionalLParen()) {
// TYPE:     if (odsParser.parseKeyword("x")) return {};
// TYPE:   } else {
// TYPE:     _result_b = ::mlir::FieldParser<int>::parse(odsParser);
// TYPE:     if (::mlir::failed(_result_b))
// TYPE:       return {};
// TYPE:     if (odsParser.parseRParen()) return {};
// TYPE:   }
// TYPE:   _result_a = ::mlir::FieldParser<int>::parse(odsParser);
// TYPE:   if (::mlir::failed(_result_a))
// TYPE:     return {};

// TYPE: void TestJType::print(::mlir::AsmPrinter &odsPrinter) const {
// TYPE:   if (!(getB() == int())) {
// TYPE:     odsPrinter << "(";
// TYPE:     if (!(getB() == int()))
// TYPE:       odsPrinter.printStrippedAttrOrType(getB());
// TYPE:     odsPrinter << ")";
// TYPE:   } else {
// TYPE:     odsPrinter << ' ' << "x";
// TYPE:   }
// TYPE:   odsPrinter.printStrippedAttrOrType(getA());

def TypeH : TestType<"TestJ"> {
  let parameters = (ins "int":$a, OptionalParameter<"int">:$b);
  let mnemonic = "type_j";
  let assemblyFormat = "(`(` $b^ `)`) : (`x`)? $a";
}

// TYPE: ::mlir::Type TestKType::parse(::mlir::AsmParser &odsParser) {
// TYPE:   _result_a.value_or(10)

// TYPE: void TestKType::print(::mlir::AsmPrinter &odsPrinter) const {
// TYPE:   if (!(getA() == 10))

def TypeI : TestType<"TestK"> {
  let parameters = (ins DefaultValuedParameter<"int", "10">:$a);
  let mnemonic = "type_k";
  let assemblyFormat = "$a";
}

// TYPE: ::mlir::Type TestLType::parse
// TYPE:   auto odsCustomLoc = odsParser.getCurrentLocation()
// TYPE:   auto odsCustomResult = parseA(odsParser,
// TYPE-NEXT: _result_a
// TYPE:   if (::mlir::failed(odsCustomResult)) return {}
// TYPE:   if (::mlir::failed(_result_a))
// TYPE-NEXT: odsParser.emitError(odsCustomLoc,
// TYPE:   auto odsCustomResult = parseB(odsParser,
// TYPE-NEXT: _result_b
// TYPE-NEXT: *_result_a

// TYPE: void TestLType::print
// TYPE:   printA(odsPrinter
// TYPE-NEXT: getA()
// TYPE:   printB(odsPrinter
// TYPE-NEXT: getB()
// TYPE-NEXT: getA()

def TypeJ : TestType<"TestL"> {
  let parameters = (ins "int":$a, OptionalParameter<"Attribute">:$b);
  let mnemonic = "type_j";
  let assemblyFormat = "custom<A>($a) custom<B>($b, ref($a))";
}

// TYPE: ::mlir::Type TestMType::parse
// TYPE: FailureOr<float> _result_a
// TYPE: return TestMType::get
// TYPE: static_cast<int>((*_result_a))

def ConvertFromStorageParameter : TypeParameter<"int", ""> {
  let cppStorageType = "float";
  let convertFromStorage = "static_cast<int>($_self)";
}

def TypeK : TestType<"TestM"> {
  let parameters = (ins ConvertFromStorageParameter:$a);
  let mnemonic = "type_k";
  let assemblyFormat = "$a";
}

// TYPE-LABEL: ::mlir::Type TestNType::parse
// TYPE: parseFoo(
// TYPE-NEXT: ::mlir::detail::unwrapForCustomParse(_result_a),
// TYPE-NEXT: 1);

// TYPE-LABEL: void TestNType::print
// TYPE: printFoo(
// TYPE-NEXT: getA(),
// TYPE-NEXT: 1);

def TypeL : TestType<"TestN"> {
  let parameters = (ins "int":$a);
  let mnemonic = "type_l";
  let assemblyFormat = [{ custom<Foo>($a, "1") }];
}

// TYPE-LABEL: ::mlir::Type TestOType::parse
// TYPE: if (odsParser.parseOptionalQuestion())
// TYPE: _result_a =
// TYPE: else

// TYPE-LABEL: void TestOType::print
// TYPE: if (!(!(getA() == int())))
// TYPE: odsPrinter << ' ' << "?"
// TYPE: else
// TYPE: odsPrinter.printStrippedAttrOrType(getA())

def TypeM : TestType<"TestO"> {
  let parameters = (ins OptionalParameter<"int">:$a);
  let mnemonic = "type_m";
  let assemblyFormat = "(`?`) : ($a^)?";
}

// TYPE-LABEL: ::mlir::Type TestPType::parse
// TYPE: if (odsParser.parseOptionalQuestion())
// TYPE: bool _seen_a
// TYPE: bool _seen_b
// TYPE: _loop_body(_paramKey))
// TYPE: else {
// TYPE-NEXT: }

// TYPE-LABEL: void TestPType::print
// TYPE: if (!(!(getA() == int()) || !(getB() == int())))
// TYPE-NEXT: odsPrinter << "?"

def TypeN : TestType<"TestP"> {
  let parameters = (ins OptionalParameter<"int">:$a,
                        OptionalParameter<"int">:$b);
  let mnemonic = "type_n";
  let assemblyFormat = "`<` (`?`) : (struct($a, $b)^)? `>`";
}
