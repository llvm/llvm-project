// RUN: mlir-tblgen -gen-attrdef-defs -I %S/../../include %s | FileCheck %s --check-prefix=ATTR
// RUN: mlir-tblgen -gen-typedef-defs -I %S/../../include %s | FileCheck %s --check-prefix=TYPE

include "mlir/IR/OpBase.td"

/// Test that attribute and type printers and parsers are correctly generated.
def Test_Dialect : Dialect {
  let name = "TestDialect";
  let cppNamespace = "::test";
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

// ATTR: ::mlir::Attribute TestAAttr::parse(::mlir::AsmParser &parser,
// ATTR:                                    ::mlir::Type type) {
// ATTR:   FailureOr<IntegerAttr> _result_value;
// ATTR:   FailureOr<TestParamA> _result_complex;
// ATTR:   if (parser.parseKeyword("hello"))
// ATTR:     return {};
// ATTR:   if (parser.parseEqual())
// ATTR:     return {};
// ATTR:   _result_value = ::mlir::FieldParser<IntegerAttr>::parse(parser);
// ATTR:   if (failed(_result_value))
// ATTR:     return {};
// ATTR:   if (parser.parseComma())
// ATTR:     return {};
// ATTR:   _result_complex = ::parseAttrParamA(parser, type);
// ATTR:   if (failed(_result_complex))
// ATTR:     return {};
// ATTR:   if (parser.parseRParen())
// ATTR:     return {};
// ATTR:   return TestAAttr::get(parser.getContext(),
// ATTR:                         _result_value.getValue(),
// ATTR:                         _result_complex.getValue());
// ATTR: }

// ATTR: void TestAAttr::print(::mlir::AsmPrinter &printer) const {
// ATTR:   printer << ' ' << "hello";
// ATTR:   printer << ' ' << "=";
// ATTR:   printer << ' ';
// ATTR:   printer.printStrippedAttrOrType(getValue());
// ATTR:   printer << ",";
// ATTR:   printer << ' ';
// ATTR:   ::printAttrParamA(printer, getComplex());
// ATTR:   printer << ")";
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

// ATTR: ::mlir::Attribute TestBAttr::parse(::mlir::AsmParser &parser,
// ATTR:                                    ::mlir::Type type) {
// ATTR:   bool _seen_v0 = false;
// ATTR:   bool _seen_v1 = false;
// ATTR:   for (unsigned _index = 0; _index < 2; ++_index) {
// ATTR:     StringRef _paramKey;
// ATTR:     if (parser.parseKeyword(&_paramKey))
// ATTR:       return {};
// ATTR:     if (parser.parseEqual())
// ATTR:       return {};
// ATTR:     if (!_seen_v0 && _paramKey == "v0") {
// ATTR:       _seen_v0 = true;
// ATTR:       _result_v0 = ::parseAttrParamA(parser, type);
// ATTR:       if (failed(_result_v0))
// ATTR:         return {};
// ATTR:     } else if (!_seen_v1 && _paramKey == "v1") {
// ATTR:       _seen_v1 = true;
// ATTR:       _result_v1 = type ? ::parseAttrWithType(parser, type) : ::parseAttrWithout(parser);
// ATTR:       if (failed(_result_v1))
// ATTR:         return {};
// ATTR:     } else {
// ATTR:       return {};
// ATTR:     }
// ATTR:     if ((_index != 2 - 1) && parser.parseComma())
// ATTR:       return {};
// ATTR:   }
// ATTR:   return TestBAttr::get(parser.getContext(),
// ATTR:                         _result_v0.getValue(),
// ATTR:                         _result_v1.getValue());
// ATTR: }

// ATTR: void TestBAttr::print(::mlir::AsmPrinter &printer) const {
// ATTR:   printer << "v0";
// ATTR:   printer << ' ' << "=";
// ATTR:   printer << ' ';
// ATTR:   ::printAttrParamA(printer, getV0());
// ATTR:   printer << ",";
// ATTR:   printer << ' ' << "v1";
// ATTR:   printer << ' ' << "=";
// ATTR:   printer << ' ';
// ATTR:   ::printAttrB(printer, getV1());
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

// ATTR: ::mlir::Attribute TestFAttr::parse(::mlir::AsmParser &parser,
// ATTR:                                    ::mlir::Type type) {
// ATTR:   ::mlir::FailureOr<int> _result_v0;
// ATTR:   ::mlir::FailureOr<int> _result_v1;
// ATTR:   _result_v0 = ::mlir::FieldParser<int>::parse(parser);
// ATTR:   if (failed(_result_v0))
// ATTR:     return {};
// ATTR:   if (parser.parseComma())
// ATTR:     return {};
// ATTR:   _result_v1 = ::mlir::FieldParser<int>::parse(parser);
// ATTR:   if (failed(_result_v1))
// ATTR:     return {};
// ATTR:   return TestFAttr::get(parser.getContext(),
// ATTR:     _result_v0.getValue(),
// ATTR:     _result_v1.getValue());
// ATTR: }

// ATTR: void TestFAttr::print(::mlir::AsmPrinter &printer) const {
// ATTR:   printer << ' ';
// ATTR:   printer.printStrippedAttrOrType(getV0());
// ATTR:   printer << ",";
// ATTR:   printer << ' ';
// ATTR:   printer.printStrippedAttrOrType(getV1());
// ATTR: }

def AttrC : TestAttr<"TestF"> {
  let parameters = (ins "int":$v0, "int":$v1);

  let mnemonic = "attr_c";
  let assemblyFormat = "params";
}

/// Test type parser and printer that mix variables and struct are generated
/// correctly.

// TYPE: ::mlir::Type TestCType::parse(::mlir::AsmParser &parser) {
// TYPE:  FailureOr<IntegerAttr> _result_value;
// TYPE:  FailureOr<TestParamC> _result_complex;
// TYPE:  if (parser.parseKeyword("foo"))
// TYPE:    return {};
// TYPE:  if (parser.parseComma())
// TYPE:    return {};
// TYPE:  if (parser.parseColon())
// TYPE:    return {};
// TYPE:  if (parser.parseKeyword("bob"))
// TYPE:    return {};
// TYPE:  if (parser.parseKeyword("bar"))
// TYPE:    return {};
// TYPE:  _result_value = ::mlir::FieldParser<IntegerAttr>::parse(parser);
// TYPE:  if (failed(_result_value))
// TYPE:    return {};
// TYPE:  bool _seen_complex = false;
// TYPE:  for (unsigned _index = 0; _index < 1; ++_index) {
// TYPE:    StringRef _paramKey;
// TYPE:    if (parser.parseKeyword(&_paramKey))
// TYPE:      return {};
// TYPE:    if (!_seen_complex && _paramKey == "complex") {
// TYPE:      _seen_complex = true;
// TYPE:      _result_complex = ::parseTypeParamC(parser);
// TYPE:      if (failed(_result_complex))
// TYPE:        return {};
// TYPE:    } else {
// TYPE:      return {};
// TYPE:    }
// TYPE:    if ((_index != 1 - 1) && parser.parseComma())
// TYPE:      return {};
// TYPE:  }
// TYPE:  if (parser.parseRParen())
// TYPE:    return {};
// TYPE:  }

// TYPE: void TestCType::print(::mlir::AsmPrinter &printer) const {
// TYPE:   printer << ' ' << "foo";
// TYPE:   printer << ",";
// TYPE:   printer << ' ' << ":";
// TYPE:   printer << ' ' << "bob";
// TYPE:   printer << ' ' << "bar";
// TYPE:   printer << ' ';
// TYPE:   printer.printStrippedAttrOrType(getValue());
// TYPE:   printer << ' ' << "complex";
// TYPE:   printer << ' ' << "=";
// TYPE:   printer << ' ';
// TYPE:   printer << getComplex();
// TYPE:   printer << ")";
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

// TYPE: ::mlir::Type TestDType::parse(::mlir::AsmParser &parser) {
// TYPE:   _result_v0 = ::parseTypeParamC(parser);
// TYPE:   if (failed(_result_v0))
// TYPE:     return {};
// TYPE:   bool _seen_v1 = false;
// TYPE:   bool _seen_v2 = false;
// TYPE:   for (unsigned _index = 0; _index < 2; ++_index) {
// TYPE:     StringRef _paramKey;
// TYPE:     if (parser.parseKeyword(&_paramKey))
// TYPE:       return {};
// TYPE:     if (parser.parseEqual())
// TYPE:       return {};
// TYPE:     if (!_seen_v1 && _paramKey == "v1") {
// TYPE:       _seen_v1 = true;
// TYPE:       _result_v1 = someFcnCall();
// TYPE:       if (failed(_result_v1))
// TYPE:         return {};
// TYPE:     } else if (!_seen_v2 && _paramKey == "v2") {
// TYPE:       _seen_v2 = true;
// TYPE:       _result_v2 = ::parseTypeParamC(parser);
// TYPE:       if (failed(_result_v2))
// TYPE:         return {};
// TYPE:     } else  {
// TYPE:       return {};
// TYPE:     }
// TYPE:     if ((_index != 2 - 1) && parser.parseComma())
// TYPE:       return {};
// TYPE:   }
// TYPE:   _result_v3 = someFcnCall();
// TYPE:   if (failed(_result_v3))
// TYPE:     return {};
// TYPE:   return TestDType::get(parser.getContext(),
// TYPE:                         _result_v0.getValue(),
// TYPE:                         _result_v1.getValue(),
// TYPE:                         _result_v2.getValue(),
// TYPE:                         _result_v3.getValue());
// TYPE: }

// TYPE: void TestDType::print(::mlir::AsmPrinter &printer) const {
// TYPE:   printer << getV0();
// TYPE:   myPrinter(getV1());
// TYPE:   printer << ' ' << "v2";
// TYPE:   printer << ' ' << "=";
// TYPE:   printer << ' ';
// TYPE:   printer << getV2();
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

// TYPE: ::mlir::Type TestEType::parse(::mlir::AsmParser &parser) {
// TYPE:   FailureOr<IntegerAttr> _result_v0;
// TYPE:   FailureOr<IntegerAttr> _result_v1;
// TYPE:   FailureOr<IntegerAttr> _result_v2;
// TYPE:   FailureOr<IntegerAttr> _result_v3;
// TYPE:   bool _seen_v0 = false;
// TYPE:   bool _seen_v2 = false;
// TYPE:   for (unsigned _index = 0; _index < 2; ++_index) {
// TYPE:     StringRef _paramKey;
// TYPE:     if (parser.parseKeyword(&_paramKey))
// TYPE:       return {};
// TYPE:     if (parser.parseEqual())
// TYPE:       return {};
// TYPE:     if (!_seen_v0 && _paramKey == "v0") {
// TYPE:       _seen_v0 = true;
// TYPE:       _result_v0 = ::mlir::FieldParser<IntegerAttr>::parse(parser);
// TYPE:       if (failed(_result_v0))
// TYPE:         return {};
// TYPE:     } else if (!_seen_v2 && _paramKey == "v2") {
// TYPE:       _seen_v2 = true;
// TYPE:       _result_v2 = ::mlir::FieldParser<IntegerAttr>::parse(parser);
// TYPE:       if (failed(_result_v2))
// TYPE:         return {};
// TYPE:     } else  {
// TYPE:       return {};
// TYPE:     }
// TYPE:     if ((_index != 2 - 1) && parser.parseComma())
// TYPE:       return {};
// TYPE:   }
// TYPE:   bool _seen_v1 = false;
// TYPE:   bool _seen_v3 = false;
// TYPE:   for (unsigned _index = 0; _index < 2; ++_index) {
// TYPE:     StringRef _paramKey;
// TYPE:     if (parser.parseKeyword(&_paramKey))
// TYPE:       return {};
// TYPE:     if (parser.parseEqual())
// TYPE:       return {};
// TYPE:     if (!_seen_v1 && _paramKey == "v1") {
// TYPE:       _seen_v1 = true;
// TYPE:       _result_v1 = ::mlir::FieldParser<IntegerAttr>::parse(parser);
// TYPE:       if (failed(_result_v1))
// TYPE:         return {};
// TYPE:     } else if (!_seen_v3 && _paramKey == "v3") {
// TYPE:       _seen_v3 = true;
// TYPE:       _result_v3 = ::mlir::FieldParser<IntegerAttr>::parse(parser);
// TYPE:       if (failed(_result_v3))
// TYPE:         return {};
// TYPE:     } else  {
// TYPE:       return {};
// TYPE:     }
// TYPE:     if ((_index != 2 - 1) && parser.parseComma())
// TYPE:       return {};
// TYPE:   }
// TYPE:   return TestEType::get(parser.getContext(),
// TYPE:     _result_v0.getValue(),
// TYPE:     _result_v1.getValue(),
// TYPE:     _result_v2.getValue(),
// TYPE:     _result_v3.getValue());
// TYPE: }

// TYPE: void TestEType::print(::mlir::AsmPrinter &printer) const {
// TYPE:   printer << "v0";
// TYPE:   printer << ' ' << "=";
// TYPE:   printer << ' ';
// TYPE:   printer.printStrippedAttrOrType(getV0());
// TYPE:   printer << ",";
// TYPE:   printer << ' ' << "v2";
// TYPE:   printer << ' ' << "=";
// TYPE:   printer << ' ';
// TYPE:   printer.printStrippedAttrOrType(getV2());
// TYPE:   printer << "v1";
// TYPE:   printer << ' ' << "=";
// TYPE:   printer << ' ';
// TYPE:   printer.printStrippedAttrOrType(getV1());
// TYPE:   printer << ",";
// TYPE:   printer << ' ' << "v3";
// TYPE:   printer << ' ' << "=";
// TYPE:   printer << ' ';
// TYPE:   printer.printStrippedAttrOrType(getV3());
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
