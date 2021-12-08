// RUN: mlir-tblgen -gen-attrdef-defs -I %S/../../include %s | FileCheck %s --check-prefix=ATTR
// RUN: mlir-tblgen -gen-typedef-defs -I %S/../../include %s | FileCheck %s --check-prefix=TYPE

include "mlir/IR/OpBase.td"

/// Test that attribute and type printers and parsers are correctly generated.
def Test_Dialect : Dialect {
  let name = "TestDialect";
  let cppNamespace = "::test";

  let useDefaultAttributePrinterParser = 1;

  let useDefaultTypePrinterParser = 1;
}

class TestAttr<string name> : AttrDef<Test_Dialect, name>;
class TestType<string name> : TypeDef<Test_Dialect, name>;

def AttrA : TestAttr<"AttrA"> {
  let mnemonic = "attr_a";
}

// ATTR: namespace test {

// ATTR: ::mlir::Attribute TestDialect::parseAttribute(::mlir::DialectAsmParser &parser,
// ATTR:                                               ::mlir::Type type) const {
// ATTR:   ::llvm::SMLoc typeLoc = parser.getCurrentLocation();
// ATTR:   ::llvm::StringRef attrTag;
// ATTR:   if (::mlir::failed(parser.parseKeyword(&attrTag)))
// ATTR:     return {};
// ATTR:   {
// ATTR:     ::mlir::Attribute attr;
// ATTR:     auto parseResult = generatedAttributeParser(parser, attrTag, type, attr);
// ATTR:     if (parseResult.hasValue())
// ATTR:       return attr;
// ATTR:   }
// ATTR:   parser.emitError(typeLoc) << "unknown attribute `"
// ATTR:       << attrTag << "` in dialect `" << getNamespace() << "`";
// ATTR:   return {}
// ATTR: }

// ATTR: void TestDialect::printAttribute(::mlir::Attribute attr,
// ATTR:                          ::mlir::DialectAsmPrinter &printer) const {
// ATTR:   if (::mlir::succeeded(generatedAttributePrinter(attr, printer)))
// ATTR:     return;
// ATTR: }

// ATTR: } // namespace test

def TypeA : TestType<"TypeA"> {
  let mnemonic = "type_a";
}

// TYPE: namespace test {

// TYPE: ::mlir::Type TestDialect::parseType(::mlir::DialectAsmParser &parser) const {
// TYPE:   ::llvm::SMLoc typeLoc = parser.getCurrentLocation();
// TYPE:   ::llvm::StringRef mnemonic;
// TYPE:   if (parser.parseKeyword(&mnemonic))
// TYPE:     return ::mlir::Type();
// TYPE:   ::mlir::Type genType;
// TYPE:   auto parseResult = generatedTypeParser(parser, mnemonic, genType);
// TYPE:   if (parseResult.hasValue())
// TYPE:     return genType;
// TYPE:   parser.emitError(typeLoc) << "unknown  type `"
// TYPE:       << mnemonic << "` in dialect `" << getNamespace() << "`";
// TYPE:   return {};
// TYPE: }

// TYPE: void TestDialect::printType(::mlir::Type type,
// TYPE:                     ::mlir::DialectAsmPrinter &printer) const {
// TYPE:   if (::mlir::succeeded(generatedTypePrinter(type, printer)))
// TYPE:     return;
// TYPE: }

// TYPE: } // namespace test
