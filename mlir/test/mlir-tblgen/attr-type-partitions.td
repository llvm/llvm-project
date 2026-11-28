// RUN: mlir-tblgen -gen-attrdef-decls -attrdef-partition=Memory -I %S/../../include %s | FileCheck %s --check-prefix=DECLS
// RUN: mlir-tblgen -gen-attrdef-defs -attrdef-partition=Memory -I %S/../../include %s | FileCheck %s --check-prefix=DEFS
// RUN: mlir-tblgen -gen-attrdef-defs -attrdef-partition=Arith -I %S/../../include %s | FileCheck %s --check-prefix=FIRST
// RUN: mlir-tblgen -gen-typedef-defs -typedef-partition=Arith -I %S/../../include %s | FileCheck %s --check-prefix=TYPES
// RUN: mlir-tblgen -gen-attrdef-defs -I %S/../../include %s | FileCheck %s --check-prefix=ALL

include "mlir/IR/AttrTypeBase.td"

def Test_ArithPartition : Partition<"Arith">;
def Test_MemoryPartition : Partition<"Memory">;

def Test_Dialect : Dialect {
  let name = "test";
  let cppNamespace = "test";
  let hasAttributes = 1;
  let hasTypes = 1;
}

class Test_Attr<string name, string attrMnemonic> : AttrDef<Test_Dialect, name> {
  let mnemonic = attrMnemonic;
  let partition = Test_ArithPartition;
}

def IntAttr : Test_Attr<"Int", "int"> {
  let parameters = (ins "int":$value);
  let assemblyFormat = "`<` $value `>`";
}

let partition = Test_MemoryPartition in {
def AddressAttr : Test_Attr<"Address", "address"> {
  let parameters = (ins "IntAttr":$offset);
  let assemblyFormat = "`<` $offset `>`";
}
def AlignmentAttr : Test_Attr<"Alignment", "alignment">;
}

def IndexType : TypeDef<Test_Dialect, "Index"> {
  let mnemonic = "index";
  let partition = Test_ArithPartition;
}

// Only the defs of the specified partition are declared. The defs of the other
// partitions are forward declared, so that declarations can refer to them. The
// registration hook and the parse and print dispatch functions of the dialect
// are declared, as well as those of all partitions, which they register the
// defs of all partitions with and dispatch to.
// DECLS: namespace mlir {
// DECLS-NEXT: class AsmParser;
// DECLS-NEXT: class AsmPrinter;
// DECLS-NEXT: class OptionalParseResult;
// DECLS-NEXT: } // namespace mlir
// DECLS: class AddressAttr;
// DECLS-NEXT: class AlignmentAttr;
// DECLS-NEXT: class IntAttr;
// DECLS-NOT: class IntAttr :
// DECLS: class AddressAttr : public
// DECLS-NOT: class IntAttr :
// DECLS: class TestDialect;
// DECLS-NEXT: /// Registers the attributes of all partitions with `dialect`.
// DECLS-NEXT: void registerTestDialectAttributes(TestDialect *dialect);
// DECLS-NEXT: void registerTestDialectAttributesArith(TestDialect *dialect);
// DECLS-NEXT: void registerTestDialectAttributesMemory(TestDialect *dialect);
// DECLS-NEXT: ::mlir::OptionalParseResult generatedTestDialectAttributeParser(::mlir::AsmParser &parser, ::llvm::StringRef *mnemonic, ::mlir::Type type, ::mlir::Attribute &value);
// DECLS-NEXT: ::llvm::LogicalResult generatedTestDialectAttributePrinter(::mlir::Attribute def, ::mlir::AsmPrinter &printer);
// DECLS-NEXT: ::mlir::OptionalParseResult generatedTestDialectAttributeParserArith(::mlir::AsmParser &parser, ::llvm::StringRef *mnemonic, ::mlir::Type type, ::mlir::Attribute &value);
// DECLS-NEXT: ::llvm::LogicalResult generatedTestDialectAttributePrinterArith(::mlir::Attribute def, ::mlir::AsmPrinter &printer);
// DECLS-NEXT: ::mlir::OptionalParseResult generatedTestDialectAttributeParserMemory(::mlir::AsmParser &parser, ::llvm::StringRef *mnemonic, ::mlir::Type type, ::mlir::Attribute &value);
// DECLS-NEXT: ::llvm::LogicalResult generatedTestDialectAttributePrinterMemory(::mlir::Attribute def, ::mlir::AsmPrinter &printer);
// DECLS-EMPTY:
// DECLS-NEXT: } // namespace test
// DECLS-NOT: IntAttr

// The dispatch functions of a partition continue with the mnemonic that the
// dispatch functions of previous partitions parsed.
// DEFS-LABEL: #ifdef GET_ATTRDEF_LIST
// DEFS-NEXT: #undef GET_ATTRDEF_LIST
// DEFS-EMPTY:
// DEFS-NEXT: test::AddressAttr,
// DEFS-NEXT: test::AlignmentAttr
// DEFS-EMPTY:
// DEFS-NEXT: #endif
// DEFS: namespace test {
// DEFS-EMPTY:
// DEFS-NEXT: ::mlir::OptionalParseResult generatedTestDialectAttributeParserMemory(::mlir::AsmParser &parser, ::llvm::StringRef *mnemonic, ::mlir::Type type, ::mlir::Attribute &value) {
// DEFS-NEXT: return ::mlir::AsmParser::KeywordSwitch<::mlir::OptionalParseResult>(parser, mnemonic)
// DEFS: ::llvm::LogicalResult generatedTestDialectAttributePrinterMemory(::mlir::Attribute def, ::mlir::AsmPrinter &printer) {
// DEFS-NOT: registerTestDialectAttributes(
// DEFS: void registerTestDialectAttributesMemory(TestDialect *dialect) {
// DEFS-NEXT:   ::mlir::AbstractAttribute::insert<AddressAttr>(*dialect);
// DEFS-NEXT:   ::mlir::AbstractAttribute::insert<AlignmentAttr>(*dialect);
// DEFS-NEXT: }
// DEFS-EMPTY:
// DEFS-NEXT: } // namespace test
// DEFS-NOT: generatedTestDialectAttributeParser(
// DEFS-NOT: IntAttr::
// DEFS-NOT: TestDialect::parseAttribute

// The first partition by name also defines the dispatch functions and the
// registration hook of the dialect, which call those of all partitions. The
// default parser and printer of the dialect call its dispatch functions.
// FIRST: ::mlir::OptionalParseResult generatedTestDialectAttributeParser(::mlir::AsmParser &parser, ::llvm::StringRef *mnemonic, ::mlir::Type type, ::mlir::Attribute &value) {
// FIRST-NEXT:   if (::mlir::OptionalParseResult result = test::generatedTestDialectAttributeParserArith(parser, mnemonic, type, value); result.has_value())
// FIRST-NEXT:     return result;
// FIRST-NEXT:   if (::mlir::OptionalParseResult result = test::generatedTestDialectAttributeParserMemory(parser, mnemonic, type, value); result.has_value())
// FIRST-NEXT:     return result;
// FIRST-NEXT:   return std::nullopt;
// FIRST-NEXT: }
// FIRST: ::llvm::LogicalResult generatedTestDialectAttributePrinter(::mlir::Attribute def, ::mlir::AsmPrinter &printer) {
// FIRST-NEXT:   if (::mlir::succeeded(test::generatedTestDialectAttributePrinterArith(def, printer)))
// FIRST-NEXT:     return ::mlir::success();
// FIRST-NEXT:   if (::mlir::succeeded(test::generatedTestDialectAttributePrinterMemory(def, printer)))
// FIRST-NEXT:     return ::mlir::success();
// FIRST-NEXT:   return ::mlir::failure();
// FIRST-NEXT: }
// FIRST: void registerTestDialectAttributes(TestDialect *dialect) {
// FIRST-NEXT:   registerTestDialectAttributesArith(dialect);
// FIRST-NEXT:   registerTestDialectAttributesMemory(dialect);
// FIRST-NEXT: }
// FIRST-NEXT: void registerTestDialectAttributesArith(TestDialect *dialect) {
// FIRST-NEXT:   ::mlir::AbstractAttribute::insert<IntAttr>(*dialect);
// FIRST-NEXT: }
// FIRST: ::mlir::Attribute TestDialect::parseAttribute(
// FIRST: auto parseResult = generatedTestDialectAttributeParser(parser, &attrTag, type, attr);
// FIRST: if (::mlir::succeeded(generatedTestDialectAttributePrinter(attr, printer)))

// TYPES: ::mlir::OptionalParseResult generatedTestDialectTypeParserArith(::mlir::AsmParser &parser, ::llvm::StringRef *mnemonic, ::mlir::Type &value) {
// TYPES: ::mlir::OptionalParseResult generatedTestDialectTypeParser(::mlir::AsmParser &parser, ::llvm::StringRef *mnemonic, ::mlir::Type &value) {
// TYPES-NEXT:   if (::mlir::OptionalParseResult result = test::generatedTestDialectTypeParserArith(parser, mnemonic, value); result.has_value())
// TYPES: void registerTestDialectTypesArith(TestDialect *dialect) {
// TYPES-NEXT:   ::mlir::AbstractType::insert<IndexType>(*dialect);

// Without -attrdef-partition, partitions are ignored. The dispatch functions
// are local to the definitions and also available under their former names.
// ALL-NOT: generatedTestDialectAttributeParserArith
// ALL: static ::mlir::OptionalParseResult generatedTestDialectAttributeParser(
// ALL-NEXT: return ::mlir::AsmParser::KeywordSwitch<::mlir::OptionalParseResult>(parser, mnemonic)
// ALL: static ::mlir::OptionalParseResult generatedAttributeParser(
// ALL-NEXT: return test::generatedTestDialectAttributeParser(parser, mnemonic, type, value);
// ALL-NOT: generatedTestDialectAttributeParserArith
// ALL: void registerTestDialectAttributes(TestDialect *dialect) {
// ALL-NEXT:   ::mlir::AbstractAttribute::insert<IntAttr>(*dialect);
// ALL-NEXT:   ::mlir::AbstractAttribute::insert<AddressAttr>(*dialect);
// ALL-NEXT:   ::mlir::AbstractAttribute::insert<AlignmentAttr>(*dialect);
// ALL-NEXT: }
// ALL-NOT: generatedTestDialectAttributeParserArith
