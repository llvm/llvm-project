// RUN: mlir-tblgen -gen-attrdef-decls -I %S/../../include %s | FileCheck %s --check-prefix=DECL
// RUN: mlir-tblgen -gen-attrdef-defs -I %S/../../include %s | FileCheck %s --check-prefix=DEF

include "mlir/IR/OpBase.td"

// DECL: #ifdef GET_ATTRDEF_CLASSES
// DECL: #undef GET_ATTRDEF_CLASSES

// DECL: namespace mlir {
// DECL: class DialectAsmParser;
// DECL: class DialectAsmPrinter;
// DECL: } // namespace mlir

// DEF: #ifdef GET_ATTRDEF_LIST
// DEF: #undef GET_ATTRDEF_LIST
// DEF: ::mlir::test::SimpleAAttr,
// DEF: ::mlir::test::CompoundAAttr,
// DEF: ::mlir::test::IndexAttr,
// DEF: ::mlir::test::SingleParameterAttr

// DEF-LABEL: ::mlir::Attribute generatedAttributeParser(::mlir::MLIRContext *context,
// DEF-NEXT: ::mlir::DialectAsmParser &parser,
// DEF-NEXT: ::llvm::StringRef mnemonic, ::mlir::Type type) {
// DEF: if (mnemonic == ::mlir::test::CompoundAAttr::getMnemonic()) return ::mlir::test::CompoundAAttr::parse(context, parser, type);
// DEF-NEXT: if (mnemonic == ::mlir::test::IndexAttr::getMnemonic()) return ::mlir::test::IndexAttr::parse(context, parser, type);
// DEF-NEXT: return ::mlir::Attribute();

def Test_Dialect: Dialect {
// DECL-NOT: TestDialect
// DEF-NOT: TestDialect
    let name = "TestDialect";
    let cppNamespace = "::mlir::test";
}

class TestAttr<string name> : AttrDef<Test_Dialect, name> { }

def A_SimpleAttrA : TestAttr<"SimpleA"> {
// DECL: class SimpleAAttr : public ::mlir::Attribute
}

// A more complex parameterized type
def B_CompoundAttrA : TestAttr<"CompoundA"> {
  let summary = "A more complex parameterized attribute";
  let description = "This attribute is to test a reasonably complex attribute";
  let mnemonic = "cmpnd_a";
  let parameters = (
      ins
      "int":$widthOfSomething,
      "::mlir::test::SimpleTypeA": $exampleTdType,
      "SomeCppStruct": $exampleCppType,
      ArrayRefParameter<"int", "Matrix dimensions">:$dims,
      "::mlir::Type":$inner
  );

  let genVerifyDecl = 1;

// DECL-LABEL: class CompoundAAttr : public ::mlir::Attribute
// DECL: static CompoundAAttr getChecked(llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, int widthOfSomething, ::mlir::test::SimpleTypeA exampleTdType, SomeCppStruct exampleCppType, ::llvm::ArrayRef<int> dims, ::mlir::Type inner);
// DECL: static ::mlir::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, int widthOfSomething, ::mlir::test::SimpleTypeA exampleTdType, SomeCppStruct exampleCppType, ::llvm::ArrayRef<int> dims, ::mlir::Type inner);
// DECL: static constexpr ::llvm::StringLiteral getMnemonic() {
// DECL:   return ::llvm::StringLiteral("cmpnd_a");
// DECL: }
// DECL: static ::mlir::Attribute parse(::mlir::MLIRContext *context,
// DECL-NEXT: ::mlir::DialectAsmParser &parser, ::mlir::Type type);
// DECL: void print(::mlir::DialectAsmPrinter &printer) const;
// DECL: int getWidthOfSomething() const;
// DECL: ::mlir::test::SimpleTypeA getExampleTdType() const;
// DECL: SomeCppStruct getExampleCppType() const;
}

def C_IndexAttr : TestAttr<"Index"> {
    let mnemonic = "index";

    let parameters = (
      ins
      StringRefParameter<"Label for index">:$label
    );

// DECL-LABEL: class IndexAttr : public ::mlir::Attribute
// DECL: static constexpr ::llvm::StringLiteral getMnemonic() {
// DECL:   return ::llvm::StringLiteral("index");
// DECL: }
// DECL: static ::mlir::Attribute parse(::mlir::MLIRContext *context,
// DECL-NEXT: ::mlir::DialectAsmParser &parser, ::mlir::Type type);
// DECL: void print(::mlir::DialectAsmPrinter &printer) const;
}

def D_SingleParameterAttr : TestAttr<"SingleParameter"> {
  let parameters = (
    ins
    "int": $num
  );
// DECL-LABEL: struct SingleParameterAttrStorage;
// DECL-LABEL: class SingleParameterAttr
// DECL-NEXT:                   detail::SingleParameterAttrStorage
}
