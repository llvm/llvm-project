// RUN: mlir-tblgen -gen-attrdef-decls -I %S/../../include %s | FileCheck %s --check-prefix=DECL
// RUN: mlir-tblgen -gen-attrdef-defs -I %S/../../include %s | FileCheck %s --check-prefix=DEF

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

// DECL: #ifdef GET_ATTRDEF_CLASSES
// DECL: #undef GET_ATTRDEF_CLASSES

// DECL: namespace mlir {
// DECL: class AsmParser;
// DECL: class AsmPrinter;
// DECL: } // namespace mlir

// DEF: #ifdef GET_ATTRDEF_LIST
// DEF: #undef GET_ATTRDEF_LIST
// DEF: ::test::IndexAttr,
// DEF: ::test::SimpleAAttr,
// DEF: ::test::CompoundAAttr,
// DEF: ::test::SingleParameterAttr

// DEF-LABEL: ::mlir::OptionalParseResult generatedAttributeParser(
// DEF-SAME: ::mlir::AsmParser &parser,
// DEF-SAME: ::llvm::StringRef *mnemonic, ::mlir::Type type,
// DEF-SAME: ::mlir::Attribute &value) {
// DEF: return ::mlir::AsmParser::KeywordSwitch<::mlir::OptionalParseResult>(parser)
// DEF: .Case(::test::IndexAttr::getMnemonic()
// DEF-NEXT:   value = ::test::IndexAttr::parse(parser, type);
// DEF-NEXT:   return ::mlir::success(!!value);
// DEF: .Case(::test::CompoundAAttr::getMnemonic()
// DEF-NEXT: value = ::test::CompoundAAttr::parse(parser, type);
// DEF-NEXT: return ::mlir::success(!!value);
// DEF-NEXT: })
// DEF: .Default([&](llvm::StringRef keyword,
// DEF-NEXT:   *mnemonic = keyword;
// DEF-NEXT:   return std::nullopt;

def Test_Dialect: Dialect {
// DECL-NOT: TestDialect
// DEF-NOT: TestDialect
    let name = "TestDialect";
    let cppNamespace = "::test";
}

class TestAttr<string name> : AttrDef<Test_Dialect, name> { }

def C_IndexAttr : TestAttr<"Index"> {
    let mnemonic = "index";

    let parameters = (
      ins
      StringRefParameter<"Label for index">:$label
    );
  let hasCustomAssemblyFormat = 1;

// DECL-LABEL: class IndexAttr : public ::mlir::Attribute
// DECL: static constexpr ::llvm::StringLiteral getMnemonic() {
// DECL:   return {"index"};
// DECL: }
// DECL: static ::mlir::Attribute parse(
// DECL-SAME: ::mlir::AsmParser &odsParser, ::mlir::Type odsType);
// DECL: void print(::mlir::AsmPrinter &odsPrinter) const;
}

def A_SimpleAttrA : TestAttr<"SimpleA"> {
  let attrName = "test.simple";
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
      "::test::SimpleTypeA": $exampleTdType,
      APFloatParameter<"">: $apFloat,
      ArrayRefParameter<"int", "Matrix dimensions">:$dims,
      AttributeSelfTypeParameter<"">:$inner
  );

  let genVerifyDecl = 1;
  let hasCustomAssemblyFormat = 1;

// DECL-LABEL: class CompoundAAttr : public ::mlir::Attribute
// DECL: static CompoundAAttr getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, int widthOfSomething, ::test::SimpleTypeA exampleTdType, ::llvm::APFloat apFloat, ::llvm::ArrayRef<int> dims, ::mlir::Type inner);
// DECL: static ::mlir::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, int widthOfSomething, ::test::SimpleTypeA exampleTdType, ::llvm::APFloat apFloat, ::llvm::ArrayRef<int> dims, ::mlir::Type inner);
// DECL: static constexpr ::llvm::StringLiteral getMnemonic() {
// DECL:   return {"cmpnd_a"};
// DECL: }
// DECL: static ::mlir::Attribute parse(
// DECL-SAME: ::mlir::AsmParser &odsParser, ::mlir::Type odsType);
// DECL: void print(::mlir::AsmPrinter &odsPrinter) const;
// DECL: int getWidthOfSomething() const;
// DECL: ::test::SimpleTypeA getExampleTdType() const;
// DECL: ::llvm::APFloat getApFloat() const;
// DECL: ::mlir::Type getInner() const;

// Check that AttributeSelfTypeParameter is handled properly.
// DEF-LABEL: struct CompoundAAttrStorage
// DEF: CompoundAAttrStorage(
// DEF-SAME: inner(std::move(inner))

// DEF: bool operator==(const KeyTy &tblgenKey) const {
// DEF-NEXT: return
// DEF-SAME: (widthOfSomething == std::get<0>(tblgenKey)) &&
// DEF-SAME: (exampleTdType == std::get<1>(tblgenKey)) &&
// DEF-SAME: (apFloat.bitwiseIsEqual(std::get<2>(tblgenKey))) &&
// DEF-SAME: (dims == std::get<3>(tblgenKey)) &&
// DEF-SAME: (inner == std::get<4>(tblgenKey));

// DEF: static CompoundAAttrStorage *construct
// DEF: return new (allocator.allocate<CompoundAAttrStorage>())
// DEF-SAME: CompoundAAttrStorage(std::move(widthOfSomething), std::move(exampleTdType), std::move(apFloat), std::move(dims), std::move(inner));

// DEF: ::mlir::Type CompoundAAttr::getInner() const {
// DEF-NEXT: return getImpl()->inner;
}

def D_SingleParameterAttr : TestAttr<"SingleParameter"> {
  let parameters = (
    ins
    "int": $num
  );
  let attrName = "test.single_parameter";
// DECL-LABEL: struct SingleParameterAttrStorage;
// DECL-LABEL: class SingleParameterAttr
// DECL-SAME:  detail::SingleParameterAttrStorage
}

def F_ParamWithAccessorTypeAttr : TestAttr<"ParamWithAccessorType"> {
  let parameters = (ins AttrParameter<"std::string", "", "StringRef">:$param);
  let attrName = "test.param_with_accessor_type";
}

// DECL-LABEL: class ParamWithAccessorTypeAttr
// DECL: StringRef getParam()
// DEF: ParamWithAccessorTypeAttrStorage
// DEF: ParamWithAccessorTypeAttrStorage(std::string param)
// DEF: StringRef ParamWithAccessorTypeAttr::getParam()

def G_BuilderWithReturnTypeAttr : TestAttr<"BuilderWithReturnType"> {
  let parameters = (ins "int":$a);
  let genVerifyDecl = 1;
  let builders = [AttrBuilder<(ins), [{ return {}; }], "::mlir::Attribute">];
  let attrName = "test.builder_with_return_type";
}

// DECL-LABEL: class BuilderWithReturnTypeAttr
// DECL: ::mlir::Attribute get(
// DECL: ::mlir::Attribute getChecked(

def H_TestExtraClassAttr : TestAttr<"TestExtraClass"> {
  let extraClassDeclaration = [{
    /// Test Method
    static int getFoo(int i);
  }];
  let extraClassDefinition = [{
    int $cppClass::getFoo(int i) {
      return i+1;
    }
  }];
  let attrName = "test.test_extra_class";
}

// DECL-LABEL: TestExtraClassAttr : public ::mlir::Attribute
// DECL: /// Test Method
// DECL-NEXT: static int getFoo(int i);

// DEF-LABEL: int TestExtraClassAttr::getFoo(int i) {
// DEF: return i+1;
// DEF-NEXT: }
