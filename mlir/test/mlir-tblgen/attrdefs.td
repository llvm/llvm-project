// RUN: mlir-tblgen -gen-attrdef-decls -I %S/../../include %s | FileCheck %s --check-prefix=DECL
// RUN: mlir-tblgen -gen-attrdef-defs -I %S/../../include %s | FileCheck %s --check-prefix=DEF

include "mlir/IR/OpBase.td"

// DECL: #ifdef GET_ATTRDEF_CLASSES
// DECL: #undef GET_ATTRDEF_CLASSES

// DECL: namespace mlir {
// DECL: class AsmParser;
// DECL: class AsmPrinter;
// DECL: } // namespace mlir

// DEF: #ifdef GET_ATTRDEF_LIST
// DEF: #undef GET_ATTRDEF_LIST
// DEF: ::test::SimpleAAttr,
// DEF: ::test::CompoundAAttr,
// DEF: ::test::IndexAttr,
// DEF: ::test::SingleParameterAttr

// DEF-LABEL: ::mlir::OptionalParseResult generatedAttributeParser(
// DEF-SAME: ::mlir::AsmParser &parser,
// DEF-SAME: ::llvm::StringRef mnemonic, ::mlir::Type type,
// DEF-SAME: ::mlir::Attribute &value) {
// DEF: if (mnemonic == ::test::CompoundAAttr::getMnemonic()) {
// DEF-NEXT: value = ::test::CompoundAAttr::parse(parser, type);
// DEF-NEXT: return ::mlir::success(!!value);
// DEF-NEXT: }
// DEF-NEXT: if (mnemonic == ::test::IndexAttr::getMnemonic()) {
// DEF-NEXT:   value = ::test::IndexAttr::parse(parser, type);
// DEF-NEXT:   return ::mlir::success(!!value);
// DEF: return {};

def Test_Dialect: Dialect {
// DECL-NOT: TestDialect
// DEF-NOT: TestDialect
    let name = "TestDialect";
    let cppNamespace = "::test";
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
      "::test::SimpleTypeA": $exampleTdType,
      APFloatParameter<"">: $apFloat,
      ArrayRefParameter<"int", "Matrix dimensions">:$dims,
      AttributeSelfTypeParameter<"">:$inner
  );

  let genVerifyDecl = 1;

// DECL-LABEL: class CompoundAAttr : public ::mlir::Attribute
// DECL: static CompoundAAttr getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, int widthOfSomething, ::test::SimpleTypeA exampleTdType, ::llvm::APFloat apFloat, ::llvm::ArrayRef<int> dims, ::mlir::Type inner);
// DECL: static ::mlir::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, int widthOfSomething, ::test::SimpleTypeA exampleTdType, ::llvm::APFloat apFloat, ::llvm::ArrayRef<int> dims, ::mlir::Type inner);
// DECL: static constexpr ::llvm::StringLiteral getMnemonic() {
// DECL:   return {"cmpnd_a"};
// DECL: }
// DECL: static ::mlir::Attribute parse(
// DECL-SAME: ::mlir::AsmParser &parser, ::mlir::Type type);
// DECL: void print(::mlir::AsmPrinter &printer) const;
// DECL: int getWidthOfSomething() const;
// DECL: ::test::SimpleTypeA getExampleTdType() const;
// DECL: ::llvm::APFloat getApFloat() const;

// Check that AttributeSelfTypeParameter is handled properly.
// DEF-LABEL: struct CompoundAAttrStorage
// DEF: CompoundAAttrStorage(
// DEF-SAME: : ::mlir::AttributeStorage(inner),

// DEF: bool operator==(const KeyTy &tblgenKey) const {
// DEF-NEXT: return
// DEF-SAME: (widthOfSomething == std::get<0>(tblgenKey)) &&
// DEF-SAME: (exampleTdType == std::get<1>(tblgenKey)) &&
// DEF-SAME: (apFloat.bitwiseIsEqual(std::get<2>(tblgenKey))) &&
// DEF-SAME: (dims == std::get<3>(tblgenKey)) &&
// DEF-SAME: (getType() == std::get<4>(tblgenKey));

// DEF: static CompoundAAttrStorage *construct
// DEF: return new (allocator.allocate<CompoundAAttrStorage>())
// DEF-SAME: CompoundAAttrStorage(widthOfSomething, exampleTdType, apFloat, dims, inner);

// DEF: ::mlir::Type CompoundAAttr::getInner() const {
// DEF-NEXT: return getImpl()->getType().cast<::mlir::Type>();
}

def C_IndexAttr : TestAttr<"Index"> {
    let mnemonic = "index";

    let parameters = (
      ins
      StringRefParameter<"Label for index">:$label
    );

// DECL-LABEL: class IndexAttr : public ::mlir::Attribute
// DECL: static constexpr ::llvm::StringLiteral getMnemonic() {
// DECL:   return {"index"};
// DECL: }
// DECL: static ::mlir::Attribute parse(
// DECL-SAME: ::mlir::AsmParser &parser, ::mlir::Type type);
// DECL: void print(::mlir::AsmPrinter &printer) const;
}

def D_SingleParameterAttr : TestAttr<"SingleParameter"> {
  let parameters = (
    ins
    "int": $num
  );
// DECL-LABEL: struct SingleParameterAttrStorage;
// DECL-LABEL: class SingleParameterAttr
// DECL-SAME:  detail::SingleParameterAttrStorage
}

// An attribute testing AttributeSelfTypeParameter.
def E_AttrWithTypeBuilder : TestAttr<"AttrWithTypeBuilder"> {
  let mnemonic = "attr_with_type_builder";
  let parameters = (ins "::mlir::IntegerAttr":$attr);
  let typeBuilder = "$_attr.getType()";
}

// DEF-LABEL: struct AttrWithTypeBuilderAttrStorage
// DEF: AttrWithTypeBuilderAttrStorage(::mlir::IntegerAttr attr)
// DEF-SAME: : ::mlir::AttributeStorage(attr.getType()), attr(attr)

def F_ParamWithAccessorTypeAttr : TestAttr<"ParamWithAccessorType"> {
  let parameters = (ins AttrParameter<"std::string", "", "StringRef">:$param);
}

// DECL-LABEL: class ParamWithAccessorTypeAttr
// DECL: StringRef getParam()
// DEF: ParamWithAccessorTypeAttrStorage
// DEF: ParamWithAccessorTypeAttrStorage(std::string param)
// DEF: StringRef ParamWithAccessorTypeAttr::getParam()
