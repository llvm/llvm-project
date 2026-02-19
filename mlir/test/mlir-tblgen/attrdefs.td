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
// DECL: static ::llvm::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, int widthOfSomething, ::test::SimpleTypeA exampleTdType, ::llvm::APFloat apFloat, ::llvm::ArrayRef<int> dims, ::mlir::Type inner);
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

// DEF: CompoundAAttr CompoundAAttr::getChecked(
// DEF-SAME:   int widthOfSomething, ::test::SimpleTypeA exampleTdType, ::llvm::APFloat apFloat, ::llvm::ArrayRef<int> dims, ::mlir::Type inner
// DEF-SAME: )
// DEF-NEXT: return Base::getChecked(emitError, context, std::move(widthOfSomething), std::move(exampleTdType), std::move(apFloat), std::move(dims), std::move(inner));

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

def I_TestGenMnemonicAliasAttr : TestAttr<"TestGenMnemonicAlias"> {
  let mnemonic = "test_gen_mnemonic_alias";
  let genMnemonicAlias = 1;
}

// DECL-LABEL: class TestGenMnemonicAliasAttr : public ::mlir::Attribute
// DECL-SAME: ::mlir::OpAsmAttrInterface::Trait
// DECL: ::mlir::OpAsmAliasResult getAlias(::llvm::raw_ostream &os) const;

// DEF-LABEL: ::mlir::OpAsmAliasResult TestGenMnemonicAliasAttr::getAlias(::llvm::raw_ostream &os) const {
// DEF-NEXT: os << "test_gen_mnemonic_alias";
// DEF-NEXT: return ::mlir::OpAsmAliasResult::OverridableAlias;
// DEF-NEXT: }

def J_CustomStorageCtorAttr : AttrDef<Test_Dialect, "CustomStorageCtorAttr"> {
  let attrName = "test_custom_storage_ctor_attr";
  let parameters = (ins "bool":$flag);
  let hasStorageCustomConstructor = 1;
}

// Note: ';' at the end of construct method declaration is important - otherwise
// one cannot provide custom definition

// DEF-LABEL: struct CustomStorageCtorAttrAttrStorage : public ::mlir::AttributeStorage
// DEF: static CustomStorageCtorAttrAttrStorage *construct
// DEF-SAME: (::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey);

// Test inheritable extra class declarations/definitions for attributes.

class InheritableTestAttr<string name> : AttrDef<Test_Dialect, name> {
  let inheritableExtraClassDeclaration = [{
    int getInheritedHelper();
  }];
  let inheritableExtraClassDefinition = [{
    int $cppClass::getInheritedHelper() { return 42; }
  }];
}

// Both inheritable and regular extra declarations should appear.
def K_InheritableAttrA : InheritableTestAttr<"InheritableA"> {
  let attrName = "test.inheritable_a";
  let extraClassDeclaration = [{
    void doA();
  }];
  let extraClassDefinition = [{
    void $cppClass::doA() {}
  }];
}

// DECL-LABEL: class InheritableAAttr
// DECL: int getInheritedHelper();
// DECL: void doA();

// DEF-LABEL: int InheritableAAttr::getInheritedHelper()
// DEF: return 42;
// DEF-LABEL: void InheritableAAttr::doA()

// Only inheritable declarations (no extraClassDeclaration).
def L_InheritableAttrB : InheritableTestAttr<"InheritableB"> {
  let attrName = "test.inheritable_b";
}

// DECL-LABEL: class InheritableBAttr
// DECL: int getInheritedHelper();

// Discard inheritable declarations by setting to empty.
def M_InheritableAttrC : InheritableTestAttr<"InheritableC"> {
  let attrName = "test.inheritable_c";
  let inheritableExtraClassDeclaration = [{}];
  let inheritableExtraClassDefinition = [{}];
}

// DECL-LABEL: class InheritableCAttr
// DECL-NOT: int getInheritedHelper();

// Middle-of-stack: accumulates with base inheritable declarations.
class InheritableMiddleAttr<string name> : InheritableTestAttr<name> {
  let inheritableExtraClassDeclaration = [{
    int getMiddleHelper();
  }];
  let inheritableExtraClassDefinition = [{
    int $cppClass::getMiddleHelper() { return 1; }
  }];
}

// Concrete attr inheriting from middle gets both base and middle.
def N_InheritableAttrD : InheritableMiddleAttr<"InheritableD"> {
  let attrName = "test.inheritable_d";
}

// DECL-LABEL: class InheritableDAttr
// DECL: int getInheritedHelper();
// DECL: int getMiddleHelper();

// DEF-LABEL: int InheritableDAttr::getInheritedHelper()
// DEF: return 42;
// DEF-LABEL: int InheritableDAttr::getMiddleHelper()
// DEF: return 1;

// Passthrough: middle class doesn't set inheritable, base value passes through.
class InheritablePassthroughAttr<string name> : InheritableTestAttr<name> {}

def O_InheritableAttrE : InheritablePassthroughAttr<"InheritableE"> {
  let attrName = "test.inheritable_e";
}

// DECL-LABEL: class InheritableEAttr
// DECL: int getInheritedHelper();
