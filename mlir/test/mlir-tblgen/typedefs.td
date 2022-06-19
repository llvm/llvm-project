// RUN: mlir-tblgen -gen-typedef-decls -I %S/../../include %s | FileCheck %s --check-prefix=DECL
// RUN: mlir-tblgen -gen-typedef-defs -I %S/../../include %s | FileCheck %s --check-prefix=DEF

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

// DECL: #ifdef GET_TYPEDEF_CLASSES
// DECL: #undef GET_TYPEDEF_CLASSES

// DECL: namespace mlir {
// DECL: class AsmParser;
// DECL: class AsmPrinter;
// DECL: } // namespace mlir

// DEF: #ifdef GET_TYPEDEF_LIST
// DEF: #undef GET_TYPEDEF_LIST
// DEF: ::test::SimpleAType,
// DEF: ::test::CompoundAType,
// DEF: ::test::IndexType,
// DEF: ::test::SingleParameterType,
// DEF: ::test::IntegerType

// DEF-LABEL: ::mlir::OptionalParseResult generatedTypeParser(
// DEF-SAME: ::mlir::AsmParser &parser,
// DEF-SAME: ::llvm::StringRef mnemonic,
// DEF-SAME: ::mlir::Type &value) {
// DEF: if (mnemonic == ::test::CompoundAType::getMnemonic()) {
// DEF-NEXT:   value = ::test::CompoundAType::parse(parser);
// DEF-NEXT:   return ::mlir::success(!!value);
// DEF-NEXT: }
// DEF-NEXT: if (mnemonic == ::test::IndexType::getMnemonic()) {
// DEF-NEXT:   value = ::test::IndexType::parse(parser);
// DEF-NEXT:   return ::mlir::success(!!value);
// DEF: return {};

def Test_Dialect: Dialect {
// DECL-NOT: TestDialect
  let name = "TestDialect";
  let cppNamespace = "::test";
}

class TestType<string name> : TypeDef<Test_Dialect, name> { }

def A_SimpleTypeA : TestType<"SimpleA"> {
// DECL: class SimpleAType : public ::mlir::Type
}

def RTLValueType : Type<CPred<"isRTLValueType($_self)">, "Type"> {
  string cppType = "::mlir::Type";
}

// A more complex parameterized type
def B_CompoundTypeA : TestType<"CompoundA"> {
  let summary = "A more complex parameterized type";
  let description = "This type is to test a reasonably complex type";
  let mnemonic = "cmpnd_a";
  let parameters = (ins
    "int":$widthOfSomething,
    "::test::SimpleTypeA": $exampleTdType,
    "SomeCppStruct": $exampleCppType,
    ArrayRefParameter<"int", "Matrix dimensions">:$dims,
    RTLValueType:$inner
  );

  let genVerifyDecl = 1;
  let hasCustomAssemblyFormat = 1;

// DECL-LABEL: class CompoundAType : public ::mlir::Type
// DECL: static CompoundAType getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, int widthOfSomething, ::test::SimpleTypeA exampleTdType, SomeCppStruct exampleCppType, ::llvm::ArrayRef<int> dims, ::mlir::Type inner);
// DECL: static ::mlir::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, int widthOfSomething, ::test::SimpleTypeA exampleTdType, SomeCppStruct exampleCppType, ::llvm::ArrayRef<int> dims, ::mlir::Type inner);
// DECL: static constexpr ::llvm::StringLiteral getMnemonic() {
// DECL:   return {"cmpnd_a"};
// DECL: }
// DECL: static ::mlir::Type parse(::mlir::AsmParser &odsParser);
// DECL: void print(::mlir::AsmPrinter &odsPrinter) const;
// DECL: int getWidthOfSomething() const;
// DECL: ::test::SimpleTypeA getExampleTdType() const;
// DECL: SomeCppStruct getExampleCppType() const;
}

def C_IndexType : TestType<"Index"> {
  let mnemonic = "index";

  let parameters = (ins
    StringRefParameter<"Label for index">:$label
  );
  let hasCustomAssemblyFormat = 1;

// DECL-LABEL: class IndexType : public ::mlir::Type
// DECL: static constexpr ::llvm::StringLiteral getMnemonic() {
// DECL:   return {"index"};
// DECL: }
// DECL: static ::mlir::Type parse(::mlir::AsmParser &odsParser);
// DECL: void print(::mlir::AsmPrinter &odsPrinter) const;
}

def D_SingleParameterType : TestType<"SingleParameter"> {
  let parameters = (ins
    "int": $num
  );
// DECL-LABEL: struct SingleParameterTypeStorage;
// DECL-LABEL: class SingleParameterType
// DECL-SAME:  detail::SingleParameterTypeStorage
}

def E_IntegerType : TestType<"Integer"> {
  let mnemonic = "int";
  let genVerifyDecl = 1;
  let hasCustomAssemblyFormat = 1;
  let parameters = (ins
      "SignednessSemantics":$signedness,
      TypeParameter<"unsigned", "Bitwidth of integer">:$width
  );

// DECL-LABEL: IntegerType : public ::mlir::Type

  let extraClassDeclaration = [{
  /// Signedness semantics.
  enum SignednessSemantics {
    Signless, /// No signedness semantics
    Signed,   /// Signed integer
    Unsigned, /// Unsigned integer
  };

  /// This extra function is necessary since it doesn't include signedness
  static IntegerType getChecked(unsigned width, Location location);

  /// Return true if this is a signless integer type.
  bool isSignless() const { return getSignedness() == Signless; }
  /// Return true if this is a signed integer type.
  bool isSigned() const { return getSignedness() == Signed; }
  /// Return true if this is an unsigned integer type.
  bool isUnsigned() const { return getSignedness() == Unsigned; }
  }];

// DECL: /// Signedness semantics.
// DECL-NEXT: enum SignednessSemantics {
// DECL-NEXT:   Signless, /// No signedness semantics
// DECL-NEXT:   Signed,   /// Signed integer
// DECL-NEXT:   Unsigned, /// Unsigned integer
// DECL-NEXT: };
// DECL: /// This extra function is necessary since it doesn't include signedness
// DECL-NEXT: static IntegerType getChecked(unsigned width, Location location);

// DECL: /// Return true if this is a signless integer type.
// DECL-NEXT: bool isSignless() const { return getSignedness() == Signless; }
// DECL-NEXT: /// Return true if this is a signed integer type.
// DECL-NEXT: bool isSigned() const { return getSignedness() == Signed; }
// DECL-NEXT: /// Return true if this is an unsigned integer type.
// DECL-NEXT: bool isUnsigned() const { return getSignedness() == Unsigned; }
}
