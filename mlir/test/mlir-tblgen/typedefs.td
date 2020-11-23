// RUN: mlir-tblgen -gen-typedef-decls -I %S/../../include %s | FileCheck %s --check-prefix=DECL
// RUN: mlir-tblgen -gen-typedef-defs -I %S/../../include %s | FileCheck %s --check-prefix=DEF

include "mlir/IR/OpBase.td"

// DECL: #ifdef GET_TYPEDEF_CLASSES
// DECL: #undef GET_TYPEDEF_CLASSES

// DECL: namespace mlir {
// DECL: class DialectAsmParser;
// DECL: class DialectAsmPrinter;
// DECL: } // namespace mlir

// DECL: ::mlir::Type generatedTypeParser(::mlir::MLIRContext* ctxt, ::mlir::DialectAsmParser& parser, ::llvm::StringRef mnenomic);
// DECL: ::mlir::LogicalResult generatedTypePrinter(::mlir::Type type, ::mlir::DialectAsmPrinter& printer);

// DEF: #ifdef GET_TYPEDEF_LIST
// DEF: #undef GET_TYPEDEF_LIST
// DEF: ::mlir::test::SimpleAType,
// DEF: ::mlir::test::CompoundAType,
// DEF: ::mlir::test::IndexType,
// DEF: ::mlir::test::SingleParameterType,
// DEF: ::mlir::test::IntegerType

// DEF-LABEL: ::mlir::Type generatedTypeParser(::mlir::MLIRContext* ctxt, ::mlir::DialectAsmParser& parser, ::llvm::StringRef mnemonic)
// DEF: if (mnemonic == ::mlir::test::CompoundAType::getMnemonic()) return ::mlir::test::CompoundAType::parse(ctxt, parser);
// DEF return ::mlir::Type();

def Test_Dialect: Dialect {
// DECL-NOT: TestDialect
// DEF-NOT: TestDialect
    let name = "TestDialect";
    let cppNamespace = "::mlir::test";
}

class TestType<string name> : TypeDef<Test_Dialect, name> { }

def A_SimpleTypeA : TestType<"SimpleA"> {
// DECL: class SimpleAType: public ::mlir::Type
}

def RTLValueType : Type<CPred<"isRTLValueType($_self)">, "Type"> {
  string cppType = "::mlir::Type";
}

// A more complex parameterized type
def B_CompoundTypeA : TestType<"CompoundA"> {
  let summary = "A more complex parameterized type";
  let description = "This type is to test a reasonably complex type";
  let mnemonic = "cmpnd_a";
  let parameters = (
      ins
      "int":$widthOfSomething,
      "::mlir::test::SimpleTypeA": $exampleTdType,
      "SomeCppStruct": $exampleCppType,
      ArrayRefParameter<"int", "Matrix dimensions">:$dims,
      RTLValueType:$inner
  );

  let genVerifyInvariantsDecl = 1;

// DECL-LABEL: class CompoundAType: public ::mlir::Type
// DECL: static ::mlir::LogicalResult verifyConstructionInvariants(::mlir::Location loc, int widthOfSomething, ::mlir::test::SimpleTypeA exampleTdType, SomeCppStruct exampleCppType, ::llvm::ArrayRef<int> dims, ::mlir::Type inner);
// DECL: static ::mlir::Type getChecked(::mlir::Location loc, int widthOfSomething, ::mlir::test::SimpleTypeA exampleTdType, SomeCppStruct exampleCppType, ::llvm::ArrayRef<int> dims, ::mlir::Type inner);
// DECL: static ::llvm::StringRef getMnemonic() { return "cmpnd_a"; }
// DECL: static ::mlir::Type parse(::mlir::MLIRContext* ctxt, ::mlir::DialectAsmParser& parser);
// DECL: void print(::mlir::DialectAsmPrinter& printer) const;
// DECL: int getWidthOfSomething() const;
// DECL: ::mlir::test::SimpleTypeA getExampleTdType() const;
// DECL: SomeCppStruct getExampleCppType() const;
}

def C_IndexType : TestType<"Index"> {
    let mnemonic = "index";

    let parameters = (
      ins
      StringRefParameter<"Label for index">:$label
    );

// DECL-LABEL: class IndexType: public ::mlir::Type
// DECL: static ::llvm::StringRef getMnemonic() { return "index"; }
// DECL: static ::mlir::Type parse(::mlir::MLIRContext* ctxt, ::mlir::DialectAsmParser& parser);
// DECL: void print(::mlir::DialectAsmPrinter& printer) const;
}

def D_SingleParameterType : TestType<"SingleParameter"> {
  let parameters = (
    ins
    "int": $num
  );
// DECL-LABEL: struct SingleParameterTypeStorage;
// DECL-LABEL: class SingleParameterType
// DECL-NEXT:                   detail::SingleParameterTypeStorage
}

def E_IntegerType : TestType<"Integer"> {
    let mnemonic = "int";
    let genVerifyInvariantsDecl = 1;
    let parameters = (
        ins
        "SignednessSemantics":$signedness,
        TypeParameter<"unsigned", "Bitwidth of integer">:$width
    );

// DECL-LABEL: IntegerType: public ::mlir::Type

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
