// RUN: mlir-tblgen -gen-dialect-interface-decls -I %S/../../include %s | FileCheck %s --check-prefix=DECL

include "mlir/IR/Interfaces.td"

def NoDefaultMethod : DialectInterface<"NoDefaultMethod"> {
  let description = [{
    This is an example dialect interface without default method body.
  }];

  let cppNamespace = "::mlir::example";

  let methods = [
    InterfaceMethod<
      "Check if it's an example dialect", "bool", "isExampleDialect", (ins)
    >,
    InterfaceMethod<
      "second method to check if multiple methods supported",
      "unsigned", "supportSecondMethod", (ins "::mlir::Type":$type)
      >
  ];
}

// DECL:   class NoDefaultMethod : public {{.*}}DialectInterface::Base<NoDefaultMethod>
// DECL:   public:
/// DECL:   NoDefaultMethod(::mlir::Dialect *dialect) : Base(dialect) {}
// DECL:   virtual bool isExampleDialect() const {}
// DECL:   virtual unsigned supportSecondMethod(::mlir::Type type) const {}

def WithDefaultMethodInterface : DialectInterface<"WithDefaultMethodInterface"> {
  let description = [{
    This is an example dialect interface with default method bodies.
  }];

  let cppNamespace = "::mlir::example";

  let methods = [
      InterfaceMethod<
        "Check if it's an example dialect", "bool", "isExampleDialect", (ins),
        [{
          return true;
        }]
      >,
      InterfaceMethod<
        "second method to check if multiple methods supported",
        "unsigned", "supportSecondMethod", (ins "::mlir::Type":$type)
      >
  ];
}

// DECL:   virtual bool isExampleDialect() const {
// DECL-NEXT:   return true;
// DECL-NEXT:   }

def OnlyDeclarationInterface : DialectInterface<"OnlyDeclarationInterface"> {
  let description = [{
    This is an example dialect interface with only declarations.
  }];

  let cppNamespace = "::mlir::example";

  let aliasDeclarations = [
      AliasDeclaration<"FirstAlias", "unsigned">];

  let methods = [
      InterfaceMethodDeclaration<
        "a method declaration", "FirstAlias", "exampleMethodDeclaration", 
        (ins "::mlir::Type":$type)
      >
  ];
}

// DECL:   class OnlyDeclarationInterface : public {{.*}}DialectInterface::Base<OnlyDeclarationInterface>
// DECL:   using FirstAlias = unsigned;
// DECL:   virtual FirstAlias exampleMethodDeclaration(::mlir::Type type) const;
