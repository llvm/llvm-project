// RUN: mlir-tblgen -gen-dialect-interface-decls -I %S/../../include %s | FileCheck %s --check-prefix=DECL

include "mlir/IR/Interfaces.td"

def NoDefaultMethod : DialectInterface<"NoDefaultMethod"> {
  let description = [{
    This is an example dialect interface without default method body.
  }];

  let cppNamespace = "::mlir::example";

  let methods = [
      InterfaceMethod<
      /*desc=*/        "Check if it's an example dialect",
      /*returnType=*/  "bool",
      /*methodName=*/  "isExampleDialect",
      /*args=*/        (ins)
      >,
      InterfaceMethod<
      /*desc=*/        "second method to check if multiple methods supported",
      /*returnType=*/  "unsigned",
      /*methodName=*/  "supportSecondMethod",
      /*args=*/        (ins "::mlir::Type":$type)
      >

  ];
}

// DECL:   class NoDefaultMethod : public {{.*}}DialectInterface::Base<NoDefaultMethod>
// DECL:   public:
// DECL:   virtual bool isExampleDialect() const {}
// DECL:   virtual unsigned supportSecondMethod(::mlir::Type type) const {}
// DECL:   NoDefaultMethod(::mlir::Dialect *dialect) : Base(dialect) {}

def WithDefaultMethodInterface : DialectInterface<"WithDefaultMethodInterface"> {
  let description = [{
    This is an example dialect interface with default method bodies.
  }];

  let cppNamespace = "::mlir::example";

  let methods = [
      InterfaceMethod<
      /*desc=*/        "Check if it's an example dialect",
      /*returnType=*/  "bool",
      /*methodName=*/  "isExampleDialect",
      /*args=*/        (ins),
      /*methodBody=*/  [{
          return true;
         }]
      >,
      InterfaceMethod<
      /*desc=*/        "second method to check if multiple methods supported",
      /*returnType=*/  "unsigned",
      /*methodName=*/  "supportSecondMethod",
      /*args=*/        (ins "::mlir::Type":$type)
      >

  ];
}

// DECL:   virtual bool isExampleDialect() const {
// DECL-NEXT:   return true;
// DECL-NEXT:   }

def PureVirtualInterface : DialectInterface<"PureVirtualInterface"> {
  let description = [{
    This is an example dialect interface with pure virtual methods.
  }];

  let cppNamespace = "::mlir::example";

  let aliasDeclarations = [
      AliasDeclaration<"FirstAlias", "unsigned">];

  let methods = [
      PureVirtualInterfaceMethod<
      /*desc=*/        "Check if it's an example dialect",
      /*returnType=*/  "bool",
      /*methodName=*/  "isExampleDialect",
      /*args=*/        (ins)
      >,
      InterfaceMethod<
      /*desc=*/        "second method to check if multiple methods supported",
      /*returnType=*/  "FirstAlias",
      /*methodName=*/  "supportSecondMethod",
      /*args=*/        (ins "::mlir::Type":$type)
      >,
      InterfaceMethodDeclaration<
      /*desc=*/        "a method declaration",
      /*returnType=*/  "::mlir::Type",
      /*methodName=*/  "exampleMethodDeclaration",
      /*args=*/        (ins "::mlir::Type":$type)
      >
  ];
}

// DECL:   class PureVirtualInterface : public {{.*}}DialectInterface::Base<PureVirtualInterface>
// DECL:   public:
// DECL:   using FirstAlias = unsigned;
// DECL:   virtual bool isExampleDialect() const = 0;
// DECL:   virtual FirstAlias supportSecondMethod(::mlir::Type type) const {}
// DECL:   virtual ::mlir::Type exampleMethodDeclaration(::mlir::Type type) const;
// DECL:   protected:
// DECL-NEXT:   PureVirtualInterface(::mlir::Dialect *dialect) : Base(dialect) {}

