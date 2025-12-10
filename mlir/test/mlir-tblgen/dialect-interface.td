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
// DECL-NEXT:   NoDefaultMethod(::mlir::Dialect *dialect) : Base(dialect) {}
// DECL:   virtual bool isExampleDialect() const {}
// DECL:   virtual unsigned supportSecondMethod(::mlir::Type type) const {}

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

// DECL:  virtual bool isExampleDialect() const {
// DECL-NEXT:  return true;
// DECL-NEXT: }

