// RUN: mlir-tblgen --gen-type-interface-decls -I %S/../../include %s | FileCheck %s

include "mlir/IR/OpBase.td"

def TestEmptyMethodBodyTypeInterface : TypeInterface<"TestEmptyMethodBodyTypeInterface"> {
  let cppNamespace = "::TestEmptyMethodBodyTypeInterface";
  let description = [{
    Treat method body with only spaces as empty.
  }];
  let methods = [
    InterfaceMethod<
      /*desc=*/        [{ Trim spaces of method body and default implementation }],
      /*returnType=*/  "StringRef",
      /*methodName=*/  "trimSpaces",
      /*args=*/        (ins),
      // CHECK-LABEL: StringRef detail::TestEmptyMethodBodyTypeInterfaceInterfaceTraits::Model<ConcreteType>::trimSpaces
      // CHECK-SAME: {
      // CHECK-NEXT: return (::llvm::cast<ConcreteType>(tablegen_opaque_val)).trimSpaces();
      // CHECK-NEXT: }
      /*methodBody=*/  [{  }],
      /*defaultImpl=*/ [{ return StringRef(); }]
    >
  ];
}

def TestEmptyDefaultImplTypeInterface : TypeInterface<"TestEmptyDefaultImplTypeInterface"> {
  let cppNamespace = "::TestEmptyDefaultImplTypeInterface";
  let description = [{
    Treat default implementation with only spaces as empty.
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/        [{ Trim spaces of default implementation }],
      /*returnType=*/  "StringRef",
      /*methodName=*/  "trimSpaces",
      /*args=*/        (ins),
      /*methodBody=*/  [{ return StringRef(); }],
      // COM: Don't generate default implementation
      // CHECK-NOT: StringRef detail::TestEmptyDefaultImplTypeInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteType>::trimSpaces
      /*defaultImpl=*/ [{
      }]
    >
  ];
}
