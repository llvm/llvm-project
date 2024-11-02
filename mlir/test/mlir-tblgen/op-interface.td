// RUN: mlir-tblgen -gen-op-interface-decls -I %S/../../include %s | FileCheck %s --check-prefix=DECL
// RUN: mlir-tblgen -gen-op-decls -I %S/../../include %s | FileCheck %s --check-prefix=OP_DECL

include "mlir/IR/OpBase.td"

def ExtraShardDeclsInterface : OpInterface<"ExtraShardDeclsInterface"> {
  let extraSharedClassDeclaration = [{
    bool sharedMethodDeclaration() {
      return $_op.someOtherMethod();
    }
  }];
}

// DECL: class ExtraShardDeclsInterface
// DECL:      bool sharedMethodDeclaration() {
// DECL-NEXT:   return (*this).someOtherMethod();
// DECL-NEXT: }

// DECL: struct ExtraShardDeclsInterfaceTrait
// DECL:      bool sharedMethodDeclaration() {
// DECL-NEXT:   return (*static_cast<ConcreteOp *>(this)).someOtherMethod();
// DECL-NEXT: }

def TestOpInterface : OpInterface<"TestOpInterface"> {
  let description = [{some op interface description}];

  let methods = [
    InterfaceMethod<
      /*desc=*/[{some function comment}],
      /*retTy=*/"int",
      /*methodName=*/"foo",
      /*args=*/(ins "int":$input)
    >,
    InterfaceMethod<
      /*desc=*/[{some function comment}],
      /*retTy=*/"int",
      /*methodName=*/"default_foo",
      /*args=*/(ins "int":$input),
      /*body=*/[{}],
      /*defaultBody=*/[{ return 0; }]
    >,
  ];
}

def TestOpInterfaceVerify : OpInterface<"TestOpInterfaceVerify"> {
  let verify = [{
    return foo();
  }];
}

def TestOpInterfaceVerifyRegion : OpInterface<"TestOpInterfaceVerifyRegion"> {
  let verify = [{
    return foo();
  }];
  let verifyWithRegions = 1;
}

// Define Ops with TestOpInterface and
// DeclareOpInterfaceMethods<TestOpInterface> traits to check that there
// are not duplicated C++ classes generated.
def TestDialect : Dialect {
  let name = "test";
}

def OpInterfaceOp : Op<TestDialect, "op_interface_op", [TestOpInterface]>;

def DeclareMethodsOp : Op<TestDialect, "declare_methods_op",
                          [DeclareOpInterfaceMethods<TestOpInterface>]>;

def DeclareMethodsWithDefaultOp : Op<TestDialect, "declare_methods_op",
      [DeclareOpInterfaceMethods<TestOpInterface, ["default_foo"]>]>;

// DECL-LABEL: TestOpInterfaceInterfaceTraits
// DECL: class TestOpInterface : public ::mlir::OpInterface<TestOpInterface, detail::TestOpInterfaceInterfaceTraits>

// DECL: /// some function comment
// DECL: int foo(int input);

// DECL-LABEL: struct TestOpInterfaceVerifyTrait
// DECL: verifyTrait

// DECL-LABEL: struct TestOpInterfaceVerifyRegionTrait
// DECL: verifyRegionTrait

// Method implementations come last, after all class definitions.
// DECL: template<typename ConcreteOp>
// DECL: int detail::TestOpInterfaceInterfaceTraits::Model<ConcreteOp>::foo

// OP_DECL-LABEL: class DeclareMethodsOp : public
// OP_DECL: int foo(int input);
// OP_DECL-NOT: int default_foo(int input);

// OP_DECL-LABEL: class DeclareMethodsWithDefaultOp : public
// OP_DECL: int foo(int input);
// OP_DECL: int default_foo(int input);
