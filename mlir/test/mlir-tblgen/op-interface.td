// RUN: mlir-tblgen -gen-op-interface-decls -I %S/../../include %s | FileCheck %s --check-prefix=DECL
// RUN: mlir-tblgen -gen-op-decls -I %S/../../include %s | FileCheck %s --check-prefix=OP_DECL

include "mlir/IR/OpBase.td"

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
// DECL: class TestOpInterface : public OpInterface<TestOpInterface, detail::TestOpInterfaceInterfaceTraits>
// DECL: int foo(int input);

// DECL-NOT: TestOpInterface

// OP_DECL-LABEL: class DeclareMethodsOp : public
// OP_DECL: int foo(int input);
// OP_DECL-NOT: int default_foo(int input);

// OP_DECL-LABEL: class DeclareMethodsWithDefaultOp : public
// OP_DECL: int foo(int input);
// OP_DECL: int default_foo(int input);
