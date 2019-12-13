// RUN: mlir-tblgen -gen-op-interface-decls -I %S/../../include %s | FileCheck %s --check-prefix=DECL --dump-input-on-failure

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

// DECL-LABEL: TestOpInterfaceInterfaceTraits
// DECL: class TestOpInterface : public OpInterface<TestOpInterface, detail::TestOpInterfaceInterfaceTraits>
// DECL: int foo(int input);

// DECL-NOT: TestOpInterface
