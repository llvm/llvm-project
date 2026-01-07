// RUN: mlir-tblgen -gen-op-interface-decls -I %S/../../include %s | FileCheck %s --check-prefix=DECL
// RUN: mlir-tblgen -gen-op-interface-defs -I %S/../../include %s | FileCheck %s --check-prefix=DEF
// RUN: mlir-tblgen -gen-op-decls -I %S/../../include %s | FileCheck %s --check-prefix=OP_DECL
// RUN: mlir-tblgen -gen-op-interface-docs -I %S/../../include %s | FileCheck %s --check-prefix=DOCS

include "mlir/IR/OpBase.td"

def ExtraClassOfInterface : OpInterface<"ExtraClassOfInterface"> {
  let extraClassOf = "return $_op->someOtherMethod();";
}

// DECL: class ExtraClassOfInterface
// DECL:   static bool classof(::mlir::Operation * base) {
// DECL-NEXT:     auto* interface = getInterfaceFor(base);
// DECL-NEXT:     if (!interface)
// DECL-NEXT:       return false;
// DECL-NEXT:     ExtraClassOfInterface odsInterfaceInstance(base, interface);
// DECL-NEXT:     return odsInterfaceInstance->someOtherMethod();
// DECL-NEXT:   }

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

def TestInheritanceMultiBaseInterface : OpInterface<"TestInheritanceMultiBaseInterface"> {
  let methods = [
    InterfaceMethod<
      /*desc=*/[{some function comment}],
      /*retTy=*/"int",
      /*methodName=*/"baz",
      /*args=*/(ins "int":$input)
    >
  ];
}

def TestInheritanceBaseInterface : OpInterface<"TestInheritanceBaseInterface", [TestInheritanceMultiBaseInterface]> {
  let methods = [
    InterfaceMethod<
      /*desc=*/[{some function comment}],
      /*retTy=*/"int",
      /*methodName=*/"foo",
      /*args=*/(ins "int":$input)
    >
  ];
}
def TestInheritanceMiddleBaseInterface
 : OpInterface<"TestInheritanceMiddleBaseInterface", [TestInheritanceBaseInterface]> {
  let methods = [
    InterfaceMethod<
      /*desc=*/[{some function comment}],
      /*retTy=*/"int",
      /*methodName=*/"bar",
      /*args=*/(ins "int":$input)
    >
  ];
}
def TestInheritanceZDerivedInterface
  : OpInterface<"TestInheritanceZDerivedInterface", [TestInheritanceMiddleBaseInterface]>;

// DECL: struct TestInheritanceZDerivedInterfaceInterfaceTraits
// DECL: struct Concept {
// DECL:     const TestInheritanceMultiBaseInterface::Concept *implTestInheritanceMultiBaseInterface = nullptr;
// DECL-NOT: const TestInheritanceMultiBaseInterface::Concept
// DECL:     const TestInheritanceBaseInterface::Concept *implTestInheritanceBaseInterface = nullptr;
// DECL:     const TestInheritanceMiddleBaseInterface::Concept *implTestInheritanceMiddleBaseInterface = nullptr;

// DECL:     void initializeInterfaceConcept(::mlir::detail::InterfaceMap &interfaceMap) {
// DECL:       implTestInheritanceBaseInterface = interfaceMap.lookup<TestInheritanceBaseInterface>();
// DECL:       assert(implTestInheritanceBaseInterface && "`TestInheritanceZDerivedInterface` expected its base interface `TestInheritanceBaseInterface` to be registered");
// DECL:       implTestInheritanceMiddleBaseInterface = interfaceMap.lookup<TestInheritanceMiddleBaseInterface>();
// DECL:       assert(implTestInheritanceMiddleBaseInterface
// DECL:     }

// DECL:    //===----------------------------------------------------------------===//
// DECL:    // Inherited from TestInheritanceBaseInterface
// DECL:    //===----------------------------------------------------------------===//
// DECL:    operator TestInheritanceBaseInterface () const {
// DECL:      return TestInheritanceBaseInterface(*this, getImpl()->implTestInheritanceBaseInterface);
// DECL:    }
// DECL:    /// some function comment
// DECL:    int foo(int input);

// DECL:    //===----------------------------------------------------------------===//
// DECL:    // Inherited from TestInheritanceMiddleBaseInterface
// DECL:    //===----------------------------------------------------------------===//
// DECL:    operator TestInheritanceMiddleBaseInterface () const {
// DECL:      return TestInheritanceMiddleBaseInterface(*this, getImpl()->implTestInheritanceMiddleBaseInterface);
// DECL:    }
// DECL:    /// some function comment
// DECL:    int bar(int input);

// DEF: int TestInheritanceZDerivedInterface::foo(int input) {
// DEF-NEXT:    getImpl()->implTestInheritanceBaseInterface->foo(getImpl()->implTestInheritanceBaseInterface, getOperation(), input);

// DEF: int TestInheritanceZDerivedInterface::bar(int input) {
// DEF-NEXT:   return getImpl()->implTestInheritanceMiddleBaseInterface->bar(getImpl()->implTestInheritanceMiddleBaseInterface, getOperation(), input);

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
      /*methodName=*/"body_foo",
      /*args=*/(ins "int":$input),
      /*body=*/[{ return 0; }]
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

def OpInterfaceInterfacesOp : Op<TestDialect, "op_inherit_interface_op", [TestInheritanceZDerivedInterface]>;

def DeclareMethodsOp : Op<TestDialect, "declare_methods_op",
                          [DeclareOpInterfaceMethods<TestOpInterface>]>;

def DeclareMethodsWithDefaultOp : Op<TestDialect, "declare_methods_op",
      [DeclareOpInterfaceMethods<TestOpInterface, ["default_foo"]>]>;

// DECL-LABEL: TestOpInterfaceInterfaceTraits
// DECL: class TestOpInterface : public ::mlir::OpInterface<TestOpInterface, detail::TestOpInterfaceInterfaceTraits>

// DECL: /// some function comment
// DECL: int foo(int input);

// Trait declarations / definitions come after interface definitions.
// DECL: struct ExtraShardDeclsInterfaceTrait : public
// DECL:      bool sharedMethodDeclaration() {
// DECL-NEXT:   return (*static_cast<ConcreteOp *>(this)).someOtherMethod();
// DECL-NEXT: }

// DECL-LABEL: struct TestOpInterfaceVerifyTrait : public
// DECL: verifyTrait

// DECL-LABEL: struct TestOpInterfaceVerifyRegionTrait : public
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

// OP_DECL: class OpInterfaceInterfacesOp :
// OP_DECL-SAME: TestInheritanceBaseInterface::Trait, TestInheritanceMiddleBaseInterface::Trait, TestInheritanceZDerivedInterface::Trait

// DOCS-LABEL: {{^}}## TestOpInterface (`TestOpInterface`)
// DOCS: some op interface description

// DOCS: {{^}}### Methods:

// DOCS: {{^}}#### `foo`
// DOCS: some function comment

// DOCS: {{^}}#### `body_foo`
// DOCS: some function comment

// DOCS: {{^}}#### `default_foo`
// DOCS: some function comment
