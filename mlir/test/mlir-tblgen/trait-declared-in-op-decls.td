// RUN: mlir-tblgen -gen-op-decls -I %S/../../include %s | FileCheck %s

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaceBase.td"

def Test_Dialect : Dialect {
  let name = "test";
  let cppNamespace = "::test";
}

def NoHeaderTrait : NativeOpTrait<"NoHeaderTrait">;
def OpTraitA : NativeOpTrait<"OpTraitA"> {
  let declaredIn = ["test/OpTraitA.h"];
}
// A trait can be declared across several headers.
def OpTraitB : NativeOpTrait<"OpTraitB"> {
  let declaredIn = ["test/OpTraitB.h", "test/OpTraitBUtils.h"];
}

def BaseOpInterface : OpInterface<"BaseOpInterface"> {
  let cppNamespace = "::test";
  let declaredIn = ["test/BaseOpInterface.h"];
}
def DerivedOpInterface : OpInterface<"DerivedOpInterface", [BaseOpInterface]> {
  let cppNamespace = "::test";
  let declaredIn = ["test/DerivedOpInterface.h"];
}
def DeclaredMethodsOpInterface : OpInterface<"DeclaredMethodsOpInterface"> {
  let cppNamespace = "::test";
  let declaredIn = ["test/DeclaredMethodsOpInterface.h"];
  let methods = [InterfaceMethod<"", "void", "foo">];
}

def TraitEffectInterface
    : EffectOpInterfaceBase<"TraitEffectOpInterface", "::test::TraitEffect"> {
  let cppNamespace = "::test";
  let declaredIn = ["test/TraitEffects.h"];
}
def DecoratorEffectInterface
    : EffectOpInterfaceBase<"DecoratorEffectOpInterface",
                            "::test::DecoratorEffect"> {
  let cppNamespace = "::test";
  let declaredIn = ["test/DecoratorEffects.h"];
}

// The headers of all ops are emitted once, sorted, at the start of the class
// declarations. Traits without `declaredIn` don't add an include.

// CHECK-LABEL: #ifdef GET_OP_CLASSES
// CHECK-NEXT: #undef GET_OP_CLASSES
// CHECK-EMPTY:
// CHECK-NEXT: #include "mlir/Bytecode/BytecodeOpInterface.h"
// CHECK-NEXT: #include "mlir/IR/Builders.h"
// CHECK-NEXT: #include "mlir/IR/OpDefinition.h"
// CHECK-NEXT: #include "test/BaseOpInterface.h"
// CHECK-NEXT: #include "test/DeclaredMethodsOpInterface.h"
// CHECK-NEXT: #include "test/DecoratorEffects.h"
// CHECK-NEXT: #include "test/DerivedOpInterface.h"
// CHECK-NEXT: #include "test/OpTraitA.h"
// CHECK-NEXT: #include "test/OpTraitB.h"
// CHECK-NEXT: #include "test/OpTraitBUtils.h"
// CHECK-NEXT: #include "test/TraitEffects.h"
// CHECK-EMPTY:
// CHECK-NEXT: namespace test {
// CHECK-NOT: #include

// Base interfaces of an interface are added as op traits, so their headers are
// included too.
def InterfaceOp : Op<Test_Dialect, "interface", [
    NoHeaderTrait, OpTraitA, DerivedOpInterface]>;

// The header is propagated through `DeclareOpInterfaceMethods`.
def DeclareMethodsOp : Op<Test_Dialect, "declare_methods", [
    OpTraitA, DeclareOpInterfaceMethods<DeclaredMethodsOpInterface>]>;

// Trait lists are flattened.
def TraitListOp : Op<Test_Dialect, "trait_list", [TraitList<[OpTraitB]>]>;

// The header is propagated through side effect traits and decorators.
def EffectTraitOp : Op<Test_Dialect, "effect_trait", [
    SideEffectsTraitBase<TraitEffectInterface, [
      SideEffect<TraitEffectInterface, "Read", DefaultResource, 0,
                 FullEffect>]>]>;
def EffectDecoratorOp : Op<Test_Dialect, "effect_decorator"> {
  let arguments = (ins Arg<AnyType, "", [
    SideEffect<DecoratorEffectInterface, "Read", DefaultResource, 0,
               FullEffect>]>:$value);
}

// Ops with properties implement the bytecode interface.
def PropertiesOp : Op<Test_Dialect, "properties"> {
  let arguments = (ins I32Attr:$value);
}
