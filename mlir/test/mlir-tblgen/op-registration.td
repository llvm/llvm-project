// RUN: mlir-tblgen -gen-op-decls -I %S/../../include %s | FileCheck %s --check-prefix=DECL
// RUN: mlir-tblgen -gen-op-defs -I %S/../../include %s | FileCheck %s --check-prefix=DEF
// RUN: mlir-tblgen -gen-op-decls -op-shard-count=2 -I %S/../../include %s | FileCheck %s --check-prefix=SPLIT
// RUN: mlir-tblgen -gen-op-defs -op-shard-count=2 -I %S/../../include %s | FileCheck %s --check-prefix=SPLIT
// RUN: mlir-tblgen -gen-op-decls -op-registration-name=Ext -I %S/../../include %s | FileCheck %s --check-prefix=NAME
// RUN: mlir-tblgen -gen-op-defs -op-registration-name=Ext -I %S/../../include %s | FileCheck %s --check-prefix=NAME

include "mlir/IR/OpBase.td"

// The registration hook is only generated for the dialects using the default
// op registration, also when sharding, and only registers the ops of its
// dialect. The ops are registered with globally qualified names, as the hook
// is defined in the dialect namespace, where `test` in `test::nested::AOp` would
// name the op class `test::nested::test`.

def Test_Dialect : Dialect {
  let name = "test";
  let cppNamespace = "test::nested";
  let useDefaultOperationRegistration = 1;
}

def Other_Dialect : Dialect {
  let name = "other";
  let cppNamespace = "::other";
  let useDefaultOperationRegistration = 0;
}

def Test_AOp : Op<Test_Dialect, "a">;
def Other_COp : Op<Other_Dialect, "c">;
def Test_BOp : Op<Test_Dialect, "b">;
def Test_test : Op<Test_Dialect, "test">;

// DECL-NOT: registerOtherDialectOperations
// DECL: class TestDialect;
// DECL-NEXT: /// Registers the operations defined in this file with `dialect`.
// DECL-NEXT: void registerTestDialectOperations(TestDialect *dialect);
// DECL-NOT: registerOtherDialectOperations

// DEF-NOT: registerOtherDialectOperations
// DEF: void registerTestDialectOperations(TestDialect *dialect) {
// DEF-NEXT:   ::mlir::RegisteredOperationName::insert<::test::nested::AOp>(*dialect);
// DEF-NEXT:   ::mlir::RegisteredOperationName::insert<::test::nested::BOp>(*dialect);
// DEF-NEXT:   ::mlir::RegisteredOperationName::insert<::test::nested::test>(*dialect);
// DEF-NEXT: }
// DEF-NOT: registerOtherDialectOperations
// DEF: #endif // GET_OP_CLASSES

// NAME-NOT: registerTestDialectOperations
// NAME: void registerExtOperations(TestDialect *dialect)
// NAME-NOT: registerTestDialectOperations

// SPLIT-NOT: registerOtherDialectOperations
// SPLIT: registerTestDialectOperations(
// SPLIT: registerTestDialectOperations0(
// SPLIT: registerTestDialectOperations1(
// SPLIT-NOT: registerOtherDialectOperations
