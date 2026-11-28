// RUN: mlir-tblgen -gen-op-defs -op-shard-count=2 -I %S/../../include %s | FileCheck %s --check-prefix=DEFS
// RUN: mlir-tblgen -gen-op-decls -op-shard-count=2 -I %S/../../include %s | FileCheck %s --check-prefix=DECLS
// RUN: mlir-tblgen -gen-op-defs -op-shard-count=2 -op-registration-name=Ext -I %S/../../include %s | FileCheck %s --check-prefix=NAME

include "mlir/IR/OpBase.td"

def Test_Dialect : Dialect {
  let name = "test";
  let cppNamespace = "test";
  let useDefaultOperationRegistration = 1;
}

class Test_Op<string mnemonic, list<Trait> traits = []> 
    : Op<Test_Dialect, mnemonic, traits>;

def OpA : Test_Op<"a">;
def OpB : Test_Op<"b">;
def OpC : Test_Op<"c">;

// A dialect in the global namespace, whose op lands in the second shard.
def Other_Dialect : Dialect {
  let name = "other";
  let cppNamespace = "";
  let useDefaultOperationRegistration = 1;
}
def Other_DOp : Op<Other_Dialect, "d">;

// DECLS: OpA
// DECLS: OpB
// DECLS: OpC
// DECLS: class TestDialect;
// DECLS: registerTestDialectOperations(
// DECLS: registerTestDialectOperations0(
// DECLS: registerTestDialectOperations1(

// The op list contains the ops of all shards.
// DEFS-LABEL: #ifdef GET_OP_LIST
// DEFS: test::OpA,
// DEFS-NEXT: test::OpB,
// DEFS-NEXT: test::OpC,
// DEFS-NEXT: DOp
// DEFS-NEXT: #endif // GET_OP_LIST

// DEFS-LABEL: GET_OP_DEFS_0
// DEFS: OpAAdaptor
// DEFS: OpBAdaptor
// DEFS: void registerTestDialectOperations(TestDialect *dialect) {
// DEFS-NEXT:   registerTestDialectOperations0(dialect);
// DEFS-NEXT:   registerTestDialectOperations1(dialect);
// DEFS-NEXT: }
// DEFS: void registerTestDialectOperations0(

// DEFS-LABEL: GET_OP_DEFS_1
// DEFS: OpCAdaptor
// DEFS: void registerTestDialectOperations1(TestDialect *dialect) {
// DEFS-NEXT:   ::mlir::RegisteredOperationName::insert<::test::OpC>(*dialect);
// DEFS-NEXT: }

// Every dialect has the hooks of all shards, defined without qualification so
// that they are valid in the global namespace too. The ops are named with
// global qualification, also in the global namespace.
// DEFS: void registerOtherDialectOperations1(OtherDialect *dialect) {
// DEFS-NEXT:   ::mlir::RegisteredOperationName::insert<::DOp>(*dialect);
// DEFS-NEXT: }

// The registration name applies to the hooks of all shards.
// NAME-LABEL: GET_OP_DEFS_0
// NAME: void registerExtOperations(TestDialect *dialect) {
// NAME-NEXT:   registerExtOperations0(dialect);
// NAME-NEXT:   registerExtOperations1(dialect);
// NAME-NEXT: }
// NAME: void registerExtOperations0(
// NAME-LABEL: GET_OP_DEFS_1
// NAME: void registerExtOperations1(
