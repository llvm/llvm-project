// RUN: mlir-tblgen -gen-op-defs -op-shard-count=2 -I %S/../../include %s | FileCheck %s --check-prefix=DEFS
// RUN: mlir-tblgen -gen-op-decls -op-shard-count=2 -I %S/../../include %s | FileCheck %s --check-prefix=DECLS

include "mlir/IR/OpBase.td"

def Test_Dialect : Dialect {
  let name = "test";
  let cppNamespace = "test";
}

class Test_Op<string mnemonic, list<Trait> traits = []> 
    : Op<Test_Dialect, mnemonic, traits>;

def OpA : Test_Op<"a">;
def OpB : Test_Op<"b">;
def OpC : Test_Op<"c">;

// DECLS: OpA
// DECLS: OpB
// DECLS: OpC
// DECLS: registerTestDialectOperations(
// DECLS: registerTestDialectOperations0(
// DECLS: registerTestDialectOperations1(

// DEFS-LABEL: GET_OP_DEFS_0
// DEFS: void test::registerTestDialectOperations(
// DEFS: void test::registerTestDialectOperations0(
// DEFS: OpAAdaptor
// DEFS: OpBAdaptor

// DEFS-LABEL: GET_OP_DEFS_1
// DEFS: void test::registerTestDialectOperations1(
// DEFS: OpCAdaptor
