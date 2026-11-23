// RUN: mlir-tblgen -gen-op-decls -op-partition=memory -I %S/../../include %s | FileCheck %s --check-prefix=DECLS
// RUN: mlir-tblgen -gen-op-defs -op-partition=memory -I %S/../../include %s | FileCheck %s --check-prefix=DEFS
// RUN: mlir-tblgen -gen-op-defs -op-partition=Arith -I %S/../../include %s | FileCheck %s --check-prefix=FIRST
// RUN: mlir-tblgen -gen-op-decls -I %S/../../include %s | FileCheck %s --check-prefix=ALL-DECLS
// RUN: mlir-tblgen -gen-op-defs -I %S/../../include %s | FileCheck %s --check-prefix=ALL-DEFS

include "mlir/IR/OpBase.td"

def Test_ArithPartition : Partition<"Arith">;
def Test_MemoryPartition : Partition<"memory">;

def Test_Dialect : Dialect {
  let name = "test";
  let cppNamespace = "test";
}

class Test_Op<string mnemonic, list<Trait> traits = []>
    : Op<Test_Dialect, mnemonic, traits> {
  let partition = Test_ArithPartition;
}

def AddOp : Test_Op<"add">, Arguments<(ins I32:$lhs, I32:$rhs)>;
def MulOp : Test_Op<"mul">, Arguments<(ins I32:$lhs, I32:$rhs)>;

let partition = Test_MemoryPartition in {
def LoadOp : Test_Op<"load">, Arguments<(ins I64:$addr)>;
def StoreOp : Test_Op<"store">, Arguments<(ins I32:$value, I64:$addr)>;
def FenceOp : Test_Op<"fence">;
}

// Only the ops of the specified partition are declared. The ops of the other
// partitions are forward declared, so that declarations can refer to them.
// The registration hooks of all partitions are declared, as well as the hook
// that registers the ops of all partitions.
// DECLS: namespace test {
// DECLS-EMPTY:
// DECLS-NEXT: class AddOp;
// DECLS: class MulOp;
// DECLS: class FenceOp;
// DECLS: class LoadOp;
// DECLS: class StoreOp;
// DECLS-LABEL: #ifdef GET_OP_CLASSES
// DECLS-NOT: AddOp
// DECLS-NOT: MulOp
// DECLS: class LoadOp : public
// DECLS-NOT: AddOp
// DECLS-NOT: MulOp
// DECLS: #endif // GET_OP_CLASSES
// DECLS: namespace test {
// DECLS-EMPTY:
// DECLS-NEXT: void registerTestDialectOperations(test::TestDialect *dialect);
// DECLS-NEXT: void registerTestDialectOperationsArith(test::TestDialect *dialect);
// DECLS-NEXT: void registerTestDialectOperationsMemory(test::TestDialect *dialect);
// DECLS-EMPTY:
// DECLS-NEXT: } // namespace test

// The registration hook of the partition replaces the op list.
// DEFS-NOT: GET_OP_LIST
// DEFS-LABEL: #ifdef GET_OP_CLASSES
// DEFS: static ::llvm::LogicalResult __mlir_ods_local_type_constraint_6memory_op2Dpartitions{{[0-9]+}}(
// DEFS-NOT: AddOp
// DEFS: LoadOpAdaptor
// DEFS: StoreOpAdaptor
// DEFS-NOT: AddOp
// DEFS-NOT: registerTestDialectOperations(
// DEFS: void test::registerTestDialectOperationsMemory(test::TestDialect *dialect) {
// DEFS-NEXT:   ::mlir::RegisteredOperationName::insert<test::FenceOp>(*dialect);
// DEFS-NEXT:   ::mlir::RegisteredOperationName::insert<test::LoadOp>(*dialect);
// DEFS-NEXT:   ::mlir::RegisteredOperationName::insert<test::StoreOp>(*dialect);
// DEFS-NEXT: }
// DEFS-EMPTY:
// DEFS-NEXT: #endif // GET_OP_CLASSES

// The first partition by name also defines the hook that registers the ops of
// all partitions.
// FIRST-LABEL: #ifdef GET_OP_CLASSES
// FIRST: void test::registerTestDialectOperations(test::TestDialect *dialect) {
// FIRST-NEXT:   test::registerTestDialectOperationsArith(dialect);
// FIRST-NEXT:   test::registerTestDialectOperationsMemory(dialect);
// FIRST-NEXT: }
// FIRST: void test::registerTestDialectOperationsArith(test::TestDialect *dialect) {
// FIRST-NEXT:   ::mlir::RegisteredOperationName::insert<test::AddOp>(*dialect);
// FIRST-NEXT:   ::mlir::RegisteredOperationName::insert<test::MulOp>(*dialect);
// FIRST-NEXT: }

// Without -op-partition, partitions are ignored.
// ALL-DECLS-NOT: register
// ALL-DEFS-NOT: register
// ALL-DEFS-LABEL: #ifdef GET_OP_LIST
// ALL-DEFS: test::AddOp,
// ALL-DEFS: test::FenceOp,
// ALL-DEFS: test::LoadOp,
// ALL-DEFS: test::MulOp,
// ALL-DEFS: test::StoreOp
// ALL-DEFS-NOT: register
