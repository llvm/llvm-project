// RUN: mlir-tblgen -gen-op-decls -op-partition=memory -I %S/../../include %s | FileCheck %s --check-prefix=DECLS
// RUN: mlir-tblgen -gen-op-defs -op-partition=memory -I %S/../../include %s | FileCheck %s --check-prefix=DEFS
// RUN: mlir-tblgen -gen-op-defs -op-partition=empty -I %S/../../include %s | FileCheck %s --check-prefix=EMPTY
// RUN: mlir-tblgen -gen-op-defs -I %S/../../include %s | FileCheck %s --check-prefix=ALL
// RUN: mlir-tblgen -gen-dialect-decls -I %S/../../include %s | FileCheck %s --check-prefix=DIALECT-DECLS

include "mlir/IR/OpBase.td"

def Test_ArithPartition : Partition<"Arith">;
def Test_MemoryPartition : Partition<"memory">;
def Test_EmptyPartition : Partition<"empty">;

def Test_Dialect : Dialect {
  let name = "test";
  let cppNamespace = "test";
  let opPartitions = [Test_ArithPartition, Test_MemoryPartition,
                      Test_EmptyPartition];
  let defaultOpPartition = Test_ArithPartition;
}

class Test_Op<string mnemonic, list<Trait> traits = []>
    : Op<Test_Dialect, mnemonic, traits>;

// These ops are in the default partition, Arith.
def AddOp : Test_Op<"add">, Arguments<(ins I32:$lhs, I32:$rhs)>;
def MulOp : Test_Op<"mul">, Arguments<(ins I32:$lhs, I32:$rhs)>;

let partition = Test_MemoryPartition in {
def LoadOp : Test_Op<"load">, Arguments<(ins I64:$addr)>;
def StoreOp : Test_Op<"store">, Arguments<(ins I32:$value, I64:$addr)>;
def FenceOp : Test_Op<"fence">;
}

// Only the ops of the requested partition are declared. The ops of the other
// partitions are forward declared, so that declarations can refer to them.
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
// DECLS-NOT: register

// The registration hooks replace the op list.
// DEFS-NOT: GET_OP_LIST
// DEFS-LABEL: #ifdef GET_OP_CLASSES
// DEFS: static ::llvm::LogicalResult __mlir_ods_local_type_constraint_6memory_op2Dpartitions{{[0-9]+}}(
// DEFS-NOT: AddOp
// DEFS: LoadOpAdaptor
// DEFS: StoreOpAdaptor
// DEFS-NOT: AddOp
// DEFS: void test::TestDialect::registerOperationsMemory() {
// DEFS-NEXT:   ::mlir::RegisteredOperationName::insert<test::FenceOp>(*this);
// DEFS-NEXT:   ::mlir::RegisteredOperationName::insert<test::LoadOp>(*this);
// DEFS-NEXT:   ::mlir::RegisteredOperationName::insert<test::StoreOp>(*this);
// DEFS-NEXT: }
// DEFS-EMPTY:
// DEFS-NEXT: #endif // GET_OP_CLASSES

// A partition without ops still gets its hook, which the dialect calls.
// EMPTY-LABEL: #ifdef GET_OP_CLASSES
// EMPTY: void test::TestDialect::registerOperationsEmpty() {
// EMPTY-NEXT: }
// EMPTY-EMPTY:
// EMPTY-NEXT: #endif // GET_OP_CLASSES

// Without a requested partition, all ops are generated with both the op list
// and the hooks of all partitions, so that a dialect can adopt partitions
// before it generates its ops by partition. Ops without a partition are in the
// default partition.
// ALL-LABEL: #ifdef GET_OP_LIST
// ALL: test::AddOp,
// ALL: test::FenceOp,
// ALL: test::LoadOp,
// ALL: test::MulOp,
// ALL: test::StoreOp
// ALL-LABEL: #ifdef GET_OP_CLASSES
// ALL: static ::llvm::LogicalResult __mlir_ods_local_type_constraint_op2Dpartitions{{[0-9]+}}(
// ALL: void test::TestDialect::registerOperationsArith() {
// ALL-NEXT:   ::mlir::RegisteredOperationName::insert<test::AddOp>(*this);
// ALL-NEXT:   ::mlir::RegisteredOperationName::insert<test::MulOp>(*this);
// ALL-NEXT: }
// ALL: void test::TestDialect::registerOperationsMemory() {
// ALL: void test::TestDialect::registerOperationsEmpty() {

// The hooks are private members of the dialect class; the dialect's own
// declarations that follow stay public.
// DIALECT-DECLS: class TestDialect
// DIALECT-DECLS: private:
// DIALECT-DECLS-NEXT: /// Registers the operations of all op partitions of the dialect.
// DIALECT-DECLS-NEXT: void registerOperations() {
// DIALECT-DECLS-NEXT:   registerOperationsArith();
// DIALECT-DECLS-NEXT:   registerOperationsMemory();
// DIALECT-DECLS-NEXT:   registerOperationsEmpty();
// DIALECT-DECLS-NEXT: }
// DIALECT-DECLS-NEXT: /// Registers the operations in op partition `Arith`.
// DIALECT-DECLS-NEXT: void registerOperationsArith();
// DIALECT-DECLS-NEXT: /// Registers the operations in op partition `memory`.
// DIALECT-DECLS-NEXT: void registerOperationsMemory();
// DIALECT-DECLS-NEXT: /// Registers the operations in op partition `empty`.
// DIALECT-DECLS-NEXT: void registerOperationsEmpty();
// DIALECT-DECLS-NEXT: public:
// DIALECT-DECLS-NEXT: };
