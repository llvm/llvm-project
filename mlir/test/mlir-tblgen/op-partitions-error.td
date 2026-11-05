// RUN: not mlir-tblgen -gen-op-decls -I %S/../../include -DERROR1 %s 2>&1 | FileCheck --check-prefix=ERROR1 %s
// RUN: not mlir-tblgen -gen-op-defs -I %S/../../include -DERROR2 %s 2>&1 | FileCheck --check-prefix=ERROR2 %s
// RUN: not mlir-tblgen -gen-op-defs -I %S/../../include -DERROR3 %s 2>&1 | FileCheck --check-prefix=ERROR3 %s
// RUN: not mlir-tblgen -gen-op-defs -op-partition=Memroy -I %S/../../include %s 2>&1 | FileCheck --check-prefix=ERROR4 %s
// RUN: not mlir-tblgen -gen-op-defs -op-partition=Memory -op-shard-count=2 -I %S/../../include %s 2>&1 | FileCheck --check-prefix=ERROR5 %s
// RUN: not mlir-tblgen -gen-op-defs -op-shard-count=2 -I %S/../../include %s 2>&1 | FileCheck --check-prefix=ERROR5 %s
// RUN: not mlir-tblgen -gen-dialect-decls -I %S/../../include -DERROR6 %s 2>&1 | FileCheck --check-prefix=ERROR6 %s
// RUN: not mlir-tblgen -gen-dialect-decls -I %S/../../include -DERROR7 %s 2>&1 | FileCheck --check-prefix=ERROR7 %s
// RUN: not mlir-tblgen -gen-op-doc -op-partition=Memroy -I %S/../../include %s 2>&1 | FileCheck --check-prefix=ERROR4 %s
// RUN: not mlir-tblgen -gen-op-defs -I %S/../../include -DERROR8 %s 2>&1 | FileCheck --check-prefix=ERROR8 %s
// RUN: not mlir-tblgen -gen-op-defs -I %S/../../include -DERROR9 %s 2>&1 | FileCheck --check-prefix=ERROR9 %s

include "mlir/IR/OpBase.td"

def MemoryPartition : Partition<"Memory">;
def CorePartition : Partition<"Core">;
#ifdef ERROR6
// ERROR6: error: assertion failed: op partition name 'my-partition' is not a C++ identifier
def InvalidPartition : Partition<"my-partition">;
#endif
def SnakeCasePartition : Partition<"control_flow">;
def CamelCasePartition : Partition<"ControlFlow">;

def Test_Dialect : Dialect {
  let name = "test";
  let opPartitions = [MemoryPartition];
#ifdef ERROR7
  // ERROR7: error: op partitions 'control_flow' and 'ControlFlow' of dialect 'test' have the same registration hook 'registerOperationsControlFlow'
  let opPartitions = [MemoryPartition, SnakeCasePartition, CamelCasePartition];
#endif
#ifdef ERROR9
  // ERROR9: error: assertion failed: default op partition 'Core' (record 'CorePartition') of dialect 'test' is not one of its `opPartitions`
  let defaultOpPartition = CorePartition;
#endif
}

def LoadOp : Op<Test_Dialect, "load"> {
  let partition = MemoryPartition;
}

#ifdef ERROR1
// ERROR1: error: assertion failed: op 'NoPartitionOp' is in no partition, but dialect 'test' has `opPartitions` and no `defaultOpPartition`
def NoPartitionOp : Op<Test_Dialect, "no_partition">;
#endif

#ifdef ERROR2
// ERROR2: error: assertion failed: op 'UnknownPartitionOp' is in partition 'Core' (record 'CorePartition'), which is not one of the `opPartitions` of dialect 'test'
def UnknownPartitionOp : Op<Test_Dialect, "unknown_partition"> {
  let partition = CorePartition;
}
#endif

#ifdef ERROR3
def Other_Dialect : Dialect {
  let name = "other";
}
// ERROR3: error: assertion failed: op 'UnpartitionedDialectOp' is in partition 'Memory' (record 'MemoryPartition'), which is not one of the `opPartitions` of dialect 'other'
def UnpartitionedDialectOp : Op<Other_Dialect, "op"> {
  let partition = MemoryPartition;
}
#endif

#ifdef ERROR8
// Partitions are matched by record, not by name.
def OtherMemoryPartition : Partition<"Memory">;
// ERROR8: error: assertion failed: op 'OtherMemoryOp' is in partition 'Memory' (record 'OtherMemoryPartition'), which is not one of the `opPartitions` of dialect 'test'
def OtherMemoryOp : Op<Test_Dialect, "other_memory"> {
  let partition = OtherMemoryPartition;
}
#endif

// ERROR4: error: requested op partition 'Memroy' is not declared in the `opPartitions` of any dialect with ops

// ERROR5: error: dialect 'test' has `opPartitions`, which cannot be combined with -op-shard-count
