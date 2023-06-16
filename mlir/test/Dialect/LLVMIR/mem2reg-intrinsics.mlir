// RUN: mlir-opt %s --pass-pipeline="builtin.module(llvm.func(mem2reg{region-simplify=false}))" --split-input-file | FileCheck %s

// CHECK-LABEL: llvm.func @basic_memset
llvm.func @basic_memset() -> i32 {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
  %memset_value = llvm.mlir.constant(42 : i8) : i8
  %memset_len = llvm.mlir.constant(4 : i32) : i32
  // CHECK-DAG: %[[MEMSET_VALUE:.*]] = llvm.mlir.constant(42 : i8) : i8
  // CHECK-DAG: %[[C8:.*]] = llvm.mlir.constant(8 : i32) : i32
  // CHECK-DAG: %[[C16:.*]] = llvm.mlir.constant(16 : i32) : i32
  "llvm.intr.memset"(%1, %memset_value, %memset_len) <{isVolatile = false}> : (!llvm.ptr, i8, i32) -> ()
  // CHECK-NOT: "llvm.intr.memset"
  // CHECK: %[[VALUE_8:.*]] = llvm.zext %[[MEMSET_VALUE]] : i8 to i32
  // CHECK: %[[SHIFTED_8:.*]] = llvm.shl %[[VALUE_8]], %[[C8]]
  // CHECK: %[[VALUE_16:.*]] = llvm.or %[[VALUE_8]], %[[SHIFTED_8]]
  // CHECK: %[[SHIFTED_16:.*]] = llvm.shl %[[VALUE_16]], %[[C16]]
  // CHECK: %[[VALUE_32:.*]] = llvm.or %[[VALUE_16]], %[[SHIFTED_16]]
  // CHECK-NOT: "llvm.intr.memset"
  %2 = llvm.load %1 {alignment = 4 : i64} : !llvm.ptr -> i32
  // CHECK: llvm.return %[[VALUE_32]] : i32
  llvm.return %2 : i32
}

// -----

// CHECK-LABEL: llvm.func @allow_dynamic_value_memset
// CHECK-SAME: (%[[MEMSET_VALUE:.*]]: i8)
llvm.func @allow_dynamic_value_memset(%memset_value: i8) -> i32 {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
  %memset_len = llvm.mlir.constant(4 : i32) : i32
  // CHECK-DAG: %[[C8:.*]] = llvm.mlir.constant(8 : i32) : i32
  // CHECK-DAG: %[[C16:.*]] = llvm.mlir.constant(16 : i32) : i32
  "llvm.intr.memset"(%1, %memset_value, %memset_len) <{isVolatile = false}> : (!llvm.ptr, i8, i32) -> ()
  // CHECK-NOT: "llvm.intr.memset"
  // CHECK: %[[VALUE_8:.*]] = llvm.zext %[[MEMSET_VALUE]] : i8 to i32
  // CHECK: %[[SHIFTED_8:.*]] = llvm.shl %[[VALUE_8]], %[[C8]]
  // CHECK: %[[VALUE_16:.*]] = llvm.or %[[VALUE_8]], %[[SHIFTED_8]]
  // CHECK: %[[SHIFTED_16:.*]] = llvm.shl %[[VALUE_16]], %[[C16]]
  // CHECK: %[[VALUE_32:.*]] = llvm.or %[[VALUE_16]], %[[SHIFTED_16]]
  // CHECK-NOT: "llvm.intr.memset"
  %2 = llvm.load %1 {alignment = 4 : i64} : !llvm.ptr -> i32
  // CHECK: llvm.return %[[VALUE_32]] : i32
  llvm.return %2 : i32
}

// -----

// CHECK-LABEL: llvm.func @exotic_target_memset
llvm.func @exotic_target_memset() -> i40 {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.alloca %0 x i40 {alignment = 4 : i64} : (i32) -> !llvm.ptr
  %memset_value = llvm.mlir.constant(42 : i8) : i8
  %memset_len = llvm.mlir.constant(5 : i32) : i32
  // CHECK-DAG: %[[MEMSET_VALUE:.*]] = llvm.mlir.constant(42 : i8) : i8
  // CHECK-DAG: %[[C8:.*]] = llvm.mlir.constant(8 : i40) : i40
  // CHECK-DAG: %[[C16:.*]] = llvm.mlir.constant(16 : i40) : i40
  // CHECK-DAG: %[[C32:.*]] = llvm.mlir.constant(32 : i40) : i40
  "llvm.intr.memset"(%1, %memset_value, %memset_len) <{isVolatile = false}> : (!llvm.ptr, i8, i32) -> ()
  // CHECK-NOT: "llvm.intr.memset"
  // CHECK: %[[VALUE_8:.*]] = llvm.zext %[[MEMSET_VALUE]] : i8 to i40
  // CHECK: %[[SHIFTED_8:.*]] = llvm.shl %[[VALUE_8]], %[[C8]]
  // CHECK: %[[VALUE_16:.*]] = llvm.or %[[VALUE_8]], %[[SHIFTED_8]]
  // CHECK: %[[SHIFTED_16:.*]] = llvm.shl %[[VALUE_16]], %[[C16]]
  // CHECK: %[[VALUE_32:.*]] = llvm.or %[[VALUE_16]], %[[SHIFTED_16]]
  // CHECK: %[[SHIFTED_COMPL:.*]] = llvm.shl %[[VALUE_32]], %[[C32]]
  // CHECK: %[[VALUE_COMPL:.*]] = llvm.or %[[VALUE_32]], %[[SHIFTED_COMPL]]
  // CHECK-NOT: "llvm.intr.memset"
  %2 = llvm.load %1 {alignment = 4 : i64} : !llvm.ptr -> i40
  // CHECK: llvm.return %[[VALUE_COMPL]] : i40
  llvm.return %2 : i40
}

// -----

// CHECK-LABEL: llvm.func @no_volatile_memset
llvm.func @no_volatile_memset() -> i32 {
  // CHECK-DAG: %[[ALLOCA_LEN:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-DAG: %[[ALLOCA:.*]] = llvm.alloca %[[ALLOCA_LEN]] x i32
  // CHECK-DAG: %[[MEMSET_VALUE:.*]] = llvm.mlir.constant(42 : i8) : i8
  // CHECK-DAG: %[[MEMSET_LEN:.*]] = llvm.mlir.constant(4 : i32) : i32
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
  %memset_value = llvm.mlir.constant(42 : i8) : i8
  %memset_len = llvm.mlir.constant(4 : i32) : i32
  // CHECK: "llvm.intr.memset"(%[[ALLOCA]], %[[MEMSET_VALUE]], %[[MEMSET_LEN]]) <{isVolatile = true}>
  "llvm.intr.memset"(%1, %memset_value, %memset_len) <{isVolatile = true}> : (!llvm.ptr, i8, i32) -> ()
  %2 = llvm.load %1 {alignment = 4 : i64} : !llvm.ptr -> i32
  llvm.return %2 : i32
}

// -----

// CHECK-LABEL: llvm.func @no_partial_memset
llvm.func @no_partial_memset() -> i32 {
  // CHECK-DAG: %[[ALLOCA_LEN:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-DAG: %[[ALLOCA:.*]] = llvm.alloca %[[ALLOCA_LEN]] x i32
  // CHECK-DAG: %[[MEMSET_VALUE:.*]] = llvm.mlir.constant(42 : i8) : i8
  // CHECK-DAG: %[[MEMSET_LEN:.*]] = llvm.mlir.constant(2 : i32) : i32
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
  %memset_value = llvm.mlir.constant(42 : i8) : i8
  %memset_len = llvm.mlir.constant(2 : i32) : i32
  // CHECK: "llvm.intr.memset"(%[[ALLOCA]], %[[MEMSET_VALUE]], %[[MEMSET_LEN]]) <{isVolatile = false}>
  "llvm.intr.memset"(%1, %memset_value, %memset_len) <{isVolatile = false}> : (!llvm.ptr, i8, i32) -> ()
  %2 = llvm.load %1 {alignment = 4 : i64} : !llvm.ptr -> i32
  llvm.return %2 : i32
}

// -----

// CHECK-LABEL: llvm.func @no_overflowing_memset
llvm.func @no_overflowing_memset() -> i32 {
  // CHECK-DAG: %[[ALLOCA_LEN:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-DAG: %[[ALLOCA:.*]] = llvm.alloca %[[ALLOCA_LEN]] x i32
  // CHECK-DAG: %[[MEMSET_VALUE:.*]] = llvm.mlir.constant(42 : i8) : i8
  // CHECK-DAG: %[[MEMSET_LEN:.*]] = llvm.mlir.constant(6 : i32) : i32
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
  %memset_value = llvm.mlir.constant(42 : i8) : i8
  %memset_len = llvm.mlir.constant(6 : i32) : i32
  // CHECK: "llvm.intr.memset"(%[[ALLOCA]], %[[MEMSET_VALUE]], %[[MEMSET_LEN]]) <{isVolatile = false}>
  "llvm.intr.memset"(%1, %memset_value, %memset_len) <{isVolatile = false}> : (!llvm.ptr, i8, i32) -> ()
  %2 = llvm.load %1 {alignment = 4 : i64} : !llvm.ptr -> i32
  llvm.return %2 : i32
}

// -----

// CHECK-LABEL: llvm.func @only_byte_aligned_integers_memset
llvm.func @only_byte_aligned_integers_memset() -> i10 {
  // CHECK-DAG: %[[ALLOCA_LEN:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-DAG: %[[ALLOCA:.*]] = llvm.alloca %[[ALLOCA_LEN]] x i10
  // CHECK-DAG: %[[MEMSET_VALUE:.*]] = llvm.mlir.constant(42 : i8) : i8
  // CHECK-DAG: %[[MEMSET_LEN:.*]] = llvm.mlir.constant(2 : i32) : i32
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.alloca %0 x i10 {alignment = 4 : i64} : (i32) -> !llvm.ptr
  %memset_value = llvm.mlir.constant(42 : i8) : i8
  %memset_len = llvm.mlir.constant(2 : i32) : i32
  // CHECK: "llvm.intr.memset"(%[[ALLOCA]], %[[MEMSET_VALUE]], %[[MEMSET_LEN]]) <{isVolatile = false}>
  "llvm.intr.memset"(%1, %memset_value, %memset_len) <{isVolatile = false}> : (!llvm.ptr, i8, i32) -> ()
  %2 = llvm.load %1 {alignment = 4 : i64} : !llvm.ptr -> i10
  llvm.return %2 : i10
}
