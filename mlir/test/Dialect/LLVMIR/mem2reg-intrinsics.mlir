// RUN: mlir-opt %s --pass-pipeline="builtin.module(llvm.func(mem2reg))" --split-input-file | FileCheck %s

// CHECK-LABEL: llvm.func @basic_memset
// CHECK-SAME: (%[[MEMSET_VALUE:.*]]: i8)
llvm.func @basic_memset(%memset_value: i8) -> i32 {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
  %memset_len = llvm.mlir.constant(4 : i32) : i32
  "llvm.intr.memset"(%1, %memset_value, %memset_len) <{isVolatile = false}> : (!llvm.ptr, i8, i32) -> ()
  // CHECK-NOT: "llvm.intr.memset"
  // CHECK: %[[VALUE_8:.*]] = llvm.zext %[[MEMSET_VALUE]] : i8 to i32
  // CHECK: %[[C8:.*]] = llvm.mlir.constant(8 : i32) : i32
  // CHECK: %[[SHIFTED_8:.*]] = llvm.shl %[[VALUE_8]], %[[C8]]
  // CHECK: %[[VALUE_16:.*]] = llvm.or %[[VALUE_8]], %[[SHIFTED_8]]
  // CHECK: %[[C16:.*]] = llvm.mlir.constant(16 : i32) : i32
  // CHECK: %[[SHIFTED_16:.*]] = llvm.shl %[[VALUE_16]], %[[C16]]
  // CHECK: %[[VALUE_32:.*]] = llvm.or %[[VALUE_16]], %[[SHIFTED_16]]
  // CHECK-NOT: "llvm.intr.memset"
  %2 = llvm.load %1 {alignment = 4 : i64} : !llvm.ptr -> i32
  // CHECK: llvm.return %[[VALUE_32]] : i32
  llvm.return %2 : i32
}

// -----

// CHECK-LABEL: llvm.func @basic_memset_inline
// CHECK-SAME: (%[[MEMSET_VALUE:.*]]: i8)
llvm.func @basic_memset_inline(%memset_value: i8) -> i32 {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
  "llvm.intr.memset.inline"(%1, %memset_value) <{isVolatile = false, len = 4 : i32}> : (!llvm.ptr, i8) -> ()
  // CHECK-NOT: "llvm.intr.memset.inline"
  // CHECK: %[[VALUE_8:.*]] = llvm.zext %[[MEMSET_VALUE]] : i8 to i32
  // CHECK: %[[C8:.*]] = llvm.mlir.constant(8 : i32) : i32
  // CHECK: %[[SHIFTED_8:.*]] = llvm.shl %[[VALUE_8]], %[[C8]]
  // CHECK: %[[VALUE_16:.*]] = llvm.or %[[VALUE_8]], %[[SHIFTED_8]]
  // CHECK: %[[C16:.*]] = llvm.mlir.constant(16 : i32) : i32
  // CHECK: %[[SHIFTED_16:.*]] = llvm.shl %[[VALUE_16]], %[[C16]]
  // CHECK: %[[VALUE_32:.*]] = llvm.or %[[VALUE_16]], %[[SHIFTED_16]]
  // CHECK-NOT: "llvm.intr.memset.inline"
  %2 = llvm.load %1 {alignment = 4 : i64} : !llvm.ptr -> i32
  // CHECK: llvm.return %[[VALUE_32]] : i32
  llvm.return %2 : i32
}

// -----

// CHECK-LABEL: llvm.func @basic_memset_constant
llvm.func @basic_memset_constant() -> i32 {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
  %memset_value = llvm.mlir.constant(42 : i8) : i8
  %memset_len = llvm.mlir.constant(4 : i32) : i32
  "llvm.intr.memset"(%1, %memset_value, %memset_len) <{isVolatile = false}> : (!llvm.ptr, i8, i32) -> ()
  %2 = llvm.load %1 {alignment = 4 : i64} : !llvm.ptr -> i32
  // CHECK: %[[C42:.*]] = llvm.mlir.constant(42 : i8) : i8
  // CHECK: %[[VALUE_42:.*]] = llvm.zext %[[C42]] : i8 to i32
  // CHECK: %[[C8:.*]] = llvm.mlir.constant(8 : i32) : i32
  // CHECK: %[[SHIFTED_42:.*]] = llvm.shl %[[VALUE_42]], %[[C8]]  : i32
  // CHECK: %[[OR0:.*]] = llvm.or %[[VALUE_42]], %[[SHIFTED_42]]  : i32
  // CHECK: %[[C16:.*]] = llvm.mlir.constant(16 : i32) : i32
  // CHECK: %[[SHIFTED:.*]] = llvm.shl %[[OR0]], %[[C16]]  : i32
  // CHECK: %[[RES:..*]] = llvm.or %[[OR0]], %[[SHIFTED]]  : i32
  // CHECK: llvm.return %[[RES]] : i32
  llvm.return %2 : i32
}

// -----

// CHECK-LABEL: llvm.func @basic_memset_inline_constant
llvm.func @basic_memset_inline_constant() -> i32 {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
  %memset_value = llvm.mlir.constant(42 : i8) : i8
  "llvm.intr.memset.inline"(%1, %memset_value) <{isVolatile = false, len = 4}> : (!llvm.ptr, i8) -> ()
  %2 = llvm.load %1 {alignment = 4 : i64} : !llvm.ptr -> i32
  // CHECK: %[[C42:.*]] = llvm.mlir.constant(42 : i8) : i8
  // CHECK: %[[VALUE_42:.*]] = llvm.zext %[[C42]] : i8 to i32
  // CHECK: %[[C8:.*]] = llvm.mlir.constant(8 : i32) : i32
  // CHECK: %[[SHIFTED_42:.*]] = llvm.shl %[[VALUE_42]], %[[C8]]  : i32
  // CHECK: %[[OR0:.*]] = llvm.or %[[VALUE_42]], %[[SHIFTED_42]]  : i32
  // CHECK: %[[C16:.*]] = llvm.mlir.constant(16 : i32) : i32
  // CHECK: %[[SHIFTED:.*]] = llvm.shl %[[OR0]], %[[C16]]  : i32
  // CHECK: %[[RES:..*]] = llvm.or %[[OR0]], %[[SHIFTED]]  : i32
  // CHECK: llvm.return %[[RES]] : i32
  llvm.return %2 : i32
}

// -----

// CHECK-LABEL: llvm.func @exotic_target_memset
// CHECK-SAME: (%[[MEMSET_VALUE:.*]]: i8)
llvm.func @exotic_target_memset(%memset_value: i8) -> i40 {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.alloca %0 x i40 {alignment = 4 : i64} : (i32) -> !llvm.ptr
  %memset_len = llvm.mlir.constant(5 : i32) : i32
  "llvm.intr.memset"(%1, %memset_value, %memset_len) <{isVolatile = false}> : (!llvm.ptr, i8, i32) -> ()
  // CHECK-NOT: "llvm.intr.memset"
  // CHECK: %[[VALUE_8:.*]] = llvm.zext %[[MEMSET_VALUE]] : i8 to i40
  // CHECK: %[[C8:.*]] = llvm.mlir.constant(8 : i40) : i40
  // CHECK: %[[SHIFTED_8:.*]] = llvm.shl %[[VALUE_8]], %[[C8]]
  // CHECK: %[[VALUE_16:.*]] = llvm.or %[[VALUE_8]], %[[SHIFTED_8]]
  // CHECK: %[[C16:.*]] = llvm.mlir.constant(16 : i40) : i40
  // CHECK: %[[SHIFTED_16:.*]] = llvm.shl %[[VALUE_16]], %[[C16]]
  // CHECK: %[[VALUE_32:.*]] = llvm.or %[[VALUE_16]], %[[SHIFTED_16]]
  // CHECK: %[[C32:.*]] = llvm.mlir.constant(32 : i40) : i40
  // CHECK: %[[SHIFTED_COMPL:.*]] = llvm.shl %[[VALUE_32]], %[[C32]]
  // CHECK: %[[VALUE_COMPL:.*]] = llvm.or %[[VALUE_32]], %[[SHIFTED_COMPL]]
  // CHECK-NOT: "llvm.intr.memset"
  %2 = llvm.load %1 {alignment = 4 : i64} : !llvm.ptr -> i40
  // CHECK: llvm.return %[[VALUE_COMPL]] : i40
  llvm.return %2 : i40
}

// -----

// CHECK-LABEL: llvm.func @exotic_target_memset_inline
// CHECK-SAME: (%[[MEMSET_VALUE:.*]]: i8)
llvm.func @exotic_target_memset_inline(%memset_value: i8) -> i40 {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.alloca %0 x i40 {alignment = 4 : i64} : (i32) -> !llvm.ptr
  "llvm.intr.memset.inline"(%1, %memset_value) <{isVolatile = false, len = 5}> : (!llvm.ptr, i8) -> ()
  // CHECK-NOT: "llvm.intr.memset.inline"
  // CHECK: %[[VALUE_8:.*]] = llvm.zext %[[MEMSET_VALUE]] : i8 to i40
  // CHECK: %[[C8:.*]] = llvm.mlir.constant(8 : i40) : i40
  // CHECK: %[[SHIFTED_8:.*]] = llvm.shl %[[VALUE_8]], %[[C8]]
  // CHECK: %[[VALUE_16:.*]] = llvm.or %[[VALUE_8]], %[[SHIFTED_8]]
  // CHECK: %[[C16:.*]] = llvm.mlir.constant(16 : i40) : i40
  // CHECK: %[[SHIFTED_16:.*]] = llvm.shl %[[VALUE_16]], %[[C16]]
  // CHECK: %[[VALUE_32:.*]] = llvm.or %[[VALUE_16]], %[[SHIFTED_16]]
  // CHECK: %[[C32:.*]] = llvm.mlir.constant(32 : i40) : i40
  // CHECK: %[[SHIFTED_COMPL:.*]] = llvm.shl %[[VALUE_32]], %[[C32]]
  // CHECK: %[[VALUE_COMPL:.*]] = llvm.or %[[VALUE_32]], %[[SHIFTED_COMPL]]
  // CHECK-NOT: "llvm.intr.memset.inline"
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

// CHECK-LABEL: llvm.func @no_volatile_memset_inline
llvm.func @no_volatile_memset_inline() -> i32 {
  // CHECK-DAG: %[[ALLOCA_LEN:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-DAG: %[[ALLOCA:.*]] = llvm.alloca %[[ALLOCA_LEN]] x i32
  // CHECK-DAG: %[[MEMSET_VALUE:.*]] = llvm.mlir.constant(42 : i8) : i8
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
  %memset_value = llvm.mlir.constant(42 : i8) : i8
  // CHECK: "llvm.intr.memset.inline"(%[[ALLOCA]], %[[MEMSET_VALUE]]) <{isVolatile = true, len = 4 : i64}>
  "llvm.intr.memset.inline"(%1, %memset_value) <{isVolatile = true, len = 4}> : (!llvm.ptr, i8) -> ()
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

// CHECK-LABEL: llvm.func @no_partial_memset_inline
llvm.func @no_partial_memset_inline() -> i32 {
  // CHECK-DAG: %[[ALLOCA_LEN:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-DAG: %[[ALLOCA:.*]] = llvm.alloca %[[ALLOCA_LEN]] x i32
  // CHECK-DAG: %[[MEMSET_VALUE:.*]] = llvm.mlir.constant(42 : i8) : i8
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
  %memset_value = llvm.mlir.constant(42 : i8) : i8
  // CHECK: "llvm.intr.memset.inline"(%[[ALLOCA]], %[[MEMSET_VALUE]]) <{isVolatile = false, len = 2 : i64}>
  "llvm.intr.memset.inline"(%1, %memset_value) <{isVolatile = false, len = 2}> : (!llvm.ptr, i8) -> ()
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

// CHECK-LABEL: llvm.func @no_overflowing_memset_inline
llvm.func @no_overflowing_memset_inline() -> i32 {
  // CHECK-DAG: %[[ALLOCA_LEN:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-DAG: %[[ALLOCA:.*]] = llvm.alloca %[[ALLOCA_LEN]] x i32
  // CHECK-DAG: %[[MEMSET_VALUE:.*]] = llvm.mlir.constant(42 : i8) : i8
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
  %memset_value = llvm.mlir.constant(42 : i8) : i8
  // CHECK: "llvm.intr.memset.inline"(%[[ALLOCA]], %[[MEMSET_VALUE]]) <{isVolatile = false, len = 6 : i64}>
  "llvm.intr.memset.inline"(%1, %memset_value) <{isVolatile = false, len = 6}> : (!llvm.ptr, i8) -> ()
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

// -----

// CHECK-LABEL: llvm.func @only_byte_aligned_integers_memset_inline
llvm.func @only_byte_aligned_integers_memset_inline() -> i10 {
  // CHECK-DAG: %[[ALLOCA_LEN:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-DAG: %[[ALLOCA:.*]] = llvm.alloca %[[ALLOCA_LEN]] x i10
  // CHECK-DAG: %[[MEMSET_VALUE:.*]] = llvm.mlir.constant(42 : i8) : i8
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.alloca %0 x i10 {alignment = 4 : i64} : (i32) -> !llvm.ptr
  %memset_value = llvm.mlir.constant(42 : i8) : i8
  // CHECK: "llvm.intr.memset.inline"(%[[ALLOCA]], %[[MEMSET_VALUE]]) <{isVolatile = false, len = 2 : i64}>
  "llvm.intr.memset.inline"(%1, %memset_value) <{isVolatile = false, len = 2}> : (!llvm.ptr, i8) -> ()
  %2 = llvm.load %1 {alignment = 4 : i64} : !llvm.ptr -> i10
  llvm.return %2 : i10
}

// -----

// CHECK-LABEL: llvm.func @basic_memcpy
// CHECK-SAME: (%[[SOURCE:.*]]: !llvm.ptr)
llvm.func @basic_memcpy(%source: !llvm.ptr) -> i32 {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.alloca %0 x i32 : (i32) -> !llvm.ptr
  %is_volatile = llvm.mlir.constant(false) : i1
  %memcpy_len = llvm.mlir.constant(4 : i32) : i32
  "llvm.intr.memcpy"(%1, %source, %memcpy_len) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i32) -> ()
  // CHECK-NOT: "llvm.intr.memcpy"
  // CHECK: %[[LOADED:.*]] = llvm.load %[[SOURCE]] : !llvm.ptr -> i32
  // CHECK-NOT: "llvm.intr.memcpy"
  %2 = llvm.load %1 : !llvm.ptr -> i32
  // CHECK: llvm.return %[[LOADED]] : i32
  llvm.return %2 : i32
}

// -----

// CHECK-LABEL: llvm.func @basic_memcpy_dest
// CHECK-SAME: (%[[DESTINATION:.*]]: !llvm.ptr)
llvm.func @basic_memcpy_dest(%destination: !llvm.ptr) -> i32 {
  %0 = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %[[DATA:.*]] = llvm.mlir.constant(42 : i32) : i32
  %data = llvm.mlir.constant(42 : i32) : i32
  %is_volatile = llvm.mlir.constant(false) : i1
  %memcpy_len = llvm.mlir.constant(4 : i32) : i32

  %1 = llvm.alloca %0 x i32 : (i32) -> !llvm.ptr
  llvm.store %data, %1 : i32, !llvm.ptr
  "llvm.intr.memcpy"(%destination, %1, %memcpy_len) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i32) -> ()
  // CHECK-NOT: "llvm.intr.memcpy"
  // CHECK: llvm.store %[[DATA]], %[[DESTINATION]] : i32, !llvm.ptr
  // CHECK-NOT: "llvm.intr.memcpy"

  %2 = llvm.load %1 : !llvm.ptr -> i32
  // CHECK: llvm.return %[[DATA]] : i32
  llvm.return %2 : i32
}

// -----

// CHECK-LABEL: llvm.func @double_memcpy
llvm.func @double_memcpy() -> i32 {
  %0 = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %[[DATA:.*]] = llvm.mlir.constant(42 : i32) : i32
  %data = llvm.mlir.constant(42 : i32) : i32
  %is_volatile = llvm.mlir.constant(false) : i1
  %memcpy_len = llvm.mlir.constant(4 : i32) : i32

  %1 = llvm.alloca %0 x i32 : (i32) -> !llvm.ptr
  %2 = llvm.alloca %0 x i32 : (i32) -> !llvm.ptr
  llvm.store %data, %1 : i32, !llvm.ptr
  "llvm.intr.memcpy"(%2, %1, %memcpy_len) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i32) -> ()

  %res = llvm.load %2 : !llvm.ptr -> i32
  // CHECK: llvm.return %[[DATA]] : i32
  llvm.return %res : i32
}

// -----

// CHECK-LABEL: llvm.func @ignore_self_memcpy
llvm.func @ignore_self_memcpy() -> i32 {
  // CHECK-DAG: %[[ALLOCA_LEN:.*]] = llvm.mlir.constant(1 : i32) : i32
  %0 = llvm.mlir.constant(1 : i32) : i32
  %is_volatile = llvm.mlir.constant(false) : i1
  %memcpy_len = llvm.mlir.constant(4 : i32) : i32

  // CHECK-DAG: %[[ALLOCA:.*]] = llvm.alloca %[[ALLOCA_LEN]] x i32
  %1 = llvm.alloca %0 x i32 : (i32) -> !llvm.ptr
  // CHECK: "llvm.intr.memcpy"(%[[ALLOCA]], %[[ALLOCA]]
  "llvm.intr.memcpy"(%1, %1, %memcpy_len) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i32) -> ()

  %res = llvm.load %1 : !llvm.ptr -> i32
  llvm.return %res : i32
}

// -----

// CHECK-LABEL: llvm.func @ignore_partial_memcpy
// CHECK-SAME: (%[[SOURCE:.*]]: !llvm.ptr)
llvm.func @ignore_partial_memcpy(%source: !llvm.ptr) -> i32 {
  // CHECK-DAG: %[[ALLOCA_LEN:.*]] = llvm.mlir.constant(1 : i32) : i32
  %0 = llvm.mlir.constant(1 : i32) : i32
  %is_volatile = llvm.mlir.constant(false) : i1
  // CHECK-DAG: %[[MEMCPY_LEN:.*]] = llvm.mlir.constant(2 : i32) : i32
  %memcpy_len = llvm.mlir.constant(2 : i32) : i32

  // CHECK-DAG: %[[ALLOCA:.*]] = llvm.alloca %[[ALLOCA_LEN]] x i32
  %1 = llvm.alloca %0 x i32 : (i32) -> !llvm.ptr
  // CHECK: "llvm.intr.memcpy"(%[[ALLOCA]], %[[SOURCE]], %[[MEMCPY_LEN]]) <{isVolatile = false}>
  "llvm.intr.memcpy"(%1, %source, %memcpy_len) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i32) -> ()

  %res = llvm.load %1 : !llvm.ptr -> i32
  llvm.return %res : i32
}

// -----

// CHECK-LABEL: llvm.func @ignore_volatile_memcpy
// CHECK-SAME: (%[[SOURCE:.*]]: !llvm.ptr)
llvm.func @ignore_volatile_memcpy(%source: !llvm.ptr) -> i32 {
  // CHECK-DAG: %[[ALLOCA_LEN:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-DAG: %[[MEMCPY_LEN:.*]] = llvm.mlir.constant(4 : i32) : i32
  %0 = llvm.mlir.constant(1 : i32) : i32
  %is_volatile = llvm.mlir.constant(false) : i1
  %memcpy_len = llvm.mlir.constant(4 : i32) : i32

  // CHECK-DAG: %[[ALLOCA:.*]] = llvm.alloca %[[ALLOCA_LEN]] x i32
  %1 = llvm.alloca %0 x i32 : (i32) -> !llvm.ptr
  // CHECK: "llvm.intr.memcpy"(%[[ALLOCA]], %[[SOURCE]], %[[MEMCPY_LEN]]) <{isVolatile = true}>
  "llvm.intr.memcpy"(%1, %source, %memcpy_len) <{isVolatile = true}> : (!llvm.ptr, !llvm.ptr, i32) -> ()

  %res = llvm.load %1 : !llvm.ptr -> i32
  llvm.return %res : i32
}

// -----

// CHECK-LABEL: llvm.func @basic_memmove
// CHECK-SAME: (%[[SOURCE:.*]]: !llvm.ptr)
llvm.func @basic_memmove(%source: !llvm.ptr) -> i32 {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.alloca %0 x i32 : (i32) -> !llvm.ptr
  %is_volatile = llvm.mlir.constant(false) : i1
  %memmove_len = llvm.mlir.constant(4 : i32) : i32
  "llvm.intr.memmove"(%1, %source, %memmove_len) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i32) -> ()
  // CHECK-NOT: "llvm.intr.memmove"
  // CHECK: %[[LOADED:.*]] = llvm.load %[[SOURCE]] : !llvm.ptr -> i32
  // CHECK-NOT: "llvm.intr.memmove"
  %2 = llvm.load %1 : !llvm.ptr -> i32
  // CHECK: llvm.return %[[LOADED]] : i32
  llvm.return %2 : i32
}

// -----

// CHECK-LABEL: llvm.func @basic_memcpy_inline
// CHECK-SAME: (%[[SOURCE:.*]]: !llvm.ptr)
llvm.func @basic_memcpy_inline(%source: !llvm.ptr) -> i32 {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.alloca %0 x i32 : (i32) -> !llvm.ptr
  %is_volatile = llvm.mlir.constant(false) : i1
  "llvm.intr.memcpy.inline"(%1, %source) <{isVolatile = false, len = 4 : i32}> : (!llvm.ptr, !llvm.ptr) -> ()
  // CHECK-NOT: "llvm.intr.memcpy.inline"
  // CHECK: %[[LOADED:.*]] = llvm.load %[[SOURCE]] : !llvm.ptr -> i32
  // CHECK-NOT: "llvm.intr.memcpy.inline"
  %2 = llvm.load %1 : !llvm.ptr -> i32
  // CHECK: llvm.return %[[LOADED]] : i32
  llvm.return %2 : i32
}
