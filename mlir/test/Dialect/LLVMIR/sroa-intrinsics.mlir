// RUN: mlir-opt %s --pass-pipeline="builtin.module(llvm.func(sroa))" --split-input-file | FileCheck %s

// CHECK-LABEL: llvm.func @memset
llvm.func @memset() -> i32 {
  // CHECK-DAG: %[[ALLOCA_LEN:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-DAG: %[[ALLOCA:.*]] = llvm.alloca %[[ALLOCA_LEN]] x i32
  // CHECK-DAG: %[[MEMSET_VALUE:.*]] = llvm.mlir.constant(42 : i8) : i8
  // After SROA, only one i32 will be actually used, so only 4 bytes will be set.
  // CHECK-DAG: %[[MEMSET_LEN:.*]] = llvm.mlir.constant(4 : i32) : i32
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.alloca %0 x !llvm.array<10 x i32> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %memset_value = llvm.mlir.constant(42 : i8) : i8
  // 16 bytes means it will span over the first 4 i32 entries
  %memset_len = llvm.mlir.constant(16 : i32) : i32
  // CHECK: "llvm.intr.memset"(%[[ALLOCA]], %[[MEMSET_VALUE]], %[[MEMSET_LEN]]) <{isVolatile = false}>
  "llvm.intr.memset"(%1, %memset_value, %memset_len) <{isVolatile = false}> : (!llvm.ptr, i8, i32) -> ()
  %2 = llvm.getelementptr %1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<10 x i32>
  %3 = llvm.load %2 : !llvm.ptr -> i32
  llvm.return %3 : i32
}

// -----

// CHECK-LABEL: llvm.func @memset_partial
llvm.func @memset_partial() -> i32 {
  // CHECK-DAG: %[[ALLOCA_LEN:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-DAG: %[[ALLOCA:.*]] = llvm.alloca %[[ALLOCA_LEN]] x i32
  // CHECK-DAG: %[[MEMSET_VALUE:.*]] = llvm.mlir.constant(42 : i8) : i8
  // After SROA, only the second i32 will be actually used. As the memset writes up
  // to half of it, only 2 bytes will be set.
  // CHECK-DAG: %[[MEMSET_LEN:.*]] = llvm.mlir.constant(2 : i32) : i32
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.alloca %0 x !llvm.array<10 x i32> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %memset_value = llvm.mlir.constant(42 : i8) : i8
  // 6 bytes means it will span over the first i32 and half of the second i32.
  %memset_len = llvm.mlir.constant(6 : i32) : i32
  // CHECK: "llvm.intr.memset"(%[[ALLOCA]], %[[MEMSET_VALUE]], %[[MEMSET_LEN]]) <{isVolatile = false}>
  "llvm.intr.memset"(%1, %memset_value, %memset_len) <{isVolatile = false}> : (!llvm.ptr, i8, i32) -> ()
  %2 = llvm.getelementptr %1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<10 x i32>
  %3 = llvm.load %2 : !llvm.ptr -> i32
  llvm.return %3 : i32
}

// -----

// CHECK-LABEL: llvm.func @memset_full
llvm.func @memset_full() -> i32 {
  // CHECK-DAG: %[[ALLOCA_LEN:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-DAG: %[[ALLOCA:.*]] = llvm.alloca %[[ALLOCA_LEN]] x i32
  // CHECK-DAG: %[[MEMSET_VALUE:.*]] = llvm.mlir.constant(42 : i8) : i8
  // After SROA, only one i32 will be actually used, so only 4 bytes will be set.
  // CHECK-DAG: %[[MEMSET_LEN:.*]] = llvm.mlir.constant(4 : i32) : i32
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.alloca %0 x !llvm.array<10 x i32> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %memset_value = llvm.mlir.constant(42 : i8) : i8
  // 40 bytes means it will span over the entire array
  %memset_len = llvm.mlir.constant(40 : i32) : i32
  // CHECK: "llvm.intr.memset"(%[[ALLOCA]], %[[MEMSET_VALUE]], %[[MEMSET_LEN]]) <{isVolatile = false}>
  "llvm.intr.memset"(%1, %memset_value, %memset_len) <{isVolatile = false}> : (!llvm.ptr, i8, i32) -> ()
  %2 = llvm.getelementptr %1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<10 x i32>
  %3 = llvm.load %2 : !llvm.ptr -> i32
  llvm.return %3 : i32
}

// -----

// CHECK-LABEL: llvm.func @memset_too_much
llvm.func @memset_too_much() -> i32 {
  // CHECK-DAG: %[[ALLOCA_LEN:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-DAG: %[[ALLOCA:.*]] = llvm.alloca %[[ALLOCA_LEN]] x !llvm.array<10 x i32>
  // CHECK-DAG: %[[MEMSET_VALUE:.*]] = llvm.mlir.constant(42 : i8) : i8
  // CHECK-DAG: %[[MEMSET_LEN:.*]] = llvm.mlir.constant(41 : i32) : i32
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.alloca %0 x !llvm.array<10 x i32> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %memset_value = llvm.mlir.constant(42 : i8) : i8
  // 41 bytes means it will span over the entire array, and then some
  %memset_len = llvm.mlir.constant(41 : i32) : i32
  // CHECK: "llvm.intr.memset"(%[[ALLOCA]], %[[MEMSET_VALUE]], %[[MEMSET_LEN]]) <{isVolatile = false}>
  "llvm.intr.memset"(%1, %memset_value, %memset_len) <{isVolatile = false}> : (!llvm.ptr, i8, i32) -> ()
  %2 = llvm.getelementptr %1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<10 x i32>
  %3 = llvm.load %2 : !llvm.ptr -> i32
  llvm.return %3 : i32
}

// -----

// CHECK-LABEL: llvm.func @memset_no_volatile
llvm.func @memset_no_volatile() -> i32 {
  // CHECK-DAG: %[[ALLOCA_LEN:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-DAG: %[[ALLOCA:.*]] = llvm.alloca %[[ALLOCA_LEN]] x !llvm.array<10 x i32>
  // CHECK-DAG: %[[MEMSET_VALUE:.*]] = llvm.mlir.constant(42 : i8) : i8
  // CHECK-DAG: %[[MEMSET_LEN:.*]] = llvm.mlir.constant(16 : i32) : i32
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.alloca %0 x !llvm.array<10 x i32> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %memset_value = llvm.mlir.constant(42 : i8) : i8
  %memset_len = llvm.mlir.constant(16 : i32) : i32
  // CHECK: "llvm.intr.memset"(%[[ALLOCA]], %[[MEMSET_VALUE]], %[[MEMSET_LEN]]) <{isVolatile = true}>
  "llvm.intr.memset"(%1, %memset_value, %memset_len) <{isVolatile = true}> : (!llvm.ptr, i8, i32) -> ()
  %2 = llvm.getelementptr %1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<10 x i32>
  %3 = llvm.load %2 : !llvm.ptr -> i32
  llvm.return %3 : i32
}

// -----

// CHECK-LABEL: llvm.func @indirect_memset
llvm.func @indirect_memset() -> i32 {
  // CHECK-DAG: %[[ALLOCA_LEN:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-DAG: %[[ALLOCA:.*]] = llvm.alloca %[[ALLOCA_LEN]] x i32
  // CHECK-DAG: %[[MEMSET_VALUE:.*]] = llvm.mlir.constant(42 : i8) : i8
  // CHECK-DAG: %[[MEMSET_LEN:.*]] = llvm.mlir.constant(4 : i32) : i32
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.alloca %0 x !llvm.struct<"foo", (i32, i32)> : (i32) -> !llvm.ptr
  %memset_value = llvm.mlir.constant(42 : i8) : i8
  // This memset will only cover the selected element.
  %memset_len = llvm.mlir.constant(4 : i32) : i32
  %2 = llvm.getelementptr %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"foo", (i32, i32)>
  // CHECK: "llvm.intr.memset"(%[[ALLOCA]], %[[MEMSET_VALUE]], %[[MEMSET_LEN]]) <{isVolatile = false}>
  "llvm.intr.memset"(%2, %memset_value, %memset_len) <{isVolatile = false}> : (!llvm.ptr, i8, i32) -> ()
  %3 = llvm.load %2 : !llvm.ptr -> i32
  llvm.return %3 : i32
}

// -----

// CHECK-LABEL: llvm.func @invalid_indirect_memset
llvm.func @invalid_indirect_memset() -> i32 {
  // CHECK-DAG: %[[ALLOCA_LEN:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-DAG: %[[ALLOCA:.*]] = llvm.alloca %[[ALLOCA_LEN]] x !llvm.struct<"foo", (i32, i32)>
  // CHECK-DAG: %[[MEMSET_VALUE:.*]] = llvm.mlir.constant(42 : i8) : i8
  // CHECK-DAG: %[[MEMSET_LEN:.*]] = llvm.mlir.constant(6 : i32) : i32
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.alloca %0 x !llvm.struct<"foo", (i32, i32)> : (i32) -> !llvm.ptr
  %memset_value = llvm.mlir.constant(42 : i8) : i8
  // This memset will go slightly beyond one of the elements.
  %memset_len = llvm.mlir.constant(6 : i32) : i32
  // CHECK: %[[GEP:.*]] = llvm.getelementptr %[[ALLOCA]][0, 0]
  %2 = llvm.getelementptr %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"foo", (i32, i32)>
  // CHECK: "llvm.intr.memset"(%[[GEP]], %[[MEMSET_VALUE]], %[[MEMSET_LEN]]) <{isVolatile = false}>
  "llvm.intr.memset"(%2, %memset_value, %memset_len) <{isVolatile = false}> : (!llvm.ptr, i8, i32) -> ()
  %3 = llvm.load %2 : !llvm.ptr -> i32
  llvm.return %3 : i32
}

// -----

// CHECK-LABEL: llvm.func @memset_double_use
llvm.func @memset_double_use() -> i32 {
  // CHECK-DAG: %[[ALLOCA_LEN:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-DAG: %[[ALLOCA_INT:.*]] = llvm.alloca %[[ALLOCA_LEN]] x i32
  // CHECK-DAG: %[[ALLOCA_FLOAT:.*]] = llvm.alloca %[[ALLOCA_LEN]] x f32
  // CHECK-DAG: %[[MEMSET_VALUE:.*]] = llvm.mlir.constant(42 : i8) : i8
  // After SROA, only one i32 will be actually used, so only 4 bytes will be set.
  // CHECK-DAG: %[[MEMSET_LEN:.*]] = llvm.mlir.constant(4 : i32) : i32
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.alloca %0 x !llvm.struct<"foo", (i32, f32)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %memset_value = llvm.mlir.constant(42 : i8) : i8
  // 8 bytes means it will span over the two i32 entries.
  %memset_len = llvm.mlir.constant(8 : i32) : i32
  // We expect two generated memset, one for each field.
  // CHECK-NOT: "llvm.intr.memset"
  // CHECK-DAG: "llvm.intr.memset"(%[[ALLOCA_INT]], %[[MEMSET_VALUE]], %[[MEMSET_LEN]]) <{isVolatile = false}>
  // CHECK-DAG: "llvm.intr.memset"(%[[ALLOCA_FLOAT]], %[[MEMSET_VALUE]], %[[MEMSET_LEN]]) <{isVolatile = false}>
  // CHECK-NOT: "llvm.intr.memset"
  "llvm.intr.memset"(%1, %memset_value, %memset_len) <{isVolatile = false}> : (!llvm.ptr, i8, i32) -> ()
  %2 = llvm.getelementptr %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"foo", (i32, f32)>
  %3 = llvm.load %2 : !llvm.ptr -> i32
  %4 = llvm.getelementptr %1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"foo", (i32, f32)>
  %5 = llvm.load %4 : !llvm.ptr -> f32
  // We use this exotic bitcast to use the f32 easily. Semantics do not matter here.
  %6 = llvm.bitcast %5 : f32 to i32
  %7 = llvm.add %3, %6 : i32
  llvm.return %7 : i32
}

// -----

// CHECK-LABEL: llvm.func @memset_considers_alignment
llvm.func @memset_considers_alignment() -> i32 {
  // CHECK-DAG: %[[ALLOCA_LEN:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-DAG: %[[ALLOCA_INT:.*]] = llvm.alloca %[[ALLOCA_LEN]] x i32
  // CHECK-DAG: %[[MEMSET_VALUE:.*]] = llvm.mlir.constant(42 : i8) : i8
  // After SROA, only 32-bit values will be actually used, so only 4 bytes will be set.
  // CHECK-DAG: %[[MEMSET_LEN:.*]] = llvm.mlir.constant(4 : i32) : i32
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.alloca %0 x !llvm.struct<"foo", (i8, i32, f32)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %memset_value = llvm.mlir.constant(42 : i8) : i8
  // 8 bytes means it will span over the i8 and the i32 entry.
  // Because of padding, the f32 entry will not be touched.
  %memset_len = llvm.mlir.constant(8 : i32) : i32
  // Even though the two i32 are used, only one memset should be generated,
  // as the second i32 is not touched by the initial memset.
  // CHECK-NOT: "llvm.intr.memset"
  // CHECK: "llvm.intr.memset"(%[[ALLOCA_INT]], %[[MEMSET_VALUE]], %[[MEMSET_LEN]]) <{isVolatile = false}>
  // CHECK-NOT: "llvm.intr.memset"
  "llvm.intr.memset"(%1, %memset_value, %memset_len) <{isVolatile = false}> : (!llvm.ptr, i8, i32) -> ()
  %2 = llvm.getelementptr %1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"foo", (i8, i32, f32)>
  %3 = llvm.load %2 : !llvm.ptr -> i32
  %4 = llvm.getelementptr %1[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"foo", (i8, i32, f32)>
  %5 = llvm.load %4 : !llvm.ptr -> f32
  // We use this exotic bitcast to use the f32 easily. Semantics do not matter here.
  %6 = llvm.bitcast %5 : f32 to i32
  %7 = llvm.add %3, %6 : i32
  llvm.return %7 : i32
}

// -----

// CHECK-LABEL: llvm.func @memset_considers_packing
llvm.func @memset_considers_packing() -> i32 {
  // CHECK-DAG: %[[ALLOCA_LEN:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-DAG: %[[ALLOCA_INT:.*]] = llvm.alloca %[[ALLOCA_LEN]] x i32
  // CHECK-DAG: %[[ALLOCA_FLOAT:.*]] = llvm.alloca %[[ALLOCA_LEN]] x f32
  // CHECK-DAG: %[[MEMSET_VALUE:.*]] = llvm.mlir.constant(42 : i8) : i8
  // After SROA, only 32-bit values will be actually used, so only 4 bytes will be set.
  // CHECK-DAG: %[[MEMSET_LEN_WHOLE:.*]] = llvm.mlir.constant(4 : i32) : i32
  // CHECK-DAG: %[[MEMSET_LEN_PARTIAL:.*]] = llvm.mlir.constant(3 : i32) : i32
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.alloca %0 x !llvm.struct<"foo", packed (i8, i32, f32)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %memset_value = llvm.mlir.constant(42 : i8) : i8
  // 8 bytes means it will span over all the fields, because there is no padding as the struct is packed.
  %memset_len = llvm.mlir.constant(8 : i32) : i32
  // Now all fields are touched by the memset.
  // CHECK-NOT: "llvm.intr.memset"
  // CHECK: "llvm.intr.memset"(%[[ALLOCA_INT]], %[[MEMSET_VALUE]], %[[MEMSET_LEN_WHOLE]]) <{isVolatile = false}>
  // CHECK: "llvm.intr.memset"(%[[ALLOCA_FLOAT]], %[[MEMSET_VALUE]], %[[MEMSET_LEN_PARTIAL]]) <{isVolatile = false}>
  // CHECK-NOT: "llvm.intr.memset"
  "llvm.intr.memset"(%1, %memset_value, %memset_len) <{isVolatile = false}> : (!llvm.ptr, i8, i32) -> ()
  %2 = llvm.getelementptr %1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"foo", packed (i8, i32, f32)>
  %3 = llvm.load %2 : !llvm.ptr -> i32
  %4 = llvm.getelementptr %1[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"foo", packed (i8, i32, f32)>
  %5 = llvm.load %4 : !llvm.ptr -> f32
  // We use this exotic bitcast to use the f32 easily. Semantics do not matter here.
  %6 = llvm.bitcast %5 : f32 to i32
  %7 = llvm.add %3, %6 : i32
  llvm.return %7 : i32
}
