// RUN: mlir-opt %s --pass-pipeline="builtin.module(llvm.func(sroa))" --split-input-file | FileCheck %s

// CHECK-LABEL: llvm.func @basic_struct
llvm.func @basic_struct() -> i32 {
  // CHECK: %[[SIZE:.*]] = llvm.mlir.constant(1 : i32)
  %0 = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %[[ALLOCA:.*]] = llvm.alloca %[[SIZE]] x i32
  %1 = llvm.alloca %0 x !llvm.struct<"foo", (i32, f64, i32)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %2 = llvm.getelementptr inbounds %1[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"foo", (i32, f64, i32)>
  // CHECK: %[[RES:.*]] = llvm.load %[[ALLOCA]]
  %3 = llvm.load %2 : !llvm.ptr -> i32
  // CHECK: llvm.return %[[RES]] : i32
  llvm.return %3 : i32
}

// -----

// CHECK-LABEL: llvm.func @basic_array
llvm.func @basic_array() -> i32 {
  // CHECK: %[[SIZE:.*]] = llvm.mlir.constant(1 : i32)
  %0 = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %[[ALLOCA:.*]] = llvm.alloca %[[SIZE]] x i32
  %1 = llvm.alloca %0 x !llvm.array<10 x i32> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %2 = llvm.getelementptr inbounds %1[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<10 x i32>
  // CHECK: %[[RES:.*]] = llvm.load %[[ALLOCA]]
  %3 = llvm.load %2 : !llvm.ptr -> i32
  // CHECK: llvm.return %[[RES]] : i32
  llvm.return %3 : i32
}

// -----

// CHECK-LABEL: llvm.func @multi_level_direct
llvm.func @multi_level_direct() -> i32 {
  // CHECK: %[[SIZE:.*]] = llvm.mlir.constant(1 : i32)
  %0 = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %[[ALLOCA:.*]] = llvm.alloca %[[SIZE]] x i32
  %1 = llvm.alloca %0 x !llvm.struct<"foo", (i32, f64, struct<"bar", (i8, array<10 x array<10 x i32>>, i8)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %2 = llvm.getelementptr inbounds %1[0, 2, 1, 5, 8] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"foo", (i32, f64, struct<"bar", (i8, array<10 x array<10 x i32>>, i8)>)>
  // CHECK: %[[RES:.*]] = llvm.load %[[ALLOCA]]
  %3 = llvm.load %2 : !llvm.ptr -> i32
  // CHECK: llvm.return %[[RES]] : i32
  llvm.return %3 : i32
}

// -----

// The first application of SROA would generate a GEP with indices [0, 0]. This
// test ensures this GEP is not eliminated during the first application. Even
// though doing it would be correct, it would prevent the second application
// of SROA to eliminate the array. GEPs should be eliminated only when they are
// truly trivial (with indices [0]).

// CHECK-LABEL: llvm.func @multi_level_direct_two_applications
llvm.func @multi_level_direct_two_applications() -> i32 {
  // CHECK: %[[SIZE:.*]] = llvm.mlir.constant(1 : i32)
  %0 = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %[[ALLOCA:.*]] = llvm.alloca %[[SIZE]] x i32
  %1 = llvm.alloca %0 x !llvm.struct<"foo", (i32, f64, array<10 x i32>, i8)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %2 = llvm.getelementptr inbounds %1[0, 2, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"foo", (i32, f64, array<10 x i32>, i8)>
  // CHECK: %[[RES:.*]] = llvm.load %[[ALLOCA]]
  %3 = llvm.load %2 : !llvm.ptr -> i32
  // CHECK: llvm.return %[[RES]] : i32
  llvm.return %3 : i32
}

// -----

// CHECK-LABEL: llvm.func @multi_level_indirect
llvm.func @multi_level_indirect() -> i32 {
  // CHECK: %[[SIZE:.*]] = llvm.mlir.constant(1 : i32)
  %0 = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %[[ALLOCA:.*]] = llvm.alloca %[[SIZE]] x i32
  %1 = llvm.alloca %0 x !llvm.struct<"foo", (i32, f64, struct<"bar", (i8, array<10 x array<10 x i32>>, i8)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %2 = llvm.getelementptr inbounds %1[0, 2, 1, 5] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"foo", (i32, f64, struct<"bar", (i8, array<10 x array<10 x i32>>, i8)>)>
  %3 = llvm.getelementptr inbounds %2[0, 8] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<10 x i32>
  // CHECK: %[[RES:.*]] = llvm.load %[[ALLOCA]]
  %4 = llvm.load %3 : !llvm.ptr -> i32
  // CHECK: llvm.return %[[RES]] : i32
  llvm.return %4 : i32
}

// -----

// CHECK-LABEL: llvm.func @resolve_alias
// CHECK-SAME: (%[[ARG:.*]]: i32)
llvm.func @resolve_alias(%arg: i32) -> i32 {
  // CHECK: %[[SIZE:.*]] = llvm.mlir.constant(1 : i32)
  %0 = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %[[ALLOCA:.*]] = llvm.alloca %[[SIZE]] x i32
  %1 = llvm.alloca %0 x !llvm.struct<"foo", (i32, f64, i32)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %2 = llvm.getelementptr %1[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"foo", (i32, f64, i32)>
  %3 = llvm.getelementptr inbounds %1[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"foo", (i32, f64, i32)>
  // CHECK: llvm.store %[[ARG]], %[[ALLOCA]]
  llvm.store %arg, %2 : i32, !llvm.ptr
  // CHECK: %[[RES:.*]] = llvm.load %[[ALLOCA]]
  %4 = llvm.load %3 : !llvm.ptr -> i32
  // CHECK: llvm.return %[[RES]] : i32
  llvm.return %4 : i32
}

// -----

// CHECK-LABEL: llvm.func @no_non_single_support
llvm.func @no_non_single_support() -> i32 {
  // CHECK: %[[SIZE:.*]] = llvm.mlir.constant
  %0 = llvm.mlir.constant(2 : i32) : i32
  // CHECK: %[[ALLOCA:.*]] = llvm.alloca %[[SIZE]] x !llvm.struct<"foo", (i32, f64, i32)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %1 = llvm.alloca %0 x !llvm.struct<"foo", (i32, f64, i32)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  // CHECK-NOT: = llvm.alloca
  %2 = llvm.getelementptr inbounds %1[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"foo", (i32, f64, i32)>
  %3 = llvm.load %2 : !llvm.ptr -> i32
  llvm.return %3 : i32
}

// -----

// CHECK-LABEL: llvm.func @no_pointer_indexing
llvm.func @no_pointer_indexing() -> i32 {
  // CHECK: %[[SIZE:.*]] = llvm.mlir.constant(1 : i32)
  %0 = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %[[ALLOCA:.*]] = llvm.alloca %[[SIZE]] x !llvm.struct<"foo", (i32, f64, i32)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %1 = llvm.alloca %0 x !llvm.struct<"foo", (i32, f64, i32)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  // CHECK-NOT: = llvm.alloca
  %2 = llvm.getelementptr %1[1, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"foo", (i32, f64, i32)>
  %3 = llvm.load %2 : !llvm.ptr -> i32
  llvm.return %3 : i32
}

// -----

// CHECK-LABEL: llvm.func @no_direct_use
llvm.func @no_direct_use() -> i32 {
  // CHECK: %[[SIZE:.*]] = llvm.mlir.constant(1 : i32)
  %0 = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %[[ALLOCA:.*]] = llvm.alloca %[[SIZE]] x !llvm.struct<"foo", (i32, f64, i32)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %1 = llvm.alloca %0 x !llvm.struct<"foo", (i32, f64, i32)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  // CHECK-NOT: = llvm.alloca
  %2 = llvm.getelementptr %1[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"foo", (i32, f64, i32)>
  %3 = llvm.load %2 : !llvm.ptr -> i32
  llvm.call @use(%1) : (!llvm.ptr) -> ()
  llvm.return %3 : i32
}

llvm.func @use(!llvm.ptr)

// -----

// CHECK-LABEL: llvm.func @direct_promotable_use_is_fine
llvm.func @direct_promotable_use_is_fine() -> i32 {
  // CHECK: %[[SIZE:.*]] = llvm.mlir.constant(1 : i32)
  %0 = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %[[ALLOCA:.*]] = llvm.alloca %[[SIZE]] x i32
  %1 = llvm.alloca %0 x !llvm.struct<"foo", (i32, f64, i32)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %2 = llvm.getelementptr %1[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"foo", (i32, f64, i32)>
  // CHECK: %[[RES:.*]] = llvm.load %[[ALLOCA]]
  %3 = llvm.load %2 : !llvm.ptr -> i32
  // This is a direct use of the slot but it can be removed because it implements PromotableOpInterface.
  llvm.intr.lifetime.start 2, %1 : !llvm.ptr
  // CHECK: llvm.return %[[RES]] : i32
  llvm.return %3 : i32
}

// -----

// CHECK-LABEL: llvm.func @direct_promotable_use_is_fine_on_accessor
llvm.func @direct_promotable_use_is_fine_on_accessor() -> i32 {
  // CHECK: %[[SIZE:.*]] = llvm.mlir.constant(1 : i32)
  %0 = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %[[ALLOCA:.*]] = llvm.alloca %[[SIZE]] x i32
  %1 = llvm.alloca %0 x !llvm.struct<"foo", (i32, f64, i32)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %2 = llvm.getelementptr %1[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"foo", (i32, f64, i32)>
  // CHECK: %[[RES:.*]] = llvm.load %[[ALLOCA]]
  %3 = llvm.load %2 : !llvm.ptr -> i32
  // This does not provide side-effect info but it can be removed because it implements PromotableOpInterface.
  llvm.intr.lifetime.start 2, %2 : !llvm.ptr
  // CHECK: llvm.return %[[RES]] : i32
  llvm.return %3 : i32
}

// -----

// CHECK-LABEL: llvm.func @no_dynamic_indexing
// CHECK-SAME: (%[[ARG:.*]]: i32)
llvm.func @no_dynamic_indexing(%arg: i32) -> i32 {
  // CHECK: %[[SIZE:.*]] = llvm.mlir.constant(1 : i32)
  %0 = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %[[ALLOCA:.*]] = llvm.alloca %[[SIZE]] x !llvm.array<10 x i32> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %1 = llvm.alloca %0 x !llvm.array<10 x i32> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  // CHECK-NOT: = llvm.alloca
  // CHECK: %[[GEP:.*]] = llvm.getelementptr %[[ALLOCA]][0, %[[ARG]]]
  %2 = llvm.getelementptr %1[0, %arg] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.array<10 x i32>
  // CHECK: %[[RES:.*]] = llvm.load %[[GEP]]
  %3 = llvm.load %2 : !llvm.ptr -> i32
  // CHECK: llvm.return %[[RES]] : i32
  llvm.return %3 : i32
}

// -----

// CHECK-LABEL: llvm.func @no_typed_pointers
llvm.func @no_typed_pointers() -> i32 {
  // CHECK: %[[SIZE:.*]] = llvm.mlir.constant(1 : i32)
  %0 = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %[[ALLOCA:.*]] = llvm.alloca %[[SIZE]] x !llvm.array<10 x i32> {alignment = 8 : i64} : (i32) -> !llvm.ptr<array<10 x i32>>
  %1 = llvm.alloca %0 x !llvm.array<10 x i32> {alignment = 8 : i64} : (i32) -> !llvm.ptr<array<10 x i32>>
  // CHECK-NOT: = llvm.alloca
  %2 = llvm.getelementptr %1[0, 1] : (!llvm.ptr<array<10 x i32>>) -> !llvm.ptr<i32>
  %3 = llvm.load %2 : !llvm.ptr<i32>
  llvm.return %3 : i32
}
