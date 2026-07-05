// REQUIRES: asserts
// RUN: mlir-opt %s --pass-pipeline="builtin.module(llvm.func(sroa))" --split-input-file --mlir-pass-statistics 2>&1 >/dev/null | FileCheck %s

// CHECK: SROA
// CHECK-NEXT: (S) 1 destructured slots
// CHECK-NEXT: (S) 2 max subelement number
// CHECK-NEXT: (S) 1 slots with memory benefit
llvm.func @basic() -> i32 {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.alloca %0 x !llvm.struct<"foo", (i32, i32)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %2 = llvm.getelementptr inbounds %1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"foo", (i32, i32)>
  %3 = llvm.load %2 : !llvm.ptr -> i32
  llvm.return %3 : i32
}

// -----

// CHECK: SROA
// CHECK-NEXT: (S) 1 destructured slots
// CHECK-NEXT: (S) 2 max subelement number
// CHECK-NEXT: (S) 0 slots with memory benefit
llvm.func @basic_no_memory_benefit() -> i32 {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.alloca %0 x !llvm.struct<"foo", (i32, i32)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %2 = llvm.getelementptr inbounds %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"foo", (i32, i32)>
  %3 = llvm.getelementptr inbounds %1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"foo", (i32, i32)>
  %4 = llvm.load %2 : !llvm.ptr -> i32
  %5 = llvm.load %3 : !llvm.ptr -> i32
  %6 = llvm.add %4, %5 : i32
  llvm.return %6 : i32
}

// -----

// CHECK: SROA
// CHECK-NEXT: (S)  1 destructured slots
// CHECK-NEXT: (S) 10 max subelement number
// CHECK-NEXT: (S)  1 slots with memory benefit
llvm.func @basic_array() -> i32 {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.alloca %0 x !llvm.array<10 x i32> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %2 = llvm.getelementptr inbounds %1[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<10 x i32>
  %3 = llvm.load %2 : !llvm.ptr -> i32
  llvm.return %3 : i32
}

// -----

// SROA is applied repeatedly here, peeling off layers of aggregates one after
// the other, four times.

// CHECK: SROA
// CHECK-NEXT: (S)  4 destructured slots
// CHECK-NEXT: (S) 10 max subelement number
// CHECK-NEXT: (S)  4 slots with memory benefit
llvm.func @multi_level_direct() -> i32 {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.alloca %0 x !llvm.struct<"foo", (i32, f64, struct<"bar", (i8, array<10 x array<10 x i32>>, i8)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %2 = llvm.getelementptr inbounds %1[0, 2, 1, 5, 8] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"foo", (i32, f64, struct<"bar", (i8, array<10 x array<10 x i32>>, i8)>)>
  %3 = llvm.load %2 : !llvm.ptr -> i32
  llvm.return %3 : i32
}
