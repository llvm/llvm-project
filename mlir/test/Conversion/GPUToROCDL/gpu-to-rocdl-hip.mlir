// RUN: mlir-opt %s -convert-gpu-to-rocdl='runtime=HIP use-opaque-pointers=1' -split-input-file | FileCheck %s

gpu.module @test_module {
  // CHECK-DAG: llvm.mlir.global internal constant @[[$PRINT_GLOBAL0:[A-Za-z0-9_]+]]("Hello, world\0A\00")
  // CHECK-DAG: llvm.mlir.global internal constant @[[$PRINT_GLOBAL1:[A-Za-z0-9_]+]]("Hello: %d\0A\00")
  // CHECK-DAG: llvm.func @__ockl_printf_append_args(i64, i32, i64, i64, i64, i64, i64, i64, i64, i32) -> i64
  // CHECK-DAG: llvm.func @__ockl_printf_append_string_n(i64, !llvm.ptr, i64, i32) -> i64
  // CHECK-DAG: llvm.func @__ockl_printf_begin(i64) -> i64

  // CHECK-LABEL: func @test_const_printf
  gpu.func @test_const_printf() {
    // CHECK: %[[CST0:.*]] = llvm.mlir.constant(0 : i64) : i64
    // CHECK-NEXT: %[[DESC0:.*]] = llvm.call @__ockl_printf_begin(%0) : (i64) -> i64
    // CHECK-NEXT: %[[FORMATSTR:.*]] = llvm.mlir.addressof @[[$PRINT_GLOBAL0]] : !llvm.ptr
    // CHECK-NEXT: %[[FORMATSTART:.*]] = llvm.getelementptr %[[FORMATSTR]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<14 x i8>
    // CHECK-NEXT: %[[FORMATLEN:.*]] = llvm.mlir.constant(14 : i64) : i64
    // CHECK-NEXT: %[[ISLAST:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-NEXT: %[[ISNTLAST:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %{{.*}} = llvm.call @__ockl_printf_append_string_n(%[[DESC0]], %[[FORMATSTART]], %[[FORMATLEN]], %[[ISLAST]]) : (i64, !llvm.ptr, i64, i32) -> i64
    gpu.printf "Hello, world\n"
    gpu.return
  }


  // CHECK-LABEL: func @test_printf
  // CHECK: (%[[ARG0:.*]]: i32)
  gpu.func @test_printf(%arg0: i32) {
    // CHECK: %[[CST0:.*]] = llvm.mlir.constant(0 : i64) : i64
    // CHECK-NEXT: %[[DESC0:.*]] = llvm.call @__ockl_printf_begin(%0) : (i64) -> i64
    // CHECK-NEXT: %[[FORMATSTR:.*]] = llvm.mlir.addressof @[[$PRINT_GLOBAL1]] : !llvm.ptr
    // CHECK-NEXT: %[[FORMATSTART:.*]] = llvm.getelementptr %[[FORMATSTR]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<11 x i8>
    // CHECK-NEXT: %[[FORMATLEN:.*]] = llvm.mlir.constant(11 : i64) : i64
    // CHECK-NEXT: %[[ISLAST:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-NEXT: %[[ISNTLAST:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %[[DESC1:.*]] = llvm.call @__ockl_printf_append_string_n(%[[DESC0]], %[[FORMATSTART]], %[[FORMATLEN]], %[[ISNTLAST]]) : (i64, !llvm.ptr, i64, i32) -> i64
    // CHECK-NEXT: %[[NARGS1:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-NEXT: %[[ARG0_64:.*]] = llvm.zext %[[ARG0]] : i32 to i64
    // CHECK-NEXT: %{{.*}} = llvm.call @__ockl_printf_append_args(%[[DESC1]], %[[NARGS1]], %[[ARG0_64]], %[[CST0]], %[[CST0]], %[[CST0]], %[[CST0]], %[[CST0]], %[[CST0]], %[[ISLAST]]) : (i64, i32, i64, i64, i64, i64, i64, i64, i64, i32) -> i64
    gpu.printf "Hello: %d\n" %arg0 : i32
    gpu.return
  }
}
