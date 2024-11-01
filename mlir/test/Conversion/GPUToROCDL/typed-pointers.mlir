// RUN: mlir-opt %s -convert-gpu-to-rocdl="runtime=HIP use-opaque-pointers=0" -split-input-file | FileCheck --check-prefixes=CHECK,HIP %s
// RUN: mlir-opt %s -convert-gpu-to-rocdl="runtime=OpenCL use-opaque-pointers=0" | FileCheck --check-prefixes=CHECK,OCL %s

gpu.module @test_module {
  // HIP-DAG: llvm.mlir.global internal constant @[[$PRINT_GLOBAL1:[A-Za-z0-9_]+]]("Hello: %d\0A\00")
  // HIP-DAG: llvm.func @__ockl_printf_append_args(i64, i32, i64, i64, i64, i64, i64, i64, i64, i32) -> i64
  // HIP-DAG: llvm.func @__ockl_printf_append_string_n(i64, !llvm.ptr<i8>, i64, i32) -> i64
  // HIP-DAG: llvm.func @__ockl_printf_begin(i64) -> i64

  // OCL: llvm.mlir.global internal constant @[[$PRINT_GLOBAL:[A-Za-z0-9_]+]]("Hello: %d\0A\00")  {addr_space = 4 : i32}
  // OCL: llvm.func @printf(!llvm.ptr<i8, 4>, ...) -> i32
  // CHECK-LABEL: func @test_printf
  // CHECK: (%[[ARG0:.*]]: i32)
  gpu.func @test_printf(%arg0: i32) {
    // OCL: %[[IMM0:.*]] = llvm.mlir.addressof @[[$PRINT_GLOBAL]] : !llvm.ptr<array<11 x i8>, 4>
    // OCL-NEXT: %[[IMM2:.*]] = llvm.getelementptr %[[IMM0]][0, 0] : (!llvm.ptr<array<11 x i8>, 4>) -> !llvm.ptr<i8, 4>
    // OCL-NEXT: %{{.*}} = llvm.call @printf(%[[IMM2]], %[[ARG0]]) : (!llvm.ptr<i8, 4>, i32) -> i32

    // HIP: %[[CST0:.*]] = llvm.mlir.constant(0 : i64) : i64
    // HIP-NEXT: %[[DESC0:.*]] = llvm.call @__ockl_printf_begin(%0) : (i64) -> i64
    // HIP-NEXT: %[[FORMATSTR:.*]] = llvm.mlir.addressof @[[$PRINT_GLOBAL1]] : !llvm.ptr<array<11 x i8>>
    // HIP-NEXT: %[[FORMATSTART:.*]] = llvm.getelementptr %[[FORMATSTR]][0, 0] : (!llvm.ptr<array<11 x i8>>) -> !llvm.ptr<i8>
    // HIP-NEXT: %[[FORMATLEN:.*]] = llvm.mlir.constant(11 : i64) : i64
    // HIP-NEXT: %[[ISLAST:.*]] = llvm.mlir.constant(1 : i32) : i32
    // HIP-NEXT: %[[ISNTLAST:.*]] = llvm.mlir.constant(0 : i32) : i32
    // HIP-NEXT: %[[DESC1:.*]] = llvm.call @__ockl_printf_append_string_n(%[[DESC0]], %[[FORMATSTART]], %[[FORMATLEN]], %[[ISNTLAST]]) : (i64, !llvm.ptr<i8>, i64, i32) -> i64
    // HIP-NEXT: %[[NARGS1:.*]] = llvm.mlir.constant(1 : i32) : i32
    // HIP-NEXT: %[[ARG0_64:.*]] = llvm.zext %[[ARG0]] : i32 to i64
    // HIP-NEXT: %{{.*}} = llvm.call @__ockl_printf_append_args(%[[DESC1]], %[[NARGS1]], %[[ARG0_64]], %[[CST0]], %[[CST0]], %[[CST0]], %[[CST0]], %[[CST0]], %[[CST0]], %[[ISLAST]]) : (i64, i32, i64, i64, i64, i64, i64, i64, i64, i32) -> i64

    gpu.printf "Hello: %d\n" %arg0 : i32
    gpu.return
  }
}
