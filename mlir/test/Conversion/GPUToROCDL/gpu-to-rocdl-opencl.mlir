// RUN: mlir-opt %s -convert-gpu-to-rocdl=runtime=OpenCL | FileCheck %s

gpu.module @test_module {
  // CHECK: llvm.mlir.global internal constant @[[$PRINT_GLOBAL:[A-Za-z0-9_]+]]("Hello: %d\0A\00")  {addr_space = 4 : i32}
  // CHECK: llvm.func @printf(!llvm.ptr<i8, 4>, ...) -> i32
  // CHECK-LABEL: func @test_printf
  // CHECK: (%[[ARG0:.*]]: i32)
  gpu.func @test_printf(%arg0: i32) {
    // CHECK: %[[IMM0:.*]] = llvm.mlir.addressof @[[$PRINT_GLOBAL]] : !llvm.ptr<array<11 x i8>, 4>
    // CHECK-NEXT: %[[IMM1:.*]] = llvm.mlir.constant(0 : i64) : i64
    // CHECK-NEXT: %[[IMM2:.*]] = llvm.getelementptr %[[IMM0]][%[[IMM1]], %[[IMM1]]] : (!llvm.ptr<array<11 x i8>, 4>, i64, i64) -> !llvm.ptr<i8, 4>
    // CHECK-NEXT: %{{.*}} = llvm.call @printf(%[[IMM2]], %[[ARG0]]) : (!llvm.ptr<i8, 4>, i32) -> i32
    gpu.printf "Hello: %d\n" %arg0 : i32
    gpu.return
  }
}
