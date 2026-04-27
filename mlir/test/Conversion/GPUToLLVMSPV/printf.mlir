// RUN: mlir-opt %s -convert-gpu-to-llvm-spv | FileCheck %s

gpu.module @test_module {
  // CHECK: llvm.mlir.global internal constant @[[$PRINT_GLOBAL:[A-Za-z0-9_]+]]("Hello: %d\0A\00")  {addr_space = 2 : i32}
  // CHECK: llvm.func spir_funccc @_Z6printfPU3AS2Kcz(!llvm.ptr<2>, ...) -> i32
  // CHECK-LABEL: llvm.func spir_funccc @test_printf
  // CHECK: (%[[ARG0:.*]]: i32)
  gpu.func @test_printf(%arg0: i32) {
    // CHECK: %[[IMM0:.*]] = llvm.mlir.addressof @[[$PRINT_GLOBAL]] : !llvm.ptr<2>
    // CHECK-NEXT: %[[IMM2:.*]] = llvm.getelementptr %[[IMM0]][0, 0] : (!llvm.ptr<2>) -> !llvm.ptr<2>, !llvm.array<11 x i8>
    // CHECK-NEXT: %{{.*}} = llvm.call spir_funccc @_Z6printfPU3AS2Kcz(%[[IMM2]], %[[ARG0]]) vararg(!llvm.func<i32 (ptr<2>, ...)>) : (!llvm.ptr<2>, i32) -> i32
    gpu.printf "Hello: %d\n", %arg0 : i32
    gpu.return
  }
}

