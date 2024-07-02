// RUN: mlir-opt %s --convert-libc-to-llvm | FileCheck %s

module {
  // CHECK: llvm.mlir.global internal constant @cpuprintfFormat_0("Hello world %f %d %lld\0A\00") {addr_space = 0 : i32}
  // CHECK: llvm.func @printf(!llvm.ptr,
  // CHECK-NEXT: func.func @doprint(%[[ARG0:.*]]: f32, %[[ARG1:.*]]: i32, %[[ARG2:.*]]: i64)
  func.func @doprint(%t: f32, %t2: i32, %t3: i64) {
    // CHECK-NEXT: llvm.mlir.addressof
    // CHECK-DAG: %[[C1:.*]] = llvm.getelementptr
    // CHECK-SAME: !llvm.ptr, !llvm.array<24 x i8>
    // CHECK: %[[C2:.*]] = llvm.fpext %[[ARG0]] 
    // CHECK: %[[C3:.*]] = llvm.zext %[[ARG1]] 
    // CHECK-NOT: libc.printf
    // CHECK-NEXT: llvm.call @printf(%[[C1]], %[[C2]], %[[C3]], %[[ARG2]])
    libc.printf "Hello world %f %d %lld\n" %t, %t2, %t3 : f32, i32, i64
    return
  }

}