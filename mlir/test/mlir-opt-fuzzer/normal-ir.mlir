// RUN: mlir-opt-fuzzer %s -ignore_remaining_args=1 --pipeline=cse 2>&1 | FileCheck %s
// RUN: mlir-opt-fuzzer %s -ignore_remaining_args=1 --pipeline=canonicalize 2>&1 | FileCheck %s
// RUN: mlir-opt-fuzzer %s -ignore_remaining_args=1 --pipeline=remove-dead-values 2>&1 | FileCheck %s
// RUN: mlir-opt-fuzzer %s -ignore_remaining_args=1 --pipeline=rdv-privatize-cse 2>&1 | FileCheck %s

module {
  func.func @f(%arg0: i32) -> i32 {
    %0 = arith.constant 1 : i32
    %1 = arith.addi %arg0, %0 : i32
    return %1 : i32
  }
}

// CHECK: Running
// CHECK-NOT: LLVM ERROR
