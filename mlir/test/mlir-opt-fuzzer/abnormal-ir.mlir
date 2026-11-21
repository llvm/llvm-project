// RUN: mlir-opt-fuzzer %s -ignore_remaining_args=1 --pipeline=cse 2>&1 | FileCheck %s

module {
  func.func @f() -> i1 {
    return
  }
}

// This module parses but fails verification: the diagnostics reject it and
// it is silently ignored - verifier-invalid input is the normal case for
// fuzzing. What this fuzzer looks for is input that passes verification and
// then crashes or otherwise misbehaves.
// CHECK: Running
// CHECK-NOT: LLVM ERROR
// CHECK-NOT: post-pipeline verification failed
