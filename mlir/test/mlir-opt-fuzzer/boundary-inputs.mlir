// RUN: mlir-opt-fuzzer %s -ignore_remaining_args=1 --pipeline=cse 2>&1 | FileCheck %s
// RUN: echo 'this is not mlir' > %t
// RUN: mlir-opt-fuzzer %t -ignore_remaining_args=1 --pipeline=cse 2>&1 | FileCheck %s

// The parse diagnostic is swallowed by the fuzz target on purpose.
// Rejected input is not a finding and silence is the correct outcome.
// CHECK: Running
// CHECK-NOT: LLVM ERROR
// CHECK-NOT: error:
