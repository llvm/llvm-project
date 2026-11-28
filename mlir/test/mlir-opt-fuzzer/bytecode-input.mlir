// RUN: mlir-opt %s --emit-bytecode -o %t.bc
// RUN: mlir-opt-fuzzer %t.bc -ignore_remaining_args=1 --pipeline=cse 2>&1 | FileCheck %s

module {
  func.func @g() {
    return
  }
}

// Real bytecode (manufactured from %s) is rejected on the magic prefix.
// CHECK: Running
// CHECK-NOT: LLVM ERROR
