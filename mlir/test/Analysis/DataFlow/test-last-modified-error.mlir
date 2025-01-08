// RUN: not mlir-opt -test-last-modified %s 2>&1 | FileCheck %s

// test error propagation from UnderlyingValueAnalysis::visitOperation
// CHECK: this op is always fails
func.func @test() {
  %c0 = arith.constant { always_fail } 0 : i32
  return
}
