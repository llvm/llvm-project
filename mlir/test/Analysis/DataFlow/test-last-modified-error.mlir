// RUN: mlir-opt -test-last-modified -verify-diagnostics %s

// test error propagation from UnderlyingValueAnalysis::visitOperation
func.func @test() {
  // expected-error @+1 {{this op is always fails}}
  %c0 = arith.constant { always_fail } 0 : i32
  return
}
