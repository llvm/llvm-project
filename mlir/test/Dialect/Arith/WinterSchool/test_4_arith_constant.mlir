// RUN: mlir-opt %s -test-arith-reduce-float-bitwidth="patterns=arith.constant fold=false"

func.func @test_constant() -> f32 {
  %0 = arith.constant 2.0 : f32
  return %0 : f32
}
