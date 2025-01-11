// RUN: mlir-opt %s -test-arith-reduce-float-bitwidth="patterns=arith.constant,arith.addf"
// RUN: mlir-opt %s -test-arith-reduce-float-bitwidth="patterns=arith.addf"

func.func @test_add_constant(%arg0: f32) -> f32 {
  %0 = arith.constant 2.0 : f32
  %1 = arith.addf %arg0, %0 : f32
  return %1 : f32
}
