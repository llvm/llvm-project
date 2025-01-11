// RUN: mlir-opt %s -test-arith-reduce-float-bitwidth="patterns=arith.addf"

func.func @test_add(%arg0: f32, %arg1: f32) {
  %0 = arith.addf %arg0, %arg1 : f32
  return
}
