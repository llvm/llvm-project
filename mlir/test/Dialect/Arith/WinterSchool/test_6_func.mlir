// RUN: mlir-opt %s -test-arith-reduce-float-bitwidth="patterns=func.return"
// RUN: mlir-opt %s -test-arith-reduce-float-bitwidth="patterns=func.func"
// RUN: mlir-opt %s -test-arith-reduce-float-bitwidth="patterns=func.func,func.return"

func.func @test_func(%arg0: f32) -> f32 {
  return %arg0 : f32
}
