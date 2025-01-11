// RUN: mlir-opt %s -test-arith-reduce-float-bitwidth="patterns=func.func,func.return,arith.truncf"
// RUN: mlir-opt %s -test-arith-reduce-float-bitwidth="patterns=func.func,func.return,arith.truncf" -canonicalize

func.func @test_func(%arg0: f32) -> f32 {
  return %arg0 : f32
}
