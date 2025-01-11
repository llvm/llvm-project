// RUN: mlir-opt %s -test-arith-reduce-float-bitwidth="patterns=pdl_patterns" -split-input-file

func.func @test_trunc_ext(%arg0: f16) -> f16 {
  %0 = arith.extf %arg0 : f16 to f32
  %1 = arith.truncf %0 : f32 to f16
  return %1 : f16
}

// -----

func.func @test_add(%arg0: f32, %arg1: f32) -> f32 {
  %0 = arith.addf %arg0, %arg1 : f32
  return %0 : f32
}

// -----

func.func @test_add_constant(%arg0: f32) -> f32 {
  %0 = arith.constant 2.0 : f32
  %1 = arith.addf %arg0, %0 : f32
  return %1 : f32
}
