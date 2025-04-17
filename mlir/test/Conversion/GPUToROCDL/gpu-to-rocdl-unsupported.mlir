// RUN: mlir-opt %s -convert-gpu-to-rocdl -verify-diagnostics

gpu.module @test_module {
  // ROCDL lowering only suport shuffles for 32bit ints/floats, but they
  // shouldn't crash on unsupported types.
  func.func @gpu_shuffle_unsupported(%arg0 : vector<4xf16>) -> vector<4xf16> {
    %offset = arith.constant 4 : i32
    %width = arith.constant 64 : i32
    // expected-error @+1 {{failed to legalize operation 'gpu.shuffle'}}
    %shfl, %pred = gpu.shuffle xor %arg0, %offset, %width : vector<4xf16>
    return %shfl : vector<4xf16>
  }
}
