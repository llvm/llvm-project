// RUN: mlir-opt %s --verify-diagnostics -pass-pipeline="builtin.module(func.func(convert-affine-for-to-gpu{gpu-block-dims=1 gpu-thread-dims=0}))"

// Regression test: affine.for loops with iter_args (reduction loops) must be
// rejected with a proper diagnostic rather than crashing. The pass cannot
// convert reduction loops because the loop body is moved to the GPU kernel
// while block arguments referencing iter_args would become dangling uses.

func.func @reduction_loop(%arg0: memref<1024xf32>) -> f32 {
  %c10 = arith.constant 10 : index
  %cst = arith.constant 0.000000e+00 : f32
  // expected-error @+1 {{affine loop with iter_args cannot be converted to GPU kernel}}
  %result = affine.for %i = 0 to %c10 iter_args(%acc = %cst) -> (f32) {
    %val = affine.load %arg0[%i] : memref<1024xf32>
    %sum = arith.addf %acc, %val : f32
    affine.yield %sum : f32
  }
  return %result : f32
}
