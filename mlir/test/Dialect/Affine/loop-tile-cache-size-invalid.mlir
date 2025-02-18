// RUN: mlir-opt -affine-loop-tile="cache-size=0" %s -split-input-file -verify-diagnostics
// XFAIL: *
// This is @legal_loop() test case. The expected error is "illegal argument: 'cache-size' cannot be zero"
// at unknown location because of invalid command line argument.

func.func @test_cache_size_zero() {
  %0 = memref.alloc() : memref<64xf32>
  affine.for %i = 0 to 64 {
    %1 = affine.load %0[%i] : memref<64xf32>
    %2 = arith.addf %1, %1 : f32
    affine.store %2, %0[%i] : memref<64xf32>
  }
  return
}

