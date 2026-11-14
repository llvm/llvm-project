// RUN: mlir-opt %s -pass-pipeline='builtin.module(func.func(affine-loop-fusion{mode=producer}))' -o /dev/null

// Computing slice bounds can produce coefficients larger than int64_t even
// though the symbolic constant itself fits. Floor-div detection must decline
// an unrepresentable result instead of asserting during conversion.
func.func @large_symbol(%B : memref<20 x 10 x f32>, %A : memref<20 x 10 x f32>) {
  %c = arith.constant 4607252787007324200 : index
  %cst = arith.constant 1.000000e+00 : f32
  affine.for %i = 0 to 2 {
    affine.for %j = 0 to 10 {
      affine.store %cst, %B[%i, %j] : memref<20 x 10 x f32>
    }
  }
  affine.for %i = 0 to 2 {
    affine.for %j = 0 to 10 {
      %0 = affine.load %B[%i, (symbol(%c) mod 21 + %j floordiv 27 + symbol(%c)) mod 10] : memref<20 x 10 x f32>
      affine.store %0, %A[%i, %j] : memref<20 x 10 x f32>
    }
  }
  return
}
