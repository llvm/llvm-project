// RUN: mlir-opt %s -affine-super-vectorize="virtual-vector-size=32,256 test-fastest-varying=1,0 vectorize-reductions=true" -verify-diagnostics

// TODO: Vectorization of reduction loops along the reduction dimension is not
//       supported for higher-rank vectors yet, so we are just checking that an
//       error message is produced.

// expected-error@+1 {{Vectorizing reductions is supported only for 1-D vectors}}
func.func @vecdim_reduction_2d(%in: memref<256x512x1024xf32>, %out: memref<256xf32>) {
 %cst = arith.constant 0.000000e+00 : f32
 affine.for %i = 0 to 256 {
   %sum_j = affine.for %j = 0 to 512 iter_args(%red_iter_j = %cst) -> (f32) {
     %sum_k = affine.for %k = 0 to 1024 iter_args(%red_iter_k = %cst) -> (f32) {
       %ld = affine.load %in[%i, %j, %k] : memref<256x512x1024xf32>
       %add = arith.addf %red_iter_k, %ld : f32
       affine.yield %add : f32
     }
     %add = arith.addf %red_iter_j, %sum_k : f32
     affine.yield %add : f32
   }
   affine.store %sum_j, %out[%i] : memref<256xf32>
 }
 return
}

