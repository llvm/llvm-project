// RUN: mlir-opt %s -affine-super-vectorizer-test  -vectorize-affine-loop-nest -split-input-file 2>&1 |  FileCheck %s

func.func @unparallel_loop_reduction_unsupported(%in: memref<256x512xf32>, %out: memref<256xf32>) {
 // CHECK: Outermost loop cannot be parallel
 %cst = arith.constant 1.000000e+00 : f32
 %final_red = affine.for %j = 0 to 512 iter_args(%red_iter = %cst) -> (f32) {
   %add = arith.addf %red_iter, %red_iter : f32
   affine.yield %add : f32
 }
 return
}
