// RUN: mlir-opt -pass-pipeline='builtin.module(func.func(affine-loop-fusion))' %s | FileCheck %s

// Test fusion of affine nests in the presence of other region-holding ops
// (scf.for in the test case below) in the block.

// CHECK-LABEL: func @scf_and_affine
func.func @scf_and_affine(%A : memref<10xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %cst = arith.constant 0.0 : f32

  %B = memref.alloc() : memref<10xf32>
  %C = memref.alloc() : memref<10xf32>

  affine.for %j = 0 to 10 {
    %v = affine.load %A[%j] : memref<10xf32>
    affine.store %v, %B[%j] : memref<10xf32>
  }

  affine.for %j = 0 to 10 {
    %v = affine.load %B[%j] : memref<10xf32>
    affine.store %v, %C[%j] : memref<10xf32>
  }
  // Nests are fused.
  // CHECK:     affine.for %{{.*}} = 0 to 10
  // CHECK-NOT: affine.for
  // CHECK:     scf.for

  scf.for %i = %c0 to %c10 step %c1 {
    memref.store %cst, %B[%i] : memref<10xf32>
  }

  // The nests below shouldn't be fused.
  affine.for %j = 0 to 10 {
    %v = affine.load %A[%j] : memref<10xf32>
    affine.store %v, %B[%j] : memref<10xf32>
  }
  scf.for %i = %c0 to %c10 step %c1 {
    memref.store %cst, %B[%i] : memref<10xf32>
  }
  affine.for %j = 0 to 10 {
    %v = affine.load %B[%j] : memref<10xf32>
    affine.store %v, %C[%j] : memref<10xf32>
  }
  // CHECK: affine.for
  // CHECK: scf.for
  // CHECK: affine.for

  return
}
