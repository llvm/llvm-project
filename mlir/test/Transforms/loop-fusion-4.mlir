// RUN: mlir-opt -allow-unregistered-dialect %s -affine-loop-fusion -split-input-file | FileCheck %s
// RUN: mlir-opt -allow-unregistered-dialect %s -affine-loop-fusion="fusion-maximal" -split-input-file | FileCheck %s --check-prefix=MAXIMAL

// Part I of fusion tests in  mlir/test/Transforms/loop-fusion.mlir. 
// Part II of fusion tests in mlir/test/Transforms/loop-fusion-2.mlir
// Part III of fusion tests in mlir/test/Transforms/loop-fusion-3.mlir

// -----

func @reduce_add_non_maximal_f32_f32(%arg0: memref<64x64xf32, 1>, %arg1 : memref<1x64xf32, 1>, %arg2 : memref<1x64xf32, 1>) {
    %cst_0 = constant 0.000000e+00 : f32
    %cst_1 = constant 1.000000e+00 : f32
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 64 {
        %accum = affine.for %arg5 = 0 to 64 iter_args (%prevAccum = %cst_0) -> f32 {
          %4 = affine.load %arg0[%arg5, %arg4] : memref<64x64xf32, 1>
          %5 = addf %prevAccum, %4 : f32
          affine.yield %5 : f32
        }
        %accum_dbl = addf %accum, %accum : f32
        affine.store %accum_dbl, %arg1[%arg3, %arg4] : memref<1x64xf32, 1>
      }
    }
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 64 {
        // Following loop  trip count does not match the corresponding source trip count.
        %accum = affine.for %arg5 = 0 to 32 iter_args (%prevAccum = %cst_1) -> f32 {
          %4 = affine.load %arg0[%arg5, %arg4] : memref<64x64xf32, 1>
          %5 = mulf %prevAccum, %4 : f32
          affine.yield %5 : f32
        }
        %accum_sqr = mulf %accum, %accum : f32
        affine.store %accum_sqr, %arg2[%arg3, %arg4] : memref<1x64xf32, 1>
      }
    }
    return
}
// Test checks the loop structure is preserved after sibling fusion
// since the destination loop and source loop trip counts do not
// match.
// MAXIMAL-LABEL:   func @reduce_add_non_maximal_f32_f32(
// MAXIMAL:        %[[cst_0:.*]] = constant 0.000000e+00 : f32
// MAXIMAL-NEXT:        %[[cst_1:.*]] = constant 1.000000e+00 : f32
// MAXIMAL-NEXT:           affine.for %[[idx_0:.*]]= 0 to 1 {
// MAXIMAL-NEXT:             affine.for %[[idx_1:.*]] = 0 to 64 {
// MAXIMAL-NEXT:               %[[result_1:.*]] = affine.for %[[idx_2:.*]] = 0 to 32 iter_args(%[[iter_0:.*]] = %[[cst_1]]) -> (f32) {
// MAXIMAL-NEXT:                 %[[result_0:.*]] = affine.for %[[idx_3:.*]] = 0 to 64 iter_args(%[[iter_1:.*]] = %[[cst_0]]) -> (f32) {

// Expects fusion of producer into consumer at depth 4 and subsequent removal of
// source loop.
// CHECK-LABEL: func @unflatten4d
func @unflatten4d(%arg1: memref<7x8x9x10xf32>) {
  %m = memref.alloc() : memref<5040xf32>
  %cf7 = constant 7.0 : f32

  affine.for %i0 = 0 to 7 {
    affine.for %i1 = 0 to 8 {
      affine.for %i2 = 0 to 9 {
        affine.for %i3 = 0 to 10 {
          affine.store %cf7, %m[720 * %i0 + 90 * %i1 + 10 * %i2 + %i3] : memref<5040xf32>
        }
      }
    }
  }
  affine.for %i0 = 0 to 7 {
    affine.for %i1 = 0 to 8 {
      affine.for %i2 = 0 to 9 {
        affine.for %i3 = 0 to 10 {
          %v0 = affine.load %m[720 * %i0 + 90 * %i1 + 10 * %i2 + %i3] : memref<5040xf32>
          affine.store %v0, %arg1[%i0, %i1, %i2, %i3] : memref<7x8x9x10xf32>
        }
      }
    }
  }
  return
}

// CHECK:        affine.for
// CHECK-NEXT:     affine.for
// CHECK-NEXT:       affine.for
// CHECK-NEXT:         affine.for
// CHECK-NOT:    affine.for
// CHECK: return

// -----

// Expects fusion of producer into consumer at depth 2 and subsequent removal of
// source loop.
// CHECK-LABEL: func @unflatten2d_with_transpose
func @unflatten2d_with_transpose(%arg1: memref<8x7xf32>) {
  %m = memref.alloc() : memref<56xf32>
  %cf7 = constant 7.0 : f32

  affine.for %i0 = 0 to 7 {
    affine.for %i1 = 0 to 8 {
      affine.store %cf7, %m[8 * %i0 + %i1] : memref<56xf32>
    }
  }
  affine.for %i0 = 0 to 8 {
    affine.for %i1 = 0 to 7 {
      %v0 = affine.load %m[%i0 + 8 * %i1] : memref<56xf32>
      affine.store %v0, %arg1[%i0, %i1] : memref<8x7xf32>
    }
  }
  return
}

// CHECK:        affine.for
// CHECK-NEXT:     affine.for
// CHECK-NOT:    affine.for
// CHECK: return