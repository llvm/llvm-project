// RUN: mlir-opt -pass-pipeline='builtin.module(func.func(affine-loop-fusion{mode=producer fusion-maximal}))' %s | FileCheck %s

// Test fusion of affine nests inside other region-holding ops (scf.for in the
// test case below).

// CHECK-LABEL: func @fusion_inner_simple
func.func @fusion_inner_simple(%A : memref<10xf32>) {
  %cst = arith.constant 0.0 : f32

  affine.for %i = 0 to 100 {
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
  }

  // CHECK:      affine.for %{{.*}} = 0 to 100
  // CHECK-NEXT:   memref.alloc
  // CHECK-NEXT:   memref.alloc
  // CHECK-NEXT:   affine.for %{{.*}} = 0 to 10
  // CHECK-NOT:    affine.for

  return
}

// CHECK-LABEL: func @fusion_inner_simple_scf
func.func @fusion_inner_simple_scf(%A : memref<10xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  %cst = arith.constant 0.0 : f32

  scf.for %i = %c0 to %c100 step %c1 {
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
  }
  // CHECK:      scf.for
  // CHECK-NEXT:   memref.alloc
  // CHECK-NEXT:   memref.alloc
  // CHECK-NEXT:   affine.for %{{.*}} = 0 to 10
  // CHECK-NOT:    affine.for
  return
}

// CHECK-LABEL: func @fusion_inner_multiple_nests
func.func @fusion_inner_multiple_nests() {
  %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<4x4xi8>
  %alloc_10 = memref.alloc() : memref<8x4xi32>
  affine.for %arg8 = 0 to 4 {
    %alloc_14 = memref.alloc() : memref<4xi8>
    %alloc_15 = memref.alloc() : memref<8x4xi8>
    affine.for %arg9 = 0 to 4 {
      %0 = affine.load %alloc_5[%arg9, %arg8] : memref<4x4xi8>
      affine.store %0, %alloc_14[%arg9] : memref<4xi8>
    }
    %alloc_16 = memref.alloc() : memref<4xi8>
    affine.for %arg9 = 0 to 4 {
      %0 = affine.load %alloc_14[%arg9] : memref<4xi8>
      affine.store %0, %alloc_16[%arg9] : memref<4xi8>
    }
    affine.for %arg9 = 0 to 2 {
      %0 = affine.load %alloc_15[%arg9 * 4, 0] : memref<8x4xi8>
      %1 = affine.load %alloc_16[0] : memref<4xi8>
      %2 = affine.load %alloc_10[%arg9 * 4, %arg8] : memref<8x4xi32>
      %3 = arith.muli %0, %1 : i8
      %4 = arith.extsi %3 : i8 to i32
      %5 = arith.addi %4, %2 : i32
      affine.store %5, %alloc_10[%arg9 * 4 + 3, %arg8] : memref<8x4xi32>
    }
    memref.dealloc %alloc_16 : memref<4xi8>
  }
  // CHECK:      affine.for %{{.*}} = 0 to 4 {
  // Everything inside fused into two nests (the second will be DCE'd).
  // CHECK-NEXT:   memref.alloc() : memref<4xi8>
  // CHECK-NEXT:   memref.alloc() : memref<1xi8>
  // CHECK-NEXT:   memref.alloc() : memref<1xi8>
  // CHECK-NEXT:   memref.alloc() : memref<8x4xi8>
  // CHECK-NEXT:   memref.alloc() : memref<4xi8>
  // CHECK-NEXT:   affine.for %{{.*}} = 0 to 2 {
  // CHECK:        }
  // CHECK:        affine.for %{{.*}} = 0 to 4 {
  // CHECK:        }
  // CHECK-NEXT:   memref.dealloc
  // CHECK-NEXT: }
  // CHECK-NEXT: return
  return
}

// CHECK-LABEL: func @fusion_inside_scf_while
func.func @fusion_inside_scf_while(%A : memref<10xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  %cst = arith.constant 0.0 : f32

  %0 = scf.while (%arg3 = %cst) : (f32) -> (f32) {
    %1 = arith.cmpf ult, %arg3, %cst : f32
    scf.condition(%1) %arg3 : f32
  } do {
  ^bb0(%arg5: f32):

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
    %1 = arith.mulf %arg5, %cst : f32
    scf.yield %1 : f32
  }
  // CHECK:      scf.while
  // CHECK:        affine.for %{{.*}} = 0 to 10
  // CHECK-NOT:    affine.for
  // CHECK:        scf.yield
  return
}


memref.global "private" constant @__constant_10x2xf32 : memref<10x2xf32> = dense<0.000000e+00>

// CHECK-LABEL: func @fusion_inner_long
func.func @fusion_inner_long(%arg0: memref<10x2xf32>, %arg1: memref<10x10xf32>, %arg2: memref<10x2xf32>, %s: index) {
  %c0 = arith.constant 0 : index
  %cst_0 = arith.constant 1.000000e-03 : f32
  %c9 = arith.constant 9 : index
  %c10_i32 = arith.constant 10 : i32
  %c1_i32 = arith.constant 1 : i32
  %c100_i32 = arith.constant 100 : i32
  %c0_i32 = arith.constant 0 : i32
  %0 = memref.get_global @__constant_10x2xf32 : memref<10x2xf32>
  %1 = scf.for %arg3 = %c0_i32 to %c100_i32 step %c1_i32 iter_args(%arg4 = %arg0) -> (memref<10x2xf32>)  : i32 {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<10xi32>
    affine.for %arg5 = 0 to 10 {
      %3 = arith.index_cast %arg5 : index to i32
      affine.store %3, %alloc[%arg5] : memref<10xi32>
    }
    %2 = scf.for %arg5 = %c0_i32 to %c10_i32 step %c1_i32 iter_args(%arg6 = %0) -> (memref<10x2xf32>)  : i32 {
      %alloc_5 = memref.alloc() : memref<2xf32>
      affine.for %arg7 = 0 to 2 {
        %16 = affine.load %arg4[%s, %arg7] : memref<10x2xf32>
        affine.store %16, %alloc_5[%arg7] : memref<2xf32>
      }
      %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<1x2xf32>
      affine.for %arg7 = 0 to 2 {
        %16 = affine.load %alloc_5[%arg7] : memref<2xf32>
        affine.store %16, %alloc_6[0, %arg7] : memref<1x2xf32>
      }
      %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<10x2xf32>
      affine.for %arg7 = 0 to 10 {
        affine.for %arg8 = 0 to 2 {
          %16 = affine.load %alloc_6[0, %arg8] : memref<1x2xf32>
          affine.store %16, %alloc_7[%arg7, %arg8] : memref<10x2xf32>
        }
      }
      %alloc_8 = memref.alloc() {alignment = 64 : i64} : memref<10x2xf32>
      affine.for %arg7 = 0 to 10 {
        affine.for %arg8 = 0 to 2 {
          %16 = affine.load %alloc_7[%arg7, %arg8] : memref<10x2xf32>
          %17 = affine.load %arg4[%arg7, %arg8] : memref<10x2xf32>
          %18 = arith.subf %16, %17 : f32
          affine.store %18, %alloc_8[%arg7, %arg8] : memref<10x2xf32>
        }
      }
      scf.yield %alloc_8 : memref<10x2xf32>
      // CHECK:      scf.for
      // CHECK:        scf.for
      // CHECK:          affine.for %{{.*}} = 0 to 10
      // CHECK-NEXT:       affine.for %{{.*}} = 0 to 2
      // CHECK-NOT:      affine.for
      // CHECK:          scf.yield
    }
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<10x2xf32>
    affine.for %arg5 = 0 to 10 {
      affine.for %arg6 = 0 to 2 {
        affine.store %cst_0, %alloc_2[%arg5, %arg6] : memref<10x2xf32>
      }
    }
    %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<10x2xf32>
    affine.for %arg5 = 0 to 10 {
      affine.for %arg6 = 0 to 2 {
        %3 = affine.load %alloc_2[%arg5, %arg6] : memref<10x2xf32>
        %4 = affine.load %2[%arg5, %arg6] : memref<10x2xf32>
        %5 = arith.mulf %3, %4 : f32
        affine.store %5, %alloc_3[%arg5, %arg6] : memref<10x2xf32>
      }
    }
    scf.yield %alloc_3 : memref<10x2xf32>
    // The nests above will be fused as well.
    // CHECK:      affine.for %{{.*}} = 0 to 10
    // CHECK-NEXT:   affine.for %{{.*}} = 0 to 2
    // CHECK-NOT:  affine.for
    // CHECK:      scf.yield
  }
  affine.for %arg3 = 0 to 10 {
    affine.for %arg4 = 0 to 2 {
      %2 = affine.load %1[%arg3, %arg4] : memref<10x2xf32>
      affine.store %2, %arg2[%arg3, %arg4] : memref<10x2xf32>
    }
  }
  return
}
