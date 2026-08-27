// RUN: mlir-opt %s -scf-for-loop-canonicalization -split-input-file | FileCheck %s

// CHECK-LABEL: func @reduce
func.func @reduce() {
  // CHECK: %[[C64:.*]] = arith.constant 64 : index
  %c2 = arith.constant 2 : index
  %cst_0 = arith.constant -0.000000e+00 : f32
  %0 = memref.alloc() : memref<128x384xf32>
  linalg.fill ins(%cst_0 : f32) outs(%0 : memref<128x384xf32>)
  %2 = memref.alloc() : memref<128xf32>
  linalg.fill ins(%cst_0 : f32) outs(%2 : memref<128xf32>)
  scf.forall (%arg0) in (%c2) {
    %7 = affine.min affine_map<(d0) -> (d0 * -64 + 128, 64)>(%arg0)
    %8 = affine.max affine_map<(d0) -> (0, d0)>(%7)
    %9 = affine.apply affine_map<(d0) -> (d0 * 64)>(%arg0)
    %10 = affine.min affine_map<(d0, d1) -> (d1 * -64 + 128, d0)>(%8, %arg0)

    // CHECK: memref.subview %{{.*}}[%{{.*}}, 0] [%[[C64]], 384] [1, 1] : memref<128x384xf32> to memref<?x384xf32, {{.*}}>
    // CHECK: memref.subview %{{.*}}[%{{.*}}] [%[[C64]]] [1] : memref<128xf32> to memref<?xf32, {{.*}}>
    %11 = memref.subview %0[%9, 0] [%10, 384] [1, 1] :
      memref<128x384xf32> to memref<?x384xf32, affine_map<(d0, d1)[s0] -> (d0 * 384 + s0 + d1)>>
    %12 = memref.subview %2[%9] [%10] [1] :
      memref<128xf32> to memref<?xf32, affine_map<(d0)[s0] -> (d0 + s0)>>

    // CHECK: linalg.generic {{.*}} ins(%{{.*}} : memref<?x384xf32, {{.*}}>) outs(%{{.*}} : memref<?xf32, {{.*}}>)
    linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                      affine_map<(d0, d1) -> (d0)>],
                     iterator_types = ["parallel", "reduction"]}
      ins(%11 : memref<?x384xf32, affine_map<(d0, d1)[s0] -> (d0 * 384 + s0 + d1)>>)
      outs(%12 : memref<?xf32, affine_map<(d0)[s0] -> (d0 + s0)>>) {
        ^bb0(%arg1: f32, %arg2: f32):
          %14 = arith.addf %arg1, %arg2 : f32
          linalg.yield %14 : f32
      }
  }
  return
}

// -----

// Regression test for GH#127436: simplifyConstrainedMinMaxOp must handle
// null-value operands produced when scf.forall has static (integer attribute)
// bounds. In addLoopRangeConstraints, static lb/ub produce null-value symbol
// variables in FlatAffineValueConstraints (via appendSymbolVar with no SSA
// value). These must be removed before calling canonicalizeMapAndOperands to
// avoid undefined behavior (null Value used as DenseMap key or passed to
// matchPattern). The filter in simplifyConstrainedMinMaxOp compacts the
// operands by replacing null positions with constant 0 and discarding them.

// CHECK-LABEL: func @forall_static_bounds_affine_min_simplify
// CHECK: %[[C64:.*]] = arith.constant 64 : i32
// CHECK-NOT: affine.min
// CHECK: memref.store %[[C64]]
func.func @forall_static_bounds_affine_min_simplify(%A : memref<128xi32>) {
  // lb=0, ub=2, step=1 are static integers -> IntegerAttr in OpFoldResult
  // -> null-value symbols in the constraint system.
  scf.forall (%i) = (0) to (2) step (1) {
    // For i in [0, 2): min(-64*i + 128, 64) = 64 always.
    %tile = affine.min affine_map<(d0) -> (d0 * -64 + 128, 64)>(%i)
    %cast = arith.index_cast %tile : index to i32
    %c0 = arith.constant 0 : index
    memref.store %cast, %A[%c0] : memref<128xi32>
    scf.forall.in_parallel {}
  }
  return
}

// -----

// Regression test for GH#127436: same as above but with a 3-result affine.min
// to exercise multiple resultDimStart null-value dim slots.

// CHECK-LABEL: func @forall_static_bounds_multi_result_min_simplify
// CHECK: %[[C32:.*]] = arith.constant 32 : i32
// CHECK-NOT: affine.min
// CHECK: memref.store %[[C32]]
func.func @forall_static_bounds_multi_result_min_simplify(%A : memref<128xi32>) {
  scf.forall (%i) = (0) to (2) step (1) {
    // 3 results; for i in [0, 2): all results are >= 32, so min = 32.
    %tile = affine.min affine_map<(d0) -> (d0 * -64 + 128, d0 * -64 + 96, 32)>(%i)
    %cast = arith.index_cast %tile : index to i32
    %c0 = arith.constant 0 : index
    memref.store %cast, %A[%c0] : memref<128xi32>
    scf.forall.in_parallel {}
  }
  return
}
