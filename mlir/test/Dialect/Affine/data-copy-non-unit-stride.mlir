// RUN: mlir-opt %s -affine-data-copy-generate="fast-mem-capacity=64 fast-mem-space=0" | FileCheck %s

// Issue #54994: a sparse write (16 iterations, 31-cell bounding box) must
// emit a copy-IN so the post-loop copy-OUT doesn't clobber gap cells.

// CHECK-LABEL: func @non_unit_stride_store
// CHECK:         [[BUF:%[a-zA-Z0-9_]+]] = memref.alloc
// CHECK:         affine.for %[[I:.*]] = 0 to 31 {
// CHECK-NEXT:      affine.load %{{.*}}[%[[I]]] : memref<32xf32>
// CHECK-NEXT:      affine.store %{{.*}}, [[BUF]][%[[I]]]
// CHECK-NEXT:    }
// CHECK:         affine.for %{{.*}} = 0 to 16 {
// CHECK-NEXT:      affine.store %{{.*}}, [[BUF]][%{{.*}} * 2]
// CHECK-NEXT:    }
// CHECK:         affine.for %{{.*}} = 0 to 31 {
func.func @non_unit_stride_store(%arg0: memref<32xf32>) {
  %cst = arith.constant 0.000000e+00 : f32
  affine.for %arg1 = 0 to 16 {
    affine.store %cst, %arg0[%arg1 * 2] : memref<32xf32>
  }
  return
}

// -----

// Dense tile (iteration count 512 = bounding-box volume): no extra copy-IN.

// CHECK-LABEL: func @densely_tiled_access
// CHECK:         memref.alloc
func.func @densely_tiled_access(%arg0: memref<512xf32>) {
  affine.for %kT = 0 to 32 {
    affine.for %kk = 0 to 16 {
      %k = affine.apply affine_map<(d0, d1) -> (16 * d0 + d1)>(%kT, %kk)
      %v = affine.load %arg0[%k] : memref<512xf32>
      affine.store %v, %arg0[%k] : memref<512xf32>
    }
  }
  return
}
