// RUN: mlir-opt %s -allow-unregistered-dialect -test-loop-fusion -test-loop-fusion-transformation -split-input-file -canonicalize | FileCheck %s

// CHECK-LABEL: func @slice_depth1_loop_nest() {
func @slice_depth1_loop_nest() {
  %0 = memref.alloc() : memref<100xf32>
  %cst = constant 7.000000e+00 : f32
  affine.for %i0 = 0 to 16 {
    affine.store %cst, %0[%i0] : memref<100xf32>
  }
  affine.for %i1 = 0 to 5 {
    %1 = affine.load %0[%i1] : memref<100xf32>
    "prevent.dce"(%1) : (f32) -> ()
  }
  // CHECK:      affine.for %[[IV0:.*]] = 0 to 5 {
  // CHECK-NEXT:   affine.store %{{.*}}, %{{.*}}[%[[IV0]]] : memref<100xf32>
  // CHECK-NEXT:   affine.load %{{.*}}[%[[IV0]]] : memref<100xf32>
  // CHECK-NEXT:   "prevent.dce"(%1) : (f32) -> ()
  // CHECK-NEXT: }
  // CHECK-NEXT: return
  return
}

// -----

// CHECK-LABEL: func @should_fuse_reduction_to_pointwise() {
func @should_fuse_reduction_to_pointwise() {
  %a = memref.alloc() : memref<10x10xf32>
  %b = memref.alloc() : memref<10xf32>
  %c = memref.alloc() : memref<10xf32>

  %cf7 = constant 7.0 : f32

  affine.for %i0 = 0 to 10 {
    affine.for %i1 = 0 to 10 {
      %v0 = affine.load %b[%i0] : memref<10xf32>
      %v1 = affine.load %a[%i0, %i1] : memref<10x10xf32>
      %v3 = addf %v0, %v1 : f32
      affine.store %v3, %b[%i0] : memref<10xf32>
    }
  }
  affine.for %i2 = 0 to 10 {
    %v4 = affine.load %b[%i2] : memref<10xf32>
    affine.store %v4, %c[%i2] : memref<10xf32>
  }

  // Match on the fused loop nest.
  // Should fuse in entire inner loop on %i1 from source loop nest, as %i1
  // is not used in the access function of the store/load on %b.
  // CHECK:       affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT:    affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT:      affine.load %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:      affine.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<10x10xf32>
  // CHECK-NEXT:      addf %{{.*}}, %{{.*}} : f32
  // CHECK-NEXT:      affine.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:    }
  // CHECK-NEXT:    affine.load %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:    affine.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:  }
  // CHECK-NEXT:  return
  return
}

// -----

// CHECK-LABEL: func @should_fuse_avoiding_dependence_cycle() {
func @should_fuse_avoiding_dependence_cycle() {
  %a = memref.alloc() : memref<10xf32>
  %b = memref.alloc() : memref<10xf32>
  %c = memref.alloc() : memref<10xf32>

  %cf7 = constant 7.0 : f32

  // Set up the following dependences:
  // 1) loop0 -> loop1 on memref '%{{.*}}'
  // 2) loop0 -> loop2 on memref '%{{.*}}'
  // 3) loop1 -> loop2 on memref '%{{.*}}'
  affine.for %i0 = 0 to 10 {
    %v0 = affine.load %a[%i0] : memref<10xf32>
    affine.store %v0, %b[%i0] : memref<10xf32>
  }
  affine.for %i1 = 0 to 10 {
    affine.store %cf7, %a[%i1] : memref<10xf32>
    %v1 = affine.load %c[%i1] : memref<10xf32>
    "prevent.dce"(%v1) : (f32) -> ()
  }
  affine.for %i2 = 0 to 10 {
    %v2 = affine.load %b[%i2] : memref<10xf32>
    affine.store %v2, %c[%i2] : memref<10xf32>
  }
  // Fusing loop first loop into last would create a cycle:
  //   {1} <--> {0, 2}
  // However, we can avoid the dependence cycle if we first fuse loop0 into
  // loop1:
  //   {0, 1) --> {2}
  // Then fuse this loop nest with loop2:
  //   {0, 1, 2}
  //
  // CHECK:      affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT:   affine.load %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:   affine.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:   affine.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:   affine.load %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:   "prevent.dce"
  // CHECK-NEXT:   affine.load %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:   affine.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT: }
  // CHECK-NEXT: return
  return
}
