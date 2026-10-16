// RUN: mlir-opt %s -pass-pipeline='builtin.module(func.func(affine-loop-fusion{maximal mode=sibling}))' -split-input-file | FileCheck %s
// RUN: mlir-opt %s -pass-pipeline='builtin.module(func.func(affine-loop-fusion))' -split-input-file | FileCheck %s

// Test cases specifically for sibling fusion. Note that sibling fusion test
// cases also exist in loop-fusion*.mlir.

// CHECK-LABEL: func @disjoint_stores
func.func @disjoint_stores(%0: memref<8xf32>) {
  %alloc_1 = memref.alloc() : memref<16xf32>
  // The affine stores below are to different parts of the memrefs. Sibling
  // fusion helps improve reuse and is valid.
  affine.for %arg2 = 0 to 8 {
    %2 = affine.load %0[%arg2] : memref<8xf32>
    affine.store %2, %alloc_1[%arg2] : memref<16xf32>
  }
  affine.for %arg2 = 0 to 8 {
    %2 = affine.load %0[%arg2] : memref<8xf32>
    %3 = arith.negf %2 : f32
    affine.store %3, %alloc_1[%arg2 + 8] : memref<16xf32>
  }
  // CHECK: affine.for
  // CHECK-NOT: affine.for
  return
}

// -----

// Sibling nests with multiple loads to the shared memref (at different
// offsets) are fused.

// CHECK-LABEL: func @multiple_loads
// CHECK-SAME:    (%{{.*}}: memref<65xf32>, %[[B:[a-z0-9]+]]: memref<64xf32>, %[[C:[a-z0-9]+]]: memref<64xf32>)
func.func @multiple_loads(%a: memref<65xf32>, %b: memref<64xf32>, %c: memref<64xf32>) {
  affine.for %i = 0 to 64 {
    %0 = affine.load %a[%i] : memref<65xf32>
    %1 = affine.load %a[%i + 1] : memref<65xf32>
    %2 = arith.addf %0, %1 : f32
    affine.store %2, %b[%i] : memref<64xf32>
  }
  affine.for %i = 0 to 64 {
    %0 = affine.load %a[%i] : memref<65xf32>
    %1 = affine.load %a[%i + 1] : memref<65xf32>
    %2 = arith.mulf %0, %1 : f32
    affine.store %2, %c[%i] : memref<64xf32>
  }
  // CHECK:       affine.for %[[I:.*]] = 0 to 64 {
  // CHECK-NEXT:    affine.load %{{.*}}[%[[I]]]
  // CHECK-NEXT:    affine.load %{{.*}}[%[[I]] + 1]
  // CHECK-NEXT:    arith.addf
  // CHECK-NEXT:    affine.store %{{.*}}, %[[B]][%[[I]]]
  // CHECK-NEXT:    affine.load %{{.*}}[%[[I]]]
  // CHECK-NEXT:    affine.load %{{.*}}[%[[I]] + 1]
  // CHECK-NEXT:    arith.mulf
  // CHECK-NEXT:    affine.store %{{.*}}, %[[C]][%[[I]]]
  // CHECK-NEXT:  }
  // CHECK-NEXT:  return
  return
}

// -----

// Only one of the sibling nests has multiple loads to the shared memref.

// CHECK-LABEL: func @single_and_multiple_loads
// CHECK-SAME:    (%{{.*}}: memref<65xf32>, %[[B:[a-z0-9]+]]: memref<64xf32>, %[[C:[a-z0-9]+]]: memref<64xf32>)
func.func @single_and_multiple_loads(%a: memref<65xf32>, %b: memref<64xf32>, %c: memref<64xf32>) {
  affine.for %i = 0 to 64 {
    %0 = affine.load %a[%i] : memref<65xf32>
    %1 = affine.load %a[%i + 1] : memref<65xf32>
    %2 = arith.addf %0, %1 : f32
    affine.store %2, %b[%i] : memref<64xf32>
  }
  affine.for %i = 0 to 64 {
    %0 = affine.load %a[%i] : memref<65xf32>
    %1 = arith.mulf %0, %0 : f32
    affine.store %1, %c[%i] : memref<64xf32>
  }
  // CHECK:       affine.for %[[I:.*]] = 0 to 64 {
  // CHECK-NEXT:    affine.load %{{.*}}[%[[I]]]
  // CHECK-NEXT:    affine.load %{{.*}}[%[[I]] + 1]
  // CHECK-NEXT:    arith.addf
  // CHECK-NEXT:    affine.store %{{.*}}, %[[B]][%[[I]]]
  // CHECK-NEXT:    affine.load %{{.*}}[%[[I]]]
  // CHECK-NEXT:    arith.mulf
  // CHECK-NEXT:    affine.store %{{.*}}, %[[C]][%[[I]]]
  // CHECK-NEXT:  }
  // CHECK-NEXT:  return
  return
}

// -----

// 2-d sibling nests with multiple loads to the shared memref are fused at the
// innermost depth.

// CHECK-LABEL: func @multiple_loads_2d
// CHECK-SAME:    (%{{.*}}: memref<33x32xf32>, %[[B:[a-z0-9]+]]: memref<32x32xf32>, %[[C:[a-z0-9]+]]: memref<32x32xf32>)
func.func @multiple_loads_2d(%a: memref<33x32xf32>, %b: memref<32x32xf32>, %c: memref<32x32xf32>) {
  affine.for %i = 0 to 32 {
    affine.for %j = 0 to 32 {
      %0 = affine.load %a[%i, %j] : memref<33x32xf32>
      %1 = affine.load %a[%i + 1, %j] : memref<33x32xf32>
      %2 = arith.addf %0, %1 : f32
      affine.store %2, %b[%i, %j] : memref<32x32xf32>
    }
  }
  affine.for %i = 0 to 32 {
    affine.for %j = 0 to 32 {
      %0 = affine.load %a[%i, %j] : memref<33x32xf32>
      %1 = affine.load %a[%i + 1, %j] : memref<33x32xf32>
      %2 = arith.mulf %0, %1 : f32
      affine.store %2, %c[%i, %j] : memref<32x32xf32>
    }
  }
  // CHECK:       affine.for %[[I:.*]] = 0 to 32 {
  // CHECK-NEXT:    affine.for %[[J:.*]] = 0 to 32 {
  // CHECK:           affine.store %{{.*}}, %[[B]][%[[I]], %[[J]]]
  // CHECK-NOT:       affine.for
  // CHECK:           affine.store %{{.*}}, %[[C]][%[[I]], %[[J]]]
  // CHECK-NEXT:    }
  // CHECK-NEXT:  }
  // CHECK-NEXT:  return
  return
}
