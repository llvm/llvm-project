// RUN: mlir-opt %s -allow-unregistered-dialect -test-multi-buffering=multiplier=5 -cse -split-input-file | FileCheck %s

// CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0) -> (((d0 - 1) floordiv 3) mod 5)>

// CHECK-LABEL: func @multi_buffer
func.func @multi_buffer(%a: memref<1024x1024xf32>) {
// CHECK-DAG: %[[A:.*]] = memref.alloc() {someAttribute} : memref<5x4x128xf32>
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C3:.*]] = arith.constant 3 : index
  %0 = memref.alloc() {someAttribute} : memref<4x128xf32>
  %c1024 = arith.constant 1024 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
// CHECK: scf.for %[[IV:.*]] = %[[C1]]
  scf.for %arg2 = %c1 to %c1024 step %c3 {
// CHECK: %[[I:.*]] = affine.apply #[[$MAP1]](%[[IV]])
// CHECK: %[[SV:.*]] = memref.subview %[[A]][%[[I]], 0, 0] [1, 4, 128] [1, 1, 1] : memref<5x4x128xf32> to memref<4x128xf32, strided<[128, 1], offset: ?>>
   %1 = memref.subview %a[%arg2, 0] [4, 128] [1, 1] :
    memref<1024x1024xf32> to memref<4x128xf32, affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>>
// CHECK: memref.copy %{{.*}}, %[[SV]] : memref<4x128xf32, #{{.*}}> to memref<4x128xf32, strided<[128, 1], offset: ?>>
   memref.copy %1, %0 : memref<4x128xf32, affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>> to memref<4x128xf32>
// CHECK: "some_use"(%[[SV]]) : (memref<4x128xf32, strided{{.*}}>) -> ()
    "some_use"(%0) : (memref<4x128xf32>) -> ()
// CHECK: "some_use"(%[[SV]]) : (memref<4x128xf32, strided{{.*}}>) -> ()
   "some_use"(%0) : (memref<4x128xf32>) -> ()
  }
  return
}

// -----

// CHECK-LABEL: func @multi_buffer_affine
func.func @multi_buffer_affine(%a: memref<1024x1024xf32>) {
// CHECK-DAG: %[[A:.*]] = memref.alloc() : memref<5x4x128xf32>
  %0 = memref.alloc() : memref<4x128xf32>
  %c1024 = arith.constant 1024 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
// CHECK: affine.for %[[IV:.*]] = 1
  affine.for %arg2 = 1 to 1024 step 3 {
// CHECK: %[[I:.*]] = affine.apply #[[$MAP1]](%[[IV]])
// CHECK: %[[SV:.*]] = memref.subview %[[A]][%[[I]], 0, 0] [1, 4, 128] [1, 1, 1] : memref<5x4x128xf32> to memref<4x128xf32, strided<[128, 1], offset: ?>>
   %1 = memref.subview %a[%arg2, 0] [4, 128] [1, 1] :
    memref<1024x1024xf32> to memref<4x128xf32, affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>>
// CHECK: memref.copy %{{.*}}, %[[SV]] : memref<4x128xf32, #{{.*}}> to memref<4x128xf32, strided<[128, 1], offset: ?>>
   memref.copy %1, %0 : memref<4x128xf32, affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>> to memref<4x128xf32>
// CHECK: "some_use"(%[[SV]]) : (memref<4x128xf32, strided{{.*}}>) -> ()
    "some_use"(%0) : (memref<4x128xf32>) -> ()
// CHECK: "some_use"(%[[SV]]) : (memref<4x128xf32, strided{{.*}}>) -> ()
   "some_use"(%0) : (memref<4x128xf32>) -> ()
  }
  return
}

// -----

// CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0) -> (((d0 - 1) floordiv 3) mod 5)>

// CHECK-LABEL: func @multi_buffer_subview_use
func.func @multi_buffer_subview_use(%a: memref<1024x1024xf32>) {
// CHECK-DAG: %[[A:.*]] = memref.alloc() : memref<5x4x128xf32>
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C3:.*]] = arith.constant 3 : index
  %0 = memref.alloc() : memref<4x128xf32>
  %c1024 = arith.constant 1024 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
// CHECK: scf.for %[[IV:.*]] = %[[C1]]
  scf.for %arg2 = %c1 to %c1024 step %c3 {
// CHECK: %[[I:.*]] = affine.apply #[[$MAP1]](%[[IV]])
// CHECK: %[[SV:.*]] = memref.subview %[[A]][%[[I]], 0, 0] [1, 4, 128] [1, 1, 1] : memref<5x4x128xf32> to memref<4x128xf32, strided<[128, 1], offset: ?>>
   %1 = memref.subview %a[%arg2, 0] [4, 128] [1, 1] :
    memref<1024x1024xf32> to memref<4x128xf32, affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>>
// CHECK: memref.copy %{{.*}}, %[[SV]] : memref<4x128xf32, #{{.*}}> to memref<4x128xf32, strided<[128, 1], offset: ?>>
   memref.copy %1, %0 : memref<4x128xf32, affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>> to memref<4x128xf32>
// CHECK: %[[SV1:.*]] = memref.subview %[[SV]][0, 1] [4, 127] [1, 1] : memref<4x128xf32, strided<[128, 1], offset: ?>> to memref<4x127xf32, strided<[128, 1], offset: ?>>
   %s = memref.subview %0[0, 1] [4, 127] [1, 1] :
      memref<4x128xf32> to memref<4x127xf32, affine_map<(d0, d1) -> (d0 * 128 + d1 + 1)>>
// CHECK: "some_use"(%[[SV1]]) : (memref<4x127xf32, strided<[128, 1], offset: ?>>) -> ()
   "some_use"(%s) : (memref<4x127xf32, affine_map<(d0, d1) -> (d0 * 128 + d1 + 1)>>) -> ()
// CHECK: "some_use"(%[[SV]]) : (memref<4x128xf32, strided<[128, 1], offset: ?>>) -> ()
   "some_use"(%0) : (memref<4x128xf32>) -> ()
  }
  return
}

// -----

// CHECK-LABEL: func @multi_buffer_negative
func.func @multi_buffer_negative(%a: memref<1024x1024xf32>) {
// CHECK-NOT: %{{.*}} = memref.alloc() : memref<5x4x128xf32>
//     CHECK: %{{.*}} = memref.alloc() : memref<4x128xf32>
  %0 = memref.alloc() : memref<4x128xf32>
  %c1024 = arith.constant 1024 : index
  %c0 = arith.constant 0 : index
  %c3 = arith.constant 3 : index
  scf.for %arg2 = %c0 to %c1024 step %c3 {
   "blocking_use"(%0) : (memref<4x128xf32>) -> ()
   %1 = memref.subview %a[%arg2, 0] [4, 128] [1, 1] :
    memref<1024x1024xf32> to memref<4x128xf32, affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>>
   memref.copy %1, %0 : memref<4x128xf32, affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>> to memref<4x128xf32>
   "some_use"(%0) : (memref<4x128xf32>) -> ()
  }
  return
}

// -----

// Test that multi-buffering correctly propagates strided layout through expand_shape.

// CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0) -> (((d0 - 1) floordiv 3) mod 5)>

// CHECK-LABEL: func @multi_buffer_expand_shape
func.func @multi_buffer_expand_shape(%a: memref<1024x1024xf32>) {
// CHECK-DAG: %[[ALLOC:.*]] = memref.alloc() : memref<5x4x128xf32>
  %0 = memref.alloc() : memref<4x128xf32>
  %c1024 = arith.constant 1024 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
// CHECK: scf.for %[[IV:.*]] = %{{.*}}
  scf.for %arg2 = %c1 to %c1024 step %c3 {
// CHECK: %[[I:.*]] = affine.apply #[[$MAP1]](%[[IV]])
// CHECK: %[[SV:.*]] = memref.subview %[[ALLOC]][%[[I]], 0, 0] [1, 4, 128] [1, 1, 1] : memref<5x4x128xf32> to memref<4x128xf32, strided<[128, 1], offset: ?>>
    %1 = memref.subview %a[%arg2, 0] [4, 128] [1, 1] :
        memref<1024x1024xf32> to memref<4x128xf32, affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>>
// CHECK: memref.copy %{{.*}}, %[[SV]] : memref<4x128xf32, #{{.*}}> to memref<4x128xf32, strided<[128, 1], offset: ?>>
    memref.copy %1, %0 : memref<4x128xf32, affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>> to memref<4x128xf32>
// CHECK: %[[EXPANDED:.*]] = memref.expand_shape %[[SV]] {{\[\[}}0, 1], [2, 3]] output_shape [2, 2, 64, 2] : memref<4x128xf32, strided<[128, 1], offset: ?>> into memref<2x2x64x2xf32, strided<[256, 128, 2, 1], offset: ?>>
    %expanded = memref.expand_shape %0 [[0, 1], [2, 3]] output_shape [2, 2, 64, 2]
        : memref<4x128xf32> into memref<2x2x64x2xf32>
// CHECK: "some_use"(%[[EXPANDED]]) : (memref<2x2x64x2xf32, strided<[256, 128, 2, 1], offset: ?>>) -> ()
    "some_use"(%expanded) : (memref<2x2x64x2xf32>) -> ()
  }
  return
}

// -----

// Test that multi-buffering correctly propagates strided layout through collapse_shape.

// CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0) -> (((d0 - 1) floordiv 3) mod 5)>

// CHECK-LABEL: func @multi_buffer_collapse_shape
func.func @multi_buffer_collapse_shape(%a: memref<1024x1024xf32>) {
// CHECK-DAG: %[[ALLOC:.*]] = memref.alloc() : memref<5x4x128xf32>
  %0 = memref.alloc() : memref<4x128xf32>
  %c1024 = arith.constant 1024 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
// CHECK: scf.for %[[IV:.*]] = %{{.*}}
  scf.for %arg2 = %c1 to %c1024 step %c3 {
// CHECK: %[[I:.*]] = affine.apply #[[$MAP1]](%[[IV]])
// CHECK: %[[SV:.*]] = memref.subview %[[ALLOC]][%[[I]], 0, 0] [1, 4, 128] [1, 1, 1] : memref<5x4x128xf32> to memref<4x128xf32, strided<[128, 1], offset: ?>>
    %1 = memref.subview %a[%arg2, 0] [4, 128] [1, 1] :
        memref<1024x1024xf32> to memref<4x128xf32, affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>>
// CHECK: memref.copy %{{.*}}, %[[SV]] : memref<4x128xf32, #{{.*}}> to memref<4x128xf32, strided<[128, 1], offset: ?>>
    memref.copy %1, %0 : memref<4x128xf32, affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>> to memref<4x128xf32>
// CHECK: %[[COLLAPSED:.*]] = memref.collapse_shape %[[SV]] {{\[\[}}0, 1]] : memref<4x128xf32, strided<[128, 1], offset: ?>> into memref<512xf32, strided<[1], offset: ?>>
    %collapsed = memref.collapse_shape %0 [[0, 1]]
        : memref<4x128xf32> into memref<512xf32>
// CHECK: "some_use"(%[[COLLAPSED]]) : (memref<512xf32, strided<[1], offset: ?>>) -> ()
    "some_use"(%collapsed) : (memref<512xf32>) -> ()
  }
  return
}

// -----

// Test that multi-buffering correctly propagates through memref.cast.

// CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0) -> (((d0 - 1) floordiv 3) mod 5)>

// CHECK-LABEL: func @multi_buffer_cast
func.func @multi_buffer_cast(%a: memref<1024x1024xf32>) {
// CHECK-DAG: %[[ALLOC:.*]] = memref.alloc() : memref<5x4x128xf32>
  %0 = memref.alloc() : memref<4x128xf32>
  %c1024 = arith.constant 1024 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
// CHECK: scf.for %[[IV:.*]] = %{{.*}}
  scf.for %arg2 = %c1 to %c1024 step %c3 {
// CHECK: %[[I:.*]] = affine.apply #[[$MAP1]](%[[IV]])
// CHECK: %[[SV:.*]] = memref.subview %[[ALLOC]][%[[I]], 0, 0] [1, 4, 128] [1, 1, 1] : memref<5x4x128xf32> to memref<4x128xf32, strided<[128, 1], offset: ?>>
    %1 = memref.subview %a[%arg2, 0] [4, 128] [1, 1] :
        memref<1024x1024xf32> to memref<4x128xf32, affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>>
// CHECK: memref.copy %{{.*}}, %[[SV]] : memref<4x128xf32, #{{.*}}> to memref<4x128xf32, strided<[128, 1], offset: ?>>
    memref.copy %1, %0 : memref<4x128xf32, affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>> to memref<4x128xf32>
// CHECK: %[[CAST:.*]] = memref.cast %[[SV]] : memref<4x128xf32, strided<[128, 1], offset: ?>> to memref<?x128xf32>
    %casted = memref.cast %0 : memref<4x128xf32> to memref<?x128xf32>
// CHECK: "some_use"(%[[CAST]]) : (memref<?x128xf32>) -> ()
    "some_use"(%casted) : (memref<?x128xf32>) -> ()
  }
  return
}

// -----

// Test that multi-buffering correctly propagates through chained view-like ops.

// CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0) -> (((d0 - 1) floordiv 3) mod 5)>

// CHECK-LABEL: func @multi_buffer_chained_view_ops
func.func @multi_buffer_chained_view_ops(%a: memref<1024x1024xf32>) {
// CHECK-DAG: %[[ALLOC:.*]] = memref.alloc() : memref<5x4x128xf32>
  %0 = memref.alloc() : memref<4x128xf32>
  %c1024 = arith.constant 1024 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
// CHECK: scf.for %[[IV:.*]] = %{{.*}}
  scf.for %arg2 = %c1 to %c1024 step %c3 {
// CHECK: %[[I:.*]] = affine.apply #[[$MAP1]](%[[IV]])
// CHECK: %[[SV:.*]] = memref.subview %[[ALLOC]][%[[I]], 0, 0] [1, 4, 128] [1, 1, 1] : memref<5x4x128xf32> to memref<4x128xf32, strided<[128, 1], offset: ?>>
    %1 = memref.subview %a[%arg2, 0] [4, 128] [1, 1] :
        memref<1024x1024xf32> to memref<4x128xf32, affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>>
// CHECK: memref.copy %{{.*}}, %[[SV]] : memref<4x128xf32, #{{.*}}> to memref<4x128xf32, strided<[128, 1], offset: ?>>
    memref.copy %1, %0 : memref<4x128xf32, affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>> to memref<4x128xf32>
// CHECK: %[[EXPANDED:.*]] = memref.expand_shape %[[SV]] {{\[\[}}0, 1], [2, 3]] output_shape [2, 2, 64, 2] : memref<4x128xf32, strided<[128, 1], offset: ?>> into memref<2x2x64x2xf32, strided<[256, 128, 2, 1], offset: ?>>
    %expanded = memref.expand_shape %0 [[0, 1], [2, 3]] output_shape [2, 2, 64, 2]
        : memref<4x128xf32> into memref<2x2x64x2xf32>
// CHECK: %[[CAST:.*]] = memref.cast %[[EXPANDED]] : memref<2x2x64x2xf32, strided<[256, 128, 2, 1], offset: ?>> to memref<?x2x64x2xf32>
    %casted = memref.cast %expanded : memref<2x2x64x2xf32> to memref<?x2x64x2xf32>
// CHECK: "some_use"(%[[CAST]]) : (memref<?x2x64x2xf32>) -> ()
    "some_use"(%casted) : (memref<?x2x64x2xf32>) -> ()
  }
  return
}
