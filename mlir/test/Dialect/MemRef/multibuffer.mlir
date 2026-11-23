// RUN: mlir-opt %s -allow-unregistered-dialect -test-multi-buffering=multiplier=5 -cse -split-input-file | FileCheck %s

// CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0) -> (((d0 - 1) floordiv 3) mod 5)>

// CHECK-LABEL: func @multi_buffer
func.func @multi_buffer(%a: memref<1024x1024xf32>) {
// CHECK-DAG: %[[A:.*]] = memref.alloc() alignment = 64 {someAttribute} : memref<5x4x128xf32>
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C3:.*]] = arith.constant 3 : index
  %0 = memref.alloc() alignment = 64 {someAttribute} : memref<4x128xf32>
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

// -----

// Padded strided alloc: trailing strides are unchanged. The new leading stride
// is the original outer pitch (2 * 256), not outerStride * multiplier.
// CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0) -> (((d0 - 1) floordiv 3) mod 5)>

// CHECK-LABEL: func @multi_buffer_strided
func.func @multi_buffer_strided(%src: memref<2x4x8x6xf32, strided<[256, 64, 8, 1]>>) {
// CHECK-DAG: %[[A:.*]] = memref.alloc() : memref<5x2x4x8x6xf32, strided<[512, 256, 64, 8, 1]>>
  %0 = memref.alloc() : memref<2x4x8x6xf32, strided<[256, 64, 8, 1]>>
  %c1024 = arith.constant 1024 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
// CHECK: scf.for %[[IV:.*]] = %{{.*}}
  scf.for %arg2 = %c1 to %c1024 step %c3 {
// CHECK: %[[I:.*]] = affine.apply #[[$MAP1]](%[[IV]])
// CHECK: %[[SV:.*]] = memref.subview %[[A]][%[[I]], 0, 0, 0, 0] [1, 2, 4, 8, 6] [1, 1, 1, 1, 1] : memref<5x2x4x8x6xf32, strided<[512, 256, 64, 8, 1]>> to memref<2x4x8x6xf32, strided<[256, 64, 8, 1], offset: ?>>
    memref.copy %src, %0 : memref<2x4x8x6xf32, strided<[256, 64, 8, 1]>> to memref<2x4x8x6xf32, strided<[256, 64, 8, 1]>>
// CHECK: "some_use"(%[[SV]]) : (memref<2x4x8x6xf32, strided<[256, 64, 8, 1], offset: ?>>) -> ()
    "some_use"(%0) : (memref<2x4x8x6xf32, strided<[256, 64, 8, 1]>>) -> ()
  }
  return
}

// -----

// Identity alloc (no explicit strides) expands to a dense multi-buffer.
// Same shape as @multi_buffer_strided, packed leading stride is 2*192 = 384.
// CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0) -> (((d0 - 1) floordiv 3) mod 5)>

// CHECK-LABEL: func @multi_buffer_identity
func.func @multi_buffer_identity(%src: memref<2x4x8x6xf32>) {
// CHECK-DAG: %[[A:.*]] = memref.alloc() : memref<5x2x4x8x6xf32>
  %0 = memref.alloc() : memref<2x4x8x6xf32>
  %c1024 = arith.constant 1024 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  scf.for %arg2 = %c1 to %c1024 step %c3 {
// CHECK: %[[I:.*]] = affine.apply #[[$MAP1]](%{{.*}})
// CHECK: %[[SV:.*]] = memref.subview %[[A]][%[[I]], 0, 0, 0, 0] [1, 2, 4, 8, 6] [1, 1, 1, 1, 1] : memref<5x2x4x8x6xf32> to memref<2x4x8x6xf32, strided<[192, 48, 6, 1], offset: ?>>
    memref.copy %src, %0 : memref<2x4x8x6xf32> to memref<2x4x8x6xf32>
// CHECK: "some_use"(%[[SV]]) : (memref<2x4x8x6xf32, strided<[192, 48, 6, 1], offset: ?>>) -> ()
    "some_use"(%0) : (memref<2x4x8x6xf32>) -> ()
  }
  return
}

// -----

// Alloc alignment is applied to the slot stride so every buffer stays aligned.
// Outer pitch 24 elements, 64-byte align with f32 -> round up to 32 elements.
// CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0) -> (((d0 - 1) floordiv 3) mod 5)>

// CHECK-LABEL: func @multi_buffer_strided_aligned
func.func @multi_buffer_strided_aligned(%src: memref<3x5xf32, strided<[8, 1]>>) {
// CHECK-DAG: %[[A:.*]] = memref.alloc() alignment = 64 : memref<5x3x5xf32, strided<[32, 8, 1]>>
  %0 = memref.alloc() alignment = 64 : memref<3x5xf32, strided<[8, 1]>>
  %c1024 = arith.constant 1024 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  scf.for %arg2 = %c1 to %c1024 step %c3 {
// CHECK: %[[I:.*]] = affine.apply #[[$MAP1]](%{{.*}})
// CHECK: %[[SV:.*]] = memref.subview %[[A]][%[[I]], 0, 0] [1, 3, 5] [1, 1, 1] : memref<5x3x5xf32, strided<[32, 8, 1]>> to memref<3x5xf32, strided<[8, 1], offset: ?>>
    memref.copy %src, %0 : memref<3x5xf32, strided<[8, 1]>> to memref<3x5xf32, strided<[8, 1]>>
// CHECK: "some_use"(%[[SV]]) : (memref<3x5xf32, strided<[8, 1], offset: ?>>) -> ()
    "some_use"(%0) : (memref<3x5xf32, strided<[8, 1]>>) -> ()
  }
  return
}

// -----

// Padded NHWC-like si8 scratch: 32-byte footprint rounded up to 64-byte align.
// CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0) -> (((d0 - 1) floordiv 3) mod 5)>

// CHECK-LABEL: func @multi_buffer_padded_si8_aligned
func.func @multi_buffer_padded_si8_aligned(%src: memref<1x1x4x8xsi8, strided<[32, 32, 8, 1]>>) {
// CHECK-DAG: %[[A:.*]] = memref.alloc() alignment = 64 : memref<5x1x1x4x8xsi8, strided<[64, 32, 32, 8, 1]>>
  %0 = memref.alloc() alignment = 64 : memref<1x1x4x8xsi8, strided<[32, 32, 8, 1]>>
  %c1024 = arith.constant 1024 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  scf.for %arg2 = %c1 to %c1024 step %c3 {
// CHECK: %[[I:.*]] = affine.apply #[[$MAP1]](%{{.*}})
// Rank reduction drops an inner size-1 dim, so the slot is printed as
// strided<[64, 32, 8, 1]>. With unit N/H that is the same addressing.
// CHECK: %[[SV:.*]] = memref.subview %[[A]][%[[I]], 0, 0, 0, 0] [1, 1, 1, 4, 8] [1, 1, 1, 1, 1] : memref<5x1x1x4x8xsi8, strided<[64, 32, 32, 8, 1]>> to memref<1x1x4x8xsi8, strided<[64, 32, 8, 1], offset: ?>>
    memref.copy %src, %0 : memref<1x1x4x8xsi8, strided<[32, 32, 8, 1]>> to memref<1x1x4x8xsi8, strided<[32, 32, 8, 1]>>
// CHECK: "some_use"(%[[SV]]) : (memref<1x1x4x8xsi8, strided<[64, 32, 8, 1], offset: ?>>) -> ()
    "some_use"(%0) : (memref<1x1x4x8xsi8, strided<[32, 32, 8, 1]>>) -> ()
  }
  return
}

// -----

// Identity alloc where slot size does not divide the alloc alignment:
// 192 bytes rounded up to 256-byte alignment gets an explicit strided layout
// with leading stride 256 so every slot stays 256-byte aligned.
// CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0) -> (((d0 - 1) floordiv 3) mod 5)>

// CHECK-LABEL: func @multi_buffer_identity_aligned
func.func @multi_buffer_identity_aligned(%src: memref<192xi8>) {
// CHECK-DAG: %[[A:.*]] = memref.alloc() alignment = 256 : memref<5x192xi8, strided<[256, 1]>>
  %0 = memref.alloc() alignment = 256 : memref<192xi8>
  %c1024 = arith.constant 1024 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  scf.for %arg2 = %c1 to %c1024 step %c3 {
// CHECK: %[[I:.*]] = affine.apply #[[$MAP1]](%{{.*}})
// CHECK: %[[SV:.*]] = memref.subview %[[A]][%[[I]], 0] [1, 192] [1, 1] : memref<5x192xi8, strided<[256, 1]>> to memref<192xi8, strided<[1], offset: ?>>
    memref.copy %src, %0 : memref<192xi8> to memref<192xi8>
// CHECK: "some_use"(%[[SV]]) : (memref<192xi8, strided<[1], offset: ?>>) -> ()
    "some_use"(%0) : (memref<192xi8>) -> ()
  }
  return
}

// -----

// The view is created before the copy. It does not read the buffer, and the
// copy dominates the read, so the alloc stays 1-D and is replaced by a view of
// the current slot. The user view keeps its constant byte shift.
// CHECK-DAG: #[[$MAP_VIEW:.*]] = affine_map<(d0) -> (((d0 - 1) floordiv 3) * 192 - (((d0 - 1) floordiv 3) floordiv 5) * 960)>

// CHECK-LABEL: func @multi_buffer_view
func.func @multi_buffer_view(%src: memref<192xi8>) {
// CHECK-DAG: %[[A:.*]] = memref.alloc() : memref<960xi8>
// CHECK-DAG: %[[C128:.*]] = arith.constant 128 : index
  %0 = memref.alloc() : memref<192xi8>
  %c128 = arith.constant 128 : index
  %c1024 = arith.constant 1024 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
// CHECK: scf.for %[[IV:.*]] =
  scf.for %arg2 = %c1 to %c1024 step %c3 {
// CHECK: %[[OFF:.*]] = affine.apply #[[$MAP_VIEW]](%[[IV]])
// CHECK: %[[SLOT:.*]] = memref.view %[[A]][%[[OFF]]][] : memref<960xi8> to memref<192xi8>
// CHECK: %[[V:.*]] = memref.view %[[SLOT]][%[[C128]]][] : memref<192xi8> to memref<16xf32>
    %1 = memref.view %0[%c128][] : memref<192xi8> to memref<16xf32>
// CHECK: memref.copy %{{.*}}, %[[SLOT]] : memref<192xi8> to memref<192xi8>
    memref.copy %src, %0 : memref<192xi8> to memref<192xi8>
// CHECK: "some_use"(%[[V]]) : (memref<16xf32>) -> ()
    "some_use"(%1) : (memref<16xf32>) -> ()
  }
  return
}

// -----

// Viewed byte alloc with an alignment: each slot starts at a multiple of 256
// bytes, so the slot stride is 256 and the last slot is not padded
// (4 * 256 + 192 = 1216 bytes).
// CHECK-DAG: #[[$MAP_VIEW_ALIGNED:.*]] = affine_map<(d0) -> (((d0 - 1) floordiv 3) * 256 - (((d0 - 1) floordiv 3) floordiv 5) * 1280)>

// CHECK-LABEL: func @multi_buffer_view_aligned
func.func @multi_buffer_view_aligned(%src: memref<192xi8>) {
// CHECK-DAG: %[[A:.*]] = memref.alloc() alignment = 256 : memref<1216xi8>
// CHECK-DAG: %[[C128:.*]] = arith.constant 128 : index
  %0 = memref.alloc() alignment = 256 : memref<192xi8>
  %c128 = arith.constant 128 : index
  %c1024 = arith.constant 1024 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
// CHECK: scf.for %[[IV:.*]] =
  scf.for %arg2 = %c1 to %c1024 step %c3 {
// CHECK: %[[OFF:.*]] = affine.apply #[[$MAP_VIEW_ALIGNED]](%[[IV]])
// CHECK: %[[SLOT:.*]] = memref.view %[[A]][%[[OFF]]][] : memref<1216xi8> to memref<192xi8>
// CHECK: %[[V:.*]] = memref.view %[[SLOT]][%[[C128]]][] : memref<192xi8> to memref<16xf32>
    %1 = memref.view %0[%c128][] : memref<192xi8> to memref<16xf32>
    memref.copy %src, %0 : memref<192xi8> to memref<192xi8>
// CHECK: "some_use"(%[[V]]) : (memref<16xf32>) -> ()
    "some_use"(%1) : (memref<16xf32>) -> ()
  }
  return
}

// -----

// The view is outside the copy's loop. Inserting the slot in the inner loop
// would not dominate the view, so the alloc is left unchanged.
// CHECK-LABEL: func @multi_buffer_view_nested
func.func @multi_buffer_view_nested(%src: memref<192xi8>) {
// CHECK-NOT: memref<960xi8>
//     CHECK: %[[A:.*]] = memref.alloc() : memref<192xi8>
  %0 = memref.alloc() : memref<192xi8>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c128 = arith.constant 128 : index
  scf.for %i = %c0 to %c4 step %c1 {
//     CHECK: %[[V:.*]] = memref.view %[[A]][%{{.*}}][] : memref<192xi8> to memref<16xf32>
    %v = memref.view %0[%c128][] : memref<192xi8> to memref<16xf32>
    scf.for %j = %c0 to %c4 step %c1 {
      memref.copy %src, %0 : memref<192xi8> to memref<192xi8>
      "some_use"(%v) : (memref<16xf32>) -> ()
    }
  }
  return
}

// -----

// A dynamic viewed alloc has no constant slot size. Leave it unchanged instead
// of expanding it to a ranked memref that memref.view cannot use.
// CHECK-LABEL: func @multi_buffer_view_dynamic
func.func @multi_buffer_view_dynamic(%src: memref<?xi8>, %n: index) {
// CHECK-NOT: memref.alloc() : memref<5x
//     CHECK: %[[A:.*]] = memref.alloc(%{{.*}}) : memref<?xi8>
  %0 = memref.alloc(%n) : memref<?xi8>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  scf.for %iv = %c0 to %c8 step %c1 {
//     CHECK: %[[V:.*]] = memref.view %[[A]][%{{.*}}][] : memref<?xi8> to memref<16xf32>
    %1 = memref.view %0[%c0][] : memref<?xi8> to memref<16xf32>
//     CHECK: memref.copy %{{.*}}, %[[A]] : memref<?xi8> to memref<?xi8>
    memref.copy %src, %0 : memref<?xi8> to memref<?xi8>
//     CHECK: "some_use"(%[[V]]) : (memref<16xf32>) -> ()
    "some_use"(%1) : (memref<16xf32>) -> ()
  }
  return
}
