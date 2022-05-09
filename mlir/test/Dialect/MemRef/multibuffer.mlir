// RUN: mlir-opt %s -allow-unregistered-dialect -test-multi-buffering=multiplier=5 -cse -split-input-file | FileCheck %s

// CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0, d1)[s0] -> (d0 * 128 + s0 + d1)>
// CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0, d1, d2) -> (((d0 - d1) floordiv d2) mod 5)>

// CHECK-LABEL: func @multi_buffer
func.func @multi_buffer(%a: memref<1024x1024xf32>) {
// CHECK-DAG: %[[A:.*]] = memref.alloc() : memref<5x4x128xf32>
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C3:.*]] = arith.constant 3 : index
  %0 = memref.alloc() : memref<4x128xf32>
  %c1024 = arith.constant 1024 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
// CHECK: scf.for %[[IV:.*]] = %[[C1]]
  scf.for %arg2 = %c1 to %c1024 step %c3 {
// CHECK: %[[I:.*]] = affine.apply #[[$MAP1]](%[[IV]], %[[C1]], %[[C3]])
// CHECK: %[[SV:.*]] = memref.subview %[[A]][%[[I]], 0, 0] [1, 4, 128] [1, 1, 1] : memref<5x4x128xf32> to memref<4x128xf32, #[[$MAP0]]>
   %1 = memref.subview %a[%arg2, 0] [4, 128] [1, 1] :
    memref<1024x1024xf32> to memref<4x128xf32, affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>>
// CHECK: memref.copy %{{.*}}, %[[SV]] : memref<4x128xf32, #{{.*}}> to memref<4x128xf32, #[[$MAP0]]>
   memref.copy %1, %0 : memref<4x128xf32, affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>> to memref<4x128xf32>
// CHECK: "some_use"(%[[SV]]) : (memref<4x128xf32, #[[$MAP0]]>) -> ()
    "some_use"(%0) : (memref<4x128xf32>) -> ()
// CHECK: "some_use"(%[[SV]]) : (memref<4x128xf32, #[[$MAP0]]>) -> ()
   "some_use"(%0) : (memref<4x128xf32>) -> ()
  }
  return
}

// -----

// CHECK-LABEL: func @multi_buffer_affine
func.func @multi_buffer_affine(%a: memref<1024x1024xf32>) {
// CHECK-DAG: %[[A:.*]] = memref.alloc() : memref<5x4x128xf32>
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C3:.*]] = arith.constant 3 : index
  %0 = memref.alloc() : memref<4x128xf32>
  %c1024 = arith.constant 1024 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
// CHECK: affine.for %[[IV:.*]] = 1
  affine.for %arg2 = 1 to 1024 step 3 {
// CHECK: %[[I:.*]] = affine.apply #[[$MAP1]](%[[IV]], %[[C1]], %[[C3]])
// CHECK: %[[SV:.*]] = memref.subview %[[A]][%[[I]], 0, 0] [1, 4, 128] [1, 1, 1] : memref<5x4x128xf32> to memref<4x128xf32, #[[$MAP0]]>
   %1 = memref.subview %a[%arg2, 0] [4, 128] [1, 1] :
    memref<1024x1024xf32> to memref<4x128xf32, affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>>
// CHECK: memref.copy %{{.*}}, %[[SV]] : memref<4x128xf32, #{{.*}}> to memref<4x128xf32, #[[$MAP0]]>
   memref.copy %1, %0 : memref<4x128xf32, affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>> to memref<4x128xf32>
// CHECK: "some_use"(%[[SV]]) : (memref<4x128xf32, #[[$MAP0]]>) -> ()
    "some_use"(%0) : (memref<4x128xf32>) -> ()
// CHECK: "some_use"(%[[SV]]) : (memref<4x128xf32, #[[$MAP0]]>) -> ()
   "some_use"(%0) : (memref<4x128xf32>) -> ()
  }
  return
}

// -----

// CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0, d1)[s0] -> (d0 * 128 + s0 + d1)>
// CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0, d1, d2) -> (((d0 - d1) floordiv d2) mod 5)>

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
// CHECK: %[[I:.*]] = affine.apply #[[$MAP1]](%[[IV]], %[[C1]], %[[C3]])
// CHECK: %[[SV:.*]] = memref.subview %[[A]][%[[I]], 0, 0] [1, 4, 128] [1, 1, 1] : memref<5x4x128xf32> to memref<4x128xf32, #[[$MAP0]]>
   %1 = memref.subview %a[%arg2, 0] [4, 128] [1, 1] :
    memref<1024x1024xf32> to memref<4x128xf32, affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>>
// CHECK: memref.copy %{{.*}}, %[[SV]] : memref<4x128xf32, #{{.*}}> to memref<4x128xf32, #[[$MAP0]]>
   memref.copy %1, %0 : memref<4x128xf32, affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>> to memref<4x128xf32>
// CHECK: %[[SV1:.*]] = memref.subview %[[SV]][0, 1] [4, 127] [1, 1] : memref<4x128xf32, #[[$MAP0]]> to memref<4x127xf32, #[[$MAP0]]>
   %s = memref.subview %0[0, 1] [4, 127] [1, 1] :
      memref<4x128xf32> to memref<4x127xf32, affine_map<(d0, d1) -> (d0 * 128 + d1 + 1)>>
// CHECK: "some_use"(%[[SV1]]) : (memref<4x127xf32, #[[$MAP0]]>) -> ()
   "some_use"(%s) : (memref<4x127xf32, affine_map<(d0, d1) -> (d0 * 128 + d1 + 1)>>) -> ()
// CHECK: "some_use"(%[[SV]]) : (memref<4x128xf32, #[[$MAP0]]>) -> ()
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

