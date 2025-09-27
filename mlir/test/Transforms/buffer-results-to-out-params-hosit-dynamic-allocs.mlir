// RUN: mlir-opt -allow-unregistered-dialect -p 'builtin.module(buffer-results-to-out-params{hoist-dynamic-allocs})' %s -split-input-file | FileCheck %s

func.func private @single_alloc(%size : index) -> (memref<?xf32>) {
  %alloc = memref.alloc(%size) : memref<?xf32>
  return %alloc : memref<?xf32>
}

func.func @single_alloc_test(%size : index) {
  %alloc = call @single_alloc(%size) : (index) -> (memref<?xf32>)
  "test.sink"(%alloc) : (memref<?xf32>) -> ()
}

// CHECK-LABEL: func.func private @single_alloc(
//  CHECK-SAME:   %{{.*}}: index,
//  CHECK-SAME:   %{{.*}}: memref<?xf32>) {

// CHECK-LABEL: func.func @single_alloc_test(
//  CHECK-SAME:   %[[size:.*]]: index) {
//       CHECK:   %[[alloc:.*]] = memref.alloc(%[[size]]) : memref<?xf32>
//       CHECK:   call @single_alloc(%[[size]], %[[alloc]]) : (index, memref<?xf32>) -> ()
//       CHECK:   "test.sink"(%[[alloc]]) : (memref<?xf32>) -> ()
//       CHECK: }

// -----

func.func private @mult_alloc(%size0 : index, %size1 : index) -> (memref<?x?xf32>, memref<?xf32>) {
  %alloc0 = memref.alloc(%size0, %size1) : memref<?x?xf32>
  %alloc1 = memref.alloc(%size1) : memref<?xf32>
  return %alloc0, %alloc1 : memref<?x?xf32>, memref<?xf32>
}

func.func @mult_alloc_test(%size0 : index, %size1: index) {
  %alloc0, %alloc1 = call @mult_alloc(%size0, %size1) : (index, index) -> (memref<?x?xf32>, memref<?xf32>)
  "test.sink"(%alloc0, %alloc1) : (memref<?x?xf32>, memref<?xf32>) -> ()
}

// CHECK-LABEL: func private @mult_alloc(
//  CHECK-SAME:    %{{.*}}: index,  %{{.*}}: index,
//  CHECK-SAME:    %{{.*}}: memref<?x?xf32>, %{{.*}}: memref<?xf32>) {

// CHECK-LABEL: func @mult_alloc_test(
//  CHECK-SAME:   %[[size0:.*]]: index,
//  CHECK-SAME:   %[[size1:.*]]: index) {
//       CHECK:   %[[alloc0:.*]] = memref.alloc(%[[size0]], %[[size1]]) : memref<?x?xf32>
//       CHECK:   %[[alloc1:.*]] = memref.alloc(%[[size1]]) : memref<?xf32>
//       CHECK:   call @mult_alloc(%[[size0]], %[[size1]], %[[alloc0]], %[[alloc1]]) : (index, index, memref<?x?xf32>, memref<?xf32>) -> ()
//       CHECK:   "test.sink"(%[[alloc0]], %[[alloc1]]) : (memref<?x?xf32>, memref<?xf32>) -> ()
//       CHECK: }


// -----

func.func private @complex_alloc(%size0 : index, %size1 : index) -> (memref<?x?xf32>, memref<4xf32>, memref<?xf32>) {
  %alloc0 = memref.alloc(%size0, %size1) : memref<?x?xf32>
  %alloc1 = memref.alloc() : memref<4xf32>
  %alloc2 = memref.alloc(%size1) : memref<?xf32>
  return %alloc0, %alloc1, %alloc2 : memref<?x?xf32>, memref<4xf32>, memref<?xf32>
}

func.func @complex_alloc_test(%size0 : index, %size1: index) {
  %alloc0, %alloc1, %alloc2 = call @complex_alloc(%size0, %size1) : (index, index) -> (memref<?x?xf32>, memref<4xf32>, memref<?xf32>)
  "test.sink"(%alloc0, %alloc1, %alloc2) : (memref<?x?xf32>, memref<4xf32>, memref<?xf32>) -> ()
}

// CHECK-LABEL: func private @complex_alloc(
//  CHECK-SAME:   %{{.*}}: index, %{{.*}}: index,
//  CHECK-SAME:   %{{.*}}: memref<?x?xf32>,
//  CHECK-SAME:   %{{.*}}: memref<4xf32>,
//  CHECK-SAME:   %{{.*}}: memref<?xf32>) {

// CHECK-LABEL: func @complex_alloc_test(
//  CHECK-SAME:   %[[size0:.*]]: index,
//  CHECK-SAME:   %[[size1:.*]]: index) {
//       CHECK:   %[[alloc0:.*]] = memref.alloc(%[[size0]], %[[size1]]) : memref<?x?xf32>
//       CHECK:   %[[alloc1:.*]] = memref.alloc() : memref<4xf32>
//       CHECK:   %[[alloc2:.*]] = memref.alloc(%[[size1]]) : memref<?xf32>
//       CHECK:   call @complex_alloc(%[[size0]], %[[size1]], %[[alloc0]], %[[alloc1]], %[[alloc2]]) : (index, index, memref<?x?xf32>, memref<4xf32>, memref<?xf32>) -> ()
//       CHECK:   "test.sink"(%[[alloc0]], %[[alloc1]], %[[alloc2]]) : (memref<?x?xf32>, memref<4xf32>, memref<?xf32>) -> ()
//       CHECK: }
