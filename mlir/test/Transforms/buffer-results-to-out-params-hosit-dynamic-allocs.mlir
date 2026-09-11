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

// -----

func.func private @duplicate_dynamic(%size : index)
    -> (memref<?xf32>, memref<?xf32>) {
  %alloc = memref.alloc(%size) : memref<?xf32>
  return %alloc, %alloc : memref<?xf32>, memref<?xf32>
}

func.func @duplicate_dynamic_test(%size : index) {
  %alloc0, %alloc1 = call @duplicate_dynamic(%size)
      : (index) -> (memref<?xf32>, memref<?xf32>)
  "test.sink"(%alloc0, %alloc1) : (memref<?xf32>, memref<?xf32>) -> ()
}

// CHECK-LABEL: func.func private @duplicate_dynamic(
//  CHECK-SAME:   %{{.*}}: index,
//  CHECK-SAME:   %[[OUT0:.*]]: memref<?xf32>,
//  CHECK-SAME:   %[[OUT1:.*]]: memref<?xf32>) {
//       CHECK:   memref.copy %[[OUT0]], %[[OUT1]] : memref<?xf32> to memref<?xf32>
//       CHECK:   return

// CHECK-LABEL: func.func @duplicate_dynamic_test(
//  CHECK-SAME:   %[[SIZE:.*]]: index) {
//       CHECK:   %[[ALLOC0:.*]] = memref.alloc(%[[SIZE]]) : memref<?xf32>
//       CHECK:   %[[ALLOC1:.*]] = memref.alloc(%[[SIZE]]) : memref<?xf32>
//       CHECK:   call @duplicate_dynamic(%[[SIZE]], %[[ALLOC0]], %[[ALLOC1]]) : (index, memref<?xf32>, memref<?xf32>) -> ()
//       CHECK:   "test.sink"(%[[ALLOC0]], %[[ALLOC1]]) : (memref<?xf32>, memref<?xf32>) -> ()

// -----

// CHECK-LABEL: func.func private @multiple_dynamic_returns_with_non_alloc(
// CHECK-SAME:    %[[SIZE:.*]]: index, %[[INPUT:.*]]: memref<?xf32>, %[[COND:.*]]: i1, %[[OUT0:.*]]: memref<?xf32>, %[[OUT1:.*]]: memref<?xf32>) {
// CHECK:        cf.cond_br %[[COND]], ^[[THEN:.*]], ^[[ELSE:.*]]
// CHECK:      ^[[THEN]]:
// CHECK:        memref.copy %[[OUT0]], %[[OUT1]] : memref<?xf32> to memref<?xf32>
// CHECK:        return
// CHECK:      ^[[ELSE]]:
// CHECK:        memref.copy %[[INPUT]], %[[OUT0]] : memref<?xf32> to memref<?xf32>
// CHECK:        memref.copy %[[INPUT]], %[[OUT1]] : memref<?xf32> to memref<?xf32>
// CHECK:        return
func.func private @multiple_dynamic_returns_with_non_alloc(
    %size: index, %input: memref<?xf32>, %cond: i1)
    -> (memref<?xf32>, memref<?xf32>) {
  %a = memref.alloc(%size) : memref<?xf32>
  cf.cond_br %cond, ^then, ^else
^then:
  return %a, %a : memref<?xf32>, memref<?xf32>
^else:
  return %input, %input : memref<?xf32>, memref<?xf32>
}
