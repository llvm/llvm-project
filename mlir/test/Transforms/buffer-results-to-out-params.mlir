// RUN: mlir-opt -buffer-results-to-out-params -split-input-file -verify-diagnostics %s | FileCheck %s

// CHECK-LABEL:   func @basic(
// CHECK-SAME:                %[[ARG:.*]]: memref<f32>) {
// CHECK:           %[[RESULT:.*]] = "test.source"() : () -> memref<f32>
// CHECK:           linalg.copy(%[[RESULT]], %[[ARG]]) : memref<f32>, memref<f32>
// CHECK:           return
// CHECK:         }
func @basic() -> (memref<f32>) {
  %0 = "test.source"() : () -> (memref<f32>)
  return %0 : memref<f32>
}

// CHECK-LABEL:   func @presence_of_existing_arguments(
// CHECK-SAME:                                         %[[ARG0:.*]]: memref<1xf32>,
// CHECK-SAME:                                         %[[ARG1:.*]]: memref<2xf32>) {
// CHECK:           %[[RESULT:.*]] = "test.source"() : () -> memref<2xf32>
// CHECK:           linalg.copy(%[[RESULT]], %[[ARG1]]) : memref<2xf32>, memref<2xf32>
// CHECK:           return
// CHECK:         }
func @presence_of_existing_arguments(%arg0: memref<1xf32>) -> (memref<2xf32>) {
  %0 = "test.source"() : () -> (memref<2xf32>)
  return %0 : memref<2xf32>
}

// CHECK-LABEL:   func @multiple_results(
// CHECK-SAME:                           %[[ARG0:.*]]: memref<1xf32>,
// CHECK-SAME:                           %[[ARG1:.*]]: memref<2xf32>) {
// CHECK:           %[[RESULTS:.*]]:2 = "test.source"() : () -> (memref<1xf32>, memref<2xf32>)
// CHECK:           linalg.copy(%[[RESULTS]]#0, %[[ARG0]]) : memref<1xf32>, memref<1xf32>
// CHECK:           linalg.copy(%[[RESULTS]]#1, %[[ARG1]]) : memref<2xf32>, memref<2xf32>
// CHECK:           return
// CHECK:         }
func @multiple_results() -> (memref<1xf32>, memref<2xf32>) {
  %0, %1 = "test.source"() : () -> (memref<1xf32>, memref<2xf32>)
  return %0, %1 : memref<1xf32>, memref<2xf32>
}

// CHECK-LABEL:   func @non_memref_types(
// CHECK-SAME:                           %[[OUTPARAM:.*]]: memref<f32>) -> (i1, i32) {
// CHECK:           %[[RESULT1:.*]]:3 = "test.source"() : () -> (i1, memref<f32>, i32)
// CHECK:           linalg.copy(%[[RESULT1]]#1, %[[OUTPARAM]]) : memref<f32>, memref<f32>
// CHECK:           return %[[RESULT1]]#0, %[[RESULT1]]#2 : i1, i32
// CHECK:         }
func @non_memref_types() -> (i1, memref<f32>, i32) {
  %0, %1, %2 = "test.source"() : () -> (i1, memref<f32>, i32)
  return %0, %1, %2 : i1, memref<f32>, i32
}

// CHECK: func private @external_function(memref<f32>)
func private @external_function() -> (memref<f32>)
// CHECK: func private @result_attrs(memref<f32> {test.some_attr})
func private @result_attrs() -> (memref<f32> {test.some_attr})
// CHECK: func private @mixed_result_attrs(memref<1xf32>, memref<2xf32> {test.some_attr}, memref<3xf32>)
func private @mixed_result_attrs() -> (memref<1xf32>, memref<2xf32> {test.some_attr}, memref<3xf32>)

// -----

// CHECK-LABEL: func private @callee(memref<1xf32>)
func private @callee() -> memref<1xf32>

// CHECK-LABEL:   func @call_basic() {
// CHECK:           %[[OUTPARAM:.*]] = memref.alloc() : memref<1xf32>
// CHECK:           call @callee(%[[OUTPARAM]]) : (memref<1xf32>) -> ()
// CHECK:           "test.sink"(%[[OUTPARAM]]) : (memref<1xf32>) -> ()
// CHECK:           return
// CHECK:         }
func @call_basic() {
  %0 = call @callee() : () -> memref<1xf32>
  "test.sink"(%0) : (memref<1xf32>) -> ()
  return
}

// -----

// CHECK-LABEL: func private @callee(memref<1xf32>, memref<2xf32>)
func private @callee() -> (memref<1xf32>, memref<2xf32>)

// CHECK-LABEL:   func @call_multiple_result() {
// CHECK:           %[[RESULT0:.*]] = memref.alloc() : memref<1xf32>
// CHECK:           %[[RESULT1:.*]] = memref.alloc() : memref<2xf32>
// CHECK:           call @callee(%[[RESULT0]], %[[RESULT1]]) : (memref<1xf32>, memref<2xf32>) -> ()
// CHECK:           "test.sink"(%[[RESULT0]], %[[RESULT1]]) : (memref<1xf32>, memref<2xf32>) -> ()
// CHECK:         }
func @call_multiple_result() {
  %0, %1 = call @callee() : () -> (memref<1xf32>, memref<2xf32>)
  "test.sink"(%0, %1) : (memref<1xf32>, memref<2xf32>) -> ()
}

// -----

// CHECK-LABEL: func private @callee(memref<1xf32>) -> (i1, i32)
func private @callee() -> (i1, memref<1xf32>, i32)

// CHECK-LABEL:   func @call_non_memref_result() {
// CHECK:           %[[RESULT0:.*]] = memref.alloc() : memref<1xf32>
// CHECK:           %[[NON_MEMREF_RESULTS:.*]]:2 = call @callee(%[[RESULT0]]) : (memref<1xf32>) -> (i1, i32)
// CHECK:           "test.sink"(%[[NON_MEMREF_RESULTS]]#0, %[[RESULT0]], %[[NON_MEMREF_RESULTS]]#1) : (i1, memref<1xf32>, i32) -> ()
// CHECK:         }
func @call_non_memref_result() {
  %0, %1, %2 = call @callee() : () -> (i1, memref<1xf32>, i32)
  "test.sink"(%0, %1, %2) : (i1, memref<1xf32>, i32) -> ()
}

// -----

func private @callee() -> (memref<?xf32>)

func @call_non_memref_result() {
  // expected-error @+1 {{cannot create out param for dynamically shaped result}}
  %0 = call @callee() : () -> (memref<?xf32>)
  "test.sink"(%0) : (memref<?xf32>) -> ()
}
