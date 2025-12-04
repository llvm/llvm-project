// RUN: mlir-opt -p 'builtin.module(buffer-results-to-out-params{modify-public-functions})' %s | FileCheck %s

// Test if `public` functions' return values are transformed into out parameters
// when `buffer-results-to-out-params` is invoked with `modifyPublicFunctions`.

// CHECK-LABEL:   func.func @basic(
// CHECK-SAME:                     %[[ARG0:.*]]: memref<f32>) {
// CHECK:           %[[VAL_0:.*]] = "test.source"() : () -> memref<f32>
// CHECK:           memref.copy %[[VAL_0]], %[[ARG0]] : memref<f32> to memref<f32>
// CHECK:           return
// CHECK:         }
func.func @basic() -> (memref<f32>) {
  %0 = "test.source"() : () -> (memref<f32>)
  return %0 : memref<f32>
}

// CHECK-LABEL:   func.func @presence_of_existing_arguments(
// CHECK-SAME:      %[[ARG0:.*]]: memref<1xf32>,
// CHECK-SAME:      %[[ARG1:.*]]: memref<2xf32>) {
// CHECK:           %[[VAL_0:.*]] = "test.source"() : () -> memref<2xf32>
// CHECK:           memref.copy %[[VAL_0]], %[[ARG1]] : memref<2xf32> to memref<2xf32>
// CHECK:           return
// CHECK:         }
func.func @presence_of_existing_arguments(%arg0: memref<1xf32>) -> (memref<2xf32>) {
  %0 = "test.source"() : () -> (memref<2xf32>)
  return %0 : memref<2xf32>
}

// CHECK-LABEL:   func.func @multiple_results(
// CHECK-SAME:      %[[ARG0:.*]]: memref<1xf32>,
// CHECK-SAME:      %[[ARG1:.*]]: memref<2xf32>) {
// CHECK:           %[[VAL_0:.*]]:2 = "test.source"() : () -> (memref<1xf32>, memref<2xf32>)
// CHECK:           memref.copy %[[VAL_0]]#0, %[[ARG0]] : memref<1xf32> to memref<1xf32>
// CHECK:           memref.copy %[[VAL_0]]#1, %[[ARG1]] : memref<2xf32> to memref<2xf32>
// CHECK:           return
// CHECK:         }
func.func @multiple_results() -> (memref<1xf32>, memref<2xf32>) {
  %0, %1 = "test.source"() : () -> (memref<1xf32>, memref<2xf32>)
  return %0, %1 : memref<1xf32>, memref<2xf32>
}
