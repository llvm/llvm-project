// RUN: mlir-opt %s -finalizing-bufferize -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL:   func @eliminate_materializations(
// CHECK-SAME:                                     %[[ARG:.*]]: memref<f32>) -> memref<f32> {
// CHECK:           return %[[ARG]] : memref<f32>
func.func @eliminate_materializations(%arg0: memref<f32>) -> memref<f32> {
  %0 = bufferization.to_tensor %arg0 : memref<f32>
  %1 = bufferization.to_memref %0 : memref<f32>
  return %1 : memref<f32>
}

// -----

func.func @unable_to_convert_lone_buffer_cast() -> memref<f32> {
  // expected-error @+1 {{failed to legalize operation 'test.source'}}
  %0 = "test.source"() : () -> tensor<f32>
  %1 = bufferization.to_memref %0 : memref<f32>
  return %1 : memref<f32>
}

// -----

func.func @unable_to_convert_lone_tensor_load(%arg0: memref<f32>) {
  %0 = bufferization.to_tensor %arg0 : memref<f32>
  // expected-error @+1 {{failed to legalize operation 'test.sink'}}
  "test.sink"(%0) : (tensor<f32>) -> ()
  return
}

// -----

// CHECK-LABEL: func @dyn_layout_to_no_layout_cast(
//  CHECK-SAME:     %[[arg:.*]]: memref<?xf32, strided<[1], offset: ?>>)
//       CHECK:   %[[c0:.*]] = arith.constant 0 : index
//       CHECK:   %[[dim:.*]] = memref.dim %[[arg]], %[[c0]]
//       CHECK:   %[[alloc:.*]] = memref.alloc(%[[dim]]) : memref<?xf32>
//       CHECK:   memref.copy %[[arg]], %[[alloc]]
//       CHECK:   return %[[alloc]]
func.func @dyn_layout_to_no_layout_cast(%m: memref<?xf32, strided<[1], offset: ?>>) -> memref<?xf32> {
  %0 = bufferization.to_tensor %m : memref<?xf32, strided<[1], offset: ?>>
  %1 = bufferization.to_memref %0 : memref<?xf32>
  return %1 : memref<?xf32>
}

// -----

// CHECK-LABEL: func @fancy_layout_to_no_layout_cast(
//  CHECK-SAME:     %[[arg:.*]]: memref<?xf32, strided<[100], offset: ?>>)
//       CHECK:   %[[c0:.*]] = arith.constant 0 : index
//       CHECK:   %[[dim:.*]] = memref.dim %[[arg]], %[[c0]]
//       CHECK:   %[[alloc:.*]] = memref.alloc(%[[dim]]) : memref<?xf32>
//       CHECK:   memref.copy %[[arg]], %[[alloc]]
//       CHECK:   return %[[alloc]]
func.func @fancy_layout_to_no_layout_cast(%m: memref<?xf32, strided<[100], offset: ?>>) -> memref<?xf32> {
  %0 = bufferization.to_tensor %m : memref<?xf32, strided<[100], offset: ?>>
  %1 = bufferization.to_memref %0 : memref<?xf32>
  return %1 : memref<?xf32>
}

// -----

// CHECK-LABEL: func @static_layout_to_no_layout_cast(
//  CHECK-SAME:     %[[arg:.*]]: memref<?xf32, strided<[1], offset: 25>>)
//       CHECK:   %[[c0:.*]] = arith.constant 0 : index
//       CHECK:   %[[dim:.*]] = memref.dim %[[arg]], %[[c0]]
//       CHECK:   %[[alloc:.*]] = memref.alloc(%[[dim]]) : memref<?xf32>
//       CHECK:   memref.copy %[[arg]], %[[alloc]]
//       CHECK:   return %[[alloc]]
func.func @static_layout_to_no_layout_cast(%m: memref<?xf32, strided<[1], offset: 25>>) -> memref<?xf32> {
  %0 = bufferization.to_tensor %m : memref<?xf32, strided<[1], offset: 25>>
  %1 = bufferization.to_memref %0 : memref<?xf32>
  return %1 : memref<?xf32>
}

// -----

// TODO: to_memref with layout maps not supported yet. This should fold to a
// memref.cast.
func.func @no_layout_to_dyn_layout_cast(%m: memref<?xf32>) -> memref<?xf32, strided<[1], offset: ?>> {
  %0 = bufferization.to_tensor %m : memref<?xf32>
  // expected-error @+1 {{failed to materialize conversion for result #0 of operation 'bufferization.to_memref' that remained live after conversion}}
  %1 = bufferization.to_memref %0 : memref<?xf32, strided<[1], offset: ?>>
  // expected-note @+1 {{see existing live user here}}
  return %1 : memref<?xf32, strided<[1], offset: ?>>
}

// -----

func.func @illegal_unranked_to_rank(%m: memref<*xf32>) -> memref<?xf32> {
  // expected-note @+1 {{prior use here}}
  %0 = bufferization.to_tensor %m : memref<*xf32>
  // expected-error @+1 {{expects different type than prior uses: 'tensor<?xf32>' vs 'tensor<*xf32>'}}
  %1 = bufferization.to_memref %0 : memref<?xf32>
  return %1 : memref<?xf32>
}
