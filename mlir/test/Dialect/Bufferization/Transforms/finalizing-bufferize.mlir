// RUN: mlir-opt %s -finalizing-bufferize -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func @eliminate_materializations(
//  CHECK-SAME:     %[[ARG:.*]]: memref<f32>) -> memref<f32> {
func @eliminate_materializations(%arg0: memref<f32>) -> memref<f32> {
  %0 = bufferization.to_tensor %arg0 : memref<f32>
  %1 = bufferization.to_memref %0 : memref<f32>
  // CHECK: return %[[ARG]] : memref<f32>
  return %1 : memref<f32>
}

// -----

func @unable_to_convert_lone_buffer_cast() -> memref<f32> {
  // expected-error @+1 {{failed to legalize operation 'test.source'}}
  %0 = "test.source"() : () -> tensor<f32>
  %1 = bufferization.to_memref %0 : memref<f32>
  return %1 : memref<f32>
}

// -----

func @unable_to_convert_lone_tensor_load(%arg0: memref<f32>) {
  %0 = bufferization.to_tensor %arg0 : memref<f32>
  // expected-error @+1 {{failed to legalize operation 'test.sink'}}
  "test.sink"(%0) : (tensor<f32>) -> ()
  return
}

// -----

// CHECK: #[[$MAP1:.*]] = affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>
#map1 = affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>

// CHECK-LABEL: func @insert_memref_cast(
//  CHECK-SAME:     %[[arg0:.*]]: memref<?xf32>
func @insert_memref_cast(%arg0: memref<?xf32>) -> memref<?xf32, #map1> {
  %0 = bufferization.to_tensor %arg0 : memref<?xf32>
  %1 = bufferization.to_memref %0 : memref<?xf32, #map1>
  // CHECK: %[[r:.*]] = memref.cast %[[arg0]] : memref<?xf32> to memref<?xf32, #[[$MAP1]]>
  // CHECK: return %[[r]]
  return %1 : memref<?xf32, #map1>
}

// -----

// CHECK: #[[$MAP2:.*]] = affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>
#map2 = affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>

// CHECK-LABEL: func @insert_buffer_copy(
//  CHECK-SAME:     %[[arg0:.*]]: memref<?xf32, #[[$MAP2]]>
func @insert_buffer_copy(%arg0: memref<?xf32, #map2>) -> memref<?xf32> {
  // CHECK: %[[c0:.*]] = arith.constant 0 : index
  // CHECK: %[[dim0:.*]] = memref.dim %[[arg0]], %[[c0]]
  // CHECK: %[[alloc:.*]] = memref.alloc(%[[dim0]]) : memref<?xf32>
  // CHECK: memref.copy %[[arg0]], %[[alloc]]
  %0 = bufferization.to_tensor %arg0 : memref<?xf32, #map2>
  %1 = bufferization.to_memref %0 : memref<?xf32>

  // CHECK: return %[[alloc]]
  return %1 : memref<?xf32>
}
