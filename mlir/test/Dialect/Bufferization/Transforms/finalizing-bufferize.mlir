// RUN: mlir-opt %s -finalizing-bufferize -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL:   func @eliminate_materializations(
// CHECK-SAME:                                     %[[ARG:.*]]: memref<f32>) -> memref<f32> {
// CHECK:           return %[[ARG]] : memref<f32>
func @eliminate_materializations(%arg0: memref<f32>) -> memref<f32> {
  %0 = bufferization.to_tensor %arg0 : memref<f32>
  %1 = bufferization.to_memref %0 : memref<f32>
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

//       CHECK: #[[$map1:.*]] = affine_map<(d0)[s0] -> (d0 + s0)>
// CHECK-LABEL: func @dyn_layout_to_no_layout_cast(
//  CHECK-SAME:     %[[arg:.*]]: memref<?xf32, #[[$map1]]>)
//       CHECK:   %[[c0:.*]] = arith.constant 0 : index
//       CHECK:   %[[dim:.*]] = memref.dim %[[arg]], %[[c0]]
//       CHECK:   %[[alloc:.*]] = memref.alloc(%[[dim]]) : memref<?xf32>
//       CHECK:   memref.copy %[[arg]], %[[alloc]]
//       CHECK:   return %[[alloc]]
#map1 = affine_map<(d0)[s0] -> (d0 + s0)>
func @dyn_layout_to_no_layout_cast(%m: memref<?xf32, #map1>) -> memref<?xf32> {
  %0 = bufferization.to_tensor %m : memref<?xf32, #map1>
  %1 = bufferization.to_memref %0 : memref<?xf32>
  return %1 : memref<?xf32>
}

// -----

//       CHECK: #[[$map2:.*]] = affine_map<(d0)[s0] -> (d0 * 100 + s0)>
// CHECK-LABEL: func @fancy_layout_to_no_layout_cast(
//  CHECK-SAME:     %[[arg:.*]]: memref<?xf32, #[[$map2]]>)
//       CHECK:   %[[c0:.*]] = arith.constant 0 : index
//       CHECK:   %[[dim:.*]] = memref.dim %[[arg]], %[[c0]]
//       CHECK:   %[[alloc:.*]] = memref.alloc(%[[dim]]) : memref<?xf32>
//       CHECK:   memref.copy %[[arg]], %[[alloc]]
//       CHECK:   return %[[alloc]]
#map2 = affine_map<(d0)[s0] -> (d0 * 100 + s0)>
func @fancy_layout_to_no_layout_cast(%m: memref<?xf32, #map2>) -> memref<?xf32> {
  %0 = bufferization.to_tensor %m : memref<?xf32, #map2>
  %1 = bufferization.to_memref %0 : memref<?xf32>
  return %1 : memref<?xf32>
}

// -----

//       CHECK: #[[$map3:.*]] = affine_map<(d0)[s0] -> (d0 + 25)>
// CHECK-LABEL: func @static_layout_to_no_layout_cast(
//  CHECK-SAME:     %[[arg:.*]]: memref<?xf32, #[[$map3]]>)
//       CHECK:   %[[c0:.*]] = arith.constant 0 : index
//       CHECK:   %[[dim:.*]] = memref.dim %[[arg]], %[[c0]]
//       CHECK:   %[[alloc:.*]] = memref.alloc(%[[dim]]) : memref<?xf32>
//       CHECK:   memref.copy %[[arg]], %[[alloc]]
//       CHECK:   return %[[alloc]]
#map3 = affine_map<(d0)[s0] -> (d0 + 25)>
func @static_layout_to_no_layout_cast(%m: memref<?xf32, #map3>) -> memref<?xf32> {
  %0 = bufferization.to_tensor %m : memref<?xf32, #map3>
  %1 = bufferization.to_memref %0 : memref<?xf32>
  return %1 : memref<?xf32>
}

// -----

// TODO: to_memref with layout maps not supported yet. This should fold to a
// memref.cast.
#map4 = affine_map<(d0)[s0] -> (d0 + s0)>
func @no_layout_to_dyn_layout_cast(%m: memref<?xf32>) -> memref<?xf32, #map4> {
  %0 = bufferization.to_tensor %m : memref<?xf32>
  // expected-error @+1 {{failed to materialize conversion for result #0 of operation 'bufferization.to_memref' that remained live after conversion}}
  %1 = bufferization.to_memref %0 : memref<?xf32, #map4>
  // expected-note @+1 {{see existing live user here}}
  return %1 : memref<?xf32, #map4>
}

// -----

func @illegal_unranked_to_rank(%m: memref<*xf32>) -> memref<?xf32> {
  // expected-note @+1 {{prior use here}}
  %0 = bufferization.to_tensor %m : memref<*xf32>
  // expected-error @+1 {{expects different type than prior uses: 'tensor<?xf32>' vs 'tensor<*xf32>'}}
  %1 = bufferization.to_memref %0 : memref<?xf32>
  return %1 : memref<?xf32>
}
