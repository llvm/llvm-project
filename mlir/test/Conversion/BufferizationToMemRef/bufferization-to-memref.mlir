// RUN: mlir-opt -verify-diagnostics -convert-bufferization-to-memref -split-input-file %s | FileCheck %s

// CHECK-LABEL: @conversion_static
func @conversion_static(%arg0 : memref<2xf32>) -> memref<2xf32> {
  %0 = bufferization.clone %arg0 : memref<2xf32> to memref<2xf32>
  memref.dealloc %arg0 : memref<2xf32>
  return %0 : memref<2xf32>
}

// CHECK:      %[[ALLOC:.*]] = memref.alloc
// CHECK-NEXT: memref.copy %[[ARG:.*]], %[[ALLOC]]
// CHECK-NEXT: memref.dealloc %[[ARG]]
// CHECK-NEXT: return %[[ALLOC]]

// -----

// CHECK-LABEL: @conversion_dynamic
func @conversion_dynamic(%arg0 : memref<?xf32>) -> memref<?xf32> {
  %1 = bufferization.clone %arg0 : memref<?xf32> to memref<?xf32>
  memref.dealloc %arg0 : memref<?xf32>
  return %1 : memref<?xf32>
}

// CHECK:      %[[CONST:.*]] = arith.constant
// CHECK-NEXT: %[[DIM:.*]] = memref.dim %[[ARG:.*]], %[[CONST]]
// CHECK-NEXT: %[[ALLOC:.*]] = memref.alloc(%[[DIM]])
// CHECK-NEXT: memref.copy %[[ARG]], %[[ALLOC]]
// CHECK-NEXT: memref.dealloc %[[ARG]]
// CHECK-NEXT: return %[[ALLOC]]

// -----

func @conversion_unknown(%arg0 : memref<*xf32>) -> memref<*xf32> {
// expected-error@+1 {{failed to legalize operation 'bufferization.clone' that was explicitly marked illegal}}
  %1 = bufferization.clone %arg0 : memref<*xf32> to memref<*xf32>
  memref.dealloc %arg0 : memref<*xf32>
  return %1 : memref<*xf32>
}

// -----

// CHECK: #[[$MAP:.*]] = affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>
#map = affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>
// CHECK-LABEL: func @conversion_with_layout_map(
//  CHECK-SAME:     %[[ARG:.*]]: memref<?xf32, #[[$MAP]]>
//       CHECK:   %[[C0:.*]] = arith.constant 0 : index
//       CHECK:   %[[DIM:.*]] = memref.dim %[[ARG]], %[[C0]]
//       CHECK:   %[[ALLOC:.*]] = memref.alloc(%[[DIM]]) : memref<?xf32>
//       CHECK:   %[[CASTED:.*]] = memref.cast %[[ALLOC]] : memref<?xf32> to memref<?xf32, #[[$MAP]]>
//       CHECK:   memref.copy
//       CHECK:   memref.dealloc
//       CHECK:   return %[[CASTED]]
func @conversion_with_layout_map(%arg0 : memref<?xf32, #map>) -> memref<?xf32, #map> {
  %1 = bufferization.clone %arg0 : memref<?xf32, #map> to memref<?xf32, #map>
  memref.dealloc %arg0 : memref<?xf32, #map>
  return %1 : memref<?xf32, #map>
}

// -----

// This bufferization.clone cannot be lowered because a buffer with this layout
// map cannot be allocated (or casted to).

#map2 = affine_map<(d0)[s0] -> (d0 * 10 + s0)>
func @conversion_with_invalid_layout_map(%arg0 : memref<?xf32, #map2>)
    -> memref<?xf32, #map2> {
// expected-error@+1 {{failed to legalize operation 'bufferization.clone' that was explicitly marked illegal}}
  %1 = bufferization.clone %arg0 : memref<?xf32, #map2> to memref<?xf32, #map2>
  memref.dealloc %arg0 : memref<?xf32, #map2>
  return %1 : memref<?xf32, #map2>
}
