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
