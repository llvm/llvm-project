// RUN: mlir-opt -verify-diagnostics -convert-bufferization-to-memref -split-input-file %s | FileCheck %s

// CHECK-LABEL: @conversion_static
func.func @conversion_static(%arg0 : memref<2xf32>) -> memref<2xf32> {
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
func.func @conversion_dynamic(%arg0 : memref<?xf32>) -> memref<?xf32> {
  %1 = bufferization.clone %arg0 : memref<?xf32> to memref<?xf32>
  memref.dealloc %arg0 : memref<?xf32>
  return %1 : memref<?xf32>
}

// CHECK:      %[[CONST:.*]] = arith.constant
// CHECK:      %[[DIM:.*]] = memref.dim %[[ARG:.*]], %[[CONST]]
// CHECK-NEXT: %[[ALLOC:.*]] = memref.alloc(%[[DIM]])
// CHECK-NEXT: memref.copy %[[ARG]], %[[ALLOC]]
// CHECK-NEXT: memref.dealloc %[[ARG]]
// CHECK-NEXT: return %[[ALLOC]]

// -----

// CHECK-LABEL: @conversion_unknown
func.func @conversion_unknown(%arg0 : memref<*xf32>) -> memref<*xf32> {
  %1 = bufferization.clone %arg0 : memref<*xf32> to memref<*xf32>
  memref.dealloc %arg0 : memref<*xf32>
  return %1 : memref<*xf32>
}

// CHECK:      %[[RANK:.*]] = memref.rank %[[ARG:.*]]
// CHECK-NEXT: %[[ALLOCA:.*]] = memref.alloca(%[[RANK]])
// CHECK-NEXT: %[[FOR:.*]] = scf.for
// CHECK-NEXT: %[[DIM:.*]] = memref.dim %[[ARG:.*]] %[[ARG:.*]]
// CHECK-NEXT: memref.store %[[DIM:.*]], %[[ALLOCA:.*]][%[[ARG:.*]]]
// CHECK-NEXT: %[[MUL:.*]] = arith.muli %[[ARG:.*]], %[[DIM:.*]]
// CHECK-NEXT: scf.yield %[[MUL:.*]]
// CHECK:      %[[ALLOC:.*]] = memref.alloc(%[[FOR:.*]])
// CHECK-NEXT: %[[RESHAPE:.*]] = memref.reshape %[[ALLOC:.*]]
// CHECK-NEXT: memref.copy
// CHECK-NEXT: memref.dealloc
// CHECK-NEXT: return %[[RESHAPE:.*]]

// -----

// CHECK-LABEL: func @conversion_with_layout_map(
//  CHECK-SAME:     %[[ARG:.*]]: memref<?xf32, strided<[?], offset: ?>>
//       CHECK:   %[[C0:.*]] = arith.constant 0 : index
//       CHECK:   %[[DIM:.*]] = memref.dim %[[ARG]], %[[C0]]
//       CHECK:   %[[ALLOC:.*]] = memref.alloc(%[[DIM]]) : memref<?xf32>
//       CHECK:   %[[CASTED:.*]] = memref.cast %[[ALLOC]] : memref<?xf32> to memref<?xf32, strided<[?], offset: ?>>
//       CHECK:   memref.copy
//       CHECK:   memref.dealloc
//       CHECK:   return %[[CASTED]]
func.func @conversion_with_layout_map(%arg0 : memref<?xf32, strided<[?], offset: ?>>) -> memref<?xf32, strided<[?], offset: ?>> {
  %1 = bufferization.clone %arg0 : memref<?xf32, strided<[?], offset: ?>> to memref<?xf32, strided<[?], offset: ?>>
  memref.dealloc %arg0 : memref<?xf32, strided<[?], offset: ?>>
  return %1 : memref<?xf32, strided<[?], offset: ?>>
}

// -----

// This bufferization.clone cannot be lowered because a buffer with this layout
// map cannot be allocated (or casted to).

func.func @conversion_with_invalid_layout_map(%arg0 : memref<?xf32, strided<[10], offset: ?>>)
    -> memref<?xf32, strided<[10], offset: ?>> {
// expected-error@+1 {{failed to legalize operation 'bufferization.clone' that was explicitly marked illegal}}
  %1 = bufferization.clone %arg0 : memref<?xf32, strided<[10], offset: ?>> to memref<?xf32, strided<[10], offset: ?>>
  memref.dealloc %arg0 : memref<?xf32, strided<[10], offset: ?>>
  return %1 : memref<?xf32, strided<[10], offset: ?>>
}

// -----
// Test: check that the dealloc lowering pattern is registered.

// CHECK-NOT: func @deallocHelper
// CHECK-LABEL: func @conversion_dealloc_simple
// CHECK-SAME: [[ARG0:%.+]]: memref<2xf32>
// CHECK-SAME: [[ARG1:%.+]]: i1
func.func @conversion_dealloc_simple(%arg0: memref<2xf32>, %arg1: i1) {
  bufferization.dealloc (%arg0 : memref<2xf32>) if (%arg1)
  return
}

//      CHECK: scf.if [[ARG1]] {
// CHECK-NEXT:   memref.dealloc [[ARG0]] : memref<2xf32>
// CHECK-NEXT: }
// CHECK-NEXT: return
