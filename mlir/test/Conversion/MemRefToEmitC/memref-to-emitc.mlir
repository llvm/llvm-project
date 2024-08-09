// RUN: mlir-opt -convert-memref-to-emitc %s -split-input-file | FileCheck %s

// CHECK-LABEL: memref_store
// CHECK-SAME:  %[[v:.*]]: f32, %[[i:.*]]: index, %[[j:.*]]: index
func.func @memref_store(%v : f32, %i: index, %j: index) {
  // CHECK-NEXT: %[[ALLOCA:.*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.array<4x8xf32>
  %0 = memref.alloca() : memref<4x8xf32>

  // CHECK-NEXT: %[[SUBSCRIPT:.*]] = emitc.subscript %[[ALLOCA]][%[[i]], %[[j]]] : (!emitc.array<4x8xf32>, index, index) -> !emitc.lvalue<f32>
  // CHECK-NEXT: emitc.assign %[[v]] : f32 to %[[SUBSCRIPT]] : <f32>
  memref.store %v, %0[%i, %j] : memref<4x8xf32>
  return
}

// -----

// CHECK-LABEL: memref_load
// CHECK-SAME:  %[[i:.*]]: index, %[[j:.*]]: index
func.func @memref_load(%i: index, %j: index) -> f32 {
  // CHECK-NEXT: %[[ALLOCA:.*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.array<4x8xf32>
  %0 = memref.alloca() : memref<4x8xf32>

  // CHECK-NEXT: %[[SUBSCRIPT:.*]] = emitc.subscript %[[ALLOCA]][%[[i]], %[[j]]] : (!emitc.array<4x8xf32>, index, index) -> !emitc.lvalue<f32>
  // CHECK-NEXT: %[[LOAD:.*]] = emitc.load %[[SUBSCRIPT]] : <f32>
  %1 = memref.load %0[%i, %j] : memref<4x8xf32>
  // CHECK-NEXT: return %[[LOAD]] : f32
  return %1 : f32
}

// -----

// CHECK-LABEL: globals
module @globals {
  memref.global "private" constant @internal_global : memref<3x7xf32> = dense<4.0>
  // CHECK-NEXT: emitc.global static const @internal_global : !emitc.array<3x7xf32> = dense<4.000000e+00>
  memref.global @public_global : memref<3x7xf32>
  // CHECK-NEXT: emitc.global extern @public_global : !emitc.array<3x7xf32>
  memref.global @uninitialized_global : memref<3x7xf32> = uninitialized
  // CHECK-NEXT: emitc.global extern @uninitialized_global : !emitc.array<3x7xf32>

  // CHECK-LABEL: use_global
  func.func @use_global() {
    // CHECK-NEXT: emitc.get_global @public_global : !emitc.array<3x7xf32>
    %0 = memref.get_global @public_global : memref<3x7xf32>
    return
  }
}
