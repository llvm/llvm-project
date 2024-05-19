// RUN: mlir-opt -convert-memref-to-emitc %s -split-input-file | FileCheck %s

// CHECK-LABEL: memref_store
// CHECK-SAME:  %[[v:.*]]: f32, %[[i:.*]]: index, %[[j:.*]]: index
func.func @memref_store(%v : f32, %i: index, %j: index) {
  // CHECK: %[[ALLOCA:.*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.array<4x8xf32>
  // CHECK: %[[SCALAR:.*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> f32
  %0 = memref.alloca() : memref<4x8xf32>
  %s = memref.alloca() : memref<f32>

  // CHECK: %[[SUBSCRIPT:.*]] = emitc.subscript %[[ALLOCA]][%[[i]], %[[j]]] : (!emitc.array<4x8xf32>, index, index) -> f32
  // CHECK: emitc.assign %[[v]] : f32 to %[[SUBSCRIPT:.*]] : f32
  memref.store %v, %0[%i, %j] : memref<4x8xf32>
  // CHECK: emitc.assign %[[v]] : f32 to %[[SCALAR]] : f32
  memref.store %v, %s[] : memref<f32>
  return
}

// -----

// CHECK-LABEL: memref_load
// CHECK-SAME:  %[[i:.*]]: index, %[[j:.*]]: index
func.func @memref_load(%i: index, %j: index) -> (f32, f32) {
  // CHECK: %[[ALLOCA:.*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.array<4x8xf32>
  %0 = memref.alloca() : memref<4x8xf32>
  // CHECK: %[[SCALAR:.*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> f32
  %s = memref.alloca() : memref<f32>

  // CHECK: %[[LOAD:.*]] = emitc.subscript %[[ALLOCA]][%[[i]], %[[j]]] : (!emitc.array<4x8xf32>, index, index) -> f32
  // CHECK: %[[VAR:.*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> f32
  // CHECK: emitc.assign %[[LOAD]] : f32 to %[[VAR]] : f32
  %1 = memref.load %0[%i, %j] : memref<4x8xf32>
  // CHECK: %[[VAR_S:.*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> f32
  // CHECK: emitc.assign %[[SCALAR]] : f32 to %[[VAR_S]] : f32
  %sv = memref.load %s[] : memref<f32>
  // CHECK: return %[[VAR]], %[[VAR_S]] : f32, f32
  return %1, %sv : f32, f32
}

// -----

// CHECK-LABEL: globals
module @globals {
  memref.global "private" constant @internal_global : memref<3x7xf32> = dense<4.0>
  // CHECK: emitc.global static const @internal_global : !emitc.array<3x7xf32> = dense<4.000000e+00>
  memref.global @public_global : memref<3x7xf32>
  // CHECK: emitc.global extern @public_global : !emitc.array<3x7xf32>
  memref.global @uninitialized_global : memref<3x7xf32> = uninitialized
  // CHECK: emitc.global extern @uninitialized_global : !emitc.array<3x7xf32>
  memref.global "private" constant @internal_global_scalar : memref<f32> = dense<4.0>
  // CHECK: emitc.global static const @internal_global_scalar : f32 = 4.000000e+00

  func.func @use_global() {
    // CHECK: emitc.get_global @public_global : !emitc.array<3x7xf32>
    %0 = memref.get_global @public_global : memref<3x7xf32>
    %1 = memref.get_global @internal_global_scalar : memref<f32>
    return
  }
}
